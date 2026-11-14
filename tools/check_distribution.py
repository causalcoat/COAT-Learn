#!/usr/bin/env python3
"""Inspect publication archives locally; this script never uploads anything."""
import argparse
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
import os
import re
import tarfile
import zipfile
from packaging.version import Version


def check_version(value, target):
    version = Version(value)
    if target != 'build-only':
        tag = os.environ.get('GITHUB_REF_NAME', '')
        if not re.fullmatch(r'v\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?', tag):
            raise ValueError('Publishing requires a vX.Y.Z or vX.Y.ZrcN Git tag')
        if version != Version(tag[1:]) or version.local or version.is_devrelease:
            raise ValueError('Distribution version must match the tag, without local/dev suffixes')
        if target == 'pypi' and version.is_prerelease:
            raise ValueError('The production path publishes final releases; use TestPyPI for this prerelease')
    return version


def check_metadata(raw, target):
    meta = BytesParser(policy=default).parsebytes(raw)
    if meta['Name'] != 'coatlearn':
        raise ValueError('Unexpected distribution name')
    if not meta['Description-Content-Type'].startswith('text/markdown'):
        raise ValueError('Missing Markdown description')
    if not meta['License-Expression']:
        raise ValueError('Missing SPDX license expression')
    description = meta.get_payload()
    for a, b in re.findall(r'\]\((?:<([^>]+)>|([^\s)]+))\)', description):
        target_url = a or b
        if not re.match(r'https?://|mailto:|#', target_url):
            raise ValueError('Relative README link would be broken on PyPI: ' + target_url)
    return check_version(meta['Version'], target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--target', choices=('build-only', 'testpypi', 'pypi'), default='build-only')
    args = parser.parse_args()
    wheels = list(args.directory.glob('*.whl'))
    sources = list(args.directory.glob('*.tar.gz'))
    if len(wheels) != 1 or len(sources) != 1:
        parser.error('Use a fresh output directory containing exactly one wheel and one source archive')
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        meta = next(name for name in names if name.endswith('.dist-info/METADATA'))
        wheel_version = check_metadata(archive.read(meta), args.target)
        assert 'coatlearn/cli/__init__.py' in names
        assert not any('/tests/' in n or n.endswith('/AGENTS.md') for n in names)
        assert 'coatlearn = coatlearn.cli:main' in archive.read(next(n for n in names if n.endswith('/entry_points.txt'))).decode()
    with tarfile.open(sources[0]) as archive:
        names = archive.getnames()
        prefix = names[0].split('/')[0]
        source_version = check_metadata(archive.extractfile(prefix + '/PKG-INFO').read(), args.target)
        assert source_version == wheel_version
        for name in ('data/LICENSE', 'docs/assets/docs.js', 'docs/assets/vendor/docsify-search.min.js', 'skills/coatlearn/SKILL.md'):
            assert prefix + '/' + name in names, name
        csvs = [n for n in names if n.startswith(prefix + '/data/') and n.endswith('.csv')]
        assert len(csvs) == 2
        assert not any(n.endswith('/.env') or n.endswith('/AGENTS.md') or '/tests/' in n for n in names)
    print('Distribution name, Git version, metadata, absolute README links, CLI, docs and data: OK')
    print('Version:', wheel_version, '| target:', args.target, '| no upload performed')


if __name__ == '__main__':
    main()
