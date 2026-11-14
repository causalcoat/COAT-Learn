#!/usr/bin/env python3
"""Generate PyPI-safe README links from the authoritative English README."""
from pathlib import Path
import argparse
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://github.com/causalcoat/COAT-Learn/blob/main/'


def render():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    def replace(match):
        target = match.group(1) or match.group(2)
        if urlsplit(target).scheme or target.startswith(('#', '//')):
            return match.group(0)
        return '](<' + SOURCE + target + '>)'
    return re.sub(r'\]\((?:<([^>]+)>|([^\s)]+))\)', replace, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    target = ROOT / 'README-PyPI.md'
    expected = render()
    if args.check:
        if not target.is_file() or target.read_text(encoding='utf-8') != expected:
            parser.error('Run python tools/prepare_pypi_readme.py and commit the generated README.')
        print('PyPI README matches README.md; repository-relative links are absolute.')
    else:
        target.write_text(expected, encoding='utf-8')
        print('Generated README-PyPI.md from README.md.')


if __name__ == '__main__':
    main()
