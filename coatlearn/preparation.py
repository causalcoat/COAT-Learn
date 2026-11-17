"""Explicit preparation of example files and optional NLTK corpora.

Importing this module never accesses the network. Only prepare_resources may
copy or download resources, and existing destination files are never replaced.
"""

import csv
import os
from pathlib import Path
import shutil
import tempfile
import tomllib
from urllib.request import urlopen


_BASE_URL = "https://raw.githubusercontent.com/causalcoat/COAT-Learn/main/"
_DATA = {
    "mb": "data/apple_reviews_PAG_graph1.csv",
    "pag": "data/apple_reviews_PAG_graph1.csv",
    "adj": "data/hillstrom_email_visit_post.csv",
}


def _validate(path, name):
    try:
        if name.endswith('.toml'):
            with path.open('rb') as handle:
                tomllib.load(handle)
        elif name.endswith('.csv'):
            with path.open('r', encoding='utf-8-sig', newline='') as handle:
                header = next(csv.reader(handle), [])
            if not header or any(not column.strip() for column in header):
                raise ValueError('empty CSV header')
        elif path.stat().st_size == 0:
            raise ValueError('empty file')
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        raise ValueError(f'Invalid resource: {name}') from None


def _prepare_file(root, source_root, name):
    destination = root / name
    if destination.exists():
        _validate(destination, name)
        return 'reused'

    destination.parent.mkdir(parents=True, exist_ok=True)
    source = source_root / name
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.coatlearn-', delete=False) as handle:
            temporary = Path(handle.name)
            if source.is_file():
                with source.open('rb') as input_file:
                    shutil.copyfileobj(input_file, handle)
                action = 'copied'
            else:
                try:
                    with urlopen(_BASE_URL + name, timeout=60) as response:
                        shutil.copyfileobj(response, handle)
                except Exception:
                    raise RuntimeError(f'Could not download resource: {name}') from None
                action = 'downloaded'
        _validate(temporary, name)
        try:
            # A hard link publishes the complete file without replacing a file
            # that another process may have created since the existence check.
            os.link(temporary, destination)
        except FileExistsError:
            _validate(destination, name)
            return 'reused'
        return action
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _prepare_wordnet():
    import nltk

    download_dir = Path.home() / 'nltk_data'
    # nltk normally includes this directory; account for custom NLTK_DATA setups.
    if str(download_dir) not in nltk.data.path:
        nltk.data.path.append(str(download_dir))
    result = {'enabled': True, 'downloaded': [], 'reused': []}
    for package in ('wordnet', 'omw-1.4'):
        def available():
            for resource in (f'corpora/{package}', f'corpora/{package}.zip'):
                try:
                    nltk.data.find(resource)
                    return True
                except LookupError:
                    pass
            return False

        if available():
            result['reused'].append(package)
            continue
        download_dir.mkdir(parents=True, exist_ok=True)
        try:
            success = nltk.download(package, download_dir=str(download_dir), quiet=True, raise_on_error=True)
        except Exception:
            raise RuntimeError(f'Could not download NLTK resource: {package}') from None
        if not success or not available():
            raise RuntimeError(f'Could not download NLTK resource: {package}')
        result['downloaded'].append(package)
    return result


def prepare_resources(*, method='mb', directory='.', decision_kind='model', wordnet=None, source_root=None) -> dict:
    """Prepare examples explicitly, reusing valid existing files.

    source_root optionally identifies a source checkout for local copying.
    WordNet is prepared by default only for decision_kind='llm'. The return
    value contains relative paths grouped by downloaded, copied and reused.
    Invalid existing resources raise ValueError and are left untouched.
    """
    if method not in (*_DATA, 'all', 'none'):
        raise ValueError('method must be mb, pag, adj, all or none')
    if decision_kind not in ('model', 'llm'):
        raise ValueError('decision_kind must be model or llm')
    if wordnet is not None and not isinstance(wordnet, bool):
        raise ValueError('wordnet must be a bool or None')
    root = Path(directory).expanduser().resolve()
    source = Path(source_root).expanduser().resolve() if source_root is not None else Path(__file__).resolve().parent.parent
    methods = tuple(_DATA) if method == 'all' else (() if method == 'none' else (method,))
    names = [f'config/{item}.toml' for item in methods]
    names += list(dict.fromkeys(_DATA[item] for item in methods))
    if methods:
        names += ['data/README.md', 'data/LICENSE']
    result = {'method': method, 'directory': str(root), 'downloaded': [], 'copied': [], 'reused': []}
    for name in names:
        result[_prepare_file(root, source, name)].append(name)
    enabled = decision_kind == 'llm' if wordnet is None else wordnet
    result['wordnet'] = _prepare_wordnet() if enabled else {'enabled': False, 'downloaded': [], 'reused': []}
    return result
