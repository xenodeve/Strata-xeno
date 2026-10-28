"""Verify the complete importable compiler subset before any upstream import."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from tools.qwenputer.tokenizer import NativeInputError


REVISION = 'ca13310e2f9f788d1dd3feeea8213c1b0c23913c'
MANIFEST_SHA256 = 'a5519351051213218a2236f1a4737af0e26e310fe3fb74d3f9269ef5bfd8c7c5'


def require_compiler_environment():
    # Setting the variable after Python starts does not fix its hash seed.
    if os.environ.get('PYTHONHASHSEED') != '0' or sys.flags.hash_randomization != 0:
        raise NativeInputError('compiler_hash_seed')


def verify_source(root: Path) -> dict:
    root = Path(root).resolve()
    try:
        with (root / 'UPSTREAM.json').open('rb') as file:
            raw = file.read(65537)
    except OSError as error:
        raise NativeInputError('compiler_manifest') from error
    if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise NativeInputError('compiler_manifest')
    manifest = json.loads(raw)
    expected = set()
    expected_directories = set()
    for item in manifest['files']:
        name = item['path']
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise NativeInputError('compiler_source')
        try:
            with path.open('rb') as file:
                data = file.read(item['bytes'] + 1)
        except OSError as error:
            raise NativeInputError('compiler_source') from error
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if (len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']
                or blob != item['git_blob']):
            raise NativeInputError('compiler_source')
        if name.startswith('experimental/riscv/'):
            expected.add(name)
            expected_directories.update(p.as_posix() for p in Path(name).parents if p.as_posix() != '.')
    actual = set()
    for path in (root / 'experimental/riscv').rglob('*'):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise NativeInputError('compiler_reparse')
        if not path.resolve().is_relative_to(root):
            raise NativeInputError('compiler_source')
        name = path.relative_to(root).as_posix()
        if stat.S_ISDIR(info.st_mode):
            if name not in expected_directories:
                raise NativeInputError('compiler_unreviewed_directory')
        elif stat.S_ISREG(info.st_mode):
            actual.add(name)
        else:
            raise NativeInputError('compiler_unreviewed_file')
    if actual != expected:
        # Pyd/pyc or extra packages could win Python's import lookup even when
        # every expected .py remains byte-identical. Fail before importing.
        raise NativeInputError('compiler_unreviewed_file')
    return manifest
