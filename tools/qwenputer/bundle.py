"""Assemble verified offline circuit/oracle deliverables, never execute inputs."""
import hashlib
import json
from pathlib import Path
import re

from tools.qwenputer.export import canonical_json
from tools.qwenputer.tokenizer import NativeInputError


def file_digest(path, limit):
    if path.is_symlink() or getattr(path.lstat(), 'st_file_attributes', 0) & 0x400:
        raise NativeInputError('bundle_reparse')
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as file:
        while data := file.read(1024 * 1024):
            size += len(data)
            if size > limit:
                raise NativeInputError('bundle_file_size')
            digest.update(data)
    return {'bytes': size, 'sha256': digest.hexdigest()}


def assemble_bundle(export_dir, oracle_dir, isa_dir, output_dir):
    export_dir, oracle_dir, isa_dir, output_dir = map(Path, (export_dir, oracle_dir, isa_dir, output_dir))
    if output_dir.exists():
        raise NativeInputError('bundle_output')
    def read_manifest(path):
        with path.open('rb') as f:
            data = f.read(1048577)
        if len(data) > 1048576:
            raise NativeInputError('bundle_manifest_size')
        try:
            return json.loads(data)
        except ValueError as error:
            raise NativeInputError('bundle_manifest') from error
    base = read_manifest(export_dir / 'manifest.json')
    oracle = read_manifest(oracle_dir / 'manifest.json')
    isa = read_manifest(isa_dir / 'manifest.json')
    if (oracle.get('ir') is not True or oracle.get('artifact_sha256') != base['artifact_sha256']
            or isa.get('source_sha256') != base['identities']['source_sha256']
            or isa.get('tokenizer_sha256') != base['identities']['tokenizer_sha256']):
        raise NativeInputError('bundle_binding')
    files = {}
    for name in ('native.qput', 'LICENSE.compiler', 'NOTICE.txt', 'UPSTREAM.json'):
        path = export_dir / name
        files[name] = (path, file_digest(path, 268435456))
    if files['native.qput'][1] != {'bytes': base['artifact_bytes'], 'sha256': base['artifact_sha256']}:
        raise NativeInputError('bundle_artifact_digest')
    for role, directory, manifest in (('oracle', oracle_dir, oracle), ('isa', isa_dir, isa)):
        listed = manifest.get('files')
        if not isinstance(listed, dict) or not 1 <= len(listed) <= 1000:
            raise NativeInputError('bundle_file_count')
        for name, expected in sorted(listed.items()):
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', name):
                raise NativeInputError('bundle_file_name')
            path = directory / name
            if file_digest(path, 268435456) != expected:
                raise NativeInputError('bundle_input_digest')
            files[role + '/' + name] = (path, expected)
        path = directory / 'manifest.json'
        files[role + '/manifest.json'] = (path, file_digest(path, 1048576))
    path = export_dir / 'manifest.json'
    files['export-manifest.json'] = (path, file_digest(path, 1048576))
    if 'oracle/oracle.jsonl' in files:
        files['oracle.jsonl'] = files['oracle/oracle.jsonl']
    if sum(record['bytes'] for _, record in files.values()) > 4 * 1024 ** 3:
        raise NativeInputError('bundle_total_size')
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, (source, expected) in sorted(files.items()):
        target = output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        digest, count = hashlib.sha256(), 0
        with source.open('rb') as src, target.open('xb') as dst:
            while data := src.read(1024 * 1024):
                count += len(data)
                if count > expected['bytes']:
                    raise NativeInputError('bundle_input_changed')
                digest.update(data)
                dst.write(data)
        if {'bytes': count, 'sha256': digest.hexdigest()} != expected:
            raise NativeInputError('bundle_input_changed')
    result = dict(base, oracle_status='complete', native_execution_tested=False,
                  oracle_cases=oracle['cases'], isa_cases=isa['cases'],
                  compiler_patch_sha256=hashlib.sha256(b'[]').hexdigest(),
                  files={name: record for name, (_, record) in sorted(files.items())})
    (output_dir / 'manifest.json').write_bytes(canonical_json(result))
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--oracle-dir', type=Path, required=True)
    parser.add_argument('--isa-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = assemble_bundle(args.export_dir, args.oracle_dir, args.isa_dir, args.output)
    print(json.dumps({k: result[k] for k in ('artifact_sha256', 'oracle_status', 'oracle_cases',
                                           'isa_cases', 'native_execution_tested')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
