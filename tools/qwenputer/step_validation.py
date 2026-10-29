"""Offline native CUDA replay/free-running checks; never serving code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess

from tools.qwenputer.format import decode_sections
from tools.qwenputer.bundle import file_digest


def fault_cases(cases):
    return {c['name']: c['fault'] for c in cases if c.get('fault') is not None}


def inputs_data(stream, text_count):
    if not isinstance(stream, list) or not 1 <= len(stream) <= 200000:
        raise ValueError('stream_shape')
    result = bytearray()
    for token in stream:
        if type(token) is not int or token < 0:
            raise ValueError('stream_token')
        kind, token_id = (1, token - (1 << 24)) if token >= 1 << 24 else (0, token)
        if token_id >= (text_count if kind else 230324):
            raise ValueError('stream_token')
        result += struct.pack('<II', kind, token_id)
    return bytes(result)


def check_lags(artifact, carry_path, native_path):
    """Project exported history through channels, then compare the original IR carry.

    This independent checking code runs after the CUDA execution interval.
    It does not call compiler functions or execute the RISC-V reference machine.
    """
    import numpy as np
    s = decode_sections(artifact.read_bytes())
    rows = lambda name, fmt: list(struct.iter_unpack(fmt, s[name][1]))
    config = rows('config', '<16I')[0]
    exprs, terms = rows('exprs', '<ii'), rows('terms', '<if')
    channels, maps = rows('chans', '<i4f'), rows('maps', '<ii')
    heads = rows('heads', '<21if')
    levels = sorted({h[1] for h in heads})
    history = np.fromfile(native_path, dtype='<f4').reshape(len(levels), 3, config[0])
    checked = 0
    for entry in json.loads(carry_path.read_bytes()):
        key, expected = entry['key'], entry['values']
        node, kind, *indices = key
        if kind not in ('q', 'k', 'v'):
            raise ValueError('oracle_lag_kind')
        vi, lane = indices if kind == 'v' else (0, indices[0])
        head = next(h for h in heads if h[0] == node and h[2] == vi)
        pair = {'q': 6, 'k': 8, 'v': 10}[kind]
        candidates = maps[head[pair]:head[pair] + head[pair + 1]]
        channel = channels[next(e for col, e in candidates if col == lane)]
        start, count = exprs[channel[0]]
        actual = []
        for lag in reversed(range(len(expected))):
            total = np.float32(0)
            for dim, coefficient in terms[start:start + count]:
                total = np.float32(total + np.float32(coefficient) * history[levels.index(head[1]), lag, dim])
            actual.append(float(total))
        if actual != expected:
            raise ValueError(f'lag_mismatch:{key}:{actual}:{expected}')
        checked += len(expected)
    return checked


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case', action='append')
    p.add_argument('--group', choices=('oracle', 'isa', 'all'), default='all')
    p.add_argument('--runner', action='store_true', help='prefix/seed-only autonomous CUDA runner')
    p.add_argument('--chunk', type=int, default=128)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error('output must be new')
    if not 1 <= args.chunk <= 256:
        p.error('chunk must be 1..256')
    args.output.mkdir(parents=True)
    binary_digest = file_digest(args.binary, 64 << 20)['sha256']
    manifest_path = args.bundle / 'manifest.json'
    if manifest_path.stat().st_size > 1048576:
        raise ValueError('bundle_manifest_size')
    manifest = json.loads(manifest_path.read_bytes())
    def verify(path):
        name = path.relative_to(args.bundle).as_posix()
        if name not in manifest['files'] or file_digest(path, 256 << 20) != manifest['files'][name]:
            raise ValueError('fixture_digest:' + name)
    verify(args.bundle / 'native.qput')
    identity_keys = ('abi_sha256', 'source_sha256', 'firmware_sha256', 'boot_sha256',
                     'model_sha256', 'tokenizer_sha256')
    if args.binding.read_bytes() != b''.join(bytes.fromhex(manifest['identities'][k]) for k in identity_keys):
        raise ValueError('independent_binding')
    artifact = args.bundle / 'native.qput'
    text_count = struct.unpack('<16I', decode_sections(artifact.read_bytes())['config'][1])[2]
    selected = set(args.case or [])
    groups = ('oracle', 'isa') if args.group == 'all' else (args.group,)
    faults = {}
    if 'isa' in groups and args.runner:
        verify(args.bundle / 'isa/cases.json')
        faults = fault_cases(json.loads((args.bundle / 'isa/cases.json').read_bytes()))
    results = []
    for group in groups:
        root = args.bundle / group
        files = sorted(root.glob('*.reference.json' if group == 'oracle' else '*.stream.json'))
        for source in files:
            name = source.name.split('.')[0]
            if not re.fullmatch('[a-z0-9-]+', name):
                raise ValueError('fixture_name')
            if selected and name not in selected:
                continue
            data = json.loads(source.read_bytes())
            verify(source)
            stream = data['stream'] if group == 'oracle' else data
            label = group + '-' + name
            inputs = args.output / (label + '.inputs')
            inputs.write_bytes(inputs_data(stream, text_count))
            final = root / (name + ('.final-state.f32' if group == 'oracle' else '.final.f32'))
            verify(root / (name + '.residual.f32'))
            verify(final)
            command = [str(args.binary), str(artifact), str(args.binding), str(inputs)]
            command += [str(final), str(args.chunk)] if args.runner else [str(root / (name + '.residual.f32')), str(final)]
            if args.runner and name in faults:
                command += ['--expect', '6']
            dump = args.output / (label + '.lags.f32')
            if group == 'oracle':
                verify(root / (name + '.final-lags.json'))
                command += ['--dump', str(dump)]
            else:
                verify(root / (name + '.init.f32'))
                command += ['--initial', str(root / (name + '.init.f32'))]
            run = subprocess.run(command, capture_output=True, text=True, timeout=300,
                                 creationflags=0x08000000 if os.name == 'nt' else 0)
            (args.output / (label + '.log')).write_text(run.stdout + run.stderr, encoding='utf-8')
            if run.returncode:
                print(run.stdout + run.stderr, flush=True)
                raise RuntimeError('native_step_failed:' + label)
            if args.runner and name in faults:
                reason = re.search(r'terminal=6 error=(\d+)', run.stdout)
                if not reason or int(reason[1]) != 0x100 + faults[name]:
                    raise ValueError('native_fault_reason:' + name)
            checked = check_lags(artifact, root / (name + '.final-lags.json'), dump) if group == 'oracle' else 0
            actual_steps = re.search(r'free_running_steps=(\d+)', run.stdout) if args.runner else None
            if args.runner and not actual_steps:
                raise ValueError('native_runner_evidence')
            result = {'case': label, 'steps': int(actual_steps[1]) if args.runner else len(stream),
                      'expected_stream_steps': len(stream), 'lag_values': checked, 'passed': True,
                      'expected_terminal': 'fault' if args.runner and name in faults else 'success',
                      'level': ('native CUDA compiled FFN/GDN; autonomous from prefix/seed' if args.runner
                                else 'native CUDA compiled FFN/GDN; supplied trace replay')}
            results.append(result)
            print(json.dumps(result), flush=True)
    if not results or (selected and not selected.issubset({r['case'].split('-', 1)[1] for r in results})):
        raise ValueError('fixture_selection')
    if file_digest(args.binary, 64 << 20)['sha256'] != binary_digest:
        raise ValueError('binary_changed_during_validation')
    report = {'results': results, 'binary_sha256': binary_digest,
              'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
              'free_running': args.runner, 'language_model_integrated': False}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'PASS {len(results)} native CUDA development ' + ('autonomous runs' if args.runner else 'replays'), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
