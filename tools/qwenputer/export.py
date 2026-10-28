"""Compile reviewed IR into a data-only native sparse-circuit artifact."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

from tools.qwenputer.features import TRACE_VOCAB, lower_feature
from tools.qwenputer.tokenizer import NativeInputError, NativeText

MATH_ABI = ('sparse-f32-v1;ordered-dot-mul-add;no-fma;no-ftz;'
            'ordered-ffn-and-output-reduction;head128;lag3;decay-1-half-zero;'
            'overwrite-beta;query-sum-in-map-order;ref-cols0-1;sqrt-rn;div-rn')
ABI_SHA256 = hashlib.sha256(MATH_ABI.encode('ascii')).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def fp32(value):
    try:
        result = struct.unpack('<f', struct.pack('<f', value))[0]
    except (ValueError, TypeError, OverflowError, struct.error) as error:
        raise NativeInputError('coefficient_fp32') from error
    if not math.isfinite(result) or result != value:
        raise NativeInputError('coefficient_fp32')
    return result


def records(fmt, values):
    pack = struct.Struct(fmt)
    return pack.size, b''.join(pack.pack(*row) for row in values)


def compile_sections(circuit, text, identity):
    """Pure lowering: imports no compiler and executes no firmware/callbacks."""
    dims = circuit.dims
    if not 1 <= len(dims) <= 16384 or len(circuit.nodes) > 16384:
        raise NativeInputError('circuit_shape')
    digests = dict(identity, abi_sha256=ABI_SHA256)
    required = ('abi_sha256', 'source_sha256', 'firmware_sha256', 'boot_sha256',
                'model_sha256', 'tokenizer_sha256')
    if any(not isinstance(digests.get(k), str) or not re.fullmatch('[a-f0-9]{64}', digests[k])
           for k in required) or digests['tokenizer_sha256'] != text.binding_sha256:
        raise NativeInputError('artifact_binding')
    exprs, terms, channels, maps, outputs = [], [], [], [], []
    ffns, ffn_terms, heads, head_levels, initial = [], [], [], [], []
    expressions, channel_ids = {}, {}

    def dim_index(value):
        if type(value) is not int or not 0 <= value < len(dims):
            raise NativeInputError('expression_dim')
        return value

    def expression(expr):
        row = tuple((dim_index(d), fp32(c)) for d, c in expr.terms.items())
        # The original order is load-bearing for the FP32 numerical contract.
        key = b''.join(struct.pack('<if', *term) for term in row)
        if key not in expressions:
            expressions[key] = len(exprs)
            exprs.append((len(terms), len(row)))
            terms.extend(row)
        return expressions[key]

    def channel(c):
        taps = tuple(fp32(x) for x in c.taps)
        if len(taps) != 4:
            raise NativeInputError('channel_taps')
        row = (expression(c.expr), *taps)
        key = struct.pack('<i4f', *row)
        if key not in channel_ids:
            channel_ids[key] = len(channels)
            channels.append(row)
        return channel_ids[key]

    def mapping(values, convert):
        start = len(maps)
        for lane, value in values.items():
            if type(lane) is not int or not 0 <= lane < 128:
                raise NativeInputError('head_lane')
            maps.append((lane, convert(value)))
        return start, len(values)

    dim_kinds = {'one': 0, 'token': 1, 'ffn': 2, 'head': 3}
    dimensions = []
    for dim in dims:
        if dim.kind not in dim_kinds or type(dim.level) is not int or not 0 <= dim.level <= 128:
            raise NativeInputError('dimension_kind_level')
        dimensions.append((dim_kinds[dim.kind], dim.level))
    features = [(0, *lower_feature('one'))] + [
        (d, *lower_feature(dims[d].name)) for d in circuit.token_fns]
    state_offset = 0
    for node_index, node in enumerate(circuit.nodes):
        if type(node.level) is not int or not 1 <= node.level <= 128:
            raise NativeInputError('node_level')
        if type(node).__name__ == 'FFN':
            if node.level % 2:
                raise NativeInputError('ffn_level')
            start = len(ffn_terms)
            for coefficient, a, b in node.terms:
                ffn_terms.append((fp32(coefficient), expression(a), expression(b)))
            ffns.append((node_index, node.level, dim_index(node.out), int(node.into), start, len(node.terms)))
        elif type(node).__name__ == 'Head':
            if node.level % 2 != 1 or node.n_cols != 128 or node.levels not in ('bit', 'byte', 'decay'):
                raise NativeInputError('head_geometry')
            qstart, qcount = mapping(node.q, channel)
            kstart, kcount = mapping(node.k, channel)
            value_heads = node.vheads()
            if not 1 <= len(value_heads) <= 2 or (node.init is not None and len(node.init) != len(value_heads)):
                raise NativeInputError('head_values')
            for vi, vh in enumerate(value_heads):
                vstart, vcount = mapping(vh.v, channel)
                zstart, zcount = mapping(vh.z, expression)
                ostart = len(outputs)
                for col, destination, weight in vh.out:
                    if type(col) is not int or not 0 <= col < 128:
                        raise NativeInputError('head_column')
                    outputs.append((col, dim_index(destination), fp32(weight)))
                initstart = len(initial)
                for row, values in sorted((node.init[vi] if node.init is not None else {}).items()):
                    if type(row) is not int or not 0 <= row < 128 or len(values) != 128:
                        raise NativeInputError('head_init')
                    vals = tuple(fp32(v) for v in values)
                    if (any(v not in (0, 2 ** 24) for v in vals[:2])
                            or (node.levels == 'bit' and any(v not in (0, 128) for v in vals[2:]))
                            or (node.levels == 'byte' and any(v != int(v) or not 0 <= v <= 255 for v in vals[2:]))):
                        raise NativeInputError('head_init_levels')
                    initial.append((len(heads), row, *vals))
                heads.append((node_index, node.level, vi, int(vh.into), 128, 128,
                              qstart, qcount, kstart, kcount, vstart, vcount, zstart, zcount,
                              ostart, len(vh.out), expression(vh.a), -1 if vh.beta is None else expression(vh.beta),
                              initstart, len(initial) - initstart, state_offset, fp32(node.tiny)))
                head_levels.append({'bit': 0, 'byte': 1, 'decay': 2}[node.levels])
                state_offset += 128 * 128
        else:
            raise NativeInputError('unknown_node')
    keep = [dim_index(d) for d in getattr(circuit, 'keep', [])]
    owner = getattr(circuit, 'owner', None)
    if owner is not None and (len(owner.terms) != 1 or next(iter(owner.terms.values())) != 1):
        raise NativeInputError('owner_dimension')
    owner_dim = dim_index(next(iter(owner.terms))) if owner is not None else 0xffffffff
    sections = {
        'dims': records('<2i', dimensions), 'features': records('<4i', features),
        'exprs': records('<2i', exprs), 'terms': records('<if', terms),
        'chans': records('<i4f', channels), 'maps': records('<2i', maps),
        'ffns': records('<6i', ffns), 'ffn_terms': records('<fii', ffn_terms),
        'heads': records('<21if', heads), 'head_levels': (1, bytes(head_levels)),
        'outputs': records('<iif', outputs),
        'init': records('<2i128f', initial), 'keep': records('<i', ((d,) for d in keep)),
        'text_rows': (16, b''.join(text.row_bytes(q) for q in range(text.n))),
        'text_lens': records('<I', ((len(row),) for row in text.bytes)),
        'text_types': (1, bytes(text.token_types)),
        'printable': records('<i', ((q,) for q in text.printable)),
        'identity': (32, b''.join(bytes.fromhex(digests[k]) for k in required)),
        'config': records('<16I', [(len(dims), TRACE_VOCAB, text.n, len(heads), len(ffns),
                                  max((n.level for n in circuit.nodes), default=0), state_offset,
                                  text.TCO, text.TCC, text.NL, 80, 16, 255, 128, 3, owner_dim)]),
    }
    metadata = {'schema': 'STQPUT01-1.0', 'identities': digests, 'math_abi': MATH_ABI,
                'compiler_patches': [], 'feature_names': [dims[d].name for d, *_ in features],
                'compiler_environment': {'python_hash_seed': '0', 'hash_randomization': 0},
                'dim_names': [d.name for d in dims], 'node_names': [n.name for n in circuit.nodes],
                'limits': {'artifact_bytes': 268435456, 'dimensions': 16384, 'nodes': 16384,
                           'trace_steps': 200000, 'output_text_tokens': 4096},
                'native_runtime': 'sparse-ffn-gdn', 'language_model_included': False,
                'model_geometry': {'architecture': 'qwen4exp', 'layers': 48, 'hidden': 2560},
                'firmware_distribution': 'local-test-only; linked-runtime notices not yet verified'}
    sections['meta'] = (1, canonical_json(metadata))
    return sections


def export_artifact(source_dir, tokenizer_dir, firmware_path, boot_dir, output_dir,
                    model_sha256, execute_reviewed_reference=False, progress=None):
    if execute_reviewed_reference is not True:
        raise NativeInputError('execution_not_authorized')
    from tools.qwenputer.source import require_compiler_environment
    require_compiler_environment()
    from tools.qwenputer.bootstrap import preflight, capture_boot
    from tools.qwenputer.features import check_features
    from tools.qwenputer.format import encode_sections, decode_sections
    from tools.qwenputer.source import REVISION, MANIFEST_SHA256, verify_source
    source_dir, boot_dir, output_dir = map(Path, (source_dir, boot_dir, output_dir))
    report = preflight(source_dir, tokenizer_dir, firmware_path)
    if report['firmware_sha256'] != 'b643ad61291a947fa6c7498de9170b1d4da062dc1f34810feba9c4eebdec862d':
        raise NativeInputError('firmware_execution_scope')
    if not re.fullmatch('[a-f0-9]{64}', model_sha256) or output_dir.exists():
        raise NativeInputError('export_output_binding')

    def bounded_read(path, limit):
        with path.open('rb') as file:
            data = file.read(limit + 1)
        if len(data) > limit:
            raise NativeInputError('bootstrap_size')
        return data

    manifest = json.loads(bounded_read(boot_dir / 'manifest.json', 65536))
    state_data = bounded_read(boot_dir / 'boot-state.json', 4 * 1024 * 1024)
    trace_data = bounded_read(boot_dir / 'boot-trace.json', 4 * 1024 * 1024)
    for name, data in (('boot-state.json', state_data), ('boot-trace.json', trace_data)):
        if manifest.get(name) != {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}:
            raise NativeInputError('bootstrap_digest')
    if (manifest.get('source_revision') != REVISION
            or manifest.get('source_manifest_sha256') not in (MANIFEST_SHA256,
                    '69cb09356ea256f3094173779b7ee6391b8f3906f403d394f7107929e43d55c8')
            or manifest.get('firmware_sha256') != report['firmware_sha256']
            or manifest.get('tokenizer_binding_sha256') != report['tokenizer_binding_sha256']):
        raise NativeInputError('bootstrap_binding')
    state, trace = json.loads(state_data), json.loads(trace_data)
    if any(name == 'chipformer' or name.startswith('chipformer.') for name in sys.modules):
        raise NativeInputError('compiler_already_loaded')
    text = NativeText.from_pack(tokenizer_dir)
    verify_source(source_dir)
    previous = list(sys.path)
    sys.path.insert(0, str(source_dir.resolve() / 'experimental/riscv'))
    try:
        from chipformer.rv import core, ref
        machine = ref.Machine(rom=state['rom'], ram=state['ram'], x=state['registers'],
                              anchor=state['anchor'], n=state['n'], drom=bytes.fromhex(state['drom_hex']),
                              bn=state['digits'], halted=True, exit_code=0)
        if capture_boot(machine, trace, 200000) != state:
            raise NativeInputError('bootstrap_state')
        circuit = core.build(text=text)
        core.snapshot(circuit, machine)
        lowered, feature_proof = check_features(circuit, text, progress)
        identity = {'source_sha256': MANIFEST_SHA256, 'firmware_sha256': report['firmware_sha256'],
                    'boot_sha256': hashlib.sha256(state_data).hexdigest(),
                    'model_sha256': model_sha256, 'tokenizer_sha256': text.binding_sha256}
        sections = compile_sections(circuit, text, identity)
        artifact = encode_sections(sections)
        if decode_sections(artifact) != sections:
            raise NativeInputError('artifact_roundtrip')
        result = {'version': 1, 'kind': 'offline-native-circuit-export', 'native_execution_tested': False,
                  'revision': REVISION, 'identities': dict(identity, abi_sha256=ABI_SHA256),
                  'stats': circuit.stats(), 'feature_proof': feature_proof,
                  'artifact_sha256': hashlib.sha256(artifact).hexdigest(), 'artifact_bytes': len(artifact),
                  'sections': {k: {'stride': v[0], 'records': len(v[1]) // v[0],
                                   'sha256': hashlib.sha256(v[1]).hexdigest()} for k, v in sorted(sections.items())},
                  'oracle_status': 'pending', 'redistribution': 'blocked-pending-firmware-linked-runtime-notices'}
    finally:
        sys.path[:] = previous
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'native.qput').write_bytes(artifact)
    (output_dir / 'manifest.json').write_bytes(canonical_json(result))
    (output_dir / 'LICENSE.compiler').write_bytes((source_dir / 'LICENSE').read_bytes())
    (output_dir / 'UPSTREAM.json').write_bytes((source_dir / 'UPSTREAM.json').read_bytes())
    (output_dir / 'NOTICE.txt').write_text(
        'Local tests only. Compiler: MIT, see LICENSE.compiler and UPSTREAM.json.\n'
        'Firmware digest is pinned, but its picolibc/libgcc linked-runtime provenance and notices\n'
        'have not been verified. Do not redistribute this generated artifact.\n', encoding='utf-8')
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'third_party/qwenputer')
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--firmware', type=Path, required=True)
    parser.add_argument('--boot-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--execute-reviewed-reference', action='store_true')
    args = parser.parse_args(argv)
    result = export_artifact(args.source_root, args.tokenizer_dir, args.firmware, args.boot_dir,
                             args.output, args.model_sha256, args.execute_reviewed_reference,
                             lambda n, total: print(f'features {n}/{total}', flush=True))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
