"""First-party FFN/GDN operator fixtures, with analytic expected values and no firmware."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

from tools.qwenputer.features import FAMILIES, feature_values
from tools.qwenputer.format import decode_sections, encode_sections


def generate(base_artifact, output):
    import numpy as np
    if output.exists():
        raise ValueError('output_exists')
    sections = decode_sections(base_artifact.read_bytes())
    records = lambda name, fmt: list(struct.iter_unpack(fmt, sections[name][1]))
    features = records('features', '<4i')
    config = list(records('config', '<16I')[0])
    dim = lambda opcode, p0, p1: next(d for d, op, a, b in features if (op, a, b) == (opcode, p0, p1))
    go_dim, b0, b1 = dim(2, 18, 0), dim(3, 3, 0), dim(3, 3, 1)
    base, bases = 0, {}
    for name, size in FAMILIES:
        bases[name] = base
        base += size
    stream = [bases['F2'], bases['GO'], bases['F2'] + 1, bases['F2'] + 2, bases['F2']]
    output.mkdir(parents=True)
    cases = []
    for hot, key_always in ((0, True), (1, True), (2, True), (128, True), (2, False)):
        name = f'q{hot}-key' + ('-always' if key_always else '-gated')
        root = output / name
        root.mkdir()
        s = dict(sections)
        expressions, terms, channels = records('exprs', '<ii'), records('terms', '<if'), records('chans', '<i4f')
        def expression(row):
            e = len(expressions)
            expressions.append((len(terms), len(row)))
            terms.extend(row)
            return e
        def channel(e):
            c = len(channels)
            channels.append((e, 1., 0., 0., 0.))
            return c
        one = expression([(0, 1.)])
        q = channel(expression([(0, 128.)]))
        go = expression([(go_dim, 1.)])
        key = q if key_always else channel(expression([(go_dim, 128.)]))
        ref = channel(expression([(0, float(1 << 24))]))
        value = channel(expression([(go_dim, 128.)]))
        gate = expression([(0, 128.)])
        decay = expression([(b0, 110.), (b1, 238.)])
        head_value = expression([(24, 1.)])
        maps = [(lane, q) for lane in reversed(range(hot))]
        maps += [(0, key), (0, ref), (1, ref), (2, value), (2, gate)]
        init = []
        state = np.zeros((128, 128), dtype=np.float64)
        state[:, :2] = 1 << 24
        for row in range(128):
            init.append((0, row, *state[row]))
        head = (0, 1, 0, 0, 128, 128, 0, hot, hot, 1, hot + 1, 3,
                hot + 4, 1, 0, 1, decay, go, 0, 128, 0, 0.)
        def pack(name, fmt, rows):
            s[name] = struct.calcsize(fmt), b''.join(struct.pack(fmt, *r) for r in rows)
        pack('exprs', '<ii', expressions)
        pack('terms', '<if', terms)
        pack('chans', '<i4f', channels)
        pack('maps', '<ii', maps)
        pack('heads', '<21if', [head])
        s['head_levels'] = 1, bytes([0])
        pack('init', '<2i128f', init)
        pack('outputs', '<iif', [(2, 24, 1.)])
        pack('ffn_terms', '<fii', [(256., head_value, one), (3., one, one)])
        pack('ffns', '<6i', [(1, 2, 25, 0, 0, 1), (2, 2, 25, 1, 1, 1)])
        c = list(config)
        c[3], c[4], c[5], c[6] = 1, 2, 2, 16384
        pack('config', '<16I', [c])
        identity = bytearray(s['identity'][1])
        identity[32:64] = hashlib.sha256(Path(__file__).read_bytes()).digest()
        identity[64:96] = hashlib.sha256(b'no firmware; first-party analytic operator fixture').digest()
        identity[96:128] = hashlib.sha256(state.astype('<f4').tobytes()).digest()
        s['identity'] = 32, bytes(identity)
        s['meta'] = 1, json.dumps({'kind': 'first-party analytic FFN/GDN unit fixture',
                                  'firmware_executed': False, 'queries': hot, 'key_always': key_always}).encode('ascii')
        artifact = encode_sections(s)
        (root / 'native.qput').write_bytes(artifact)
        (root / 'binding.bin').write_bytes(identity)
        (root / 'inputs.bin').write_bytes(b''.join(struct.pack('<II', 0, t) for t in stream))
        matrix = np.zeros((len(stream), config[0]), dtype=np.float64)
        text = SimpleNamespace(n=config[2])
        for d, *descriptor in features:
            matrix[:, d] = feature_values(tuple(descriptor), 0, np.array(stream, dtype=np.int64), text)
        # Independent analytic expectation, not the runtime expression evaluator.
        # GO writes row zero; F2 selects keep, halve, clear. A present key with
        # beta zero must not overwrite the state. Queries read 0/1/2/128 rows.
        for i, token in enumerate(stream):
            if token == bases['F2'] + 1:
                state *= .5
            elif token == bases['F2'] + 2:
                state *= 0
            if token == bases['GO']:
                state[0] = 0
                state[0, :2] = 1 << 24
                state[0, 2] = 128
            data_sum = sum(state[row, 2] for row in reversed(range(hot)))
            reference_sum = sum(state[row, 0] for row in reversed(range(hot)))
            result = 0 if not reference_sum else data_sum * 8 / reference_sum * 128
            matrix[i, 24] = result
            matrix[i, 25] = result * 256 + 3
        (root / 'residual.f32').write_bytes(matrix.astype('<f4').tobytes())
        (root / 'final.f32').write_bytes(state.astype('<f4').tobytes())
        cases.append(name)
    (output / 'cases.json').write_text(json.dumps(cases), encoding='utf-8')
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(generate(args.artifact, args.output)))


if __name__ == '__main__':
    main()
