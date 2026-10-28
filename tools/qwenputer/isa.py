"""First-party synthetic RV datapath vectors, never a downloaded firmware binary."""
from tools.qwenputer.features import FAMILIES

BASES = {}
_base = 0
for _name, _size in FAMILIES:
    BASES[_name] = _base
    _base += _size


def tok(name, payload=0):
    return BASES[name] + payload


def value_tokens(value):
    value &= 0xffffffff
    signed = value if value < 0x80000000 else value - 0x100000000
    if -16384 <= signed < 16384:
        return [tok('F0', value & 32767)]
    if -536870912 <= signed < 536870912:
        return [tok('V', value & 32767), tok('F1', (value >> 15) & 32767)]
    return [tok('V', value & 32767), tok('V', (value >> 15) & 32767), tok('F2', value >> 30)]


def rword(f7, f3):
    return f7 << 25 | 2 << 20 | 1 << 15 | f3 << 12 | 3 << 7 | 0x33


def iword(imm, rs1, f3, rd, opcode):
    return (imm & 4095) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | opcode


def sword(f3):
    return 2 << 20 | 1 << 15 | f3 << 12 | 0x23


def register_value(row, columns):
    from tools.qwenputer.tokenizer import NativeInputError
    bits = [sum(float(row[c]) for c in columns[b]) / 128 for b in range(32)]
    if any(bit not in (0, 1) for bit in bits):
        raise NativeInputError('register_bit')
    return sum(int(bit) << i for i, bit in enumerate(bits))


def instruction_cases():
    cases = []
    def add(name, word, a=0, b=0, result=None, record=None, **extra):
        record = list(record if record is not None else value_tokens(result))
        cases.append(dict(name=name, word=word, a=a, b=b, expected_register3=result,
                          record=record, **extra))
    for name, f7, f3, a, b, result in (
        ('add-overflow', 0, 0, 0x7fffffff, 1, 0x80000000),
        ('sub', 0x20, 0, 0, 1, 0xffffffff), ('and', 0, 7, 0xa5, 0x3c, 0x24),
        ('or', 0, 6, 0xa5, 0x3c, 0xbd), ('xor', 0, 4, 0xa5, 0x3c, 0x99),
        ('slt', 0, 2, 0xfffffff9, 3, 1), ('sltu', 0, 3, 0xfffffff9, 3, 0),
        ('sll', 0, 1, 0x80000001, 4, 16), ('srl', 0, 5, 0x80000001, 4, 0x08000000),
        ('sra', 0x20, 5, 0x80000001, 4, 0xf8000000),
        ('mul', 1, 0, 0xfffffff9, 3, 0xffffffeb),
        ('mulh-signed', 1, 1, 0xfffffff9, 3, 0xffffffff),
        ('mulhsu', 1, 2, 0xfffffff9, 3, 0xffffffff),
        ('mulhu', 1, 3, 0xfffffff9, 3, 2),
        ('sh1add', 0x10, 2, 7, 3, 17), ('sh2add', 0x10, 4, 7, 3, 31),
        ('sh3add', 0x10, 6, 7, 3, 59)):
        add(name, rword(f7, f3), a, b, result)
    add('addi', iword(-9, 1, 0, 3, 0x13), 7, result=0xfffffffe)
    add('lui', 0x12345 << 12 | 3 << 7 | 0x37, result=0x12345000)
    add('auipc', 1 << 12 | 3 << 7 | 0x17, result=0x41000)
    add('misa', 0x301 << 20 | 2 << 12 | 3 << 7 | 0x73, result=0x40001100)
    add('fence', 0xf, result=0, record=[tok('NT')])
    for name, f3, address, result in (('lb', 0, 0x103, 0xffffff80), ('lbu', 4, 0x103, 128),
                                     ('lh', 1, 0x102, 0xffff80ff), ('lhu', 5, 0x102, 0x80ff),
                                     ('lw', 2, 0x100, 0x80ff7f01)):
        add('load-' + name, iword(0, 1, f3, 3, 3), address, result=result, ram_word=0x80ff7f01)
    for name, f3, address, value, merged in (
        ('byte', 0, 0x101, 0x11, 0xaabb11dd), ('half', 1, 0x102, 0x1122, 0x1122ccdd),
        ('word', 2, 0x100, 0x11223344, 0x11223344), ('secondary-half', 1, 0x142, 0x1122, 0x1122ccdd)):
        half = (address >> 6) & 1
        store = [tok('ST' + str(half))] if f3 == 2 else [tok('V', merged & 32767),
                        tok('V', (merged >> 15) & 32767), tok('V', merged >> 30), tok('SS' + str(half))]
        add('store-' + name, sword(f3), address, value, merged,
            record=store + value_tokens(merged), ram_word=0xaabbccdd,
            following=[iword(0, 4, 2, 3, 3)])
    # Branch target is PC+8, both fall-through and target contain a HALT ecall.
    for name, f3, a, b, taken in (('beq', 0, 7, 7, True), ('bne', 1, 7, 3, True),
                                 ('blt', 4, 0xffffffff, 3, True), ('bge', 5, 3, 0xffffffff, True),
                                 ('bltu', 6, 3, 0xffffffff, True), ('bgeu', 7, 0xffffffff, 3, True),
                                 ('beq-false', 0, 7, 3, False)):
        word = 2 << 20 | 1 << 15 | f3 << 12 | 4 << 8 | 0x63
        add(name, word, a, b, 0, [tok('J', 2) if taken else tok('NT')], following=[0x73])
    add('jal', 8 << 20 | 3 << 7 | 0x6f, result=0x40004, record=[tok('J', 2)], following=[0x73])
    add('jalr', iword(0, 1, 0, 3, 0x67), 0x40008, result=0x40004,
        record=[tok('J', 2)], following=[0x73])
    add('refresh', rword(0, 0), 17, 23, 40,
        record=[tok('R', 63)] + value_tokens(40), initial_n=63)
    for fault, word, a, b, a7 in (
        (0, 0, 0, 0, 93), (1, 0x100073, 0, 0, 93),
        (2, iword(0, 1, 1, 3, 3), 0x101, 0, 93), (3, sword(1), 0x101, 0, 93),
        (4, iword(0, 1, 2, 3, 3), 0x9000, 0, 93), (5, sword(2), 0x8000, 0, 93),
        (6, iword(0, 1, 0, 3, 0x67), 0x40002, 0, 93), (7, 0x73, 0, 0, 2)):
        record = [tok('FLT', fault)] + [tok('HEX', (0x40000 >> (4 * (7 - k))) & 15) for k in range(8)]
        add('fault-' + str(fault), word, a, b, None, record=record, fault=fault, a7=a7)
    add('putchar', 0x73, result=93, record=[tok('BYTE', 65), -65] + value_tokens(93) + value_tokens(93),
        following=[iword(93, 0, 0, 3, 0x13), iword(93, 0, 0, 17, 0x13)], a7=1, a0=65)
    return cases


def main(argv=None):
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import sys
    import numpy as np
    from tools.qwenputer.export import canonical_json
    from tools.qwenputer.oracle import decode_prediction
    from tools.qwenputer.source import require_compiler_environment, verify_source, MANIFEST_SHA256
    from tools.qwenputer.tokenizer import NativeText, NativeInputError
    from tools.qwenputer.features import TEXT_TOKEN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'third_party/qwenputer')
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute-reviewed-compiler', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute_reviewed_compiler:
        raise NativeInputError('execution_not_authorized')
    require_compiler_environment()
    verify_source(args.source_root)
    if args.output.exists() or any(n == 'chipformer' or n.startswith('chipformer.') for n in sys.modules):
        raise NativeInputError('isa_output_or_import')
    text = NativeText.from_pack(args.tokenizer_dir)
    sys.path.insert(0, str(args.source_root.resolve() / 'experimental/riscv'))
    from chipformer.rv import core, ref
    from chipformer.q35 import ir
    circuit = core.build(text=text)
    cases = instruction_cases()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'cases.json').write_bytes(canonical_json(cases))
    owner_dim = next(iter(circuit.owner.terms))

    def dense_state(carry=None):
        result = []
        for ni, node in enumerate(circuit.nodes):
            if type(node).__name__ == 'Head':
                for vi, _ in enumerate(node.vheads()):
                    rows = ((node.init[vi] if node.init is not None else {}) if carry is None
                            else carry.get((ni, 'S', vi), {}))
                    data = np.zeros((128, 128), dtype='<f4')
                    for row, values in rows.items():
                        data[row] = values
                    result.append(data)
        return np.stack(result)

    results = []
    for case in cases:
        start = case.get('initial_n', 0)
        rom = [0] * 65536
        words = [case['word'], *case.get('following', []), 0x73]
        rom[start:start + len(words)] = words
        ram = [0] * 8192
        ram[(case['a'] & ~3) // 4 if 0x100 <= case['a'] < 0x180 else 64] = case.get('ram_word', 0)
        x = [0] * 32
        x[1], x[2], x[4] = case['a'], case['b'], case['a'] & ~3
        x[10], x[17] = case.get('a0', 0), case.get('a7', 93)
        # This constructs table data only. Never call Machine.record/run or load
        # an ELF for these first-party one-instruction datapath fixtures.
        machine = ref.Machine(rom=rom, ram=ram, x=x, anchor=0, n=start)
        core.snapshot(circuit, machine)
        (args.output / (case['name'] + '.init.f32')).write_bytes(dense_state().tobytes())
        stream = [tok('GO')] + [TEXT_TOKEN + text.byte_token[-t] if t < 0 else t for t in case['record']]
        stream += [tok('HALT'), TEXT_TOKEN + text.NL]
        carry = {}
        matrix = ir.evaluate(circuit, stream, check=True, carry=carry, kernel=False)
        for i, row in enumerate(matrix[:-1]):
            predicted = decode_prediction(row, circuit.keep, owner_dim, text)
            if predicted != stream[i + 1]:
                raise NativeInputError(f'isa_prediction:{case["name"]}:{i}:{predicted}:{stream[i + 1]}')
        if case['expected_register3'] is not None:
            node = circuit.tables['regs.rs1']
            ni = next(i for i, n in enumerate(circuit.nodes) if n is node)
            row = carry[(ni, 'S', 0)].get(3, np.zeros(128))
            actual = register_value(row, circuit.reg_cols)
            if actual != case['expected_register3']:
                raise NativeInputError(f'isa_register:{case["name"]}:{actual}:{case["expected_register3"]}')
        (args.output / (case['name'] + '.stream.json')).write_bytes(canonical_json(stream))
        (args.output / (case['name'] + '.residual.f32')).write_bytes(matrix.astype('<f4').tobytes())
        (args.output / (case['name'] + '.final.f32')).write_bytes(dense_state(carry).tobytes())
        results.append({'name': case['name'], 'tokens': len(stream), 'passed': True})
        print('IR ISA PASS ' + case['name'], flush=True)
    (args.output / 'results.json').write_bytes(canonical_json(results))
    manifest = {'kind': 'first-party-synthetic-IR-datapath-tests', 'source_sha256': MANIFEST_SHA256,
                'new_external_firmware_executed': False, 'native_cuda_tested': False,
                'cases': len(results), 'tokenizer_sha256': text.binding_sha256,
                'files': {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in sorted(args.output.iterdir()) if p.is_file()}}
    (args.output / 'manifest.json').write_bytes(canonical_json(manifest))
    print(f'PASS {len(results)} synthetic instruction/fault fixtures; original IR, not Native CUDA', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
