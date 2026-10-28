"""Test-only independent mathematical oracle. Never import from serving code.

Python's AST and Fraction provide an implementation independent of the RV
firmware's shunting-yard parser and decimal digit-file algorithms. Output is a
canonical rational value, not a claim about firmware decimal display spelling.
"""
import ast
from fractions import Fraction
import math
import re

from tools.qwenputer.tokenizer import NativeInputError

DEV_SEED = 20261028
HELD_OUT_SEED = 20261103
DEVELOPMENT_CASES = (
    ('multiply', '17*23'), ('negative-floor', '-7//3'), ('negative-modulo', '-7%3'),
    ('negative-divisor', '7%-3'), ('precedence', '2**3**2'), ('unary', '-2**2'),
    ('rational', '1/3+1/6'), ('decimal', '0.1+0.2'), ('gcd', 'gcd(48,18)'),
    ('pow', '2**100'), ('limit-valid', '10**60'), ('zero-division', '1/0'),
    ('digit-overflow', '10**62'), ('huge-exponent', '2**100000000'),
    ('syntax-error', '1+'), ('complexity', '(' * 16 + '1' + ')' * 16),
    ('literal-62', '1' + '0' * 61), ('subtract-62', '1' + '0' * 61 + '-1'),
    ('capacity-pow', '10**61'), ('capacity-add', '1' + '0' * 61 + '+0'),
    ('capacity-mul', '1' + '0' * 61 + '*1'),
)
KNOWN_CAPACITY_REFUSALS = {
    '10**61': 'upstream conservative multiplication capacity: final operand lengths 30+33 > MAXD62',
    '1' + '0' * 61 + '+0': 'upstream conservative addition capacity: max operand length62 +1 > MAXD62',
    '1' + '0' * 61 + '*1': 'upstream conservative multiplication capacity: operand lengths62+1 > MAXD62',
}


def development_cases():
    """Freeze this corpus before Native implementation; held-out cases stay separate."""
    import random
    rng = random.Random(DEV_SEED)
    cases = list(DEVELOPMENT_CASES)
    for i in range(8):
        a, b = rng.randint(-9999, 9999), rng.randint(1, 999)
        cases.append((f'seeded-{i}', f'({a}){rng.choice(("+", "-", "*", "//", "%"))}{b}'))
    return cases


def held_out_cases():
    import random
    rng = random.Random(HELD_OUT_SEED)
    cases, seen = [], set()
    while len(cases) < 64:
        a, b, c = rng.randint(-99999, 99999), rng.randint(1, 9999), rng.randint(-999, 999)
        expression = f'({a}){rng.choice(("+", "-", "*", "//", "%", "/"))}{b}'
        if len(cases) % 4 == 0:
            expression = f'({expression})+({c})'
        if expression in seen:
            continue
        seen.add(expression)
        cases.append({'id': f'held-{len(cases):02d}', 'expression': expression,
                      'expected_rational': exact_answer(expression), 'firmware_executed': False})
    return cases


def exact_answer(expression):
    if not isinstance(expression, str) or not 1 <= len(expression.encode('utf-8')) <= 255:
        raise NativeInputError('oracle_input')
    source = expression.replace('^', '**').strip()
    if not re.fullmatch(r'[0-9+*/%()., gcd\-]+', source):
        raise NativeInputError('oracle_syntax')
    # Firmware numbers are decimal even with leading zeroes; Python's parser
    # rejects 001. Normalize only integer literal spelling, never evaluate it.
    source = re.sub(r'(?<![a-z0-9_.])([0-9]+)(?![0-9.])',
                    lambda m: m[1].lstrip('0') or '0', source)
    try:
        root = ast.parse(source, mode='eval')
    except (SyntaxError, RecursionError) as error:
        raise NativeInputError('oracle_syntax') from error
    if sum(1 for _ in ast.walk(root)) > 128:
        raise NativeInputError('oracle_complexity')

    def bounded(value):
        if max(len(str(abs(value.numerator))), len(str(value.denominator))) > 62:
            raise NativeInputError('oracle_digits')
        return value

    def evaluate(node, depth=0):
        if depth > 16:
            raise NativeInputError('oracle_complexity')
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            token = ast.get_source_segment(source, node)
            if not re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', token):
                raise NativeInputError('oracle_number')
            return bounded(Fraction(token))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = evaluate(node.operand, depth + 1)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'gcd':
            if len(node.args) != 2 or node.keywords:
                raise NativeInputError('oracle_gcd')
            a, b = [evaluate(n, depth + 1) for n in node.args]
            if a.denominator != 1 or b.denominator != 1:
                raise NativeInputError('oracle_integer')
            return Fraction(math.gcd(a.numerator, b.numerator))
        if not isinstance(node, ast.BinOp):
            raise NativeInputError('oracle_syntax')
        a, b = evaluate(node.left, depth + 1), evaluate(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            value = a + b
        elif isinstance(node.op, ast.Sub):
            value = a - b
        elif isinstance(node.op, ast.Mult):
            value = a * b
        elif isinstance(node.op, ast.Div):
            value = a / b
        elif isinstance(node.op, (ast.FloorDiv, ast.Mod, ast.Pow)):
            if a.denominator != 1 or b.denominator != 1:
                raise NativeInputError('oracle_integer')
            if isinstance(node.op, ast.Pow):
                if not 0 <= b <= 999:
                    raise NativeInputError('oracle_exponent')
                value = a ** int(b)
            elif isinstance(node.op, ast.FloorDiv):
                value = Fraction(a // b)
            else:
                value = a % b
        else:
            raise NativeInputError('oracle_operator')
        return bounded(value)

    try:
        value = evaluate(root.body)
    except (ZeroDivisionError, OverflowError, RecursionError) as error:
        raise NativeInputError('oracle_domain') from error
    return str(value)


def reference_case(machine, text, expression, max_trace_tokens=200000):
    from tools.qwenputer.call import CallContext, eligible_call
    from tools.qwenputer.features import FAMILIES, TEXT_TOKEN
    import time
    call = '\n<function=calc><parameter=expression>' + expression + '</parameter></function>\n'
    qids = text.encode(call)
    eligible_call(text, qids, CallContext(live_assistant=True, enrolled_calc=True))
    bases, base = {}, 0
    for name, size in FAMILIES:
        bases[name] = (base, size)
        base += size
    stream = [TEXT_TOKEN + text.TCO] + [TEXT_TOKEN + q for q in qids] + [TEXT_TOKEN + text.TCC, bases['GO'][0]]
    machine.text = [bytes(16) for _ in range(128)]
    for i, qid in enumerate(qids, 1):
        machine.text[i] = text.row_bytes(qid)
    output, trace_tokens = [], 1
    started = time.perf_counter()
    while not machine.halted:
        record = machine.record()
        if trace_tokens + len(record) > max_trace_tokens:
            raise NativeInputError('reference_trace_limit')
        trace_tokens += len(record)
        for token in record:
            stream.append(token)
            if bases['BYTE'][0] <= token < bases['BYTE'][0] + bases['BYTE'][1]:
                payload = token - bases['BYTE'][0]
                if not 0 <= payload < len(text.printable):
                    raise NativeInputError('reference_byte')
                output.append(text.printable[payload])
                stream.append(TEXT_TOKEN + text.printable[payload])
            elif token == bases['HALT'][0]:
                stream.append(TEXT_TOKEN + text.NL)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if machine.fault is not None or machine.exit_code != 0:
        raise NativeInputError('reference_fault')
    try:
        first = output.index(text.printable[256])
        last = output.index(text.printable[257], first + 1)
        body = text.decode_bytes(output[first + 1:last]).decode('utf-8')
    except (ValueError, UnicodeError) as error:
        raise NativeInputError('reference_response') from error
    answer = body.strip()
    try:
        expected = exact_answer(expression)
    except NativeInputError:
        expected = None
    refusal = KNOWN_CAPACITY_REFUSALS.get(expression)
    expected_error = expression == '(' * 16 + '1' + ')' * 16 or expected is None or refusal is not None
    if expected_error:
        passed = answer in ('error', 'error (too complex: split it)')
    else:
        try:
            passed = Fraction(answer) == Fraction(expected)
        except (ValueError, ZeroDivisionError):
            passed = False
    return {'level': 'upstream/reference firmware; not Native CUDA', 'expression': expression,
            'expected_rational': expected, 'expected_error': expected_error,
            'actual_body': body, 'passed': passed, 'mathematical_success': passed and not expected_error,
            'refusal': refusal, 'stream': stream, 'text_ids': qids,
            'trace_tokens': trace_tokens, 'instructions': machine.steps,
            'record_loop_ms': elapsed_ms,
            'timing_scope': 'Machine.record and trace/echo capture only; excludes all oracles'}


def decode_prediction(row, keep, owner_dim, text):
    from tools.qwenputer.features import FAMILIES, TEXT_TOKEN
    if row[owner_dim] == 0:
        return None
    if row[owner_dim] != 1 or any(abs(row[d]) != 2 ** -20 for d in keep[:21]):
        raise NativeInputError('prediction_code')
    code = sum((1 << bit) for bit, d in enumerate(keep[:21]) if row[d] > 0)
    family, payload = code & 31, code >> 5
    if family == len(FAMILIES):
        if payload >= len(text.printable):
            raise NativeInputError('prediction_text')
        return TEXT_TOKEN + text.printable[payload]
    if family >= len(FAMILIES) or payload >= FAMILIES[family][1]:
        raise NativeInputError('prediction_trace')
    return sum(size for _, size in FAMILIES[:family]) + payload


def main(argv=None):
    import argparse
    import hashlib
    import json
    from pathlib import Path
    import struct
    import sys
    import numpy as np
    from tools.qwenputer.bootstrap import preflight, capture_boot
    from tools.qwenputer.export import canonical_json, compile_sections
    from tools.qwenputer.format import encode_sections
    from tools.qwenputer.source import verify_source
    from tools.qwenputer.tokenizer import NativeText

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'third_party/qwenputer')
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--firmware', type=Path, required=True)
    parser.add_argument('--boot-dir', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append', help='development case name; defaults to all')
    parser.add_argument('--ir', action='store_true', help='also generate checked exact IR tensors')
    parser.add_argument('--execute-reviewed-reference', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute_reviewed_reference:
        raise NativeInputError('execution_not_authorized')
    from tools.qwenputer.source import require_compiler_environment
    require_compiler_environment()
    report = preflight(args.source_root, args.tokenizer_dir, args.firmware)
    if report['firmware_sha256'] != 'b643ad61291a947fa6c7498de9170b1d4da062dc1f34810feba9c4eebdec862d':
        raise NativeInputError('firmware_execution_scope')
    if args.output.exists():
        raise NativeInputError('oracle_output')

    def read(path, limit):
        with path.open('rb') as f:
            data = f.read(limit + 1)
        if len(data) > limit:
            raise NativeInputError('oracle_input_size')
        return data

    manifest = json.loads(read(args.artifact_dir / 'manifest.json', 65536))
    state_data = read(args.boot_dir / 'boot-state.json', 4 * 1024 * 1024)
    if (hashlib.sha256(state_data).hexdigest() != manifest['identities']['boot_sha256']
            or report['firmware_sha256'] != manifest['identities']['firmware_sha256']
            or report['tokenizer_binding_sha256'] != manifest['identities']['tokenizer_sha256']):
        raise NativeInputError('oracle_binding')
    state = json.loads(state_data)
    boot_manifest = json.loads(read(args.boot_dir / 'manifest.json', 65536))
    boot_trace_data = read(args.boot_dir / 'boot-trace.json', 4 * 1024 * 1024)
    if boot_manifest.get('boot-trace.json') != {'bytes': len(boot_trace_data),
                                               'sha256': hashlib.sha256(boot_trace_data).hexdigest()}:
        raise NativeInputError('oracle_boot_trace')
    boot_trace = json.loads(boot_trace_data)
    artifact = read(args.artifact_dir / 'native.qput', 256 * 1024 * 1024)
    if hashlib.sha256(artifact).hexdigest() != manifest['artifact_sha256']:
        raise NativeInputError('oracle_artifact_digest')
    text = NativeText.from_pack(args.tokenizer_dir)
    verify_source(args.source_root)
    if any(name == 'chipformer' or name.startswith('chipformer.') for name in sys.modules):
        raise NativeInputError('compiler_already_loaded')
    sys.path.insert(0, str(args.source_root.resolve() / 'experimental/riscv'))
    from chipformer.rv import core, ref
    from chipformer.q35 import ir

    def machine():
        # Independent state per case; no trial is seeded by a previous answer.
        return ref.Machine(rom=list(state['rom']), ram=list(state['ram']), x=list(state['registers']),
                           anchor=state['anchor'], n=state['n'], drom=bytes.fromhex(state['drom_hex']),
                           bn=[list(row) for row in state['digits']])

    circuit = core.build(text=text)
    boot = machine()
    boot.halted, boot.exit_code = True, 0
    capture_boot(boot, boot_trace, 200000)
    core.snapshot(circuit, boot)
    sections = compile_sections(circuit, text, manifest['identities'])
    if encode_sections(sections) != artifact:
        raise NativeInputError('oracle_circuit_artifact_mismatch')
    cases = development_cases()
    selected = [(name, expr) for name, expr in cases if args.case is None or name in args.case]
    if not selected or (args.case is not None and set(args.case) != {n for n, _ in selected}):
        raise NativeInputError('oracle_case')
    args.output.mkdir(parents=True, exist_ok=False)
    corpus = {'corpus_version': 2, 'development_seed': DEV_SEED, 'held_out_seed': HELD_OUT_SEED,
              'held_out_executed': False, 'cases': cases, 'compiler_revision': manifest['revision'],
              'firmware_sha256': report['firmware_sha256'], 'artifact_sha256': manifest['artifact_sha256']}
    (args.output / 'corpus.json').write_bytes(canonical_json(corpus))
    results = []
    for name, expression in selected:
        result = reference_case(machine(), text, expression)
        result['name'] = name
        (args.output / (name + '.reference.json')).write_bytes(canonical_json(result))
        print(json.dumps({k: result[k] for k in ('name', 'expression', 'actual_body', 'expected_rational',
                                               'expected_error', 'passed', 'mathematical_success',
                                               'refusal', 'trace_tokens', 'instructions')}), flush=True)
        if not result['passed']:
            raise NativeInputError('reference_mathematical_mismatch:' + name)
        if args.ir:
            # One reference batch uses the original IR and validates every node's
            # BF16/exactness invariants. No oracle is inside production timing.
            stream = result['stream']
            if len(stream) * len(circuit.dims) * 8 > 1024 * 1024 * 1024:
                raise NativeInputError('oracle_batch_memory')
            carry = {}
            matrix = ir.evaluate(circuit, stream, check=True, carry=carry, kernel=False)
            owner_dim = next(iter(circuit.owner.terms))
            for i, row in enumerate(matrix[:-1]):
                predicted = decode_prediction(row, circuit.keep, owner_dim, text)
                if predicted is not None and predicted != stream[i + 1]:
                    raise NativeInputError(f'ir_prediction_mismatch:{name}:{i}:{predicted}:{stream[i + 1]}')
            (args.output / (name + '.residual.f32')).write_bytes(matrix.astype('<f4').tobytes())
            (args.output / (name + '.retained.f32')).write_bytes(matrix[:, circuit.keep + [owner_dim]].astype('<f4').tobytes())
            initial_state = np.zeros((len(sections['heads'][1]) // 88, 128, 128), dtype='<f4')
            hi = 0
            for ni, node in enumerate(circuit.nodes):
                if type(node).__name__ == 'Head':
                    for vi, _ in enumerate(node.vheads()):
                        for row, values in carry.get((ni, 'S', vi), {}).items():
                            initial_state[hi, row] = values
                        hi += 1
            (args.output / (name + '.final-state.f32')).write_bytes(initial_state.tobytes())
            conv = [{'key': list(k), 'values': np.asarray(v).tolist()}
                    for k, v in carry.items() if len(k) < 2 or k[1] != 'S']
            (args.output / (name + '.final-lags.json')).write_bytes(canonical_json(conv))
            result['ir_checked_tokens'] = len(stream)
            print(f'IR PASS {name}: {len(stream)} tokens, {len(circuit.dims)} residual dimensions', flush=True)
        results.append({k: v for k, v in result.items() if k not in ('stream', 'text_ids')})
    (args.output / 'oracle.jsonl').write_bytes(b''.join(canonical_json(r) + b'\n' for r in results))
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output / 'manifest.json').write_bytes(canonical_json({'files': files, 'ir': args.ir,
                    'cases': len(results), 'artifact_sha256': manifest['artifact_sha256'],
                    'held_out_executed': False, 'production_execution': False}))
    print(f'PASS {len(results)} development cases; IR={args.ir}; not Native CUDA', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
