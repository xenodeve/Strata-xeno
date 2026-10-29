"""Offline acceptance oracle. Never import this module from serving code.

The production binary receives only artifact identities and prefix inputs. All
expected-answer computation and result checking happens after that process exits.
"""
from fractions import Fraction
import re
from tools.qwenputer.oracle import exact_answer


def classify_result(result, text, expression, artifact_sha256):
    rejected = lambda outcome: {'passed': False, 'outcome': outcome}
    if (result.get('kind') != 'native-execution'
            or result.get('evidence_level') != 'observed-execution'
            or result.get('artifact_sha256') != artifact_sha256):
        return rejected('identity-error')
    if result.get('reason') != 3 or result.get('error') != 0:
        return rejected('execution-incomplete')
    ids = result.get('text_ids')
    if (not isinstance(ids, list) or len(ids) > 4096
            or any(type(q) is not int or q < 0 for q in ids)):
        return rejected('result-envelope')
    opening, closing = text.printable[256:258]
    if ids.count(opening) != 1 or ids.count(closing) != 1:
        return rejected('result-envelope')
    start, end = ids.index(opening), ids.index(closing)
    if end <= start:
        return rejected('result-envelope')
    try:
        before = text.decode_bytes(ids[:start])
        after = text.decode_bytes(ids[end + 1:])
        body = text.decode_bytes(ids[start + 1:end]).decode('utf-8').strip()
    except (ValueError, IndexError, UnicodeError):
        return rejected('result-envelope')
    # calc-bridge ends a response with an empty thinking block. It is protocol
    # framing, not additional evidence or an arithmetic answer. Reject any other
    # non-whitespace suffix rather than silently dropping unexpected output.
    if before.strip() or after.strip() not in (b'', b'<think>\n\n</think>') or '\x00' in body:
        return rejected('result-envelope')
    if body in ('error', 'error (too complex: split it)'):
        return dict(rejected('firmware-refusal'), body=body)
    if not re.fullmatch(r'[+-]?[0-9]+(?:/[0-9]+|\.[0-9]+)?', body):
        return rejected('result-number')
    # This independent oracle is deliberately outside the measured invocation.
    expected = exact_answer(expression)
    try:
        actual = Fraction(body)
    except (ValueError, ZeroDivisionError):
        return rejected('result-number')
    passed = actual == Fraction(expected)
    return {'passed': passed, 'outcome': 'verified-selected-test' if passed else 'wrong-answer',
            'body': body, 'expected_rational': expected, 'actual_rational': str(actual),
            'evidence_level': 'selected-tests-passed' if passed else 'observed-execution'}


def main(argv=None):
    import argparse
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import time
    from tools.qwenputer.call import CallContext, eligible_call
    from tools.qwenputer.step_validation import inputs_data
    from tools.qwenputer.tokenizer import NativeText
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output must be a fresh evidence directory')
    frozen = (args.bundle / 'heldout.json').read_bytes()
    corpus_sha = hashlib.sha256(frozen).hexdigest()
    if corpus_sha != 'd1ddd54c5c86d93eab55e3bee98873be16fa777b175583e76b5ce77e34da32f3':
        parser.error('frozen held-out corpus identity mismatch')
    corpus = json.loads(frozen)
    cases = corpus['cases'] if isinstance(corpus, dict) else corpus
    if len(cases) != 64 or any(c.get('firmware_executed') is not False for c in cases):
        parser.error('unexpected held-out case status')
    artifact = args.bundle / 'native.qput'
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    binary_sha = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    text = NativeText.from_pack(args.tokenizer)
    report = dict(corpus_sha256=corpus_sha, artifact_sha256=artifact_sha,
                  binary_sha256=binary_sha, level='native CUDA autonomous; no language model',
                  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  binding_sha256=hashlib.sha256(args.binding.read_bytes()).hexdigest(),
                  tokenizer_sha256=text.binding_sha256,
                  oracle_in_timed_interval=False, results=[])
    args.output.mkdir(parents=True)
    for case in cases:
        expression = case['expression']
        call = '\n<function=calc><parameter=expression>' + expression + '</parameter></function>\n'
        qids = text.encode(call)
        eligible_call(text, qids, CallContext(live_assistant=True, enrolled_calc=True))
        path = args.output / (case['id'] + '.prefix')
        path.write_bytes(inputs_data([(1 << 24) + q for q in [text.TCO, *qids, text.TCC]], text.n))
        command = [str(args.binary), str(artifact), str(args.binding), str(path)]
        if hashlib.sha256(args.binary.read_bytes()).hexdigest() != binary_sha:
            raise ValueError('executable changed during acceptance')
        started = time.perf_counter()
        run = subprocess.run(command, capture_output=True, timeout=120,
                             creationflags=0x08000000 if os.name == 'nt' else 0)
        invocation_ms = (time.perf_counter() - started) * 1000
        if hashlib.sha256(args.binary.read_bytes()).hexdigest() != binary_sha:
            raise ValueError('executable changed during acceptance')
        (args.output / (case['id'] + '.stdout')).write_bytes(run.stdout)
        (args.output / (case['id'] + '.stderr')).write_bytes(run.stderr)
        if run.returncode:
            checked = {'passed': False, 'outcome': 'process-error', 'exit': run.returncode}
            result = {}
        else:
            result = json.loads(run.stdout)
            checked = classify_result(result, text, expression, artifact_sha)
            if exact_answer(expression) != case['expected_rational']:
                raise ValueError('frozen independent oracle disagreement')
        record = dict(checked, case=case['id'], expression=expression, invocation_ms=invocation_ms,
                      command=command, execution=result)
        report['results'].append(record)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({k: record.get(k) for k in ('case', 'expression', 'body', 'outcome', 'passed') }), flush=True)
    return 0 if all(r['passed'] for r in report['results']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
