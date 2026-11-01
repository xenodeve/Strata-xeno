"""Differential runner for the #279 sandbox: CPython against tests/igc/sandbox_main.

A program runs under `python -I -B -X utf8 file.py` and under our binary, and the two must agree on stdout and on the
class of an uncaught error. CPython is the reference at test time only; the header does not depend on it.

A first line `# expect: <Status>` marks a case that ends in a limit by design. CPython is not run at all for those:
`[0]*10**10` or `while True: pass` there would take the machine's RAM or never end.

Verdicts:
  match     same stdout, same exception class
  expected  a `# expect:` case that ended in that status
  refused   Unsupported, or ModuleNotFoundError for a module CPython has and the allowlist does not
  limit     StepLimit, MemoryLimit or RecursionLimit on a program CPython ran
  MISMATCH  our answer differs from CPython's, always a bug
  CRASH     no status line, or a non-zero exit
  HANG      our binary passed the timeout
  CPYTHON-TIMEOUT  the reference itself did not finish

    python -I -B tools/igc/sandbox_oracle.py --binary sandbox.exe corpus tests/igc/sandbox/dev
    python -I -B tools/igc/sandbox_oracle.py --binary sandbox.exe humaneval HumanEval.jsonl.gz --split even
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000
LIMIT_STATUSES = {'StepLimit', 'MemoryLimit', 'RecursionLimit'}
ANSWERING_STATUSES = {'Ok', 'Raised', 'Syntax'}
EXPECT = re.compile(r'#\s*expect:\s*([A-Za-z]+)')
EXCEPTION_LINE = re.compile(r'^([A-Za-z_][A-Za-z0-9_.]*)(?::|$)')
STATUS_LINE = re.compile(r'^status=(\w+) steps=(\d+) peak=(\d+) truncated=([01]) ms=([\d.]+)(?: error=(.*))?$')
# The binary's own output limit stays out of the comparison unless a case asks for it.
BINARY_FLAGS = ['--output', '1000000']


@dataclass
class Run:
    stdout: str = ''
    exception: str | None = None
    timed_out: bool = False


@dataclass
class Ours(Run):
    status: str | None = None
    error: str = ''
    steps: int = 0
    peak: int = 0
    truncated: bool = False
    ms: float = 0.0
    exit_code: int = 0


@dataclass
class Verdict:
    kind: str
    name: str
    detail: str = ''
    ours: Ours | None = None
    reference: Run | None = None


def _run(command: list[str], timeout: float) -> tuple[bytes, bytes, int, bool]:
    flags = CREATE_NO_WINDOW if os.name == 'nt' else 0
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                               creationflags=flags)
    try:
        out, err = process.communicate(timeout=timeout)
        return out, err, process.returncode, False
    except subprocess.TimeoutExpired:
        process.kill()
        out, err = process.communicate()
        return out, err, process.returncode, True


def _text(raw: bytes) -> str:
    return raw.decode('utf-8', 'replace').replace('\r\n', '\n')


def exception_class(stderr: str) -> str | None:
    """The class of the uncaught error: the last line at column 0 that reads `Name: message` or `Name`."""
    if 'Traceback (most recent call last)' not in stderr and 'Error' not in stderr:
        return None
    for line in reversed(stderr.strip().splitlines()):
        if line[:1].isspace():
            continue
        match = EXCEPTION_LINE.match(line)
        if match:
            return match.group(1).rsplit('.', 1)[-1]
    return None


def run_cpython(path: Path, timeout: float, python: str = sys.executable) -> Run:
    out, err, _, timed_out = _run([python, '-I', '-B', '-X', 'utf8', str(path)], timeout)
    stderr = _text(err)
    return Run(_text(out), None if timed_out else exception_class(stderr), timed_out)


def run_binary(binary: Path, path: Path, timeout: float, flags: list[str] | None = None) -> Ours:
    out, err, code, timed_out = _run([str(binary), str(path)] + (BINARY_FLAGS if flags is None else flags), timeout)
    ours = Ours(_text(out), timed_out=timed_out, exit_code=code)
    lines = _text(err).strip().splitlines()
    match = STATUS_LINE.match(lines[-1]) if lines else None
    if match:
        ours.status, ours.error = match.group(1), match.group(6) or ''
        ours.steps, ours.peak, ours.truncated = int(match.group(2)), int(match.group(3)), match.group(4) == '1'
        ours.ms = float(match.group(5))
        if ours.status in ('Raised', 'Syntax'):
            head = re.match(r'[A-Za-z_][\w.]*', ours.error)
            ours.exception = head.group(0) if head else None
    return ours


def expected_status(source: str) -> str | None:
    first = source.split('\n', 1)[0]
    match = EXPECT.match(first.strip())
    return match.group(1) if match else None


def first_difference(ours: str, reference: str) -> str:
    a, b = ours.split('\n'), reference.split('\n')
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            at = next((k for k, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            start = max(at - 40, 0)
            return f'stdout line {i + 1} col {at}: ours {x[start:at + 120]!r} cpython {y[start:at + 120]!r}'
    return f'stdout lengths differ: ours {len(a)} lines, cpython {len(b)} lines'


def compare(name: str, expect: str | None, ours: Ours, reference: Run | None) -> Verdict:
    def verdict(kind: str, detail: str = '') -> Verdict:
        return Verdict(kind, name, detail, ours, reference)

    if ours.timed_out:
        return verdict('HANG')
    if ours.status is None or ours.exit_code != 0:
        return verdict('CRASH', f'exit {ours.exit_code:#x}')
    if expect is not None:
        if ours.status == expect:
            return verdict('expected')
        return verdict('MISMATCH', f'expected status {expect}, got {ours.status}: {ours.error}')
    assert reference is not None
    if reference.timed_out:
        return verdict('CPYTHON-TIMEOUT')
    prefix = reference.stdout.startswith(ours.stdout)
    if ours.status == 'Unsupported':
        return verdict('refused' if prefix else 'MISMATCH', ours.error)
    if ours.status in LIMIT_STATUSES:
        return verdict('limit' if prefix else 'MISMATCH', ours.error)
    if ours.status not in ANSWERING_STATUSES:
        return verdict('MISMATCH', f'status {ours.status}')
    if ours.exception == 'ModuleNotFoundError' and reference.exception != 'ModuleNotFoundError' and prefix:
        return verdict('refused', ours.error)
    if ours.stdout != reference.stdout:
        return verdict('MISMATCH', first_difference(ours.stdout, reference.stdout))
    if ours.exception != reference.exception:
        return verdict('MISMATCH', f'exception: ours {ours.error!r} cpython {reference.exception!r}')
    return verdict('match')


def check_file(binary: Path, path: Path, timeout: float) -> Verdict:
    expect = expected_status(path.read_text(encoding='utf-8', errors='surrogateescape'))
    ours = run_binary(binary, path, timeout)
    reference = None if expect is not None else run_cpython(path, timeout)
    return compare(path.name, expect, ours, reference)


def humaneval_program(task: dict) -> str:
    return task['prompt'] + task['canonical_solution'] + '\n' + task['test'] + f"\ncheck({task['entry_point']})\n"


def task_number(task: dict) -> int:
    return int(task['task_id'].rsplit('/', 1)[-1])


def load_humaneval(path: Path, split: str) -> list[dict]:
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        tasks = [json.loads(line) for line in handle if line.strip()]
    wanted = {'even': 0, 'odd': 1}
    return [t for t in tasks if split == 'all' or task_number(t) % 2 == wanted[split]]


def summarize(verdicts: list[Verdict]) -> dict:
    counts = collections.Counter(v.kind for v in verdicts)
    return {'total': len(verdicts), **dict(sorted(counts.items()))}


def reason(error: str) -> str:
    """The refused feature without its line number."""
    return re.sub(r' \(line \d+\)$', '', error)


def report(verdicts: list[Verdict], stream=sys.stdout) -> None:
    print(json.dumps(summarize(verdicts)), file=stream)
    for kind in ('refused', 'limit'):
        reasons = collections.Counter(reason(v.detail) for v in verdicts if v.kind == kind)
        for text, count in reasons.most_common():
            print(f'  {kind} {count}x {text}', file=stream)
    for v in verdicts:
        if v.kind in ('MISMATCH', 'CRASH', 'HANG', 'CPYTHON-TIMEOUT'):
            print(f'  {v.kind} {v.name}: {v.detail[:400]}', file=stream)


def run_corpus(binary: Path, directory: Path, timeout: float, only: str | None = None) -> list[Verdict]:
    paths = sorted(p for p in directory.glob('*.py') if only is None or only in p.name)
    return [check_file(binary, p, timeout) for p in paths]


def run_humaneval(binary: Path, data: Path, split: str, timeout: float, out: Path | None) -> list[Verdict]:
    verdicts = []
    sources = {}
    with tempfile.TemporaryDirectory(prefix='sandbox-humaneval-') as scratch:
        for task in load_humaneval(data, split):
            path = Path(scratch) / f"task_{task_number(task):03d}.py"
            source = humaneval_program(task)
            path.write_text(source, encoding='utf-8')
            verdicts.append(check_file(binary, path, timeout))
            sources[path.name] = source
    if out:
        rows = [{'name': v.name, 'kind': v.kind, 'detail': v.detail, 'steps': v.ours.steps if v.ours else None,
                 'peak': v.ours.peak if v.ours else None, 'ms': v.ours.ms if v.ours else None,
                 'source': sources[v.name] if v.kind != 'match' else None} for v in verdicts]
        out.write_text(json.dumps(rows, indent=1), encoding='utf-8')
    return verdicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=10.0)
    commands = parser.add_subparsers(dest='command', required=True)
    corpus = commands.add_parser('corpus')
    corpus.add_argument('directory', type=Path)
    corpus.add_argument('--only')
    tasks = commands.add_parser('humaneval')
    tasks.add_argument('data', type=Path)
    tasks.add_argument('--split', choices=['even', 'odd', 'all'], required=True)
    tasks.add_argument('--out', type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    if args.command == 'corpus':
        verdicts = run_corpus(args.binary, args.directory, args.timeout, args.only)
    else:
        verdicts = run_humaneval(args.binary, args.data, args.split, args.timeout, args.out)
    report(verdicts)
    print(f'  {time.perf_counter() - start:.1f} s')
    return 0 if all(v.kind in ('match', 'expected') for v in verdicts) else 1


if __name__ == '__main__':
    sys.exit(main())
