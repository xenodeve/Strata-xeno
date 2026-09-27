"""Per-round latency budget from a `strata generate` stdout/stderr pair.

Usage: python tests/xeno/latency_breakdown.py RUN_DIR [RUN_DIR ...]

Reads RUN_DIR/on.stdout (+ on.stderr for prefill) and prints where each decode round's
milliseconds go, as ms/round and as a share of the measured round, so the largest stage
is the first thing to attack.  Every number is copied from Strata's own counters; the
"unaccounted" row is round time minus the stages Strata reports, not a guess.
"""
import re
import sys
from pathlib import Path

FLOAT = r'([0-9]+(?:\.[0-9]+)?)'


def parse(stdout: str, stderr: str = '') -> dict:
    out = {}
    m = re.search(r'^decode\s+(\d+) tokens in ' + FLOAT + r' ms', stdout, re.M)
    if m:
        out['tokens'], out['decode_ms'] = int(m.group(1)), float(m.group(2))
    m = re.search(r'^speculation\s+(\d+) rounds', stdout, re.M)
    if m:
        out['rounds'] = int(m.group(1))
    m = re.search(r'^verify window\s+wait for rings ' + FLOAT + r'\s+pool ' + FLOAT + r'\s+host ' + FLOAT
                  + r'\s+commit ' + FLOAT + r' ms/round; CPU experts ' + FLOAT + r' distinct / ' + FLOAT, stdout, re.M)
    if m:
        out.update(wait_rings=float(m.group(1)), pool=float(m.group(2)), host=float(m.group(3)),
                   commit=float(m.group(4)), cpu_distinct=float(m.group(5)), cpu_routed=float(m.group(6)))
    m = re.search(r'^pool multi\s+gate/up ' + FLOAT + r'\s+quantize ' + FLOAT + r'\s+down ' + FLOAT
                  + r' ms/round; ' + FLOAT + r' GB/s', stdout, re.M)
    if m:
        out.update(pool_gateup=float(m.group(1)), pool_quant=float(m.group(2)), pool_down=float(m.group(3)),
                   pool_gbs=float(m.group(4)))
    m = re.search(r'^dispatch\s+plan ' + FLOAT + r'\s+activation quantize ' + FLOAT + r'\s+jobs ' + FLOAT
                  + r'\s+run ' + FLOAT, stdout, re.M)
    if m:
        out.update(disp_plan=float(m.group(1)), disp_actq=float(m.group(2)), disp_jobs=float(m.group(3)),
                   disp_run=float(m.group(4)))
    m = re.search(r'^mtp\s+' + FLOAT + r' ms/round drafting', stdout, re.M)
    if m:
        out['mtp'] = float(m.group(1))
    m = re.search(r'prefill (\d+) tokens in (\d+) chunks, ' + FLOAT + r' ms .*?host ' + FLOAT + r' ms\), resident (\d+); PLE '
                  + FLOAT, stderr)
    if m:
        out.update(prefill_tokens=int(m.group(1)), prefill_ms=float(m.group(3)), prefill_host_ms=float(m.group(4)),
                   prefill_ple_ms=float(m.group(6)))
    return out


def budget(p: dict) -> list:
    """Rows of (stage, ms/round, share of round).  Stages overlap only where Strata says they do:
    `wait for rings` is the GPU waiting on the CPU, `pool` is the CPU work it waits for."""
    if 'rounds' not in p or 'decode_ms' not in p:
        return []
    per_round = p['decode_ms'] / p['rounds']
    stages = [('GPU waits for CPU experts (wait for rings)', p.get('wait_rings', 0.0)),
              ('MTP drafting', p.get('mtp', 0.0)),
              ('host', p.get('host', 0.0)),
              ('commit', p.get('commit', 0.0))]
    rows = [('round total', per_round, 1.0)]
    rows += [(name, ms, ms / per_round) for name, ms in stages]
    rest = per_round - sum(ms for _, ms in stages)
    rows.append(('GPU compute + unaccounted', rest, rest / per_round))
    rows.append(('  (CPU pool call, overlaps GPU)', p.get('pool', 0.0), p.get('pool', 0.0) / per_round))
    for key, name in (('pool_gateup', '    pool gate/up'), ('pool_down', '    pool down'),
                      ('disp_plan', '    dispatch plan')):
        if key in p:
            rows.append((name, p[key], p[key] / per_round))
    return rows


def main(argv: list) -> int:
    for arg in argv:
        run = Path(arg)
        stdout = (run / 'on.stdout').read_text(encoding='utf-8', errors='replace')
        stderr_path = run / 'on.stderr'
        stderr = stderr_path.read_text(encoding='utf-8', errors='replace') if stderr_path.exists() else ''
        p = parse(stdout, stderr)
        print(f'== {run.name}: {p.get("tokens", "?")} tokens, {p.get("rounds", "?")} rounds, '
              f'CPU experts {p.get("cpu_distinct", "?")} distinct/layer, pool {p.get("pool_gbs", "?")} GB/s')
        for name, ms, share in budget(p):
            print(f'  {name:<46} {ms:8.2f} ms  {share * 100:5.1f} %')
        if 'prefill_ms' in p:
            print(f'  prefill {p["prefill_tokens"]} tok {p["prefill_ms"]:.0f} ms: expert upload host '
                  f'{p["prefill_host_ms"]:.0f} ms ({p["prefill_host_ms"] / p["prefill_ms"] * 100:.0f} %), '
                  f'PLE {p["prefill_ple_ms"]:.0f} ms')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
