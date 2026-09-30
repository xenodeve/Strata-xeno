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
    # verify.cpp: wait for rings is host polling the primary GPU doorbell;
    # pool is the following expert callback, including secondary completion.
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
    m = re.search(r'^dispatch detail\s+CPU pool ' + FLOAT + r' secondary finish ' + FLOAT +
                  r' ms/round$', stdout, re.M)
    if m:
        out.update(cpu_pool_self=float(m.group(1)), secondary_finish_total=float(m.group(2)))
    m = re.search(r'^mtp\s+' + FLOAT + r' ms/round drafting', stdout, re.M)
    if m:
        out['mtp'] = float(m.group(1))
    m = re.search(r'^secondary host\s+plan ' + FLOAT + r'\s+switch ' + FLOAT +
                  r'\s+enqueue ' + FLOAT + r'\s+query ' + FLOAT +
                  r'\s+copyout ' + FLOAT + r' ms/round$', stdout, re.M)
    if m:
        out.update(secondary_host_plan=float(m.group(1)), secondary_host_switch=float(m.group(2)),
                   secondary_host_enqueue=float(m.group(3)), secondary_host_query=float(m.group(4)),
                   secondary_host_copyout=float(m.group(5)))
    m = re.search(r'^secondary device\s+H2D ' + FLOAT + r'\s+clear ' + FLOAT +
                  r'\s+quantize ' + FLOAT + r'\s+expert ' + FLOAT + r'\s+D2H ' + FLOAT +
                  r' ms/round; (\d+) launches$', stdout, re.M)
    if m:
        out.update(secondary_h2d=float(m.group(1)), secondary_clear=float(m.group(2)),
                   secondary_quantize=float(m.group(3)), secondary_expert=float(m.group(4)),
                   secondary_d2h=float(m.group(5)), secondary_launches=int(m.group(6)))
    m = re.search(r'^secondary D2H bytes\s+full ' + FLOAT + r' MiB requested ' + FLOAT +
                  r' MiB$', stdout, re.M)
    if m:
        out.update(secondary_d2h_full_mib=float(m.group(1)),
                   secondary_d2h_requested_mib=float(m.group(2)))
    m = re.search(r'prefill (\d+) tokens in (\d+) chunks, ' + FLOAT + r' ms .*?host ' + FLOAT + r' ms\), resident (\d+); PLE '
                  + FLOAT, stderr)
    if m:
        out.update(prefill_tokens=int(m.group(1)), prefill_ms=float(m.group(3)), prefill_host_ms=float(m.group(4)),
                   prefill_ple_ms=float(m.group(6)))
    tiers = ('pinned', 'pageable', 'peer', 'nvme')
    m = re.search(r'prefill sources' + ''.join(rf' {k} (\d+) \(' + FLOAT + r' GB\)' for k in tiers), stderr)
    if m:
        for i, k in enumerate(tiers):
            out[f'prefill_src_{k}'] = int(m.group(2 * i + 1))
            out[f'prefill_src_{k}_gb'] = float(m.group(2 * i + 2))
    m = re.search(r'prefill sources.*; rows (\d+) (\d+) (\d+) (\d+)', stderr)
    if m:
        for i, k in enumerate(tiers):
            out[f'prefill_rows_{k}'] = int(m.group(i + 1))
    return out


def budget(p: dict) -> list:
    """Host wall intervals; indented diagnostics nest inside the expert callback.

    This is not a device-time pie. CUDA work overlaps host intervals, and the
    remainder cannot be called GPU compute without device-side evidence.
    """
    if 'rounds' not in p or 'decode_ms' not in p:
        return []
    per_round = p['decode_ms'] / p['rounds']
    stages = [('Host prepares verifier window', p.get('host', 0.0)),
              ('Host waits for primary GPU doorbell', p.get('wait_rings', 0.0)),
              ('Host services expert callback', p.get('pool', 0.0)),
              ('MTP drafting', p.get('mtp', 0.0)),
              ('Commit', p.get('commit', 0.0))]
    rows = [('round total', per_round, 1.0)]
    rows += [(name, ms, ms / per_round) for name, ms in stages]
    rest = per_round - sum(ms for _, ms in stages)
    rows.append(('Other/unaccounted wall time', rest, rest / per_round))
    for key, name in (('disp_plan', '  dispatch plan (inside callback)'),
                      ('disp_run', '  dispatch run (CPU pool + secondary finish)'),
                      ('cpu_pool_self', '    CPU pool self-time'),
                      ('secondary_finish_total', '    secondary finish total'),
                      ('pool_gateup', '  CPU gate/up phase'), ('pool_down', '  CPU down phase')):
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
        if 'secondary_launches' in p:
            print(f'  secondary host (overlaps device): plan {p["secondary_host_plan"]:.3f}, '
                  f'switch {p["secondary_host_switch"]:.3f}, enqueue {p["secondary_host_enqueue"]:.3f}, '
                  f'query {p["secondary_host_query"]:.3f}, '
                  f'copyout {p["secondary_host_copyout"]:.3f} ms/round')
            print(f'  secondary stream (includes enqueue gaps): H2D {p["secondary_h2d"]:.3f}, '
                  f'clear {p["secondary_clear"]:.3f}, quantize {p["secondary_quantize"]:.3f}, '
                  f'expert {p["secondary_expert"]:.3f}, D2H {p["secondary_d2h"]:.3f} ms/round '
                  f'({p["secondary_launches"]} launches)')
            if 'secondary_d2h_full_mib' in p:
                print(f'  D2H requested {p["secondary_d2h_requested_mib"]:.2f} MiB of '
                      f'{p["secondary_d2h_full_mib"]:.2f} MiB full routed rows')
        if 'prefill_ms' in p:
            print(f'  prefill {p["prefill_tokens"]} tok {p["prefill_ms"]:.0f} ms: expert upload host '
                  f'{p["prefill_host_ms"]:.0f} ms ({p["prefill_host_ms"] / p["prefill_ms"] * 100:.0f} %), '
                  f'PLE {p["prefill_ple_ms"]:.0f} ms')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
