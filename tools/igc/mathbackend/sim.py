"""The coordinator's runner: every sim_<module>.run(seed, n) of the math backend, then one coverage table (docs/plans/2026-11-05-mip-math-backend-coverage.md).

    PYTHONPATH=. python -m tools.igc.mathbackend.sim --seed 1 --n 60 [--modules numbers,algebra]

Run it with the interpreter that has SymPy and mpmath (the oracles live in the sim files, never in the backend). Per operation: cases generated, answered, agreeing with the oracle,
refused (NotCovered, with reasons) and disagreeing (must be 0). Per family (`kind`) and in total: answered / generated and agree / answered. Operations registered with no case, and cases
for operations that are not registered, are listed: a registered operation nobody tests is a gap, not a pass.
"""
import argparse
import collections
import importlib
import json
import sys
import time
from pathlib import Path

if __name__ == '__main__' and __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.igc.mathbackend import MODULES, registry

STATS = ('generated', 'answered', 'agree', 'disagree')


def normalise(result):
    """{op: {generated, answered, agree, refused: {reason: n}, disagree, seconds}} from what a sim module returned (either {'ops': {...}} or the mapping itself)."""
    ops = {}
    if isinstance(result, dict):
        for key in ('ops', 'per_operation', 'per_op'):
            if isinstance(result.get(key), dict):
                ops = result[key]
                break
        else:
            ops = result
    out = {}
    for name, s in ops.items():
        if not isinstance(s, dict) or 'generated' not in s or '.' not in name:
            continue
        refused = s.get('refused', {})
        if isinstance(refused, int):
            reasons = s.get('refusal_reasons') or s.get('refusals') or {}
            refused = dict(reasons) if sum(reasons.values()) == refused else ({'unspecified': refused} if refused else {})
        # a case the module meant the backend to refuse (past a bound) and the backend did: a correct refusal, counted apart from the unexpected ones
        expected_ok = int(s.get('expected_refusals_ok', 0))
        out[name] = dict(generated=int(s.get('generated', 0)), answered=int(s.get('answered', 0)), agree=int(s.get('agree', 0)),
                         disagree=int(s.get('disagree', 0)), refused=dict(refused), expected_refusals=expected_ok, seconds=float(s.get('seconds', 0.0)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--modules', default=','.join(MODULES))
    ap.add_argument('--out', default='build-mathbackend/coverage.json')
    args = ap.parse_args(argv)
    reg = registry()
    results, missing = {}, []
    for m in args.modules.split(','):
        try:
            mod = importlib.import_module(f'tools.igc.mathbackend.sim_{m}')
        except ModuleNotFoundError as e:
            if e.name == f'tools.igc.mathbackend.sim_{m}':
                missing.append(m)
                continue
            raise
        t0 = time.monotonic()
        results[m] = normalise(mod.run(args.seed, args.n))
        print(f'[{m}] {len(results[m])} operations with cases in {time.monotonic() - t0:.0f} s', flush=True)
    seen = {name: s for r in results.values() for name, s in r.items()}
    registered = {n: v for n, v in reg.items() if n.split('.')[0] in results}
    by_kind = collections.defaultdict(lambda: collections.Counter())
    ops_by_kind = collections.defaultdict(lambda: [0, 0])
    for name, v in registered.items():
        k = v['kind']
        ops_by_kind[k][0] += 1
        s = seen.get(name)
        if s:
            ops_by_kind[k][1] += 1
            for f in STATS:
                by_kind[k][f] += s[f]
            by_kind[k]['refused'] += sum(s['refused'].values())
            by_kind[k]['expected'] += s.get('expected_refusals', 0)
    print(f'\n{"family":34s} {"ops":>9s} {"cases":>7s} {"answered":>9s} {"agree":>7s} {"refused":>8s} {"expected":>8s} {"disagree":>8s}')
    tot = collections.Counter()
    for k in sorted(by_kind.keys() | ops_by_kind.keys()):
        c, o = by_kind[k], ops_by_kind[k]
        print(f'{k:34s} {o[1]:3d}/{o[0]:<5d} {c["generated"]:7d} {c["answered"]:9d} {c["agree"]:7d} {c["refused"]:8d} {c["expected"]:8d} {c["disagree"]:8d}')
        tot.update(c)
        tot['ops'] += o[0]
        tot['tested'] += o[1]
    print(f'{"TOTAL":34s} {tot["tested"]:3d}/{tot["ops"]:<5d} {tot["generated"]:7d} {tot["answered"]:9d} {tot["agree"]:7d} {tot["refused"]:8d} {tot["expected"]:8d} {tot["disagree"]:8d}')
    untested = sorted(n for n in registered if n not in seen)
    stray = sorted(n for n in seen if n not in reg)
    bad = sorted((n, s['disagree']) for n, s in seen.items() if s['disagree'])
    if untested:
        print(f'\nregistered with no case ({len(untested)}):', ', '.join(untested))
    if stray:
        print(f'\ncases for operations that are not registered ({len(stray)}):', ', '.join(stray))
    if bad:
        print('\nDISAGREEMENTS:', bad)
    reasons = collections.Counter()
    for s in seen.values():
        reasons.update(s['refused'])
    if reasons:
        print('\nrefusals by reason:', dict(reasons.most_common(12)))
    if missing:
        print('\nno sim file yet for:', ', '.join(missing))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(seed=args.seed, n=args.n, results=results, untested=untested, stray=stray, missing=missing), indent=1), encoding='utf-8')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
