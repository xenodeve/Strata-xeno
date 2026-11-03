"""Rows for the next arm: the questions a first pass got wrong and the feature can reach, plus an optional seeded share of the right ones.
Easy questions show nothing, and neither do misses with nothing to compute: a wrong fact or a misread question is out of reach.
select_rows.py records.json rows.jsonl output.jsonl [share of right] | --verdicts records.json verdicts.json (for --stop-on-loss)"""
import json, random, re, sys
from tools.igc.external_score import final_number
from tools.igc.standard_score import bbh_right, hle_right, letter_answer, math_right

LETTER_SUITES = ('mmlupro', 'supergpqa', 'gpqa')


def right(row, text):
    """The one rule each suite is scored by: the bench's early stop, the reader's triage and the row selection all use it."""
    if row['suite'] in ('stdmath', 'arith'):   # arith: Qwenputer's DeepMind / NUPA sets, boxed numbers
        return math_right(text, row['answer'])
    if row['suite'] == 'bbh':
        return bbh_right(text, row['answer'])
    if row['suite'] == 'hle':
        return hle_right(text, row['answer'])
    if row['suite'] in LETTER_SUITES:
        return letter_answer(text) == row['answer']
    value = final_number(text)   # GSM8K and the other integer suites
    return value is not None and abs(value - float(row['answer'])) < 1e-6


CALCULATION = re.compile(r'\d\s*(?:[-+*/^=<>×÷]|\\times|\\frac|\\cdot)\s*[\d(\\]')


def reachable(row, text):
    """MIP computes what the reasoning writes: a miss with no calculation in its reply gives it nothing to check."""
    if row.get('calculation') is False:
        return False
    return bool(CALCULATION.search(text))


def select(records, rows, share=.2, seed=1):
    by_id = {r['id']: r for r in rows}
    wrong, good, out_of_reach = [], [], []
    for record in records:
        row = by_id.get(record['id'])
        if row is None:
            continue
        if right(row, record['text']):
            good.append(row)
        else:
            (wrong if reachable(row, record['text']) else out_of_reach).append(row)
    random.Random(seed).shuffle(good)
    return wrong, good[:round(len(good) * share)], len(good), len(out_of_reach)


def verdicts(records):
    """The base arm's verdicts for external_bench --stop-on-loss, by the same rule the bench judges the new arm with."""
    return {r['id']: 'ok' if right(r, r['text']) else 'wrong' for r in records}


if __name__ == '__main__' and sys.argv[1] == '--verdicts':   # --verdicts records.json verdicts.json
    found = verdicts(json.load(open(sys.argv[2], encoding='utf-8')))
    open(sys.argv[3], 'w', encoding='utf-8').write(json.dumps(found, indent=1))
    print(sys.argv[3], sum(v == 'ok' for v in found.values()), 'ok of', len(found))
elif __name__ == '__main__':
    records = json.load(open(sys.argv[1], encoding='utf-8'))
    rows = [json.loads(line) for line in open(sys.argv[2], encoding='utf-8')]
    wrong, kept, right_total, dropped = select(records, rows, float(sys.argv[4]) if len(sys.argv) > 4 else 0)
    with open(sys.argv[3], 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(json.dumps(dict(r, first='wrong'), ensure_ascii=False) + '\n' for r in wrong)
        f.writelines(json.dumps(dict(r, first='right'), ensure_ascii=False) + '\n' for r in kept)
    n = len(wrong) + right_total + dropped
    print(sys.argv[3], 'first pass', n, 'right', right_total, f'{100 * right_total / max(n, 1):.2f} %', 'kept', len(wrong), 'misses,', dropped, 'out of reach,', len(kept), 'right')
