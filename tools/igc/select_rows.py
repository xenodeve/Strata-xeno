"""Rows for the next arm: the questions a first pass got wrong, plus a seeded share of the right ones so harm still shows.
Easy questions the model already gets show nothing and cost the same. select_rows.py records.json rows.jsonl output.jsonl [share of right]"""
import json, random, sys
from tools.igc.standard_score import bbh_right, letter_answer, math_right


def right(row, text):
    if row['suite'] == 'stdmath':
        return math_right(text, row['answer'])
    if row['suite'] == 'bbh':
        return bbh_right(text, row['answer'])
    return letter_answer(text) == row['answer']


def select(records, rows, share=.2, seed=1):
    by_id = {r['id']: r for r in rows}
    wrong, good = [], []
    for record in records:
        row = by_id.get(record['id'])
        if row is not None:
            (good if right(row, record['text']) else wrong).append(row)
    random.Random(seed).shuffle(good)
    return wrong, good[:round(len(good) * share)], len(good)


if __name__ == '__main__':
    records = json.load(open(sys.argv[1], encoding='utf-8'))
    rows = [json.loads(line) for line in open(sys.argv[2], encoding='utf-8')]
    wrong, kept, right_total = select(records, rows, float(sys.argv[4]) if len(sys.argv) > 4 else .2)
    with open(sys.argv[3], 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(json.dumps(dict(r, first='wrong'), ensure_ascii=False) + '\n' for r in wrong)
        f.writelines(json.dumps(dict(r, first='right'), ensure_ascii=False) + '\n' for r in kept)
    n = len(wrong) + right_total
    print(sys.argv[3], 'first pass', n, 'right', right_total, f'{100 * right_total / max(n, 1):.2f} %', 'kept', len(wrong), 'wrong and', len(kept), 'right')
