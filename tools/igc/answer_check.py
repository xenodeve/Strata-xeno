"""A second reader for external answers. Rule and reader agree that it is right: done. They differ: read the case and say
who erred and why. They agree that it is wrong: read the case and say what went wrong."""
import argparse
from collections import Counter
import json
from pathlib import Path
from tools.igc.external_score import final_number

RULES = """You compare a reply with the reference answer of a question. Do not solve the question.
same: the value the reply settles on as its answer equals the reference, in any wording or unit form.
different: it settles on another value, or gives no value.
hedged: it offers several values and does not choose one.
Answer with one JSON object only: {"verdict": "same|different|hedged", "value": "<the value it settles on>", "reason": "<short>"}"""


def reply_of(text):
    return text.rsplit('</think>', 1)[1].replace('<|im_end|>', '').strip() if '</think>' in text else None


def prompt(case):
    return f"{RULES}\n\nQuestion:\n{case['question']}\n\nReference answer: {case['gold']}\n\nReply:\n{reply_of(case['reply'])}"


def scored_right(record):
    value = final_number(record['text'])
    return value is not None and abs(value - float(record['answer'])) < 1e-6


def triage(records, verdicts):
    rows = []
    for r in records:
        rule = scored_right(r)
        verdict = verdicts.get(r['id'])
        if reply_of(r['text']) is None:
            route = 'agreed-wrong'          # nothing to compare: the limit or a loop took it
        elif verdict is None or verdict['verdict'] not in ('same', 'different', 'hedged'):
            route = 'disagree'
        else:
            judge = verdict['verdict'] == 'same'
            route = 'disagree' if judge != rule else 'agreed-right' if rule else 'agreed-wrong'
        rows.append(dict(id=r['id'], route=route, rule=rule, verdict=verdict, finish=r['finish'], tokens=r['tokens'],
                         gold=r['answer'], rule_value=final_number(r['text'])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', type=Path)
    parser.add_argument('--verdicts', type=Path, required=True, help='JSON of id -> verdict, from the reader')
    args = parser.parse_args()
    records = json.loads((args.arm / 'records.json').read_text(encoding='utf-8'))
    rows = triage(records, json.loads(args.verdicts.read_text(encoding='utf-8')))
    (args.arm / 'triage.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print(args.arm.name, dict(Counter(r['route'] for r in rows)))
    for r in rows:
        if r['route'] != 'agreed-right':
            print(r['route'], r['id'], 'gold', r['gold'], 'rule', r['rule_value'], r['finish'], r['verdict'])


if __name__ == '__main__':
    main()
