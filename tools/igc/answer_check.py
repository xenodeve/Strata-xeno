"""A second reader for external answers. Rule and reader agree that it is right: done. They differ: read the case and say
who erred and why. They agree that it is wrong: read the case and say what went wrong."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
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


def load_env(path):
    rows = (line.split('=', 1) for line in path.read_text(encoding='utf-8').splitlines()
            if '=' in line and not line.lstrip().startswith('#'))
    return {key.strip(): value.strip() for key, value in rows}


def parse_verdict(text):
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        return None
    try:
        verdict = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return verdict if isinstance(verdict, dict) and 'verdict' in verdict else None


def ask(env, text):
    body = dict(model=env['ANSWER_CHECK_MODEL'], temperature=0, messages=[dict(role='user', content=text)])
    request = urllib.request.Request(env['ANSWER_CHECK_BASE_URL'].rstrip('/') + '/chat/completions',
                                     json.dumps(body).encode('utf-8'),
                                     {'Authorization': 'Bearer ' + env['ANSWER_CHECK_API_KEY'], 'Content-Type': 'application/json',
                                      'User-Agent': 'strata-answer-check/1.0'})   # Cloudflare refuses urllib's own (1010)
    with urllib.request.urlopen(request, timeout=180) as reply:
        return json.load(reply)['choices'][0]['message']['content']


def read_one(env, case, tries=4, wait=2.):
    for attempt in range(tries):
        try:
            verdict = parse_verdict(ask(env, prompt(case)))
            if verdict:
                return verdict
        except (urllib.error.URLError, OSError, KeyError, IndexError, TypeError, ValueError) as error:
            # The reason only: an error's text can carry the request.
            print('reader', case['id'], type(error).__name__, getattr(error, 'code', ''), file=sys.stderr)
        time.sleep(wait * 2 ** attempt)
    return None


def read_all(env, records, questions, path):
    """Asks for what `path` does not hold yet and saves as it goes, so a stopped run goes on where it was."""
    verdicts = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    cases = [dict(id=r['id'], question=questions[r['id']], gold=r['answer'], reply=r['text']) for r in records
             if r['id'] not in verdicts and reply_of(r['text']) is not None]
    with ThreadPoolExecutor(int(env.get('ANSWER_CHECK_WORKERS') or 4)) as pool:
        for case, verdict in zip(cases, pool.map(lambda c: read_one(env, c), cases)):
            if verdict:
                verdicts[case['id']] = verdict
                path.write_text(json.dumps(verdicts, ensure_ascii=False, indent=1), encoding='utf-8')
    return verdicts


def read_records(arm):
    """The bench rewrites this file after each case, so a read can meet it half written."""
    try:
        return json.loads((arm / 'records.json').read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_questions(arm):
    return {c['id']: c['prompt'] for c in json.loads((arm / 'cases.json').read_text(encoding='utf-8'))}


def follow(env, arm, path, every=15.):
    """Reads a running arm beside it: the bench never waits for a verdict. Ends when the arm's report is written."""
    verdicts = {}
    while True:
        finished = (arm / 'report.json').exists()
        records = read_records(arm)
        if records:
            verdicts = read_all(env, records, read_questions(arm), path)
        if finished and records is not None:
            return verdicts
        time.sleep(every)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--follow', action='store_true', help='read a running arm as its answers arrive')
    parser.add_argument('arm', type=Path)
    parser.add_argument('--verdicts', type=Path, help='JSON of id -> verdict; default verdicts.json in the arm')
    parser.add_argument('--env', type=Path, default=Path('.env'))
    args = parser.parse_args()
    env = load_env(args.env)
    missing = [k for k in ('ANSWER_CHECK_BASE_URL', 'ANSWER_CHECK_API_KEY', 'ANSWER_CHECK_MODEL') if not env.get(k)]
    if missing:
        raise SystemExit(f'{args.env} has no value for {", ".join(missing)}')
    records = [] if args.follow else read_records(args.arm)
    questions = None if args.follow else read_questions(args.arm)
    path = args.verdicts or args.arm / 'verdicts.json'
    verdicts = follow(env, args.arm, path) if args.follow else read_all(env, records, questions, path)
    records = read_records(args.arm)
    rows = triage(records, verdicts)
    (args.arm / 'triage.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print(args.arm.name, dict(Counter(r['route'] for r in rows)))
    for r in rows:
        if r['route'] != 'agreed-right':
            print(r['route'], r['id'], 'gold', r['gold'], 'rule', r['rule_value'], r['finish'], r['verdict'])


if __name__ == '__main__':
    main()
