"""Where long thinking goes: when the answer first appears, and what each sentence after it does."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

MOVES = dict(
    recheck=r"\b(wait|double-check|re-?check|verify|check|confirm|ensure|again|recalc|recompute)\b",
    wording=r"\b(interpret|ambiguous|wording|phrase|meaning|could mean|maybe they|parse|intended|typo)\b",
    alternative=r"\b(alternatively|another|what if|could there be|could be|or maybe|consider if|suppose)\b",
    enumerate=r"\b(for [a-z]=|case \d|try |k=\d|n=\d)",
    planning=r"\b(need|let's craft|final|answer concise|provide)\b")


def sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+|\n+', text) if s.strip()]


def audit(text, answer):
    thinking = text.split('</think>')[0]
    rows = sentences(thinking)
    plain = str(int(answer)) if float(answer) == int(answer) else str(answer)
    # As a result, not as a number the question gave: after `=`, or after a word that states an outcome.
    pattern = re.compile(rf'(?:=|(?:is|are|answer|total|gives?|get|so|thus):?)\s*\$?\s*{re.escape(plain)}(?![\d]|[.,]\d)', re.I)
    first = next((i for i, s in enumerate(rows) if pattern.search(s.replace(',', ''))), None)
    seen, repeated, moves = set(), 0, Counter()
    for s in rows[(first or 0):]:
        for name, form in MOVES.items():
            if re.search(form, s, re.I):
                moves[name] += 1
                break
        else:
            moves['working'] += 1
    for s in rows:
        repeated += s in seen
        seen.add(s)
    return dict(sentences=len(rows), first_sentence=first, repeated_sentences=repeated, moves=moves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms', type=Path, nargs='+')
    parser.add_argument('--at-least', type=int, default=1000, help='tokens in a case that counts as long')
    args = parser.parse_args()
    for arm in args.arms:
        for r in json.loads((arm / 'records.json').read_text(encoding='utf-8')):
            if r['tokens'] < args.at_least:
                continue
            row = audit(r['text'], r['answer'])
            place = '-' if row['first_sentence'] is None else f"{row['first_sentence'] / max(row['sentences'], 1):.2f}"
            print(arm.name, r['id'], r['finish'], r['tokens'], 'sentences', row['sentences'], 'answer-at', place,
                  'repeated', row['repeated_sentences'], dict(row['moves']))


if __name__ == '__main__':
    main()
