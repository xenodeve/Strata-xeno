"""contest_math.py: recent public contest problems as rows (MathArena's datasets: AIME 2025 / 2026, HMMT Feb 2025, CMIMC 2025), and their scoring (#298).

Why: LiveBench's public questions stop at the 2024-11-25 release (README: "not all questions for this release are public"; the Hugging Face
sets were last pushed 2025-04-07), so a model trained after that may have seen them. These are newer, with a short exact answer each. They do
not replace LiveBench; they check, on questions the base cannot have memorised as easily, that a gain seen there is not recall.

Answer rule: the last `\\boxed{}` of the reply is read with mathexpr (an integer, a fraction, a decimal, `\\frac{a}{b}`) and must equal the
reference exactly; when mathexpr cannot read either side the two strings are compared after dropping spaces and `\\left`, `\\right`, `\\,`.

    python -m tools.igc.contest_math rows <data dir> <out.jsonl>     # <data dir>/<set>/data/train-*.parquet exported to <set>.jsonl first
"""
import json
import re
import sys
from pathlib import Path

from tools.igc.external_score import question_id
from tools.igc.livebench import last_boxed_only_string, remove_boxed
from tools.igc.mathexpr import evaluate
from tools.igc.standard_score import reply

SUITE = 'contest'
TAIL = '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'


def rows(data, sets=None, limit=30000):
    out = []
    for path in sorted(Path(data).glob('*.jsonl')):
        name = path.stem
        if sets and name not in sets:
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            q = r['problem'] + TAIL
            out.append(dict(id=question_id(SUITE, q), suite=SUITE, question=q, prompt=q, answer=str(r['answer']), limit=limit, group=name,
                            calculation=True, source_id=f"{name}-{r['problem_idx']}"))
    return out


def _plain(s):
    s = re.sub(r'\\(?:left|right|,|;|!)', '', s)
    s = s.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac').replace('\\%', '').replace('$', '')
    return re.sub(r'\s+', '', s)


def _value(s):
    try:
        return evaluate(_plain(s))
    except Exception:
        return None


def right(row, text):
    boxed = last_boxed_only_string(reply(text))
    if boxed is None:
        return False
    got, want = remove_boxed(boxed), row['answer']
    a, b = _value(got), _value(want)
    if a is not None and b is not None:
        return a == b
    return _plain(got) == _plain(want)


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'rows':   # rows <data dir> <out.jsonl>
    rs = rows(sys.argv[2])
    Path(sys.argv[3]).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rs), encoding='utf-8', newline='\n')
    from collections import Counter
    print(len(rs), 'rows', dict(Counter(r['group'] for r in rs)))
