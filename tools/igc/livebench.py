"""livebench.py: LiveBench's questions as rows, and LiveBench's own scoring rules (#298: the standard external set, compared with the base).

The rules are ported from github.com/LiveBench/LiveBench (Apache-2.0), `livebench/process_results/...` and `gen_ground_truth_judgment.py`,
at commit 1d1006a0 (2026-10-18), not imported: the originals pull in openai, lark and latex2sympy2 at import time. What differs:
- AMPS_Hard compares two LaTeX answers with sympy (optional here: imported when the first AMPS_Hard answer is scored). The original asks
  an OpenAI model (`gpt-5-mini`) when sympy says "not equal" or cannot parse. This port sends nothing anywhere: such an answer scores 0 and
  `score_detail` says `unverified`, so the number of answers that were never decided is counted, not hidden.
- Everything else is the original, line for line in its decisions.

The public questions are LiveBench's older releases (the Hugging Face datasets were last updated 2025-04-07; the project's releases run to
2026-06-25 and the newest questions are not public). A row keeps its release date and removal date; `rows(..., current_only=True)` keeps the
questions not removed by 2025-04-02, the last public removal. Scores here are for the public set only, not the leaderboard's.

    python -m tools.igc.livebench rows <data dir> <out.jsonl>        # the data dir holds math.jsonl and reasoning.jsonl (parquet, exported)
    python -m tools.igc.livebench selftest <data dir>                # every question's own ground truth must score 1
"""
import itertools
import json
import re
import sys
from pathlib import Path

from tools.igc.external_score import question_id
from tools.igc.standard_score import reply

SUITE = 'livebench'


# ------------------------------------------------------------------------------------------------ rows
def rows(data, current_only=True, tasks=None, limit=30000):
    """The math and reasoning questions as tir4b rows. `question` is the first turn, verbatim: LiveBench's own prompt, with its format rules."""
    out = []
    for name in ('math', 'reasoning'):
        path = Path(data) / f'{name}.jsonl'
        if not path.exists():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            if current_only and r.get('livebench_removal_date') not in (None, '') and str(r['livebench_removal_date']) <= '2025-04-02':
                continue
            if tasks and r['task'] not in tasks:
                continue
            turns = r['turns']
            q = turns[0] if isinstance(turns, list) else str(turns)
            out.append(dict(id=question_id(SUITE, q), suite=SUITE, question=q, prompt=q, answer=str(r['ground_truth']), limit=limit,
                            group=r['task'], subtask=r.get('subtask') or r['task'], category=r.get('category', name),
                            release=str(r.get('livebench_release_date'))[:10], removal=str(r.get('livebench_removal_date') or '')[:10],
                            lb_id=str(r['question_id'])))
    return out


# ------------------------------------------------------------------------------------------------ the original helpers
def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if '\\boxed ' in string:
        return '\\boxed ' + string.split('\\boxed ')[-1].split('$')[0]
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None
    i, right, depth = idx, None, 0
    while i < len(string):
        if string[i] == '{':
            depth += 1
        if string[i] == '}':
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    return None if right is None else string[idx:right + 1].replace('$', '').replace('fbox', 'boxed')


def remove_boxed(s):
    if '\\boxed ' in s:
        return s[len('\\boxed '):]
    left = '\\boxed{'
    assert s[:len(left)] == left and s[-1] == '}'
    return s[len(left):-1]


def levenshtein(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(len(a) + 1):
        dp[i][0] = i
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] if a[i - 1] == b[j - 1] else 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(a)][len(b)]


# ------------------------------------------------------------------------------------------------ math competitions (AMC, AIME) and olympiad
def amc_choice_text(statement, letter):
    matches = re.findall(r'\\textbf{\(([A-E])\)\s?}(.*?)(?:\\qquad|\$)', statement)
    answer = {m[0]: m[1].strip() for m in matches}.get(letter, None)
    if not answer:
        answer = 'FAILURE'
    return answer.strip().strip('$').strip('~')


def mathcontest(ground_truth, answer, question_text):
    score = 0
    solution = re.findall(r'<solution>(.*?)</solution>', answer)
    if solution and len(set(solution[-1])) == 1 and next(iter(set(solution[-1]))).lower() == ground_truth.lower():
        score = 1
    if ground_truth * 4 in answer:
        score = 1
    if score == 0:
        answer = answer.replace('\\\\fbox{', '\\\\boxed{')
        boxed = last_boxed_only_string(answer)
        if boxed:
            got = remove_boxed(boxed).replace('\\text{', '').replace('}', '').replace('\\', '').lower()
            if got in {'a', 'b', 'c', 'd', 'e'} and got == ground_truth.lower():
                score = 1
    if score == 0:
        value = amc_choice_text(question_text, ground_truth)
        if value in answer[-(20 + len(value)):]:
            score = 1
    if score == 0:
        last = answer.strip().split('\n')[-1]
        if last.strip().replace('*', '').lower() == ground_truth.lower():
            score = 1
        elif '(' in last and ')' in last and last.split('(')[1].split(')')[0].lower() == ground_truth.lower():
            score = 1
    return score


def aime(ground_truth, answer):
    solution = re.findall(r'<solution>(.*?)</solution>', answer)
    if solution and len(set(solution[-1])) == 1 and next(iter(set(solution[-1]))).lower() == ground_truth.lower():
        return 1
    return 1 if ground_truth in answer[-50:] else 0


def _digits_at_ends(s):
    start = 0
    while start < len(s) and not s[start].isdigit():
        start += 1
    end = start
    while end < len(s) and s[end].isdigit():
        end += 1
    return s[start:end], len(s) - (end - start)


def _olympiad_numbers(generation):
    numbers = None
    if 'answer:' in generation.lower():
        lines = generation.lower().strip().split('\n')
        line = idx = None
        for i, ln in enumerate(lines):
            if 'answer:' in ln:
                line, idx = ln, i
        text = line.split('answer:')[1].replace('answer:', '').replace('**', '').replace('.', '').strip()
        if text == '' and idx < len(lines) - 1:
            text = lines[idx + 1].replace('answer:', '').replace('**', '').replace('.', '').strip()
        numbers = []
        for n in text.split(','):
            n = n.strip().split(' ')[-1].replace('$', '').replace('{', '').replace('}', '').replace('\\', '').replace('boxed', '').replace('<', '').replace('>', '')
            try:
                numbers.append(int(n))
            except ValueError:
                numbers.append('NO ANSWER')
        if not numbers or set(numbers) == {'NO ANSWER'}:
            numbers = None
    if numbers is None and '\\boxed' in generation:
        boxed = last_boxed_only_string(generation)
        string = remove_boxed(boxed) if boxed is not None else generation
        string = string.replace('\\text{', '').replace('}', '').replace('\\', '')
        numbers = []
        for n in string.strip().split(','):
            try:
                numbers.append(int(n.strip()))
            except ValueError:
                numbers.append('NO ANSWER')
        if not numbers or set(numbers) == {'NO ANSWER'}:
            numbers = None
    if numbers is None:
        numbers = []
        for n in generation.strip().lower().split('\n')[-1].strip().split(','):
            n, _ = _digits_at_ends(n)
            if not n.strip():
                continue
            try:
                numbers.append(int(n.strip()))
            except ValueError:
                numbers.append('NO ANSWER')
        if not numbers or set(numbers) == {'NO ANSWER'}:
            numbers = None
    if numbers is None:
        parts = [k.strip() for k in generation.lower().split('answer:')[-1].split(',')]
        numbers = []
        for i, n in enumerate(parts):
            n, removed = _digits_at_ends(n)
            if n != '' and n != '₂':
                numbers.append(int(n))
            if i > 0 and removed > 0:
                break
    return numbers


def olympiad(ground_truth, answer):
    """imo / usamo: the share of the masked formulae put back in the right place (edit distance, as the original does for these two)."""
    truth = [int(n) for n in ground_truth.split(',')]
    got = _olympiad_numbers(answer)
    denom = max(len(got), len(truth))
    return 1 - levenshtein(got, truth) / denom if denom else 0


# ------------------------------------------------------------------------------------------------ AMPS_Hard (sympy)
def _normalize_final(final):
    final = final.split('=')[-1]
    final = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final)
    for tag in ('text', 'textbf', 'overline'):
        final = re.sub(r'(\\' + tag + r'\{)(.*?)(\})', '\\2', final)
    final = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final)
    final = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final)
    final = re.sub(r'(sqrt)([^{\[])', 'sqrt{\\2}', final)
    final = final.replace('$', '')
    if final.replace(',', '').isdigit():
        final = final.replace(',', '')
    return final


def _sympy_parse(x):
    import sympy
    from sympy.parsing.latex import parse_latex
    try:
        parsed = parse_latex(x, backend='lark')
    except Exception:
        try:
            parsed = parse_latex(x.replace('\\\\', '\\'), backend='lark')
        except Exception:
            try:
                from latex2sympy2 import latex2sympy
                return [latex2sympy(x)]
            except Exception:
                return []
    return list(getattr(parsed, 'children', [parsed]))


def _sympy_compare(x1, x2):
    import lark   # noqa: F401 - sympy's LaTeX reader needs it; without it every parse comes back empty and every answer "unverified" in silence (lb1, 2026-11-04)
    import sympy
    one, two = _sympy_parse(x1), _sympy_parse(x2)
    for a in one:
        for b in two:
            try:
                diff = sympy.simplify(a - b)
                if diff == 0 or abs(diff) < 0.001:
                    return True
            except Exception:
                continue
    return False


def _compare_in_process(queue, x1, x2):
    try:
        queue.put(_sympy_compare(x1, x2))
    except BaseException as e:   # ImportError included: the caller tells it from a False
        queue.put(e)


def _sympy_equal(x1, x2, seconds=60):
    """True when sympy proves the two LaTeX answers equal. False when it cannot: the original then asks an LLM; here that is `unverified`.
    Run in its own process with a time limit, as the original does: simplify on a model's answer can run for ever."""
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    proc = ctx.Process(target=_compare_in_process, args=(queue, x1, x2))
    proc.start()
    proc.join(seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False
    got = queue.get() if not queue.empty() else False
    if isinstance(got, ImportError):
        raise got
    return got is True


def amps_hard(ground_truth, answer):
    """(score, note). note: 'string' (the closing text equals the answer), 'sympy', 'wrong' (decided by sympy), 'unverified' (sympy could not tell)."""
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[-1]
    for old, new in (('+C', ''), ('+ C', ''), ('+ c', ''), ('+c', ''), ('\\\\fbox{', '\\\\boxed{'), ('\\dfrac', '\\frac'), ('\\tfrac', '\\frac'),
                     ('\\left', ''), ('\\right', ''), ('\\bigl', ''), ('\\bigr', ''), ('\\Bigl', ''), ('\\Bigr', ''), ('\\,', ''), ('\\;', ''),
                     ('\n', ''), ('\\cdot', '*')):
        answer = answer.replace(old, new)
    for old, new in (('\\left', ''), ('\\right', ''), (' ^', '^'), ('\\ ', '*')):
        ground_truth = ground_truth.replace(old, new)
    parsed = None
    boxed = last_boxed_only_string(answer)
    if boxed:
        parsed = _normalize_final(remove_boxed(boxed))
    if parsed is None:
        last = answer.split('\n')[-1]
        if last.count('$') >= 2:
            close = last.rfind('$')
            if last[close - 1] == '$':
                close -= 1
            math = last[last.rfind('$', 0, close) + 1:close]
            if '=' in math:
                math = math.split('=')[-1].strip()
            elif '\\quad \\text{or} \\quad' in math:
                math = math.split('\\quad \\text{or} \\quad')[-1].strip()
            parsed = _normalize_final(math)
    if parsed is not None:
        if parsed.strip() == ground_truth.strip():
            return 1, 'string'
        try:
            if _sympy_equal(ground_truth, parsed):
                return 1, 'sympy'
        except ImportError:
            return 0, 'unverified: sympy or lark is not installed'   # not a wrong answer: the scorer could not look
        return 0, 'unverified'   # the original asks an LLM here
    tail = answer[:-1] if answer.endswith('.') else answer
    return (1, 'string') if ground_truth == tail[-len(ground_truth):] else (0, 'wrong')


# ------------------------------------------------------------------------------------------------ reasoning
def zebra(ground_truth, answer):
    truth = ground_truth.split(',')
    solution = re.findall(r'<solution>(.*?)</solution>', answer) or re.findall(r'</solution>(.*?)</solution>', answer)
    if not solution:
        boxed = last_boxed_only_string(answer.replace('\\\\fbox{', '\\\\boxed{'))
        if boxed:
            solution.append(remove_boxed(boxed).replace('\\text{', '').replace('}', '').replace('\\', ''))
    if not solution:
        last = answer.strip().split('\n')[-1]
        if last.count(',') == len(truth) - 1:
            solution.append(last)
    if not solution:
        return 0
    if len(solution) > 1:
        every = []
        for m in solution:
            every += m.split(',')
        text = every[-len(truth):]
    else:
        text = solution[-1].split(',')
    right = 0
    for i, want in enumerate(truth):
        want = want.strip().lower().replace('-', ' ')
        if i >= len(text):
            continue
        got = text[i].strip().lower().replace('-', ' ').replace('position', '')
        if want == got or want in got:
            right += 1
    return ((right == len(truth)) + right / len(truth)) / 2


def spatial(ground_truth, answer):
    if answer == ground_truth:
        return 1
    numbers = {w: str(i) for i, w in enumerate(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
                                                 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'])}
    bold = re.findall(r'\*\*([^\*]+)\*\*', answer)
    gt = ground_truth.strip().lower()
    score = 0
    for i in range(3):
        if len(bold) > i:
            word = bold[-i - 1].strip().lower()
            if word == gt or numbers.get(word) == gt:
                score = 1
            for shape in ('tetrahedra', 'tetrahedron', 'triangle', 'square'):
                if gt == shape and shape in word and len(word) < 2 * len(shape) + 5:
                    score = 1
    if score == 0:
        boxed = last_boxed_only_string(answer.replace('\\\\fbox{', '\\\\boxed{'))
        if boxed:
            got = remove_boxed(boxed)
            for old in ('\\textbf{', '\\mathbf{', '\\text{', '}'):
                got = got.replace(old, '')
            if got == ground_truth:
                score = 1
    return score


def web_of_lies(ground_truth, answer):
    parsed = None
    solution = re.findall(r'<solution>(.*?)</solution>', answer) or re.findall(r'</solution>(.*?)</solution>', answer)
    if solution:
        parsed = solution[-1]
    bold = re.findall(r'\*\*(.*?)\*\*', answer)
    if parsed is None and bold:
        words = [w.lower().strip().replace(',', '').replace('.', '')[0:max(len(w), 3)] for m in bold for w in m.split()]
        parsed = []
        i = len(words) - 1
        while i >= 0 and len(parsed) < 3:
            if words[i] in ('yes', 'no', 'unknown'):
                parsed = [words[i]] + parsed
            i -= 1
        parsed = ', '.join(parsed) if parsed else None
    if parsed is None or parsed.strip() == '':
        answer = answer.replace('\\\\boxed{\\\\textbf{', '\\\\boxed{').replace('\\\\fbox{', '\\\\boxed{').replace('\\textbf{', '\\boxed{')
        boxed = last_boxed_only_string(answer)
        if boxed:
            parsed = remove_boxed(boxed)
    if parsed is None:
        final, at = None, -1
        for comb in itertools.product(['yes', 'no', 'unknown'], repeat=3):
            i = answer.lower().find(', '.join(comb))
            if i != -1 and i > at:
                final, at = comb, i
        if final is not None:
            parsed = ', '.join(final)
    score = 0
    if parsed and parsed == ground_truth.lower():
        score = 1
    if parsed and parsed.count('yes') + parsed.count('no') + parsed.count('unknown') == 3 and ground_truth.lower() in parsed:
        score = 1
    return score


# ------------------------------------------------------------------------------------------------ dispatch
def score_detail(row, text):
    """(score in 0..1, note). `text` as the bench passes it: the thinking, `</think>`, the reply; nothing but the reply is scored."""
    said = reply(text)
    truth, sub, task = row['answer'], row.get('subtask') or row['group'], row['group']
    parts = sub.split('_')
    if parts[0] in ('amc', 'smc') or (len(parts) > 1 and parts[1] == 'amc'):
        return mathcontest(truth, said, row['question']), 'amc'
    if parts[0] == 'aime':
        return aime(truth, said), 'aime'
    if parts[0] in ('imo', 'usamo'):
        return olympiad(truth, said), 'olympiad'
    if 'amps_hard' in sub or 'amps_hard' in task.lower():
        return amps_hard(truth, said)
    if task == 'web_of_lies_v2':
        return web_of_lies(truth, said), 'web_of_lies_v2'
    if task == 'zebra_puzzle':
        return zebra(truth, said), 'zebra'
    if task == 'spatial':
        return spatial(truth, said), 'spatial'
    raise ValueError(f'no LiveBench scorer for task {task!r} subtask {sub!r}')


def score(row, text):
    return score_detail(row, text)[0]


def right(row, text):
    """Full marks. The partial credit of zebra puzzles and olympiad orderings is in `score`."""
    return score(row, text) >= 1.0


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'rows':   # rows <data dir> <out.jsonl>
    rs = rows(sys.argv[2])
    Path(sys.argv[3]).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rs), encoding='utf-8', newline='\n')
    from collections import Counter
    print(len(rs), 'rows', dict(Counter(r['group'] for r in rs)))
elif __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'selftest':   # selftest <data dir>
    from collections import Counter
    bad, notes = [], Counter()
    for r in rows(sys.argv[2], current_only=False):
        gt = r['answer']
        # the ground truth written the way the question asks, once per task
        if r['group'] == 'math_comp':
            said = f'<solution>{gt * 5}</solution>' if gt in 'ABCDE' else f'<solution>{gt}</solution>'
        elif r['group'] == 'AMPS_Hard':
            said = f'\\boxed{{{gt}}}'
        elif r['group'] == 'olympiad':
            said = 'answer: ' + gt
        elif r['group'] == 'spatial':
            said = f'**{gt}**'
        else:
            said = f'<solution>{gt}</solution>'
        s, note = score_detail(r, '</think>' + said)
        notes[(r['group'], note)] += 1
        if s < 1.0:
            bad.append((r['group'], r['subtask'], gt[:40], note))
    print('own ground truth scored 1:', sum(notes.values()) - len(bad), 'of', sum(notes.values()), '; notes', dict(notes))
    for b in bad[:12]:
        print('  not 1:', b)
