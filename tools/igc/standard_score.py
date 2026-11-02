"""Scoring for the published few-shot protocols. MATH: the extraction and normalization of Lewkowycz et al. 2022, appendix D.
BBH: the text after `the answer is`, as the CoT exemplars end. Strings are compared after normalizing: the harness's sympy fallback is not
used, so a right answer in another form counts wrong here and MATH is a lower bound."""
import re
from fractions import Fraction

SUBSTITUTIONS = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''), (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
                 ('\\text{and}', ','), ('\\text{m}', '\\text{}')]
REMOVED = ['square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft', 'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes',
           'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
           '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}',
           r'\;', r',\!', '{,}', '"', '\\dots']


def normalize_math(answer):
    answer = answer.split('=')[-1]
    # Same value, other spelling: the harness leaves these to sympy, which is not used here.
    for spelling in ('tfrac', 'dfrac', 'cfrac'):
        answer = answer.replace(spelling, 'frac')
    answer = answer.replace('\\left', '').replace('\\right', '').replace('\\!', '').replace('\\%', '').replace('%', '')
    answer = re.sub(r'(frac\{[^{}]*\})([^{\s])', r'\1{\2}', answer)
    for word in REMOVED:
        if word.isalpha():
            answer = re.sub(rf'(?<!\\)\b{word}\b', '', answer)
    for before, after in SUBSTITUTIONS:
        answer = answer.replace(before, after)
    for word in REMOVED:
        if word.isalpha():
            parts = re.split(r'(\\[a-zA-Z]+)', answer)
            for i in range(0, len(parts), 2):
                parts[i] = parts[i].replace(word, '')
            answer = ''.join(parts)
        else:
            answer = answer.replace(word, '')
    answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', answer)
    answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', answer)
    answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', answer)
    answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', answer)
    answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', answer)
    answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', answer)
    answer = re.sub(r'(sqrt)(\[[^\]]*\])(\w)', '\\1\\2{\\3}', answer)
    answer = re.sub(r'(sqrt)(?!\[)([^{])', 'sqrt{\\2}', answer)
    answer = answer.replace('$', '')
    if re.fullmatch(r'-?\d{1,3}(,\d{3})+', answer):
        answer = answer.replace(',', '')
    return answer


def last_boxed(solution):
    """The gold answer of a MATH row: the content of its last \\boxed."""
    if '\\boxed ' in solution:
        return solution.split('\\boxed ')[-1].split('$')[0]
    start = solution.rfind('\\boxed')
    if start < 0:
        start = solution.rfind('\\fbox')
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(solution)):
        depth += (solution[i] == '{') - (solution[i] == '}')
        if solution[i] == '}' and depth == 0:
            return solution[solution.index('{', start) + 1:i]
    return None


def reply(text):
    """Nothing when the thinking never closed: a guess in a cut-off reasoning is not an answer, as in final_number."""
    parts = text.rsplit('</think>', 1)
    return parts[1].replace('<|im_end|>', '') if len(parts) == 2 else ''


def math_answer(text):
    # The base model copies the exemplars' `I hope it is correct.`; a post-trained model stops at the answer.
    said = reply(text)
    found = re.search(r'[Tt]he final answer is(.*?)(?:\.\s*I hope it is correct|\.?\s*(?:\n|$))', said, re.S)
    if found:
        return found.group(1).strip()
    boxed = last_boxed(said)   # a reply with the thinking on ends in a box, not in the exemplars' sentence
    if boxed is not None:
        return boxed
    # No box, when the prompt asked for none (m5o/m5t, #288): the last bold answer, else the last displayed line
    bold = re.findall(r'\*\*\s*\$?\\?\(?\s*([^*]+?)\s*\\?\)?\$?\s*\*\*', said)
    if bold:
        return bold[-1].rstrip('.')
    shown = re.findall(r'\\\[(.*?)\\\]|\$\$(.*?)\$\$', said, re.S)
    if shown:
        last = next(x for x in shown[-1] if x).strip()
        return last.splitlines()[-1].strip().rstrip('.,') if last else None
    return None


def math_value(answer):
    """The number a normalized answer spells, as a Fraction, or None: `\\frac{5}{2}`, `-\\frac{3}{4}`, `5/2`, `2.5`."""
    s = answer.strip()
    if s.startswith('{') and s.endswith('}') and s.count('{') == 1:
        s = s[1:-1]
    sign = -1 if s.startswith('-') else 1
    s = s.lstrip('-')
    m = re.fullmatch(r'\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}', s) or re.fullmatch(r'(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)', s)
    try:
        if m:
            return sign * Fraction(m.group(1)) / Fraction(m.group(2))
        if re.fullmatch(r'\d+(?:\.\d+)?', s):
            return sign * Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def math_right(text, gold):
    answer = math_answer(text)
    if answer is None:
        return False
    got, want = normalize_math(answer), normalize_math(gold)
    if got == want:
        return True
    a, b = math_value(got), math_value(want)   # the same number in another spelling: 5/2 and 2.5
    return a is not None and a == b


def bbh_answer(text):
    found = re.search(r'the answer is\s*(.*?)(?:\.\s|\.$|\n|$)', reply(text), re.I | re.S)
    if not found:
        return None
    return found.group(1).strip().strip('*').strip()


def bbh_right(text, gold):
    answer = bbh_answer(text)
    return answer is not None and answer.lower().rstrip('.') == gold.strip().lower().rstrip('.')


def hle_answer(text):
    """The line the benchmark's prompt asks for, last one wins."""
    found = re.findall(r'Exact Answer:\s*(.+)', reply(text))
    return found[-1].strip() if found else None


def hle_number(text):
    """The number a short answer states, as an exact fraction: after its last `=`, the first one. None when it holds none."""
    text = re.sub(r'\\d?frac\{(\d+)\}\{(\d+)\}', r'\1/\2', str(text)).replace(',', '').rsplit('=', 1)[-1]
    found = re.search(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
    if not found:
        return None
    try:
        return Fraction(found.group())
    except (ValueError, ZeroDivisionError):
        return None


def hle_right(text, gold):
    """Exact, except that a gold with decimals allows one unit in its last place. Only numeric golds are scored here, which needs no judge."""
    got, want = hle_answer(text), hle_number(gold)
    value = None if got is None else hle_number(got)
    if value is None or want is None:
        return False
    places = len(str(gold).split('.')[1]) if '.' in str(gold) else None
    return value == want if places is None else abs(value - want) <= Fraction(1, 10 ** places)


def letter_answer(text):
    """MMLU-Pro: `the answer is (X)`. SuperGPQA ends in `Answer: X.` The last one in the reply counts, whichever form: a reply weighs options on its
    way there, and `Maybe the answer is A?` settles nothing."""
    found = re.findall(r'[Aa]nswer(?: is|:)\s*\(?\$?\*{0,2}([A-P])\b(?!\s*\?)', reply(text))
    return found[-1] if found else None
