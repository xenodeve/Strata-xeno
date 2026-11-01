"""Scoring for the published few-shot protocols. MATH: the extraction and normalization of Lewkowycz et al. 2022, appendix D.
BBH: the text after `the answer is`, as the CoT exemplars end. Strings are compared after normalizing: the harness's sympy fallback is not
used, so a right answer in another form counts wrong here and MATH is a lower bound."""
import re

SUBSTITUTIONS = [('an ', ''), ('a ', ''), ('.$', '$'), ('\\$', ''), (r'\ ', ''), (' ', ''), ('mbox', 'text'), (',\\text{and}', ','),
                 ('\\text{and}', ','), ('\\text{m}', '\\text{}')]
REMOVED = ['square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft', 'hours', 'km', 'units', '\\ldots', 'sue', 'points', 'feet', 'minutes',
           'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds', 'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
           '\\text{s}', '\\text{.}', '\\text{\ns}', '\\text{}^2', '\\text{}^3', '\\text{\n}', '\\text{}', r'\mathrm{th}', r'^\circ', r'^{\circ}',
           r'\;', r',\!', '{,}', '"', '\\dots']


def normalize_math(answer):
    answer = answer.split('=')[-1]
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
    return text.rsplit('</think>', 1)[-1].replace('<|im_end|>', '')


def math_answer(text):
    # The base model copies the exemplars' `I hope it is correct.`; a post-trained model stops at the answer.
    found = re.search(r'[Tt]he final answer is(.*?)(?:\.\s*I hope it is correct|\.?\s*(?:\n|$))', reply(text), re.S)
    if found:
        return found.group(1).strip()
    return last_boxed(reply(text))   # a reply with the thinking on ends in a box, not in the exemplars' sentence


def math_right(text, gold):
    answer = math_answer(text)
    return answer is not None and normalize_math(answer) == normalize_math(gold)


def bbh_answer(text):
    found = re.search(r'the answer is\s*(.*?)(?:\.\s|\.$|\n|$)', reply(text), re.I | re.S)
    if not found:
        return None
    return found.group(1).strip().strip('*').strip()


def bbh_right(text, gold):
    answer = bbh_answer(text)
    return answer is not None and answer.lower().rstrip('.') == gold.strip().lower().rstrip('.')


def letter_answer(text):
    """MMLU-Pro: `the answer is (X)`, then `Answer: X`. SuperGPQA ends in `Answer: X.`; the last such line counts, as both evaluators read it."""
    found = re.findall(r'answer is \(?\$?([A-P])\b', reply(text)) or re.findall(r'[Aa]nswer:\s*\(?\$?([A-P])\b', reply(text))
    return found[-1] if found else None
