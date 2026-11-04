"""question_read.py: the computation a question states, read by software before the model starts (MIP catch point 1, #298).

pc4-whole lost 4 of 160 DeepMind rows because the model copied the question's numbers wrong, then the software computed exactly what
it wrote. Reading the question itself removes that step. A question gives either an expression (`What is 20*(-8)/(-60)*329?`) or the
English for one (`Subtract 3 from 7`, `What is 3 less than 7?`, `the product of 2 and 5`).

`read(question)` -> (expression in plain notation, exact value) or None. None when the question holds no single computation this
reader is sure of: a word problem, two candidate expressions, or symbols.
"""
import re
from fractions import Fraction

from tools.igc.mathexpr import evaluate

NUM = r'(-?\d[\d,]*(?:\.\d+)?)'
LEAD = r'(?:what is|what\'s|calculate|compute|evaluate|work out|find|determine|give)(?: the value of)?'
# (pattern, the expression it means); a and b are the first and second number as written
VERBAL = [
    (rf'(?:the )?product of {NUM} and {NUM}', '({a})*({b})'),
    (rf'multiply {NUM} (?:and|by) {NUM}', '({a})*({b})'),
    (rf'{NUM} (?:times|multiplied by) {NUM}', '({a})*({b})'),
    (rf'divide {NUM} by {NUM}', '({a})/({b})'),
    (rf'{NUM} divided by {NUM}', '({a})/({b})'),
    (rf'(?:the )?(?:sum|total) of {NUM} and {NUM}', '({a})+({b})'),
    (rf'(?:sum|add|put together|combine) {NUM} and {NUM}', '({a})+({b})'),
    (rf'add {NUM} to {NUM}', '({b})+({a})'),
    (rf'{NUM} plus {NUM}', '({a})+({b})'),
    (rf'{NUM} more than {NUM}', '({b})+({a})'),
    (rf'subtract {NUM} from {NUM}', '({b})-({a})'),
    (rf'take {NUM} (?:away )?from {NUM}', '({b})-({a})'),
    (rf'{NUM} take away {NUM}', '({a})-({b})'),
    (rf'(?:the )?distance between {NUM} and {NUM}', '|({a})-({b})|'),
    (rf'{NUM} less than {NUM}', '({b})-({a})'),
    (rf'{NUM} minus {NUM}', '({a})-({b})'),
    (rf'(?:the )?difference between {NUM} and {NUM}', '({a})-({b})'),
]


def _clean(text):
    first = text.strip().split('\n')[0].strip()   # the question, without the answer-format line after it
    return re.sub(r'\s+', ' ', first)


def read(question):
    q = _clean(question)
    body = re.sub(rf'^{LEAD}\s*', '', q, flags=re.I).rstrip(' ?.')
    for pattern, form in VERBAL:
        m = re.fullmatch(pattern, body, flags=re.I)
        if m:
            a, b = (x.replace(',', '') for x in m.groups())
            expr = form.format(a=a, b=b)
            value = evaluate(expr)
            return (expr, value) if value is not None else None
    if re.search(r'[A-Za-z]', body):   # words left: a word problem, or symbols
        return None
    value = evaluate(body)
    return (body, value) if value is not None and re.search(r'[-+*/×÷^]', body.lstrip('-')) else None


def as_answer(value):
    """The value as DeepMind spells it: an integer, a terminating decimal, or a reduced fraction."""
    from tools.igc.tir4b import spelled
    return spelled(Fraction(value))
