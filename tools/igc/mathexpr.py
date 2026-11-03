"""Exact arithmetic in every way a model writes it: ASCII, LaTeX at any depth, Unicode. The reader MIP's inline value and hand check run on.

The regex rewrite it replaces read one LaTeX layer and cut numbers at their commas: on Qwenputer's DeepMind rows the 4B rewrote the
question's nested expression as \\frac inside \\frac and it was never computed (14 rows lost), and `12,510,204 \\times 100` was read as
204 x 100. Other models write other forms, so the reader takes them all and refuses only what is not exact arithmetic: a symbol, a word, a
mixed number (`2\\frac{1}{3}` is ambiguous), a root or power that is not exact, a value too large to be a step.

evaluate(text) -> Fraction | None            the value of the whole text, or None
expression_before(text) -> (expr, value)     the longest arithmetic expression that ends the text (what precedes an `=`), or None
simple_quotient(expr) -> bool                an integer over an integer and nothing else (a long division's piece)"""
import re
from fractions import Fraction
from math import factorial, isqrt

MUL, DIV, POW = '*', '/', '^'
_SPACE = re.compile(r'\\[,;:! ]|\\q?quad\b|~')
_SIZES = re.compile(r'\\(?:left|right|[bB]ig{1,2}[lr]?|middle)\s*')
_SUPERSCRIPT = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'}
_WORDS = {r'\times': MUL, r'\cdot': MUL, r'\ast': MUL, r'\div': DIV, r'\frac': 'frac', r'\dfrac': 'frac', r'\tfrac': 'frac', r'\sqrt': 'sqrt',
          r'\%': '%', r'\{': '(', r'\}': ')', r'\lbrace': '(', r'\rbrace': ')', r'\lparen': '(', r'\rparen': ')'}
_SINGLE = {'*': MUL, '×': MUL, '·': MUL, '⋅': MUL, '∙': MUL, '/': DIV, '÷': DIV, '^': POW, '+': '+', '-': '-', '−': '-', '–': '-',
           '(': '(', ')': ')', '[': '(', ']': ')', '{': '{', '}': '}', '!': '!', '%': '%'}
_NUMBER = re.compile(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\.\d+')
LIMIT = 10 ** 40   # a numerator past this is no step of a calculation
DENOMINATOR = 10 ** 15


class _NotExact(Exception):
    pass


def _tokens(text):
    s = re.sub(r'(?<=\d)\\,(?=\d{3}(?!\d))', '', text)   # 1\,000: a thin space between digit groups, before it becomes a space
    s = _SIZES.sub('', _SPACE.sub(' ', s))
    s = ''.join('^' + _SUPERSCRIPT[c] if c in _SUPERSCRIPT else c for c in s).replace('**', '^')
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        m = _NUMBER.match(s, i)
        if m and (c.isdigit() or c == '.'):
            raw = m.group().replace(',', '')
            out.append(('num', Fraction(raw) if 'e' not in raw.lower() else Fraction(raw.lower())))
            i = m.end()
            if i < len(s) and (s[i].isalpha() or s[i] == '_' or (s[i] == '.' and i + 1 < len(s) and s[i + 1].isdigit())):
                raise _NotExact('a number glued to a symbol, or 1.2.3')
            continue
        if c == '\\':
            m = re.match(r'\\(?:[A-Za-z]+|.)', s[i:])
            word = m.group()
            if word not in _WORDS:
                raise _NotExact(word)   # \pi, \text, \cdots: not arithmetic
            out.append(('op', _WORDS[word]))
            i += len(word)
            continue
        if c in _SINGLE:
            out.append(('op', _SINGLE[c]))
            i += 1
            continue
        raise _NotExact(c)   # a letter, a symbol, `$`, `=`
    return out


class _Parser:
    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, value=None):
        kind, v = self.peek()
        if kind is None or (value is not None and v != value):
            raise _NotExact('expected ' + str(value))
        self.i += 1
        return kind, v

    def expr(self):
        value = self.term()
        while self.peek()[1] in ('+', '-'):
            op = self.take()[1]
            right = self.term()
            value = value + right if op == '+' else value - right
        return value

    def term(self):
        value = self.unary()
        while True:
            kind, v = self.peek()
            if v in (MUL, DIV):
                self.take()
                right = self.unary()
                if v == DIV and right == 0:
                    raise _NotExact('division by zero')
                value = value * right if v == MUL else value / right
            elif v in ('(', 'frac', 'sqrt', '{') and self.i and self.t[self.i - 1][1] in (')', '}', '!', '%') or \
                    v in ('(', 'sqrt') and self.i and self.t[self.i - 1][0] == 'num':
                value = value * self.unary()   # 2(3 + 4), (2)(3), (a)\frac{..}{..}
            elif v == 'frac' and self.i and self.t[self.i - 1][0] == 'num':
                raise _NotExact('a mixed number: 2\\frac{1}{3} is 2 1/3 or 2/3')
            elif kind == 'num' and self.i and self.t[self.i - 1][1] == ')':
                value = value * self.unary()   # (2)3
            else:
                return value
            self.bound(value)

    def unary(self):
        if self.peek()[1] in ('-', '+'):
            sign = -1 if self.take()[1] == '-' else 1
            return sign * self.unary()
        return self.power()

    def power(self):
        base = self.postfix()
        if self.peek()[1] == POW:
            self.take()
            exponent = self.unary_exponent()
            if exponent.denominator != 1 or abs(exponent) > 64:
                raise _NotExact('a power that is not a small integer one')
            if base == 0 and exponent < 0:
                raise _NotExact('0 to a negative power')
            if abs(base.numerator) > 1 and abs(exponent) * max(len(str(abs(base.numerator))), len(str(base.denominator))) > 60:
                raise _NotExact('too large')
            base = base ** int(exponent)
        return self.bound(base)

    def unary_exponent(self):
        if self.peek()[1] in ('-', '+'):
            sign = -1 if self.take()[1] == '-' else 1
            return sign * self.unary_exponent()
        return self.postfix()

    def postfix(self):
        value = self.primary()
        while self.peek()[1] in ('!', '%'):
            op = self.take()[1]
            if op == '!':
                if value.denominator != 1 or not 0 <= value <= 30:
                    raise _NotExact('a factorial past 30!')
                value = Fraction(factorial(int(value)))
            else:
                value = value / 100
        return value

    def primary(self):
        kind, v = self.peek()
        if kind == 'num':
            self.take()
            return v
        if v in ('(', '{'):
            self.take()
            value = self.expr()
            self.take(')' if v == '(' else '}')
            return value
        if v == 'frac':
            self.take()
            top, bottom = self.argument(), self.argument()
            if bottom == 0:
                raise _NotExact('division by zero')
            return top / bottom
        if v == 'sqrt':
            self.take()
            inside = self.argument()
            if inside < 0:
                raise _NotExact('the root of a negative')
            a, b = isqrt(inside.numerator), isqrt(inside.denominator)
            if a * a != inside.numerator or b * b != inside.denominator:
                raise _NotExact('a root that is not exact')
            return Fraction(a, b)
        raise _NotExact('expected a number')

    def argument(self):
        """\\frac and \\sqrt take a {group} or one digit (\\frac12)."""
        if self.peek()[1] == '{':
            self.take()
            value = self.expr()
            self.take('}')
            return value
        kind, v = self.take()
        if kind != 'num' or v.denominator != 1 or not 0 <= v <= 9:
            raise _NotExact('an argument with no braces')
        return v

    @staticmethod
    def bound(value):
        if abs(value.numerator) > LIMIT or value.denominator > DENOMINATOR:
            raise _NotExact('too large to be a step')
        return value


def evaluate(text):
    try:
        tokens = _tokens(text)
        if not tokens:
            return None
        p = _Parser(tokens)
        value = p.expr()
        if p.i != len(tokens):
            return None
        return _Parser.bound(value)
    except (_NotExact, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None


_OPERATOR = re.compile(r'[+\-*/^×÷·⋅∙−!%]|\\(?:times|cdot|ast|div|d?t?frac|sqrt)|[²³]')
_START = re.compile(r'[\d.(\[{\\\-−]')
_BOUNDARY = set(' \t$([{=:;,') | {'\\'}


def _symbol_before(head):
    """True when the text before a `-` ends in an operand (a number, a bracket, a one-letter symbol, `b_2`): the `-` is then binary."""
    h = head.rstrip()
    if not h:
        return False
    if h[-1].isdigit() or h[-1] in ')]}':
        return True
    word = re.search(r'([A-Za-z]+)(?:_\{?\w+\}?|\d*)$', h)
    return bool(word) and (len(word.group(1)) == 1 or '_' in h[word.start():])


def expression_before(text):
    """(expression, value) for the longest exact arithmetic expression that ends `text` on its last line, or None. Each start is tried from
    the left: the first that reads whole and holds an operator wins, so `12,510,204 \\times 100` is not cut at a comma and `-(25 * 130)`
    keeps its sign, while `x - 3 \\times 4` gives 3 x 4: the `-` after a symbol is not a sign."""
    line = text.rsplit('\n', 1)[-1].rstrip()
    for i in range(len(line)):
        if not _START.match(line[i]) or (i and line[i - 1] not in _BOUNDARY):
            continue
        if line[i] in '-−' and _symbol_before(line[:i]):
            continue
        expr = line[i:].strip()
        if not _OPERATOR.search(expr.lstrip('-−')):
            continue
        value = evaluate(expr)
        if value is not None:
            return expr, value
    return None


def simple_quotient(expr):
    """An integer over an integer and nothing else: `21 \\div 19`, `21/19`, `\\frac{-2142}{-7037}`."""
    bare = re.sub(r'\s|\\left|\\right|[()]', '', expr)
    return bool(re.fullmatch(r'-?\d+(?:/|\\div|÷)-?\d+|\\d?t?frac\{-?\d+\}\{-?\d+\}', bare))
