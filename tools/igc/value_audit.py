"""Checks every plain `a op b ... = value` in an arm's text, so a wrong value is seen even when the final answer survives."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re

MINUS = chr(0x2212)
NUMBER = r'\d[\d,]*(?:\.\d+)?'
EXPRESSION = re.compile(rf'(?<![\w.,^)}}=*/+\-\\])({NUMBER}(?:\s*(?:[+*/×\-]|\\times|\\cdot|\\div)\s*{NUMBER})+)\s*=\s*(-?{NUMBER})(?![\d^(]|[.,]\d|\s*[*/+\-×]|\s*\\(?:times|cdot|div))')


def evaluate(text):
    text = re.sub(r'\\times|\\cdot|×', '*', text).replace('\\div', '/').replace(',', '')
    tokens = re.findall(r'\d+(?:\.\d+)?|[+*/\-]', text)
    terms, sign, value = [], 1, None
    pending = None
    for token in tokens:
        if token in '*/':
            pending = token
        elif token in '+-':
            terms.append(sign * value)
            sign, value, pending = 1 if token == '+' else -1, None, None
        else:
            number = Fraction(token)
            if pending == '*':
                value *= number
            elif pending == '/':
                if number == 0:
                    return None
                value /= number
            else:
                value = number
    terms.append(sign * value)
    return sum(terms)


def audit(text):
    """(expression, written, right) for each wrong equality; the count of those checked."""
    checked, wrong = 0, []
    for match in EXPRESSION.finditer(text.replace(MINUS, '-')):
        right = evaluate(match[1])
        if right is None:
            continue
        written = Fraction(match[2].replace(',', ''))
        checked += 1
        if abs(written - right) >= Fraction(10) ** -len((match[2].split('.') + [''])[1]):   # rounded or cut, not wrong
            wrong.append((match[1], match[2], float(right)))
    return checked, wrong


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms', type=Path, nargs='+')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    for arm in args.arms:
        totals = Counter()
        for record in json.loads((arm / 'records.json').read_text(encoding='utf-8')):
            checked, wrong = audit(record['text'])
            totals.update(checked=checked, wrong=len(wrong))
            if args.list:
                for item in wrong:
                    print(' ', record['id'], item)
        print(arm.name, dict(totals))


if __name__ == '__main__':
    main()
