"""Word problems with long numbers, for firing on natural language: the answer needs a calculation no expression states.

A small fixed set, in the form external_bench reads through --math-file. Not a benchmark of the model.
"""
import argparse
import json
from pathlib import Path
import random

FORMS = (
    ('A warehouse holds {a} crates. Each crate holds {b} bottles. How many bottles are there in the warehouse? Reason in plain words.', lambda a, b: a * b),
    ('A farm sold {a} boxes at {b} dollars each. How many dollars did it receive? Reason in plain words.', lambda a, b: a * b),
    ('A library had {a} books and bought {b} more. How many books does it have now? Reason in plain words.', lambda a, b: a + b),
    ('A town had {a} people and {b} of them moved away. How many people are left? Reason in plain words.', lambda a, b: a - b),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=48)
    args = parser.parse_args()
    random_source = random.Random(20261031)
    with args.output.open('w', encoding='utf-8') as f:
        for i in range(args.count):
            form, answer = FORMS[i % len(FORMS)]
            digits = 3 + i % 3
            a = random_source.randint(10 ** (digits + 1), 10 ** (digits + 2) - 1)
            b = random_source.randint(10 ** digits, 10 ** (digits + 1) - 1)
            f.write(json.dumps(dict(problem=form.format(a=a, b=b), answer=str(answer(a, b)))) + '\n')


if __name__ == '__main__':
    main()
