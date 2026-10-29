"""Separately versioned matched development preparation; no model execution."""
import argparse
from collections import defaultdict
import hashlib
from itertools import islice, product
import json
from pathlib import Path
import random

SOURCE_FILE_SHA256 = '0aa4fa0c34e45b47f213279a638d7d11d1a04f0256877ddcf5e8be6c803452d9'
SOURCE_ROWS_SHA256 = 'fe0e9d86210b4c0c7ef544a7b4a5982dc9c8f53c3a3eebada7a2e34b2356fa7c'
SEED = 20261108
PATTERNS = {
    ('seen', 'en', 'compute'): 'Calculate {expr}. Give the exact integer result.',
    ('seen', 'en', 'quote-copy'): 'Copy the quoted expression "{expr}" exactly. Do not calculate it.',
    ('seen', 'th', 'compute'): 'คำนวณ {expr} แล้วตอบเป็นจำนวนเต็ม',
    ('seen', 'th', 'quote-copy'): 'คัดลอกข้อความ "{expr}" ตามเดิม ห้ามคำนวณ',
    ('new', 'en', 'compute'): 'Work out {expr} and give its value.',
    ('new', 'en', 'quote-copy'): 'Treat "{expr}" as quoted data; copy its characters without evaluation.',
    ('new', 'th', 'compute'): 'ช่วยคิดโจทย์ {expr} และบอกค่าที่ได้',
    ('new', 'th', 'quote-copy'): 'ข้อความ "{expr}" เป็นข้อมูลอ้างอิง ให้คัดลอกโดยไม่หาผลลัพธ์',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def build(pool):
    pairs = list(islice(pool, 16385))
    if len(pairs) > 16384 or any(not isinstance(p, (tuple, list)) or len(p) != 2
            or any(type(x) is not int or abs(x) > 99 for x in p) for p in pairs):
        raise ValueError('bounded integer pair pool')
    strata = defaultdict(list)
    for a, b in sorted(set(map(tuple, pairs))):
        strata[(len(str(abs(a))), len(str(abs(b))), a < 0, b < 0)].append((a, b))
    expected = set(product((1, 2), (1, 2), (False, True), (False, True)))
    if set(strata) != expected:
        raise ValueError('incomplete length/sign strata')
    rng = random.Random(SEED)
    rows = []
    covered = {'train': set(), 'development': set()}
    tags = lambda pair: {(side, x) for side, x in enumerate(pair) if abs(x) in (3, 8)}
    for key in sorted(strata):
        critical = [p for p in strata[key] if any(abs(x) in (3, 8) for x in p)]
        ordinary = [p for p in strata[key] if p not in critical]
        rng.shuffle(critical)
        rng.shuffle(ordinary)
        if 1 in key[:2]:
            if len(critical) < 3 or len(ordinary) < 3:
                raise ValueError('insufficient critical/ordinary pairs')
            selected = {}
            for split, count, remainder in [('train', 2, ordinary[:2]), ('development', 1, ordinary[2:3])]:
                chosen = []
                for _ in range(count):
                    index = max(range(len(critical)), key=lambda i: len(tags(critical[i]) - covered[split]))
                    pair = critical.pop(index)
                    covered[split].update(tags(pair))
                    chosen.append(pair)
                selected[split] = chosen + remainder
        else:
            if len(ordinary) < 6:
                raise ValueError('insufficient two-digit pairs')
            selected = {'train': ordinary[:4], 'development': ordinary[4:6]}
        for split, selected_pairs in selected.items():
            for a, b in selected_pairs:
                for op, symbol in ((1, '+'), (2, '-'), (3, '*')):
                    expression = f'({a}) {symbol} ({b})'
                    pair_group, group = digest([a, b])[:20], digest([a, b, op])[:20]
                    families = ('seen',) if split == 'train' else ('seen', 'new')
                    for family, language, intent in product(families, ('en', 'th'), ('compute', 'quote-copy')):
                        prompt = PATTERNS[family, language, intent].format(expr=expression)
                        rows.append(dict(id=f'{group}-{family}-{language}-{intent}', group=group,
                            pair_group=pair_group, split=split, family=family, language=language, intent=intent,
                            lhs=a, rhs=b, operation=op, ready_op=op if intent == 'compute' else 0,
                            expression=expression, prompt=prompt,
                            prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest()))
    required = set(product((0, 1), (-8, -3, 3, 8)))
    if any(covered[split] != required for split in covered):
        raise ValueError('critical signed-slot coverage incomplete')
    rng.shuffle(rows)
    return dict(version=1, kind='matched-pair-current-readout-development', seed=SEED,
                rows=rows, rows_sha256=digest(rows),
                scope='signed ±99; historically inspected pool, not fresh acceptance',
                sampling='equal length/sign/op strata; half critical pairs in strata containing one-digit operands',
                heldout='pair-disjoint experiment development; all pool values have historical exposure')


def write_new(output, value):
    output = Path(output)
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    if len(data) > 4 * 1024**2 or len(value['rows']) != 1536 or digest(value['rows']) != value['rows_sha256']:
        raise ValueError('bounded output manifest')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'corpus.json').write_bytes(data)


def load_audited_pool(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 8 * 1024**2:
        raise ValueError('bounded audited pool file')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_FILE_SHA256:
        raise ValueError('unrecognized pool identity')
    value = json.loads(raw)
    if len(value['rows']) != 10624 or digest(value['rows']) != SOURCE_ROWS_SHA256:
        raise ValueError('audited pool rows identity')
    return [(r['lhs'], r['rhs']) for r in value['rows']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = build(load_audited_pool(args.source_corpus))
    value['source_pool'] = dict(file_sha256=SOURCE_FILE_SHA256, rows_sha256=SOURCE_ROWS_SHA256,
        historical_exposure=True, selection='subset of already audited pair pool; no acceptance manifest access')
    write_new(args.output, value)
    print(json.dumps(dict(rows=len(value['rows']), rows_sha256=value['rows_sha256'],
                         source_file_sha256=SOURCE_FILE_SHA256)))


if __name__ == '__main__':
    main()
