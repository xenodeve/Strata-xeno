"""Offline firing test: a trained sidecar on real reasoning traces, at each confidence threshold.

The number the model wrote is the answer key. Every labelled number is a chance to fire, so copied numbers
count as misfires. `tools/igc/rule_pointer.py` is the floor this has to beat.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import torch
from tools.igc.auto_label import OPERATIONS, label_text
from tools.igc.rule_pointer import interval
from tools.igc.sidecar_data import table
from tools.igc.sidecar_fit import MAX_ENTRIES, MAX_TOKENS, OPERATIONS as NAMES, Sidecar, ZERO_FROM, shape_for

THRESHOLDS = (.5, .8, .9, .95, .98, .99, .995)


def slot(prompt, thinking, row, tokenizer):
    """Tail of the context up to the slot, with the table entries still inside it."""
    text = prompt + '\n' + thinking[:row['start']]
    entries = table(text, len(text))
    encoded = tokenizer.encode(text, add_special_tokens=False)
    start = max(0, len(encoded.ids) - MAX_TOKENS)
    ids = [i if i < ZERO_FROM else 0 for i in encoded.ids[start:]]
    offsets = encoded.offsets[start:]
    kept = []
    ends = []
    for entry in entries:
        position = next((i for i, (_, stop) in enumerate(offsets) if stop >= entry['end']), None)
        if position is not None and offsets[0][0] < entry['end']:
            kept.append(entry['value'])
            ends.append(position)
    return ids, ends[-MAX_ENTRIES:], kept[-MAX_ENTRIES:]


def score(model, ids, ends, device):
    tokens = torch.tensor([ids], device=device)
    with torch.no_grad():
        outputs = model(tokens, torch.tensor([len(ids)], device=device), torch.tensor([ends], device=device))
    probabilities = [o.softmax(-1)[0] for o in outputs]
    best = [int(p.argmax()) for p in probabilities]
    return best, float(probabilities[0][best[0]] * probabilities[1][best[1]] * probabilities[2][best[2]])


def main():
    from tokenizers import Tokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('traces', 'table', 'tokenizer', 'weights'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--target', type=int, choices=(5, 30, 100), required=True)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import numpy as np
    model = Sidecar(torch.from_numpy(np.load(args.table)).float(), *shape_for(args.target)).to(device).eval()
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    points = {}
    for trace in json.loads(args.traces.read_text(encoding='utf-8')):
        for row in label_text(trace['prompt'], trace['thinking']):
            if row['label'] == 'computed' and row['level'] == 0:
                continue
            ids, ends, values = slot(trace['prompt'], trace['thinking'], row, tokenizer)
            if not ends:
                continue
            (op, left, right), confidence = score(model, ids, ends, device)
            value = OPERATIONS[NAMES[op]](values[left], values[right])
            good = value == Fraction(row['value'].replace(',', ''))
            points.setdefault(trace['suite'], []).append((confidence, good, row['label'] == 'computed', row['label'] == 'computed' and row['level'] > 0))
    for suite, rows in sorted(points.items()):
        slots = sum(r[3] for r in rows)
        for threshold in THRESHOLDS:
            fired = [r for r in rows if r[0] >= threshold]
            right = sum(r[1] for r in fired)
            caught = sum(r[1] and r[3] for r in fired)
            low, high = interval(right, len(fired)) if fired else (0., 1.)
            print(f'{suite} {args.target}M >= {threshold}: fired {len(fired)} right {right} precision 95% [{low:.3f}, {high:.3f}] recall {caught}/{slots}')


if __name__ == '__main__':
    main()
