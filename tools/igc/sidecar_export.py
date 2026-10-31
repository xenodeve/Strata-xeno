"""Sidecar weights as one flat file for the engine, and a parity fixture for its test.

Layout, little endian: magic, then int32 width, layers, heads, vocabulary, table width, common; then tensors in the
fixed order `tensors()` gives. The table is float16, everything else float32.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tools.igc.sidecar_fit import COMMON, MAX_ENTRIES, MAX_TOKENS, OPERATIONS, Sidecar, shape_for

MAGIC = b'STRSIDE1'


def tensors(state, layers):
    """Name order is the file format; the engine reads the same order."""
    names = ['project.0.weight', 'project.0.bias', 'project.1.weight', 'project.1.bias', 'learned.weight', 'position.weight']
    for i in range(layers):
        prefix = f'encoder.layers.{i}.'
        names += [prefix + n for n in ('norm1.weight', 'norm1.bias', 'self_attn.in_proj_weight', 'self_attn.in_proj_bias',
                                       'self_attn.out_proj.weight', 'self_attn.out_proj.bias', 'norm2.weight', 'norm2.bias',
                                       'linear1.weight', 'linear1.bias', 'linear2.weight', 'linear2.bias')]
    names += ['norm.weight', 'norm.bias', 'operation.weight', 'operation.bias', 'left.weight', 'left.bias', 'right.weight', 'right.bias',
              'entry.weight', 'entry.bias', 'recency.weight', 'scale']
    return [(n, state[n]) for n in names]


def export(weights, table_path, target, output):
    table = torch.from_numpy(np.load(table_path)).float()
    state = torch.load(weights, map_location='cpu')
    width, layers, heads = shape_for(target)
    with output.open('wb') as f:
        f.write(MAGIC)
        np.array([width, layers, heads, table.shape[0], table.shape[1], COMMON, MAX_TOKENS, MAX_ENTRIES], dtype=np.int32).tofile(f)
        table.numpy().astype(np.float16).tofile(f)
        state['index'].numpy().astype(np.int32).tofile(f)
        for _, tensor in tensors(state, layers):
            tensor.numpy().astype(np.float32).tofile(f)


def fixture(weights, table_path, target, output):
    """Outputs of the Python model for fixed inputs, with the table as the engine sees it."""
    table = torch.from_numpy(np.load(table_path).astype(np.float16).astype(np.float32))
    width, layers, heads = shape_for(target)
    model = Sidecar(table, width, layers, heads).eval()
    model.load_state_dict(torch.load(weights, map_location='cpu'), strict=False)
    torch.manual_seed(20261031)
    cases = []
    for length, entries in ((7, 2), (64, 5), (300, 12), (512, 64)):
        tokens = torch.randint(0, table.shape[0], (1, length))
        ends = torch.sort(torch.randperm(length)[:entries]).values[None]
        with torch.no_grad():
            operation, left, right = model(tokens, torch.tensor([length]), ends)
        cases.append(dict(tokens=tokens[0].tolist(), ends=ends[0].tolist(), operation=operation[0].tolist(), left=left[0].tolist(), right=right[0].tolist()))
    output.write_text(json.dumps(dict(operations=OPERATIONS, cases=cases)), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('weights', 'table', 'output', 'fixture'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--target', type=int, choices=(5, 30, 100), default=5)
    args = parser.parse_args()
    export(args.weights, args.table, args.target, args.output)
    fixture(args.weights, args.table, args.target, args.fixture)
