"""Sidecar size sweep: operation and table pointers at a slot. Runs on Colab; never predicts a value.

Input is `sidecar_data.py` output plus a frozen embedding table (`embed_table.py`). Examples are split by
question, so a problem never sits on both sides. Reported: exact structure rate on held questions.
"""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn

OPERATIONS = ('add', 'sub', 'mul', 'div')
ZERO_FROM = 248070   # audio tokens of the larger family members have no row in the frozen table
MAX_TOKENS = 512
MAX_ENTRIES = 64


class Sidecar(nn.Module):
    """Causal encoder over frozen embeddings; each table entry is its last token's state."""

    def __init__(self, table, width, layers, heads):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(table, freeze=True)
        self.project = nn.Linear(table.shape[1], width)
        self.position = nn.Embedding(MAX_TOKENS, width)
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.operation = nn.Linear(width, len(OPERATIONS))
        self.left = nn.Linear(width, width)
        self.right = nn.Linear(width, width)
        self.entry = nn.Linear(width, width)

    def forward(self, tokens, lengths, entry_ends):
        steps = tokens.shape[1]
        x = self.project(self.embedding(tokens)) + self.position(torch.arange(steps, device=tokens.device))
        padding = torch.arange(steps, device=tokens.device)[None] >= lengths[:, None]
        causal = nn.Transformer.generate_square_subsequent_mask(steps, device=tokens.device, dtype=torch.bool)
        h = self.encoder(x, mask=causal, src_key_padding_mask=padding)
        query = h[torch.arange(len(h)), lengths - 1]
        valid = entry_ends >= 0
        entries = self.entry(h[torch.arange(len(h))[:, None], entry_ends.clamp(min=0)])
        score = lambda head: (entries @ head(query)[:, :, None]).squeeze(-1).masked_fill(~valid, -1e9)
        return self.operation(query), score(self.left), score(self.right)


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def shape_for(target):
    """Width and depth near a non-embedding parameter target."""
    return {5: (256, 6, 4), 30: (544, 8, 8), 100: (896, 10, 8)}[target]


def encode(rows, tokenizer):
    """Tokens up to the slot, and the last token of each table entry inside them."""
    out = []
    for row in rows:
        encoded = tokenizer.encode(row['text'][:row['slot']], add_special_tokens=False)
        ids = [i if i < ZERO_FROM else 0 for i in encoded.ids]
        offsets = encoded.offsets
        ends = []
        for _, end in row['table']:
            position = next((i for i, (_, stop) in enumerate(offsets) if stop >= end), None)
            ends.append(position if position is not None else -1)
        if not ids or len(ids) > MAX_TOKENS or len(ends) > MAX_ENTRIES or -1 in (ends[row['left']], ends[row['right']]):
            continue
        out.append(dict(ids=ids, ends=ends, op=OPERATIONS.index(row['op']), left=row['left'], right=row['right'], key=row['text']))
    return out


def batch(rows, device):
    steps = max(len(r['ids']) for r in rows)
    width = max(len(r['ends']) for r in rows)
    tokens = torch.zeros(len(rows), steps, dtype=torch.long)
    ends = torch.full((len(rows), width), -1, dtype=torch.long)
    for i, r in enumerate(rows):
        tokens[i, :len(r['ids'])] = torch.tensor(r['ids'])
        ends[i, :len(r['ends'])] = torch.tensor(r['ends'])
    labels = [torch.tensor([r[k] for r in rows]).to(device) for k in ('op', 'left', 'right')]
    return tokens.to(device), torch.tensor([len(r['ids']) for r in rows]).to(device), ends.to(device), labels


def evaluate(model, rows, device):
    model.eval()
    right = 0
    with torch.no_grad():
        for s in range(0, len(rows), 64):
            tokens, lengths, ends, labels = batch(rows[s:s + 64], device)
            guesses = [o.argmax(-1) for o in model(tokens, lengths, ends)]
            right += int(((guesses[0] == labels[0]) & (guesses[1] == labels[1]) & (guesses[2] == labels[2])).sum())
    return right, len(rows)


def fit(data, table_path, tokenizer_path, output, target, epochs, seed):
    from tokenizers import Tokenizer
    random.seed(seed); torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = encode(json.loads(data.read_text(encoding='utf-8')), Tokenizer.from_file(str(tokenizer_path)))
    keys = sorted({r['key'] for r in rows})
    random.shuffle(keys)
    held = set(keys[:len(keys) // 10])
    train = [r for r in rows if r['key'] not in held]
    test = [r for r in rows if r['key'] in held]
    table = torch.from_numpy(np.load(table_path)).float()
    width, layers, heads = shape_for(target)
    model = Sidecar(table, width, layers, heads).to(device)
    print('non-embedding parameters', trainable(model), 'train', len(train), 'held', len(test), flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=.01)
    loss_of = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        random.shuffle(train)
        for s in range(0, len(train), 32):
            tokens, lengths, ends, labels = batch(train[s:s + 32], device)
            optimizer.zero_grad(set_to_none=True)
            loss = sum(loss_of(o, l) for o, l in zip(model(tokens, lengths, ends), labels))
            loss.backward()
            optimizer.step()
        print('epoch', epoch, round(float(loss.detach()), 4), 'held exact', evaluate(model, test, device), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith('embedding.')}
    torch.save(state, output / f'sidecar-{target}m.pt')
    (output / f'sidecar-{target}m.json').write_text(json.dumps(dict(target=target, parameters=trainable(model), held=evaluate(model, test, device), seed=seed)), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'table', 'tokenizer', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--target', type=int, choices=(5, 30, 100), required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261031)
    args = parser.parse_args()
    fit(args.data, args.table, args.tokenizer, args.output, args.target, args.epochs, args.seed)
