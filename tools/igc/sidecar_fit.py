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

OPERATIONS = ('add', 'sub', 'mul', 'div', 'none')
NONE = OPERATIONS.index('none')
ZERO_FROM = 248070   # audio tokens of the larger family members have no row in the frozen table
MAX_TOKENS = 512
MAX_ENTRIES = 64
COMMUTATIVE = ('add', 'mul')


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
        self.recency = nn.Embedding(MAX_ENTRIES, width)
        self.norm = nn.LayerNorm(width)
        self.scale = nn.Parameter(torch.tensor(10.))

    def forward(self, tokens, lengths, entry_ends):
        steps = tokens.shape[1]
        x = self.project(self.embedding(tokens)) + self.position(torch.arange(steps, device=tokens.device))
        padding = torch.arange(steps, device=tokens.device)[None] >= lengths[:, None]
        causal = nn.Transformer.generate_square_subsequent_mask(steps, device=tokens.device, dtype=torch.bool)
        h = self.norm(self.encoder(x, mask=causal, src_key_padding_mask=padding))
        query = h[torch.arange(len(h)), lengths - 1]
        valid = entry_ends >= 0
        rank = (valid.sum(1, keepdim=True) - 1 - torch.arange(valid.shape[1], device=tokens.device)[None]).clamp(min=0, max=MAX_ENTRIES - 1)
        entries = nn.functional.normalize(self.entry(h[torch.arange(len(h))[:, None], entry_ends.clamp(min=0)]) + self.recency(rank), dim=-1)
        score = lambda head: self.scale * (entries @ nn.functional.normalize(head(query), dim=-1)[:, :, None]).squeeze(-1).masked_fill(~valid, -1e4)
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
        start = max(0, len(encoded.ids) - MAX_TOKENS)
        ids = [i if i < ZERO_FROM else 0 for i in encoded.ids[start:]]
        offsets = encoded.offsets[start:]
        if not ids:
            continue
        ends = []
        for _, end in row['table']:
            position = next((i for i, (_, stop) in enumerate(offsets) if stop >= end), None)
            ends.append(position if position is not None and offsets[0][0] < end else -1)
        drop = max(0, len(ends) - MAX_ENTRIES)
        ends = ends[drop:]
        left, right = row['left'] - drop, row['right'] - drop
        if row['op'] == 'none':
            left = right = -100
            if max(ends, default=-1) < 0:
                continue
        elif min(left, right) < 0 or -1 in (ends[left], ends[right]):
            continue
        elif row['op'] in COMMUTATIVE:
            left, right = min(left, right), max(left, right)
        out.append(dict(ids=ids, ends=ends, op=OPERATIONS.index(row['op']), left=left, right=right, key=row['group']))
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


def buckets(rows, size):
    """Batches of near-equal length in random order, so padding does not cost a long row's time for every short one."""
    rows = random.sample(rows, len(rows))
    chunks = [sorted(rows[s:s + size * 50], key=lambda r: len(r['ids'])) for s in range(0, len(rows), size * 50)]
    parts = [chunk[s:s + size] for chunk in chunks for s in range(0, len(chunk), size)]
    random.shuffle(parts)
    return parts


def evaluate(model, rows, device):
    model.eval()
    right = 0
    heads = [0, 0, 0]
    with torch.no_grad():
        for s in range(0, len(rows), 64):
            tokens, lengths, ends, labels = batch(rows[s:s + 64], device)
            guesses = [o.argmax(-1) for o in model(tokens, lengths, ends)]
            heads = [h + int((g == l).sum()) for h, g, l in zip(heads, guesses, labels)]
            pointers = ((guesses[1] == labels[1]) & (guesses[2] == labels[2])) | (labels[0] == NONE)
            right += int(((guesses[0] == labels[0]) & pointers).sum())
    return right, len(rows), heads


def fit(data, table_path, tokenizer_path, output, target, epochs, seed, size=32):
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
    steps = max(1, epochs * (len(train) + size - 1) // size)
    scaler = torch.amp.GradScaler(device, enabled=device == 'cuda')
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, 3e-4, total_steps=steps + 1, pct_start=.05)
    loss_of = nn.CrossEntropyLoss(ignore_index=-100)
    for epoch in range(epochs):
        model.train()
        for s, part in enumerate(buckets(train, size)):
            tokens, lengths, ends, labels = batch(part, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=device == 'cuda'):
                outputs = model(tokens, lengths, ends)
                loss = sum(loss_of(o, l) for o, l in zip(outputs, labels) if (l != -100).any())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
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
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261031)
    args = parser.parse_args()
    fit(args.data, args.table, args.tokenizer, args.output, args.target, args.epochs, args.seed, args.batch)
