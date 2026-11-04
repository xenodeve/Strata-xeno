"""seal_vector.py MODEL TRACES... --out OUT.npz [--layers 16,20,24,28] [--rows N]: the SEAL steering vector for this model from its own stored thinking traces (tir4b --ids files).

For every reply: one teacher-forced pass; at every `\\n\\n` token of the reply the stream after block L is kept, labelled by the keyword type of the thought that ends there
(tools/igc/seal.py). S_L = mean(execution) - mean(reflection + transition). Layer choice: on the last 30 % of the traces (held out from the means), AUROC of the projection onto
S_L for execution against the rest, and the mean projection gap; the best layers are printed and every layer asked for is saved, so the layer is chosen from the held-out table.
The traces must be of other questions than the ones the steering is tested on."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path.cwd()))
from tools.igc.seal import find_modules, thoughts


def auroc(pos, neg):
    """Probability a random positive projects above a random negative (ties half), by ranks."""
    x = np.concatenate([pos, neg])
    r = x.argsort().argsort() + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('traces', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--layers', default='16,20,24,28')
    ap.add_argument('--rows', type=int, default=0)
    ap.add_argument('--max-len', type=int, default=16000)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(',')]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map='cuda:0').eval()
    n_layers = model.config.get_text_config().num_hidden_layers
    _, blocks = find_modules(model, n_layers)
    delim = tok.encode('\n\n', add_special_tokens=False)
    assert len(delim) == 1, delim
    delim = delim[0]
    lm = model.model.language_model if hasattr(model.model, 'language_model') else model.model
    rows = []
    for p in args.traces:
        rows += [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]
    rows = rows[:args.rows] if args.rows else rows
    data = []   # (row index, layer -> [vectors], kinds)
    for k, row in enumerate(rows):
        out = row['out_ids'][:args.max_len - len(row['prompt_ids'])]
        th = thoughts(out, tok, delim)
        if not th:
            continue
        pos = torch.tensor([len(row['prompt_ids']) + e for _, e, _ in th], device='cuda:0')
        got, hooks = {}, []
        for L in layers:
            def hook(m_, i_, o_, L=L):
                h = o_[0] if isinstance(o_, tuple) else o_
                got[L] = h[0, pos].float().cpu().numpy()
            hooks.append(blocks[L - 1].register_forward_hook(hook))
        try:
            with torch.no_grad():
                lm(input_ids=torch.tensor([row['prompt_ids'] + out], device='cuda:0'), use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        data.append((k, got, [t for _, _, t in th]))
        print(f'  trace {k + 1}/{len(rows)} {row["id"]}: {len(th)} thoughts {dict(zip(*np.unique([t for _, _, t in th], return_counts=True)))}', flush=True)
    cut = int(0.7 * len(data))
    fit, held = data[:cut], data[cut:]
    save = {}
    print(f'\nfit on {len(fit)} traces, held out {len(held)}')
    print(f'{"layer":>5s} {"E fit":>6s} {"RT fit":>7s} {"AUROC held":>11s} {"gap held":>9s} {"|S|":>8s}')
    for L in layers:
        def stack(part, want):
            xs = [g[L][[i for i, t in enumerate(kinds) if want(t)]] for _, g, kinds in part]
            xs = [x for x in xs if len(x)]
            return np.concatenate(xs) if xs else np.zeros((0, 1))
        e, rt = stack(fit, lambda t: t == 'E'), stack(fit, lambda t: t != 'E')
        if len(e) < 5 or len(rt) < 5:
            print(f'{L:5d} too few thoughts: E {len(e)}, R/T {len(rt)}')
            continue
        S = e.mean(0) - rt.mean(0)
        he, hr = stack(held, lambda t: t == 'E') @ S, stack(held, lambda t: t != 'E') @ S
        a = auroc(he, hr)
        best = max(best, (a, L), key=lambda x: x[0])
        print(f'{L:5d} {len(e):6d} {len(rt):7d} {a:11.3f} {he.mean() - hr.mean():9.2f} {np.linalg.norm(S):8.2f}')
        # the saved vector uses every trace, held out ones too: the table above is what was held out to choose the layer
        ea, rta = stack(data, lambda t: t == 'E'), stack(data, lambda t: t != 'E')
        save[f'L{L}'] = (ea.mean(0) - rta.mean(0)).astype(np.float32)
    np.savez(args.out, delim=delim, best_layer=best[1] or 0, **save)
    print(f'saved {sorted(save)} to {args.out}; best held-out layer {best[1]} (AUROC {best[0]:.3f})')


if __name__ == '__main__':
    main()
