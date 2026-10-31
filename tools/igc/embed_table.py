"""Frozen embedding table for the sidecar: a small family model's input embeddings, reduced by PCA.

Only the shard that holds the embeddings is fetched. Runs on Colab; the weights do not need to exist locally.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

NAMES = ('model.embed_tokens.weight', 'model.language_model.embed_tokens.weight', 'language_model.model.embed_tokens.weight')


def shard_of(repository):
    from huggingface_hub import hf_hub_download
    try:
        index = json.loads(Path(hf_hub_download(repository, 'model.safetensors.index.json')).read_text(encoding='utf-8'))
    except Exception:
        return 'model.safetensors', None
    for name in NAMES:
        if name in index['weight_map']:
            return index['weight_map'][name], name
    raise SystemExit('no embedding tensor: ' + ', '.join(k for k in index['weight_map'] if 'embed' in k))


def reduce(weights, width):
    """Centred PCA, so no direction is spent on the mean token."""
    weights = weights.float()
    centre = weights.mean(0)
    _, _, v = torch.pca_lowrank(weights - centre, q=width, center=False, niter=4)
    return (weights - centre) @ v


def main():
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    shard, name = shard_of(args.repository)
    with safe_open(hf_hub_download(args.repository, shard), framework='pt') as f:
        name = name or next(k for k in f.keys() if k in NAMES)
        weights = f.get_tensor(name)
    table = reduce(weights, args.width).numpy().astype(np.float32)
    np.save(args.output, table)
    print(name, tuple(weights.shape), '->', table.shape)


if __name__ == '__main__':
    main()
