"""Validate bounded first-window mapper observations, never executable artifacts.

These are observations before commit. The collector separately requires a
completed owned request and a frozen model/tokenizer/build/config manifest.
This reader alone establishes neither execution success nor model provenance.
"""
import hashlib
import json
from pathlib import Path
import stat
import numpy as np


def _read(path, size=None, limit=1 << 20):
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('observation must be a regular non-reparse file')
        if info.st_size > limit or (size is not None and info.st_size != size):
            raise ValueError('observation byte count')
        data = path.read_bytes()
        if len(data) != info.st_size:
            raise ValueError('observation changed while reading')
        return data
    except OSError as error:
        raise ValueError('observation file unavailable') from error


def load_observation(root, index):
    if type(index) is not int or not 0 <= index < 1024:
        raise ValueError('observation index')
    root = Path(root)
    prefix = f'mixed.{index}'
    meta_raw = _read(root / (prefix + '.meta.json'), limit=4096)
    meta = json.loads(meta_raw)
    if not isinstance(meta, dict):
        raise ValueError('observation metadata object')
    required = {'version':1, 'row':0, 'n_embd':2560, 'hc':4, 'n_vocab':248320}
    if any(type(meta.get(k)) is not int or meta[k] != value for k, value in required.items()):
        raise ValueError('observation format or geometry')
    if meta.get('kind') != 'first_window_observation' or meta.get('stage') != 'head_mixed_after_final_hc':
        raise ValueError('observation stage')
    for key in ('position', 'token', 'prompt_tokens'):
        if type(meta.get(key)) is not int:
            raise ValueError('observation boundary type')
    count = meta['prompt_tokens']
    if not 1 <= count <= 262144 or meta['position'] != count-1 or not 0 <= meta['token'] < 248320:
        raise ValueError('observation boundary')
    tokens_raw = _read(root / (prefix + '.tokens.bin'), count*4)
    tokens = np.frombuffer(tokens_raw, dtype='<i4')
    if np.any(tokens < 0) or np.any(tokens >= 248320) or tokens[-1] != meta['token']:
        raise ValueError('observation token prefix')
    mixed_raw = _read(root / prefix, 2560*4)
    logits_raw = _read(root / f'logits.{index}', 248320*4)
    mixed = np.frombuffer(mixed_raw, dtype='<f4')
    logits = np.frombuffer(logits_raw, dtype='<f4')
    if not np.isfinite(mixed).all() or not np.isfinite(logits).all() or not np.any(mixed) or not np.any(logits):
        raise ValueError('observation nonfinite or stale zero tensor')
    return {**meta, 'mixed':mixed.copy(), 'logits':logits.copy(), 'prefix_tokens':tokens.copy(),
            'prefix_sha256':hashlib.sha256(tokens_raw).hexdigest(),
            'mixed_sha256':hashlib.sha256(mixed_raw).hexdigest(),
            'logits_sha256':hashlib.sha256(logits_raw).hexdigest(),
            'metadata_sha256':hashlib.sha256(meta_raw).hexdigest()}
