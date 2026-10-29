"""Validate bounded first-window mapper observations, never executable artifacts.

These are observations before commit. The collector separately requires a
completed owned request and a frozen model/tokenizer/build/config manifest.
This reader alone establishes neither execution success nor model provenance.
"""
import hashlib
import json
from pathlib import Path
import stat
import struct
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


def load_preanchor(root,index,stage='final'):
    """Only observed F32 rows; absence in a valid prefix cache is explicit."""
    anchor=load_observation(root,index)
    contracts={'final':(b'IGCPREA1','preanchor','head_mixed_after_final_hc'),
               'embedding':(b'IGCEMB01','embedding','native_embedding_broadcast'),
               'attention':(b'IGCATN01','attention','native_first_attention_output')}
    if stage not in contracts:
        raise ValueError('unknown activation stage')
    expected_magic,suffix,stage_name=contracts[stage]
    raw=_read(Path(root)/f'mixed.{index}.{suffix}.bin',limit=4*1024**2)
    header=struct.Struct('<8sIIIIIQQ')
    if len(raw)<header.size:
        raise ValueError('preanchor header size')
    magic,version,hc,dim,cap,n,position,count=header.unpack_from(raw)
    count_key='preanchor_rows' if stage=='final' else f'preanchor_{suffix}_rows'
    if (magic!=expected_magic or (version,hc,dim,cap)!=(1,4,2560,80)
            or position!=anchor['position'] or count!=anchor['prompt_tokens']
            or n>min(position,cap) or anchor.get('preanchor_format')!=1
            or anchor.get(count_key)!=n or anchor.get('preanchor_capacity')!=cap
            or (stage!='final' and anchor.get(f'preanchor_{suffix}_stage')!=stage_name)):
        raise ValueError('preanchor geometry or boundary')
    row_bytes=16+hc*dim*4
    if len(raw)!=header.size+n*row_bytes:
        raise ValueError('preanchor record bytes')
    positions=[];sources=[];values=[]
    first=max(0,position-cap)
    for i in range(n):
        offset=header.size+i*row_bytes
        p,token,source=struct.unpack_from('<qII',raw,offset)
        if (not first<=p<position or source not in (1,2) or token!=anchor['prefix_tokens'][p]
                or (positions and p<=positions[-1])):
            raise ValueError('preanchor row identity or order')
        row=np.frombuffer(raw,dtype='<f4',count=hc*dim,offset=offset+16).reshape(hc,dim)
        if not np.isfinite(row).all():
            raise ValueError('preanchor nonfinite')
        positions.append(p);sources.append(source);values.append(row.copy())
    return dict(anchor=anchor,activation_stage=stage,positions=positions,sources=sources,
                residuals=np.stack(values) if values else np.empty((0,hc,dim),dtype=np.float32),
                missing_positions=[p for p in range(first,position) if p not in positions],
                preanchor_sha256=hashlib.sha256(raw).hexdigest())
