"""serve/gguf_info.py - what model is running and how it is quantized, read from the GGUF headers it was loaded from.

Only the header is read (key-values, then tensor infos), never the weights. A tensor's size is the distance to the
next tensor's offset, so the bytes per role are the file's own, not a table's; bits per weight is those bytes over the
weights the dims say. Nothing here is a guess: a file that cannot be read gives None and the app says "not measured".
"""
from __future__ import annotations

import os
import re
import struct
from collections import OrderedDict

# ggml type ids. 42 is the Q2_0 this project's engine reads (the CPU rows' gu_type 42).
TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
         29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 42: "Q2_0"}
_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}     # value type -> bytes
ROLE_ORDER = ["experts", "shared experts", "attention", "embeddings", "output", "ple table", "other"]
_KEEP = ("general.", )


def _u32(f): return struct.unpack("<I", f.read(4))[0]
def _u64(f): return struct.unpack("<Q", f.read(8))[0]


def _str(f) -> str:
    n = _u64(f)
    if n > 1 << 24:
        raise ValueError("implausible string length")
    return f.read(n).decode("utf-8", "replace")


def _skip_value(f, t: int):
    if t in _FIXED:
        f.seek(_FIXED[t], 1)
    elif t == 8:
        f.seek(_u64(f), 1)
    elif t == 9:
        et, n = _u32(f), _u64(f)
        if et in _FIXED:
            f.seek(n * _FIXED[et], 1)
        else:
            for _ in range(n):
                _skip_value(f, et)
    else:
        raise ValueError(f"unknown value type {t}")


def _value(f, t: int):
    if t == 8:
        return _str(f)
    if t in _FIXED:
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
        return struct.unpack(fmt, f.read(_FIXED[t]))[0]
    _skip_value(f, t)
    return None


def read_header(path: str) -> dict:
    """{"meta": {general.* scalars and strings}, "tensors": [{name, dims, type, offset}], "data_start", "size"}"""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = _u32(f)
        if version < 2:
            raise ValueError("GGUF version 1 is not read")
        n_tensors, n_kv = _u64(f), _u64(f)
        if n_tensors > 1 << 20 or n_kv > 1 << 20:
            raise ValueError("implausible counts")
        meta = {}
        for _ in range(n_kv):
            key, t = _str(f), _u32(f)
            v = _value(f, t) if key.startswith(_KEEP) else _skip_value(f, t)
            if v is not None:
                meta[key] = v
        tensors = []
        for _ in range(n_tensors):
            name, nd = _str(f), _u32(f)
            dims = [_u64(f) for _ in range(nd)]
            typ, off = _u32(f), _u64(f)
            tensors.append({"name": name, "dims": dims, "type": typ, "offset": off})
        align = int(meta.get("general.alignment", 32)) or 32
        pos = f.tell()
        return {"meta": meta, "tensors": tensors, "data_start": pos + (-pos) % align, "size": os.path.getsize(path)}


def role_of(name: str, shard_index: int) -> str:
    n = name.lower()
    if "shexp" in n:
        return "shared experts"
    if "_exps" in n or "ffn_gate_up" in n and "exps" in n:
        return "experts"
    if "per_layer" in n or "ple" in n.split(".")[0].split("_") or "ngram" in n:
        return "ple table"
    if "token_embd" in n:
        return "ple table" if shard_index > 0 else "embeddings"
    if n.startswith("output") or ".output" in n:
        return "output"
    if any(k in n for k in ("attn", "ssm", "linear_attn", "gdn", "conv1d")):
        return "attention"
    return "other"


def model_info(paths: list[str]) -> dict | None:
    """The model behind these GGUF files (shard 1 first), or None when there is nothing to read."""
    try:
        if not paths:
            return None
        heads = [read_header(p) for p in paths]
    except (OSError, ValueError, struct.error):
        return None
    meta = heads[0]["meta"]
    acc: dict[str, dict] = OrderedDict()
    total_bytes = total_weights = 0
    for si, h in enumerate(heads):
        ts = sorted(h["tensors"], key=lambda t: t["offset"])
        for i, t in enumerate(ts):
            end = ts[i + 1]["offset"] if i + 1 < len(ts) else h["size"] - h["data_start"]
            nbytes = max(0, end - t["offset"])
            weights = 1
            for d in t["dims"]:
                weights *= d
            r = acc.setdefault(role_of(t["name"], si), {"tensors": 0, "bytes": 0, "weights": 0, "types": {}})
            r["tensors"] += 1
            r["bytes"] += nbytes
            r["weights"] += weights
            tn = TYPES.get(t["type"], f"type {t['type']}")
            r["types"][tn] = r["types"].get(tn, 0) + nbytes
            total_bytes += nbytes
            total_weights += weights
    roles = []
    for role in ROLE_ORDER:
        r = acc.get(role)
        if r:
            roles.append({"role": role, "tensors": r["tensors"], "bytes": r["bytes"],
                          "types": [k for k, _ in sorted(r["types"].items(), key=lambda kv: -kv[1])],
                          "bpw": round(r["bytes"] * 8 / r["weights"], 3) if r["weights"] else None})
    m = re.search(r"-((?:IQ|Q|TQ)\d\w*|BF16|F16)-\d{5}-of-\d{5}\.gguf$", os.path.basename(paths[0]), re.I)
    source = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(paths[0])))) if m else None
    return {"variant": m.group(1) if m else None, "source": source or None, "name": meta.get("general.name"), "basename": meta.get("general.basename"), "finetune": meta.get("general.finetune"),
            "size_label": meta.get("general.size_label"), "architecture": meta.get("general.architecture"),
            "files": [os.path.basename(p) for p in paths], "bytes": total_bytes,
            "bpw": round(total_bytes * 8 / total_weights, 3) if total_weights else None, "roles": roles}


def files_from_args(args: list[str]) -> list[str]:
    """The GGUF files an engine command line loads: --native (shard 1) and --ple-gguf (the PLE table's shard)."""
    out = []
    for key in ("--native", "--ple-gguf"):
        for i, a in enumerate(args[:-1]):
            if a == key and str(args[i + 1]).lower().endswith(".gguf"):
                out.append(args[i + 1])
    return out
