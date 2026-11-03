"""tools/qwen35_to_qwen4exp.py - Qwen3.5-4B (Hugging Face safetensors) into the files the engine loads for qwen4exp (#297).

    python tools/qwen35_to_qwen4exp.py --hf D:\\models\\Qwen3.5-4B --gguf <out>.gguf --pack <pack dir>
    python tools/qwen35_to_qwen4exp.py --hf-repo Qwen/Qwen3.5-4B --gguf <out>.gguf --pack <pack dir>   (no download)
    python tools/qwen35_to_qwen4exp.py --hf D:\\models\\Qwen3.5-4B --plan                              (tensor table only)

The plan is docs/plans/2026-11-03-qwen35-4b-port.md, stage 1.  The 4B has a plain residual, a dense FFN and dense
gated attention; the engine runs one architecture, qwen4exp, with 4 hyper-connection streams, MoE plus a shared
expert, and QSA attention.  This writes the 4B in qwen4exp's shape, with weights chosen so qwen4exp's formulas
compute the 4B exactly:

  hyper-connections   hc_*_down, hc_*_up, hc_*_inject (and output_hc_down/up) are all zero.  The read's gate is
                      then sigmoid(0) = 0.5 on every stream and the write's weight 2*sigmoid(0/hc) = 1, a plain
                      residual add.  The embedding is broadcast to all 4 streams and every write adds the same
                      output, so the streams stay equal.  The read's per-stream norm gamma is 2*(1+w): HF's block
                      norms multiply by (1+w) (zero-centred), gr_read multiplies by the stored gamma, and the 0.5
                      gate halves it.  The x2 is exact.
  dense FFN           the shared expert: ffn_{gate,up,down}_shexp, with ffn_gate_inp_shexp = 0 (scalar gate
                      sigmoid(0) = 0.5) and ffn_down_shexp's Q8_0 block scales doubled, which is exact.  No routed
                      experts and no router (expert_count 0).
  GDN                 in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj / conv1d, with the value heads
                      REORDERED: HF pairs value head h with key head h // (h_v/h_k) (repeat_interleave), the engine
                      with key head h % h_k (gdn.hpp).  Engine head e holds HF head (e % h_k)*r + e // h_k.
                      ssm_a = -exp(A_log).  ssm_norm is the gated norm's weight AS STORED: HF's
                      Qwen3_5RMSNormGated multiplies by w (not 1+w), and so does gdn_out_norm.  HF's output gate is
                      silu(z); today's engine applies sigmoid(z) (fused_gdn.cu, gdn.cu) - stage 2 adds the switch.
  attention           q_proj ([q 256 | gate 256] per head, exactly the engine's q_full), k_proj, v_proj, o_proj;
                      q_norm / k_norm stored as 1+w (rms_norm_weighted multiplies by the stored value).
  head                output.weight is a copy of the tied embedding; output_hc_norm = 2*(1+norm.w) x4.
  left out            PLE (the engine runs with --no-ple), the indexer, MTP (mtp.*) and the vision tower.

Types: every projection, token_embd and output in Q8_0 (the engine's native GEMV has no BF16/F16 matrix path);
the small readers the engine takes as BF16 (hc_*, output_hc_down/up, ssm_alpha/beta, ffn_gate_inp_shexp) as BF16,
exactly as the GSQ-RCO qwen4exp files store them; norms, ssm_a, ssm_dt.bias and ssm_conv1d as F32.

The architecture name defaults to `qwen35dense`, a NEW name: today's engine refuses it (check_architecture wants
'qwen4exp'), which is right until stage 2 teaches it this geometry.  --arch qwen4exp writes the same tensors under
the old name, for experiments only.  `--plan` prints the table and what the current loader would still refuse.

The pack (--pack) is tools/iq_pack.py's standalone form: index.txt (Q8_0 rows served natively from the GGUF),
dense.bin (the BF16/F32 tensors), conversions.json and tokenizer/.  There is no native_experts.txt: the model has no
routed experts, and that file is iq_pack's completion marker, so today's engine and setup see the pack as
unfinished - stage 2's list.

The numeric gate is tools/test_qwen35_convert.py (tiny random weights in its unit tests; --full on the real model).
Reading the real model through --hf-repo uses plain HTTP range requests with no token, so nothing of the 9.3 GB
download has to land on disk.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import pathlib
import struct
import sys
import time
import urllib.request

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gguf_writer import GGUFWriter  # noqa: E402

GGML_ID = {"F32": 0, "Q8_0": 8, "BF16": 30}
FILE_TYPE_MOSTLY_Q8_0 = 7
HF_PREFIX = "model.language_model."
TOOL = "tools/qwen35_to_qwen4exp.py"


# ------------------------------------------------------------------ the 4B's geometry, from config.json
@dataclasses.dataclass
class Geometry:
    n_embd: int
    n_layers: int
    layer_types: list
    n_head: int
    n_head_kv: int
    head_dim: int
    n_rot: int
    rope_base: float
    rope_sections: list
    ssm_k_heads: int
    ssm_v_heads: int
    ssm_k_dim: int
    ssm_v_dim: int
    d_conv: int
    n_ff: int
    vocab: int
    eps: float
    context: int
    full_attention_interval: int
    hc: int = 4
    hc_lr: int = 320

    @property
    def hc_dim(self) -> int:
        return self.hc * self.n_embd

    @property
    def conv_channels(self) -> int:
        return 2 * self.ssm_k_heads * self.ssm_k_dim + self.ssm_v_heads * self.ssm_v_dim

    @property
    def value_dim(self) -> int:
        return self.ssm_v_heads * self.ssm_v_dim

    def is_attention(self, layer: int) -> bool:
        return self.layer_types[layer] == "full_attention"

    @classmethod
    def from_config(cls, cfg: dict, hc: int = 4, hc_lr: int = 320) -> "Geometry":
        t = cfg.get("text_config", cfg)
        rope = t.get("rope_parameters") or {}
        head_dim = int(t["head_dim"])
        n_rot = int(head_dim * float(rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0))))
        types = list(t["layer_types"])
        if t.get("hidden_act", "silu") != "silu":
            raise ValueError("hidden_act %r: the shared expert computes silu(gate)*up" % t.get("hidden_act"))
        if t.get("attention_bias"):
            raise ValueError("attention_bias: the engine's projections have no bias")
        if not t.get("attn_output_gate", True):
            raise ValueError("attn_output_gate false: the engine's attention always gates its output")
        g = cls(n_embd=int(t["hidden_size"]), n_layers=int(t["num_hidden_layers"]), layer_types=types,
                n_head=int(t["num_attention_heads"]), n_head_kv=int(t["num_key_value_heads"]), head_dim=head_dim,
                n_rot=n_rot, rope_base=float(rope.get("rope_theta", t.get("rope_theta", 1e7))),
                rope_sections=list(rope.get("mrope_section", [11, 11, 10])),
                ssm_k_heads=int(t["linear_num_key_heads"]), ssm_v_heads=int(t["linear_num_value_heads"]),
                ssm_k_dim=int(t["linear_key_head_dim"]), ssm_v_dim=int(t["linear_value_head_dim"]),
                d_conv=int(t["linear_conv_kernel_dim"]), n_ff=int(t["intermediate_size"]),
                vocab=int(t["vocab_size"]), eps=float(t["rms_norm_eps"]),
                context=int(t.get("max_position_embeddings", 262144)),
                full_attention_interval=int(t.get("full_attention_interval", 4)), hc=hc, hc_lr=hc_lr)
        if len(types) != g.n_layers:
            raise ValueError("layer_types has %d entries for %d layers" % (len(types), g.n_layers))
        if g.ssm_k_dim != g.ssm_v_dim:
            raise ValueError("the engine's GDN has one head dim (S) for keys and values")
        if g.ssm_v_heads % g.ssm_k_heads:
            raise ValueError("value heads must be a multiple of key heads")
        if g.n_head % g.n_head_kv:
            raise ValueError("query heads must be a multiple of key/value heads")
        return g


def v_head_order(g: Geometry) -> np.ndarray:
    """order[e] = the HF value head the engine's head e holds.

    HF (repeat_interleave): value head h reads key head h // r, r = h_v / h_k.  The engine (gdn.hpp, `idx[h] = h %
    h_k`): head e reads key head e % h_k.  So engine head e must hold an HF head h with h // r == e % h_k; taking
    h % r = e // h_k makes the map a bijection: h = (e % h_k) * r + e // h_k.
    """
    r = g.ssm_v_heads // g.ssm_k_heads
    e = np.arange(g.ssm_v_heads)
    return (e % g.ssm_k_heads) * r + e // g.ssm_k_heads


# ------------------------------------------------------------------ reading the safetensors (local or HTTP range)
_ST_DTYPE = {"BF16": 2, "F32": 4, "F16": 2}


def _values(dtype: str, raw: bytes, shape) -> np.ndarray:
    """A safetensors tensor as float32 - exact for all three float types."""
    if dtype == "BF16":
        a = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    elif dtype == "F32":
        a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif dtype == "F16":
        a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    else:
        raise ValueError("unsupported safetensors dtype %s" % dtype)
    return a.reshape(shape)


class SafetensorsSource:
    """The model's tensors by name, as float32.  `opener(file)` returns a callable read(offset, n) -> bytes."""

    def __init__(self, files: list, opener):
        self._read = {}
        self.where = {}
        for f in files:
            rd = opener(f)
            n = struct.unpack("<Q", rd(0, 8))[0]
            head = json.loads(rd(8, n).decode("utf-8"))
            base = 8 + n
            for name, info in head.items():
                if name == "__metadata__":
                    continue
                b, e = info["data_offsets"]
                self.where[name] = (f, info["dtype"], list(info["shape"]), base + b, e - b)
            self._read[f] = rd

    def has(self, name: str) -> bool:
        return name in self.where

    def dtype(self, name: str) -> str:
        return self.where[name][1]

    def get(self, name: str) -> np.ndarray:
        f, dtype, shape, off, n = self.where[name]
        raw = self._read[f](off, n)
        if len(raw) != n:
            raise IOError("%s: read %d of %d bytes" % (name, len(raw), n))
        return _values(dtype, raw, shape)


def _index_files(index: dict, single: str) -> list:
    if index is None:
        return [single]
    return sorted(set(index["weight_map"].values()))


def local_source(model_dir) -> SafetensorsSource:
    d = pathlib.Path(model_dir)
    idx = d / "model.safetensors.index.json"
    index = json.loads(idx.read_text(encoding="utf-8")) if idx.exists() else None
    files = _index_files(index, "model.safetensors")

    def opener(f):
        fh = open(d / f, "rb")

        def rd(off, n):
            fh.seek(off)
            return fh.read(n)
        return rd
    return SafetensorsSource(files, opener)


def _http_get(url: str, start: int, n: int, tries: int = 4) -> bytes:
    """One byte range, anonymously: no Authorization header is ever sent (the PC's HF token is not used)."""
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, start + n - 1),
                                                       "User-Agent": "strata-qwen35-convert"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != n:
                raise IOError("short range read: %d of %d bytes" % (len(data), n))
            return data
        except Exception as e:     # noqa: BLE001 - retried, then raised
            last = e
            time.sleep(2 + 4 * attempt)
    raise IOError("range %d+%d of %s failed: %s" % (start, n, url, last))


def remote_source(repo: str, revision: str = "main", cache_dir=None, piece: int = 64 << 20,
                  workers: int = 6) -> SafetensorsSource:
    """The model read over HTTP byte ranges from huggingface.co (public repos only).

    `cache_dir` keeps the ranges read (one file per range), so a second run over the same layers does not fetch
    them again - for development on a few layers; the full model needs ~8.3 GB there.
    """
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url_of = lambda f: "%s/%s/resolve/%s/%s" % (base, repo, revision, f)  # noqa: E731
    idx = json.loads(_http_get_all(url_of("model.safetensors.index.json")))
    files = _index_files(idx, "model.safetensors")
    pool = concurrent.futures.ThreadPoolExecutor(workers)
    cache = pathlib.Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    def opener(f):
        url = url_of(f)

        def rd(off, n):
            key = cache / ("%s.%d.%d.bin" % (f, off, n)) if cache else None
            if key is not None and key.exists() and key.stat().st_size == n:
                return key.read_bytes()
            parts = [(o, min(piece, off + n - o)) for o in range(off, off + n, piece)]
            data = b"".join(pool.map(lambda p: _http_get(url, p[0], p[1]), parts))
            if key is not None and n > 4096:
                tmp = key.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(key)
            return data
        return rd
    return SafetensorsSource(files, opener)


def _http_get_all(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "strata-qwen35-convert"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


# ------------------------------------------------------------------ encodings
def _roundf(x: np.ndarray) -> np.ndarray:
    """C roundf (halves away from zero), as ggml's quantize_row_q8_0_ref and gguf-py's np_roundf use."""
    a = np.abs(x)
    f = np.floor(a)
    return np.sign(x) * (f + np.floor(2 * (a - f)))


Q8_BLOCK = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])


def quantize_q8_0(values: np.ndarray, scale: float = 1.0) -> bytes:
    """float32 (rows, n) with n % 32 == 0 -> Q8_0 blocks, row-major.  ggml's reference quantizer: d = amax/127,
    q = roundf(x / d), d stored as fp16.  `scale` multiplies the STORED fp16 d afterwards (2.0 for the shared
    expert's down projection): a power of two times an fp16 is exact, so it is the same dequantized tensor as
    quantize(w) times `scale`, and the codes do not move."""
    x = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if x.size % 32:
        raise ValueError("Q8_0 needs a multiple of 32 elements, got %d" % x.size)
    out = np.empty(x.size // 32, dtype=Q8_BLOCK)
    step = 1 << 22                                    # blocks per slice: bounds the temporaries for the embedding
    for s in range(0, out.size, step):
        b = x[s * 32:(s + step) * 32].reshape(-1, 32)
        d = np.max(np.abs(b), axis=1) / np.float32(127)
        with np.errstate(divide="ignore"):
            inv = np.where(d > 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
        q = _roundf(b * inv[:, None])
        d16 = d.astype(np.float16)
        if scale != 1.0:
            with np.errstate(over="ignore"):
                d16 = (d16 * np.float16(scale)).astype(np.float16)
        if not np.isfinite(d16).all():
            raise ValueError("a Q8_0 scale overflows fp16")
        out["d"][s:s + b.shape[0]] = d16
        out["qs"][s:s + b.shape[0]] = q.astype(np.int8)
    return out.tobytes()


def dequantize_q8_0(raw: bytes) -> np.ndarray:
    b = np.frombuffer(raw, dtype=Q8_BLOCK)
    return (b["d"].astype(np.float32)[:, None] * b["qs"].astype(np.float32)).reshape(-1)


def bf16_bytes(values: np.ndarray, name: str) -> bytes:
    """float32 values that ARE bf16 (they came from bf16, or are zero) -> the bf16 bytes.  Refuses anything that
    would round: these tensors are copies, never re-quantizations."""
    u = np.ascontiguousarray(values, dtype=np.float32).reshape(-1).view(np.uint32)
    if np.count_nonzero(u & 0xFFFF):
        raise ValueError("%s: values are not exactly bf16" % name)
    return (u >> 16).astype("<u2").tobytes()


@dataclasses.dataclass
class Tensor:
    name: str
    type_name: str      # F32 | BF16 | Q8_0
    shape: list         # GGUF order: shape[0] = ne0 is the contiguous axis
    data: bytes

    def values(self) -> np.ndarray:
        """What the engine reads, as float32 in (ne1, ne0) row-major (or (ne0,) for 1-D)."""
        if self.type_name == "Q8_0":
            v = dequantize_q8_0(self.data)
        elif self.type_name == "BF16":
            v = (np.frombuffer(self.data, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            v = np.frombuffer(self.data, dtype="<f4")          # read-only view, no copy (the embedding is 2.5 GB)
        return v.reshape(tuple(reversed(self.shape)))


def _t(name, type_name, arr: np.ndarray, scale: float = 1.0, proj: str = "Q8_0") -> Tensor:
    """arr in row-major (rows, ne0) - an HF Linear weight (out, in) is already that - or 1-D.

    `proj="F32"` writes the Q8_0 tensors unquantized instead (times `scale`, exact): a check-only form that isolates
    the mapping from the quantization.  The engine cannot load it."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    if type_name == "Q8_0" and proj == "F32":
        type_name, a, scale = "F32", (a * np.float32(scale)).astype(np.float32), 1.0
    shape = [int(a.shape[-1])] + ([int(a.shape[0])] if a.ndim == 2 else [])
    if a.ndim > 2:
        raise ValueError("%s: %d-D" % (name, a.ndim))
    if type_name == "Q8_0":
        data = quantize_q8_0(a, scale)
    elif type_name == "BF16":
        if scale != 1.0:
            raise ValueError("scale is a Q8_0 option")
        data = bf16_bytes(a, name)
    else:
        if scale != 1.0:
            raise ValueError("scale is a Q8_0 option")
        data = a.astype("<f4").tobytes()
    return Tensor(name, type_name, shape, data)


# ------------------------------------------------------------------ the mapping
def plan(g: Geometry) -> list:
    """Every tensor written, as (name, type, GGUF shape), in file order.  The mapping below is checked against it."""
    E, H = g.n_embd, g.hc_dim
    rows = [("token_embd.weight", "Q8_0", [E, g.vocab]),
            ("output_hc_norm.weight", "F32", [H]),
            ("output_hc_down.weight", "BF16", [H, g.hc_lr]),
            ("output_hc_up.weight", "BF16", [g.hc_lr, H]),
            ("output.weight", "Q8_0", [E, g.vocab])]
    for l in range(g.n_layers):
        p = "blk.%d." % l
        for part in ("attn", "ffn"):
            rows += [(p + "hc_%s_norm.weight" % part, "F32", [H]),
                     (p + "hc_%s_down.weight" % part, "BF16", [H, g.hc_lr]),
                     (p + "hc_%s_up.weight" % part, "BF16", [g.hc_lr, H]),
                     (p + "hc_%s_inject.weight" % part, "BF16", [H, g.hc])]
        if g.is_attention(l):
            rows += [(p + "attn_q.weight", "Q8_0", [E, 2 * g.n_head * g.head_dim]),
                     (p + "attn_k.weight", "Q8_0", [E, g.n_head_kv * g.head_dim]),
                     (p + "attn_v.weight", "Q8_0", [E, g.n_head_kv * g.head_dim]),
                     (p + "attn_output.weight", "Q8_0", [g.n_head * g.head_dim, E]),
                     (p + "attn_q_norm.weight", "F32", [g.head_dim]),
                     (p + "attn_k_norm.weight", "F32", [g.head_dim])]
        else:
            rows += [(p + "attn_qkv.weight", "Q8_0", [E, g.conv_channels]),
                     (p + "attn_gate.weight", "Q8_0", [E, g.value_dim]),
                     (p + "ssm_out.weight", "Q8_0", [g.value_dim, E]),
                     (p + "ssm_conv1d.weight", "F32", [g.d_conv, g.conv_channels]),
                     (p + "ssm_alpha.weight", "BF16", [E, g.ssm_v_heads]),
                     (p + "ssm_beta.weight", "BF16", [E, g.ssm_v_heads]),
                     (p + "ssm_a", "F32", [g.ssm_v_heads]),
                     (p + "ssm_dt.bias", "F32", [g.ssm_v_heads]),
                     (p + "ssm_norm.weight", "F32", [g.ssm_v_dim])]
        rows += [(p + "ffn_gate_inp_shexp.weight", "BF16", [E]),
                 (p + "ffn_gate_shexp.weight", "Q8_0", [E, g.n_ff]),
                 (p + "ffn_up_shexp.weight", "Q8_0", [E, g.n_ff]),
                 (p + "ffn_down_shexp.weight", "Q8_0", [g.n_ff, E])]
    return rows


def _hc_norm(w: np.ndarray, g: Geometry) -> np.ndarray:
    """HF's zero-centred gamma w (the norm multiplies by 1+w) -> gr_read's stored per-stream gamma, x4 streams.
    The x2 undoes the read's sigmoid(0) = 0.5 gate.  Computed in float32: 1+w of a bf16 w is exact in f32 (|w| is
    far below 2^8), and x2 is exact."""
    return np.tile((np.float32(2) * (np.float32(1) + w.astype(np.float32))).astype(np.float32), g.hc)


def convert_globals(src: SafetensorsSource, g: Geometry, proj: str = "Q8_0") -> dict:
    emb = src.get(HF_PREFIX + "embed_tokens.weight")
    if emb.shape != (g.vocab, g.n_embd):
        raise ValueError("embed_tokens is %s, config says (%d, %d)" % (emb.shape, g.vocab, g.n_embd))
    if src.has("lm_head.weight"):
        raise ValueError("this checkpoint has its own lm_head.weight; the converter assumes the tied embedding")
    tok = _t("token_embd.weight", "Q8_0", emb, proj=proj)
    out = {"token_embd.weight": tok,
           # the head is the same matrix (tie_word_embeddings): the same Q8_0 bytes
           "output.weight": Tensor("output.weight", tok.type_name, list(tok.shape), tok.data),
           "output_hc_norm.weight": _t("output_hc_norm.weight", "F32", _hc_norm(src.get(HF_PREFIX + "norm.weight"), g)),
           "output_hc_down.weight": _t("output_hc_down.weight", "BF16", np.zeros((g.hc_lr, g.hc_dim), np.float32)),
           "output_hc_up.weight": _t("output_hc_up.weight", "BF16", np.zeros((g.hc_dim, g.hc_lr), np.float32))}
    del emb
    return out


def convert_layer(src: SafetensorsSource, g: Geometry, l: int, proj: str = "Q8_0") -> dict:
    hf = lambda s: src.get("%slayers.%d.%s" % (HF_PREFIX, l, s))  # noqa: E731
    p = "blk.%d." % l
    out = {}
    put = lambda t: out.__setitem__(t.name, t)  # noqa: E731
    zeros = lambda *s: np.zeros(s, np.float32)  # noqa: E731
    for part, norm in (("attn", "input_layernorm.weight"), ("ffn", "post_attention_layernorm.weight")):
        put(_t(p + "hc_%s_norm.weight" % part, "F32", _hc_norm(hf(norm), g)))
        put(_t(p + "hc_%s_down.weight" % part, "BF16", zeros(g.hc_lr, g.hc_dim)))
        put(_t(p + "hc_%s_up.weight" % part, "BF16", zeros(g.hc_dim, g.hc_lr)))
        put(_t(p + "hc_%s_inject.weight" % part, "BF16", zeros(g.hc, g.hc_dim)))
    if g.is_attention(l):
        put(_t(p + "attn_q.weight", "Q8_0", hf("self_attn.q_proj.weight"), proj=proj))
        put(_t(p + "attn_k.weight", "Q8_0", hf("self_attn.k_proj.weight"), proj=proj))
        put(_t(p + "attn_v.weight", "Q8_0", hf("self_attn.v_proj.weight"), proj=proj))
        put(_t(p + "attn_output.weight", "Q8_0", hf("self_attn.o_proj.weight"), proj=proj))
        put(_t(p + "attn_q_norm.weight", "F32", np.float32(1) + hf("self_attn.q_norm.weight")))
        put(_t(p + "attn_k_norm.weight", "F32", np.float32(1) + hf("self_attn.k_norm.weight")))
    else:
        order = v_head_order(g)
        S, qk = g.ssm_v_dim, g.ssm_k_heads * g.ssm_k_dim
        # value-channel order inside the qkv / conv channels and inside z / o: head blocks of S, permuted
        v_ch = (order[:, None] * S + np.arange(S)[None, :]).reshape(-1)
        ch = np.concatenate([np.arange(2 * qk), 2 * qk + v_ch])
        qkv = hf("linear_attn.in_proj_qkv.weight")
        if qkv.shape != (g.conv_channels, g.n_embd):
            raise ValueError("layer %d: in_proj_qkv is %s" % (l, qkv.shape))
        put(_t(p + "attn_qkv.weight", "Q8_0", qkv[ch], proj=proj))
        put(_t(p + "attn_gate.weight", "Q8_0", hf("linear_attn.in_proj_z.weight")[v_ch], proj=proj))
        put(_t(p + "ssm_out.weight", "Q8_0", hf("linear_attn.out_proj.weight")[:, v_ch], proj=proj))
        conv = hf("linear_attn.conv1d.weight")              # (C, 1, d_conv): tap 0 is the oldest input
        put(_t(p + "ssm_conv1d.weight", "F32", conv.reshape(g.conv_channels, g.d_conv)[ch]))
        put(_t(p + "ssm_alpha.weight", "BF16", hf("linear_attn.in_proj_a.weight")[order]))
        put(_t(p + "ssm_beta.weight", "BF16", hf("linear_attn.in_proj_b.weight")[order]))
        put(_t(p + "ssm_a", "F32", -np.exp(hf("linear_attn.A_log").astype(np.float32))[order]))
        put(_t(p + "ssm_dt.bias", "F32", hf("linear_attn.dt_bias")[order]))
        put(_t(p + "ssm_norm.weight", "F32", hf("linear_attn.norm.weight")))
    put(_t(p + "ffn_gate_inp_shexp.weight", "BF16", zeros(g.n_embd)))
    put(_t(p + "ffn_gate_shexp.weight", "Q8_0", hf("mlp.gate_proj.weight"), proj=proj))
    put(_t(p + "ffn_up_shexp.weight", "Q8_0", hf("mlp.up_proj.weight"), proj=proj))
    # the scalar gate is sigmoid(0) = 0.5, so down carries x2 - in its fp16 block scales, exactly
    put(_t(p + "ffn_down_shexp.weight", "Q8_0", hf("mlp.down_proj.weight"), scale=2.0, proj=proj))
    return out


def check_against_plan(group: dict, rows: list) -> None:
    want = {n: (t, s) for n, t, s in rows}
    for name, t in group.items():
        if name not in want:
            raise AssertionError("%s is not in the plan" % name)
        if (t.type_name, list(t.shape)) != (want[name][0], list(want[name][1])):
            raise AssertionError("%s is %s %s, the plan says %s %s" % (name, t.type_name, t.shape, *want[name]))


def skipped_hf_tensors(src: SafetensorsSource) -> dict:
    """The checkpoint's tensors this converter does not write, by group."""
    groups = {}
    for n in src.where:
        if n.startswith(HF_PREFIX):
            continue
        key = n.split(".")[0] if not n.startswith("model.visual") else "model.visual"
        groups[key] = groups.get(key, 0) + 1
    return groups


# ------------------------------------------------------------------ metadata
def metadata(g: Geometry, arch: str, tokenizer: dict | None, source: str) -> list:
    """(key, gguf type, value), in write order.  The geometry keys stage 2 reads, under the architecture's prefix."""
    a = arch
    md = [("general.architecture", "string", a),
          ("general.type", "string", "model"),
          ("general.name", "string", "Qwen3.5-4B in qwen4exp's shape"),
          ("general.file_type", "u32", FILE_TYPE_MOSTLY_Q8_0),
          ("general.quantization_version", "u32", 2),
          ("general.source.repo_url", "string", source),
          (a + ".block_count", "u32", g.n_layers),
          (a + ".context_length", "u32", g.context),
          (a + ".embedding_length", "u32", g.n_embd),
          (a + ".feed_forward_length", "u32", g.n_ff),
          (a + ".expert_count", "u32", 0),
          (a + ".expert_used_count", "u32", 0),
          (a + ".expert_shared_feed_forward_length", "u32", g.n_ff),
          (a + ".attention.head_count", "u32", g.n_head),
          (a + ".attention.head_count_kv", "u32", g.n_head_kv),
          (a + ".attention.key_length", "u32", g.head_dim),
          (a + ".attention.value_length", "u32", g.head_dim),
          (a + ".attention.layer_norm_rms_epsilon", "f32", g.eps),
          (a + ".rope.dimension_count", "u32", g.n_rot),
          (a + ".rope.freq_base", "f32", g.rope_base),
          (a + ".rope.dimension_sections", "array:i32", list(g.rope_sections) + [0] * (4 - len(g.rope_sections))),
          (a + ".ssm.conv_kernel", "u32", g.d_conv),
          (a + ".ssm.state_size", "u32", g.ssm_k_dim),
          (a + ".ssm.group_count", "u32", g.ssm_k_heads),
          (a + ".ssm.time_step_rank", "u32", g.ssm_v_heads),
          (a + ".ssm.inner_size", "u32", g.value_dim),
          # HF Qwen3.5's GDN output gate; today's kernels apply sigmoid (Flash-Next).  Stage 2's switch.
          (a + ".ssm.output_gate", "string", "silu"),
          (a + ".full_attention_interval", "u32", g.full_attention_interval),
          (a + ".hyper_connection.count", "u32", g.hc),
          (a + ".hyper_connection.low_rank", "u32", g.hc_lr),
          # how the weights were made, so a reader of the file can tell a converted dense model from Flash-Next
          ("strata.convert.tool", "string", TOOL),
          ("strata.convert.hyper_connections", "string", "zero (plain residual); norms 2*(1+w) x hc"),
          ("strata.convert.shared_expert", "string", "dense FFN; gate_inp_shexp 0, down scales x2"),
          ("strata.convert.v_head_order", "string", "modulo (engine head e = HF head (e%h_k)*r + e//h_k)")]
    if tokenizer:
        md += [("tokenizer.ggml.model", "string", "gpt2"),
               ("tokenizer.ggml.pre", "string", "qwen35"),
               ("tokenizer.ggml.tokens", "array:string", tokenizer["tokens"]),
               ("tokenizer.ggml.token_type", "array:i32", tokenizer["token_type"]),
               ("tokenizer.ggml.merges", "array:string", tokenizer["merges"])]
        for k in ("eos", "padding", "bos"):
            if tokenizer.get(k) is not None:
                md.append(("tokenizer.ggml.%s_token_id" % k, "u32", tokenizer[k]))
        md.append(("tokenizer.ggml.add_bos_token", "bool", False))
        if tokenizer.get("chat_template"):
            md.append(("tokenizer.chat_template", "string", tokenizer["chat_template"]))
    return md


def hf_tokenizer(model_dir: pathlib.Path, vocab_size: int) -> dict:
    """tokenizer.json -> the GGUF tokenizer arrays, the way llama.cpp's convert_hf_to_gguf builds a gpt2 vocab:
    tokens by id, ids the file leaves empty as [PADn] UNUSED (5), added tokens CONTROL (3) when special or shaped
    <|...|> (does_token_look_special), other added tokens USER_DEFINED (4), the rest NORMAL (1).  Checked against
    the Flash-Next GGUFs, which share this vocabulary: merges identical, and the token table differs only where the
    4B's tokenizer.json has no token (248070-248076, Flash-Next's audio/tts tokens)."""
    tj = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tc = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    if tj["model"]["type"] != "BPE":
        raise ValueError("tokenizer model %r, expected BPE" % tj["model"]["type"])
    tokens = ["[PAD%d]" % i for i in range(vocab_size)]
    types = [5] * vocab_size
    for tok, i in tj["model"]["vocab"].items():
        tokens[i], types[i] = tok, 1
    for a in tj.get("added_tokens", []):
        c = a["content"]
        looks = c.startswith("<|") and c.endswith("|>")
        tokens[a["id"]], types[a["id"]] = c, (3 if a.get("special") or looks else 4)
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
    ids = {t: i for i, t in enumerate(tokens)}
    eos, pad = tc.get("eos_token"), tc.get("pad_token")
    tpl = model_dir / "chat_template.jinja"
    return {"tokens": tokens, "token_type": types, "merges": merges,
            "eos": ids.get(eos) if eos else None, "padding": ids.get(pad) if pad else None,
            # llama.cpp's gpt2 vocab for Qwen sets bos to <|endoftext|> as well (the Flash-Next GGUFs carry it)
            "bos": ids.get(pad) if pad else None,
            "chat_template": tpl.read_text(encoding="utf-8") if tpl.exists() else tc.get("chat_template")}


# ------------------------------------------------------------------ writing
class StreamingGGUF(GGUFWriter):
    """gguf_writer's metadata encoding, with the tensor data streamed group by group instead of held in memory
    (the Q8_0 model is ~5 GB)."""

    def write_stream(self, path, rows: list, groups) -> None:
        offsets, at = [], 0
        for name, type_name, shape in rows:
            offsets.append(at)
            at += _nbytes(type_name, shape)
            at = (at + self.alignment - 1) // self.alignment * self.alignment
        head = bytearray()
        head += struct.pack("<IIQQ", 0x46554747, 3, len(rows), len(self.metadata))
        for k, (t, v) in self.metadata.items():
            head += self._kv_bytes(k, t, v)
        for (name, type_name, shape), off in zip(rows, offsets):
            head += struct.pack("<Q", len(name)) + name.encode("utf-8")
            head += struct.pack("<I", len(shape)) + struct.pack("<%dQ" % len(shape), *shape)
            head += struct.pack("<IQ", GGML_ID[type_name], off)
        head += b"\0" * ((-len(head)) % self.alignment)
        tmp = pathlib.Path(str(path) + ".tmp")
        order = {name: i for i, (name, _, _) in enumerate(rows)}
        nxt = 0
        with open(tmp, "wb") as fo:
            fo.write(head)
            for group in groups:
                for name in sorted(group, key=order.__getitem__):
                    i = order[name]
                    if i != nxt:
                        raise AssertionError("tensor %s arrived out of order (expected %s)" % (name, rows[nxt][0]))
                    t = group[name]
                    if len(t.data) != _nbytes(t.type_name, t.shape):
                        raise AssertionError("%s: %d bytes" % (name, len(t.data)))
                    fo.write(t.data)
                    fo.write(b"\0" * ((-len(t.data)) % self.alignment))
                    nxt += 1
        if nxt != len(rows):
            raise AssertionError("wrote %d of %d tensors" % (nxt, len(rows)))
        tmp.replace(path)


def _nbytes(type_name: str, shape) -> int:
    n = int(np.prod(shape))
    return {"F32": 4 * n, "BF16": 2 * n, "Q8_0": n // 32 * 34}[type_name]


def groups(src: SafetensorsSource, g: Geometry, rows: list, log=print):
    out = convert_globals(src, g)
    check_against_plan(out, rows)
    yield out
    for l in range(g.n_layers):
        t0 = time.time()
        grp = convert_layer(src, g, l)
        check_against_plan(grp, rows)
        log("  layer %2d %-9s %.1f s" % (l, "attention" if g.is_attention(l) else "GDN", time.time() - t0))
        yield grp


def write_gguf(path, src: SafetensorsSource, g: Geometry, arch: str, tokenizer, source: str, log=print) -> None:
    w = StreamingGGUF()
    for k, t, v in metadata(g, arch, tokenizer, source):
        w.add(k, v, t)
    rows = plan(g)
    w.write_stream(path, [(n, t, s) for n, t, s in rows], groups(src, g, rows, log))


def write_pack(gguf_path, out_dir) -> int:
    """iq_pack's standalone index (dense.bin, index.txt, conversions.json) and the tokenizer directory.  The expert
    table is not written: this model has none (see the module docstring)."""
    import iq_pack
    import strata_tokenizer
    src = pathlib.Path(gguf_path).absolute()
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rc = iq_pack.index_standalone(src, out, iq_pack.Model(src))
    if rc:
        return rc
    strata_tokenizer.extract(src, out)
    return 0


# ------------------------------------------------------------------ what today's engine would still refuse
def stage2_refusals(g: Geometry, arch: str) -> list:
    """Read from the code at this commit (gguf_reader.hpp, layout.cpp/.hpp, iq_pack.py, the kernels); not run."""
    n_qsa = sum(g.is_attention(l) for l in range(g.n_layers))
    return [
        "check_architecture (include/strata/artifact/gguf_reader.hpp): general.architecture '%s' != 'qwen4exp'; "
        "with --arch qwen4exp it still wants block_count 48 / head_count 24 / head_count_kv 2 (Qwen4ExpGuard)" % arch,
        "ModelGeometry (include/strata/core/layout.hpp) is compiled-in Flash-Next: 48 layers, 24/2 heads, GDN 48 "
        "v-heads, 10240 conv channels, 6144 value dim, n_ff 640, 512 experts; the only keys read from the file are "
        "qwen4exp.expert_count / expert_used_count / rope.* (src/program/generate.cpp ~3905), under the literal "
        "prefix 'qwen4exp.', so a '%s' file's keys are not read at all" % arch,
        "check_all (src/core/layout.cpp): blk.N.ffn_gate_inp.weight [n_embd, n_expert] on EVERY layer (not written: "
        "n_expert 0); indexer.q_proj/k_proj/q_norm/k_norm on every attention layer (not written); the split must be "
        "%d attention / %d GDN against the geometry's 12 / 36" % (n_qsa, g.n_layers - n_qsa),
        "pack: no native_experts.txt (iq_pack's completion marker, read by src/core/expert_source.cpp and setup); "
        "iq_pack.expert_layout refuses a model with no _exps tensors",
        "GDN output gate: sigmoid(z) in fused_gdn.cu:69 / gdn.cu gdn_out_norm / native_gdn_out_norm; Qwen3.5 is "
        "silu(z) (GGUF key %s.ssm.output_gate = silu)" % arch,
        "GDN prompt kernels fix HV 48 and C 10240 (plan item 3); this model is HV %d, C %d"
        % (g.ssm_v_heads, g.conv_channels),
        "attention: qsa_real_shapes() fixes 24/2 heads (GQA 12); this model is %d/%d (GQA %d); QSA selection with "
        "an indexer past 2051 cells - this model needs dense attention" % (g.n_head, g.n_head_kv,
                                                                           g.n_head // g.n_head_kv),
        "shared expert width %d against Flash-Next's 640 (shared_expert scratch is sized from n_ff; any kernel "
        "constant for 640 is stage 2's to find)" % g.n_ff,
        "PLE: run with --no-ple (verify.cpp); MTP: none (no --mtp/--spec)",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src_g = ap.add_mutually_exclusive_group(required=True)
    src_g.add_argument("--hf", help="local HF model directory (config.json, *.safetensors, tokenizer.json)")
    src_g.add_argument("--hf-repo", help="read a public HF repo over HTTP byte ranges, e.g. Qwen/Qwen3.5-4B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--cache", help="--hf-repo only: keep the ranges read here")
    ap.add_argument("--meta", help="--hf-repo only: a directory with config.json / tokenizer.json / "
                                   "tokenizer_config.json / chat_template.jinja (default: fetched)")
    ap.add_argument("--gguf", help="write the native GGUF here")
    ap.add_argument("--pack", help="write the pack (index.txt, dense.bin, tokenizer/) here, from --gguf")
    ap.add_argument("--arch", default="qwen35dense", help="general.architecture (default qwen35dense)")
    ap.add_argument("--plan", action="store_true", help="print the tensor table and stage 2's refusals")
    a = ap.parse_args()

    if a.hf:
        meta_dir = pathlib.Path(a.hf)
    else:
        meta_dir = pathlib.Path(a.meta) if a.meta else None
        if meta_dir is None:
            import tempfile
            meta_dir = pathlib.Path(tempfile.mkdtemp(prefix="qwen35-meta-"))
            base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
            for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                (meta_dir / f).write_bytes(_http_get_all("%s/%s/resolve/%s/%s" % (base, a.hf_repo, a.revision, f)))
    cfg = json.loads((meta_dir / "config.json").read_text(encoding="utf-8"))
    g = Geometry.from_config(cfg)
    rows = plan(g)
    if a.plan:
        for n, t, s in rows:
            if not n.startswith("blk.") or n.startswith(("blk.0.", "blk.3.")):
                print("%-34s %-5s %s" % (n, t, s))
        print("... %d tensors, %.2f GiB" % (len(rows), sum(_nbytes(t, s) for _, t, s in rows) / 2**30))
        print("\nwhat today's engine still refuses (stage 2):")
        for r in stage2_refusals(g, a.arch):
            print("  - " + r)
        if not a.gguf:
            return 0
    if not a.gguf:
        ap.error("--gguf is required (or --plan)")
    src = local_source(a.hf) if a.hf else remote_source(a.hf_repo, a.revision, a.cache)
    print("skipped from the checkpoint: %s" % ", ".join("%s (%d)" % kv for kv in sorted(skipped_hf_tensors(src).items())))
    tok = hf_tokenizer(meta_dir, g.vocab)
    source = "https://huggingface.co/%s" % (a.hf_repo or "Qwen/Qwen3.5-4B")
    t0 = time.time()
    write_gguf(a.gguf, src, g, a.arch, tok, source)
    print("%s: %d tensors, %.2f GiB, %.0f s" % (a.gguf, len(rows), os.path.getsize(a.gguf) / 2**30, time.time() - t0))
    if a.pack:
        return write_pack(a.gguf, a.pack)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
