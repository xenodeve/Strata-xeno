"""ref/qwen4exp_engine.py - one decode step of the engine's qwen4exp layer, in numpy, for the Qwen3.5-4B check (#297).

A transcription of what the CUDA reference kernels compute, token by token (the decode path, which stage 2 ports
first), so converted weights can be checked against Hugging Face without a GPU or an engine build:

  layer      src/core/layer.cpp block_layer: gr_read(hc_attn) -> GDN or attention -> gr_write -> gr_read(hc_ffn)
             -> routed experts (none here) + shared expert -> gr_write.  Head: lm_head_mix = gr_read(output_hc_*)
             with no injection, then output.weight.  The embedding row is broadcast to every stream (session.cpp).
  GDN        layer.cpp gdn_layer + gdn.cu / elementwise.cu: conv over [conv_state | qkv] with tap 0 the oldest,
             silu; q, k L2-normalised per head with eps on the SUM (gdn_l2_norm); q *= 1/sqrt(S);
             gate = softplus(alpha + dt) * ssm_a, dec = exp(gate); beta = sigmoid(beta); the delta rule with value
             head h reading key head h % h_k; y = rms_norm(o per head) * ssm_norm * act(z).
             act is SIGMOID for Flash-Next; `gdn_gate="silu"` is Qwen3.5's, which the engine runs since #297
             stage 2 (strata::kernels::gdn_gate_silu, set from qwen35dense.ssm.output_gate).
  attention  layer.cpp qsa_layer, DENSE (no indexer; the 4B has none): q_full (n_head, 2*head_dim), q = first half
             and gate = second half of each head's row; rms_norm_weighted(q / k per head, stored gamma) then NEOX rope
             on the first n_rot dims (rope.cu, the table built in float64); softmax(q.k / sqrt(head_dim)) over every
             cached cell, query head h reading kv head h // (n_head / n_head_kv); attn * sigmoid(gate); attn_output.
  shared     shared_expert.cu: sigmoid(x . gate_inp_shexp) * down(silu(gate x) * up x), added to the routed sum.

Weights are a dict name -> float32 array in (ne1, ne0) row-major (an engine GEMV is y[o] = sum_i W[o, i] x[i]), as
`tools/qwen35_to_qwen4exp.Tensor.values()` returns them.  Activations are float32.  The engine quantizes GEMV inputs
(Q8_0 / Q8_1 / Q8_K images, bf16 for the BF16 readers); `act_q8=True` models a Q8_0 image of every Q8_0 GEMV's
input as an approximation of that, and is off by default.  `bf16_act=True` rounds the inputs of the BF16 readers (and
gr_read's projections) to bf16, as the legacy f32_to_bf16_bulk / project_bf16 path does; the native and fused paths
(fused_gdn_ab, native MMVF) read FP32 activations, which is the default here.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from gr import bf16_round, gr_read, gr_write, sigmoid, silu


@dataclasses.dataclass
class Geo:
    n_embd: int
    n_layers: int
    attention_layers: tuple
    n_head: int
    n_head_kv: int
    head_dim: int
    n_rot: int
    rope_base: float
    S: int            # GDN head dim (state size)
    h_k: int
    h_v: int
    d_conv: int
    hc: int
    eps: float = 1e-6

    @classmethod
    def of(cls, g) -> "Geo":
        """From tools/qwen35_to_qwen4exp.Geometry."""
        return cls(g.n_embd, g.n_layers, tuple(l for l in range(g.n_layers) if g.is_attention(l)), g.n_head,
                   g.n_head_kv, g.head_dim, g.n_rot, g.rope_base, g.ssm_k_dim, g.ssm_k_heads, g.ssm_v_heads,
                   g.d_conv, g.hc, g.eps)


def _q8_image(x: np.ndarray) -> np.ndarray:
    b = x.astype(np.float32).reshape(-1, 32)
    d = np.max(np.abs(b), axis=1, keepdims=True) / np.float32(127)
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0).astype(np.float32)
    a = np.abs(b * inv)
    q = np.sign(b) * (np.floor(a) + np.floor(2 * (a - np.floor(a))))
    return (q * d.astype(np.float16).astype(np.float32)).reshape(-1).astype(np.float32)


class Engine:
    def __init__(self, W: dict, geo: Geo, gdn_gate: str = "sigmoid", bf16_act: bool = False, act_q8: bool = False):
        if gdn_gate not in ("sigmoid", "silu"):
            raise ValueError(gdn_gate)
        self.W, self.g, self.gdn_gate, self.bf16_act, self.act_q8 = W, geo, gdn_gate, bf16_act, act_q8
        half = geo.n_rot // 2
        self.inv_freq = np.power(np.float64(geo.rope_base), -2.0 * np.arange(half, dtype=np.float64) / geo.n_rot)
        self.reset()

    def reset(self):
        g = self.g
        C = 2 * g.h_k * g.S + g.h_v * g.S
        self.conv = {l: np.zeros((g.d_conv - 1, C), np.float32) for l in range(g.n_layers)}
        self.state = {l: np.zeros((g.h_v, g.S, g.S), np.float32) for l in range(g.n_layers)}   # [h, i(key), j(val)]
        self.K = {l: [] for l in g.attention_layers}
        self.V = {l: [] for l in g.attention_layers}

    # ---- primitives
    def mv(self, name: str, x: np.ndarray, quantized: bool = True) -> np.ndarray:
        w = self.W[name]
        if quantized and self.act_q8:
            x = _q8_image(x)
        elif not quantized and self.bf16_act:
            x = bf16_round(x)
        return (w @ x.astype(np.float32)).astype(np.float32)

    @staticmethod
    def rms_rows(x: np.ndarray, w, eps: float) -> np.ndarray:
        ms = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
        y = x * (1.0 / np.sqrt(ms + np.float32(eps))).astype(np.float32)
        return (y * w if w is not None else y).astype(np.float32)

    def rope(self, x: np.ndarray, pos: int) -> np.ndarray:
        """NEOX pairs (i, i + n_rot/2) over the first n_rot dims of each row."""
        half = self.g.n_rot // 2
        ang = pos * self.inv_freq
        c, s = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
        y = x.copy()
        a, b = x[:, :half], x[:, half:2 * half]
        y[:, :half] = a * c - b * s
        y[:, half:2 * half] = a * s + b * c
        return y

    # ---- the two mixers
    def gdn(self, l: int, x: np.ndarray) -> np.ndarray:
        g, p = self.g, "blk.%d." % l
        S, hk, hv = g.S, g.h_k, g.h_v
        qk = hk * S
        qkv = self.mv(p + "attn_qkv.weight", x)
        z = self.mv(p + "attn_gate.weight", x)
        alpha = self.mv(p + "ssm_alpha.weight", x, quantized=False)
        beta = self.mv(p + "ssm_beta.weight", x, quantized=False)
        kW = self.W[p + "ssm_conv1d.weight"]                      # (C, d_conv): kW[c, i], tap 0 the oldest
        inp = np.concatenate([self.conv[l], qkv[None, :]], axis=0)  # (d_conv, C)
        h = silu(np.einsum("ic,ci->c", inp, kW).astype(np.float32))
        self.conv[l] = inp[1:].copy()
        q = h[:qk].reshape(hk, S)
        k = h[qk:2 * qk].reshape(hk, S)
        v = h[2 * qk:].reshape(hv, S)
        q = (q / np.sqrt(np.sum(q * q, axis=1, keepdims=True) + np.float32(g.eps))).astype(np.float32)
        k = (k / np.sqrt(np.sum(k * k, axis=1, keepdims=True) + np.float32(g.eps))).astype(np.float32)
        q = (q * np.float32(1.0 / np.sqrt(S))).astype(np.float32)
        a = alpha + self.W[p + "ssm_dt.bias"]
        softplus = np.where(a > 20, a, np.log1p(np.exp(np.minimum(a, 20)))).astype(np.float32)
        dec = np.exp(softplus * self.W[p + "ssm_a"]).astype(np.float32)
        b = sigmoid(beta)
        idx = np.arange(hv) % hk                                   # the engine's pairing: MODULO
        st = self.state[l] * dec[:, None, None]
        kk, qq = k[idx], q[idx]                                    # (hv, S)
        sk = np.einsum("hij,hi->hj", st, kk)
        d = (v - sk) * b[:, None]
        st = st + kk[:, :, None] * d[:, None, :]
        o = np.einsum("hij,hi->hj", st, qq).astype(np.float32)
        self.state[l] = st.astype(np.float32)
        act = sigmoid if self.gdn_gate == "sigmoid" else silu
        y = self.rms_rows(o, self.W[p + "ssm_norm.weight"], g.eps) * act(z.reshape(hv, S))
        return self.mv(p + "ssm_out.weight", y.reshape(-1).astype(np.float32))

    def attention(self, l: int, x: np.ndarray, pos: int) -> np.ndarray:
        g, p = self.g, "blk.%d." % l
        D = g.head_dim
        q_full = self.mv(p + "attn_q.weight", x).reshape(g.n_head, 2 * D)
        q, gate = q_full[:, :D], q_full[:, D:].reshape(-1)
        k = self.mv(p + "attn_k.weight", x).reshape(g.n_head_kv, D)
        v = self.mv(p + "attn_v.weight", x).reshape(g.n_head_kv, D)
        q = self.rope(self.rms_rows(q, self.W[p + "attn_q_norm.weight"], g.eps), pos)
        k = self.rope(self.rms_rows(k, self.W[p + "attn_k_norm.weight"], g.eps), pos)
        self.K[l].append(k)
        self.V[l].append(v)
        K, V = np.stack(self.K[l], 1), np.stack(self.V[l], 1)      # (kv, T, D)
        grp = g.n_head // g.n_head_kv
        kv = np.arange(g.n_head) // grp                            # query head h -> kv head h // group
        sc = np.einsum("hd,htd->ht", q, K[kv]) * np.float32(1.0 / np.sqrt(D))
        sc = sc - sc.max(axis=1, keepdims=True)
        pr = np.exp(sc)
        pr = pr / pr.sum(axis=1, keepdims=True)
        attn = np.einsum("ht,htd->hd", pr, V[kv]).reshape(-1).astype(np.float32)
        return self.mv(p + "attn_output.weight", attn * sigmoid(gate))

    def shared(self, l: int, x: np.ndarray) -> np.ndarray:
        p = "blk.%d." % l
        h = silu(self.mv(p + "ffn_gate_shexp.weight", x)) * self.mv(p + "ffn_up_shexp.weight", x)
        down = self.mv(p + "ffn_down_shexp.weight", h.astype(np.float32))
        gsc = sigmoid(np.float32(np.dot(self.W[p + "ffn_gate_inp_shexp.weight"], x)))
        return (down * gsc).astype(np.float32)

    # ---- a layer, the head, a token
    def layer(self, l: int, R: np.ndarray, pos: int) -> np.ndarray:
        g, p = self.g, "blk.%d." % l
        W = self.W
        mixed, inj = gr_read(R, W[p + "hc_attn_norm.weight"], W[p + "hc_attn_down.weight"],
                             W[p + "hc_attn_up.weight"], W[p + "hc_attn_inject.weight"], g.eps, self.bf16_act)
        out = self.attention(l, mixed, pos) if l in g.attention_layers else self.gdn(l, mixed)
        R = gr_write(R, out, inj)
        mixed, inj = gr_read(R, W[p + "hc_ffn_norm.weight"], W[p + "hc_ffn_down.weight"],
                             W[p + "hc_ffn_up.weight"], W[p + "hc_ffn_inject.weight"], g.eps, self.bf16_act)
        return gr_write(R, self.shared(l, mixed), inj)          # n_expert 0: no routed part

    def embed(self, token: int) -> np.ndarray:
        return np.tile(self.W["token_embd.weight"][token].astype(np.float32), (self.g.hc, 1))

    def head(self, R: np.ndarray) -> np.ndarray:
        mixed, _ = gr_read(R, self.W["output_hc_norm.weight"], self.W["output_hc_down.weight"],
                           self.W["output_hc_up.weight"], None, self.g.eps, self.bf16_act)
        return self.mv("output.weight", mixed)

    def step(self, token: int, pos: int, layers=None):
        """-> (logits, [R after each layer]).  `layers` limits the stack (a check of the first few)."""
        R = self.embed(token)
        trace = []
        for l in range(self.g.n_layers if layers is None else layers):
            R = self.layer(l, R, pos)
            trace.append(R)
        return (self.head(R) if layers is None else None), trace
