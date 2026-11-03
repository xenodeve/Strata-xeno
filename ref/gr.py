"""ref/gr.py - the gated residual (hyper-connection) as the engine computes it, in numpy.

Transcribed from the CUDA reference `src/kernels/cuda/gr.cu` and its header `include/strata/kernels/gr.hpp`
(which name this file as their spec), for the Qwen3.5-4B conversion check (#297):

    xn_c   = R_c * rsqrt(mean(R_c^2) + eps) * w_norm_c          per stream c, over n_embd (gr_norm_kernel)
    lo     = silu((bf16(xn) @ w_down.T) / hc)                    the /hc is INSIDE the silu
    gated  = xn * sigmoid(bf16(lo) @ w_up.T)                     elementwise over hc*n_embd
    mixed  = mean over c of gated_c                              (n_embd,)
    inject = bf16(xn) @ w_inject.T                               (hc,), absent for the final mixer

    gr_write: R_out_c = R_c + block_out * 2*sigmoid(inject_c / hc)

R is (hc, n_embd).  Weights are in MANIFEST layout as float arrays: w_down (hc_lr, hc*n_embd), w_up
(hc*n_embd, hc_lr), w_inject (hc, hc*n_embd).  `bf16_act=True` is the default activation contract (xn and lo
rounded to bf16 before their projections); False is gr_set_fp32_activations(true).
"""
from __future__ import annotations

import numpy as np


def bf16_round(x: np.ndarray) -> np.ndarray:
    """f32 -> nearest-even bf16 -> f32 (finite inputs)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)).astype(np.uint32)
    return r.view(np.float32).reshape(np.shape(x))


def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def silu(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def gr_read(R: np.ndarray, w_norm: np.ndarray, w_down: np.ndarray, w_up: np.ndarray, w_inject, eps: float,
            bf16_act: bool = True):
    """-> (mixed (n_embd,), inject (hc,) or None)."""
    hc, n = R.shape
    R = R.astype(np.float32)
    ms = np.mean(R * R, axis=1, keepdims=True, dtype=np.float32)
    xn = (R * (1.0 / np.sqrt(ms + np.float32(eps))).astype(np.float32) * w_norm.reshape(hc, n)).astype(np.float32)
    flat = xn.reshape(-1)
    xq = bf16_round(flat) if bf16_act else flat
    lo = silu((w_down @ xq) / np.float32(hc))
    lq = bf16_round(lo) if bf16_act else lo
    gated = flat * sigmoid(w_up @ lq)
    mixed = gated.reshape(hc, n).mean(axis=0, dtype=np.float32)
    inject = None if w_inject is None else (w_inject @ xq).astype(np.float32)
    return mixed.astype(np.float32), inject


def gr_write(R: np.ndarray, block_out: np.ndarray, inject: np.ndarray) -> np.ndarray:
    hc = R.shape[0]
    w = (2.0 * sigmoid(inject / np.float32(hc))).astype(np.float32)
    return (R + block_out[None, :] * w[:, None]).astype(np.float32)


def _selftest() -> int:
    """The two identities the Qwen3.5 conversion rests on: zero projections make the read 0.5*norm(R_c)*w_norm on
    equal streams, and a zero injection makes the write a plain residual add."""
    rng = np.random.default_rng(1)
    hc, n, lr = 4, 64, 8
    x = rng.standard_normal(n).astype(np.float32)
    R = np.tile(x, (hc, 1))
    gamma = rng.standard_normal(n).astype(np.float32)
    mixed, inj = gr_read(R, np.tile(2 * gamma, hc), np.zeros((lr, hc * n), np.float32),
                         np.zeros((hc * n, lr), np.float32), np.zeros((hc, hc * n), np.float32), 1e-6)
    want = x / np.sqrt(np.mean(x * x) + 1e-6) * gamma
    ok = np.allclose(mixed, want, rtol=1e-6, atol=1e-6) and np.all(inj == 0)
    out = rng.standard_normal(n).astype(np.float32)
    ok &= np.array_equal(gr_write(R, out, inj), R + out[None, :])
    print("ref/gr.py selftest " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
