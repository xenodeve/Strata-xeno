"""The engine on a qwen35dense model, against the repo's reference math (#297 stage 2).

    python tools/test_qwen35_engine.py --exe build/strata.exe --work <scratch dir>        a synthetic model (GPU)
    python tools/test_qwen35_engine.py --exe build/strata.exe --work <dir> --keep           keep the files

A synthetic Qwen3.5 checkpoint in the 4B's real per-layer geometry (n_embd 2560, attention 16 q / 4 kv heads x 256 with
rope 64, GDN 16 key / 32 value heads x 128, FFN 9216), with fewer layers (default 4: three GDN, one attention) and a
small vocabulary (default 4096), random weights, is converted by tools/qwen35_to_qwen4exp.py into the GGUF and the pack
the engine boots on.  strata.exe then reads a prompt through the verify windows (several tokens a window) and
generates greedily (one token a window), dumping every position's logits (--dump-logits).  The same tokens go
through ref/qwen4exp_engine.py on the converted weights (silu GDN gate, dense attention, the shared expert alone), and
the logits are compared row by row: the relative error max|d| / max|ref|, the argmax agreement and the KL.

What this checks that the unit tests cannot: the boot path of the new architecture (geometry from the file, the
tensor set, no expert machinery), the verify window's dense skips (no router, no indexer, the identity selection),
the GDN silu gate and the GQA-4 attention kernel inside the real graph, the q8_1 widths at n_ff 9216, and the
window/commit bookkeeping (a prompt window of T tokens, then T = 1 windows).  The engine quantizes its GEMV inputs
(Q8_1) and the reference does not (act_q8 approximates it), so the rows agree to about 1e-2, not to float rounding;
the same comparison with the sigmoid gate in the reference shows what a missing switch looks like.

Not checked here: the real 4B (its 5.6 GiB files; the VM run does that: teacher-forced NLL / top-1 against
Hugging Face), long contexts, sampling.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "ref"))
import qwen35_to_qwen4exp as C  # noqa: E402
from gr import bf16_round  # noqa: E402
from qwen4exp_engine import Engine, Geo  # noqa: E402

P = C.HF_PREFIX
CREATE_NO_WINDOW = 0x08000000


def synthetic_config(layers: int, ff: int, vocab: int) -> dict:
    """Qwen3.5-4B's config.json with fewer layers, a smaller FFN if asked, and a small vocabulary."""
    t = {"hidden_size": 2560, "num_hidden_layers": layers, "num_attention_heads": 16, "num_key_value_heads": 4,
         "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
         "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "intermediate_size": ff, "vocab_size": vocab,
         "rms_norm_eps": 1e-6, "max_position_embeddings": 262144, "full_attention_interval": 4,
         "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4), "hidden_act": "silu",
         "attention_bias": False, "attn_output_gate": True, "tie_word_embeddings": True,
         "rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                             "mrope_section": [11, 11, 10], "mrope_interleaved": True}}
    return {"text_config": t}


def synthetic_weights(cfg: dict, seed: int) -> dict:
    """Random bf16 weights, scaled by fan-in so every activation stays O(1) at n_embd 2560 (the tiny unit model's
    scales, 0.045 at fan-in 64, carried to the real widths); norms around their real centres."""
    t = cfg["text_config"]
    rng = np.random.default_rng(seed)
    E, F, V = t["hidden_size"], t["intermediate_size"], t["vocab_size"]
    hk, hv, S = t["linear_num_key_heads"], t["linear_num_value_heads"], t["linear_key_head_dim"]
    nh, nkv, D = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    Cc = 2 * hk * S + hv * S
    b = lambda *s, sc: bf16_round((rng.standard_normal(s) * sc).astype(np.float32))  # noqa: E731
    fan = lambda n, k=1.0: k * 0.36 / np.sqrt(n)  # noqa: E731
    # the tied embedding is also the head: at the real model's scale (~0.02) the logits spread over a few units, so
    # the comparison sees the whole distribution (at 0.5 they were in the hundreds and every softmax one-hot)
    w = {P + "embed_tokens.weight": b(V, E, sc=0.02), P + "norm.weight": b(E, sc=0.3)}
    for l in range(t["num_hidden_layers"]):
        p = "%slayers.%d." % (P, l)
        w[p + "input_layernorm.weight"] = b(E, sc=0.3)
        w[p + "post_attention_layernorm.weight"] = b(E, sc=0.3)
        w[p + "mlp.gate_proj.weight"] = b(F, E, sc=fan(E))
        w[p + "mlp.up_proj.weight"] = b(F, E, sc=fan(E))
        w[p + "mlp.down_proj.weight"] = b(E, F, sc=fan(F))
        if t["layer_types"][l] == "full_attention":
            w[p + "self_attn.q_proj.weight"] = b(2 * nh * D, E, sc=fan(E, 2))
            w[p + "self_attn.k_proj.weight"] = b(nkv * D, E, sc=fan(E, 2))
            w[p + "self_attn.v_proj.weight"] = b(nkv * D, E, sc=fan(E))
            w[p + "self_attn.o_proj.weight"] = b(E, nh * D, sc=fan(nh * D))
            w[p + "self_attn.q_norm.weight"] = b(D, sc=0.3)
            w[p + "self_attn.k_norm.weight"] = b(D, sc=0.3)
        else:
            w[p + "linear_attn.in_proj_qkv.weight"] = b(Cc, E, sc=fan(E, 2))
            w[p + "linear_attn.in_proj_z.weight"] = b(hv * S, E, sc=fan(E, 2))
            w[p + "linear_attn.in_proj_a.weight"] = b(hv, E, sc=fan(E, 2))
            w[p + "linear_attn.in_proj_b.weight"] = b(hv, E, sc=fan(E, 2))
            w[p + "linear_attn.out_proj.weight"] = b(E, hv * S, sc=fan(hv * S))
            w[p + "linear_attn.conv1d.weight"] = b(Cc, 1, 4, sc=0.4)
            w[p + "linear_attn.A_log"] = bf16_round(np.log(rng.uniform(0.5, 8, hv)).astype(np.float32))
            w[p + "linear_attn.dt_bias"] = b(hv, sc=1.0)
            w[p + "linear_attn.norm.weight"] = bf16_round((1 + rng.standard_normal(S) * 0.1).astype(np.float32))
    return w


class DictSource:
    """SafetensorsSource's interface over a dict of float32 arrays (as tools/test_qwen35_convert.py's)."""

    def __init__(self, w: dict):
        self.w = w
        self.where = {k: None for k in w}

    def has(self, name):
        return name in self.w

    def get(self, name):
        return self.w[name].astype(np.float32)


def write_model(work: pathlib.Path, cfg: dict, seed: int) -> tuple:
    import iq_pack
    g = C.Geometry.from_config(cfg)
    src = DictSource(synthetic_weights(cfg, seed))
    gguf = work / "qwen35dense-synthetic.gguf"
    pack = work / "pack"
    t0 = time.time()
    C.write_gguf(gguf, src, g, "qwen35dense", None, "synthetic (tools/test_qwen35_engine.py)", log=lambda *a: None)
    pack.mkdir(parents=True, exist_ok=True)
    if iq_pack.index_standalone(gguf.absolute(), pack, iq_pack.Model(gguf.absolute())):
        raise SystemExit("iq_pack.index_standalone failed")
    print("model: %s (%.0f MiB) and %s in %.1f s" % (gguf, gguf.stat().st_size / 2**20, pack, time.time() - t0))
    return g, src, gguf, pack


def read_dump(path: pathlib.Path) -> np.ndarray:
    raw = path.read_bytes()
    n_vocab, rows = np.frombuffer(raw[:8], np.int32)
    return np.frombuffer(raw[8:], np.float32).reshape(int(rows), int(n_vocab))


def run_engine(exe: pathlib.Path, gguf, pack, tokens: list, max_new: int, work: pathlib.Path, extra: list) -> tuple:
    dump = work / "logits.bin"
    logpos = work / "logpos.tsv"
    if logpos.exists():
        logpos.unlink()
    cmd = [str(exe), "--pack", str(pack), "--native", str(gguf), "--tokens", ",".join(map(str, tokens)),
           "--max-new", str(max_new), "--max-context", str(max(1024, len(tokens) + max_new + 64)),
           "--dump-logits", str(dump)] + extra
    env = dict(os.environ, STRATA_LOGPOS=str(logpos))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0)
    (work / "engine.log").write_text("$ %s\n\n--- stdout\n%s\n--- stderr\n%s" % (" ".join(cmd), r.stdout, r.stderr),
                                     encoding="utf-8")
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise SystemExit("strata.exe exited %d (log: %s)" % (r.returncode, work / "engine.log"))
    gen = []
    for line in r.stdout.splitlines():
        if line.startswith("generated"):
            gen = [int(x) for x in line.split()[1:]]
    print("engine: %.1f s; %s" % (time.time() - t0, " | ".join(l.strip() for l in r.stdout.splitlines() if l.strip())))
    return read_dump(dump), gen, logpos


def kl(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    def lsm(x):
        x = x.astype(np.float64)
        m = x.max()
        return x - (m + np.log(np.sum(np.exp(x - m))))
    lp, lq = lsm(p_logits), lsm(q_logits)
    return float(np.sum(np.exp(lp) * (lp - lq)))


def compare(rows: np.ndarray, ref: np.ndarray) -> dict:
    rel = [float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30)) for a, b in zip(rows, ref)]
    top = [int(np.argmax(a) == np.argmax(b)) for a, b in zip(rows, ref)]
    kls = [kl(b, a) for a, b in zip(rows, ref)]
    ent = []   # the reference's own entropy: a comparison on one-hot softmaxes would say nothing
    for b in ref:
        x = b.astype(np.float64) - b.max()
        p = np.exp(x) / np.sum(np.exp(x))
        ent.append(float(-np.sum(p * np.log(np.maximum(p, 1e-300)))))
    return {"rows": len(rel), "rel_max": max(rel), "rel_mean": float(np.mean(rel)), "top1": sum(top),
            "kl_mean": float(np.mean(kls)), "kl_max": max(kls), "ref_logit_std": float(np.mean(np.std(ref, axis=1))),
            "ref_entropy_mean": float(np.mean(ent))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exe", required=True)
    ap.add_argument("--work", required=True, help="scratch directory for the model files and the dumps")
    ap.add_argument("--layers", type=int, default=4, help="a multiple of 4 (default 4: three GDN, one attention)")
    ap.add_argument("--ff", type=int, default=9216)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=297)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=150,
                    help="random prompt tokens (150: three 64-cell attention chunks, 19 prompt windows)")
    ap.add_argument("--tokens", help="the prompt's token ids, comma-separated (instead of --prompt-len)")
    ap.add_argument("--extra", default="", help="more strata.exe arguments, space-separated")
    ap.add_argument("--rel-gate", type=float, default=0.15,
                    help="fail above this max relative error against the act-q8 reference (measured 0.039 at 4 layers, "
                         "0.10 at 8: random weights amplify the activation rounding with depth)")
    ap.add_argument("--kl-gate", type=float, default=1e-2, help="fail above this max KL(ref || engine) per row")
    ap.add_argument("--json", help="write the numbers here")
    ap.add_argument("--keep", action="store_true", help="keep the model files")
    a = ap.parse_args()
    work = pathlib.Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    cfg = synthetic_config(a.layers, a.ff, a.vocab)
    g, src, gguf, pack = write_model(work, cfg, a.seed)
    tokens = ([int(x) for x in a.tokens.split(",")] if a.tokens else
              [int(x) for x in np.random.default_rng(a.seed + 1).integers(0, a.vocab, a.prompt_len)])
    rows, gen, logpos = run_engine(pathlib.Path(a.exe), gguf, pack, tokens, a.max_new, work, a.extra.split())
    fed = tokens + gen[:-1]                     # row p is the logits after reading fed[p]
    if rows.shape[0] != len(fed):
        raise SystemExit("the dump has %d rows, expected %d" % (rows.shape[0], len(fed)))
    converted = C.convert_globals(src, g)       # the file's tensors again (Q8_0 is deterministic), dequantized
    for l in range(g.n_layers):
        converted.update(C.convert_layer(src, g, l))
    W = {n: t.values() for n, t in converted.items()}
    result = {"tokens": tokens, "generated": gen, "geometry": {"layers": a.layers, "ff": a.ff, "vocab": a.vocab}}
    ok = True
    for name, gate, act_q8 in (("silu", "silu", False), ("silu-act-q8", "silu", True), ("sigmoid", "sigmoid", False)):
        e = Engine(W, Geo.of(g), gdn_gate=gate, act_q8=act_q8)
        ref = np.array([e.step(t, p)[0] for p, t in enumerate(fed)])
        c = compare(rows, ref)
        result[name] = c
        print("%-12s rows %d  rel max %.3e mean %.3e  top-1 %d/%d  KL mean %.3e max %.3e  (ref logit std %.2f, "
              "entropy %.2f nats)" % (name, c["rows"], c["rel_max"], c["rel_mean"], c["top1"], c["rows"], c["kl_mean"],
                                      c["kl_max"], c["ref_logit_std"], c["ref_entropy_mean"]))
        if name == "silu-act-q8":
            # the gate: the closer of the two silu references - FP32 activations (the batched prompt path's FP16 GEMM
            # inputs are nearer it) or Q8_0-rounded ones (the verify windows' Q8_1 GEMV inputs)
            best = min((result["silu"], c), key=lambda x: x["kl_mean"])
            ok = ok and best["rel_max"] <= a.rel_gate and best["kl_max"] <= a.kl_gate
        if name == "silu":
            # the greedy picks are the reference's argmax on the engine's own prefix
            picks = [int(np.argmax(ref[len(tokens) - 1 + i])) for i in range(len(gen))]
            result["greedy_agree"] = sum(int(x == y) for x, y in zip(picks, gen))
            print("%-12s %d/%d generated tokens are the reference's argmax" % ("greedy", result["greedy_agree"], len(gen)))
        if name == "sigmoid":   # the engine must be measurably closer to silu than to the gate it replaced
            ok = ok and c["kl_mean"] > 10 * max(result["silu"]["kl_mean"], 1e-12)
    lp = [l.split("\t") for l in logpos.read_text().splitlines() if l.strip()] if logpos.exists() else []
    result["logpos_rows"] = len(lp)
    print("STRATA_LOGPOS: %d rows (teacher-forced prompt positions with a target)" % len(lp))
    ok = ok and len(lp) == len(tokens) - 1
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(result, indent=1))
    if not a.keep:
        for f in (gguf, pack / "dense.bin"):
            if f.exists():
                f.unlink()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
