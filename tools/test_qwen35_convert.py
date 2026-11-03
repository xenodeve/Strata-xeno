"""The numeric gate for tools/qwen35_to_qwen4exp.py (#297 stage 1): converted weights, run through the engine's own math
(ref/gr.py, ref/qwen4exp_engine.py), must reproduce Qwen3.5 as Hugging Face computes it.

    python -m unittest tools.test_qwen35_convert          tiny random weights, numpy only, no download (CI)
    python tools/test_qwen35_convert.py --full [--hf-repo Qwen/Qwen3.5-4B | --hf DIR] [--cache DIR] [--json OUT]

The unit tests compare against `HfMath` below, a numpy transcription of transformers' modeling_qwen3_5 (value heads
paired by repeat_interleave, (1+w) block norms, the gated norm's plain w and silu(z), a plain residual).  When torch
and transformers are importable, one more test pins `HfMath` to transformers itself on the same tiny weights.  Each
property the conversion depends on has a negative test: a converter that skipped it must fail the comparison.

--full runs the real model: Hugging Face's own decoder layers (fp32, eager attention, the bf16 weights) against the
engine reference on the converted weights, layer by layer, so neither side holds the whole model.  Three engine
variants: `f32` (the mapping with unquantized projections - must agree to float rounding, or the mapping is wrong),
`q8` (the Q8_0 file, the gate) and `q8-sigmoid` (today's kernels, with sigmoid(z) as the GDN output gate).  It
reports, per layer, the error of that layer alone (fed HF's own input) and of the stack, then the logits' top-1
agreement.  It needs torch and transformers >= 5 (qwen3_5).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile
import time
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "ref"))
import qwen35_to_qwen4exp as C  # noqa: E402
from gr import bf16_round  # noqa: E402
from qwen4exp_engine import Engine, Geo  # noqa: E402

P = C.HF_PREFIX


# ------------------------------------------------------------------ Hugging Face's math, in numpy
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sig(x)


def _rms1(x, w, eps):
    """Qwen3_5RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w)."""
    return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * (1.0 + w)


class HfMath:
    """modeling_qwen3_5's text model, one token at a time (the recurrent form of the gated delta rule)."""

    def __init__(self, get, t: dict):
        self.get, self.t = get, t
        rope = t["rope_parameters"]
        self.D = t["head_dim"]
        self.n_rot = int(self.D * rope["partial_rotary_factor"])
        self.inv = 1.0 / (rope["rope_theta"] ** (np.arange(0, self.n_rot, 2, dtype=np.float64) / self.n_rot))
        n = t["num_hidden_layers"]
        self.conv = [None] * n
        self.st = [None] * n
        self.K = [[] for _ in range(n)]
        self.V = [[] for _ in range(n)]

    def w(self, l, s):
        return self.get("%slayers.%d.%s" % (P, l, s))

    def gdn(self, l, x):
        t = self.t
        hk, hv, S = t["linear_num_key_heads"], t["linear_num_value_heads"], t["linear_key_head_dim"]
        qkv = self.w(l, "linear_attn.in_proj_qkv.weight") @ x
        z = self.w(l, "linear_attn.in_proj_z.weight") @ x
        b = self.w(l, "linear_attn.in_proj_b.weight") @ x
        a = self.w(l, "linear_attn.in_proj_a.weight") @ x
        cw = self.w(l, "linear_attn.conv1d.weight")[:, 0, :]          # (C, d_conv)
        dc = cw.shape[1]
        if self.conv[l] is None:
            self.conv[l] = np.zeros((dc - 1, qkv.size))
        buf = np.concatenate([self.conv[l], qkv[None]], 0)
        self.conv[l] = buf[1:]
        h = _silu(np.einsum("ic,ci->c", buf, cw))
        q, k, v = h[:hk * S].reshape(hk, S), h[hk * S:2 * hk * S].reshape(hk, S), h[2 * hk * S:].reshape(hv, S)
        beta = _sig(b)
        g = -np.exp(self.w(l, "linear_attn.A_log")) * np.log1p(np.exp(a + self.w(l, "linear_attn.dt_bias")))
        r = hv // hk
        q, k = np.repeat(q, r, 0), np.repeat(k, r, 0)                 # repeat_interleave: head h reads h // r
        q = q / np.sqrt(np.sum(q * q, -1, keepdims=True) + 1e-6) / np.sqrt(S)
        k = k / np.sqrt(np.sum(k * k, -1, keepdims=True) + 1e-6)
        st = np.zeros((hv, S, S)) if self.st[l] is None else self.st[l]
        st = st * np.exp(g)[:, None, None]
        mem = np.einsum("hij,hi->hj", st, k)
        st = st + k[:, :, None] * ((v - mem) * beta[:, None])[:, None, :]
        self.st[l] = st
        o = np.einsum("hij,hi->hj", st, q)
        y = o / np.sqrt(np.mean(o * o, -1, keepdims=True) + t["rms_norm_eps"]) * self.w(l, "linear_attn.norm.weight")
        y = y * _silu(z.reshape(hv, S))
        return self.w(l, "linear_attn.out_proj.weight") @ y.reshape(-1)

    def rope(self, x, pos):
        half = self.n_rot // 2
        f = pos * self.inv
        c, s = np.cos(f), np.sin(f)
        y = x.copy()
        a, b = x[:, :half], x[:, half:self.n_rot]
        y[:, :half], y[:, half:self.n_rot] = a * c - b * s, b * c + a * s
        return y

    def attn(self, l, x, pos):
        t, D, eps = self.t, self.D, self.t["rms_norm_eps"]
        nh, nkv = t["num_attention_heads"], t["num_key_value_heads"]
        qg = (self.w(l, "self_attn.q_proj.weight") @ x).reshape(nh, 2 * D)
        q, gate = qg[:, :D], qg[:, D:].reshape(-1)
        q = self.rope(_rms1(q, self.w(l, "self_attn.q_norm.weight"), eps), pos)
        k = self.rope(_rms1((self.w(l, "self_attn.k_proj.weight") @ x).reshape(nkv, D),
                            self.w(l, "self_attn.k_norm.weight"), eps), pos)
        v = (self.w(l, "self_attn.v_proj.weight") @ x).reshape(nkv, D)
        self.K[l].append(k)
        self.V[l].append(v)
        K, V = np.stack(self.K[l], 1), np.stack(self.V[l], 1)
        kv = np.arange(nh) // (nh // nkv)
        sc = np.einsum("hd,htd->ht", q, K[kv]) / np.sqrt(D)
        pr = np.exp(sc - sc.max(1, keepdims=True))
        pr /= pr.sum(1, keepdims=True)
        o = np.einsum("ht,htd->hd", pr, V[kv]).reshape(-1)
        return self.w(l, "self_attn.o_proj.weight") @ (o * _sig(gate))

    def layer(self, l, h, pos):
        t, eps = self.t, self.t["rms_norm_eps"]
        x = _rms1(h, self.w(l, "input_layernorm.weight"), eps)
        h = h + (self.attn(l, x, pos) if t["layer_types"][l] == "full_attention" else self.gdn(l, x))
        x = _rms1(h, self.w(l, "post_attention_layernorm.weight"), eps)
        return h + self.w(l, "mlp.down_proj.weight") @ (_silu(self.w(l, "mlp.gate_proj.weight") @ x) *
                                                          (self.w(l, "mlp.up_proj.weight") @ x))

    def run(self, tokens):
        E = self.get(P + "embed_tokens.weight")
        out, hs = [], []
        for pos, tok in enumerate(tokens):
            h = E[tok].astype(np.float64)
            trace = []
            for l in range(self.t["num_hidden_layers"]):
                h = self.layer(l, h, pos)
                trace.append(h)
            hs.append(trace)
            out.append(E @ _rms1(h, self.get(P + "norm.weight"), self.t["rms_norm_eps"]))
        return np.array(out), hs


# ------------------------------------------------------------------ tiny model
def tiny_config(hc_lr=8):
    t = {"hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
         "head_dim": 32, "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16,
         "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4, "intermediate_size": 96, "vocab_size": 288,
         "rms_norm_eps": 1e-6, "max_position_embeddings": 4096, "full_attention_interval": 4,
         "layer_types": ["linear_attention"] * 3 + ["full_attention"], "hidden_act": "silu",
         "attention_bias": False, "attn_output_gate": True, "tie_word_embeddings": True,
         "rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                             "mrope_section": [1, 1, 2], "mrope_interleaved": True}}
    return {"text_config": t}


def tiny_weights(cfg, seed=0) -> dict:
    """Random weights, rounded to bf16 as the real checkpoint is; norms around their real centres.  The projections
    are kept small: at 3x these scales the random model amplifies a 0.3 % weight error into 20 % at layer 0 (HfMath
    itself, fed the Q8_0-rounded embedding, moves as much), and the Q8_0 test would measure that, not the file."""
    t = cfg["text_config"]
    rng = np.random.default_rng(seed)
    E, F, V = t["hidden_size"], t["intermediate_size"], t["vocab_size"]
    hk, hv, S = t["linear_num_key_heads"], t["linear_num_value_heads"], t["linear_key_head_dim"]
    nh, nkv, D = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    Cc = 2 * hk * S + hv * S
    b = lambda *s, sc=0.045: bf16_round((rng.standard_normal(s) * sc).astype(np.float32))  # noqa: E731
    w = {P + "embed_tokens.weight": b(V, E, sc=0.5), P + "norm.weight": b(E, sc=0.3)}
    for l in range(t["num_hidden_layers"]):
        p = "%slayers.%d." % (P, l)
        w[p + "input_layernorm.weight"] = b(E, sc=0.3)
        w[p + "post_attention_layernorm.weight"] = b(E, sc=0.3)
        w[p + "mlp.gate_proj.weight"] = b(F, E)
        w[p + "mlp.up_proj.weight"] = b(F, E)
        w[p + "mlp.down_proj.weight"] = b(E, F)
        if t["layer_types"][l] == "full_attention":
            w[p + "self_attn.q_proj.weight"] = b(2 * nh * D, E)
            w[p + "self_attn.k_proj.weight"] = b(nkv * D, E)
            w[p + "self_attn.v_proj.weight"] = b(nkv * D, E)
            w[p + "self_attn.o_proj.weight"] = b(E, nh * D)
            w[p + "self_attn.q_norm.weight"] = b(D, sc=0.3)
            w[p + "self_attn.k_norm.weight"] = b(D, sc=0.3)
        else:
            w[p + "linear_attn.in_proj_qkv.weight"] = b(Cc, E, sc=0.09)
            w[p + "linear_attn.in_proj_z.weight"] = b(hv * S, E, sc=0.09)
            w[p + "linear_attn.in_proj_a.weight"] = b(hv, E, sc=0.09)
            w[p + "linear_attn.in_proj_b.weight"] = b(hv, E, sc=0.09)
            w[p + "linear_attn.out_proj.weight"] = b(E, hv * S)
            w[p + "linear_attn.conv1d.weight"] = b(Cc, 1, 4, sc=0.4)
            w[p + "linear_attn.A_log"] = bf16_round(np.log(rng.uniform(0.5, 8, hv)).astype(np.float32))
            w[p + "linear_attn.dt_bias"] = b(hv, sc=1.0)
            w[p + "linear_attn.norm.weight"] = bf16_round((1 + rng.standard_normal(S) * 0.1).astype(np.float32))
    return w


class DictSource:
    """SafetensorsSource's interface over a dict of float32 arrays."""

    def __init__(self, w: dict):
        self.w = w
        self.where = {k: None for k in w}

    def has(self, name):
        return name in self.w

    def get(self, name):
        return self.w[name].astype(np.float32)


def convert_all(src, g, proj="Q8_0") -> dict:
    out = C.convert_globals(src, g, proj)
    for l in range(g.n_layers):
        out.update(C.convert_layer(src, g, l, proj))
    return out


def engine_weights(tensors: dict) -> dict:
    return {n: t.values() for n, t in tensors.items()}


def run_engine(W, g, tokens, gdn_gate="silu", **kw):
    e = Engine(W, Geo.of(g), gdn_gate=gdn_gate, **kw)
    logits, traces = [], []
    for pos, tok in enumerate(tokens):
        lg, tr = e.step(tok, pos)
        logits.append(lg)
        traces.append(tr)
    return np.array(logits), traces


def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


TOKENS = [5, 77, 200, 3, 3, 41, 150, 287, 9, 64, 120, 33]


class TinyConversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = tiny_config()
        cls.g = C.Geometry.from_config(cls.cfg, hc=4, hc_lr=8)
        cls.hfw = tiny_weights(cls.cfg)
        cls.src = DictSource(cls.hfw)
        cls.hf_logits, cls.hf_hs = HfMath(lambda n: cls.hfw[n].astype(np.float64), cls.cfg["text_config"]).run(TOKENS)
        cls.f32 = convert_all(cls.src, cls.g, "F32")
        cls.q8 = convert_all(cls.src, cls.g, "Q8_0")

    # ---- the encodings
    def test_v_head_order_is_a_bijection_that_pairs_like_hf(self):
        g = self.g
        order = C.v_head_order(g)
        self.assertEqual(sorted(order.tolist()), list(range(g.ssm_v_heads)))
        r = g.ssm_v_heads // g.ssm_k_heads
        for e, h in enumerate(order):
            self.assertEqual(e % g.ssm_k_heads, h // r)          # engine's key head == HF's key head
        real = C.Geometry.from_config({"hidden_size": 2560, "num_hidden_layers": 1, "layer_types": ["linear_attention"],
                                       "num_attention_heads": 16, "num_key_value_heads": 4, "head_dim": 256,
                                       "linear_num_key_heads": 16, "linear_num_value_heads": 32,
                                       "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                                       "linear_conv_kernel_dim": 4, "intermediate_size": 9216, "vocab_size": 248320,
                                       "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 1e7,
                                                                                  "partial_rotary_factor": 0.25}})
        self.assertEqual(C.v_head_order(real)[:4].tolist(), [0, 2, 4, 6])
        self.assertEqual(C.v_head_order(real)[16:18].tolist(), [1, 3])

    def test_q8_0_is_ggml_reference_and_x2_is_exact(self):
        rng = np.random.default_rng(3)
        w = bf16_round(rng.standard_normal((48, 96)).astype(np.float32) * 0.05)
        raw = C.quantize_q8_0(w)
        back = C.dequantize_q8_0(raw)
        self.assertLess(np.max(np.abs(back - w.reshape(-1))), np.max(np.abs(w)) / 127 * 0.51 + 1e-7)
        # doubling the stored fp16 scales == quantizing 2w (same codes, scales exactly x2)
        self.assertEqual(C.quantize_q8_0(w, scale=2.0), C.quantize_q8_0(2 * w))
        self.assertTrue(np.array_equal(C.dequantize_q8_0(C.quantize_q8_0(w, scale=2.0)), 2 * back))
        try:
            from _paths import gguf_py
            sys.path.insert(0, gguf_py())
        except SystemExit:
            # a checkout nested in the development tree (.claude/worktrees/x): the outer checkout's third_party/
            sys.path.append(str(ROOT.parents[2] / "third_party" / "llama.cpp" / "gguf-py"))
        try:
            from gguf import GGMLQuantizationType as Q, quants
        except ImportError:
            self.skipTest("gguf-py not found; the reference quantizer is checked where it is")
        self.assertEqual(quants.quantize(w, Q.Q8_0).tobytes(), raw)

    def test_every_tensor_matches_the_plan(self):
        rows = C.plan(self.g)
        self.assertEqual(sorted(self.q8), sorted(n for n, _, _ in rows))
        C.check_against_plan(self.q8, rows)
        names = set(self.q8)
        for bad in ("ffn_gate_inp.weight", "indexer.q_proj.weight", "ffn_gate_exps.weight"):
            self.assertFalse(any(n.endswith(bad) for n in names), bad)

    def test_gguf_and_pack_round_trip(self):
        import gguf_reader
        tok = {"tokens": [chr(0x100 + i) for i in range(self.g.vocab)], "token_type": [1] * self.g.vocab,
               "merges": [], "eos": 2, "padding": 1, "bos": 1, "chat_template": "{{ messages }}"}
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "tiny.gguf"
            w = C.StreamingGGUF()
            for k, t, v in C.metadata(self.g, "qwen35dense", tok, "test"):
                w.add(k, v, t)
            rows = C.plan(self.g)
            w.write_stream(path, rows, C.groups(self.src, self.g, rows, log=lambda *_: None))
            gf = gguf_reader.GGUFFile(path)
            self.assertEqual(gf.metadata["general.architecture"], "qwen35dense")
            self.assertEqual(gf.metadata["qwen35dense.expert_count"], 0)
            self.assertEqual(gf.metadata["qwen35dense.ssm.time_step_rank"], self.g.ssm_v_heads)
            raw = path.read_bytes()
            for t in gf.tensors:
                want = self.q8[t.name]
                self.assertEqual((t.type_name, list(t.shape)), (want.type_name, want.shape), t.name)
                at = gf.data_start + t.offset
                self.assertEqual(raw[at:at + t.expected_bytes()], want.data, t.name)
            pack = pathlib.Path(d) / "pack"
            self.assertEqual(C.write_pack(path, pack), 0)
            idx = {ln.split()[0]: ln.split() for ln in (pack / "index.txt").read_text().splitlines()
                   if not ln.startswith("#")}
            self.assertEqual(len(idx), len(rows))
            self.assertEqual(idx["blk.0.attn_qkv.weight"][1:3], ["0", "0"])           # native Q8_0 row
            self.assertEqual(idx["blk.0.hc_attn_down.weight"][2], "4")                 # BF16 in dense.bin
            self.assertEqual(idx["blk.0.hc_attn_norm.weight"][2], "2")                 # F32
            self.assertEqual(idx["blk.0.ffn_gate_inp_shexp.weight"][2], "4")
            self.assertTrue((pack / "tokenizer" / "vocab.json").exists())
            self.assertFalse((pack / "native_experts.txt").exists())

    # ---- the numbers
    def test_mapping_is_exact_with_unquantized_projections(self):
        lg, tr = run_engine(engine_weights(self.f32), self.g, TOKENS)
        for pos in range(len(TOKENS)):
            for l in range(self.g.n_layers):
                R = tr[pos][l]
                self.assertTrue(np.array_equal(R, np.tile(R[0], (self.g.hc, 1))), "streams diverged")
                self.assertLess(rel(R[0], self.hf_hs[pos][l]), 2e-5, (pos, l))
        self.assertLess(rel(lg, self.hf_logits), 2e-5)

    def test_q8_0_file_tracks_hf(self):
        lg, tr = run_engine(engine_weights(self.q8), self.g, TOKENS)
        self.assertLess(rel(lg, self.hf_logits), 0.03)
        self.assertGreaterEqual((lg.argmax(1) == self.hf_logits.argmax(1)).mean(), 0.9)

    # ---- each property the conversion depends on: without it the comparison must fail
    def _broken(self, edit, gdn_gate="silu"):
        W = engine_weights(self.f32)
        W = {k: np.array(v) for k, v in W.items()}
        edit(W)
        lg, _ = run_engine(W, self.g, TOKENS, gdn_gate=gdn_gate)
        return rel(lg, self.hf_logits)

    def test_negative_without_v_head_reorder(self):
        import unittest.mock as m
        with m.patch.object(C, "v_head_order", lambda g: np.arange(g.ssm_v_heads)):
            W = engine_weights(convert_all(self.src, self.g, "F32"))
        lg, _ = run_engine(W, self.g, TOKENS)
        self.assertGreater(rel(lg, self.hf_logits), 1e-2)

    def test_negative_without_shared_expert_x2(self):
        def edit(W):
            for l in range(self.g.n_layers):
                W["blk.%d.ffn_down_shexp.weight" % l] /= 2
        self.assertGreater(self._broken(edit), 1e-2)

    def test_negative_with_todays_sigmoid_gdn_gate(self):
        self.assertGreater(self._broken(lambda W: None, gdn_gate="sigmoid"), 1e-2)

    def test_negative_hc_norm_without_x2(self):
        def edit(W):
            for l in range(self.g.n_layers):
                W["blk.%d.hc_ffn_norm.weight" % l] /= 2
        self.assertGreater(self._broken(edit), 1e-2)

    def test_negative_gated_norm_as_one_plus_w(self):
        def edit(W):
            for l in range(self.g.n_layers):
                if not self.g.is_attention(l):
                    W["blk.%d.ssm_norm.weight" % l] += 1
        self.assertGreater(self._broken(edit), 1e-2)

    def test_negative_qk_norm_without_one_plus(self):
        def edit(W):
            for l in self.g_attention():
                W["blk.%d.attn_q_norm.weight" % l] -= 1
        self.assertGreater(self._broken(edit), 1e-2)

    def g_attention(self):
        return [l for l in range(self.g.n_layers) if self.g.is_attention(l)]

    # ---- HfMath is transformers
    def test_hf_math_is_transformers(self):
        try:
            import torch
            from transformers.models.qwen3_5 import modeling_qwen3_5 as M
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        except ImportError:
            self.skipTest("torch / transformers with qwen3_5 not installed")
        t = dict(self.cfg["text_config"])
        tc = Qwen3_5TextConfig(**t)
        tc._attn_implementation = "eager"
        model = M.Qwen3_5ForCausalLM(tc).float().eval()
        sd = {}
        for k, v in self.hfw.items():
            sd[k.replace("model.language_model.", "model.")] = torch.from_numpy(np.array(v, dtype=np.float32))
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        self.assertEqual([k for k in missing if "rotary" not in k], [])
        self.assertEqual(unexpected, [])
        with torch.no_grad():
            got = model(torch.tensor([TOKENS])).logits[0].double().numpy()
        self.assertLess(rel(got, self.hf_logits), 1e-4)


# ------------------------------------------------------------------ the real model (--full)
PROMPTS = [
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
    "Solve for x: 3x + 7 = 22. First subtract 7 from both sides, which gives",
    "ภาษาไทยเป็นภาษาราชการของประเทศไทย และมีผู้พูดเป็นภาษาแม่ประมาณ",
]


def full_check(a) -> int:
    import torch
    from tokenizers import Tokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    torch.set_grad_enabled(False)
    torch.set_num_threads(a.threads)

    meta = pathlib.Path(a.hf or a.meta)
    cfg = json.loads((meta / "config.json").read_text(encoding="utf-8"))
    g = C.Geometry.from_config(cfg)
    if a.layers:
        g.n_layers = a.layers
    src = C.local_source(a.hf) if a.hf else C.remote_source(a.hf_repo, cache_dir=None)
    cached = C.remote_source(a.hf_repo, cache_dir=a.cache) if (a.hf_repo and a.cache) else src
    cache_layers = set(int(x) for x in a.cache_layers.split(",")) if a.cache_layers else set()
    tk = Tokenizer.from_file(str(meta / "tokenizer.json"))
    prompts = [tk.encode(p).ids for p in PROMPTS]
    print("prompts: %s tokens" % [len(p) for p in prompts], flush=True)

    t = dict(cfg["text_config"])
    t.pop("model_type", None)
    tc = Qwen3_5TextConfig(**t)
    tc._attn_implementation = "eager"
    rot = M.Qwen3_5TextRotaryEmbedding(tc)

    def memo(source):
        box = {}

        class S:
            where = source.where

            def has(self, n):
                return source.has(n)

            def get(self, n):
                if n not in box:
                    box[n] = source.get(n)
                return box[n]
        return S()

    t0 = time.time()
    gsrc = memo(cached)
    variants = {"f32": "F32", "q8": "Q8_0"}
    glob = {v: C.convert_globals(gsrc, g, q) for v, q in variants.items()}
    Wg = {v: {n: tt.values() for n, tt in glob[v].items()} for v in variants}
    for v in Wg:
        Wg[v]["output.weight"] = Wg[v]["token_embd.weight"]
    emb = gsrc.get(P + "embed_tokens.weight")
    hf_norm = gsrc.get(P + "norm.weight")
    print("globals converted in %.0f s" % (time.time() - t0), flush=True)

    geo = Geo.of(g)
    arms = [("f32", "silu", {}), ("q8", "silu", {}), ("q8-sigmoid", "sigmoid", {}), ("q8-actq8", "silu", {"act_q8": True})]
    if not a.all_arms:
        arms = arms[:3]
    wkey = lambda name: "f32" if name == "f32" else "q8"  # noqa: E731
    engines = {name: [Engine(dict(Wg[wkey(name)]), geo, gdn_gate=gate, **kw) for _ in prompts]
               for name, gate, kw in arms}
    hf_h = [torch.from_numpy(emb[p]).unsqueeze(0) for p in prompts]                  # (1, T, E) float32
    R = {name: [np.stack([engines[name][i].embed(tok) for tok in p]) for i, p in enumerate(prompts)]
         for name, _, _ in arms}                                                       # (T, hc, E)
    report = {"prompts": [len(p) for p in prompts], "layers": []}
    for l in range(g.n_layers):
        tl = time.time()
        lsrc = memo(cached if l in cache_layers else src)
        hf_layer = M.Qwen3_5DecoderLayer(tc, l).float().eval()
        sd = {}
        pre = "%slayers.%d." % (P, l)
        for n in lsrc.where:
            if n.startswith(pre):
                sd[n[len(pre):]] = torch.from_numpy(np.ascontiguousarray(lsrc.get(n)))
        hf_layer.load_state_dict(sd, strict=True)
        conv = {v: C.convert_layer(lsrc, g, l, q) for v, q in variants.items()}
        Wl = {v: {n: tt.values() for n, tt in conv[v].items()} for v in variants}
        row = {"layer": l, "kind": "attention" if g.is_attention(l) else "GDN"}
        new_hf = []
        for i, p in enumerate(prompts):
            T = len(p)
            pos = torch.arange(T).unsqueeze(0)
            pe = rot(hf_h[i], pos)
            mask = torch.full((1, 1, T, T), torch.finfo(torch.float32).min).triu(1)
            out = hf_layer(hf_h[i], position_embeddings=pe, attention_mask=mask, position_ids=pos)
            out = out[0] if isinstance(out, tuple) else out
            new_hf.append(out if out.dim() == 3 else out.unsqueeze(0))
        for name, gate, kw in arms:
            W = Wl[wkey(name)]
            iso_err, iso_rel, st_err, st_rel, spread = 0.0, 0.0, 0.0, 0.0, 0.0
            for i, p in enumerate(prompts):
                hin = hf_h[i][0].numpy()
                hout = new_hf[i][0].numpy() if new_hf[i].dim() == 3 else new_hf[i].numpy()
                # alone: the engine layer fed HF's own input, with fresh state
                e1 = Engine({**Wg[wkey(name)], **W}, geo, gdn_gate=gate, **kw)
                e2 = engines[name][i]
                e2.W.update(W)
                for tpos in range(len(p)):
                    out1 = e1.layer(l, np.tile(hin[tpos], (g.hc, 1)), tpos)
                    d = np.abs(out1[0].astype(np.float64) - hout[tpos])
                    iso_err = max(iso_err, float(d.max()))
                    iso_rel = max(iso_rel, float(d.max() / np.abs(hout[tpos]).max()))
                    R[name][i][tpos] = e2.layer(l, R[name][i][tpos], tpos)
                    Rt = R[name][i][tpos]
                    spread = max(spread, float(np.abs(Rt - Rt[0]).max()))
                    d = np.abs(Rt[0].astype(np.float64) - hout[tpos])
                    st_err = max(st_err, float(d.max()))
                    st_rel = max(st_rel, float(d.max() / np.abs(hout[tpos]).max()))
                for n in W:
                    e2.W.pop(n, None)
            row[name] = {"alone_max_abs": iso_err, "alone_rel": iso_rel, "stack_max_abs": st_err,
                         "stack_rel": st_rel, "stream_spread": spread}
        hf_h = [h if h.dim() == 3 else h.unsqueeze(0) for h in new_hf]
        row["hf_max_abs"] = float(max(np.abs(h.numpy()).max() for h in hf_h))
        row["seconds"] = round(time.time() - tl, 1)
        report["layers"].append(row)
        print("layer %2d %-9s |h| %8.2f  " % (l, row["kind"], row["hf_max_abs"]) + "  ".join(
            "%s alone %.2e stack %.2e" % (name, row[name]["alone_rel"], row[name]["stack_rel"]) for name, _, _ in arms)
            + "  (%.0f s)" % row["seconds"], flush=True)
        del hf_layer, sd, conv, Wl, lsrc
    if g.n_layers == len(g.layer_types):
        normed = [(_rms1(h[0].numpy().astype(np.float64), hf_norm, g.eps)) for h in hf_h]
        hf_logits = [n @ emb.T.astype(np.float64) for n in normed]
        report["logits"] = {}
        for name, _, _ in arms:
            agree, n_tok, mx, kls = 0, 0, 0.0, []
            for i, p in enumerate(prompts):
                lg = np.stack([engines[name][i].head(R[name][i][tpos]) for tpos in range(len(p))]).astype(np.float64)
                ref = hf_logits[i]
                agree += int((lg.argmax(1) == ref.argmax(1)).sum())
                n_tok += len(p)
                mx = max(mx, float(np.abs(lg - ref).max()))
                lp = ref - ref.max(1, keepdims=True)
                lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
                lq = lg - lg.max(1, keepdims=True)
                lq = lq - np.log(np.exp(lq).sum(1, keepdims=True))
                kls += (np.exp(lp) * (lp - lq)).sum(1).tolist()
            report["logits"][name] = {"top1_agree": agree, "tokens": n_tok, "max_abs": mx,
                                      "kl_mean": float(np.mean(kls)), "kl_max": float(np.max(kls))}
            print("logits %-10s top-1 %d/%d  max|d| %.3f  KL mean %.2e max %.2e" % (
                name, agree, n_tok, mx, np.mean(kls), np.max(kls)), flush=True)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("DONE %.0f s" % (time.time() - t0), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="the real model (see the module docstring)")
    ap.add_argument("--hf")
    ap.add_argument("--hf-repo", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--meta", help="--hf-repo: the directory holding config.json and tokenizer.json")
    ap.add_argument("--cache", help="--hf-repo: keep the ranges of the globals and --cache-layers here")
    ap.add_argument("--cache-layers", default="", help="comma-separated layers whose ranges go to --cache")
    ap.add_argument("--layers", type=int, default=0, help="only the first N layers (no logits)")
    ap.add_argument("--all-arms", action="store_true", help="also the Q8_0-activation arm")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json")
    a, rest = ap.parse_known_args()
    if a.full:
        if not (a.hf or a.meta):
            ap.error("--full needs --hf DIR, or --meta DIR with --hf-repo")
        return full_check(a)
    unittest.main(argv=[sys.argv[0]] + rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
