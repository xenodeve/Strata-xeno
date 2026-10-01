"""serve/test_gguf_info.py - the model's name and quantization, read from the GGUF header (xeno UI): per role (experts,
attention, embeddings, the PLE table) the type, the bytes it really takes, and the bits per weight those bytes make.

    python -m unittest serve.test_gguf_info -v
"""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import gguf_info  # noqa: E402


def s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def kv_str(k, v):
    return s(k) + struct.pack("<I", 8) + s(v)


def kv_u32(k, v):
    return s(k) + struct.pack("<I", 4) + struct.pack("<I", v)


def kv_str_array(k, vals):
    return s(k) + struct.pack("<I", 9) + struct.pack("<IQ", 8, len(vals)) + b"".join(s(v) for v in vals)


def kv_u8_array(k, n):
    return s(k) + struct.pack("<I", 9) + struct.pack("<IQ", 0, n) + bytes(n)


def make_gguf(path: Path, tensors, kvs, pad_to=32):
    """tensors: [(name, dims, ggml_type, nbytes)] laid out one after the other."""
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), len(kvs)) + b"".join(kvs)
    off, infos = 0, b""
    for name, dims, typ, nbytes in tensors:
        infos += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", typ, off)
        off += nbytes
    blob = head + infos
    blob += bytes((-len(blob)) % pad_to)
    blob += bytes(off)
    path.write_bytes(blob)


class Reads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def model(self):
        p = self.dir / "M-Q2_0-00001-of-00002.gguf"
        make_gguf(p, [
            ("token_embd.weight", [64, 100], 8, 6800),                       # Q8_0, 6400 weights
            ("blk.0.attn_q.weight", [64, 64], 1, 8192),                      # F16: 16 bits per weight
            ("blk.0.ffn_gate_exps.weight", [64, 32, 8], 42, 4096),           # Q2_0: 16384 weights, 2 bits
            ("blk.0.ssm_a", [64], 0, 256),                                   # F32
        ], [kv_str("general.architecture", "qwen3next"), kv_str("general.name", "Qwen3.8 Flash Next GSQ RCO"),
            kv_str("general.finetune", "GSQ-RCO"), kv_str("general.size_label", "80B-A3B"), kv_u32("general.file_type", 2),
            kv_str_array("tokenizer.ggml.tokens", ["a", "bb", "ccc"] * 50), kv_u8_array("tokenizer.ggml.token_type", 150)])
        return p

    def test_name_and_architecture_from_the_header(self):
        info = gguf_info.model_info([str(self.model())])
        self.assertEqual(info["name"], "Qwen3.8 Flash Next GSQ RCO")
        self.assertEqual((info["architecture"], info["finetune"], info["size_label"]), ("qwen3next", "GSQ-RCO", "80B-A3B"))
        self.assertEqual(info["files"], ["M-Q2_0-00001-of-00002.gguf"])

    def test_each_role_has_its_type_bytes_and_bits_per_weight(self):
        roles = {r["role"]: r for r in gguf_info.model_info([str(self.model())])["roles"]}
        self.assertEqual(roles["experts"]["types"], ["Q2_0"])
        self.assertAlmostEqual(roles["experts"]["bpw"], 2.0, places=2)               # 4096 B * 8 / 16384 weights
        self.assertEqual(roles["attention"]["types"], ["F16", "F32"])                  # the biggest type first: the ssm_a is F32
        self.assertAlmostEqual(roles["attention"]["bpw"], (8192 + 256) * 8 / (4096 + 64), places=2)
        self.assertEqual(roles["embeddings"]["types"], ["Q8_0"])
        self.assertEqual(roles["experts"]["tensors"], 1)
        self.assertEqual(roles["experts"]["bytes"], 4096)                            # from the offsets, not a table

    def test_variant_and_source_come_from_the_file_and_folder_names(self):
        d = self.dir / "Org-Model-GSQ" / "Q2_0"
        d.mkdir(parents=True)
        p = d / "Model-GSQ-Q2_0-00001-of-00002.gguf"
        make_gguf(p, [("blk.0.attn_q.weight", [8, 8], 1, 128)], [kv_str("general.architecture", "x")])
        info = gguf_info.model_info([str(p)])
        self.assertEqual((info["variant"], info["source"]), ("Q2_0", "Org-Model-GSQ"))
        q = self.dir / "plain.gguf"
        make_gguf(q, [("blk.0.attn_q.weight", [8, 8], 1, 128)], [kv_str("general.architecture", "x")])
        self.assertEqual((gguf_info.model_info([str(q)])["variant"], gguf_info.model_info([str(q)])["source"]), (None, None))

    def test_the_whole_model_bits_per_weight(self):
        info = gguf_info.model_info([str(self.model())])
        total_w = 6400 + 4096 + 16384 + 64
        total_b = 6800 + 8192 + 4096 + 256
        self.assertAlmostEqual(info["bpw"], total_b * 8 / total_w, places=2)
        self.assertEqual(info["bytes"], total_b)

    def test_the_second_shard_adds_the_ple_table(self):
        a = self.model()
        b = self.dir / "M-Q2_0-00002-of-00002.gguf"
        make_gguf(b, [("per_layer_token_embd.weight", [160, 1000], 20, 90000)], [kv_str("general.architecture", "qwen3next")])
        roles = {r["role"]: r for r in gguf_info.model_info([str(a), str(b)])["roles"]}
        self.assertEqual(roles["ple table"]["types"], ["IQ4_NL"])
        self.assertEqual(roles["ple table"]["bytes"], 90000)

    def test_an_unknown_type_is_named_by_its_number_not_guessed(self):
        p = self.dir / "x.gguf"
        make_gguf(p, [("blk.0.attn_q.weight", [8, 8], 99, 64)], [kv_str("general.architecture", "x")])
        roles = {r["role"]: r for r in gguf_info.model_info([str(p)])["roles"]}
        self.assertEqual(roles["attention"]["types"], ["type 99"])
        self.assertAlmostEqual(roles["attention"]["bpw"], 8.0, places=2)             # bytes and dims are known whatever the type is called

    def test_bad_files_are_none_not_a_crash(self):
        bad = self.dir / "bad.gguf"
        bad.write_bytes(b"NOTGGUF" * 10)
        self.assertIsNone(gguf_info.model_info([str(bad)]))
        self.assertIsNone(gguf_info.model_info([str(self.dir / "missing.gguf")]))
        self.assertIsNone(gguf_info.model_info([]))

    def test_model_files_come_from_the_engine_arguments(self):
        args = ["--pack", "--native", "C:\\m\\a-00001-of-00002.gguf", "--ple-gguf", "C:\\m\\a-00002-of-00002.gguf", "--x", "1"]
        self.assertEqual(gguf_info.files_from_args(args), ["C:\\m\\a-00001-of-00002.gguf", "C:\\m\\a-00002-of-00002.gguf"])
        self.assertEqual(gguf_info.files_from_args(["--pack", "pack/full"]), [])


if __name__ == "__main__":
    unittest.main()
