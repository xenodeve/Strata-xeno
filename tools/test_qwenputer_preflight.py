"""Behavior tests for the task-owned P0 inventory; no GPU/downloads."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "qwenputer_preflight", Path(__file__).parent / "qwenputer" / "preflight.py")
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class PreflightTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_hash_is_the_bytes_digest_not_a_path_or_timestamp(self):
        p = self.root / "small.bin"
        p.write_bytes(b"abc")
        self.assertEqual(preflight.sha256_file(p),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_native_and_ple_missing_are_both_reported_without_loading(self):
        path = self.root / "missing.gguf"
        cfg = self.root / "config.json"
        cfg.write_text(json.dumps({"exe": str(path), "tokenizer": str(self.root / "tok"),
                                   "args": ["--native", str(path), "--ple-gguf", str(path)]}))
        rows = {x["role"]: x for x in preflight.collect_inputs(cfg)["inputs"]}
        self.assertIn("native", rows)
        self.assertIn("ple", rows)
        self.assertFalse(rows["native"]["exists"])
        self.assertFalse(rows["ple"]["exists"])

    def test_directory_inputs_pin_file_bytes_and_preserve_config(self):
        pack = self.root / "pack"
        pack.mkdir()
        (pack / "weights.bin").write_bytes(b"abc")
        cfg = self.root / "config.json"
        before = json.dumps({"args": ["--pack", str(pack)]}).encode()
        cfg.write_bytes(before)
        rows = preflight.collect_inputs(cfg)["inputs"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["files"][0]["sha256"],
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertEqual(cfg.read_bytes(), before)

    def test_search_finds_only_requested_model_names(self):
        sub = self.root / "models"
        sub.mkdir()
        (sub / "wanted.gguf").write_bytes(b"GGUF")
        (sub / "different.gguf").write_bytes(b"GGUF")
        result = preflight.find_models([self.root], {"wanted.gguf"})
        self.assertEqual([Path(x["path"]).name for x in result["matches"]], ["wanted.gguf"])

    def test_search_records_inaccessible_or_missing_root(self):
        result = preflight.find_models([self.root / "absent"], {"wanted.gguf"})
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(result["errors"][0]["root"], str(self.root / "absent"))


if __name__ == "__main__":
    unittest.main()
