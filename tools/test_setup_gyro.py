"""Tests for setup.py's Gyro family (xeno, docs/GYRO.md): agentionai's Gyro-S / Gyro-M files at a pinned revision
with their sizes and SHA-256, no low-RAM mode and no images, the engine compiled with agentionai's ggml in its own
folder (engine-gyro/) and recompiled there after an update, never a ready-made engine.  Nothing is downloaded or
compiled.

    python -m unittest tools.test_setup_gyro
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

M_FILE = "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf"
S_FILE = "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf"


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class Registry(unittest.TestCase):
    def test_family_and_sizes(self):
        fam = setup.FAMILIES["gyro"]
        self.assertEqual(fam["tag"], "gyro-")
        self.assertEqual(fam["engine"], "gyro")
        names = [m for m in setup.MODELS if "gyro" in setup.MODELS[m].get("families", ("qwen", "swift"))]
        self.assertEqual(sorted(names), ["M", "S"])
        # the names a pack, config and start script get, as tools/gyro_setup.sh and the hub use them
        self.assertEqual((fam["tag"] + "M").lower(), "gyro-m")
        self.assertEqual((fam["tag"] + "S").lower(), "gyro-s")

    def test_files_are_one_shard_each(self):
        fam = setup.FAMILIES["gyro"]
        self.assertEqual(setup.model_file(fam, "M", 1), M_FILE)
        self.assertEqual(setup.model_file(fam, "S", 1), S_FILE)
        self.assertEqual(setup.model_shards(fam, "M"), 1)
        self.assertEqual(setup.model_shards(fam, "S"), 1)

    def test_pinned_revision_sizes_and_hashes(self):
        fam = setup.FAMILIES["gyro"]
        self.assertIn("/resolve/8fa6f45c64be05fc7e629749dcb6660e7d7acb0f/", fam["hf"])
        for name, model in ((M_FILE, "M"), (S_FILE, "S")):
            size, sha = fam["sha256"][name]
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(size / 1e9, setup.MODELS[model]["download_gb"], delta=0.1)

    def test_a_gyro_file_is_recognized_not_refused(self):
        self.assertEqual(setup.gguf_choice(M_FILE), ("gyro", "M"))
        self.assertEqual(setup.gguf_choice(S_FILE), ("gyro", "S"))
        self.assertIsNone(setup.gguf_unsupported(M_FILE))
        self.assertIsNone(setup.gguf_unsupported(S_FILE))
        self.assertIn("Gyro-S / Gyro-M", setup.SUPPORTED_GGUFS)

    def test_no_low_ram_mode_and_no_images(self):
        for model in ("M", "S"):
            self.assertFalse(setup.low_ram_wanted(model, 24))
            self.assertFalse(setup.low_ram_wanted(model, 24, "on"))
            self.assertFalse(setup.low_ram_fits(model, 24, 16))
            self.assertIs(setup.MODELS[model].get("vision", setup.FAMILIES["gyro"].get("vision")), False)
            self.assertTrue(setup.MODELS[model].get("nvidia_only"))

    def test_s_is_experimental_and_sorts_after_m(self):
        self.assertTrue(setup.MODELS["S"].get("experimental"))
        self.assertFalse(setup.MODELS["M"].get("experimental"))
        names = ["S", "M"]
        names.sort(key=lambda m: bool(setup.MODELS[m].get("experimental")))   # as main() orders a family's sizes
        self.assertEqual(names, ["M", "S"])

    def test_agentionai_llama_cpp_is_pinned(self):
        self.assertRegex(setup.AGENTION_LLAMA_COMMIT, r"^[0-9a-f]{40}$")
        self.assertIn(setup.AGENTION_LLAMA_COMMIT, setup.AGENTION_LLAMA_ZIP)
        self.assertTrue(setup.AGENTION_LLAMA_ZIP.startswith("https://github.com/agentionai/llama.cpp/archive/"))


class EngineFolder(unittest.TestCase):
    def test_engine_dir_and_config(self):
        self.assertEqual(setup.engine_dir(gyro=True).name, "engine-gyro")
        self.assertEqual(setup.engine_dir(13).name, "engine")
        self.assertEqual(setup.engine_dir(12).name, setup.ENGINE12_DIR)
        self.assertTrue(setup.config_gyro({"exe": str(setup.ROOT / "engine-gyro" / setup.EXE)}))
        self.assertFalse(setup.config_gyro({"exe": str(setup.ROOT / "engine" / setup.EXE)}))
        self.assertFalse(setup.config_gyro({}))

    def test_get_llama_cpp_keeps_the_trees_apart(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)):
            for sub in ("llama.cpp", "llama.cpp-agention"):
                (Path(d) / "third_party" / sub / "ggml").mkdir(parents=True)
                (Path(d) / "third_party" / sub / "ggml" / "CMakeLists.txt").write_text("")
                (Path(d) / "third_party" / sub / "gguf-py").mkdir()
            self.assertEqual(setup.get_llama_cpp().name, "llama.cpp")
            self.assertEqual(setup.get_llama_cpp(agention=True).name, "llama.cpp-agention")

    def _engine(self, d: Path, meta: dict):
        eng = d / "engine-gyro"
        eng.mkdir()
        (eng / setup.EXE).write_bytes(b"exe")
        (eng / "BUILD.json").write_text(json.dumps(meta), encoding="utf-8")
        return eng

    def test_update_recompiles_with_agentionai_ggml_after_a_pull(self):
        calls = []
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)), \
                mock.patch.object(setup, "source_hash", lambda parts: "new"), \
                mock.patch.object(setup, "update_engine_gpu", lambda tk=13: {"arch": "120", "index": 0}), \
                mock.patch.object(setup, "get_llama_cpp", lambda agention=False: Path("agention" if agention else "stock")), \
                mock.patch.object(setup, "build_engine", lambda *a, **k: calls.append((a, k))):
            self._engine(Path(d), {"source": "local", "src": "old", "archs": [89],
                                   "ggml": f"agentionai/llama.cpp@{setup.AGENTION_LLAMA_COMMIT[:7]}"})
            quiet(setup.update_gyro_engine)
        self.assertEqual(len(calls), 1)
        args, kw = calls[0]
        self.assertEqual(args[3], Path("agention"))
        self.assertEqual(kw, {"toolkit": 13, "gyro": True})
        self.assertEqual(args[0]["archs"], [89, 120])

    def test_update_leaves_an_up_to_date_engine_alone(self):
        calls = []
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)), \
                mock.patch.object(setup, "source_hash", lambda parts: "same"), \
                mock.patch.object(setup, "build_engine", lambda *a, **k: calls.append((a, k))):
            self._engine(Path(d), {"source": "local", "src": "same",
                                   "ggml": f"agentionai/llama.cpp@{setup.AGENTION_LLAMA_COMMIT[:7]}"})
            quiet(setup.update_gyro_engine)
        self.assertEqual(calls, [])

    def test_a_new_ggml_pin_recompiles(self):
        calls = []
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)), \
                mock.patch.object(setup, "source_hash", lambda parts: "same"), \
                mock.patch.object(setup, "update_engine_gpu", lambda tk=13: {"arch": "89", "index": 0}), \
                mock.patch.object(setup, "get_llama_cpp", lambda agention=False: Path("agention")), \
                mock.patch.object(setup, "build_engine", lambda *a, **k: calls.append((a, k))):
            self._engine(Path(d), {"source": "local", "src": "same", "ggml": "agentionai/llama.cpp@0000000"})
            quiet(setup.update_gyro_engine)
        self.assertEqual(len(calls), 1)

    def test_update_installed_engine_visits_the_gyro_engine(self):
        seen = []
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)), \
                mock.patch.object(setup, "update_gyro_engine", lambda: seen.append("gyro")):
            self._engine(Path(d), {"source": "local", "src": "x"})
            quiet(setup.update_installed_engine, "https://example.com/")   # no engine/ here: only the Gyro one
        self.assertEqual(seen, ["gyro"])

    def test_build_engine_reuses_a_matching_gyro_engine_and_refuses_cuda12(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)), \
                mock.patch.object(setup, "source_hash", lambda parts: "same"), \
                mock.patch.object(setup, "cpu_info", lambda: (None, True)), \
                mock.patch.object(setup, "cpu_floor", lambda avx2: ""):
            eng = self._engine(Path(d), {"source": "local", "src": "same", "vision_src": None, "archs": [120],
                                         "ggml": f"agentionai/llama.cpp@{setup.AGENTION_LLAMA_COMMIT[:7]}"})
            got, out = quiet(setup.build_engine, {"arch": "120"}, "none", True, Path("x"), toolkit=13, gyro=True)
            self.assertEqual(got, eng)
            self.assertIn("already built", out)
            with self.assertRaises(SystemExit):
                quiet(setup.build_engine, {"arch": "120"}, "none", True, Path("x"), toolkit=12, gyro=True)


if __name__ == "__main__":
    unittest.main()
