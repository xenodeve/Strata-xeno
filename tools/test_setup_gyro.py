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
        self.assertEqual(fam["smaller"], "Gyro-S, --model S")
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

    def test_both_sizes_are_regular_choices_m_first(self):
        """Gyro-S runs on agentionai's rc2 code like Gyro-M (only Gyro-M's TQ2_T decode tuning is not its): neither is
        experimental, and the menu lists M first."""
        self.assertFalse(setup.MODELS["S"].get("experimental"))
        self.assertFalse(setup.MODELS["M"].get("experimental"))
        names = [m for m in setup.MODELS if "gyro" in setup.MODELS[m].get("families", ("qwen", "swift"))]
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
                                   "ggml": setup.AGENTION_GGML})
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
                                   "ggml": setup.AGENTION_GGML})
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
                                         "ggml": setup.AGENTION_GGML})
            got, out = quiet(setup.build_engine, {"arch": "120"}, "none", True, Path("x"), toolkit=13, gyro=True)
            self.assertEqual(got, eng)
            self.assertIn("already built", out)
            with self.assertRaises(SystemExit):
                quiet(setup.build_engine, {"arch": "120"}, "none", True, Path("x"), toolkit=12, gyro=True)


class FakeGGUF:
    """gguf_reader.GGUFFile: a Gyro file holds the experts and the PLE table in one shard."""
    def __init__(self, path):
        import types
        self.tensors = [types.SimpleNamespace(name=n) for n in ("blk.0.ffn_up_exps.weight", "per_layer_token_embd.weight")]


class Install(unittest.TestCase):
    """setup.main() for --family gyro, every outside effect mocked (no download, no compile)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "data" / "mtp" / "rt").mkdir(parents=True)
        (self.t / "data" / "mtp" / "rt" / "experts.bin").write_bytes(b"")
        self.downloads, self.runs, self.verified, self.builds = [], [], [], []

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, argv, ram=63.7, answers=None, model="M", amd=(), family=True):
        import types
        self.asked = []

        def fake_input(prompt=""):
            if answers is None:
                raise AssertionError(f"asked {prompt!r}")
            self.asked.append(prompt)
            return next((v for k, v in answers.items() if k in prompt), "")

        found = [{"index": 0, "name": "NVIDIA GeForce RTX 5070", "vram_gb": 11.9, "arch": "120", "driver": "580.97"}]

        def fake_download(url, dst, what=None):
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        def fake_build(gpu, vision, yes, llama, toolkit=None, gyro=False):
            self.builds.append({"llama": llama, "toolkit": toolkit, "gyro": gyro, "vision": vision})
            eng = setup.engine_dir(toolkit or 13, gyro)
            eng.mkdir(exist_ok=True)
            (eng / setup.EXE).write_bytes(b"exe")
            (eng / "BUILD.json").write_text(json.dumps({"source": "local", "version": "0.1.42", "archs": [120],
                                                        "cuda_dirs": ["C:/cuda/bin"]}))
            return eng

        def fake_run(cmd, cwd=None, env=None, check=True, quiet=False):
            self.runs.append(([str(x) for x in cmd], dict(env or {})))

        def no_prebuilt(*a, **k):
            raise AssertionError("a ready-made engine was fetched for Gyro")

        patches = [
            mock.patch.object(setup, "ROOT", self.t),
            mock.patch.object(setup, "GPU_PICK", None),
            mock.patch.object(setup, "data_folder", lambda d: (self.t / "data", [])),
            mock.patch.object(setup, "installed_configs", lambda: []),
            mock.patch.object(setup, "gpus", lambda: found),
            mock.patch.object(setup, "amd_gpus", lambda *a: list(amd)),
            mock.patch.object(setup, "ram_gb", lambda: ram),
            mock.patch.object(setup, "cpu_info", lambda: ("Test CPU", True, True)),
            mock.patch.object(setup, "cpu_cores", lambda: None),
            mock.patch.object(setup, "page_file_gb", lambda: 16.0),
            mock.patch.object(setup, "free_gb", lambda p: 500.0),
            mock.patch.object(setup, "rotational_disk", lambda p: None),
            mock.patch.object(setup, "is_wsl", lambda: False),
            mock.patch.object(setup, "pip_install", lambda *a, **k: None),
            mock.patch.object(setup, "get_llama_cpp",
                              lambda agention=False: self.t / ("llama.cpp-agention" if agention else "llama.cpp")),
            mock.patch.object(setup, "get_prebuilt", no_prebuilt),
            mock.patch.object(setup, "build_engine", fake_build),
            mock.patch.object(setup, "download", fake_download),
            mock.patch.object(setup, "check_shards", lambda shards: None),
            mock.patch.object(setup, "verify_sha256", lambda s, size, sha: self.verified.append((s.name, size, sha))),
            mock.patch.object(setup, "run", fake_run),
            mock.patch.object(setup, "refresh_draft_vocab", lambda *a, **k: None),
            mock.patch.object(setup, "write_run_script", lambda tag, cfg, port, *_: self.t / f"start-{tag}.bat"),
            mock.patch.object(setup, "saved_calibration", lambda cfg: None),
            mock.patch.object(setup, "start", mock.Mock(side_effect=AssertionError("started"))),
            mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=FakeGGUF)}),
            mock.patch.object(sys, "argv", ["setup.py", *(["--family", "gyro"] if family else []),
                                            *(["--model", model] if model else []),
                                            *(["--yes"] if answers is None else []), "--no-start",
                                            "--models-dir", str(self.t / "models"), *argv]),
            mock.patch("builtins.input", fake_input),
        ]
        with contextlib.ExitStack() as st:
            for p in patches:
                st.enter_context(p)
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    code = setup.main()
            except SystemExit as e:                    # setup stopped (fail): its exit code
                code = e.code
            out = buf.getvalue()
        cfg = self.t / f"strata-gyro-{(model or 'm').lower()}.json"
        return code, out, (json.loads(cfg.read_text()) if cfg.exists() else None)

    def test_install_gyro_m(self):
        code, out, cfg = self.main(["--context", "65536"])
        self.assertEqual(code, 0, out)
        # one GGUF from the pinned revision, its size and SHA-256 checked; no image encoder
        self.assertEqual(self.downloads, [setup.FAMILIES["gyro"]["hf"] + M_FILE])
        self.assertEqual(self.verified, [(M_FILE, *setup.GYRO_FILES[M_FILE])])
        # the engine compiled with agentionai's tree into engine-gyro/ (no ready-made engine)
        self.assertEqual(len(self.builds), 1)
        self.assertTrue(self.builds[0]["gyro"])
        self.assertEqual(self.builds[0]["llama"].name, "llama.cpp-agention")
        self.assertEqual(self.builds[0]["vision"], "none")
        self.assertEqual(Path(cfg["exe"]).parent.name, "engine-gyro")
        self.assertTrue(setup.config_gyro(cfg))
        self.assertEqual(cfg["lib_dirs"], ["C:/cuda/bin"])
        # the pack with --compat-bf16, through agentionai's gguf-py
        packs = [(c, e) for c, e in self.runs if c[1].endswith("iq_pack.py")]
        self.assertEqual(len(packs), 1, self.runs)
        cmd, env = packs[0]
        self.assertIn("--compat-bf16", cmd)
        self.assertNotIn("--experts-bin", cmd)
        self.assertTrue(cmd[cmd.index("--gguf") + 1].endswith(M_FILE))
        self.assertTrue(cmd[cmd.index("--out") + 1].replace("\\", "/").endswith("packs/gyro-m"))
        self.assertTrue(env["STRATA_GGUF_PY"].replace("\\", "/").endswith("llama.cpp-agention/gguf-py"))
        # the config: the same file as --native and --ple-gguf, no low-RAM flags, no images
        args = cfg["args"]
        self.assertTrue(args[args.index("--native") + 1].endswith(M_FILE))
        self.assertEqual(args[args.index("--ple-gguf") + 1], args[args.index("--native") + 1])
        for flag in ("--mmap-experts", "--resident-experts", "--vision", "--resident-budget-gib"):
            self.assertNotIn(flag, args)
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-gyro-m")
        self.assertNotIn("vision", cfg)

    def test_gyro_m_is_labelled_recommended(self):
        """The developer's label (2026-10-14): Gyro-M's quality is close to ISTA's IQ3_S at a smaller size."""
        self.assertTrue(setup.MODELS["M"].get("recommended"))
        self.assertFalse(setup.MODELS["S"].get("recommended"))
        self.assertEqual(setup.FAMILIES["gyro"].get("recommended"), "Gyro-M")
        _, out, _ = self.main([], answers={"Context": "1", "Which model?": str(list(setup.FAMILIES).index("gyro") + 1)},
                              model=None, family=False)
        self.assertRegex(out, r"Qwen3\.8-Flash-Next \(Gyro\).*\[recommended: Gyro-M\]")
        size_lines = [ln for ln in out.splitlines() if ln.strip().startswith(("1) M ", "2) S "))]
        self.assertEqual(len(size_lines), 2, out)
        self.assertTrue(size_lines[0].rstrip().endswith("[recommended]"))
        self.assertNotIn("[recommended]", size_lines[1])

    def test_the_sizes_say_what_they_compare_with(self):
        self.assertIn("IQ3_S", setup.MODELS["M"]["about"])     # the developer's tests
        self.assertIn("IQ2_XS", setup.MODELS["S"]["about"])    # agentionai's model card (KLD, top-1)

    def test_the_default_size_follows_the_ram(self):
        """Gyro-M from its measured 48 GB of RAM, Gyro-S below it."""
        self.main([], ram=32.0, answers={"Context": "1"}, model=None)
        self.assertEqual([q for q in self.asked if q.startswith("Which size?")], ["Which size? [2]: "])   # 2 = S
        self.asked = []
        self.main([], ram=63.7, answers={"Context": "1"}, model=None)
        self.assertEqual([q for q in self.asked if q.startswith("Which size?")], ["Which size? [1]: "])   # 1 = M

    def test_amd_is_refused_before_any_download(self):
        amd = [{"index": 0, "name": "AMD Radeon RX 9070 XT", "vram_gb": 16.0, "arch": "gfx1201", "count": 1}]
        with mock.patch.object(setup, "amd_problem", lambda g: None):
            code, out, cfg = self.main(["--backend", "hip"], amd=amd)
        self.assertNotEqual(code, 0)
        self.assertIn("needs an NVIDIA GPU in setup", out)
        self.assertEqual(self.downloads, [])
        self.assertIsNone(cfg)


if __name__ == "__main__":
    unittest.main()
