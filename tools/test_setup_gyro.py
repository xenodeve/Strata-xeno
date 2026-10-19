"""Tests for setup.py's Gyro family (xeno, docs/GYRO.md): agentionai's Gyro-S / Gyro-M files at a pinned revision
with their sizes and SHA-256, no low-RAM mode and no images (on the fork), the engine compiled with agentionai's ggml in its own
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
import strata_mcp as M  # noqa: E402

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
        self.assertEqual(fam["smaller"], "Gyro-S, --model Gyro-S")
        names = [m for m in setup.MODELS if "gyro" in setup.MODELS[m].get("families", ("qwen", "swift"))]
        self.assertEqual(sorted(names), ["Gyro-M", "Gyro-S"])      # one catalog since v0.2.0-beta (#268): upstream's names
        # the names a pack, config and start script get stay the fork's, as tools/gyro_setup.sh and the hub use them
        self.assertEqual(setup.install_tag(fam, "Gyro-M").lower(), "gyro-m")
        self.assertEqual(setup.install_tag(fam, "Gyro-S").lower(), "gyro-s")
        self.assertEqual(setup.install_tag(setup.FAMILIES["swift"], "IQ2_XS"), "swift-IQ2_XS")

    def test_the_fork_names_are_aliases(self):
        """--model M / S (the fork's names before v0.2.0) are Gyro-M / Gyro-S; a gyro-m config is read back as Gyro-M."""
        self.assertEqual(setup.GYRO_ALIASES, {"M": "Gyro-M", "S": "Gyro-S"})
        with tempfile.TemporaryDirectory() as t:
            for name, size in (("strata-gyro-m.json", "Gyro-M"), ("strata-gyro-s.json", "Gyro-S"),
                               ("strata-gyro-gyro-m.json", "Gyro-M")):
                (Path(t) / name).write_text(json.dumps({"args": []}))
                ch = setup.choices_from_config(Path(t) / name)
                self.assertEqual((ch["family"], ch["model"]), ("gyro", size), name)

    def test_files_are_one_shard_each(self):
        fam = setup.FAMILIES["gyro"]
        self.assertEqual(setup.model_file(fam, "Gyro-M", 1), M_FILE)
        self.assertEqual(setup.model_file(fam, "Gyro-S", 1), S_FILE)
        self.assertEqual(setup.model_shards(fam, "Gyro-M"), 1)
        self.assertEqual(setup.model_shards(fam, "Gyro-S"), 1)

    def test_pinned_revision_sizes_and_hashes(self):
        fam = setup.FAMILIES["gyro"]
        self.assertIn("/resolve/8fa6f45c64be05fc7e629749dcb6660e7d7acb0f/", fam["hf"])
        for name, model in ((M_FILE, "Gyro-M"), (S_FILE, "Gyro-S")):
            size, sha = fam["sha256"][name]
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
            self.assertAlmostEqual(size / 1e9, setup.MODELS[model]["download_gb"], delta=0.1)

    def test_a_gyro_file_is_recognized_not_refused(self):
        self.assertEqual(setup.gguf_choice(M_FILE), ("gyro", "Gyro-M"))
        self.assertEqual(setup.gguf_choice(S_FILE), ("gyro", "Gyro-S"))
        self.assertIsNone(setup.gguf_unsupported(M_FILE))
        self.assertIsNone(setup.gguf_unsupported(S_FILE))
        self.assertIn("Gyro-S / Gyro-M", setup.SUPPORTED_GGUFS)

    def test_no_low_ram_mode_and_no_images(self):
        for model in ("Gyro-M", "Gyro-S"):
            self.assertFalse(setup.low_ram_wanted(model, 24))
            self.assertFalse(setup.low_ram_wanted(model, 24, "on"))
            self.assertFalse(setup.low_ram_fits(model, 24, 16))
            self.assertIs(setup.MODELS[model].get("vision", setup.FAMILIES["gyro"].get("vision")), False)

    def test_both_sizes_are_regular_choices_m_first(self):
        """Gyro-S runs on agentionai's rc2 code like Gyro-M (only Gyro-M's TQ2_T decode tuning is not its): neither is
        experimental, and the menu lists M first."""
        self.assertFalse(setup.MODELS["Gyro-S"].get("experimental"))
        self.assertFalse(setup.MODELS["Gyro-M"].get("experimental"))
        names = [m for m in setup.MODELS if "gyro" in setup.MODELS[m].get("families", ("qwen", "swift"))]
        names.sort(key=lambda m: bool(setup.MODELS[m].get("experimental")))   # as main() orders a family's sizes
        self.assertEqual(names, ["Gyro-M", "Gyro-S"])

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
                                   "ggml": setup.AGENTION_GGML_STAMP})
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
                                   "ggml": setup.AGENTION_GGML_STAMP})
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
                                         "ggml": setup.AGENTION_GGML_STAMP})
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
            # this checkout's version (0.2.0: a Gyro size needs GYRO_ENGINE, which the fork's compiled engine is)
            (eng / "BUILD.json").write_text(json.dumps({"source": "local", "version": "0.1.45.3", "archs": [120],
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
        self.assertTrue(setup.MODELS["Gyro-M"].get("recommended"))
        self.assertFalse(setup.MODELS["Gyro-S"].get("recommended"))
        self.assertEqual(setup.FAMILIES["gyro"].get("recommended"), "Gyro-M")
        _, out, _ = self.main([], answers={"Context": "1", "Which model?": str(list(setup.FAMILIES).index("gyro") + 1)},
                              model=None, family=False)
        self.assertRegex(out, r"Qwen3\.8-Flash-Next \(Gyro\).*\[recommended: Gyro-M\]")
        size_lines = [ln for ln in out.splitlines() if ln.strip().startswith(("1) Gyro-M", "2) Gyro-S"))]
        self.assertEqual(len(size_lines), 2, out)
        self.assertTrue(size_lines[0].rstrip().endswith("[recommended]"))
        self.assertNotIn("[recommended]", size_lines[1])

    def test_the_sizes_say_what_they_compare_with(self):
        self.assertIn("IQ3_S", setup.MODELS["Gyro-M"]["about"])     # the developer's tests
        self.assertIn("IQ2_XS", setup.MODELS["Gyro-S"]["about"])    # agentionai's model card (KLD, top-1)

    def test_the_default_size_follows_the_ram(self):
        """Gyro-M from 35 GB of RAM + VRAM together (upstream 0.2.0's figure, mem_total), Gyro-S below it (the test's card
        has 11.9 GB: 20 GB of RAM makes 31.9 GB).  Step 1's RAM floor (the smallest model that is not counted as RAM +
        VRAM, the Coder's 32 GB, less 4) asks "Go on anyway?" below 28 GB of RAM, and no RAM is both >= 28 and below
        35 - 11.9 = 23.1 GB, so the Gyro-S case answers it with y (upstream's step 1, unchanged by the fork)."""
        self.main([], ram=20.0, answers={"Context": "1", "Go on anyway": "y"}, model=None)
        self.assertEqual([q for q in self.asked if q.startswith("  Go on anyway?")], ["  Go on anyway? [n]: "])
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


# Upstream v0.2.0's tests of its Gyro catalog (setup.MODELS "Gyro-S" / "Gyro-M", GYRO_ENGINE, mem_total, the MCP
# server's recommendation).  Since the merge of v0.2.0-beta (#268) it is the one catalog; the fork keeps "vision": False.


def hardware(ram, vram):
    return {"ram_gb": ram, "gpus": [{"usable": True, "vram_gb": vram, "vendor": "nvidia", "index": 0, "name": "RTX"}],
            "cpu": {"avx2": True}}


class GyroCatalog(unittest.TestCase):
    def test_the_two_sizes_and_their_numbers(self):
        self.assertEqual(setup.FAMILIES["gyro"]["file"].format(q="Gyro-S"), "Qwen3.8-Flash-Next-Gyro-S.gguf")
        for model, name, download, memory in (("Gyro-S", "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf", 58.5, 27.6),
                                              ("Gyro-M", "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf", 92.0, 35.0)):
            d = setup.MODELS[model]
            self.assertEqual(setup.model_file(setup.FAMILIES["gyro"], model, 1), name)
            self.assertEqual((d["download_gb"], d["ram_gb"]), (download, memory))
            size, sha = setup.GYRO_FILES[name]
            self.assertAlmostEqual(size / 1e9, download, delta=0.05)
            self.assertEqual(len(sha), 64)
            self.assertEqual(setup.model_shards(setup.FAMILIES["gyro"], model), 1)
            self.assertIs(d.get("vision"), False)          # xeno: upstream True; not tried with the fork's engine-gyro
            self.assertTrue(d["mem_total"])
            self.assertEqual(d["families"], ("gyro",))

    def test_pinned_revision_and_license(self):
        repo = "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"
        self.assertEqual(len(setup.HF_REVISIONS[repo]), 40)
        self.assertIn(setup.HF_REVISIONS[repo], setup.FAMILIES["gyro"]["hf"])
        self.assertIn("Qwen Community License 1.0", setup.FAMILIES["gyro"]["license"])

    def test_never_the_default_family(self):
        self.assertEqual(setup.menu_families(False)[0], "qwen")
        self.assertNotIn("gyro", setup.RECOMMENDED_SIZES)

    def test_memory_is_vram_plus_ram(self):
        self.assertEqual(setup.mem_have("Gyro-S", 16.0, 12.0), 28.0)
        self.assertEqual(setup.mem_have("Q2_0", 16.0, 12.0), 16.0)
        names = ["Gyro-S", "Gyro-M"]
        self.assertEqual(setup.recommended_size("gyro", names, 24.0, 12.0), "Gyro-M")     # 36 GB
        self.assertEqual(setup.recommended_size("gyro", names, 16.0, 12.0), "Gyro-S")     # 28 GB
        self.assertEqual(setup.recommended_size("gyro", names, 8.0, 8.0), "Gyro-S")       # nothing fits: the first

    def test_mcp_table_matches(self):
        for m in ("Gyro-S", "Gyro-M"):
            for k in ("download_gb", "ram_gb", "arena_gb", "mem_total", "engine", "vision"):   # vision: False in both
                self.assertEqual(setup.MODELS[m][k], M.FALLBACK_MODELS[m][k], (m, k))
        self.assertEqual(setup.FAMILIES["gyro"]["tag"], M.FALLBACK_FAMILIES["gyro"]["tag"])

    def test_mcp_recommends_gyro_only_where_nothing_else_fits(self):
        s = M.Strata(ROOT)
        for ram, vram, family in ((64.0, 12.0, "qwen"), (48.0, 12.0, "qwen"), (32.0, 12.0, "swift"), (28.0, 8.0, "coder")):
            self.assertEqual(s.recommend(hardware(ram, vram))["family"], family, (ram, vram))
        got = s.recommend(hardware(20.0, 12.0))                  # 32 GB in total, the Coder's 32 GB of RAM missing
        self.assertEqual((got["family"], got["model"]), ("gyro", "Gyro-S"))
        got = s.recommend(hardware(24.0, 16.0))                  # 40 GB in total: the owner's 40 GB tier, Swift 1.5's IQ2_XS
        self.assertEqual((got["family"], got["model"]), ("swift", "IQ2_XS"))
        self.assertIn("Gyro-M", got["why"])                      # named as the other choice (NVIDIA and AMD)
        intel = hardware(24.0, 16.0)
        intel["gpus"] = [{**g, "vendor": "intel"} for g in intel["gpus"]]
        self.assertNotIn("Gyro", s.recommend(intel)["why"])      # never Intel Arc
        self.assertIsNone(s.recommend(hardware(12.0, 8.0))["family"])
        self.assertIsNone(s.recommend(hardware(16.0, 11.0))["family"])   # 27 GB: below Gyro-S's 27.6 + room

    def test_mcp_table_verdict_counts_vram(self):
        s = M.Strata(ROOT)
        fam = next(f for f in s.model_table(hardware(16.0, 16.0)) if f["family"] == "gyro")
        by = {x["model"]: x for x in fam["sizes"]}
        self.assertEqual(by["Gyro-S"]["on_this_pc"], "fits")
        self.assertEqual(by["Gyro-M"]["on_this_pc"], "tight")
        # the fork's Gyro reads no pictures ("vision": False in setup.py and the MCP table; upstream's: True)
        self.assertFalse(by["Gyro-S"]["images"])
        self.assertFalse(fam["images"])

    def test_mcp_table_gives_ram_plus_vram_without_the_ngram_table(self):
        """The Models page shows what a version needs of RAM + VRAM together, the n-gram table (on the SSD) not counted."""
        s = M.Strata(ROOT)
        table = s.model_table(hardware(64.0, 12.0))
        gyro = {x["model"]: x for x in next(f for f in table if f["family"] == "gyro")["sizes"]}
        self.assertEqual(gyro["Gyro-S"]["memory_gb"], 27.6)           # stated in total (mem_total), not the 58.5 GB download
        self.assertTrue(gyro["Gyro-S"]["mem_total"])
        swift = {x["model"]: x for x in next(f for f in table if f["family"] == "swift")["sizes"]}
        self.assertEqual(swift["IQ2_XS"]["memory_gb"], 36.5)          # 35.5 GB of experts + ~1 GB dense
        self.assertFalse(swift["IQ2_XS"]["mem_total"])


class GyroEngineVersionTest(unittest.TestCase):
    """Gyro's expert files are native_experts.txt v5 (Hadamard-folded). The released 0.1.45 engine reads only up to v4, so the
    minimum is 0.1.45.1, the first engine package built after the v5 reader (APR port, f47b0385f) was merged."""

    def test_gyro_needs_an_engine_that_reads_v5_files(self):
        self.assertEqual(setup.GYRO_ENGINE, (0, 1, 45, 1))
        self.assertGreater(setup.GYRO_ENGINE, (0, 1, 45))            # the released 0.1.45 is not enough
        sizes = [m for m in setup.MODELS.values() if "gyro" in str(m.get("file", "")).lower()]
        self.assertEqual(len(sizes), 2)
        self.assertTrue(all(m["engine"] == (0, 1, 45, 1) for m in sizes))

    def test_every_catalog_agrees(self):
        for m in ("Gyro-S", "Gyro-M"):
            self.assertEqual(setup.MODELS[m]["engine"], M.FALLBACK_MODELS[m]["engine"], m)

    def test_the_catalog_tells_the_app_the_minimum(self):
        table = M.Strata(ROOT).model_table(hardware(64.0, 12.0))
        gyro = {x["model"]: x for x in next(f for f in table if f["family"] == "gyro")["sizes"]}
        self.assertEqual({k: v["engine_min"] for k, v in gyro.items()}, {"Gyro-S": "0.1.45.1", "Gyro-M": "0.1.45.1"})
        qwen = {x["model"]: x for x in next(f for f in table if f["family"] == "qwen")["sizes"]}
        self.assertIsNone(qwen["Q2_0"]["engine_min"])                # a size with no minimum of its own says so

    def test_the_pack_reader_is_in_the_source_this_minimum_describes(self):
        text = (ROOT / "src" / "kernels" / "cpu" / "expert_layout.cpp").read_text(encoding="utf-8")
        self.assertIn("native_experts.txt v5 declares Hadamard-folded experts", text)


if __name__ == "__main__":
    unittest.main()
