"""Exercise the real P0 binary; missing binaries are not red-test evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("QWENPUTER_PROBE"), "requires a built CUDA probe and GPU")
class ProbeTest(unittest.TestCase):
    def run_probe(self, *extra):
        binary = Path(os.environ["QWENPUTER_PROBE"])
        self.assertTrue(binary.is_file(), "build the probe before running its behavior tests")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            proc = subprocess.run([str(binary), "--device", "0", "--json", str(path), *extra],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  timeout=30, creationflags=0x08000000 if os.name == "nt" else 0)
            self.assertTrue(path.is_file(), f"probe produced no report: {proc.stderr}")
            return proc, json.loads(path.read_text(encoding="utf-8"))

    def test_real_graph_zero_one_n_relaunch_bound_and_cancel(self):
        proc, report = self.run_probe()
        self.assertEqual(proc.returncode, 0, report)
        self.assertTrue(report["ok"], report)
        self.assertIn("context_switch_checked", report)
        if report["device_count"] > 1:
            self.assertTrue(report["context_switch_checked"])
        cases = {x["name"]: x for x in report["cases"]}
        expected = {"zero": (0, 0), "one": (1, 1), "n": (17, 153),
                    "repeat_n": (5, 15), "repeat_zero": (0, 0), "output_bound": (3, 6),
                    "rebuilt_one": (1, 1), "rebuilt_zero": (0, 0)}
        for name, (steps, checksum) in expected.items():
            with self.subTest(case=name):
                self.assertEqual(cases[name]["steps"], steps)
                self.assertEqual(cases[name]["checksum"], checksum)
        self.assertEqual(cases["output_bound"]["reason"], "output_bound")
        self.assertEqual(cases["cancel"]["reason"], "cancelled")
        self.assertGreater(cases["cancel"]["steps"], 0)
        self.assertLess(cases["cancel"]["steps"], 4096)
        if report["context_switch_checked"]:
            self.assertEqual(cases["after_device_switch"]["steps"], 1)
            self.assertEqual(cases["after_device_switch"]["checksum"], 1)

    def test_negative_control_detects_real_step_count_mutation(self):
        proc, report = self.run_probe("--negative-control")
        self.assertEqual(proc.returncode, 1, report)
        self.assertFalse(report["ok"])
        one = next(x for x in report["cases"] if x["name"] == "one")
        self.assertEqual(one["expected_steps"], 1)
        self.assertEqual(one["steps"], 2)
        self.assertFalse(one["ok"])


if __name__ == "__main__":
    unittest.main()
