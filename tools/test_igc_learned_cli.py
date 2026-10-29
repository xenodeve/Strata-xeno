"""Native learned-controller admission checks; never load a model in this suite."""
import os
from pathlib import Path
import subprocess
import unittest


@unittest.skipUnless(os.environ.get("STRATA_IGC_ENGINE_EXE"), "set the diagnostic engine executable")
class LearnedAdmission(unittest.TestCase):
    def invoke(self, *args):
        exe = Path(os.environ["STRATA_IGC_ENGINE_EXE"])
        self.assertTrue(exe.is_file())
        result = subprocess.run(
            [str(exe), *args], capture_output=True, timeout=30,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        self.assertNotEqual(result.returncode, 0)
        return (result.stdout + result.stderr).decode("utf-8", errors="replace")

    def test_requires_actual_native_serve_before_model_allocation(self):
        self.assertIn("IGC learned diagnostic requires native serve", self.invoke(
            "--igc-learned-diagnostic-bundle", "missing-controller.bin"))

    def test_controlled_fixture_is_not_a_learned_route(self):
        self.assertIn("IGC learned diagnostic cannot combine", self.invoke(
            "--igc-learned-diagnostic-bundle", "missing-controller.bin",
            "--igc-output-diagnostic-artifact", "missing-output.bin"))

    @unittest.skipUnless(os.environ.get("STRATA_IGC_MODEL_INPUT"), "select an existing model for metadata-only admission")
    def test_valid_native_route_reaches_bundle_validation(self):
        model = os.environ["STRATA_IGC_MODEL_INPUT"]
        text = self.invoke("--serve", "--native", model, "--ple-gguf", model,
                           "--igc-learned-diagnostic-bundle", "missing-controller.bin")
        self.assertIn("strata IGC learned: validating bundle", text)
        self.assertNotIn("IGC learned diagnostic requires native serve", text)

    @unittest.skipUnless(os.environ.get("STRATA_IGC_MODEL_INPUT"), "select an existing model for metadata-only admission")
    def test_effective_model_override_cannot_escape_content_receipt(self):
        model = os.environ["STRATA_IGC_MODEL_INPUT"]
        for flag in ("--embd-gguf", "--native-head-gguf", "--native-dense-gguf", "--ple-gguf"):
            with self.subTest(flag=flag):
                text = self.invoke("--serve", "--native", model, "--ple-gguf", model,
                                   flag, "unbound-override.gguf", "--igc-learned-diagnostic-bundle", "missing-controller.bin")
                self.assertIn("IGC learned diagnostic requires bound model sources", text)


if __name__ == "__main__":
    unittest.main()
