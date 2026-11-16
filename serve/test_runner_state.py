"""serve/test_runner_state.py - the runner's hooks for Strata Desktop (#310 part 3): the device label the Client sends
(Task 2) and, in Task 3, the states it reports.  No HTTP: the Client's headers are read as built.

    python -m pytest serve/test_runner_state.py -q
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import runner as runner_mod  # noqa: E402

URL = "http://127.0.0.1:9"
DEV = {"id": "x" * 22, "name": "pc"}


class Label(unittest.TestCase):
    def test_label_default_unchanged(self):
        c = runner_mod.Client(URL, "k", DEV)
        self.assertEqual(c._headers["X-Strata-Device-Name"], "Strata runner on pc")
        self.assertEqual(c._headers["X-Strata-Device"], "x" * 22)

    def test_label_given(self):
        c = runner_mod.Client(URL, "k", DEV, label="Strata Desktop on pc")
        self.assertEqual(c._headers["X-Strata-Device-Name"], "Strata Desktop on pc")
        self.assertEqual(c._headers["X-Strata-Device"], "x" * 22)

    def test_device_label(self):
        self.assertEqual(runner_mod.device_label(DEV), "Strata runner on pc")
        self.assertEqual(runner_mod.device_label(DEV, "Strata Desktop"), "Strata Desktop on pc")
        self.assertEqual(runner_mod.device_label({"id": "x" * 22, "name": "เครื่อง-1"}, "Strata Desktop"),
                         "Strata Desktop on " + "?" * len("เครื่อง") + "-1")         # printable ASCII only: what a header can carry
        self.assertEqual(runner_mod.device_label({"id": "x" * 22}, "A"), "A on " + runner_mod.machine_name())

    def test_label_is_made_header_safe(self):
        c = runner_mod.Client(URL, "k", DEV, label="Strata Desktop on เครื่อง")
        self.assertEqual(c._headers["X-Strata-Device-Name"], "Strata Desktop on " + "?" * len("เครื่อง"))


if __name__ == "__main__":
    unittest.main()
