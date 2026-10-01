"""serve/test_telemetry.py - the hardware sampler (xeno UI S6): every card on its own, throttle reasons decoded, a
history per card, the sampling rate, and the storage facts (model, bus, which disk holds the model).

    python -m unittest serve.test_telemetry -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import storage  # noqa: E402
from serve.telemetry import Telemetry, throttle_names  # noqa: E402


class FakeCard:
    def __init__(self, name, util, temp):
        self._name, self._util, self._temp = name, util, temp

    def ok(self):
        return True

    def name(self):
        return self._name

    def read(self):
        return {"util": self._util, "mem_used": 4 * 2**30, "mem_total": 12 * 2**30, "temp": self._temp, "power": 90.0,
                "power_limit": 220.0, "pcie_gen": 4, "pcie_gen_max": 4, "pcie_width": 16, "pcie_width_max": 16,
                "pcie_rx_mb": 12.5, "pcie_tx_mb": 3.0, "sm_clock": 2400, "mem_clock": 10500, "throttle": ["power cap"]}


def bare(cards) -> Telemetry:
    t = Telemetry.__new__(Telemetry)
    t.gpus = list(enumerate(cards))
    t.gpu = cards[0]
    t.ps = None
    t.fallback = SimpleNamespace(cpu=lambda: 1.0, ram=lambda: (1, 2))
    t.extra = None
    t._disk_prev = None
    t.disks_now = None
    return t


class PerCard(unittest.TestCase):
    def test_every_card_is_listed_with_its_own_facts_even_when_there_is_one(self):
        for cards in ([FakeCard("RTX 4070 SUPER", 10, 55)], [FakeCard("RTX 5060 Ti", 80, 70), FakeCard("RTX 4070 SUPER", 10, 55)]):
            s = bare(cards).sample()
            self.assertEqual(len(s["gpus"]), len(cards))
            g = s["gpus"][0]
            self.assertEqual((g["index"], g["name"], g["sm_clock"], g["throttle"]), (0, cards[0]._name, 2400, ["power cap"]))
            self.assertEqual((g["pcie_gen"], g["pcie_gen_max"], g["pcie_width"], g["pcie_width_max"]), (4, 4, 16, 16))
            self.assertEqual((g["pcie_rx_mb"], g["pcie_tx_mb"]), (12.5, 3.0))

    def test_the_old_aggregate_keys_are_unchanged(self):
        s = bare([FakeCard("A", 80, 70), FakeCard("B", 10, 55)]).sample()
        self.assertEqual((s["gpu_util"], s["gpu_temp"], s["gpu_mem_total"]), (45.0, 70, 24 * 2**30))

    def test_the_history_is_kept_per_card(self):
        t = bare([FakeCard("A", 80, 70), FakeCard("B", 10, 55)])
        keys = t.history_keys(t.sample())
        for k in ("gpu0_util", "gpu1_util", "gpu1_temp", "gpu0_pcie_rx_mb", "gpu1_power", "gpu0_mem_used"):
            self.assertIn(k, keys)


class Throttle(unittest.TestCase):
    def test_reasons_are_named_and_idle_is_not_one(self):
        self.assertEqual(throttle_names(0), [])
        self.assertEqual(throttle_names(0x1), [])                      # GPU idle: not a throttle
        self.assertEqual(throttle_names(0x4), ["power cap"])
        self.assertEqual(throttle_names(0x4 | 0x20), ["power cap", "thermal slowdown"])
        self.assertEqual(throttle_names(0x40 | 0x80), ["hardware thermal slowdown", "hardware power brake"])
        self.assertEqual(throttle_names(None), None)                   # NVML could not say: not "none"


class Rate(unittest.TestCase):
    def test_five_hz_while_a_request_runs_one_hz_idle(self):
        t = bare([FakeCard("A", 1, 1)])
        t.busy_fn = lambda: True
        self.assertEqual(t.interval(), 0.2)
        t.busy_fn = lambda: False
        self.assertEqual(t.interval(), 1.0)
        t.busy_fn = None
        self.assertEqual(t.interval(), 1.0)

    def test_the_history_takes_one_point_a_second_whatever_the_rate(self):
        t = bare([FakeCard("A", 1, 1)])
        t.last_record = 0.0
        self.assertTrue(t.should_record(10.0))
        self.assertFalse(t.should_record(10.2))
        self.assertFalse(t.should_record(10.8))
        self.assertTrue(t.should_record(11.0))


class Storage(unittest.TestCase):
    PS = json.dumps([{"DeviceId": "0", "FriendlyName": "Samsung SSD 990 PRO 2TB", "BusType": "NVMe", "MediaType": "SSD", "Size": 2000398934016},
                     {"DeviceId": "1", "FriendlyName": "WDC WD40EZRZ", "BusType": "SATA", "MediaType": "HDD", "Size": 4000787030016}])

    def fake_run(self, outputs):
        def run(cmd, **_):
            text = next((v for k, v in outputs.items() if k in " ".join(cmd)), "")
            return SimpleNamespace(returncode=0, stdout=text)
        return run

    def test_windows_disks_with_model_bus_media(self):
        disks = storage.physical_disks(os_name="nt", run=self.fake_run({"Get-PhysicalDisk": self.PS}))
        self.assertEqual([(d["index"], d["model"], d["bus"], d["media"]) for d in disks],
                         [(0, "Samsung SSD 990 PRO 2TB", "NVMe", "SSD"), (1, "WDC WD40EZRZ", "SATA", "HDD")])
        self.assertAlmostEqual(disks[0]["size_gb"], 1862.9, places=0)

    def test_a_single_disk_is_an_object_not_a_list(self):
        one = json.dumps({"DeviceId": "0", "FriendlyName": "X", "BusType": "NVMe", "MediaType": "SSD", "Size": 1})
        self.assertEqual(len(storage.physical_disks(os_name="nt", run=self.fake_run({"Get-PhysicalDisk": one}))), 1)

    def test_which_disk_holds_a_file(self):
        run = self.fake_run({"Get-Partition": "1\n"})
        self.assertEqual(storage.disk_of_path("D:\\models\\x.gguf", os_name="nt", run=run), 1)

    def test_failures_are_none_not_a_guess(self):
        def boom(*a, **k):
            raise OSError("no powershell")
        self.assertEqual(storage.physical_disks(os_name="nt", run=boom), [])
        self.assertIsNone(storage.disk_of_path("D:\\x", os_name="nt", run=boom))
        self.assertIsNone(storage.disk_of_path("relative.gguf", os_name="nt", run=self.fake_run({})))   # no drive letter

    def test_per_disk_rates_from_two_counter_snapshots(self):
        c = lambda rb, wb, rc, rt: SimpleNamespace(read_bytes=rb, write_bytes=wb, read_count=rc, read_time=rt)
        prev = {"PhysicalDrive0": c(0, 0, 0, 0), "PhysicalDrive1": c(0, 0, 0, 0)}
        cur = {"PhysicalDrive0": c(2 * 2**20, 1 * 2**20, 100, 50), "PhysicalDrive1": c(0, 0, 0, 0)}
        rates = storage.disk_rates(prev, cur, dt=2.0)
        self.assertEqual(rates[0], {"index": 0, "read_mb": 1.0, "write_mb": 0.5, "read_ms_op": 0.5})
        self.assertEqual(rates[1], {"index": 1, "read_mb": 0.0, "write_mb": 0.0, "read_ms_op": None})   # no reads: no latency


if __name__ == "__main__":
    unittest.main()
