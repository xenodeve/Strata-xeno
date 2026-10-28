"""P0 memory semantics and small Windows limit controls; no model or GPU."""
import os
import ctypes
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    from tools.qwenputer import memory as telemetry
except ImportError:
    telemetry = None


class MemorySemanticsTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(telemetry, "P0 needs authoritative memory telemetry")

    def test_system_commit_uses_pages_and_keeps_boot_peak_distinct(self):
        info = SimpleNamespace(CommitTotal=9, CommitLimit=20, CommitPeak=18,
                               PhysicalTotal=32, PhysicalAvailable=5, PageSize=4096)
        row = telemetry.performance_fields(info)
        self.assertEqual(row["system_commit_bytes"], 9 * 4096)
        self.assertEqual(row["system_commit_available_bytes"], 11 * 4096)
        self.assertEqual(row["system_commit_peak_since_boot_bytes"], 18 * 4096)
        self.assertEqual(row["physical_available_bytes"], 5 * 4096)
        self.assertEqual(row["source"], "GetPerformanceInfo")

    def test_process_private_and_lifetime_peak_are_not_system_or_resident_bytes(self):
        info = SimpleNamespace(PrivateUsage=123, PeakPagefileUsage=456,
                               WorkingSetSize=78, PeakWorkingSetSize=90)
        row = telemetry.process_fields(info)
        self.assertEqual(row["process_private_commit_bytes"], 123)
        self.assertEqual(row["process_private_commit_peak_lifetime_bytes"], 456)
        self.assertEqual(row["working_set_bytes"], 78)
        self.assertNotIn("system_commit_bytes", row)

    def test_disabled_job_limits_are_null_not_zero_bytes(self):
        info = SimpleNamespace(basic=SimpleNamespace(flags=0x2000), process_memory=123,
                               job_memory=456, peak_process=789, peak_job=900)
        row = telemetry.job_fields(info)
        self.assertIsNone(row["process_commit_limit_bytes"])
        self.assertIsNone(row["job_commit_limit_bytes"])
        self.assertEqual(row["peak_process_commit_bytes"], 789)

    def test_process_limit_event_binds_to_failing_pid(self):
        result = telemetry.classify_commit_failure(1455, 42, [{"message_id": 9, "pid": 42}])
        self.assertEqual(result["classification"], "process_guard_observed")
        unrelated = telemetry.classify_commit_failure(1455, 43, [{"message_id": 9, "pid": 42}])
        self.assertEqual(unrelated["classification"], "inconclusive_commit_limit")

    def test_job_guard_event_is_separate_from_process_guard(self):
        result = telemetry.classify_commit_failure(1455, 42, [{"message_id": 10, "pid": 42}])
        self.assertEqual(result["classification"], "job_guard_observed")

    def test_missing_event_and_soft_threshold_cannot_prove_os_exhaustion(self):
        for events in ([], [{"message_id": 11, "pid": 42, "violation_flags": 0x200}]):
            result = telemetry.classify_commit_failure(1455, 42, events)
            self.assertEqual(result["classification"], "inconclusive_commit_limit")
            self.assertFalse(result["os_limit_proven"])

    def test_non_commit_error_is_not_relabelled_memory_exhaustion(self):
        self.assertEqual(telemetry.classify_commit_failure(487, 42, [])["classification"],
                         "not_commit_limit_error")

    @unittest.skipUnless(os.name == "nt", "Windows error state")
    def test_dequeued_hard_event_survives_later_port_error(self):
        class Port:
            calls = 0
            def GetQueuedCompletionStatus(self, port, message, key, pid, timeout):
                self.calls += 1
                if self.calls == 1:
                    message._obj.value, key._obj.value, pid._obj.value = 9, 1, 42
                    return True
                ctypes.set_last_error(6)
                return False
        monitor = telemetry.JobMemoryMonitor.__new__(telemetry.JobMemoryMonitor)
        monitor.kernel, monitor.port, monitor.start, monitor.events = Port(), 1, 0, []
        with self.assertRaises(OSError):
            monitor.poll()
        self.assertEqual(len(monitor.events), 1, "a dequeued event cannot be recovered from Windows")
        self.assertEqual(monitor.events[0]["message_id"], 9)
        self.assertEqual(monitor.events[0]["pid"], 42)

    def test_collection_errors_cannot_bypass_owned_job_cleanup(self):
        finish = getattr(telemetry, "finish_owned_job_telemetry", None)
        self.assertTrue(callable(finish), "cleanup must survive telemetry errors")
        closed = []
        class Job:
            def close(self): closed.append("job")
        class Monitor:
            events = [{"message_id": 9, "pid": 42}]
            def poll(self): raise OSError("port error")
            def snapshot(self): raise OSError("snapshot error")
            def close(self): closed.append("port")
        result = finish(Job(), Monitor())
        self.assertCountEqual(closed, ["job", "port"])
        self.assertEqual(len(result["telemetry_errors"]), 2)
        self.assertEqual(result["job_events"][0]["message_id"], 9)

    @unittest.skipUnless(os.name == "nt", "Windows error state")
    def test_soft_enrichment_error_preserves_prior_hard_event_and_error(self):
        class Port:
            calls = 0
            def GetQueuedCompletionStatus(self, port, message, key, pid, timeout):
                self.calls += 1
                if self.calls <= 2:
                    message._obj.value, key._obj.value, pid._obj.value = (9 if self.calls == 1 else 11), 1, 42
                    return True
                ctypes.set_last_error(258)
                return False
        monitor = telemetry.JobMemoryMonitor.__new__(telemetry.JobMemoryMonitor)
        monitor.kernel, monitor.port, monitor.start, monitor.events = Port(), 1, 0, []
        def refuse_query(*args): raise OSError("violation query refused")
        monitor._query = refuse_query
        monitor.poll()
        self.assertEqual([e["message_id"] for e in monitor.events], [9, 11])
        self.assertIn("violation query refused", monitor.events[1]["violation_query_error"])

    def test_partial_report_is_durable_before_fallible_postrun_probes(self):
        persist = getattr(telemetry, "persist_report", None)
        self.assertTrue(callable(persist), "post-run probes must not erase the failure evidence")
        # The executor's virtualized default TEMP permits file creation but
        # refuses atomic replace. This workspace is an explicit writable root.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            path = Path(temp) / "result.json"
            def failed_probe():
                self.assertEqual(json.loads(path.read_text())["job_events"][0]["message_id"], 9)
                raise OSError("GetPerformanceInfo failed")
            report = {"job_events": [{"message_id": 9, "pid": 42}]}
            persist(path, report, {"host_after": failed_probe, "later": lambda: 123})
            saved = json.loads(path.read_text())
            self.assertEqual(saved["job_events"], report["job_events"])
            self.assertEqual(saved["later"], 123)
            self.assertEqual(saved["postrun_errors"][0]["operation"], "host_after")
            self.assertEqual(saved["postrun_errors"][0]["type"], "OSError")
            self.assertEqual(saved["postrun_errors"][0]["error"], "GetPerformanceInfo failed")

    @unittest.skipUnless(os.name == "nt", "Windows APIs")
    def test_live_system_and_current_process_have_distinct_sources(self):
        system = telemetry.system_snapshot()
        process = telemetry.process_snapshot(os.getpid())
        self.assertGreater(system["system_commit_limit_bytes"], system["system_commit_bytes"])
        self.assertGreater(process["process_private_commit_bytes"], 0)
        self.assertGreaterEqual(process["process_private_commit_peak_lifetime_bytes"],
                                process["process_private_commit_bytes"])
        self.assertEqual(process["source"], "GetProcessMemoryInfo")

    @unittest.skipUnless(os.name == "nt", "Windows APIs")
    def test_explicit_none_disables_only_owned_process_commit_guard(self):
        from tools.qwenputer.headless import OwnedJob
        job, monitor = OwnedJob(), None
        try:
            try:
                monitor = telemetry.JobMemoryMonitor(job, None)
            except (TypeError, ValueError) as error:
                self.fail(f"explicit diagnostic no-cap mode is required: {error}")
            row = monitor.snapshot()
            self.assertIsNone(row["process_commit_limit_bytes"])
            self.assertIsNone(row["job_commit_limit_bytes"])
            self.assertEqual(row["limit_flags"] & 0x300, 0)
            self.assertEqual(row["limit_flags"] & 0x2000, 0x2000)
            monitor.set_process_limit(64 * 1024**2)
            self.assertEqual(monitor.snapshot()["process_commit_limit_bytes"], 64 * 1024**2)
            monitor.set_process_limit(None)
            self.assertIsNone(monitor.snapshot()["process_commit_limit_bytes"])
        finally:
            if monitor is not None:
                monitor.close()
            job.close()

    @unittest.skipUnless(os.name == "nt", "Windows APIs")
    def test_small_real_guard_control_preserves_limit_and_never_claims_os_exhaustion(self):
        control = getattr(telemetry, "small_limit_control", None)
        self.assertTrue(callable(control), "test a real small job, not only simulated event dictionaries")
        row = control()
        self.assertEqual(row["allocation_winerror"], 1455)
        self.assertTrue(row["first_commit_succeeded"])
        self.assertLessEqual(row["job_final"]["process_commit_limit_bytes"], 64 * 1024**2)
        self.assertLessEqual(row["job_final"]["peak_process_commit_bytes"],
                             row["job_final"]["process_commit_limit_bytes"])
        self.assertIn(row["classification"]["classification"],
                      ("process_guard_observed", "inconclusive_commit_limit"))
        self.assertFalse(row["classification"]["os_limit_proven"])
        self.assertTrue(any(e["message_id"] == 11 for e in row["events"]),
                        "the soft job threshold notification must actually be exercised")


if __name__ == "__main__":
    unittest.main()
