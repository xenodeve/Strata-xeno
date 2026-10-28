"""Windows P0 diagnostics: system commitment, private commit and owned-job limits.

Completion-port messages 9/10 positively identify an observed hard-limit event.
Their absence never proves that Windows, rather than a job, refused an allocation.
Message 11 is a guaranteed *soft threshold* notification, not an allocation failure.
No function changes the system, pagefile, serving profile or another process's job.
"""
from __future__ import annotations

import ctypes as c
from ctypes import wintypes as w
import os
import time


class PerformanceInfo(c.Structure):
    _fields_ = [("cb", w.DWORD)] + [(name, c.c_size_t) for name in (
        "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
        "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
        (name, w.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]


class ProcessCounters(c.Structure):
    _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD)] + [
        (name, c.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
        "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]


class BasicLimit(c.Structure):
    _fields_ = [("user", c.c_int64), ("job_user", c.c_int64), ("flags", w.DWORD),
                ("min_ws", c.c_size_t), ("max_ws", c.c_size_t), ("processes", w.DWORD),
                ("affinity", c.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]


class ExtendedLimit(c.Structure):
    _fields_ = [("basic", BasicLimit), ("io", c.c_uint64 * 6),
                ("process_memory", c.c_size_t), ("job_memory", c.c_size_t),
                ("peak_process", c.c_size_t), ("peak_job", c.c_size_t)]


class NotificationLimit(c.Structure):
    _fields_ = [("read", c.c_uint64), ("write", c.c_uint64), ("user", c.c_int64),
                ("memory", c.c_uint64), ("tolerance", w.DWORD), ("interval", w.DWORD),
                ("flags", w.DWORD)]


class Violation(c.Structure):
    _fields_ = [("flags", w.DWORD), ("violation_flags", w.DWORD)] + [
        (name, c.c_uint64) for name in ("read", "read_limit", "write", "write_limit",
                                       "user", "user_limit", "memory", "memory_limit")] + [
        ("tolerance", w.DWORD), ("tolerance_limit", w.DWORD)]


def performance_fields(info):
    page = int(info.PageSize)
    if page <= 0 or info.CommitTotal > info.CommitLimit:
        raise ValueError("invalid GetPerformanceInfo page/commit values")
    return {"source": "GetPerformanceInfo", "page_size_bytes": page,
            "system_commit_bytes": int(info.CommitTotal) * page,
            "system_commit_limit_bytes": int(info.CommitLimit) * page,
            "system_commit_available_bytes": int(info.CommitLimit - info.CommitTotal) * page,
            "system_commit_peak_since_boot_bytes": int(info.CommitPeak) * page,
            "physical_total_bytes": int(info.PhysicalTotal) * page,
            "physical_available_bytes": int(info.PhysicalAvailable) * page}


def process_fields(info):
    return {"source": "GetProcessMemoryInfo",
            "process_private_commit_bytes": int(info.PrivateUsage),
            "process_private_commit_peak_lifetime_bytes": int(info.PeakPagefileUsage),
            "working_set_bytes": int(info.WorkingSetSize),
            "working_set_peak_lifetime_bytes": int(info.PeakWorkingSetSize)}


def job_fields(info):
    flags = int(info.basic.flags)
    return {"source": "QueryInformationJobObject/ExtendedLimitInformation", "limit_flags": flags,
            "process_commit_limit_bytes": int(info.process_memory) if flags & 0x100 else None,
            "job_commit_limit_bytes": int(info.job_memory) if flags & 0x200 else None,
            "peak_process_commit_bytes": int(info.peak_process),
            "peak_job_commit_bytes": int(info.peak_job)}


def _libraries():
    if os.name != "nt":
        raise RuntimeError("Windows memory diagnostics require Windows")
    k = c.WinDLL("kernel32", use_last_error=True)
    p = c.WinDLL("psapi", use_last_error=True)
    return k, p


def system_snapshot():
    _, p = _libraries()
    p.GetPerformanceInfo.argtypes = (c.POINTER(PerformanceInfo), w.DWORD)
    p.GetPerformanceInfo.restype = w.BOOL
    info = PerformanceInfo()
    info.cb = c.sizeof(info)
    if not p.GetPerformanceInfo(c.byref(info), c.sizeof(info)):
        raise c.WinError(c.get_last_error())
    return performance_fields(info)


def process_snapshot(pid):
    k, p = _libraries()
    k.OpenProcess.argtypes = (w.DWORD, w.BOOL, w.DWORD)
    k.OpenProcess.restype = w.HANDLE
    k.CloseHandle.argtypes = (w.HANDLE,)
    p.GetProcessMemoryInfo.argtypes = (w.HANDLE, c.POINTER(ProcessCounters), w.DWORD)
    p.GetProcessMemoryInfo.restype = w.BOOL
    handle = k.OpenProcess(0x1000 | 0x10, False, pid)
    if not handle:
        raise c.WinError(c.get_last_error())
    try:
        info = ProcessCounters()
        info.cb = c.sizeof(info)
        if not p.GetProcessMemoryInfo(handle, c.byref(info), c.sizeof(info)):
            raise c.WinError(c.get_last_error())
        return {"pid": int(pid), **process_fields(info)}
    finally:
        k.CloseHandle(handle)


def classify_commit_failure(winerror, failing_pid, events):
    classification = "not_commit_limit_error"
    matching = [e for e in events if failing_pid is not None and e.get("pid") == failing_pid]
    if winerror == 1455:
        classification = "inconclusive_commit_limit"
        if any(e["message_id"] == 9 for e in matching):
            classification = "process_guard_observed"
        elif any(e["message_id"] == 10 for e in matching):
            classification = "job_guard_observed"
    return {"classification": classification, "os_limit_proven": False,
            "note": "Hard-limit notifications are not guaranteed; absence and sampled headroom are inconclusive. "
                    "A matching event identifies a limit event in this run, not the exact failed call by itself."}


class JobMemoryMonitor:
    """Attach before starting an owned job; retain events and peaks through drain."""
    def __init__(self, owned_job, process_limit_bytes, notify_job_bytes=None):
        self.job = owned_job
        self.kernel = k = owned_job.kernel
        self.port = None
        self.events = []
        self.start = time.monotonic()
        self.notification_threshold = notify_job_bytes
        k.CreateIoCompletionPort.argtypes = (w.HANDLE, w.HANDLE, c.c_size_t, w.DWORD)
        k.CreateIoCompletionPort.restype = w.HANDLE
        k.GetQueuedCompletionStatus.argtypes = (w.HANDLE, c.POINTER(w.DWORD),
                                               c.POINTER(c.c_size_t), c.POINTER(c.c_void_p), w.DWORD)
        k.GetQueuedCompletionStatus.restype = w.BOOL
        self.port = k.CreateIoCompletionPort(c.c_void_p(-1), None, 0, 1)
        if not self.port:
            raise c.WinError(c.get_last_error())
        try:
            class Associate(c.Structure):
                _fields_ = [("key", c.c_void_p), ("port", w.HANDLE)]
            self._set(7, Associate(1, self.port))
            self.set_process_limit(process_limit_bytes)
            if notify_job_bytes is not None:
                self.set_notification_threshold(notify_job_bytes)
        except BaseException:
            self.close()
            raise

    def _set(self, kind, info):
        if not self.kernel.SetInformationJobObject(self.job.handle, kind, c.byref(info), c.sizeof(info)):
            raise c.WinError(c.get_last_error())

    def _query(self, kind, cls):
        info = cls()
        if not self.kernel.QueryInformationJobObject(self.job.handle, kind, c.byref(info), c.sizeof(info), None):
            raise c.WinError(c.get_last_error())
        return info

    def set_process_limit(self, amount):
        if amount is not None and amount <= 0:
            raise ValueError("process guard must be positive or explicitly None")
        info = self._query(9, ExtendedLimit)
        if amount is None:
            # Explicit per-test exception; preserve kill-on-close and every
            # unrelated flag. This changes only this task-owned job.
            info.basic.flags &= ~0x100
            info.process_memory = 0
        else:
            info.basic.flags |= 0x100
            info.process_memory = amount
        self._set(9, info)
        actual = self.snapshot()["process_commit_limit_bytes"]
        if amount is None:
            if actual is not None:
                raise RuntimeError("Windows retained the diagnostic process guard")
        elif actual is None or not 0 < actual <= amount or amount - actual >= 65536:
            raise RuntimeError("Windows did not apply the bounded process guard")

    def snapshot(self):
        return {**job_fields(self._query(9, ExtendedLimit)),
                "notification_job_commit_bytes": self.notification_threshold}

    def set_notification_threshold(self, amount):
        if amount <= 0:
            raise ValueError("notification threshold must be positive")
        info = NotificationLimit()
        info.flags, info.memory = 0x200, amount
        self._set(12, info)
        self.notification_threshold = amount

    def poll(self):
        new = []
        while True:
            message, key, value = w.DWORD(), c.c_size_t(), c.c_void_p()
            if not self.kernel.GetQueuedCompletionStatus(self.port, c.byref(message), c.byref(key),
                                                        c.byref(value), 0):
                error = c.get_last_error()
                if error == 258:  # WAIT_TIMEOUT: queue drained, not a missing limit event proof
                    break
                raise c.WinError(error)
            event = {"seconds": time.monotonic() - self.start, "message_id": message.value,
                     "completion_key": key.value, "pid": value.value}
            # A dequeued notification cannot be recovered from Windows if a later
            # query fails. Retain it before enriching it or reading another event.
            self.events.append(event)
            new.append(event)
            if message.value == 11:
                try:
                    violation = self._query(13, Violation)
                    event.update(violation_flags=violation.violation_flags,
                                 notification_job_commit_bytes=violation.memory,
                                 notification_job_limit_bytes=violation.memory_limit)
                except OSError as error:
                    event["violation_query_error"] = str(error)
        return new

    def close(self):
        if self.port:
            self.kernel.CloseHandle(self.port)
            self.port = None


def finish_owned_job_telemetry(job, monitor):
    """Retain partial evidence and close the owned job even if collection fails.

    Call after normal terminate_and_drain; on setup/collection failure the job's
    kill-on-close policy is still applied. A caller must not label an unverified
    drain as successful merely because this final collector returned.
    """
    result = {"telemetry_errors": []}
    try:
        try:
            monitor.poll()
        except OSError as error:
            result["telemetry_errors"].append({"operation": "poll", "error": str(error)})
        try:
            result["job_final"] = monitor.snapshot()
        except OSError as error:
            result["telemetry_errors"].append({"operation": "snapshot", "error": str(error)})
        result["job_events"] = list(monitor.events)
    finally:
        try:
            monitor.close()
        finally:
            job.close()
    return result


def persist_report(path, report, postrun_probes=None):
    """Save existing evidence first; optional probes cannot erase it on failure."""
    import json
    def save():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(path)
    save()
    for name, probe in (postrun_probes or {}).items():
        try:
            report[name] = probe()
        except Exception as error:
            report.setdefault("postrun_errors", []).append(
                {"operation": name, "type": type(error).__name__, "error": str(error)})
        save()
    return report


def small_limit_control():
    """One child, <=64 MiB guard, only 1 MiB successful extra commitment; no GPU.

    The 8 MiB request is deliberately larger than the remaining process guard.
    This validates observable Windows semantics without exhausting system memory.
    """
    import json
    from pathlib import Path
    import subprocess
    import sys
    from tools.qwenputer.headless import OwnedJob

    code = "\n".join([
        "import ctypes as c,json,os,sys",
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})",
        "from tools.qwenputer.memory import process_snapshot",
        "k=c.WinDLL('kernel32',use_last_error=True)",
        "k.VirtualAlloc.argtypes=(c.c_void_p,c.c_size_t,c.c_uint32,c.c_uint32)",
        "k.VirtualAlloc.restype=c.c_void_p",
        "print(json.dumps(process_snapshot(os.getpid())),flush=True)",
        "assert sys.stdin.buffer.read(1)==b'G'",
        "first=k.VirtualAlloc(None,1024**2,0x3000,4)",
        "c.set_last_error(0)",
        "second=k.VirtualAlloc(None,8*1024**2,0x3000,4)",
        "error=c.get_last_error()",
        "print(json.dumps({'first_commit_succeeded':bool(first), 'second_commit_succeeded':bool(second),"
        " 'allocation_winerror':error,'allocation_request_bytes':8*1024**2,'pid':os.getpid()}),flush=True)",
        "assert sys.stdin.buffer.read(1)==b'Q'",
    ])
    job, proc, monitor = OwnedJob(), None, None
    try:
        monitor = JobMemoryMonitor(job, 64 * 1024**2)
        proc = subprocess.Popen([sys.executable, "-B", "-u", "-c", code], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=0x08000000)
        # communicate cannot be used until the two bounded handshakes finish.
        # A reader timeout below keeps a broken fixture from blocking the task.
        import queue
        import threading
        lines = queue.Queue()
        def read_lines():
            for line in proc.stdout:
                lines.put(line)
        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        baseline = json.loads(lines.get(timeout=10))
        job.assign(proc)
        limit = ((baseline["process_private_commit_bytes"] + 2 * 1024**2 + 4095) // 4096) * 4096
        if limit > 64 * 1024**2:
            raise RuntimeError("small control's startup exceeded its fixed 64 MiB budget")
        monitor.set_process_limit(limit)
        monitor.set_notification_threshold(baseline["process_private_commit_bytes"] + 512 * 1024)
        proc.stdin.write(b"G")
        proc.stdin.flush()
        result = json.loads(lines.get(timeout=10))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            monitor.poll()
            if {9, 11}.issubset({e["message_id"] for e in monitor.events}):
                break
            time.sleep(0.01)
        result.update(system=system_snapshot(), process=process_snapshot(proc.pid))
        proc.stdin.write(b"Q")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(timeout=5)
        job.terminate_and_drain(proc.returncode)
        monitor.poll()
        result.update(job_final=monitor.snapshot(), events=monitor.events,
                      classification=classify_commit_failure(result["allocation_winerror"], proc.pid,
                                                             monitor.events))
        reader.join(timeout=1)
        return result
    finally:
        job.close()
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
        if monitor is not None:
            monitor.close()
