"""Run one existing tool with the installed MSVC environment and no window."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


class OwnedJob:
    """One non-inheritable Windows job; only our waiting worker is assigned."""
    def __init__(self):
        import ctypes
        from ctypes import wintypes as w
        self.ctypes = ctypes
        self.kernel = k = ctypes.WinDLL("kernel32", use_last_error=True)

        class Basic(ctypes.Structure):
            _fields_ = [("user", ctypes.c_int64), ("job_user", ctypes.c_int64), ("flags", w.DWORD),
                        ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t), ("process_limit", w.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", ctypes.c_uint64 * 6),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

        class Accounting(ctypes.Structure):
            _fields_ = [("times", ctypes.c_int64 * 4), ("faults", w.DWORD),
                        ("total", w.DWORD), ("active", w.DWORD), ("terminated", w.DWORD)]

        self.accounting_type = Accounting
        k.CreateJobObjectW.argtypes = (w.LPVOID, w.LPCWSTR)
        k.CreateJobObjectW.restype = w.HANDLE
        k.SetInformationJobObject.argtypes = (w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD)
        k.AssignProcessToJobObject.argtypes = (w.HANDLE, w.HANDLE)
        k.TerminateJobObject.argtypes = (w.HANDLE, w.UINT)
        k.QueryInformationJobObject.argtypes = (w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD, w.LPVOID)
        k.CloseHandle.argtypes = (w.HANDLE,)
        self.handle = k.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, proc):
        if not self.kernel.AssignProcessToJobObject(self.handle, int(proc._handle)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def terminate_and_drain(self, code):
        if not self.kernel.TerminateJobObject(self.handle, code):
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        deadline = time.monotonic() + 10
        while True:
            info = self.accounting_type()
            if not self.kernel.QueryInformationJobObject(self.handle, 1, self.ctypes.byref(info),
                                                         self.ctypes.sizeof(info), None):
                raise self.ctypes.WinError(self.ctypes.get_last_error())
            if info.active == 0:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("task-owned job did not drain; completion is uncertain")
            time.sleep(0.01)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def run_owned_tool(command, env, log, timeout):
    if os.name != "nt":
        raise RuntimeError("this no-window launcher requires Windows")
    job = OwnedJob()
    proc = None
    try:
        # The worker cannot spawn the actual command until job assignment succeeds.
        # This removes the assign-after-spawn race for grandchildren.
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", *command],
                                env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=0x08000000)
        job.assign(proc)
        proc.stdin.write(b"G")
        proc.stdin.close()
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.write("TIMEOUT: stopping and draining the task-owned process tree\n")
            code = 124
        job.terminate_and_drain(code)
        proc.wait(timeout=10)
        return code
    finally:
        # On setup failure the worker has not received G, so no tool was started.
        job.close()
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def worker(command):
    if sys.stdin.buffer.read(1) != b"G":
        return 125
    return subprocess.Popen(command, stdin=subprocess.DEVNULL, creationflags=0x08000000).wait()


def msvc_environment(vcvars: Path, scratch: Path) -> dict[str, str]:
    if not vcvars.is_file() or '"' in str(vcvars):
        raise ValueError("invalid vcvars path")
    scratch.parent.mkdir(parents=True, exist_ok=True)
    # cmd /c has its own quote parser: do not pass a quoted command through
    # subprocess.list2cmdline a second time. Run a task-owned batch file instead.
    scratch.write_text(f'@echo off\ncall "{vcvars}" >nul\nif errorlevel 1 exit /b 1\nset\n',
                       encoding="utf-8")
    setup = subprocess.run(["cmd.exe", "/d", "/c", str(scratch)],
                           capture_output=True, text=True, creationflags=0x08000000, timeout=30)
    if setup.returncode:
        raise RuntimeError(f"MSVC setup failed ({setup.returncode}): {setup.stderr}")
    env = {}
    for line in setup.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            env[key.upper()] = value
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--vcvars", type=Path)
    parser.add_argument("--cuda", type=Path)
    parser.add_argument("--primary-uuid")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    flags = 0x08000000 if os.name == "nt" else 0
    env = dict(os.environ)
    if args.vcvars:
        env.update(msvc_environment(args.vcvars, args.log.with_suffix(".env.cmd")))
    if args.cuda:
        env["CUDA_PATH"] = str(args.cuda)
        env["PATH"] = os.pathsep.join([str(args.cuda / "bin"), str(args.cuda / "bin" / "x64"),
                                       env.get("PATH", "")])
    if args.primary_uuid:
        env["CUDA_VISIBLE_DEVICES"] = args.primary_uuid
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", encoding="utf-8") as log:
        log.write(f"command={command!r}\ncreationflags=0x{flags:08x}\n")
        log.flush()
        code = run_owned_tool(command, env, log, args.timeout)
        log.write(f"DONE exit={code}\n")
    print(f"DONE exit={code} log={args.log}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(worker(sys.argv[2:]) if sys.argv[1:2] == ["--worker"] else main())
