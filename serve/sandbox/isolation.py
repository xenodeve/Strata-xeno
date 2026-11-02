"""serve/sandbox/isolation.py - the operating system's wall around code the model wrote.

`Isolation.start(policy)` returns a `Box` or raises `IsolationError`. There is no unsandboxed fallback: a wall that
cannot be built stops the feature (rule I4 of docs/plans/2026-11-02-sandbox-workspace-design.md). Nothing here is wired
into the server yet.

Backends:

- `AppContainerBackend` (Windows): a new AppContainer profile per box, no capabilities, inside a Job Object. The box runs
  a private copy of CPython under the scratch root, so no ACL outside the scratch root ever changes.
- `MxcBackend` (Windows): only when `wxc-exec.exe` is already installed, and only if asked for by name: it has not passed
  the escape suite, and its schema has no memory, process or disk limit.
- `BubblewrapBackend` (Linux): `bwrap` plus rlimits.
- `NoneBackend`: no wall at all. Refuses on Windows; on Linux it needs STRATA_SANDBOX_THROWAWAY=1 (the Colab VM).

What a Job Object does not do on Windows, and what stands in for it: there is no disk quota without elevation, so
`disk_bytes` is a watcher that polls and kills. It stops a box, but only after the fact: a writer overshoots by what it
writes between two polls, and SetEndOfFile takes all it asks for in one call before any poll can see it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

CREATE_NO_WINDOW = 0x08000000
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
THROWAWAY_ENV = "STRATA_SANDBOX_THROWAWAY"
ROOT_ENV = "STRATA_SANDBOX_ROOT"
REPARSE_POINT = 0x400

# Everything a box may import beyond the standard library. Extra packages mean a new runtime, not a new ACL.
RUNTIME_PACKAGES = ("sympy", "numpy", "numpy.libs", "mpmath", "networkx", "pytest", "_pytest", "py.py", "pluggy",
                    "iniconfig", "packaging", "colorama", "pygments")
RUNTIME_SKIP = ("site-packages", "test", "tests", "idlelib", "tkinter", "turtledemo", "ensurepip", "__phello__")
RUNTIME_LAYOUT = 1


class IsolationError(RuntimeError):
    """The wall could not be built or could not be confirmed. Nothing ran."""


@dataclass(frozen=True)
class Policy:
    readonly: Sequence[str] = ()
    readwrite: Sequence[str] = ()
    denied: Sequence[str] = ()
    network: bool = False
    memory_bytes: int = 1 << 30
    seconds: float = 60.0
    processes: int = 32
    disk_bytes: int = 1 << 30
    output_bytes: int = 1 << 20
    cpu_percent: int = 0    # hard cap on the whole machine's CPU, 0 for none

    def __post_init__(self):
        for name in ("memory_bytes", "seconds", "processes", "disk_bytes", "output_bytes"):
            if getattr(self, name) <= 0:
                raise IsolationError(f"policy {name} must be above zero")
        if not 0 <= self.cpu_percent <= 100:
            raise IsolationError("policy cpu_percent must be 0 to 100")


@dataclass
class RunResult:
    exit_code: Optional[int]
    stdout: bytes
    stderr: bytes
    wall_ms: float
    killed: Optional[str] = None    # "timeout", "time", "output", "disk" or "kill"; None if it ended by itself
    truncated: bool = False


@dataclass
class Usage:
    wall_seconds: float
    cpu_seconds: float        # coarse: a program of some tens of ms can read 0.0 on Windows
    peak_memory_bytes: int    # Windows counts the allocation it refused: can read above the limit by that one request
    processes_total: int      # on Windows both counts include a console host per program, which takes no limit slot
    processes_active: int
    disk_bytes: int
    written_bytes: int


def _long(path) -> str:
    """A path Windows takes as written, for names a box can make and a plain path cannot reach: `con`, `name.`, and anything
    past 260 characters when long paths are off."""
    text = str(path)
    if sys.platform != "win32" or text.startswith("\\\\?\\"):
        return text
    return "\\\\?\\" + os.path.abspath(text)


def _walk(top) -> "iter":
    """(path, lstat, is_link) for everything under top. Never enters a link: a box can plant a junction to anywhere,
    and this code runs with the user's full rights over what it finds."""
    stack = [_long(top)]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                listed = list(entries)
        except OSError:
            continue
        for entry in listed:
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            linked = stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & REPARSE_POINT)
            yield entry.path, st, linked
            if stat.S_ISDIR(st.st_mode) and not linked:
                stack.append(entry.path)


def _refuse_links(top) -> None:
    for path, _, linked in _walk(top):
        if linked:
            raise IsolationError(f"{path} is a link, refusing to change ACLs through it")


def _remove(action: Callable[[str], None], path: str) -> None:
    try:
        action(path)
    except PermissionError:    # a box can mark what it leaves read-only, folders too
        try:
            os.chmod(path, stat.S_IWRITE)
            action(path)
        except OSError:
            pass
    except OSError:
        pass


def _remove_tree(top) -> None:
    """Delete top without following any link inside it. What cannot be deleted stays for the next sweep."""
    top = _long(top)
    if not os.path.lexists(top):
        return
    dirs = [top]
    for path, st, _ in _walk(top):
        if stat.S_ISDIR(st.st_mode):
            dirs.append(path)
        else:
            _remove(os.unlink, path)
    for path in reversed(dirs):
        _remove(os.rmdir, path)


class Box:
    """A running wall. `run` starts a program inside it, `kill` ends everything in it and removes its files, and it stays
    usable between runs: a process a run leaves behind keeps running until `kill` or a limit ends it."""

    backend = ""
    isolated = True
    policy: Policy
    workdir: Path
    tmpdir: Path

    def run(self, argv: Sequence[str], stdin: Union[bytes, str, None] = None, timeout: Optional[float] = None) -> RunResult:
        raise NotImplementedError

    def popen(self, argv: Sequence[str]):
        """A program that stays, for a kernel that keeps its names between blocks: text `stdin` and `stdout`, `kill` and `wait`, as `subprocess.Popen` has.
        The box's limits still hold. Its output has no cap of its own, because the reader takes one line at a time and owns what it keeps."""
        raise NotImplementedError

    def kill(self) -> None:
        raise NotImplementedError

    def usage(self) -> Usage:
        raise NotImplementedError

    def __enter__(self) -> "Box":
        return self

    def __exit__(self, *exc) -> None:
        self.kill()

    def _start_watch(self) -> None:
        self._began = time.monotonic()
        self._deadline = self._began + self.policy.seconds
        self._stopping = threading.Event()
        self._swap = threading.Lock()    # held while reading the job, which a kill can replace
        self._reason: Optional[str] = None
        self._watcher = threading.Thread(target=self._watch, daemon=True, name="sandbox-watch")
        self._watcher.start()

    def _watch(self) -> None:
        """A box with no watcher has no disk limit and no clock, so a failure here ends the box and `run` then refuses."""
        try:
            self._enforce()
        except Exception:
            self._terminate("watch")

    def _enforce(self) -> None:
        """The wall clock and the disk ceiling. The walk is the expensive part, so it runs when the job has written a
        real amount, and otherwise at most 5 % of the time; the slow cadence is what sees SetEndOfFile, which writes nothing."""
        step = max(1 << 20, self.policy.disk_bytes // 16)
        walked_at, seen, gap = 0.0, 0, 0.1
        while not self._stopping.wait(0.02):
            now = time.monotonic()
            if now >= self._deadline:
                self._terminate("time")
                return
            with self._swap:
                if not self._alive():
                    continue
                written = self._written()
                if written - seen < step and now - walked_at < gap:
                    continue
                used = self._disk_used()
            done = time.monotonic()
            walked_at, seen, gap = done, written, max(0.1, (done - now) * 20)
            if used > self.policy.disk_bytes:
                self._terminate("disk")

    def _still_open(self) -> float:
        """The seconds this box has left, or IsolationError."""
        if self._closed:
            raise IsolationError("the box is closed")
        left = self._deadline - time.monotonic()
        if left <= 0:
            raise IsolationError("the box is out of time")
        if not self._watcher.is_alive():
            raise IsolationError("the box's watcher has stopped, so its limits no longer hold")
        return left

    def _stop_watch(self) -> None:
        self._stopping.set()
        if self._watcher is not threading.current_thread():
            self._watcher.join(2)

    def _terminate(self, reason: str) -> None:
        raise NotImplementedError

    def _alive(self) -> bool:
        raise NotImplementedError

    def _written(self) -> int:
        return 0

    def _disk_used(self) -> int:
        raise NotImplementedError


if sys.platform == "win32":
    import ctypes
    from ctypes import byref, sizeof, wintypes as w

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _adv = ctypes.WinDLL("advapi32", use_last_error=True)
    _uenv = ctypes.WinDLL("userenv", use_last_error=True)
    _sh32 = ctypes.WinDLL("shell32", use_last_error=True)
    _u32 = ctypes.WinDLL("user32", use_last_error=True)

    _WINSTA_BOX = 0x20123       # enumerate, read attributes, atoms: no clipboard, no screen, no new desktops, no exit
    _DESKTOP_BOX = 0x20087      # its own windows: no hooks, no journal, no switching
    _WINSTA_OWNER = 0xF037F
    _DESKTOP_OWNER = 0xF01FF
    _HRESULT_EXISTS = 0x800700B7
    _ALL_APP_PACKAGES = "S-1-15-2-1"
    _PACKAGE_SID_PREFIX = "S-1-15-2-"
    _SE_FILE_OBJECT = 1
    _DACL_INFO = 4
    _PROTECTED_DACL = 0x80000000
    _ACL_REVISION = 2
    _GRANT, _REVOKE = 1, 4
    _INHERIT = 3    # objects and containers
    _STAT_ONLY = 0x00100080     # FILE_READ_ATTRIBUTES and SYNCHRONIZE: no listing
    _READ_EXECUTE = 0x1200A9
    _MODIFY = 0x1301BF
    _ALLOWED_ACE = 0
    _INHERIT_ONLY_ACE = 0x08
    _INHERITED_ACE = 0x10

    _ATTR_HANDLE_LIST = 0x20002
    _ATTR_SECURITY_CAPABILITIES = 0x20009
    _ATTR_JOB_LIST = 0x2000D
    _EXTENDED_STARTUPINFO = 0x00080000
    _UNICODE_ENVIRONMENT = 0x00000400
    _SUSPENDED = 0x00000004
    _USE_STDHANDLES = 0x00000100
    _WAIT_OBJECT_0 = 0

    _LIMIT_JOB_TIME = 0x00000004
    _LIMIT_ACTIVE_PROCESS = 0x00000008
    _LIMIT_PRIORITY_CLASS = 0x00000020
    _LIMIT_JOB_MEMORY = 0x00000200
    _LIMIT_DIE_ON_UNHANDLED = 0x00000400
    _LIMIT_KILL_ON_CLOSE = 0x00002000
    _JOB_UI_ALL = 0xFF    # handles, clipboard both ways, system parameters, display, atoms, desktops, exit
    _JOB_UI_INFO, _JOB_ACCOUNTING_INFO, _JOB_LIMIT_INFO, _JOB_CPU_INFO = 4, 8, 9, 15
    _CPU_ENABLE_HARD_CAP = 0x1 | 0x4
    _KILLED_EXIT_CODE = 0xE5C0DEAD
    _STATUS_QUOTA_EXCEEDED = 0xC0000044

    _TOKEN_QUERY = 0x0008
    _TOKEN_CAPABILITIES, _TOKEN_IS_APP_CONTAINER, _TOKEN_APP_CONTAINER_SID = 30, 29, 31
    _FIND_STREAM_STANDARD = 0

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                      "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", w.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _BasicAccounting(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_longlong), ("TotalKernelTime", ctypes.c_longlong),
                    ("ThisPeriodTotalUserTime", ctypes.c_longlong), ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                    ("TotalPageFaultCount", w.DWORD), ("TotalProcesses", w.DWORD), ("ActiveProcesses", w.DWORD),
                    ("TotalTerminatedProcesses", w.DWORD)]

    class _AccountingAndIo(ctypes.Structure):
        _fields_ = [("BasicInfo", _BasicAccounting), ("IoInfo", _IoCounters)]

    class _CpuRate(ctypes.Structure):
        _fields_ = [("ControlFlags", w.DWORD), ("CpuRate", w.DWORD)]

    class _SecurityCapabilities(ctypes.Structure):
        _fields_ = [("AppContainerSid", w.LPVOID), ("Capabilities", w.LPVOID), ("CapabilityCount", w.DWORD),
                    ("Reserved", w.DWORD)]

    class _StartupInfo(ctypes.Structure):
        _fields_ = [("cb", w.DWORD), ("lpReserved", w.LPWSTR), ("lpDesktop", w.LPWSTR), ("lpTitle", w.LPWSTR),
                    ("dwX", w.DWORD), ("dwY", w.DWORD), ("dwXSize", w.DWORD), ("dwYSize", w.DWORD),
                    ("dwXCountChars", w.DWORD), ("dwYCountChars", w.DWORD), ("dwFillAttribute", w.DWORD),
                    ("dwFlags", w.DWORD), ("wShowWindow", w.WORD), ("cbReserved2", w.WORD), ("lpReserved2", w.LPVOID),
                    ("hStdInput", w.HANDLE), ("hStdOutput", w.HANDLE), ("hStdError", w.HANDLE)]

    class _StartupInfoEx(ctypes.Structure):
        _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", w.LPVOID)]

    class _ProcessInformation(ctypes.Structure):
        _fields_ = [("hProcess", w.HANDLE), ("hThread", w.HANDLE), ("dwProcessId", w.DWORD), ("dwThreadId", w.DWORD)]

    class _Trustee(ctypes.Structure):
        _fields_ = [("pMultipleTrustee", w.LPVOID), ("MultipleTrusteeOperation", ctypes.c_int),
                    ("TrusteeForm", ctypes.c_int), ("TrusteeType", ctypes.c_int), ("ptstrName", w.LPVOID)]

    class _ExplicitAccess(ctypes.Structure):
        _fields_ = [("grfAccessPermissions", w.DWORD), ("grfAccessMode", ctypes.c_int), ("grfInheritance", w.DWORD),
                    ("Trustee", _Trustee)]

    class _AceHeader(ctypes.Structure):
        _fields_ = [("AceType", ctypes.c_ubyte), ("AceFlags", ctypes.c_ubyte), ("AceSize", w.WORD)]

    class _AllowedAce(ctypes.Structure):
        _fields_ = [("Header", _AceHeader), ("Mask", w.DWORD), ("SidStart", w.DWORD)]

    class _AclSizeInformation(ctypes.Structure):
        _fields_ = [("AceCount", w.DWORD), ("AclBytesInUse", w.DWORD), ("AclBytesFree", w.DWORD)]

    class _StreamData(ctypes.Structure):
        _fields_ = [("StreamSize", ctypes.c_longlong), ("cStreamName", w.WCHAR * 296)]

    class _TokenGroupsHead(ctypes.Structure):
        _fields_ = [("GroupCount", w.DWORD)]

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = [("nLength", w.DWORD), ("lpSecurityDescriptor", w.LPVOID), ("bInheritHandle", w.BOOL)]

    _u32.CreateWindowStationW.argtypes = (w.LPCWSTR, w.DWORD, w.DWORD, ctypes.POINTER(_SecurityAttributes))
    _u32.CreateWindowStationW.restype = w.HANDLE
    _u32.CreateDesktopW.argtypes = (w.LPCWSTR, w.LPCWSTR, w.LPVOID, w.DWORD, w.DWORD,
                                    ctypes.POINTER(_SecurityAttributes))
    _u32.CreateDesktopW.restype = w.HANDLE
    _u32.CloseDesktop.argtypes = (w.HANDLE,)
    _u32.GetProcessWindowStation.restype = w.HANDLE
    _u32.SetProcessWindowStation.argtypes = (w.HANDLE,)
    _u32.GetUserObjectInformationW.argtypes = (w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD, ctypes.POINTER(w.DWORD))
    _adv.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (w.LPCWSTR, w.DWORD, ctypes.POINTER(w.LPVOID),
                                                                          w.LPVOID)

    _uenv.CreateAppContainerProfile.restype = ctypes.c_long
    _uenv.CreateAppContainerProfile.argtypes = (w.LPCWSTR, w.LPCWSTR, w.LPCWSTR, w.LPVOID, w.DWORD,
                                                ctypes.POINTER(w.LPVOID))
    _uenv.DeleteAppContainerProfile.restype = ctypes.c_long
    _uenv.DeleteAppContainerProfile.argtypes = (w.LPCWSTR,)
    _uenv.GetAppContainerFolderPath.restype = ctypes.c_long
    _uenv.GetAppContainerFolderPath.argtypes = (w.LPCWSTR, ctypes.POINTER(w.LPWSTR))

    _adv.ConvertSidToStringSidW.argtypes = (w.LPVOID, ctypes.POINTER(w.LPWSTR))
    _adv.ConvertStringSidToSidW.argtypes = (w.LPCWSTR, ctypes.POINTER(w.LPVOID))
    _adv.EqualSid.argtypes = (w.LPVOID, w.LPVOID)
    _adv.FreeSid.argtypes = (w.LPVOID,)
    _adv.FreeSid.restype = w.LPVOID
    _adv.GetNamedSecurityInfoW.restype = w.DWORD
    _adv.GetNamedSecurityInfoW.argtypes = (w.LPCWSTR, ctypes.c_int, w.DWORD, w.LPVOID, w.LPVOID,
                                           ctypes.POINTER(w.LPVOID), w.LPVOID, ctypes.POINTER(w.LPVOID))
    _adv.SetNamedSecurityInfoW.restype = w.DWORD
    _adv.SetNamedSecurityInfoW.argtypes = (w.LPWSTR, ctypes.c_int, w.DWORD, w.LPVOID, w.LPVOID, w.LPVOID, w.LPVOID)
    _adv.SetEntriesInAclW.restype = w.DWORD
    _adv.SetEntriesInAclW.argtypes = (w.ULONG, ctypes.POINTER(_ExplicitAccess), w.LPVOID, ctypes.POINTER(w.LPVOID))
    _adv.GetAclInformation.argtypes = (w.LPVOID, w.LPVOID, w.DWORD, ctypes.c_int)
    _adv.GetAce.argtypes = (w.LPVOID, w.DWORD, ctypes.POINTER(w.LPVOID))
    _adv.InitializeAcl.argtypes = (w.LPVOID, w.DWORD, w.DWORD)
    _adv.AddAce.argtypes = (w.LPVOID, w.DWORD, w.DWORD, w.LPVOID, w.DWORD)
    _adv.OpenProcessToken.argtypes = (w.HANDLE, w.DWORD, ctypes.POINTER(w.HANDLE))
    _adv.GetTokenInformation.argtypes = (w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD, ctypes.POINTER(w.DWORD))
    _adv.OpenSCManagerW.argtypes = (w.LPCWSTR, w.LPCWSTR, w.DWORD)
    _adv.OpenSCManagerW.restype = w.LPVOID
    _adv.OpenServiceW.argtypes = (w.LPVOID, w.LPCWSTR, w.DWORD)
    _adv.OpenServiceW.restype = w.LPVOID
    _adv.QueryServiceStatus.argtypes = (w.LPVOID, w.LPVOID)
    _adv.CloseServiceHandle.argtypes = (w.LPVOID,)

    _k32.LocalFree.argtypes = (w.LPVOID,)
    _k32.LocalFree.restype = w.LPVOID
    _k32.CloseHandle.argtypes = (w.HANDLE,)
    _k32.CreateJobObjectW.restype = w.HANDLE
    _k32.CreateJobObjectW.argtypes = (w.LPVOID, w.LPCWSTR)
    _k32.SetInformationJobObject.argtypes = (w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD)
    _k32.QueryInformationJobObject.argtypes = (w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD, ctypes.POINTER(w.DWORD))
    _k32.TerminateJobObject.argtypes = (w.HANDLE, w.UINT)
    _k32.IsProcessInJob.argtypes = (w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL))
    _k32.InitializeProcThreadAttributeList.argtypes = (w.LPVOID, w.DWORD, w.DWORD, ctypes.POINTER(ctypes.c_size_t))
    _k32.UpdateProcThreadAttribute.argtypes = (w.LPVOID, w.DWORD, ctypes.c_size_t, w.LPVOID, ctypes.c_size_t, w.LPVOID,
                                               w.LPVOID)
    _k32.DeleteProcThreadAttributeList.argtypes = (w.LPVOID,)
    _k32.CreateProcessW.argtypes = (w.LPCWSTR, w.LPWSTR, w.LPVOID, w.LPVOID, w.BOOL, w.DWORD, w.LPVOID, w.LPCWSTR,
                                    ctypes.POINTER(_StartupInfoEx), ctypes.POINTER(_ProcessInformation))
    _k32.ResumeThread.argtypes = (w.HANDLE,)
    _k32.ResumeThread.restype = w.DWORD
    _k32.TerminateProcess.argtypes = (w.HANDLE, w.UINT)
    _k32.WaitForSingleObject.argtypes = (w.HANDLE, w.DWORD)
    _k32.WaitForSingleObject.restype = w.DWORD
    _k32.GetExitCodeProcess.argtypes = (w.HANDLE, ctypes.POINTER(w.DWORD))
    _k32.CreatePipe.argtypes = (ctypes.POINTER(w.HANDLE), ctypes.POINTER(w.HANDLE), w.LPVOID, w.DWORD)
    _k32.SetHandleInformation.argtypes = (w.HANDLE, w.DWORD, w.DWORD)
    _k32.PeekNamedPipe.argtypes = (w.HANDLE, w.LPVOID, w.DWORD, w.LPVOID, ctypes.POINTER(w.DWORD), w.LPVOID)
    _k32.ReadFile.argtypes = (w.HANDLE, w.LPVOID, w.DWORD, ctypes.POINTER(w.DWORD), w.LPVOID)
    _k32.WriteFile.argtypes = (w.HANDLE, w.LPCVOID, w.DWORD, ctypes.POINTER(w.DWORD), w.LPVOID)
    _k32.OpenProcess.argtypes = (w.DWORD, w.BOOL, w.DWORD)
    _k32.OpenProcess.restype = w.HANDLE
    _k32.SetErrorMode.argtypes = (w.UINT,)
    _k32.SetErrorMode.restype = w.UINT
    _k32.FindFirstStreamW.argtypes = (w.LPCWSTR, ctypes.c_int, w.LPVOID, w.DWORD)
    _k32.FindFirstStreamW.restype = w.HANDLE
    _k32.FindNextStreamW.argtypes = (w.HANDLE, w.LPVOID)
    _k32.FindClose.argtypes = (w.HANDLE,)

    _INVALID_HANDLE = w.HANDLE(-1).value

    def _check(ok, what):
        if not ok:
            raise IsolationError(f"{what}: {ctypes.WinError(ctypes.get_last_error())}")
        return ok

    def _sid_text(sid) -> str:
        text = w.LPWSTR()
        _check(_adv.ConvertSidToStringSidW(sid, byref(text)), "ConvertSidToStringSid")
        out = text.value
        _k32.LocalFree(text)
        return out

    def _sid_from(text: str):
        sid = w.LPVOID()
        _check(_adv.ConvertStringSidToSidW(text, byref(sid)), "ConvertStringSidToSid")
        return sid

    def _edit_dacl(path, sid, mode: int, rights: int, inherit: int) -> None:
        """Add one entry to a path's DACL and keep the rest. Windows spreads an inheritable entry to what is inside, so
        every caller has checked the tree holds no link first."""
        old, descriptor, new = w.LPVOID(), w.LPVOID(), w.LPVOID()
        status = _adv.GetNamedSecurityInfoW(str(path), _SE_FILE_OBJECT, _DACL_INFO, None, None, byref(old), None,
                                            byref(descriptor))
        if status:
            raise IsolationError(f"reading the ACL of {path}: {ctypes.WinError(status)}")
        try:
            entry = _ExplicitAccess(rights, mode, inherit, _Trustee(None, 0, 0, 0, sid))
            status = _adv.SetEntriesInAclW(1, byref(entry), old, byref(new))
            if status:
                raise IsolationError(f"building the ACL of {path}: {ctypes.WinError(status)}")
            status = _adv.SetNamedSecurityInfoW(str(path), _SE_FILE_OBJECT, _DACL_INFO, None, None, new, None)
            if status:
                raise IsolationError(f"writing the ACL of {path}: {ctypes.WinError(status)}")
        finally:
            _k32.LocalFree(new)
            _k32.LocalFree(descriptor)

    def _aces(dacl) -> "iter":
        """(ace pointer, its header, whether it allows an AppContainer) for each entry of a DACL."""
        info = _AclSizeInformation()
        _check(_adv.GetAclInformation(dacl, byref(info), sizeof(info), 2), "GetAclInformation")
        for index in range(info.AceCount):
            ace = w.LPVOID()
            _check(_adv.GetAce(dacl, index, byref(ace)), "GetAce")
            entry = ctypes.cast(ace, ctypes.POINTER(_AllowedAce)).contents
            is_allow = entry.Header.AceType == _ALLOWED_ACE
            sid = ace.value + _AllowedAce.SidStart.offset
            yield ace, entry.Header, is_allow and _sid_text(sid).startswith(_PACKAGE_SID_PREFIX)

    def _open_to_containers(path) -> bool:
        """True if the path's own DACL lets an AppContainer in. Outside the scratch root such a path cannot be denied, since
        that would mean changing an ACL there."""
        dacl, descriptor = w.LPVOID(), w.LPVOID()
        status = _adv.GetNamedSecurityInfoW(str(path), _SE_FILE_OBJECT, _DACL_INFO, None, None, byref(dacl), None,
                                            byref(descriptor))
        if status:
            return False
        try:
            if not dacl:
                return True
            return any(opens and not header.AceFlags & _INHERIT_ONLY_ACE for _, header, opens in _aces(dacl))
        finally:
            _k32.LocalFree(descriptor)

    def _withdraw(path) -> None:
        """Deny a path by taking away every AppContainer's allow entry and cutting it off from its parent's. A deny entry
        is not an option: on Windows 26300 a box read a file straight through a deny for its own SID and for ALL APPLICATION
        PACKAGES, both first in the DACL. It lasts past the box on purpose: giving the path back would reopen it to any
        other box that holds a grant on the parent."""
        dacl, descriptor = w.LPVOID(), w.LPVOID()
        status = _adv.GetNamedSecurityInfoW(str(path), _SE_FILE_OBJECT, _DACL_INFO, None, None, byref(dacl), None,
                                            byref(descriptor))
        if status:
            raise IsolationError(f"reading the ACL of {path}: {ctypes.WinError(status)}")
        try:
            size = _AclSizeInformation()
            _check(_adv.GetAclInformation(dacl, byref(size), sizeof(size), 2), "GetAclInformation")
            kept = ctypes.create_string_buffer(size.AclBytesInUse + size.AclBytesFree + 8)
            _check(_adv.InitializeAcl(kept, sizeof(kept), _ACL_REVISION), "InitializeAcl")
            for ace, header, opens in _aces(dacl):
                if not opens:
                    header.AceFlags &= ~_INHERITED_ACE & 0xFF
                    _check(_adv.AddAce(kept, _ACL_REVISION, 0xFFFFFFFF, ace, header.AceSize), "AddAce")
            status = _adv.SetNamedSecurityInfoW(str(path), _SE_FILE_OBJECT, _DACL_INFO | _PROTECTED_DACL, None, None, kept,
                                                None)
            if status:
                raise IsolationError(f"writing the ACL of {path}: {ctypes.WinError(status)}")
        finally:
            _k32.LocalFree(descriptor)
        if _open_to_containers(path):
            raise IsolationError(f"{path} is still open to AppContainers after denying it")

    def _stream_bytes(path: str) -> int:
        """The file's size in all its streams. The size in a directory listing goes stale while a handle is open, and it
        never counts `name:stream`, so neither can serve a quota."""
        data = _StreamData()
        found = _k32.FindFirstStreamW(_long(path), _FIND_STREAM_STANDARD, byref(data), 0)
        if found == _INVALID_HANDLE:
            return 0
        try:
            total = data.StreamSize
            while _k32.FindNextStreamW(found, byref(data)):
                total += data.StreamSize
            return total
        finally:
            _k32.FindClose(found)

    def _disk_used(roots: Sequence[Path]) -> int:
        total = 0
        for root in roots:
            total += _stream_bytes(str(root))
            for path, _, linked in _walk(root):
                if not linked:
                    total += _stream_bytes(path)
        return total

    def _service_running(name: str) -> bool:
        manager = _adv.OpenSCManagerW(None, None, 0x0001)
        if not manager:
            return False
        try:
            service = _adv.OpenServiceW(manager, name, 0x0004)
            if not service:
                return False
            try:
                status = (w.DWORD * 7)()
                return bool(_adv.QueryServiceStatus(service, status)) and status[1] == 4
            finally:
                _adv.CloseServiceHandle(service)
        finally:
            _adv.CloseServiceHandle(manager)

    def _user_sid_text() -> str:
        token = w.HANDLE()
        _check(_adv.OpenProcessToken(w.HANDLE(-1), _TOKEN_QUERY, byref(token)), "OpenProcessToken")
        try:
            buffer, got = ctypes.create_string_buffer(256), w.DWORD()
            _check(_adv.GetTokenInformation(token, 1, buffer, sizeof(buffer), byref(got)), "reading the token's user")
            return _sid_text(ctypes.cast(buffer, ctypes.POINTER(w.LPVOID)).contents)
        finally:
            _k32.CloseHandle(token)

    def _attributes(sddl: str) -> _SecurityAttributes:
        descriptor = w.LPVOID()
        _check(_adv.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1, byref(descriptor), None), "SDDL")
        return _SecurityAttributes(sizeof(_SecurityAttributes), descriptor, False)

    _station_lock = threading.Lock()
    _station = [None, "", ""]    # handle, name, the user's SID

    def _object_name(handle) -> str:
        name, got = ctypes.create_unicode_buffer(256), w.DWORD()
        _check(_u32.GetUserObjectInformationW(handle, 2, name, sizeof(name), byref(got)), "reading a station's name")
        return name.value

    def _box_desktop(sid_text: str) -> Tuple[int, str]:
        """(handle, station\\desktop): where the box's windows go, which is nowhere the developer can see, and a clipboard
        that is not theirs. Windows gives a non-admin a station only with no name, one per logon session, so every box of
        this process shares it; each has a desktop that only its own SID may open."""
        with _station_lock:
            if _station[0] is None:
                _station[2] = _user_sid_text()
                security = _attributes(f"D:(A;;GA;;;{_station[2]})(A;;GA;;;SY)(A;;{_WINSTA_BOX:#x};;;{_ALL_APP_PACKAGES})")
                handle = _u32.CreateWindowStationW(None, 0, _WINSTA_OWNER, byref(security))
                _k32.LocalFree(security.lpSecurityDescriptor)
                _station[0] = _check(handle, "CreateWindowStation")
                _station[1] = _object_name(handle)
            name = f"box-{uuid.uuid4().hex[:12]}"
            security = _attributes(f"D:(A;;GA;;;{_station[2]})(A;;GA;;;SY)(A;;{_DESKTOP_BOX:#x};;;{sid_text})")
            previous = _u32.GetProcessWindowStation()
            _check(_u32.SetProcessWindowStation(_station[0]), "SetProcessWindowStation")
            try:
                desktop = _u32.CreateDesktopW(name, None, None, 0, _DESKTOP_OWNER, byref(security))
                failed = ctypes.get_last_error()
            finally:
                _u32.SetProcessWindowStation(previous)
                _k32.LocalFree(security.lpSecurityDescriptor)
            if not desktop:
                raise IsolationError(f"CreateDesktop: {ctypes.WinError(failed)}")
            return desktop, f"{_station[1]}\\{name}"

    def _pipe(child_reads: bool) -> Tuple[int, int]:
        """(child's end, our end). Only the child's end is inheritable, and the attribute list names it, so nothing else
        of this process reaches the box."""
        read_end, write_end = w.HANDLE(), w.HANDLE()
        _check(_k32.CreatePipe(byref(read_end), byref(write_end), None, 1 << 16), "CreatePipe")
        child, ours = (read_end, write_end) if child_reads else (write_end, read_end)
        _check(_k32.SetHandleInformation(child, 1, 1), "SetHandleInformation")
        return child.value, ours.value

    def _drain(handle, sink: bytearray, room: int) -> Tuple[bool, int]:
        """Take what is in the pipe now without waiting. (writer still there, bytes taken); only `room` of them are kept. It
        stops once past `room` and not when the pipe is empty: a fast writer can keep it from ever being empty, and the
        caller could then never reach the cap."""
        chunk = ctypes.create_string_buffer(1 << 16)
        taken = 0
        while taken <= room:
            ready, got = w.DWORD(), w.DWORD()
            if not _k32.PeekNamedPipe(handle, None, 0, None, byref(ready), None):
                return False, taken
            if not ready.value:
                return True, taken
            if not _k32.ReadFile(handle, chunk, min(sizeof(chunk), ready.value), byref(got), None):
                return False, taken
            keep = max(0, min(got.value, room - taken))
            sink += chunk.raw[:keep]
            taken += got.value
        return True, taken

    class _Stays:
        """A program left running in a box: what `Box.popen` gives."""

        def __init__(self, process, stdin, stdout, read_err):
            self._process = process
            self.stdin = stdin
            self.stdout = stdout
            threading.Thread(target=self._discard, args=(read_err,), daemon=True).start()

        @staticmethod
        def _discard(handle) -> None:
            chunk = ctypes.create_string_buffer(1 << 12)
            got = w.DWORD()
            while _k32.ReadFile(handle, chunk, sizeof(chunk), byref(got), None):
                pass
            _k32.CloseHandle(handle)

        def kill(self) -> None:
            if self._process is not None:
                _k32.TerminateProcess(self._process.hProcess, _KILLED_EXIT_CODE)

        def wait(self, timeout: Optional[float] = None) -> None:
            if self._process is None:
                return
            _k32.WaitForSingleObject(self._process.hProcess, 0xFFFFFFFF if timeout is None else int(timeout * 1000))
            _k32.CloseHandle(self._process.hProcess)
            self._process = None
            for stream in (self.stdin, self.stdout):
                try:
                    stream.close()
                except OSError:
                    pass

    def _feed(handle: int, data: bytes) -> None:
        written, offset = w.DWORD(), 0
        while offset < len(data):
            part = data[offset:offset + (1 << 16)]
            if not _k32.WriteFile(handle, part, len(part), byref(written), None):
                break
            offset += written.value
        _k32.CloseHandle(handle)

    class _Job:
        """One unnamed job, so no box can open it by name. Kill-on-close ties it to this process's life. A terminated job
        refuses every new process for good, even when empty, so a box that goes on after a kill gets a new job."""

        def __init__(self, policy: Policy, cpu_spent: float = 0.0):
            self.ended = False
            self.handle = _check(_k32.CreateJobObjectW(None, None), "CreateJobObject")
            try:
                self._apply(policy, max(policy.seconds - cpu_spent, 0.001))
            except IsolationError:
                _k32.CloseHandle(self.handle)
                self.handle = None
                raise

        def _apply(self, policy: Policy, cpu_seconds: float) -> None:
            limits = _ExtendedLimits()
            basic = limits.BasicLimitInformation
            basic.LimitFlags = (_LIMIT_KILL_ON_CLOSE | _LIMIT_DIE_ON_UNHANDLED | _LIMIT_ACTIVE_PROCESS | _LIMIT_JOB_MEMORY
                                | _LIMIT_JOB_TIME | _LIMIT_PRIORITY_CLASS)
            basic.ActiveProcessLimit = policy.processes
            basic.PerJobUserTimeLimit = self.cpu_limit = int(cpu_seconds * 10_000_000)
            basic.PriorityClass = BELOW_NORMAL_PRIORITY_CLASS
            limits.JobMemoryLimit = policy.memory_bytes
            _check(_k32.SetInformationJobObject(self.handle, _JOB_LIMIT_INFO, byref(limits), sizeof(limits)),
                   "setting the job's limits")
            back = _ExtendedLimits()
            _check(_k32.QueryInformationJobObject(self.handle, _JOB_LIMIT_INFO, byref(back), sizeof(back), None),
                   "reading the job's limits back")
            if (back.BasicLimitInformation.LimitFlags & basic.LimitFlags != basic.LimitFlags
                    or back.BasicLimitInformation.ActiveProcessLimit != policy.processes
                    or back.JobMemoryLimit != policy.memory_bytes):
                raise IsolationError("the job did not take its limits")
            ui = w.DWORD(_JOB_UI_ALL)
            _check(_k32.SetInformationJobObject(self.handle, _JOB_UI_INFO, byref(ui), sizeof(ui)),
                   "setting the job's UI limits")
            if policy.cpu_percent:
                rate = _CpuRate(_CPU_ENABLE_HARD_CAP, policy.cpu_percent * 100)
                _check(_k32.SetInformationJobObject(self.handle, _JOB_CPU_INFO, byref(rate), sizeof(rate)),
                       "setting the job's CPU cap")

        def accounting(self) -> _AccountingAndIo:
            info = _AccountingAndIo()
            _check(_k32.QueryInformationJobObject(self.handle, _JOB_ACCOUNTING_INFO, byref(info), sizeof(info), None),
                   "reading the job's accounting")
            return info

        def out_of_cpu(self) -> bool:
            """A coarse backstop and not a limit to design on: Windows checks it only now and then. With 2 s allowed, four
            spinners used 14.9 and 25.5 s of CPU and were stopped 3.9 and 6.7 s in (2026-11-02). The wall clock is what
            bounds a box."""
            return self.accounting().BasicInfo.TotalUserTime >= self.cpu_limit

        def peak_memory(self) -> int:
            info = _ExtendedLimits()
            _check(_k32.QueryInformationJobObject(self.handle, _JOB_LIMIT_INFO, byref(info), sizeof(info), None),
                   "reading the job's peak")
            return info.PeakJobMemoryUsed

        def terminate(self) -> None:
            if self.handle:
                self.ended = True
                _k32.TerminateJobObject(self.handle, _KILLED_EXIT_CODE)

        def wait_empty(self, seconds: float) -> bool:
            end = time.monotonic() + seconds
            while self.accounting().BasicInfo.ActiveProcesses:
                if time.monotonic() >= end:
                    return False
                time.sleep(0.001)
            return True

        def close(self) -> None:
            if self.handle:
                _k32.CloseHandle(self.handle)
                self.handle = None

    class _Profile:
        """An AppContainer of its own for every box: boxes with one SID could open each other's folders."""

        def __init__(self, name: str):
            self.name = name
            self.sid = w.LPVOID()
            status = _uenv.CreateAppContainerProfile(name, name, name, None, 0, byref(self.sid)) & 0xFFFFFFFF
            if status == _HRESULT_EXISTS:
                raise IsolationError(f"AppContainer profile {name} already exists")
            if status:
                raise IsolationError(f"CreateAppContainerProfile: HRESULT {status:#010x}")
            self.sid_text = _sid_text(self.sid)
            folder = w.LPWSTR()
            status = _uenv.GetAppContainerFolderPath(self.sid_text, byref(folder)) & 0xFFFFFFFF
            if status:
                self.close()
                raise IsolationError(f"GetAppContainerFolderPath: HRESULT {status:#010x}")
            self.folder = Path(folder.value)
            ctypes.windll.ole32.CoTaskMemFree(folder)

        def close(self) -> None:
            if self.sid:
                _adv.FreeSid(self.sid)
                self.sid = None
                _uenv.DeleteAppContainerProfile(self.name)

    class Scratch:
        """The one tree whose ACLs this code touches. A folder it did not make is refused, so a wrong setting cannot turn a
        user's folder into a scratch root."""

        MARK = ".strata-sandbox-root"

        def __init__(self, root=None):
            local = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
            self.root = Path(os.path.abspath(root or os.environ.get(ROOT_ENV) or os.path.join(local, "Strata", "sandbox")))
            if self.root.exists():
                if not (self.root / self.MARK).is_file():
                    raise IsolationError(f"{self.root} exists and is not a scratch root this code made")
            else:
                self.root.mkdir(parents=True)
                (self.root / self.MARK).write_text("Scratch root of the Strata sandbox. Safe to delete.\n", encoding="utf-8")
            if os.path.normcase(os.path.realpath(self.root)) != os.path.normcase(str(self.root)):
                raise IsolationError(f"{self.root} goes through a link")
            for part in ("boxes", "runtime", "shared"):
                (self.root / part).mkdir(exist_ok=True)

        def make(self, name: str) -> Path:
            """A folder a Policy may name in readonly, readwrite or denied."""
            path = self.root / "shared" / name
            path.mkdir(parents=True, exist_ok=True)
            return path

        def inside(self, path) -> Optional[Path]:
            """The path if it is really under the scratch root; None if it is outside. A path that reaches the root only
            through a link is refused."""
            plain = os.path.abspath(path)
            if not os.path.normcase(plain).startswith(os.path.normcase(str(self.root)) + os.sep):
                real = os.path.normcase(os.path.realpath(plain))
                if real.startswith(os.path.normcase(str(self.root)) + os.sep):
                    raise IsolationError(f"{path} reaches the scratch root through a link")
                return None
            if os.path.normcase(os.path.realpath(plain)) != os.path.normcase(plain):
                raise IsolationError(f"{path} goes through a link")
            return Path(plain)

        def sweep(self) -> None:
            """Remove what a crashed server left: its job died with it, its folders, profiles and half-built runtimes did not."""
            def gone(pid: str) -> bool:
                if not pid.isdigit() or int(pid) == os.getpid():
                    return False
                process = _k32.OpenProcess(0x00100000, False, int(pid))
                if not process:
                    return True
                ended = _k32.WaitForSingleObject(process, 0) == _WAIT_OBJECT_0    # any open handle keeps a dead one openable
                _k32.CloseHandle(process)
                return ended

            for entry in (self.root / "boxes").iterdir():
                parts = entry.name.split(".")
                if len(parts) == 4 and gone(parts[2]):
                    _remove_tree(entry)
                    _uenv.DeleteAppContainerProfile(entry.name)
            for entry in (self.root / "runtime").iterdir():
                parts = entry.name.split("-")
                if parts[0] == ".building" and gone(parts[1]):
                    _remove_tree(entry)

    def _site_dirs() -> List[Path]:
        import site
        found = [Path(p) for p in site.getsitepackages()]
        found.append(Path(site.getusersitepackages()))
        return [p for p in found if p.is_dir()]

    def _runtime_key(packages: Sequence[str]) -> str:
        text = "\n".join((sys.version, str(RUNTIME_LAYOUT), os.path.realpath(sys.base_prefix), *packages))
        return f"py{sys.version_info[0]}{sys.version_info[1]}-{hashlib.sha256(text.encode()).hexdigest()[:12]}"

    def build_runtime(scratch: Scratch, packages: Sequence[str] = RUNTIME_PACKAGES) -> Path:
        """A private CPython under the scratch root, built once. Every AppContainer may read it, so a new box needs no ACL
        work on thousands of files: the entry goes on the empty root first and each copied file inherits it. Copies and
        not hard links: a link shares its ACL with the user's own file."""
        key = _runtime_key(packages)
        final = scratch.root / "runtime" / key
        if (final / "manifest.json").is_file():
            return final
        building = scratch.root / "runtime" / f".building-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        building.mkdir()
        try:
            everyone = _sid_from(_ALL_APP_PACKAGES)
            try:
                _edit_dacl(building, everyone, _GRANT, _READ_EXECUTE, _INHERIT)
            finally:
                _k32.LocalFree(everyone)
            base = Path(sys.base_prefix)
            for name in (f"python{sys.version_info[0]}{sys.version_info[1]}.dll", "python3.dll", "python.exe",
                         "vcruntime140.dll", "vcruntime140_1.dll"):
                if (base / name).exists():
                    shutil.copy2(base / name, building / name)
            skip = shutil.ignore_patterns(*RUNTIME_SKIP)
            shutil.copytree(base / "DLLs", building / "DLLs", ignore=skip)
            shutil.copytree(base / "Lib", building / "Lib", ignore=skip)
            target = building / "Lib" / "site-packages"
            target.mkdir()
            versions = {}
            sites = _site_dirs()
            for package in packages:
                source = next((s / package for s in sites if (s / package).exists()), None)
                if source is None:
                    raise IsolationError(f"package {package} is not installed, cannot copy it into the runtime")
                if source.is_file():
                    shutil.copy2(source, target / package)
                    continue
                shutil.copytree(source, target / package, ignore=shutil.ignore_patterns("tests", "test"))
                for info in source.parent.glob(f"{package.replace('-', '_')}-*.dist-info"):
                    shutil.copytree(info, target / info.name)
                    versions[package] = info.name[len(package) + 1:-len(".dist-info")]
            subprocess.run([sys.executable, "-I", "-m", "compileall", "-q", "-j", "1", str(building / "Lib")],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS)
            files = bytes_total = 0
            for path, st, _ in _walk(building):
                if not stat.S_ISDIR(st.st_mode):
                    files += 1
                    bytes_total += st.st_size
            (building / "manifest.json").write_text(json.dumps(
                {"python": sys.version, "packages": versions, "files": files, "bytes": bytes_total}, indent=1), encoding="utf-8")
            try:
                os.rename(building, final)
            except OSError:
                if not (final / "manifest.json").is_file():
                    raise
                _remove_tree(building)
        except BaseException:
            _remove_tree(building)
            raise
        return final

    class _AppContainerBox(Box):
        backend = "appcontainer"
        isolated = True

        def __init__(self, policy: Policy, scratch: Scratch, runtime: Path):
            self.policy = policy
            self._scratch = scratch
            self._runtime = runtime
            self._lock = threading.Lock()    # one run at a time, and kill waits for it to leave
            self._state = threading.Lock()
            self._closed = False
            self._gone = threading.Event()
            self._reason: Optional[str] = None
            self._job = self._profile = self._desktop = None
            self._spent = {"cpu": 0.0, "peak": 0, "processes": 0, "written": 0}    # of the jobs a kill has ended
            self._grants: List[Path] = []
            self._rw: List[Path] = []
            self.name = f"strata.box.{os.getpid()}.{uuid.uuid4().hex[:12]}"
            self.dir = scratch.root / "boxes" / self.name
            self._watcher = None
            try:
                self._job = _Job(policy)
                self._profile = _Profile(self.name)
                self._desktop, self._desktop_name = _box_desktop(self._profile.sid_text)
                self.workdir = self.dir / "work"
                self.tmpdir = self.dir / "tmp"
                self.workdir.mkdir(parents=True)
                self.tmpdir.mkdir()
                for path in (self.workdir, self.tmpdir):
                    _edit_dacl(path, self._profile.sid, _GRANT, _MODIFY, _INHERIT)
                # pytest stats the parent of what it collects, and an AccessDenied there ends the whole session.
                _edit_dacl(self.dir, self._profile.sid, _GRANT, _STAT_ONLY, 0)
                # CreateProcess rewrites TEMP, TMP and LOCALAPPDATA to this path under the LOCALAPPDATA it is given.
                (self.tmpdir / "Packages" / self.name / "AC" / "Temp").mkdir(parents=True)
                self._apply_paths()
                self._env = self._environment()
                self._start_watch()
            except BaseException:
                self.kill()
                raise

        def _apply_paths(self) -> None:
            sid = self._profile.sid
            for writable, paths in ((False, self.policy.readonly), (True, self.policy.readwrite)):
                for raw in paths:
                    path = self._scratch.inside(raw)
                    if path is None:
                        raise IsolationError(f"{raw} is outside the scratch root; copy it in, this backend will not change "
                                             "ACLs on the user's own folders")
                    _refuse_links(path)
                    _edit_dacl(path, sid, _GRANT, _MODIFY if writable else _READ_EXECUTE, _INHERIT)
                    self._grants.append(path)
                    if writable:
                        self._rw.append(path)
            for raw in self.policy.denied:
                path = self._scratch.inside(raw)
                if path is not None:
                    if path.exists():
                        _refuse_links(path)
                        _withdraw(path)
                    continue
                nearest = Path(os.path.abspath(raw))
                while not nearest.exists() and nearest.parent != nearest:
                    nearest = nearest.parent
                if _open_to_containers(nearest):
                    raise IsolationError(f"{raw} is open to every AppContainer and is outside the scratch root, so it "
                                         "cannot be denied")

        def _environment(self) -> Dict[str, str]:
            """Written out from nothing, so no key in the server's environment reaches the box. LOCALAPPDATA is not a
            courtesy: CreateProcess fails with ERROR_ENVVAR_NOT_FOUND without it."""
            root = os.environ.get("SystemRoot", r"C:\Windows")
            tmp = str(self.tmpdir)
            env = {"SystemRoot": root, "windir": root, "SystemDrive": root[:2],
                   "ComSpec": os.path.join(root, "System32", "cmd.exe"), "PATHEXT": ".COM;.EXE;.BAT;.CMD",
                   "PATH": ";".join((str(self._runtime), os.path.join(root, "System32"), root)),
                   "USERPROFILE": tmp, "HOME": tmp, "APPDATA": tmp, "LOCALAPPDATA": tmp, "TEMP": tmp, "TMP": tmp,
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0", "PYTHONUTF8": "1",
                   "MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
            for name in ("PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS", "OS"):
                if name in os.environ:
                    env[name] = os.environ[name]
            return env

        def _resolve(self, program: str) -> str:
            if program in ("python", "python.exe", "python3"):
                return str(self._runtime / "python.exe")
            if os.path.isabs(program):
                return program
            for folder in (str(self.workdir), *self._env["PATH"].split(";")):
                for suffix in ("", ".exe"):
                    candidate = os.path.join(folder, program + suffix)
                    if os.path.isfile(candidate):
                        return candidate
            raise IsolationError(f"{program} was not found on the box's PATH")

        def _spawn(self, argv: Sequence[str], handles: Sequence[int]) -> _ProcessInformation:
            """Start suspended, confirm the wall from the outside, and only then let the first instruction run."""
            exe = self._resolve(argv[0])
            line = ctypes.create_unicode_buffer(subprocess.list2cmdline([exe, *argv[1:]]))
            text = "".join(f"{k}={v}\0" for k, v in sorted(self._env.items(), key=lambda kv: kv[0].upper())) + "\0"
            block = ctypes.create_unicode_buffer(text, len(text))
            caps = _SecurityCapabilities(self._profile.sid, None, 0, 0)
            listed = (w.HANDLE * len(handles))(*handles)
            job = w.HANDLE(self._job.handle)
            size = ctypes.c_size_t()
            _k32.InitializeProcThreadAttributeList(None, 3, 0, byref(size))
            attributes = ctypes.create_string_buffer(size.value)
            _check(_k32.InitializeProcThreadAttributeList(attributes, 3, 0, byref(size)), "attribute list")
            try:
                for attribute, value, length in ((_ATTR_SECURITY_CAPABILITIES, byref(caps), sizeof(caps)),
                                                 (_ATTR_HANDLE_LIST, listed, sizeof(listed)),
                                                 (_ATTR_JOB_LIST, byref(job), sizeof(job))):
                    _check(_k32.UpdateProcThreadAttribute(attributes, 0, attribute, value, length, None, None),
                           "UpdateProcThreadAttribute")
                info = _StartupInfoEx()
                info.StartupInfo.cb = sizeof(info)
                info.StartupInfo.dwFlags = _USE_STDHANDLES
                info.StartupInfo.lpDesktop = self._desktop_name
                info.StartupInfo.hStdInput, info.StartupInfo.hStdOutput, info.StartupInfo.hStdError = handles
                info.lpAttributeList = ctypes.cast(attributes, w.LPVOID)
                process = _ProcessInformation()
                flags = _EXTENDED_STARTUPINFO | _UNICODE_ENVIRONMENT | _SUSPENDED | CREATE_NO_WINDOW
                _check(_k32.CreateProcessW(exe, line, None, None, True, flags, block, str(self.workdir), byref(info),
                                           byref(process)), f"starting {exe}")
            finally:
                _k32.DeleteProcThreadAttributeList(attributes)
            try:
                self._confirm(process.hProcess)
                if _k32.ResumeThread(process.hThread) == 0xFFFFFFFF:
                    raise IsolationError(f"ResumeThread: {ctypes.WinError(ctypes.get_last_error())}")
            except BaseException:
                _k32.TerminateProcess(process.hProcess, _KILLED_EXIT_CODE)
                _k32.CloseHandle(process.hThread)
                _k32.CloseHandle(process.hProcess)
                raise
            _k32.CloseHandle(process.hThread)
            return process

        def _confirm(self, process) -> None:
            inside = w.BOOL()
            _check(_k32.IsProcessInJob(process, self._job.handle, byref(inside)), "IsProcessInJob")
            if not inside:
                raise IsolationError("the process is not in the job")
            token = w.HANDLE()
            _check(_adv.OpenProcessToken(process, _TOKEN_QUERY, byref(token)), "OpenProcessToken")
            try:
                flag, got = w.DWORD(), w.DWORD()
                _check(_adv.GetTokenInformation(token, _TOKEN_IS_APP_CONTAINER, byref(flag), sizeof(flag), byref(got)),
                       "reading the token")
                if not flag:
                    raise IsolationError("the process token is not an AppContainer")
                buffer = ctypes.create_string_buffer(256)
                _check(_adv.GetTokenInformation(token, _TOKEN_APP_CONTAINER_SID, buffer, sizeof(buffer), byref(got)),
                       "reading the token's SID")
                held = ctypes.cast(buffer, ctypes.POINTER(w.LPVOID)).contents
                if not _adv.EqualSid(held, self._profile.sid):
                    raise IsolationError("the process runs as another AppContainer")
                _check(_adv.GetTokenInformation(token, _TOKEN_CAPABILITIES, buffer, sizeof(buffer), byref(got)),
                       "reading the token's capabilities")
                if ctypes.cast(buffer, ctypes.POINTER(_TokenGroupsHead)).contents.GroupCount:
                    raise IsolationError("the process token holds capabilities")
            finally:
                _k32.CloseHandle(token)

        def run(self, argv, stdin=None, timeout=None) -> RunResult:
            if isinstance(stdin, str):
                stdin = stdin.encode("utf-8")
            with self._lock:
                left = self._still_open()
                wait, out_of_time = (left, "time") if timeout is None or left <= timeout else (timeout, "timeout")
                self._reason = None
                began = time.perf_counter()
                if self._job.ended or self._job.out_of_cpu():
                    self._renew_job()
                    self._still_open()
                child_in, feed = _pipe(True)
                child_out, read_out = _pipe(False)
                child_err, read_err = _pipe(False)
                try:
                    process = self._spawn(argv, (child_in, child_out, child_err))
                except BaseException:
                    for handle in (feed, read_out, read_err):
                        _k32.CloseHandle(handle)
                    raise
                finally:
                    for handle in (child_in, child_out, child_err):
                        _k32.CloseHandle(handle)
                threading.Thread(target=_feed, args=(feed, stdin or b""), daemon=True).start()
                return self._pump(process, read_out, read_err, began, wait, out_of_time)

        def popen(self, argv):
            import msvcrt
            with self._lock:
                self._still_open()
                if self._job.ended or self._job.out_of_cpu():
                    self._renew_job()
                    self._still_open()
                child_in, feed = _pipe(True)
                child_out, read_out = _pipe(False)
                child_err, read_err = _pipe(False)
                try:
                    process = self._spawn(argv, (child_in, child_out, child_err))
                except BaseException:
                    for handle in (feed, read_out, read_err):
                        _k32.CloseHandle(handle)
                    raise
                finally:
                    for handle in (child_in, child_out, child_err):
                        _k32.CloseHandle(handle)
            stdin = os.fdopen(msvcrt.open_osfhandle(feed, 0), "w", encoding="utf-8", newline="\n")
            stdout = os.fdopen(msvcrt.open_osfhandle(read_out, os.O_RDONLY), "r", encoding="utf-8", newline="\n")
            return _Stays(process, stdin, stdout, read_err)

        def _renew_job(self) -> None:
            """The limits carry over: the new job gets what is left of the CPU time, so kills cannot refill it."""
            old = self._job
            old.terminate()
            if not old.wait_empty(2):
                raise IsolationError("the killed job did not empty")
            with self._swap:
                info = old.accounting()
                self._spent["cpu"] += (info.BasicInfo.TotalUserTime + info.BasicInfo.TotalKernelTime) / 1e7
                self._spent["peak"] = max(self._spent["peak"], old.peak_memory())
                self._spent["processes"] += info.BasicInfo.TotalProcesses
                self._spent["written"] += info.IoInfo.WriteTransferCount
                if self._spent["cpu"] >= self.policy.seconds:
                    raise IsolationError("the box is out of time: it has used its CPU")
                self._job = _Job(self.policy, self._spent["cpu"])
                old.close()

        def _pump(self, process, read_out, read_err, began, wait, out_of_time) -> RunResult:
            """One loop over both pipes and the process. No reader thread: a child left behind keeps the pipe open, and a
            blocked read would hold this call until it ended."""
            cap = self.policy.output_bytes
            out, err = bytearray(), bytearray()
            total = 0
            truncated = False
            give_up = time.monotonic() + wait
            pipes = [(read_out, out), (read_err, err)]
            try:
                while True:
                    for handle, sink in pipes:
                        _, taken = _drain(handle, sink, max(0, cap - total))
                        total += taken
                    if total > cap and not truncated:
                        truncated = True
                        self._terminate("output")
                    remaining = give_up - time.monotonic()
                    if remaining <= 0:
                        self._terminate(out_of_time)
                        _k32.WaitForSingleObject(process.hProcess, 2000)
                        break
                    if _k32.WaitForSingleObject(process.hProcess, min(5, int(remaining * 1000) + 1)) == _WAIT_OBJECT_0:
                        break
                for handle, sink in pipes:
                    _, taken = _drain(handle, sink, max(0, cap - total))
                    total += taken
                code = w.DWORD()
                _k32.GetExitCodeProcess(process.hProcess, byref(code))
                if self._reason is None and code.value == _STATUS_QUOTA_EXCEEDED and self._job.out_of_cpu():
                    self._job.ended = True    # Windows ended the job itself, so nothing here saw it happen
                    self._reason = "time"
                return RunResult(code.value, bytes(out), bytes(err), (time.perf_counter() - began) * 1000,
                                 self._reason, truncated or total > cap)
            finally:
                _k32.CloseHandle(process.hProcess)
                _k32.CloseHandle(read_out)
                _k32.CloseHandle(read_err)

        def _terminate(self, reason: str) -> None:
            if self._reason is None:
                self._reason = reason
            if self._job:
                self._job.terminate()

        def _alive(self) -> bool:
            return bool(self._job and self._job.handle and self._job.accounting().BasicInfo.ActiveProcesses)

        def _written(self) -> int:
            return self._job.accounting().IoInfo.WriteTransferCount

        def _disk_used(self) -> int:
            return _disk_used([self.workdir, self.tmpdir, self._profile.folder, *self._rw])

        def usage(self) -> Usage:
            if self._closed:
                raise IsolationError("the box is closed")    # a NULL job would answer for this process's own job
            spent = self._spent
            with self._swap:
                info = self._job.accounting()
                basic = info.BasicInfo
                return Usage(time.monotonic() - self._began,
                             spent["cpu"] + (basic.TotalUserTime + basic.TotalKernelTime) / 1e7,
                             max(spent["peak"], self._job.peak_memory()), spent["processes"] + basic.TotalProcesses,
                             basic.ActiveProcesses, self._disk_used(), spent["written"] + info.IoInfo.WriteTransferCount)

        def kill(self) -> None:
            """Safe from any thread while a run is inside: it ends the programs first, which ends the run, and only then
            takes the lock to clear up. When it returns the box is gone, also for a second caller."""
            with self._state:
                first = not self._closed
                self._closed = True
            if not first:
                self._gone.wait(15)
                return
            try:
                if self._watcher:
                    self._stop_watch()
                self._terminate("kill")
                with self._lock:
                    if self._job:
                        self._job.terminate()    # a run may have renewed it before it saw the box close
                        self._job.wait_empty(5)
                        self._job.close()
                    for path in self._grants:
                        self._revoke(path)
                    _remove_tree(self.dir)
                    if self._desktop:
                        _u32.CloseDesktop(self._desktop)
                    if self._profile:
                        self._profile.close()
            finally:
                self._gone.set()

        def _revoke(self, path: Path) -> None:
            """The box's links go first: revoking on a tree spreads into it, and the box had write access here."""
            for found, st, linked in list(_walk(path)):
                if linked:
                    try:
                        os.rmdir(found) if stat.S_ISDIR(st.st_mode) else os.unlink(found)
                    except OSError:
                        return
            try:
                _edit_dacl(path, self._profile.sid, _REVOKE, 0, _INHERIT if path.is_dir() else 0)
            except IsolationError:
                pass


class AppContainerBackend:
    name = "appcontainer"
    _swept = set()

    @staticmethod
    def available() -> Tuple[bool, str]:
        if sys.platform != "win32":
            return False, "Windows only"
        if not hasattr(_uenv, "CreateAppContainerProfile"):
            return False, "this Windows has no AppContainer"
        if _sh32.IsUserAnAdmin():
            return False, "the server runs elevated; a box under an admin token has not been tested"
        for service in ("BFE", "MpsSvc"):    # an AppContainer's lack of network is enforced by these two
            if not _service_running(service):
                return False, f"the firewall service {service} is not running, so a box's network may not be shut"
        return True, ""

    @classmethod
    def start(cls, policy: Policy, root=None) -> Box:
        ok, why = cls.available()
        if not ok:
            raise IsolationError(f"AppContainer backend: {why}")
        if policy.network:
            raise IsolationError("network is not built for this backend: rule I3 needs per-destination rules")
        scratch = Scratch(root)
        if scratch.root not in cls._swept:
            cls._swept.add(scratch.root)
            scratch.sweep()
        _k32.SetErrorMode(0x0001 | 0x0002 | 0x8000)    # no crash or missing-DLL dialog, the box inherits this
        return _AppContainerBox(policy, scratch, build_runtime(scratch))


class MxcBackend:
    """Microsoft's Execution Containers through `wxc-exec.exe`. Written from the published schema 1.0.0 and never run:
    the tool is not on this machine. That schema has `process.timeout` and no memory, process or disk limit for
    `processcontainer`, so this backend cannot meet rule I7 by itself."""

    name = "mxc"
    schema_version = "1.0.0"

    @staticmethod
    def executable() -> Optional[str]:
        return shutil.which("wxc-exec.exe")

    @classmethod
    def available(cls) -> Tuple[bool, str]:
        if sys.platform != "win32":
            return False, "Windows only"
        return (True, "") if cls.executable() else (False, "wxc-exec.exe is not installed")

    @classmethod
    def config(cls, policy: Policy, argv: Sequence[str], cwd: Optional[str] = None) -> dict:
        if policy.network:
            raise IsolationError("network is not built for this backend")
        process = {"commandLine": subprocess.list2cmdline(list(argv)), "inheritDefaultEnv": False,
                   "timeout": int(policy.seconds * 1000)}
        if cwd:
            process["cwd"] = str(cwd)
        return {"version": cls.schema_version, "containment": "processcontainer", "process": process,
                "filesystem": {"readonlyPaths": [str(p) for p in policy.readonly],
                               "readwritePaths": [str(p) for p in policy.readwrite],
                               "deniedPaths": [str(p) for p in policy.denied]},
                "network": {"egress": {"default": "deny"}, "ingress": {"default": "deny", "hostLoopback": "deny"}},
                "ui": {"disable": True, "clipboard": "none", "injection": False}}

    @classmethod
    def write_config(cls, policy: Policy, argv: Sequence[str], path, cwd: Optional[str] = None) -> Path:
        path = Path(path)
        path.write_text(json.dumps(cls.config(policy, argv, cwd), indent=2), encoding="utf-8")
        return path

    @classmethod
    def start(cls, policy: Policy, root=None) -> Box:
        ok, why = cls.available()
        raise IsolationError(f"MXC backend: {why or 'present, but it has not passed the escape suite'}")


class _PosixBox(Box):
    """A session of its own with rlimits, behind `bwrap` or behind nothing at all."""

    def __init__(self, policy: Policy, backend: str, isolated: bool, wrap: Callable[["_PosixBox", List[str]], List[str]]):
        self.policy = policy
        self.backend = backend
        self.isolated = isolated
        self._wrap = wrap
        self.dir = Path(tempfile.mkdtemp(prefix="strata-box-"))
        self.workdir = self.dir / "work"
        self.tmpdir = self.dir / "tmp"
        self.workdir.mkdir()
        self.tmpdir.mkdir()
        self._lock = threading.Lock()
        self._state = threading.Lock()
        self._closed = False
        self._group: Optional[int] = None
        self._spawned = 0
        self._start_watch()

    def _limits(self) -> None:
        import resource
        seconds = int(self.policy.seconds) + 1
        memory = self.policy.memory_bytes
        for limit, value in ((resource.RLIMIT_AS, memory), (resource.RLIMIT_CPU, seconds),
                             (resource.RLIMIT_NPROC, self.policy.processes), (resource.RLIMIT_CORE, 0),
                             (resource.RLIMIT_FSIZE, self.policy.disk_bytes)):
            hard = resource.getrlimit(limit)[1]    # asking above the hard limit raises
            value = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(limit, (value, value))

    def _environment(self) -> Dict[str, str]:
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(self.tmpdir), "TMPDIR": str(self.tmpdir), "LANG": "C.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg"}
        return env

    def run(self, argv, stdin=None, timeout=None) -> RunResult:
        import selectors
        if isinstance(stdin, str):
            stdin = stdin.encode("utf-8")
        with self._lock:
            left = self._still_open()
            wait, out_of_time = (left, "time") if timeout is None or left <= timeout else (timeout, "timeout")
            self._reason = None
            began = time.perf_counter()
            process = subprocess.Popen(self._wrap(self, list(argv)), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, cwd=self.workdir, env=self._environment(),
                                       start_new_session=True, preexec_fn=self._limits)
            self._group = process.pid
            self._spawned += 1
            threading.Thread(target=self._send, args=(process, stdin or b""), daemon=True).start()
            cap = self.policy.output_bytes
            sinks = {process.stdout: bytearray(), process.stderr: bytearray()}
            select = selectors.DefaultSelector()
            for stream in sinks:
                select.register(stream, selectors.EVENT_READ)
            total = 0
            truncated = False
            give_up = time.monotonic() + wait
            while select.get_map():
                remaining = give_up - time.monotonic()
                if remaining <= 0:
                    self._terminate(out_of_time)
                    break
                ready = select.select(min(remaining, 0.05))
                for key, _ in ready:
                    chunk = os.read(key.fileobj.fileno(), 1 << 16)
                    if not chunk:
                        select.unregister(key.fileobj)
                        continue
                    sinks[key.fileobj] += chunk[:max(0, cap - total)]
                    total += len(chunk)
                    if total > cap and not truncated:
                        truncated = True
                        self._terminate("output")
                if not ready and process.poll() is not None:    # a child left behind keeps the pipe open, so no EOF comes
                    break
            select.close()
            try:
                code = process.wait(max(0.1, give_up - time.monotonic()))
            except subprocess.TimeoutExpired:
                self._terminate(out_of_time)
                code = process.wait(2)
            return RunResult(code, bytes(sinks[process.stdout]), bytes(sinks[process.stderr]),
                             (time.perf_counter() - began) * 1000, self._reason, truncated)

    def popen(self, argv):
        with self._lock:
            self._still_open()
            process = subprocess.Popen(self._wrap(self, list(argv)), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=self.workdir,
                                       env=self._environment(), start_new_session=True, preexec_fn=self._limits, text=True, encoding="utf-8")
            self._group = process.pid
            self._spawned += 1
            return process

    @staticmethod
    def _send(process, data: bytes) -> None:
        try:
            process.stdin.write(data)
            process.stdin.close()
        except OSError:
            pass

    def _terminate(self, reason: str) -> None:
        import signal
        if self._reason is None:
            self._reason = reason
        if self._group:
            try:
                os.killpg(self._group, signal.SIGKILL)
            except OSError:
                pass

    def _alive(self) -> bool:
        if not self._group:
            return False
        try:
            os.killpg(self._group, 0)
            return True
        except OSError:
            return False

    def _disk_used(self) -> int:
        total = 0
        for root in (self.workdir, self.tmpdir):
            for _, st, _ in _walk(root):
                total += st.st_blocks * 512 if hasattr(st, "st_blocks") else st.st_size
        return total

    def usage(self) -> Usage:
        import resource
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        return Usage(time.monotonic() - self._began, child.ru_utime + child.ru_stime, child.ru_maxrss * 1024, self._spawned,
                     1 if self._alive() else 0, self._disk_used(), 0)

    def kill(self) -> None:
        with self._state:
            if self._closed:
                return
            self._closed = True
        self._stop_watch()
        self._terminate("kill")
        with self._lock:
            shutil.rmtree(self.dir, ignore_errors=True)


def bwrap_argv(policy: Policy, box: "_PosixBox", argv: Sequence[str]) -> List[str]:
    """Nothing is visible but the system's read-only folders and what the policy names. `--unshare-all` includes the
    network. A denied file becomes /dev/null and a denied folder an empty tmpfs."""
    if policy.network:
        raise IsolationError("network is not built for this backend")
    line = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--ro-bind", "/usr", "/usr",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    for name in ("/bin", "/sbin", "/lib", "/lib64", "/etc/alternatives", "/etc/ld.so.cache"):
        line += ["--ro-bind-try", name, name]
    for path in policy.readonly:
        line += ["--ro-bind", str(path), str(path)]
    for path in policy.readwrite:
        line += ["--bind", str(path), str(path)]
    line += ["--bind", str(box.workdir), str(box.workdir), "--bind", str(box.tmpdir), str(box.tmpdir)]
    for path in policy.denied:
        line += ["--tmpfs", str(path)] if os.path.isdir(path) else ["--ro-bind", "/dev/null", str(path)]
    return line + ["--chdir", str(box.workdir), "--", *argv]


class BubblewrapBackend:
    name = "bubblewrap"

    @staticmethod
    def available() -> Tuple[bool, str]:
        if not sys.platform.startswith("linux"):
            return False, "Linux only"
        return (True, "") if shutil.which("bwrap") else (False, "bwrap is not installed")

    @classmethod
    def start(cls, policy: Policy, root=None) -> Box:
        ok, why = cls.available()
        if not ok:
            raise IsolationError(f"bubblewrap backend: {why}")
        if policy.network:
            raise IsolationError("network is not built for this backend")
        box = _PosixBox(policy, cls.name, True, lambda b, argv: bwrap_argv(policy, b, argv))
        probe = box.run(["/usr/bin/true"], timeout=10)
        if probe.exit_code != 0:
            box.kill()
            raise IsolationError(f"bwrap did not start: {probe.stderr[:300]!r}")
        return box


class NoneBackend:
    """No wall. For the throwaway Colab VM of step S1 and nothing else."""

    name = "none"

    @staticmethod
    def available() -> Tuple[bool, str]:
        if sys.platform == "win32":
            return False, "refused on Windows"
        if not sys.platform.startswith("linux"):
            return False, "Linux only"
        if os.environ.get(THROWAWAY_ENV) != "1":
            return False, f"needs {THROWAWAY_ENV}=1: there is no wall behind this backend"
        return True, ""

    @classmethod
    def start(cls, policy: Policy, root=None) -> Box:
        ok, why = cls.available()
        if not ok:
            raise IsolationError(f"none backend: {why}")
        return _PosixBox(policy, cls.name, False, lambda b, argv: argv)


BACKENDS = {cls.name: cls for cls in (AppContainerBackend, MxcBackend, BubblewrapBackend, NoneBackend)}


class Isolation:
    @staticmethod
    def report() -> Dict[str, Tuple[bool, str]]:
        """Which backends this machine has, and why not for the others."""
        return {name: cls.available() for name, cls in BACKENDS.items()}

    @staticmethod
    def start(policy: Policy, backend: Optional[str] = None, root=None) -> Box:
        """A Box, or IsolationError. MXC is used only by name, until it has passed the escape suite."""
        if backend is not None:
            if backend not in BACKENDS:
                raise IsolationError(f"no backend named {backend}")
            return BACKENDS[backend].start(policy, root)
        if sys.platform == "win32":
            return AppContainerBackend.start(policy, root)
        if sys.platform.startswith("linux"):
            if BubblewrapBackend.available()[0]:
                return BubblewrapBackend.start(policy, root)
            return NoneBackend.start(policy, root)
        raise IsolationError(f"no backend for {sys.platform}")
