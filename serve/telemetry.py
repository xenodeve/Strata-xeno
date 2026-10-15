"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.  With the AMD backend (#301): the
  amdgpu driver's Linux sysfs files - load, VRAM, temperature and power.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc) and the disk rate is absent.
- Memory pressure (xeno #237): RAM available, the commit charge, hard page faults per second, and the engine's working
  set and page faults per second - read from the OS directly, psutil or not.
- The record kept (xeno #237): TelemetryLog keeps the last two hours in memory, one point a second, and writes them to
  the request history's folder as telemetry-YYYY-MM-DD.jsonl, so a past stall can be read back.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import json
import os
import platform
import sys
import threading
import time

HISTORY = 60


# ------------------------------------------------------------------------------------------------ NVML
class _Nvml:
    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, index=0):
        self.lib = self.dev = None
        names = ["nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                          "NVIDIA Corporation", "NVSMI", "nvml.dll")] if os.name == "nt" \
            else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        for n in names:
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if self.lib is None:
            return
        try:
            init = getattr(self.lib, "nvmlInit_v2", None) or self.lib.nvmlInit
            if init() != 0:
                self.lib = None
                return
            h = ctypes.c_void_p()
            get = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or self.lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) != 0:
                self.lib = None
                return
            self.dev = h
        except (AttributeError, OSError):
            self.lib = None

    def ok(self):
        return self.lib is not None and self.dev is not None

    def count(self):
        """How many cards NVML sees (0 when it cannot say)."""
        n = ctypes.c_uint()
        try:
            fn = getattr(self.lib, "nvmlDeviceGetCount_v2", None) or self.lib.nvmlDeviceGetCount
            return n.value if fn(ctypes.byref(n)) == 0 else 0
        except (AttributeError, OSError):
            return 0

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def _u64(self, fn):
        v = ctypes.c_ulonglong()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def name(self):
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        return None

    def read(self):
        out = {}
        u = self.Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self.Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))          # NVML_TEMPERATURE_GPU
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")        # drops at idle (power saving)
        out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        out["pcie_width_max"] = self._uint("nvmlDeviceGetMaxPcieLinkWidth")
        out["sm_clock"] = self._uint("nvmlDeviceGetClockInfo", ctypes.c_uint(1))         # NVML_CLOCK_SM, MHz
        out["mem_clock"] = self._uint("nvmlDeviceGetClockInfo", ctypes.c_uint(2))        # NVML_CLOCK_MEM, MHz
        out["throttle"] = throttle_names(self._u64("nvmlDeviceGetCurrentClocksThrottleReasons"))
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))                 # NVML_PCIE_UTIL_RX_BYTES, KB/s
        tx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(0))
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None
        out["pcie_tx_mb"] = tx / 1024.0 if tx is not None else None
        return out


def nvml_device_count():
    g = _Nvml(0)
    return g.count() if g.ok() else 0


# NVML clocks-throttle-reason bits. "GPU idle" (0x1) is the card resting, not a limit, so it is not listed.
_THROTTLE = ((0x4, "power cap"), (0x8, "hardware slowdown"), (0x20, "thermal slowdown"), (0x40, "hardware thermal slowdown"),
             (0x80, "hardware power brake"), (0x2, "application clocks"), (0x10, "sync boost"), (0x100, "display clocks"))


def throttle_names(mask):
    """The limits holding a card's clocks down, by name; None when NVML could not say (which is not "none")."""
    if mask is None:
        return None
    return [name for bit, name in sorted(_THROTTLE, key=lambda x: (x[0] not in (0x4, 0x8, 0x20, 0x40, 0x80), x[0])) if mask & bit]


# ------------------------------------------------------------------------------------------------ AMD (Linux sysfs)
SYSFS = "/sys"


def amd_device_dir(index, sysfs=None):
    """The amdgpu sysfs folder (/sys/class/drm/renderD<N>/device) of the AMD GPU that HIP numbers `index`: the KFD
    topology's GPU nodes in order, the CPU nodes skipped, linked to their render node by drm_render_minor - the
    numbering setup's amd_gpus() and HIP_VISIBLE_DEVICES use.  None when there is no such card (or no amdgpu)."""
    base = os.path.join(sysfs or SYSFS, "class", "kfd", "kfd", "topology", "nodes")
    try:
        nodes = sorted((n for n in os.listdir(base) if n.isdigit()), key=int)
    except OSError:
        return None
    gpus = []
    for n in nodes:
        try:
            with open(os.path.join(base, n, "properties"), encoding="utf-8") as f:
                props = dict(line.strip().partition(" ")[::2] for line in f if line.strip())
            if int(props.get("gfx_target_version") or 0) == 0 or int(props.get("simd_count") or 0) == 0:
                continue
            gpus.append(props)
        except (OSError, ValueError):
            continue
    if not 0 <= index < len(gpus) or not gpus[index].get("drm_render_minor"):
        return None
    dev = os.path.join(sysfs or SYSFS, "class", "drm", "renderD" + gpus[index]["drm_render_minor"].strip(), "device")
    return dev if os.path.isdir(dev) else None


class _Amd:
    """#301: an AMD card's readings from the amdgpu driver's sysfs files (Linux; no ROCm library needed), with _Nvml's
    interface: load (gpu_busy_percent), VRAM (mem_info_vram_used / _total), and from its hwmon folder the temperature
    (temp1_input, the edge sensor, m°C), power (power1_average or power1_input, µW) and its cap (power1_cap)."""

    def __init__(self, index=0, sysfs=None):
        self.dev = amd_device_dir(index, sysfs)
        self.hwmon = None
        if self.dev:
            try:
                hw = sorted(os.listdir(os.path.join(self.dev, "hwmon")))
                self.hwmon = os.path.join(self.dev, "hwmon", hw[0]) if hw else None
            except OSError:
                pass

    def ok(self):
        return self.dev is not None

    @staticmethod
    def _int(path):
        try:
            with open(path, encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError, TypeError):
            return None

    def name(self):
        try:
            with open(os.path.join(self.dev, "product_name"), encoding="utf-8") as f:
                return f.read().strip() or "AMD Radeon"
        except (OSError, TypeError):
            return "AMD Radeon"

    @staticmethod
    def _gen(text):
        """The PCIe generation of a link-speed file's contents ("16.0 GT/s PCIe" -> 4), or None."""
        try:
            return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}[float(str(text).split()[0])]
        except (ValueError, TypeError, IndexError, KeyError):
            return None

    @staticmethod
    def _bdf(s):
        """True for a sysfs pci device name ("0000:03:00.0"); nothing is imported for it."""
        return (len(s) == 12 and s[4] == ":" and s[7] == ":" and s[10] == "." and s[11] in "01234567"
                and all(c in "0123456789abcdef" for c in s[0:4] + s[5:7] + s[8:10]))

    @staticmethod
    def _read(d, name):
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                return f.read().strip()
        except (OSError, TypeError):
            return None

    def _hops(self):
        """The PCIe devices between the root port and this card, the card last: the sysfs path names every
        hop (a card behind a bridge chain has more than one; a directly attached card has one)."""
        out, prefix = [], []
        for part in os.path.realpath(self.dev or "").split("/"):
            prefix.append(part)
            if self._bdf(part):
                out.append("/".join(prefix))
        return out

    def link(self):
        """The PCIe link the card actually gets: the **narrowest/slowest hop** between the root port and the card,
        from each hop's `max_link_speed` / `max_link_width` (the capability, so a power-saving downgrade or a Gen3
        slot cannot make it read low).  A card that is Gen4 on its own hop but sits behind a Gen3 root port really
        runs at Gen3, and that is the number a PCIe bandwidth budget needs.
        `pcie_own_gen` keeps the card's own hop aside: the gap between the two is what a user has to see.

        The bottleneck is only claimed when **every** hop of the path could be read; `pcie_path` reports how many
        of them were ("4/4").  A kernel or a container that hides part of /sys/devices would otherwise drop the
        unreadable hops in silence and report the card's own Gen4 hop as the whole path - i.e. be optimistic
        exactly where it matters.  When the walk is incomplete the reading falls back to the card's own negotiated
        link and pcie_path says so."""
        hops = self._hops()
        gens, widths, read = [], [], 0
        for d in hops:
            g = self._gen(_Amd._read(d, "max_link_speed"))
            w = self._int(os.path.join(d, "max_link_width"))
            read += 1 if (g is not None or w is not None) else 0
            if g:
                gens.append(g)
            if w:
                widths.append(w)
        own = self.dev or ""
        own_gen = self._gen(_Amd._read(own, "current_link_speed"))
        whole = bool(hops) and read == len(hops)
        gen = min(gens) if (whole and gens) else own_gen
        width = min(widths) if (whole and widths) else self._int(os.path.join(own, "current_link_width"))
        return {"pcie_gen": gen, "pcie_gen_max": gen, "pcie_own_gen": own_gen, "pcie_width": width,
                "pcie_path": "%d/%d" % (read, len(hops))}


    def read(self):
        out = {"util": self._int(os.path.join(self.dev, "gpu_busy_percent")),
               "mem_used": self._int(os.path.join(self.dev, "mem_info_vram_used")),
               "mem_total": self._int(os.path.join(self.dev, "mem_info_vram_total"))}
        if self.hwmon:
            t = self._int(os.path.join(self.hwmon, "temp1_input"))
            out["temp"] = t / 1000.0 if t is not None else None
            p = self._int(os.path.join(self.hwmon, "power1_average"))
            if p is None:
                p = self._int(os.path.join(self.hwmon, "power1_input"))
            out["power"] = p / 1e6 if p is not None else None
            cap = self._int(os.path.join(self.hwmon, "power1_cap"))
            out["power_limit"] = cap / 1e6 if cap is not None else None
        out.update(self.link())
        return out


def gpu_reader(index=0, amd=False):
    """The card's readings: NVML (NVIDIA), or the amdgpu sysfs files with the AMD backend (#301)."""
    return _Amd(index) if amd else _Nvml(index)


def free_vram_mib(index=0, amd=False):
    """Free VRAM of a card in MiB, or None when it cannot be read."""
    g = gpu_reader(index, amd)
    if not g.ok():
        return None
    r = g.read()
    if r.get("mem_total") is None or r.get("mem_used") is None:
        return None
    return int((r["mem_total"] - r["mem_used"]) >> 20)


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_name():
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        if os.name == "nt":
            idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return idle.value, kern.value + user.value           # kernel time includes idle
            return None
        try:
            f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            return f[3] + f[4], sum(f)
        except (OSError, ValueError):
            return None

    def cpu(self):
        cur = self._times()
        prev, self.prev = self.prev, cur
        if not cur or not prev or cur[1] == prev[1]:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))))

    @staticmethod
    def ram():
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullTotalPhys - m.ullAvailPhys, m.ullTotalPhys
            return None, None
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


# ------------------------------------------------------------------------------------------------ memory pressure
class _MemPressure:
    """What paging looks like (xeno #237), with or without psutil: the RAM still available, the commit charge against
    its limit, the hard page faults per second (pages read back from disk, the system-wide stutter), and the engine
    process's working set and page faults per second (soft and hard; a working set the OS trimmed faults back in).
    Windows: GlobalMemoryStatusEx, the PDH counter \\Memory\\Pages Input/sec and K32GetProcessMemoryInfo.  Linux:
    /proc/meminfo, /proc/vmstat pgmajfault, /proc/<pid>/statm and /proc/<pid>/stat.  Anything unreadable is None."""

    def __init__(self):
        self.nt = os.name == "nt"
        self._pdh = self._q = self._c = None
        self._proc = (None, None)             # (pid, handle) on Windows
        self._prev_sys = None                 # Linux: (time, pgmajfault)
        self._prev_eng = None                 # (pid, time, page faults)
        if self.nt:
            self._open_pdh()

    def _open_pdh(self):
        try:
            pdh = ctypes.WinDLL("pdh")
            q, c = ctypes.c_void_p(), ctypes.c_void_p()
            if pdh.PdhOpenQueryW(None, None, ctypes.byref(q)) != 0:
                return
            if pdh.PdhAddEnglishCounterW(q, ctypes.c_wchar_p(r"\Memory\Pages Input/sec"), None, ctypes.byref(c)) != 0:
                pdh.PdhCloseQuery(q)
                return
            pdh.PdhCollectQueryData(q)        # a rate needs two collections: the first one is the base
            self._pdh, self._q, self._c = pdh, q, c
        except (OSError, AttributeError):
            self._pdh = None

    class _PdhValue(ctypes.Structure):
        _fields_ = [("CStatus", ctypes.c_ulong), ("doubleValue", ctypes.c_double)]

    class _ProcMem(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + \
                   [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                   "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                   "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    def _system_nt(self):
        out = {}

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                out["ram_avail"] = m.ullAvailPhys
                out["commit_used"] = m.ullTotalPageFile - m.ullAvailPageFile     # the page-file fields are the commit
                out["commit_limit"] = m.ullTotalPageFile
        except (OSError, AttributeError):
            pass
        if self._pdh is not None:
            v = self._PdhValue()
            try:
                if self._pdh.PdhCollectQueryData(self._q) == 0 and \
                        self._pdh.PdhGetFormattedCounterValue(self._c, 0x200, None, ctypes.byref(v)) == 0:   # PDH_FMT_DOUBLE
                    out["hard_pf_s"] = v.doubleValue
            except (OSError, AttributeError):
                pass
        return out

    def _system_linux(self):
        out = {}
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            kb = lambda k: int(info[k].split()[0]) * 1024     # noqa: E731
            out["ram_avail"] = kb("MemAvailable")
            out["commit_used"], out["commit_limit"] = kb("Committed_AS"), kb("CommitLimit")
        except (OSError, KeyError, ValueError):
            pass
        try:
            maj = next(int(line.split()[1]) for line in open("/proc/vmstat") if line.startswith("pgmajfault "))
            now, prev = time.time(), self._prev_sys
            self._prev_sys = (now, maj)
            if prev is not None and now > prev[0]:
                out["hard_pf_s"] = (maj - prev[1]) / (now - prev[0])
        except (OSError, StopIteration, ValueError):
            pass
        return out

    def _engine_nt(self, pid):
        k32 = ctypes.windll.kernel32
        if self._proc[0] != pid:
            if self._proc[1]:
                k32.CloseHandle(ctypes.c_void_p(self._proc[1]))
            k32.OpenProcess.restype = ctypes.c_void_p
            h = k32.OpenProcess(ctypes.c_ulong(0x1000 | 0x0010), False, ctypes.c_ulong(pid))   # QUERY_LIMITED | VM_READ
            self._proc = (pid, h)
        h = self._proc[1]
        if not h:
            return None
        pm = self._ProcMem()
        pm.cb = ctypes.sizeof(pm)
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not k32.K32GetProcessMemoryInfo(ctypes.c_void_p(h), ctypes.byref(pm), pm.cb):
            self._proc = (None, None)         # the engine exited: a new handle next time
            k32.CloseHandle(ctypes.c_void_p(h))
            return None
        return pm.WorkingSetSize, pm.PageFaultCount, 2**32      # a DWORD count wraps

    @staticmethod
    def _engine_linux(pid):
        try:
            rss = int(open(f"/proc/{pid}/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            s = open(f"/proc/{pid}/stat").read()
            rest = s[s.rfind(")") + 2:].split()                # field n of stat(5) is rest[n - 3]
            return rss, int(rest[7]) + int(rest[9]), None        # minflt + majflt
        except (OSError, ValueError, IndexError):
            return None

    def read(self, pid=None):
        try:
            out = self._system_nt() if self.nt else self._system_linux()
        except Exception:  # noqa: BLE001 - telemetry must never take the server down
            out = {}
        if pid:
            try:
                r = self._engine_nt(pid) if self.nt else self._engine_linux(pid)
            except Exception:  # noqa: BLE001
                r = None
            if r is not None:
                ws, faults, wrap = r
                out["engine_ws"] = ws
                now, prev = time.time(), self._prev_eng
                self._prev_eng = (pid, now, faults)
                if prev is not None and prev[0] == pid and now > prev[1]:
                    d = faults - prev[2]
                    if d < 0 and wrap:
                        d += wrap
                    if d >= 0:
                        out["engine_pf_s"] = d / (now - prev[1])
        return out


# ------------------------------------------------------------------------------------------------ the record kept
def compact_point(s, t, busy):
    """One second of readings as the log keeps it (~200 bytes): RAM, commit and the engine in MiB, faults per second,
    and per card [load %, VRAM used MiB, PCIe rx MB/s, PCIe tx MB/s]."""
    mib = lambda v: None if v is None else int(v) >> 20          # noqa: E731
    num = lambda v: None if v is None else round(v, 1)           # noqa: E731
    return {"t": round(t, 1), "busy": 1 if busy else 0, "cpu": num(s.get("cpu")), "ram_used": mib(s.get("ram_used")),
            "ram_avail": mib(s.get("ram_avail")), "commit": mib(s.get("commit_used")),
            "commit_limit": mib(s.get("commit_limit")), "hard_pf": num(s.get("hard_pf_s")),
            "eng_ws": mib(s.get("engine_ws")), "eng_pf": num(s.get("engine_pf_s")), "tok_s": num(s.get("tok_s")),
            "g": [[g.get("util"), mib(g.get("mem_used")), num(g.get("pcie_rx_mb")), num(g.get("pcie_tx_mb"))]
                  for g in s.get("gpus") or []]}


def window_summary(points, t0, t1):
    """The hardware over one request (xeno #237): the lowest RAM left, the highest commit and fault rates, the engine's
    working set range, the CPU, and each card's load and VRAM.  None when no point falls in [t0, t1]."""
    pts = [p for p in points if t0 <= p["t"] <= t1]
    if not pts:
        return None

    def col(k):
        return [p[k] for p in pts if p.get(k) is not None]

    def agg(fn, k):
        v = col(k)
        return round(fn(v), 1) if v else None

    mean = lambda v: sum(v) / len(v)                              # noqa: E731
    gpus = []
    for i in range(max((len(p.get("g") or []) for p in pts), default=0)):
        util = [p["g"][i][0] for p in pts if len(p.get("g") or []) > i and p["g"][i][0] is not None]
        mem = [p["g"][i][1] for p in pts if len(p.get("g") or []) > i and p["g"][i][1] is not None]
        gpus.append({"util_mean": round(mean(util), 1) if util else None, "util_max": max(util) if util else None,
                     "mem_max_mib": max(mem) if mem else None})
    return {"samples": len(pts), "ram_avail_min_mib": agg(min, "ram_avail"), "commit_max_mib": agg(max, "commit"),
            "commit_limit_mib": agg(max, "commit_limit"), "hard_pf_max": agg(max, "hard_pf"),
            "hard_pf_mean": agg(mean, "hard_pf"), "eng_ws_min_mib": agg(min, "eng_ws"),
            "eng_ws_max_mib": agg(max, "eng_ws"), "eng_pf_max": agg(max, "eng_pf"), "eng_pf_mean": agg(mean, "eng_pf"),
            "cpu_max": agg(max, "cpu"), "cpu_mean": agg(mean, "cpu"), "gpus": gpus}


class TelemetryLog:
    """The readings kept past the 60-second window (xeno #237), so a stall or a stutter can be read back afterwards:
    one compact point a second in memory for the last KEEP_S seconds (a finished request's hardware summary reads its
    window from them), and on disk as <dir>/telemetry-YYYY-MM-DD.jsonl - every second while a request runs, every
    IDLE_EVERY_S seconds otherwise.  Files older than `keep_days` are deleted.  A failing disk only stops the file."""

    KEEP_S = 7200
    IDLE_EVERY_S = 10.0

    def __init__(self, directory=None, keep_days=14):
        self.dir = None if directory is None else os.fspath(directory)
        self.keep_days = keep_days
        self.lock = threading.Lock()
        self.points = collections.deque(maxlen=self.KEEP_S)
        self._last_disk = 0.0
        self._day = None

    def add(self, s, t, busy):
        p = compact_point(s, t, busy)
        with self.lock:
            self.points.append(p)
        if self.dir is None or (not busy and t - self._last_disk < self.IDLE_EVERY_S):
            return
        self._last_disk = t
        try:
            day = time.strftime("%Y-%m-%d", time.localtime(t))
            if day != self._day:
                os.makedirs(self.dir, exist_ok=True)
                self._day = day
                self._prune(t)
            with open(os.path.join(self.dir, f"telemetry-{day}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(p, separators=(",", ":")) + "\n")
        except OSError:
            self.dir = None                   # a full or missing disk: keep the memory window, stop writing

    def _prune(self, now):
        try:
            names = os.listdir(self.dir)
        except OSError:
            return
        cut = time.strftime("%Y-%m-%d", time.localtime(now - self.keep_days * 86400))
        for n in names:
            if n.startswith("telemetry-") and n.endswith(".jsonl") and n[10:20] < cut:
                try:
                    os.remove(os.path.join(self.dir, n))
                except OSError:
                    pass

    def summary(self, t0, t1):
        with self.lock:
            pts = list(self.points)
        return window_summary(pts, t0, t1)

    def series(self, t0, t1, max_points=600):
        """The points of [t0, t1], thinned evenly to at most `max_points` (a request's detail file)."""
        with self.lock:
            pts = [p for p in self.points if t0 <= p["t"] <= t1]
        step = max(1, -(-len(pts) // max_points))
        return pts[::step]


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    def __init__(self, extra=None, gpu_index=0, gpu_indices=None, busy_fn=None, model_path=None, amd=False,
                 pid_fn=None, log_dir=None, keep_days=14):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_index`: the card the
        engine runs on, numbered as nvidia-smi and NVML number them (by PCI bus); `gpu_indices`: all of them when
        the model is split across several (issue #112) - the gpu_* readings are then their total (memory, power,
        PCIe traffic), mean (load) or hottest (temperature), and "gpus" has each card's own.  `amd`: the AMD backend's
        cards, numbered as HIP numbers them, read from sysfs (#301).  `pid_fn()` -> the engine's process id or None,
        for its working set and page faults; `log_dir`: where TelemetryLog writes its daily files (None: memory only),
        kept `keep_days` days (xeno #237)."""
        self.extra = extra
        self.busy_fn = busy_fn            # True while a request runs: the cards are then sampled five times a second
        self.pid_fn = pid_fn
        self.mem = _MemPressure()
        self.log = TelemetryLog(log_dir, keep_days)
        self.last_record = 0.0
        self.disks_now = None
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
        self.gpus = [(i, gpu_reader(i, amd)) for i in idx]
        self.gpus = [(i, g) for i, g in self.gpus if g.ok()] or self.gpus[:1]
        self.gpu = self.gpus[0][1]
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        self.static = {
            "gpu_name": " + ".join(g.name() or "?" for _, g in self.gpus) if self.gpu.ok() else None,
            "gpu_count": len(self.gpus),
            # #1380: the AMD readings are the amdgpu driver's Linux sysfs files; a Windows AMD card has none yet, and the
            # dashboard said "not readable (NVML)" or showed empty tiles with no word why
            "gpu_note": ("no GPU load or VRAM readings for AMD cards on Windows yet (Linux reads them from the amdgpu "
                         "driver); the engine's own VRAM figures are in its log" if amd and not self.gpu.ok() else None),
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
        }
        self._disk_prev = None
        self._stop = threading.Event()
        threading.Thread(target=self._storage_facts, args=(model_path,), daemon=True).start()
        threading.Thread(target=self._loop, daemon=True).start()

    def _storage_facts(self, model_path):
        """The disks' model / bus / media, and which one holds the model: asked once, off the start-up path."""
        from serve import storage
        try:
            disks = storage.physical_disks()
            self.static["storage"] = {"disks": disks, "model_disk": storage.disk_of_path(model_path) if model_path else None}
        except Exception:  # noqa: BLE001 - telemetry must never take the server down
            self.static["storage"] = {"disks": [], "model_disk": None}

    def interval(self):
        return 0.2 if self.busy_fn is not None and self.busy_fn() else 1.0

    def should_record(self, now):
        """The history takes one point a second, whatever the sampling rate."""
        if now - self.last_record >= 1.0:
            self.last_record = now
            return True
        return False

    @staticmethod
    def history_keys(s):
        keys = ["gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used", "disk_read_mb",
                "tok_s", "prefill_tok_s_mean", "ram_avail", "hard_pf_s", "engine_ws", "engine_pf_s"]   # the last four: #237
        for g in s.get("gpus", []):
            keys += [f"gpu{g['index']}_{k}" for k in ("util", "temp", "power", "mem_used", "pcie_rx_mb", "pcie_tx_mb")]
        return keys

    def _disk(self):
        if not self.ps:
            return None, None
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None
        if c is None:   # psutil found no disk (a gVisor container, Windows with its disk counters off)
            return None, None
        t = time.time()
        prev, self._disk_prev = self._disk_prev, (t, c.read_bytes, c.write_bytes)
        if prev is None or t <= prev[0]:
            return None, None
        dt = t - prev[0]
        return (c.read_bytes - prev[1]) / dt / 2**20, (c.write_bytes - prev[2]) / dt / 2**20

    def _disks_per(self):
        if not self.ps:
            return None
        try:
            cur = self.ps.disk_io_counters(perdisk=True)
        except (OSError, RuntimeError, ValueError):
            return None
        if cur is None:   # psutil found no disk (a gVisor container, Windows with its disk counters off)
            return None
        t = time.time()
        prev, self._disks_prev = getattr(self, "_disks_prev", None), (t, cur)
        if prev is None or t <= prev[0]:
            return None
        from serve import storage
        return storage.disk_rates(prev[1], cur, t - prev[0])

    def sample(self):
        s = {}
        if self.gpu.ok():
            reads = [(i, g.read()) for i, g in self.gpus]
            g = dict(reads[0][1])
            if len(reads) > 1:
                def vals(k):
                    return [r[k] for _, r in reads if r.get(k) is not None]
                for k in ("mem_used", "mem_total", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb"):
                    v = vals(k)
                    g[k] = sum(v) if v else None
                u = vals("util")
                g["util"] = sum(u) / len(u) if u else None
                t = vals("temp")
                g["temp"] = max(t) if t else None
            names = self.__dict__.setdefault("_card_names", {})
            s["gpus"] = [{"index": i, "name": names.setdefault(i, gc.name()), **{k: r.get(k) for k in (
                "util", "mem_used", "mem_total", "temp", "power", "power_limit", "pcie_gen", "pcie_gen_max", "pcie_width",
                "pcie_width_max", "pcie_rx_mb", "pcie_tx_mb", "sm_clock", "mem_clock", "throttle")}}
                for (i, gc), (_, r) in zip(self.gpus, reads)]
            s.update({f"gpu_{k}": v for k, v in g.items()})
        if self.ps:
            try:
                s["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                s["ram_used"], s["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        else:
            s["cpu"] = self.fallback.cpu()
            s["ram_used"], s["ram_total"] = self.fallback.ram()
        s["disk_read_mb"], s["disk_write_mb"] = self._disk()
        s["disks"] = self._disks_per()        # per disk (psutil only): MB/s and the OS-observed read latency
        mem = getattr(self, "mem", None)
        if mem is not None:                   # xeno #237: RAM left, commit, hard faults, the engine's working set
            try:
                pid = self.pid_fn() if getattr(self, "pid_fn", None) else None
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pid = None
            s.update(mem.read(pid))
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return s

    def close(self):
        """Ends the sampler thread (a server that stops, a test's service): it used to run for the life of the process."""
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            s = self.sample()
            now = time.time()
            with self.lock:
                self.now = s
                record = self.should_record(now)
                if record:
                    for k in self.history_keys(s):
                        v = s.get(k)
                        if v is None and k.startswith("gpu") and k[3].isdigit():       # a card's own series
                            idx, _, field = k[3:].partition("_")
                            v = next((g.get(field) for g in s.get("gpus", []) if str(g["index"]) == idx), None)
                        self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            if record:                        # outside self.lock: the log has its own, and a disk write may be slow
                try:
                    self.log.add(s, now, bool(self.busy_fn()) if self.busy_fn else False)
                except Exception:  # noqa: BLE001 - telemetry must never take the server down
                    pass
            self._stop.wait(self.interval())

    def request_summary(self, t0, t1):
        """The hardware over [t0, t1] for a finished request's record (xeno #237), or None."""
        return self.log.summary(t0, t1)

    def request_series(self, t0, t1):
        """The per-second points of [t0, t1] for a request's detail file (xeno #237)."""
        return self.log.series(t0, t1)

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}
