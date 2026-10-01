"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc) and the disk rate is absent.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
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


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    def __init__(self, extra=None, gpu_index=0, gpu_indices=None, busy_fn=None, model_path=None):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_index`: the card the
        engine runs on, numbered as nvidia-smi and NVML number them (by PCI bus); `gpu_indices`: all of them when
        the model is split across several (issue #112) - the gpu_* readings are then their total (memory, power,
        PCIe traffic), mean (load) or hottest (temperature), and "gpus" has each card's own."""
        self.extra = extra
        self.busy_fn = busy_fn            # True while a request runs: the cards are then sampled five times a second
        self.last_record = 0.0
        self.disks_now = None
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
        self.gpus = [(i, _Nvml(i)) for i in idx]
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
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
        }
        self._disk_prev = None
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
                "tok_s", "prefill_tok_s_mean"]
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
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return s

    def _loop(self):
        while True:
            s = self.sample()
            with self.lock:
                self.now = s
                if self.should_record(time.time()):
                    for k in self.history_keys(s):
                        v = s.get(k)
                        if v is None and k.startswith("gpu") and k[3].isdigit():       # a card's own series
                            idx, _, field = k[3:].partition("_")
                            v = next((g.get(field) for g in s.get("gpus", []) if str(g["index"]) == idx), None)
                        self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            time.sleep(self.interval())

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}
