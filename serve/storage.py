"""serve/storage.py - the disks behind the model (xeno UI S6): model, bus and media of each physical disk, which one
holds a file, and per-disk rates from two counter snapshots. Facts and OS counters only: a figure that cannot be read
is None ("not measured" in the app), never a guess.

Windows asks PowerShell once (cached by the caller); Linux reads /sys/block. Not covered here: the NVMe's own PCIe
link (SetupAPI) and a device-level latency (a boot microbench) - the app says so.
"""
from __future__ import annotations

import json
import os
import re
import subprocess


def _ps(script: str, run):
    r = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True,
            timeout=20)
    return r.stdout if getattr(r, "returncode", 1) == 0 else ""


def physical_disks(os_name: str = os.name, run=subprocess.run) -> list[dict]:
    """[{index, model, bus, media, size_gb}] - index as Windows' disk number (psutil's PhysicalDriveN)."""
    try:
        if os_name == "nt":
            out = _ps("Get-PhysicalDisk | Select-Object DeviceId,FriendlyName,BusType,MediaType,Size | ConvertTo-Json -Compress", run)
            data = json.loads(out) if out.strip() else []
            data = [data] if isinstance(data, dict) else data
            return sorted(({"index": int(d["DeviceId"]), "model": (d.get("FriendlyName") or "").strip() or None,
                            "bus": d.get("BusType") or None, "media": d.get("MediaType") or None,
                            "size_gb": round(int(d["Size"]) / 2**30, 1) if d.get("Size") else None} for d in data),
                          key=lambda d: d["index"])
        disks = []
        for i, name in enumerate(sorted(n for n in os.listdir("/sys/block") if re.match(r"(nvme\d+n\d+|sd[a-z]+|vd[a-z]+)$", n))):
            def read(p):
                try:
                    return open(f"/sys/block/{name}/{p}", encoding="utf-8").read().strip()
                except OSError:
                    return None
            rot = read("queue/rotational")
            disks.append({"index": i, "model": read("device/model"), "bus": "NVMe" if name.startswith("nvme") else "SATA/SCSI",
                          "media": None if rot is None else "HDD" if rot == "1" else "SSD",
                          "size_gb": round(int(read("size") or 0) * 512 / 2**30, 1) or None, "device": name})
        return disks
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return []


def disk_of_path(path: str, os_name: str = os.name, run=subprocess.run):
    """The disk number holding `path`, or None when it cannot be told (no drive letter, no PowerShell, a network path)."""
    try:
        if os_name != "nt":
            return None                                   # Linux: the caller matches by device name when it needs to
        m = re.match(r"^([A-Za-z]):[\\/]", str(path))
        if not m:
            return None
        out = _ps(f"(Get-Partition -DriveLetter {m.group(1)} | Select-Object -First 1).DiskNumber", run).strip()
        return int(out) if out.isdigit() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def disk_rates(prev: dict, cur: dict, dt: float) -> list[dict]:
    """Per-disk MB/s and the OS-observed read latency (ms per read, queue included) between two
    psutil.disk_io_counters(perdisk=True) snapshots."""
    out = []
    for name, c in sorted(cur.items()):
        m = re.search(r"(\d+)$", name)
        p = prev.get(name)
        if not m or p is None or dt <= 0:
            continue
        reads = c.read_count - p.read_count
        out.append({"index": int(m.group(1)), "read_mb": round((c.read_bytes - p.read_bytes) / dt / 2**20, 2),
                    "write_mb": round((c.write_bytes - p.write_bytes) / dt / 2**20, 2),
                    "read_ms_op": round((c.read_time - p.read_time) / reads, 3) if reads > 0 else None})
    return sorted(out, key=lambda r: r["index"])
