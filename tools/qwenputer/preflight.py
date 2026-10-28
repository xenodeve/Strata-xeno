"""P0 metadata and hashing only; never loads a model or runs upstream code."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def sha256_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"input changed while hashing: {path}")
    return digest.hexdigest()


def _walk(root: Path):
    def fail(error):
        raise error

    for base, dirs, names in os.walk(root, followlinks=False, onerror=fail):
        # Do not recurse through junctions or another repository's metadata.
        kept = []
        for name in sorted(dirs):
            path = Path(base) / name
            attrs = getattr(path.lstat(), "st_file_attributes", 0)
            if name != ".git" and not path.is_symlink() and not attrs & 0x400:
                kept.append(name)
        dirs[:] = kept
        for name in sorted(names):
            yield Path(base) / name


def _file(path: Path) -> dict:
    print(f"hash {path}", file=sys.stderr, flush=True)
    return {"path": str(path), "resolved_path": str(path.resolve()),
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _input(role: str, path: Path) -> dict:
    row = {"role": role, "path": str(path), "exists": path.exists()}
    if not row["exists"]:
        row["status"] = "missing"
    elif path.is_file():
        row.update(_file(path), status="pinned")
    elif path.is_dir():
        row["files"] = [_file(p) for p in _walk(path) if p.is_file()]
        tree = hashlib.sha256()
        for item in row["files"]:
            relative = str(Path(item["path"]).relative_to(path)).replace("\\", "/")
            tree.update(f"{relative}\0{item['size_bytes']}\0{item['sha256']}\n".encode())
        row.update(status="pinned", tree_sha256=tree.hexdigest(),
                   size_bytes=sum(x["size_bytes"] for x in row["files"]))
    else:
        row["status"] = "not_regular"
    return row


def collect_inputs(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = config.get("args", [])
    paths = []
    for key in ("exe", "tokenizer"):
        if config.get(key):
            paths.append((key, Path(config[key])))
    roles = {"--pack": "pack", "--native": "native", "--ple-gguf": "ple",
             "--mtp": "mtp", "--expert-profile": "expert_profile"}
    for flag, role in roles.items():
        if flag in args:
            i = args.index(flag)
            if i + 1 >= len(args):
                raise ValueError(f"missing value for {flag}")
            paths.append((role, Path(args[i + 1])))
    return {"config_path": str(config_path), "config_sha256": sha256_file(config_path),
            "args": args, "env": config.get("env", {}),
            "inputs": [_input(role, path) for role, path in paths]}


def find_models(roots: list[Path], filenames: set[str]) -> dict:
    result = {"roots": [str(p) for p in roots], "matches": [], "errors": [], "files_seen": 0}
    for root in roots:
        print(f"search {root}", file=sys.stderr, flush=True)
        try:
            if not root.is_dir():
                raise FileNotFoundError(str(root))
            for path in _walk(root):
                result["files_seen"] += 1
                if result["files_seen"] > 250_000:
                    raise RuntimeError("search file limit reached; search is incomplete")
                if path.name in filenames and path.is_file():
                    result["matches"].append(_file(path))
        except (OSError, RuntimeError) as error:
            result["errors"].append({"root": str(root), "error": str(error)})
    return result


def host_snapshot() -> dict:
    snapshot = {"platform": platform.platform(), "python": platform.python_version(),
                "logical_processors": os.cpu_count()}
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32)] + [
                (name, ctypes.c_uint64) for name in (
                    "total_physical", "available_physical", "total_pagefile", "available_pagefile",
                    "total_virtual", "available_virtual", "available_extended_virtual")]
        memory = Memory()
        memory.length = ctypes.sizeof(memory)
        if ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx(ctypes.byref(memory)):
            snapshot["memory_bytes"] = {name: getattr(memory, name) for name, _ in Memory._fields_[2:]}
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                snapshot["processor_name"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError as error:
            snapshot["processor_error"] = str(error)
    command = ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
               "--format=csv"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=20,
                              creationflags=0x08000000 if os.name == "nt" else 0)
        snapshot["nvidia_smi"] = {"command": command, "exit_code": proc.returncode,
                                  "stdout": proc.stdout, "stderr": proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        snapshot["nvidia_smi_error"] = str(error)
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, action="append", default=[])
    parser.add_argument("--model-name", action="append", default=[])
    parser.add_argument("--extra-input", type=Path, action="append", default=[])
    args = parser.parse_args()
    report = {"schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "host_before": host_snapshot(), "profile": collect_inputs(args.config),
              "model_search": find_models(args.search_root, set(args.model_name)),
              "extra_inputs": [_input("extra", path) for path in args.extra_input],
              "host_after": host_snapshot(), "finished_utc": datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    missing = [x["role"] for x in report["profile"]["inputs"] if not x["exists"]]
    print(json.dumps({"manifest": str(args.output), "missing_roles": missing,
                      "model_matches": len(report["model_search"]["matches"])}))
    print("DONE preflight", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
