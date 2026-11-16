"""Builds Strata Desktop with PyInstaller (#310 part 3), with no window:

    desktop\\.venv\\Scripts\\python desktop\\build.py

runs `desktop\\.venv\\Scripts\\pyinstaller desktop\\StrataDesktop.spec --noconfirm --distpath desktop\\dist --workpath
desktop\\build` from the repo root (CREATE_NO_WINDOW; PyInstaller's own output goes to desktop\\build\\pyinstaller.log),
then prints the exe's path, size and sha256 and the folder's size.  Exit 0 when the exe was built, else PyInstaller's
code (the end of its log is printed).  desktop\\build\\ and desktop\\dist\\ are git-ignored.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESKTOP = ROOT / "desktop"
SPEC = DESKTOP / "StrataDesktop.spec"
DIST, WORK = DESKTOP / "dist", DESKTOP / "build"
EXE = DIST / "StrataDesktop" / "StrataDesktop.exe"
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def pyinstaller() -> Path:
    scripts = DESKTOP / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    return scripts / ("pyinstaller.exe" if os.name == "nt" else "pyinstaller")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def folder_size(path: Path) -> tuple[int, int]:
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    log = WORK / "pyinstaller.log"
    cmd = [str(pyinstaller()), str(SPEC), "--noconfirm", "--distpath", str(DIST), "--workpath", str(WORK)]
    print("build:", " ".join(cmd), flush=True)
    with open(log, "w", encoding="utf-8", errors="replace") as out:
        code = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                              creationflags=NO_WINDOW).returncode
    if code != 0 or not EXE.is_file():
        print(f"build: PyInstaller exited {code}; the end of {log}:")
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))
        return code or 1
    n, size = folder_size(EXE.parent)
    print(f"exe: {EXE}")
    print(f"exe size: {EXE.stat().st_size} bytes")
    print(f"exe sha256: {sha256(EXE)}")
    print(f"folder: {n} files, {size} bytes")
    print(f"log: {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
