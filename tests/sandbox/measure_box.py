"""Latency of the AppContainer box on this machine, and what a box can still see. Run it by hand, with nothing else heavy
running: python tests/sandbox/measure_box.py [rounds]

Each row is `rounds` paired steps in one session. "outside" is the same private runtime started without a box, so the
difference is the wall and not the copy. The first box of a process is dropped: it pays for the scratch root and the sweep.
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from serve.sandbox.isolation import Isolation, Policy, Scratch, build_runtime  # noqa: E402

NO_WINDOW = 0x08000000
POLICY = Policy(memory_bytes=1 << 30, seconds=120, processes=16, disk_bytes=256 << 20)
PASS = ["python", "-c", "pass"]
SYMPY = ["python", "-c", "import sympy"]
NUMPY = ["python", "-c", "import numpy"]


def timed(work):
    began = time.perf_counter()
    out = work()
    return (time.perf_counter() - began) * 1000, out


def row(name, samples):
    print(f"{name:44} mean {statistics.mean(samples):8.1f}  median {statistics.median(samples):8.1f}  "
          f"min {min(samples):8.1f}  max {max(samples):8.1f}")


def outside(runtime, argv):
    env = {"SystemRoot": os.environ["SystemRoot"], "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
           "OPENBLAS_NUM_THREADS": "1"}
    return subprocess.run([str(runtime / "python.exe"), *argv[1:]], capture_output=True, creationflags=NO_WINDOW, env=env, check=True)


def inside(box, argv):
    result = box.run(argv, timeout=60)
    assert result.exit_code == 0, result
    return result


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    scratch = Scratch()
    ms, runtime = timed(lambda: build_runtime(scratch))
    manifest = json.loads((runtime / "manifest.json").read_text(encoding="utf-8"))
    print(f"backends: {Isolation.report()}")
    print(f"runtime {runtime}: {manifest['files']} files, {manifest['bytes'] >> 20} MiB, ready in {ms:.0f} ms")
    with Isolation.start(POLICY) as box:
        inside(box, SYMPY)
        inside(box, NUMPY)

    start, kill, first, first_sympy = [], [], [], []
    for _ in range(rounds):
        ms, box = timed(lambda: Isolation.start(POLICY))
        start.append(ms)
        first.append(inside(box, PASS).wall_ms)
        first_sympy.append(inside(box, SYMPY).wall_ms)
        kill.append(timed(box.kill)[0])
    print(f"\n{rounds} rounds, ms")
    row("box start", start)
    row("python -c pass, first run in a new box", first)
    row("import sympy, second run in that box", first_sympy)
    row("box kill and cleanup", kill)

    in_python, out_python, in_sympy, out_sympy, in_numpy, out_numpy = [], [], [], [], [], []
    with Isolation.start(POLICY) as box:
        for _ in range(rounds):
            in_python.append(inside(box, PASS).wall_ms)
            out_python.append(timed(lambda: outside(runtime, PASS))[0])
            in_sympy.append(inside(box, SYMPY).wall_ms)
            out_sympy.append(timed(lambda: outside(runtime, SYMPY))[0])
            in_numpy.append(inside(box, NUMPY).wall_ms)
            out_numpy.append(timed(lambda: outside(runtime, NUMPY))[0])
    row("python -c pass, in a warm box", in_python)
    row("python -c pass, same runtime outside", out_python)
    row("import sympy, in a warm box", in_sympy)
    row("import sympy, same runtime outside", out_sympy)
    row("import numpy, in a warm box", in_numpy)
    row("import numpy, same runtime outside", out_numpy)

    home = Path.home()
    paths = ["C:\\", "C:\\Users", "C:\\Users\\Public", str(home), str(home / "Documents"), "C:\\Windows", "C:\\Windows\\System32",
             "C:\\Windows\\System32\\config", "C:\\Program Files", "C:\\Program Files (x86)", "C:\\ProgramData", "D:\\", "D:\\Github",
             str(ROOT), sys.base_prefix, "\\\\localhost\\C$", "\\\\wsl.localhost"]
    with Isolation.start(POLICY) as box:
        shutil.copy(ROOT / "tests" / "sandbox" / "escape" / "survey.py", box.workdir)
        result = box.run(["python", "survey.py", json.dumps(paths)], timeout=60)
        print(f"\nwhat a box can still see (exit {result.exit_code}):")
        if result.exit_code:
            print(result.stderr.decode(errors="replace")[-400:])
            return
        report = json.loads(result.stdout)
        for path, seen in report.pop("paths").items():
            print(f"  {path:58} {seen if isinstance(seen, str) else str(seen) + ' names listed'}")
        for name, value in report.items():
            print(f"  {name:58} {value}")


if __name__ == "__main__":
    main()
