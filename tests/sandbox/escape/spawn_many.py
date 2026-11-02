"""Gentle check of the process limit, run before any bomb: start sleepers one after another and count the refusals."""
import subprocess
import sys

started, refused = [], 0
for _ in range(int(sys.argv[1])):
    try:
        started.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"], creationflags=0x08000000))
    except OSError:
        refused += 1
print(f"started {len(started)} refused {refused}", flush=True)
for child in started:
    child.kill()
