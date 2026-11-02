"""Breach: leave a process behind that lives on after the box ends. Tries to break away from the job first, then settles
for a detached child. The child sleeps a short while and ends by itself, so a breach cannot stay on the machine."""
import subprocess
import sys

NO_WINDOW, NEW_GROUP, BREAKAWAY = 0x08000000, 0x00000200, 0x01000000
sleeper = [sys.executable, "-c", f"import time; time.sleep({float(sys.argv[1])})"]
try:
    child = subprocess.Popen(sleeper, creationflags=NO_WINDOW | NEW_GROUP | BREAKAWAY)
    print(f"broke away pid {child.pid}", flush=True)
except OSError as error:
    print(f"breakaway refused: {error!r}", flush=True)
    child = subprocess.Popen(sleeper, creationflags=NO_WINDOW | NEW_GROUP)
    print(f"stayed pid {child.pid}", flush=True)
