"""Breach: multiply until the machine has no room. It stops by itself at the depth given (2^(depth+1)-1 processes at most),
so a limit that fails costs a bounded number of small processes and not the machine."""
import subprocess
import sys
import time

depth = int(sys.argv[1])
children = []
if depth > 0:
    for _ in range(2):
        try:
            children.append(subprocess.Popen([sys.executable, sys.argv[0], str(depth - 1)], stdout=sys.stdout,
                                             creationflags=0x08000000))
        except OSError as error:
            print(f"refused: {error!r}", flush=True)
time.sleep(1.5)
for child in children:
    child.wait()
