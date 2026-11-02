"""Not a breach: the test's helper, run outside the box. It starts a box whose program leaves a sleeper behind, prints the
sleeper's pid, and then ends the way a crashed server does, with no cleanup, so the test can see kill-on-close work."""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from serve.sandbox.isolation import Isolation, Policy  # noqa: E402

box = Isolation.start(Policy(memory_bytes=256 << 20, seconds=30, processes=4, disk_bytes=16 << 20))
shutil.copy(Path(__file__).with_name("detached_child.py"), box.workdir)
result = box.run(["python", "detached_child.py", "20"], timeout=10)
print(result.stdout.decode(), flush=True)
print(f"box {box.name}", flush=True)
os._exit(0)
