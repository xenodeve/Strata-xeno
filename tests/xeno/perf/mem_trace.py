"""RAM over time for one strata run: the machine's in-use and committed memory and the process's private bytes and
working set every 0.5 s, with each stderr line time-stamped, so the boot peak and the steady state can be read
from the same run.

usage: python mem_trace.py OUT_NAME PROMPT exe [extra args...]
"""
import json, os, subprocess, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab  # noqa: E402

out_name, prompt, exe, extra = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
out = os.path.join(ab.T, out_name)
os.makedirs(out, exist_ok=True)
cmd = json.load(open(os.path.join(ab.T, ab.REF[prompt], "on.command.json")))[1:] + ["--pool-priority", "2"] + extra
env = dict(os.environ, CUDA_VISIBLE_DEVICES="1,0")
cuda = os.path.join("C:" + os.sep, "Program Files", "NVIDIA GPU Computing Toolkit", "CUDA", "v13.3", "bin")
env["PATH"] = os.pathsep.join([cuda, os.path.join(cuda, "x64"), env["PATH"]])
t0 = time.time()
pr = subprocess.Popen([exe] + cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                      encoding="utf-8", errors="replace", creationflags=0x80 | 0x08000000)   # HIGH class, CREATE_NO_WINDOW
samples, events = [], []


def sample():
    while pr.poll() is None:
        u, _, c = ab.system_memory()
        pm = ab.process_memory(pr.pid) or (0.0, 0.0)
        samples.append((time.time() - t0, u, c, pm[0], pm[1]))
        time.sleep(0.5)


def read_err():
    for line in pr.stderr:
        events.append((time.time() - t0, line.rstrip()))


th = [threading.Thread(target=sample, daemon=True), threading.Thread(target=read_err, daemon=True)]
for t in th: t.start()
stdout = pr.stdout.read()
pr.wait()
for t in th: t.join()
with open(os.path.join(out, "trace.tsv"), "w", encoding="utf-8") as f:
    f.write("t_s\tsys_in_use_gib\tsys_commit_gib\tproc_private_gib\tproc_ws_gib\n")
    for s in samples: f.write("\t".join(f"{v:.2f}" for v in s) + "\n")
with open(os.path.join(out, "events.txt"), "w", encoding="utf-8") as f:
    for t, line in events: f.write(f"{t:8.2f}  {line}\n")
open(os.path.join(out, "stdout.txt"), "w", encoding="utf-8").write(stdout)
peak = max(samples, key=lambda s: s[1]) if samples else None
print(f"rc={pr.returncode} samples={len(samples)} peak in-use {peak[1]:.2f} GiB at {peak[0]:.1f}s" if peak else "no samples")
