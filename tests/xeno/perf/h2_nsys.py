"""H2: Nsight Systems capture of the real verifier decode on the current exe, then per-device kernel classes.

Same command as the benchmark (codex dispatch-detail code256) + best measured flags + --profile-decode-range.
Waits for a running A/B (argv[1] summary.log) to finish so the capture is not disturbed.
"""
import json, os, sqlite3, subprocess, sys, time

T = os.environ["TEMP"]
NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe"
EXE = os.path.join(T, "strata-claude-exe", "x7-sec.exe")
OUT = os.path.join(T, "strata-claude-h2-nsys")
if len(sys.argv) > 1:
    while "DONE" not in open(sys.argv[1], encoding="utf-8").read():
        time.sleep(20)
os.makedirs(OUT, exist_ok=True)
base = json.load(open(os.path.join(T, "strata-codex-dispatch-detail-code256", "on.command.json")))[1:]
extra = ["--pool-priority", "2", "--adapt-swaps", "8", "--adapt-every", "1", "--adapt-secondary", "8",
         "--profile-decode-range"]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="1,0")
cuda = os.path.join("C:" + os.sep, "Program Files", "NVIDIA GPU Computing Toolkit", "CUDA", "v13.3", "bin")
env["PATH"] = os.pathsep.join([cuda, os.path.join(cuda, "x64"), env["PATH"]])
rep = os.path.join(OUT, "decode")
cmd = [NSYS, "profile", "-t", "cuda", "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
       "--capture-range-end=stop-shutdown", "-s", "none", "-o", rep, "--force-overwrite", "true",
       "--export", "sqlite", EXE] + base + extra
json.dump(cmd, open(os.path.join(OUT, "command.json"), "w"), indent=1)
p = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
open(os.path.join(OUT, "stdout"), "w", encoding="utf-8").write(p.stdout)
open(os.path.join(OUT, "stderr"), "w", encoding="utf-8").write(p.stderr)
db = sqlite3.connect(rep + ".sqlite")
rounds = None
for line in p.stdout.splitlines():
    if line.startswith("speculation"):
        rounds = int(line.split()[1])
q = """select k.deviceId, s.value, count(*), sum(k.end - k.start) / 1e6
       from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName
       group by k.deviceId, s.value order by 4 desc"""
lines = [f"rounds {rounds}"]
tot = {}
for dev, name, n, ms in db.execute(q):
    tot[dev] = tot.get(dev, 0) + ms
    lines.append(f"dev{dev} {ms:9.1f} ms {ms / (rounds or 1):7.3f} ms/round  n={n:6}  {name}")
lines.insert(1, "kernel ms per device: " + ", ".join(f"dev{d} {v:.1f}" for d, v in tot.items()))
open(os.path.join(OUT, "kernels.txt"), "w").write("\n".join(lines))
print("\n".join(lines[:40]))
print("DONE")
