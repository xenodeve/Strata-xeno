"""Routing traces: six training prompts (trace-prompts/*.ids) + the four benchmark prompts, one run each.

Same command as the benchmark (codex dispatch-detail code256) with --tokens-file swapped, --dump-routing added (format 2, #93),
and --pool-priority 2 (AGENTS.md). Waits for a running sweep (argv[1] summary.log) to finish first.
"""
import glob, json, os, subprocess, sys, time

T = os.environ["TEMP"]
HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(T, "strata-claude-exe", "x2-exp.exe")
OUT = os.path.join(T, "strata-claude-traces")
REF = {"code": "strata-codex-dispatch-detail-code256", "thai": "strata-codex-dispatch-detail-thai256",
       "sky": "strata-codex-compact-parity-sky256", "long": "strata-codex-compact-parity-long256"}

if len(sys.argv) > 1:
    while "DONE" not in open(sys.argv[1], encoding="utf-8").read():
        time.sleep(20)
os.makedirs(OUT, exist_ok=True)
base = json.load(open(os.path.join(T, REF["code"], "on.command.json")))[1:]
i = base.index("--tokens-file")
jobs = [(os.path.basename(p)[:-4], p) for p in sorted(glob.glob(os.path.join(HERE, "trace-prompts", "*.ids")))]
jobs += [("bench-" + k, os.path.join(T, v, "prompt.ids")) for k, v in REF.items()]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="1,0")
cuda = os.path.join("C:" + os.sep, "Program Files", "NVIDIA GPU Computing Toolkit", "CUDA", "v13.3", "bin")
env["PATH"] = os.pathsep.join([cuda, os.path.join(cuda, "x64"), env["PATH"]])
log = open(os.path.join(OUT, "summary.log"), "a", encoding="utf-8")
for name, ids in jobs:
    cmd = base[:i + 1] + [ids] + base[i + 2:] + ["--dump-routing", os.path.join(OUT, name + ".bin"), "--pool-priority", "2"]
    p = subprocess.run([EXE] + cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    open(os.path.join(OUT, name + ".stdout"), "w", encoding="utf-8").write(p.stdout)
    open(os.path.join(OUT, name + ".stderr"), "w", encoding="utf-8").write(p.stderr)
    tier = [l for l in p.stdout.splitlines() if l.startswith("tier hits")]
    log.write(f"{name} rc={p.returncode} {tier[0] if tier else ''}\n"); log.flush()
log.write("DONE\n")
