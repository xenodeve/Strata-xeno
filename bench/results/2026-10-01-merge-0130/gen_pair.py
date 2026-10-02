"""#56: generate mode, the same prompt and args, the 0.1.26 exe against a build of the merge: exit code, output ids
hash and decode speed.  usage: python gen_pair.py NEW_EXE [prompt dir name] [max_new]"""
import hashlib
import json
import os
import re
import subprocess
import sys

S = os.path.dirname(os.path.abspath(__file__))
new_exe = sys.argv[1]
prompt = sys.argv[2] if len(sys.argv) > 2 else "strata-phase4-verified256-code"
max_new = sys.argv[3] if len(sys.argv) > 3 else "256"
cfg = json.load(open(os.path.join(S, "strata-live-130.json"), encoding="utf-8"))
args = list(cfg["args"])
i = args.index("--max-context")
del args[i:i + 2]
if os.environ.get("GP_TIER"):
    args[args.index("--secondary-expert-mib") + 1] = os.environ["GP_TIER"]
args += ["--tokens-file", os.path.join(os.environ["TEMP"], prompt, "prompt.ids"), "--max-new", max_new, "--greedy",
         "--max-context", "16384"]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="1,0", STRATA_PREFILL_EXPERT_SPLIT="1", STRATA_PREFILL_WAVE="1",
           STRATA_EXP_QUICK_EXIT="1")
env["PATH"] = os.pathsep.join(cfg["lib_dirs"] + [env["PATH"]])
for name, exe in (("0.1.26", r"C:\Strata-exp\run\strata-26f5974288933cb3.exe"), ("merge", new_exe)):
    p = subprocess.run([exe] + args, env=env, cwd=cfg["cwd"], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=900)
    out = re.search(r"^output\s*:\s*(.*)$", p.stdout, re.M)
    tps = re.search(r"^decode\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s", p.stdout, re.M)
    print(f"{name}: rc {p.returncode:#x} out {hashlib.sha256(out[1].encode()).hexdigest()[:8] if out else None} "
          f"decode {tps[1] if tps else None} tok/s")
    if p.returncode != 0:
        print("\n".join(p.stderr.splitlines()[-8:]))
