"""Phase 8 (#8) acceptance at the engine seam, through the speculative verifier:
  temperature 0 reproduces greedy; greedy repeats; a seeded request repeats; another seed differs; sampling is live.
  With `nospec` the same requests run with --spec 2 and no lookup drafts; comparing the two JSON dumps shows whether
  the sampled output depends on speculation (on 2026-09-30 it did not: identical for every request and seed).
usage: phase8_sampler_check.py EXE [nospec]   (EXE and the prompt ids live in %TEMP%/strata-claude-stage)"""
import os
import subprocess
import sys

D = os.path.join(os.environ["TEMP"], "strata-claude-stage")
M = r"C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0"
exe = os.path.join(D, sys.argv[1] + ".exe")
nospec = len(sys.argv) > 2 and sys.argv[2] == "nospec"
ids = [t for t in open(os.path.join(D, "prompt8k.ids"), encoding="utf-8").read().replace("\n", ",").strip(",").split(",")]
req = ",".join(t for t in ids[2000:3000] if t != "248045")
args = [exe, "--serve", "--pack", r"D:\Github\Strata\packs\q2_0",
        "--native", M + r"\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
        "--ple-gguf", M + r"\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",
        "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", r"D:\Github\Strata\mtp\rt", "--kv", "int8",
        "--max-context", "16384", "--pcie-frac", "0", "--expert-cache", "8000",
        "--expert-profile", r"C:\Users\xenod\AppData\Local\Temp\strata-benchmark-full-profile.bin",
        "--vram-reserve-mib", "2400", "--secondary-free-floor-mib", "640", "--secondary-expert-mib", "8448",
        "--pool-priority", "2", "--process-priority", "2"]
if nospec:   # another draft window and no lookup drafts: the sampled output must not depend on them
    i = args.index("--spec"); args[i + 1] = "2"
    args += ["--suffix-draft", "0"]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="1,0")
env["PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;" + env["PATH"]
with open(os.path.join(D, "phase8.stderr"), "w") as err:
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, env=env)
    for line in p.stdout:
        if line.startswith("READY"):
            break

    def gen(keys):
        p.stdin.write(f"GEN 64 {keys}{' ' if keys else ''}{req}\n")
        p.stdin.flush()
        out = []
        for line in p.stdout:
            if line.startswith("T "):
                out.append(int(line[2:]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                return out, line.strip()
        return out, "EOF"

    runs = {name: gen(keys) for name, keys in [
        ("greedy", ""), ("temp0", "temperature=0"), ("s42a", "temperature=0.8 top_p=0.95 seed=42"),
        ("s42b", "temperature=0.8 top_p=0.95 seed=42"), ("s43", "temperature=0.8 top_p=0.95 seed=43"),
        ("greedy2", "")]}
    p.stdin.close()
    p.wait(timeout=120)
import json
json.dump({k: v[0] for k, v in runs.items()}, open(os.path.join(D, "phase8_" + ("nospec" if nospec else "spec") + ".json"), "w"))
for k, (toks, done) in runs.items():
    print(f"{k:8} {len(toks):3d} tokens  {done[:60]}  {toks[:6]}")
checks = {
    "temperature 0 == greedy": runs["temp0"][0] == runs["greedy"][0] and len(runs["greedy"][0]) > 0,
    "greedy repeats": runs["greedy2"][0] == runs["greedy"][0],
    "seed 42 repeats": runs["s42a"][0] == runs["s42b"][0] and len(runs["s42a"][0]) > 0,
    "seed 43 differs from 42": runs["s43"][0] != runs["s42a"][0],
    "sampled differs from greedy": runs["s42a"][0] != runs["greedy"][0],
}
for k, v in checks.items():
    print(f"{'PASS' if v else 'FAIL'}  {k}")
