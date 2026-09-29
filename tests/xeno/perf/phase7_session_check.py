"""Phase 7 (#7) acceptance at the engine seam: a ~100K-token session, a follow-up turn, another session in between,
then a return to the first - the continuation must reuse the cached prefix and give the same tokens as a fresh
process reading the whole conversation.  Memory and TTFT from the engine's own lines.
usage: phase7_check.py EXE [CTX]"""
import os
import re
import subprocess
import sys
import time

D = os.path.join(os.environ["TEMP"], "strata-claude-stage")
M = r"C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0"
exe = os.path.join(D, sys.argv[1] + ".exe")
ctx = sys.argv[2] if len(sys.argv) > 2 else "131072"
T = "248045"   # <|im_start|>: the conversation cache's turn token
body = [t for t in open(os.path.join(D, "prompt8k.ids"), encoding="utf-8").read().replace("\n", ",").strip(",").split(",")
        if t != T]
long_text = (body * 13)[:100000]
A1 = [T] + long_text + [T] + body[100:300]
A2 = A1 + [T] + body[400:500]
B = [T] + body[1000:3000] + [T] + body[3000:3100]
A3 = A2 + [T] + body[600:650]
args = [exe, "--serve", "--pack", r"D:\Github\Strata\packs\q2_0",
        "--native", M + r"\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
        "--ple-gguf", M + r"\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",
        "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", r"D:\Github\Strata\mtp\rt", "--kv", "int8",
        "--max-context", ctx, "--pcie-frac", "0", "--expert-cache", "8000",
        "--expert-profile", r"C:\Users\xenod\AppData\Local\Temp\strata-benchmark-full-profile.bin",
        "--vram-reserve-mib", "2400", "--secondary-free-floor-mib", "640", "--secondary-expert-mib", "8448",
        "--pool-priority", "2", "--process-priority", "2"]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="1,0")
env["PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;" + env["PATH"]


def server(tag):
    err = open(os.path.join(D, f"phase7_{tag}.stderr"), "w")
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, env=env)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    return p


def gen(p, toks, k=16):
    t0 = time.time()
    p.stdin.write(f"GEN {k} {','.join(toks)}\n")
    p.stdin.flush()
    out, first = [], None
    for line in p.stdout:
        if line.startswith("T "):
            if first is None:
                first = time.time() - t0
            out.append(int(line[2:]))
        elif line.startswith("DONE") or line.startswith("ERR"):
            return out, line.strip(), first, time.time() - t0
    return out, "EOF", first, time.time() - t0


p = server("cached")
res = {}
for name, toks in (("A1", A1), ("A2", A2), ("B", B), ("A3", A3)):
    out, done, ttft, wall = gen(p, toks)
    res[name] = out
    f = done.split()
    print(f"{name}: {len(toks):6d} tokens, reused {f[8] if len(f) > 8 else '?':>6}, TTFT {ttft or 0:6.2f} s, "
          f"wall {wall:6.1f} s | {done[:70]}", flush=True)
p.stdin.close()
p.wait(timeout=300)
p = server("fresh")
out, done, ttft, wall = gen(p, A3)
print(f"A3 fresh: {len(A3)} tokens, TTFT {ttft or 0:.2f} s | {done[:70]}", flush=True)
p.stdin.close()
p.wait(timeout=300)
ok = out == res["A3"] and len(out) > 0
print(f"A3 continuation vs fresh read: {'IDENTICAL' if ok else 'DIFFERENT'} ({res['A3'][:6]} vs {out[:6]})")
txt = open(os.path.join(D, "phase7_cached.stderr"), encoding="utf-8", errors="replace").read()
for m in re.findall(r"private commit [\d.]+ GiB", txt)[-4:]:
    print("  cached server:", m)
print("PASS" if ok else "FAIL")
