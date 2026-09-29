"""#35 regression (e34a1d0): the served wave with a prompt read in parts at turn boundaries.

Serve's prompt cache reads a prompt in parts (the system prompt root, the history, the new header).  A part the lent
wave layout still holds but whose lane chunk does not run split once ran one lane with the wave link attached: it read
every other chunk and planned the 4070's stream for a lane 2 that never ran - the request hung.

  serve_wave_mixed.py EXE REF_STDOUT TIER parts2
EXE and REF_STDOUT (generate mode over promptparts2.ids, split only: dual8ks.sh) are names in %TEMP%/strata-claude-stage.
promptparts2.ids is built from prompt8k.ids when missing: <|im_start|> + 5,999 tokens, <|im_start|> + 3,500,
<|im_start|> + 400.  PASS: the request finishes and its tokens match the reference (serve stops at a stop token, so
only the tokens it produced are compared).  The old exe hung: "prompt chunk 0 of 3501", killed at 300 s."""
import os
import re
import subprocess
import sys
import threading

D = os.path.join(os.environ["TEMP"], "strata-claude-stage")
M = r"C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0"
exe, ref_name, tier = os.path.join(D, sys.argv[1] + ".exe"), sys.argv[2], sys.argv[3]
ref = re.search(r"^output\s*:\s*(.*)$", open(os.path.join(D, ref_name + ".stdout"), encoding="utf-8").read(), re.M)
want = [int(x) for x in ref.group(1).split()]
ids = open(os.path.join(D, "prompt8k.ids"), encoding="utf-8").read().replace("\n", ",").strip(",").split(",")
first, second = ",".join(ids[:6000]), ",".join(ids[4000:7500])
if not os.path.exists(os.path.join(D, "promptparts2.ids")):
    T = 248045   # <|im_start|>: the prompt cache's turn token
    body = [198 if int(t) == T else int(t) for t in ids]
    parts = [T] + body[1:6000] + [T] + body[1000:4500] + [T] + body[5000:5400]
    with open(os.path.join(D, "promptparts2.ids"), "w", encoding="utf-8") as f:
        f.write(",".join(map(str, parts)) + "\n")
if len(sys.argv) > 4 and sys.argv[4] in ("parts", "parts2"):   # one request read in parts: 6,000 / 3,500 / 400
    second = open(os.path.join(D, "promptparts2.ids"), encoding="utf-8").read().strip()
args = [exe, "--serve", "--pack", r"D:\Github\Strata\packs\q2_0",
        "--native", M + r"\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
        "--ple-gguf", M + r"\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",
        "--prefill", "auto", "--spec", "3", "--spec-min-p", "0.5", "--mtp", r"D:\Github\Strata\mtp\rt", "--kv", "int8",
        "--max-context", "16384", "--pcie-frac", "0", "--expert-cache", "8000",
        "--expert-profile", r"C:\Users\xenod\AppData\Local\Temp\strata-benchmark-full-profile.bin",
        "--vram-reserve-mib", "2400", "--secondary-free-floor-mib", "640", "--secondary-expert-mib", tier,
        "--adapt-secondary", "0", "--pool-priority", "2", "--process-priority", "2"]
env = dict(os.environ, CUDA_VISIBLE_DEVICES="1,0", STRATA_PREFILL_EXPERT_SPLIT="1", STRATA_PREFILL_WAVE="1",
           STRATA_TRACE="1")
env["PATH"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;" + env["PATH"]
with open(os.path.join(D, f"mixed_{sys.argv[1]}.stderr"), "w") as err:
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, env=env)
    timer = threading.Timer(300, p.kill)   # a hang is the failure this guards: kill and report it
    timer.start()
    for line in p.stdout:
        if line.startswith("READY"):
            break

    def gen(idstr, k):
        p.stdin.write(f"GEN {k} {idstr}\n")
        p.stdin.flush()
        out, fin = [], "NO ANSWER (killed after 300 s: hung)"
        for line in p.stdout:
            if line.startswith("T "):
                out.append(int(line[2:]))
            elif line.startswith("DONE") or line.startswith("ERR"):
                fin = line.strip()
                break
        return out, fin

    parts = len(sys.argv) > 4 and sys.argv[4] in ("parts", "parts2")
    _, done1 = gen(first, 4) if not parts else ([], "skipped (parts mode)")
    got, done2 = gen(second, len(want))
    timer.cancel()
    try:
        p.stdin.close()
        p.wait(timeout=120)
    except Exception:
        p.kill()
print(f"first (6,000): {done1}")
# a stop token ends the request early (serve stops on it; generate's --max-new does not): compare what was produced
ok = len(got) > 0 and (got == want or (" stop " in done2 and got == want[:len(got)]))
print(f"second: {done2} | tokens {got} | reference {want[:max(len(got), 4)]}")
print("PASS" if ok else "FAIL")
