"""#18: does the same exe give the same tokens after a reboot?  Capture a run now, reboot, capture again, compare.

Usage:
    python tests/xeno/perf/cross_boot.py freeze EXE            copy EXE to the kit's directory (once, before the reboot)
    python tests/xeno/perf/cross_boot.py capture NAME          run the frozen exe on the code and Thai prompts
    python tests/xeno/perf/cross_boot.py compare NAME_A NAME_B first divergent token + differing boot facts

The kit lives in %TEMP%/strata-claude-det (it survives a reboot). `capture` runs the dual-GPU serving flags with
--max-new 256 --greedy, twice per prompt so a same-boot difference is caught too, records the output ids and the
boot-dependent facts the engine logs (free VRAM and cache sizing, placement, secondary sizing, prefill sources,
tier hits), plus nvidia-smi's used VRAM per card before the run. It refuses to start below 12 GB of free RAM.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

KIT = os.path.join(os.environ.get("TEMP", "."), "strata-claude-det")
EXE = os.path.join(KIT, "det.exe")
M = r"C:\AI\models\ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO\Q2_0"
PROMPTS = {"code": "strata-codex-dispatch-detail-code256", "thai": "strata-codex-dispatch-detail-thai256"}
FACT_PATTERNS = {
    "expert cache auto": r"expert cache auto: (.*)",
    "placement": r"placement-first: (.*?); host-owned",
    "exclusive": r"exclusive: (.*)",
    "secondary": r"strata generate: staged (.*secondary experts.*?); SECONDARY",
    "prefill sources": r"prefill sources (.*)",
    "prefill": r"strata generate: prefill (\d+ tokens in \d+ chunks?).*experts streamed (\d+).*resident (\d+)",
    "tier hits": r"^tier hits\s+(.*)",
}


def facts_from(stderr: str, stdout: str) -> dict:
    out = {}
    for key, pat in FACT_PATTERNS.items():
        for text in (stderr, stdout):
            m = re.search(pat, text, re.M)
            if m:
                out[key] = " ".join(g for g in m.groups() if g)
                break
    return out


def compare(a: dict, b: dict) -> dict:
    # a run that did not finish is not a determinism result (2026-09-30: a missing DLL gave "divergent at 0")
    for tag, r in (("A", a), ("B", b)):
        if r.get("rc", 0) != 0 or not r["tokens"]:
            return {"failed": f"{tag}: rc {r.get('rc', 0)}, {len(r['tokens'])} tokens", "first_divergent_token": None,
                    "facts": {}}
    ta, tb = a["tokens"], b["tokens"]
    first = -1
    for i in range(max(len(ta), len(tb))):
        if i >= len(ta) or i >= len(tb) or ta[i] != tb[i]:
            first = i
            break
    facts = {k: (a["facts"].get(k), b["facts"].get(k)) for k in sorted(set(a["facts"]) | set(b["facts"]))
             if a["facts"].get(k) != b["facts"].get(k)}
    return {"first_divergent_token": first, "facts": facts}


def free_gb() -> int:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)"],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def command(prompt_ids: str) -> list[str]:
    T = os.environ["TEMP"]
    return [EXE, "--pack", r"D:\Github\Strata\packs\q2_0", "--native", M + r"\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
            "--ple-gguf", M + r"\Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf", "--prefill", "2048", "--spec", "4",
            "--spec-min-p", "0.5", "--mtp", r"D:\Github\Strata\mtp\rt", "--kv", "int8", "--tokens-file", prompt_ids,
            "--max-new", "256", "--greedy", "--max-context", "8192", "--pool-workers", "13", "--adapt-swaps", "8",
            "--adapt-every", "1", "--adapt-secondary", "8", "--pcie-frac", "0", "--expert-cache", "8000",
            "--expert-profile", os.path.join(T, "strata-ranked-exl3only-profile.bin"), "--no-prefill-borrow",
            "--vram-reserve-mib", "2400", "--secondary-free-floor-mib", "640", "--secondary-expert-mib", "8704",
            "--exclusive-primary-experts", "--pool-priority", "2", "--process-priority", "2"]


CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"


def engine_env(base: dict) -> dict:
    """The engine's environment: the CUDA runtime DLLs first on PATH (a fresh boot's shell may not have them)."""
    return dict(base, CUDA_VISIBLE_DEVICES="1,0", PATH=CUDA_BIN + ";" + base.get("PATH", ""))


def capture(name: str) -> None:
    os.makedirs(KIT, exist_ok=True)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    sha = hashlib.sha256(open(EXE, "rb").read()).hexdigest()[:16]
    result = {"exe_sha256": sha, "nvidia_smi_before": smi, "runs": {}}
    env = engine_env(dict(os.environ))
    for prompt, d in PROMPTS.items():
        ids = os.path.join(os.environ["TEMP"], d, "prompt.ids")
        for rep in (1, 2):
            if free_gb() < 12:
                print(f"{prompt} {rep}: SKIPPED, less than 12 GB of RAM free")
                continue
            p = subprocess.run(command(ids), capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
            open(os.path.join(KIT, f"{name}-{prompt}-{rep}.stdout"), "w", encoding="utf-8").write(p.stdout)
            open(os.path.join(KIT, f"{name}-{prompt}-{rep}.stderr"), "w", encoding="utf-8").write(p.stderr)
            m = re.search(r"^output\s+(.*)$", p.stdout, re.M)
            tokens = [int(x) for x in re.findall(r"-?\d+", m.group(1))] if m else []
            result["runs"][f"{prompt}-{rep}"] = {"rc": p.returncode, "tokens": tokens, "facts": facts_from(p.stderr, p.stdout)}
            print(f"{prompt} {rep}: rc {p.returncode}, {len(tokens)} tokens, "
                  f"{hashlib.md5(' '.join(map(str, tokens)).encode()).hexdigest()[:8]}")
    json.dump(result, open(os.path.join(KIT, name + ".json"), "w", encoding="utf-8"), indent=1)
    print(f"exe {sha}; before the run: {smi}")


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "freeze":
        os.makedirs(KIT, exist_ok=True)
        shutil.copy2(argv[1], EXE)
        print("frozen:", hashlib.sha256(open(EXE, "rb").read()).hexdigest()[:16])
    elif len(argv) >= 2 and argv[0] == "capture":
        capture(argv[1])
    elif len(argv) >= 3 and argv[0] == "compare":
        a = json.load(open(os.path.join(KIT, argv[1] + ".json"), encoding="utf-8"))
        b = json.load(open(os.path.join(KIT, argv[2] + ".json"), encoding="utf-8"))
        print(f"exe {a['exe_sha256']} vs {b['exe_sha256']}")
        print(f"VRAM before: {a['nvidia_smi_before']!r}\n         vs: {b['nvidia_smi_before']!r}")
        for k in sorted(set(a["runs"]) & set(b["runs"])):
            d = compare(a["runs"][k], b["runs"][k])
            if d.get("failed"):
                print(f"{k}: NOT COMPARED, a run failed ({d['failed']})")
                continue
            print(f"{k}: first divergent token {d['first_divergent_token']} (-1: identical)")
            for f, (x, y) in d["facts"].items():
                print(f"    {f}: {x!r}\n    {' ' * len(f)}  {y!r}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
