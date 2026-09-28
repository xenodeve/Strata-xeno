"""Same-session A/B for strata generate: ABBA per prompt, exe sha recorded, parity A vs B, per-stage table.

usage: python ab.py OUT_NAME A=exe_path B=exe_path [--prompts code,thai] [--order A,B,B,A] [--extra "args"]
       python ab.py --summarize OUT_NAME code,thai
An arm is NAME=exe or NAME=exe|args; "~--flag" drops that flag from the reference command, "@normal" runs the arm
at the normal priority class (every arm runs at HIGH class by default, AGENTS.md). Arm exes are copied to
%TEMP%/OUT_NAME/<arm>.exe first so a rebuild in place cannot change them mid-run. The prompts' reference
commands are %TEMP%/<REF dir>/on.command.json (see README.md).
"""
import hashlib, json, os, re, shutil, statistics as st, subprocess, sys, time

T = os.environ.get("TEMP", os.environ.get("TMP", "."))
REF = {"code": "strata-codex-dispatch-detail-code256", "thai": "strata-codex-dispatch-detail-thai256",
       "sky": "strata-codex-compact-parity-sky256", "long": "strata-codex-compact-parity-long256"}
F = r"([0-9.]+)"
STAGES = {"tok/s": r"->\s+" + F + " tok/s", "rings": r"wait for rings " + F, "pool": r"verify window.*pool " + F,
          "cpu_pool": r"dispatch detail\s+CPU pool " + F, "sec_fin": r"secondary finish " + F,
          "sec_launch": r"secondary timing\s+launch " + F, "cpu_GBs": r"ms/round; " + F + " GB/s",
          "mtp": r"^mtp\s+" + F, "host": r" host " + F, "commit": r"commit " + F}


def main(argv):
    out_name, arms, prompts, order, extra = argv[0], {}, ["code", "thai"], ["A", "B", "B", "A"], ["--pool-priority", "2"]   # AGENTS.md: every arm at high priority
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--prompts": prompts = argv[i + 1].split(","); i += 2; continue
        if a == "--order": order = argv[i + 1].split(","); i += 2; continue
        if a == "--extra": extra = argv[i + 1].split(); i += 2; continue
        k, v = a.split("=", 1); arms[k] = v; i += 1   # NAME=exe or NAME=exe|extra args for this arm
    out = os.path.join(T, out_name); os.makedirs(out, exist_ok=True)
    exe, sha, arm_args = {}, {}, {}
    for k, v in arms.items():
        path, _, args = v.partition("|")
        arm_args[k] = args.split()
        exe[k] = os.path.join(out, f"{k}.exe"); shutil.copy2(path, exe[k])
        sha[k] = hashlib.sha256(open(exe[k], "rb").read()).hexdigest()[:12]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1,0")
    cuda = os.path.join("C:" + os.sep, "Program Files", "NVIDIA GPU Computing Toolkit", "CUDA", "v13.3", "bin")
    env["PATH"] = os.pathsep.join([cuda, os.path.join(cuda, "x64"), env["PATH"]])
    log = open(os.path.join(out, "summary.log"), "a", encoding="utf-8")
    log.write(f"arms {json.dumps({k: [arms[k], sha[k]] for k in arms})} extra {extra}\n"); log.flush()
    for prompt in prompts:
        cmd = json.load(open(os.path.join(T, REF[prompt], "on.command.json")))[1:] + extra
        for n, arm in enumerate(order, 1):
            tag = f"{prompt}-{n}-{arm}"
            t0 = time.time()
            # every arm runs in HIGH_PRIORITY_CLASS (0x80): strata-claude-hiclass showed it cuts desktop interference
            # (code +2.6 %, thai +1.3 %, tighter spread); '@normal' in an arm's args opts that arm out
            flags = 0 if "@normal" in arm_args[arm] else 0x80
            p = subprocess.run([exe[arm]] + [c for c in cmd if '~' + c not in arm_args[arm]] + [x for x in arm_args[arm] if not x.startswith(('~', '@'))], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=flags)
            open(os.path.join(out, tag + ".stdout"), "w", encoding="utf-8").write(p.stdout)
            open(os.path.join(out, tag + ".stderr"), "w", encoding="utf-8").write(p.stderr)
            tps = re.search(STAGES["tok/s"], p.stdout)
            line = f"{tag} rc={p.returncode} sha={sha[arm]} tok/s {tps.group(1) if tps else 'NA'} wall {time.time()-t0:.0f}s"
            print(line, flush=True); log.write(line + "\n"); log.flush()
    log.write(summarize(out, prompts) + "\nDONE\n"); log.flush()
    print("DONE", flush=True)


def parse_run(text):
    """(greedy output ids, {stage: value}) of one run's stdout; (None, {}) when the run printed no output."""
    m = re.search(r"^output\s*:\s*(.*)$", text, re.M)
    if not m:
        return None, {}
    return m.group(1), {k: float(x.group(1)) for k, v in STAGES.items() if (x := re.search(v, text, re.M))}


def summarize(out, prompts):
    lines = []
    for prompt in prompts:
        runs = {}
        for f in sorted(os.listdir(out)):
            if not (f.startswith(prompt + "-") and f.endswith(".stdout")): continue
            output, vals = parse_run(open(os.path.join(out, f), encoding="utf-8").read())
            if output is None: continue
            runs.setdefault(f[:-7].split("-")[-1], []).append((output, vals))
        outputs = {o for v in runs.values() for o, _ in v}
        lines.append(f"{prompt}: {sum(len(v) for v in runs.values())} runs, outputs identical across arms: {len(outputs) == 1}")
        for arm in sorted(runs):
            vals = [v for _, v in runs[arm]]
            keys = [k for k in STAGES if all(k in v for v in vals)]
            lines.append(f"  {arm}: " + "  ".join(f"{k} {st.mean(v[k] for v in vals):.2f}" for k in keys)
                         + f"  runs {[v.get('tok/s') for v in vals]}")
    return "\n".join(lines)


if __name__ == "__main__":
    if sys.argv[1] == "--summarize":
        print(summarize(os.path.join(T, sys.argv[2]), sys.argv[3].split(",")))
    else:
        main(sys.argv[1:])
