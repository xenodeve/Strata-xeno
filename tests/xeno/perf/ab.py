"""Same-session A/B for strata generate: ABBA per prompt, exe sha recorded, parity A vs B, per-stage table.

usage: python ab.py OUT_NAME A=exe_path B=exe_path [--prompts code,thai] [--order A,B,B,A] [--extra "args"]
       python ab.py --summarize OUT_NAME code,thai
An arm is NAME=exe or NAME=exe|args; "~--flag" drops that flag from the reference command, "@normal" runs the arm
at the normal priority class (every arm runs at HIGH class by default, AGENTS.md). Arm exes are copied to
%TEMP%/OUT_NAME/<arm>.exe first so a rebuild in place cannot change them mid-run. The prompts' reference
commands are %TEMP%/<REF dir>/on.command.json (see README.md).
"""
import ctypes, hashlib, json, os, re, shutil, statistics as st, subprocess, sys, threading, time
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.qwenputer.memory import system_snapshot


class _ProcMem(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t)]


def system_memory():
    """System physical in-use/available and commitment in GiB, from GetPerformanceInfo."""
    m = system_snapshot()
    g = 2 ** 30
    return ((m['physical_total_bytes'] - m['physical_available_bytes']) / g,
            m['physical_available_bytes'] / g, m['system_commit_bytes'] / g)


def process_memory(pid):
    """(private GiB, working set GiB) of one process, or None."""
    h = ctypes.windll.kernel32.OpenProcess(0x1000 | 0x0010, False, pid)   # QUERY_LIMITED_INFORMATION | VM_READ
    if not h:
        return None
    try:
        c = _ProcMem(); c.cb = ctypes.sizeof(c)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
            return None
        return c.PrivateUsage / 2 ** 30, c.WorkingSetSize / 2 ** 30
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def collect_memory(process, receive, errors, interval=0.5):
    """Shared observer; failures remain explicit and never become zero samples."""
    try:
        while process.poll() is None:
            used, available, committed = system_memory()
            private = process_memory(process.pid)
            if private is None:
                if process.poll() is not None:
                    return
                raise OSError("running process memory is unavailable")
            receive((used, available, committed, private[0], private[1]))
            time.sleep(interval)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")


def memory_peak_line(peak, samples, errors):
    if errors or not samples:
        return f"memory unavailable       samples={samples} errors={json.dumps(errors)}\n"
    return (f"memory peak              system in use {peak[0]:.2f} GiB  committed {peak[1]:.2f} GiB  "
            f"process private {peak[2]:.2f} GiB  working set {peak[3]:.2f} GiB\n")


def _session(value):
    if (not isinstance(value, dict) or type(value.get('version')) is not int or value['version'] != 1
            or not isinstance(value.get('arms'), dict) or not 1 <= len(value['arms']) <= 16
            or any(not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', k) or not isinstance(v, str)
                   or not re.fullmatch(r'[0-9a-f]{64}', v) for k, v in value['arms'].items())
            or not isinstance(value.get('order'), list) or not 1 <= len(value['order']) <= 1024
            or any(not isinstance(k, str) or k not in value['arms'] for k in value['order'])
            or set(value['order']) != set(value['arms'])
            or not isinstance(value.get('prompts'), list) or not 1 <= len(value['prompts']) <= 64
            or any(not isinstance(p, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', p) for p in value['prompts'])
            or len(set(value['prompts'])) != len(value['prompts'])):
        raise ValueError('invalid measurement session manifest')
    return value


def _prompt_evidence(root, session, prompt):
    failures=[]; outputs=[]; memory=[]
    expected={f'{prompt}-{i}-{arm}' for i,arm in enumerate(session['order'],1)}
    for suffix in ('.stdout','.stderr','.run.json'):
        actual={p.name[:-len(suffix)] for p in root.glob('*'+suffix)
                if re.fullmatch(re.escape(prompt)+r'-[1-9][0-9]*-.+'+re.escape(suffix),p.name)}
        if actual != expected: failures.append('missing or extra scheduled '+suffix)
    for i,arm in enumerate(session['order'],1):
        tag=f'{prompt}-{i}-{arm}'; meta=root/(tag+'.run.json')
        if meta.stat().st_size>4096: raise ValueError('run receipt size')
        record=json.loads(meta.read_text(encoding='utf-8'))
        if (type(record.get('version')) is not int or record['version']!=1 or record.get('prompt')!=prompt
                or type(record.get('ordinal')) is not int or record['ordinal']!=i or record.get('arm')!=arm
                or record.get('exe_sha256')!=session['arms'][arm]
                or type(record.get('returncode')) is not int or record['returncode']!=0):
            failures.append('failed or mismatched run receipt')
        for suffix in ('stdout','stderr'):
            with (root/(tag+'.'+suffix)).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=record.get(suffix+'_sha256'):
                    failures.append('raw log identity mismatch')
        output,_=parse_run((root/(tag+'.stdout')).read_text(encoding='utf-8'))
        if output is None or not output.split() or any(not t.isdigit() for t in output.split()):
            failures.append('missing or invalid greedy token output')
        outputs.append(output); memory.append(record.get('memory_complete') is True)
    return failures, outputs, memory


def session_evidence(out, prompt):
    root = Path(out); manifest = root / 'session.json'
    if not manifest.is_file():
        return dict(complete=False, parity_qualified=False, memory_complete=False,
                    reason='unverified legacy completeness: no session manifest')
    failures=[]; memory=[]; outputs=[]
    try:
        if manifest.stat().st_size > 1024*1024: raise ValueError('manifest size')
        session = _session(json.loads(manifest.read_text(encoding='utf-8')))
        if prompt not in session['prompts']: raise ValueError('prompt not in session')
        for arm, digest in session['arms'].items():
            with (root/(arm+'.exe')).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=digest: failures.append('snapshot identity mismatch')
        for scheduled_prompt in session['prompts']:
            errors, raw, observed_memory = _prompt_evidence(root, session, scheduled_prompt)
            failures.extend(errors); memory.extend(observed_memory)
            if scheduled_prompt == prompt: outputs = raw
        complete=not failures
        return dict(complete=complete,parity_qualified=complete and len(session['arms'])>=2 and len(set(outputs))==1,
                    memory_complete=complete and all(memory),reason='; '.join(sorted(set(failures))) or 'complete session')
    except (OSError,ValueError,TypeError,AttributeError) as error:
        return dict(complete=False,parity_qualified=False,memory_complete=False,
                    reason='unverified session: '+str(error))

T = os.environ.get("TEMP", os.environ.get("TMP", "."))
REF = {"code": "strata-codex-dispatch-detail-code256", "thai": "strata-codex-dispatch-detail-thai256",
       "sky": "strata-codex-compact-parity-sky256", "long": "strata-codex-compact-parity-long256",
       "serve": "strata-claude-serve-long"}   # strata-xeno.json's serving args, an 8,024-token prompt, 128 new
F = r"([0-9.]+)"
STAGES = {"tok/s": r"->\s+" + F + " tok/s", "rings": r"wait for rings " + F, "pool": r"verify window.*pool " + F,
          "cpu_pool": r"dispatch detail\s+CPU pool " + F, "sec_fin": r"secondary finish " + F,
          "sec_launch": r"secondary timing\s+launch " + F, "cpu_GBs": r"ms/round; " + F + " GB/s",
          "mtp": r"^mtp\s+" + F, "host": r" host " + F, "commit": r"commit " + F,
          "ttft_ms": r"time to first token " + F,
          "sys_used_gib": r"^memory peak\s+system in use " + F, "sys_commit_gib": r"^memory peak.*committed " + F,
          "proc_private_gib": r"^memory peak.*process private " + F, "proc_ws_gib": r"^memory peak.*working set " + F}


def main(argv):
    out_name, arms, prompts, order, extra = argv[0], {}, ["code", "thai"], ["A", "B", "B", "A"], ["--pool-priority", "2"]   # AGENTS.md: every arm at high priority
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--prompts": prompts = argv[i + 1].split(","); i += 2; continue
        if a == "--order": order = argv[i + 1].split(","); i += 2; continue
        if a == "--extra": extra = argv[i + 1].split(); i += 2; continue
        k, v = a.split("=", 1); arms[k] = v; i += 1   # NAME=exe or NAME=exe|extra args for this arm
    _session(dict(version=1, arms={k:'0'*64 for k in arms}, prompts=prompts, order=order))
    out = os.path.join(T, out_name)
    if os.path.isdir(out) and os.listdir(out): raise ValueError('measurement requires a fresh output directory; old evidence is preserved')
    os.makedirs(out, exist_ok=True)
    exe, sha, sha_full, arm_args = {}, {}, {}, {}
    for k, v in arms.items():
        path, _, args = v.partition("|")
        arm_args[k] = args.split()
        exe[k] = os.path.join(out, f"{k}.exe"); shutil.copy2(path, exe[k])
        with open(exe[k], "rb") as stream: sha_full[k] = hashlib.file_digest(stream, 'sha256').hexdigest()
        sha[k] = sha_full[k][:12]
    session = _session(dict(version=1, arms=sha_full, prompts=prompts, order=order))
    with open(os.path.join(out, 'session.json'), 'x', encoding='utf-8') as manifest:
        json.dump(session, manifest, indent=2)
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
            flags = (0 if "@normal" in arm_args[arm] else 0x80) | 0x08000000   # + CREATE_NO_WINDOW: no console pops up on the display
            pr = subprocess.Popen([exe[arm]] + [c for c in cmd if '~' + c not in arm_args[arm]] + [x for x in arm_args[arm] if not x.startswith(('~', '@'))], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", creationflags=flags)
            peak = [0.0, 0.0, 0.0, 0.0]   # system in use, system committed, process private, process working set
            sample_count = [0]; sample_errors = []
            def receive_memory(row):
                sample_count[0] += 1
                for i, v in enumerate((row[0], row[2], row[3], row[4])): peak[i] = max(peak[i], v)
            th = threading.Thread(target=collect_memory, args=(pr, receive_memory, sample_errors), daemon=True); th.start()
            stdout, stderr = pr.communicate(); th.join()
            stdout += memory_peak_line(peak, sample_count[0], sample_errors)
            p = subprocess.CompletedProcess(pr.args, pr.returncode, stdout, stderr)
            log_hashes={}
            for suffix, text in (('stdout',p.stdout),('stderr',p.stderr)):
                path=os.path.join(out,tag+'.'+suffix)
                with open(path,'w',encoding='utf-8') as stream: stream.write(text)
                with open(path,'rb') as stream: log_hashes[suffix+'_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
            with open(os.path.join(out, tag + '.run.json'), 'x', encoding='utf-8') as receipt:
                json.dump(dict(version=1,prompt=prompt,ordinal=n,arm=arm,exe_sha256=sha_full[arm],
                               returncode=p.returncode,memory_complete=not sample_errors and bool(sample_count[0]),**log_hashes),receipt)
            tps = re.search(STAGES["tok/s"], p.stdout)
            line = f"{tag} rc={p.returncode} sha={sha[arm]} tok/s {tps.group(1) if tps else 'NA'} wall {time.time()-t0:.0f}s memory={'unavailable' if sample_errors or not sample_count[0] else 'complete'}"
            print(line, flush=True); log.write(line + "\n"); log.flush()
    log.write(summarize(out, prompts) + "\nDONE\n"); log.flush(); log.close()
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
            match=re.fullmatch(re.escape(prompt)+r'-[1-9][0-9]*-(.+)\.stdout',f)
            if match: runs.setdefault(match.group(1), []).append((output, vals))
        outputs = {o for v in runs.values() for o, _ in v}
        evidence=session_evidence(out,prompt)
        lines.append(f"{prompt}: {sum(len(v) for v in runs.values())} runs, outputs identical across arms: {evidence['parity_qualified']}; "
                     f"session complete: {evidence['complete']}; memory complete: {evidence['memory_complete']}; {evidence['reason']}")
        lines.append(f"  recorded outputs identical: {bool(outputs) and len(outputs)==1} (does not establish completeness)")
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
