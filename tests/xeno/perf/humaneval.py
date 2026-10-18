"""Task-quality gate for decode-path options: HumanEval / HumanEval+ pass@1 through the server (#255).

Teacher-forced NLL / top-1 (``qual_logpos``) can point the wrong way for a change that removes or delays experts:
on 2026-10-18 dropping Gyro-M's cold experts gained 10 points of teacher-forced code accuracy and lost 57 of 164
HumanEval problems.  Gate such changes on a task instead.

    python tests/xeno/perf/humaneval.py run <tag> <config.json> <ENV=V,ENV=V|-> [--n N] [--port P]
    python tests/xeno/perf/humaneval.py plus <tag> [<tag> ...]     # rescore run's completions on HumanEval+
    python tests/xeno/perf/humaneval.py cmp <ref-tag> <tag> [...]   # pass@1 and gained / lost per tag

``run`` starts the config's server in the background (no console window), asks every problem once (temperature 0,
``enable_thinking`` false, 512 tokens), and runs prompt + completion + the problem's tests with ``python -I`` in a
fresh empty directory (15 s each).  ``plus`` compares each completion with the canonical solution on EvalPlus's
base + plus inputs (floats: relative 1e-7, absolute ``atol``; 120 s each).  Results go to ``he-<tag>.json`` /
``he-<tag>.plus.json`` in ``--out`` (default: the current directory).

Data (downloaded once, each into its own directory):
  HumanEval      https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz
  HumanEval+     https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz
A 164-problem run of Gyro-M takes ~20 min an arm; base 155 / 164 (HumanEval), 150 / 164 (HumanEval+) on 2026-10-18,
and a rounding-only perturbation moves each by one problem, so read differences of 1-2 problems as noise.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

NO_WINDOW = 0x08000000 if os.name == "nt" else 0
ASK = ("Complete the following Python function. Reply with the complete function, including its signature, in one "
       "```python code block, and nothing else.\n\n```python\n{prompt}```")

PLUS_HARNESS = r'''
import copy, json, math, sys
sys.setrecursionlimit(100000)
CAND = json.load(open("cand.json", encoding="utf-8"))
REF = json.load(open("ref.json", encoding="utf-8"))
ENTRY, ATOL = CAND["entry"], float(REF["atol"] or 0)
nc, nr = {}, {}
exec(CAND["src"], nc)
exec(REF["src"], nr)
cand, ref = nc[ENTRY], nr[ENTRY]

def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except Exception:
            return False
        return math.isclose(fa, fb, rel_tol=1e-7, abs_tol=ATOL) or (math.isnan(fa) and math.isnan(fb))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b

for inp in REF["inputs"]:
    try:
        want = ref(*copy.deepcopy(inp))
    except Exception:
        continue
    got = cand(*copy.deepcopy(inp))
    if not same(got, want):
        print("MISMATCH", repr(inp)[:200], repr(got)[:100], repr(want)[:100])
        sys.exit(1)
print("OK")
'''


def load_problems(path):
    return [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]


def extract(text):
    """The first fenced code block of a reply (```python, ```py or bare), else the whole reply."""
    m = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    return m[0] if m else text


def _isolated(files, script, timeout, runs, python):
    """Write `files` and run `script` with `python -I` in a fresh directory under `runs`: (ok, tail of output)."""
    os.makedirs(runs, exist_ok=True)
    d = tempfile.mkdtemp(dir=runs)
    try:
        for name, body in files.items():
            open(os.path.join(d, name), "w", encoding="utf-8").write(body)
        try:
            r = subprocess.run([python, "-I", script], cwd=d, capture_output=True, text=True, timeout=timeout,
                               creationflags=NO_WINDOW)
            return r.returncode == 0, (r.stdout + r.stderr)[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def check_base(prob, code, runs, python=sys.executable):
    prog = prob["prompt"] + "\n\n" + code + "\n\n" + prob["test"] + f"\ncheck({prob['entry_point']})\n"
    return _isolated({"t.py": prog}, "t.py", 15, runs, python)


def check_plus(prob, code, runs, python=sys.executable):
    files = {
        "cand.json": json.dumps({"entry": prob["entry_point"], "src": prob["prompt"] + "\n\n" + code}),
        "ref.json": json.dumps({"atol": prob.get("atol", 0), "src": prob["prompt"] + prob["canonical_solution"],
                                "inputs": prob["base_input"] + prob["plus_input"]}),
        "h.py": PLUS_HARNESS,
    }
    return _isolated(files, "h.py", 120, runs, python)


def run(a):
    c = json.load(open(a.config, encoding="utf-8"))
    c.pop("_comment", None)
    c["port"] = a.port
    c["log"] = os.path.join(a.out, f"he-{a.tag}.log")
    if a.env and a.env != "-":
        c["env"] = dict(c.get("env", {}), **dict(kv.split("=", 1) for kv in a.env.split(",")))
    cfg = os.path.join(a.out, f"he-{a.tag}.cfg.json")
    json.dump(c, open(cfg, "w", encoding="utf-8"), indent=1)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.cuda_devices)
    env.pop("CUDA_DEVICE_ORDER", None)
    proc = subprocess.Popen([a.python, "-u", a.server, "--engine", "strata", "--config", cfg, "--port", str(a.port)],
                            env=env, cwd=os.path.dirname(os.path.dirname(a.server)),
                            stdout=open(os.path.join(a.out, f"he-{a.tag}.server.txt"), "w", encoding="utf-8"),
                            stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
    out = []
    try:
        t0 = time.time()
        while time.time() - t0 < 900:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited with {proc.returncode}")
            try:
                if json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=3)).get("status") == "ok":
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(3)
        probs = load_problems(a.data)
        for k, prob in enumerate(probs[:a.n] if a.n else probs):
            body = json.dumps({"model": "x", "messages": [{"role": "user", "content": ASK.format(prompt=prob["prompt"])}],
                               "max_tokens": 512, "temperature": 0, "stream": False,
                               "chat_template_kwargs": {"enable_thinking": False}}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            t1 = time.time()
            r = json.load(urllib.request.urlopen(req, timeout=1800))
            text = r["choices"][0]["message"].get("content") or ""
            ok, err = check_base(prob, extract(text), a.runs)
            out.append({"task_id": prob["task_id"], "pass": ok, "err": err, "s": round(time.time() - t1, 2),
                        "tokens": (r.get("usage") or {}).get("completion_tokens"), "text": text})
            if k % 20 == 0:
                print(f"{a.tag}: {k + 1} done, {sum(o['pass'] for o in out)} passed, {time.time() - t0:.0f} s", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
        json.dump(out, open(os.path.join(a.out, f"he-{a.tag}.json"), "w", encoding="utf-8"), indent=1)
    s = sum(o["s"] for o in out) or 1e-9
    print(f"{a.tag}: pass@1 {sum(o['pass'] for o in out)}/{len(out)}, "
          f"{sum(o['tokens'] or 0 for o in out) / s:.1f} completion tok/s", flush=True)


def plus(a):
    probs = {p["task_id"]: p for p in load_problems(a.data_plus)}
    for tag in a.tags:
        rows = json.load(open(os.path.join(a.out, f"he-{tag}.json"), encoding="utf-8"))
        res = []
        for o in rows:
            ok, why = check_plus(probs[o["task_id"]], extract(o["text"]), a.runs)
            res.append({"task_id": o["task_id"], "base_pass": o["pass"], "plus_pass": ok, "why": why})
        json.dump(res, open(os.path.join(a.out, f"he-{tag}.plus.json"), "w", encoding="utf-8"), indent=1)
        print(f"{tag}: HumanEval+ pass@1 {sum(x['plus_pass'] for x in res)}/{len(res)}, "
              f"timeouts {sum(1 for x in res if x['why'] == 'timeout')}", flush=True)


def compare(results):
    """{tag: {task_id: passed}} with the reference first -> one line per tag."""
    tags = list(results)
    ref = results[tags[0]]
    lines = []
    for t in tags:
        r = results[t]
        line = f"{t}: pass@1 {sum(r.values())}/{len(r)} ({100.0 * sum(r.values()) / max(len(r), 1):.1f} %)"
        if t != tags[0]:
            both = [k for k in r if k in ref]
            gain = sum(1 for k in both if r[k] and not ref[k])
            loss = sum(1 for k in both if ref[k] and not r[k])
            line += f"; vs {tags[0]}: +{gain} -{loss} (net {gain - loss:+d})"
        lines.append(line)
    return lines


def cmp(a):
    for suffix, key in ((".json", "pass"), (".plus.json", "plus_pass")):
        paths = {t: os.path.join(a.out, f"he-{t}{suffix}") for t in a.tags}
        if all(os.path.exists(p) for p in paths.values()):
            print("HumanEval+" if suffix == ".plus.json" else "HumanEval")
            results = {t: {o["task_id"]: o[key] for o in json.load(open(p, encoding="utf-8"))} for t, p in paths.items()}
            for line in compare(results):
                print("  " + line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".", help="where he-<tag>*.json are written and read")
    ap.add_argument("--runs", default=os.path.join(tempfile.gettempdir(), "strata-he-runs"),
                    help="the parent of the per-problem empty directories")
    ap.add_argument("--data", default=r"C:\Strata-exp\data-humaneval\HumanEval.jsonl.gz")
    ap.add_argument("--data-plus", default=r"C:\Strata-exp\data-humanevalplus\HumanEvalPlus.jsonl.gz")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("tag")
    r.add_argument("config")
    r.add_argument("env", help="ENV=V[,ENV=V] for this arm, or -")
    r.add_argument("--n", type=int, default=0, help="the first N problems only (0: all 164)")
    r.add_argument("--port", type=int, default=8097)
    r.add_argument("--server", default=r"C:\Strata-exp\src-main\serve\server.py")
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--cuda-devices", default="1,0")
    p = sub.add_parser("plus")
    p.add_argument("tags", nargs="+")
    c = sub.add_parser("cmp")
    c.add_argument("tags", nargs="+")
    a = ap.parse_args(argv)
    a.out = os.path.abspath(a.out)   # the server runs in its own directory and reads the config from here
    {"run": run, "plus": plus, "cmp": cmp}[a.cmd](a)


if __name__ == "__main__":
    main()
