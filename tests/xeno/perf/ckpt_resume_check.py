"""Does a request that resumes from a periodic checkpoint start from the exact state at its position? (#260)

Server A reads request 1 (a long text, so the prompt read saves periodic checkpoints every ``--prompt-cache-every``,
16384 by default), then request 2 (the same start with a different ending: the server restores the 16384 checkpoint
and reads on).  A fresh server B reads request 2 from 0.  Both run with ``STRATA_STATE_HASH`` and
``STRATA_STATE_HASH_GDN``, so the log holds the session state's hash after each request.

    python tests/xeno/perf/ckpt_resume_check.py <tag> <config.json> [ENV=V,ENV=V|-] [--max-tokens N]

Run it once per arm (for example ``STRATA_CKPT_CUTS=0`` and ``=1``).  How to read two arms:
  - request 1's hashes must be equal across the arms: a change of where the parts end may not change a full read;
  - request 2 on server A then differs between the arms only through the checkpoint's content, so a different hash
    there means one arm's checkpoint was not the state at its position (#260: a wave read saved it while the other
    lane had already moved the GDN state past it);
  - A's request 2 is not expected to equal B's: "restore + read on" is not byte-identical to "read from 0", even
    without the wave.
With ``--max-tokens`` above 1 the answers are kept too (greedy), to see whether the state difference changes the text.
Results go to ``ckr-<tag>.json`` in ``--out``; the server logs to ``ckr-<tag>-{A,B}.log``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

NO_WINDOW = 0x08000000 if os.name == "nt" else 0
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HASH = re.compile(r"STATE_HASH L=(\d+) gdn=([0-9a-f]+) .*?kv=([0-9a-f]+)")
PROMPT = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms")
PART = re.compile(r"prompt part [^:]*: (\d+) tokens \[(\d+), (\d+)\)")


def texts(src: str) -> tuple[str, str]:
    """Request 1 and request 2: the same first ~78K characters (more than 16384 tokens of C++), different endings."""
    t = open(src, encoding="utf-8").read()
    if len(t) < 203000:
        raise SystemExit(f"{src}: needs at least 203,000 characters, has {len(t)}")
    return t[:110000], t[:78000] + "\n// --- a different ending for the second request ---\n" + t[200000:203000]


def summarize(lines) -> list[dict]:
    """One entry per request in a server log: its state hash, the prompt line and the parts it was read in."""
    out, parts, last = [], [], None
    for line in lines:
        m = PART.search(line)
        if m:
            parts.append([int(m.group(2)), int(m.group(3))])
            continue
        m = HASH.search(line)
        if m:
            last = {"L": int(m.group(1)), "gdn": m.group(2), "kv": m.group(3)}
            continue
        m = PROMPT.search(line)
        if m:
            out.append({"tokens": int(m.group(1)), "reused": int(m.group(2)), "read": int(m.group(3)),
                        "ms": int(m.group(4)), "state": last, "parts": parts})
            parts, last = [], None
    return out


def serve(a, side: str):
    c = json.load(open(a.config, encoding="utf-8"))
    c.pop("_comment", None)
    c["port"] = a.port
    c["log"] = os.path.join(a.out, f"ckr-{a.tag}-{side}.log")
    if os.path.exists(c["log"]):
        os.remove(c["log"])
    env_add = {"STRATA_STATE_HASH": "1", "STRATA_STATE_HASH_GDN": "1"}
    if a.env and a.env != "-":
        env_add.update(dict(kv.split("=", 1) for kv in a.env.split(",")))
    c["env"] = dict(c.get("env", {}), **env_add)
    cfg = os.path.join(a.out, f"ckr-{a.tag}-{side}.cfg.json")
    json.dump(c, open(cfg, "w", encoding="utf-8"), indent=1)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "1,0"))   # 5060 Ti first
    env.pop("CUDA_DEVICE_ORDER", None)
    p = subprocess.Popen([a.python, "-u", a.server, "--engine", "strata", "--config", cfg, "--port", str(a.port)],
                         cwd=os.path.dirname(os.path.dirname(a.server)), env=env,
                         stdout=open(os.path.join(a.out, f"ckr-{a.tag}-{side}.server.txt"), "w", encoding="utf-8"),
                         stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise RuntimeError(f"server exited with {p.returncode}")
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=3)).get("status") == "ok":
                return p
        except Exception:  # noqa: BLE001 - not up yet
            pass
        time.sleep(3)
    p.kill()
    raise RuntimeError("the server did not report healthy in 900 s")


def ask(a, text: str) -> dict:
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": a.max_tokens,
                       "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return {"usage": r.get("usage"), "text": r["choices"][0]["message"].get("content")}


def stop(p) -> None:
    p.terminate()
    try:
        p.wait(timeout=120)
    except subprocess.TimeoutExpired:
        p.kill()
    time.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tag")
    ap.add_argument("config")
    ap.add_argument("env", nargs="?", default="-")
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--text", default=os.path.join(ROOT, "src", "core", "expert_source.cpp"))
    ap.add_argument("--server", default=os.path.join(ROOT, "serve", "server.py"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)   # the server runs in its own directory and reads the config from here
    r1, r2 = texts(a.text)
    res = {"config": a.config, "env": a.env, "max_tokens": a.max_tokens}
    for side, reqs in (("A", (("req1", r1), ("req2", r2))), ("B", (("req2", r2),))):
        p = serve(a, side)
        try:
            for name, text in reqs:
                res[f"{side}_{name}"] = ask(a, text)
        finally:
            stop(p)
        res[f"{side}_log"] = summarize(open(os.path.join(a.out, f"ckr-{a.tag}-{side}.log"), encoding="utf-8",
                                            errors="replace"))
    json.dump(res, open(os.path.join(a.out, f"ckr-{a.tag}.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for side in ("A", "B"):
        for i, e in enumerate(res[f"{side}_log"]):
            s = e["state"] or {}
            print(f"{side} request {i + 1}: {e['tokens']} tokens = {e['reused']} reused + {e['read']} read in "
                  f"{e['ms']} ms, parts {e['parts']}, gdn={s.get('gdn')} kv={s.get('kv')}")
    print("DONE")


if __name__ == "__main__":
    main()
