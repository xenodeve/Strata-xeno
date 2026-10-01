"""#91: an agent-shaped serve session (synthetic - not a recorded Claude Code session): a short turn, a ~20K-token
document with a long answer, then short follow-up turns that reuse the prefix (their new parts run as prompt windows,
phase 0).  usage: python session91.py PORT"""
import json
import sys
import time
import urllib.request

PORT = sys.argv[1]


def ask(messages, max_tokens):
    body = json.dumps({"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                       "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            d = ((ev.get("choices") or [{}])[0].get("delta") or {})
            text.append((d.get("content") or "") + (d.get("reasoning_content") or ""))
    out = "".join(text)
    print(f"  {time.time() - t0:.0f}s chars={len(out)}", flush=True)
    return out


doc = open(r"C:\Strata-exp\src-cap\docs\BLUEPRINT.md", encoding="utf-8").read()[:72000]
print("turn 1: short", flush=True)
ask([{"role": "user", "content": "Say hello in five words."}], 64)
m = [{"role": "user", "content": "Read this document and explain, step by step, how a request flows through the "
                                 "system.\n\n" + doc}]
print("turn 2: ~20K-token document, long answer", flush=True)
a = ask(m, 800)
for q, n in (("Now list the five slowest steps.", 400),
             ("สรุปเป็นภาษาไทยสั้น ๆ ว่าขั้นไหนช้าที่สุดและทำไม", 400),
             ("Write a Python function that parses the 'prompt part' log line into a dict.", 400)):
    m = m + [{"role": "assistant", "content": a}, {"role": "user", "content": q}]
    print(f"turn: {q[:40]}", flush=True)
    a = ask(m, n)
print("SESSION DONE", flush=True)
