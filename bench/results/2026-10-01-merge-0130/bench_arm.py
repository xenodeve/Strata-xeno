"""One boot of the serve-mode A/B (#54): fixed requests against the running test server, then the engine's own
per-request figures from its log. usage: python bench_arm.py ARM_NAME OUT.jsonl"""
import json
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

S = Path(__file__).parent
ENGINE_LOG = S / "engine-s7.log"
URL = "http://127.0.0.1:8091/v1/messages"
arm, out = sys.argv[1], Path(sys.argv[2])

CODE = ("Write a complete Python module that implements an LRU cache with a time-to-live per entry, thread-safe, "
        "with get, put, delete and a stats() method, plus unit tests using unittest. Code only.")
THAI = ("เขียนบทความภาษาไทยเรื่องการดูแลสุขภาพในวัยทำงาน ครอบคลุมการนอน อาหาร การออกกำลังกาย และความเครียด "
        "ยาวอย่างน้อยห้าย่อหน้า")


def ask(text, max_tokens, system="You are a helpful assistant."):
    import os
    samp = os.environ.get("BENCH_SAMPLED") == "1"   # #56: Claude Code's main turns sample
    body = {"model": "m", "max_tokens": max_tokens, "stream": False, "temperature": 1.0 if samp else 0,
            "thinking": {"type": "disabled"}, "system": system, "messages": [{"role": "user", "content": text}]}
    if samp:
        body.update(top_p=0.95, top_k=20, seed=7)
    t = time.monotonic()
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(),
                                                       {"Content-Type": "application/json"}), timeout=900) as r:
        o = json.load(r)
    return time.monotonic() - t, o


def engine_lines():
    return ENGINE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()


PROMPT = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
                    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)")
METRICS = re.compile(r"decode entries (\d+) = primary (\d+) \+ secondary (\d+) \+ pcie (\d+) \+ cpu (\d+); "
                     r"cpu experts ([\d.]+) ms")
HIT = re.compile(r"decode expert cache hit rate: ([\d.]+)%")


def figures(lines):
    rec = {}
    for l in lines:
        m = PROMPT.search(l)
        if m:
            rec.update(prompt=int(m[1]), reused=int(m[2]), read=int(m[3]), read_ms=int(m[4]), read_tps=float(m[5]),
                       gen=int(m[6]), gen_ms=int(m[7]), gen_tps=float(m[8]), acc=int(m[9]), off=int(m[10]))
        m = METRICS.search(l)
        if m:
            e, p, s, _, c = (int(m[i]) for i in range(1, 6))
            rec.update(entries=e, primary=p, secondary=s, cpu=c, cpu_share=round(c / e, 4) if e else None,
                       cpu_ms=float(m[6]))
        m = HIT.search(l)
        if m:
            rec["hit"] = float(m[1])
    return rec


sizing = [l for l in engine_lines() if "expert cache sizing" in l or "expert cache " in l and "slots" in l][:2]
ask("Say ok.", 8)                                           # warm-up (not recorded)
plan = [("code", CODE, 256), ("code", CODE, 256), ("thai", THAI, 256), ("thai", THAI, 256)]
nonce = uuid.uuid4().hex
long_text = nonce + "\n" + "\n".join(f"Entry {i:05d}: action {(i * 7919) % 99991:05d} ran." for i in range(620)) + \
    "\nWhich action did Entry 00321 run? Number only."
plan.append(("read8k", long_text, 8))
with out.open("a", encoding="utf-8") as f:
    for name, text, mt in plan:
        before = len(engine_lines())
        wall, o = ask(text, mt)
        time.sleep(0.5)
        rec = {"arm": arm, "task": name, "wall_s": round(wall, 2), **figures(engine_lines()[before:]),
               "text_sha": __import__("hashlib").sha256(json.dumps(o["content"]).encode()).hexdigest()[:8]}
        f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
print("sizing:", sizing)
