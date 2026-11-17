"""Replay a request stream through the server's cache-slot rules, without an engine (#321).

    python tools/cacheslots_replay.py TRACE.jsonl[.gz] [--slots 4] [--rows]
    python tools/cacheslots_replay.py --bench w1,w2a,w2s [--workload-root DIR] [--rows]
    python tools/cacheslots_replay.py --sequences [--rows]

TRACE is what the server wrote with STRATA_SLOT_TRACE=<file> (serve/server.py slot_trace: one JSON line per request -
the route, who sent it, the prompt's token ids, a client's own strata_cache_slot).  --bench builds the stream from
tools/workstation_bench.py's own request order instead (user A / B and their side requests; the prompt rendered by the
server's template, its bytes cut into 4-byte pieces as a stand-in for tokens).

--sequences replays the review's Claude-Code-shaped single-user sequences (tools/cacheslots_sequences.py).  A record's
"role" (main, subagent, classifier, side), when present, splits the tokens read.

Every rule of Service.with_slot is replayed on the same stream with the engine's slots (--slots, default 4):
family (CacheSlots, the default), identity (IdentityCacheSlots), continuation (ContinuationCacheSlots), hybrid
(HybridCacheSlots).  A request whose
client chose its slot goes there, as the server sends it.  The engine side is modelled as each slot holding the last
prompt sent to it; "reused" is the prefix a prompt shares with its slot's last prompt.  That is an upper bound: the
engine resumes from its last checkpoint at or before that point (turn starts, the root, ckpt_at).

Per rule: the tokens read (the prompt minus what it reused), the slot switches (a request on another slot than the one
before it: the engine parks one conversation and restores another), the evictions (a prompt landing on a slot whose
last prompt it does not continue), and the regrets: evicted prompts that a later request continued, with their tokens
(a conversation that had to be read again).  "Continues" is ContinuationCacheSlots' test (shares at least PREFIX
tokens and all of the last prompt but its last TAIL), so it is the same for every rule.

Rule-independent, per sender (key name | device): the conversations (requests linked by that same continuation test)
and how many of them are live at each request - started at or before it and continued after it - max and median.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.server import (CacheSlots, ContinuationCacheSlots, HybridCacheSlots, IdentityCacheSlots,  # noqa: E402
                          _shared_len)

RULES = {"family": CacheSlots, "identity": IdentityCacheSlots, "continuation": ContinuationCacheSlots,
         "hybrid": HybridCacheSlots}
PREFIX, TAIL = ContinuationCacheSlots.PREFIX, ContinuationCacheSlots.TAIL


def continues(last, ids) -> int:
    """The shared prefix when `ids` continues `last` (ContinuationCacheSlots' test), else 0."""
    sh = _shared_len(last, ids)
    return sh if sh >= PREFIX and sh >= len(last) - TAIL else 0


def sender(rec) -> tuple | None:
    """The identity rule's key, as Service.with_slot makes it."""
    who = rec.get("who")
    return (who.get("key"), who.get("device")) if isinstance(who, dict) else None


def label(key) -> str:
    return "server" if key is None else f"{key[0]}|{key[1]}"


def read_trace(path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    out = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rec["ids"] = list(rec.get("ids") or [])
                out.append(rec)
    return out


def replay(records: list[dict], rule, slots: int = 4) -> dict:
    """`rule`: a name of RULES, or a class with the CacheSlots interface (a variant under test)."""
    cls = RULES[rule] if isinstance(rule, str) else rule
    table = cls(slots)
    rule = rule if isinstance(rule, str) else getattr(cls, "__name__", str(cls))
    content: list = [None] * slots
    evicted: dict = {}                                  # hash of the first PREFIX tokens -> [evicted prompts]
    res = {"rule": rule, "tokens": 0, "read": 0, "reused": 0, "switches": 0, "evictions": 0, "regrets": 0,
           "regret_tokens": 0, "rows": [], "by_role": {}}
    prev_slot = None
    for i, rec in enumerate(records):
        ids, who = rec["ids"], sender(rec)
        head = hash(tuple(ids[:PREFIX]))
        for c in list(evicted.get(head, ())):           # a conversation that was evicted, continued now
            if continues(c, ids):
                res["regrets"] += 1
                res["regret_tokens"] += len(c)
                evicted[head].remove(c)
        pinned = rec.get("pinned")
        if isinstance(pinned, int) and not isinstance(pinned, bool) and 0 <= pinned < slots:
            slot = pinned
            table.record(slot, ids)
        else:
            slot = table.choose(ids, who=who)[0]
        last = content[slot]
        reused = _shared_len(last, ids) if last is not None else 0
        how = "free" if last is None else "continues" if continues(last, ids) else "evicts"
        if how == "evicts":
            res["evictions"] += 1
            evicted.setdefault(hash(tuple(last[:PREFIX])), []).append(last)
        if prev_slot is not None and slot != prev_slot:
            res["switches"] += 1
        prev_slot = slot
        content[slot] = ids
        res["tokens"] += len(ids)
        res["reused"] += reused
        res["read"] += len(ids) - reused
        role = rec.get("role") or "-"
        res["by_role"][role] = res["by_role"].get(role, 0) + len(ids) - reused
        res["rows"].append({"i": i, "route": rec.get("route"), "who": label(who), "tag": rec.get("tag"), "role": role,
                            "n": len(ids), "slot": slot, "how": how, "reused": reused, "read": len(ids) - reused})
    return res


def live_conversations(records: list[dict]) -> dict:
    """Per sender: the conversations (linked by the continuation test) and the live ones at each request."""
    convs = []                                          # [sender, first index, last index, last prompt]
    by_head: dict = {}
    for i, rec in enumerate(records):
        ids = rec["ids"]
        head = hash(tuple(ids[:PREFIX]))
        best, best_sh = None, 0
        for c in by_head.get(head, ()):
            sh = continues(c[3], ids)
            if sh > best_sh:
                best, best_sh = c, sh
        if best is None:
            best = [sender(rec), i, i, ids]
            convs.append(best)
            by_head.setdefault(head, []).append(best)
        best[2], best[3] = i, ids
    out = {}
    for key in {c[0] for c in convs}:
        mine = [c for c in convs if c[0] == key]
        live = [sum(1 for c in mine if c[1] <= t < c[2]) for t in range(len(records))]
        out[label(key)] = {"conversations": len(mine), "live_max": max(live, default=0),
                           "live_median": statistics.median(live) if live else 0}
    return out


def bench_records(mode: str, root: Path | None = None) -> list[dict]:
    """workstation_bench's request order for `mode` as trace records (user A / B; a side request is its user's)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import workstation_bench as wb
    from serve.frontend import ChatTemplate, openai_to_messages
    from serve.server import ByteTokenizer, MockEngine, Service
    root = Path(root) if root else ROOT
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, "ok", max_context=1 << 22), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    items = wb.series_items(mode, None, "01/234", 0, wb.build_turns(root), root)
    out = []
    for user, it in wb.send_order(mode, items):
        tag, msgs, extra = (tuple(it) + (None,))[:3]
        body = {"messages": msgs, "tools": wb.atb.TOOLS, "reasoning_effort": "high"}
        for k, v in (extra or {}).items():
            if v is None:
                body.pop(k, None)
            else:
                body[k] = v
        messages, tools, kw = openai_to_messages(body)
        b = svc.render_prompt(messages, tools, kw).encode("utf-8")
        name = user.split("~")[0]
        out.append({"route": "bench", "tag": tag, "role": "side" if "~" in user else "main", "who": {"key": "owner", "device": f"addr:{name}", "addr": name},
                    "ids": [hash(b[i:i + 4]) for i in range(0, len(b), 4)], "pinned": None})
    return out


def report(records: list[dict], slots: int = 4, rows: bool = False, title: str = "") -> str:
    lines = [f"== {title}{len(records)} requests, {sum(len(r['ids']) for r in records):,} prompt tokens, {slots} slots"]
    lines.append(f"{'rule':13s} {'read':>11s} {'reused':>11s} {'switches':>8s} {'evictions':>9s} {'regrets':>7s} "
                 f"{'regret tok':>11s}")
    results = [replay(records, r, slots) for r in RULES]
    for res in results:
        lines.append(f"{res['rule']:13s} {res['read']:>11,} {res['reused']:>11,} {res['switches']:>8} "
                     f"{res['evictions']:>9} {res['regrets']:>7} {res['regret_tokens']:>11,}  read by role: "
                     + ", ".join(f"{k} {v:,}" for k, v in sorted(res["by_role"].items())))
    lines.append("live conversations per sender (rule-independent): conversations, live max, live median")
    for who, v in sorted(live_conversations(records).items()):
        lines.append(f"  {who}: {v['conversations']}, {v['live_max']}, {v['live_median']}")
    if rows:
        for res in results:
            lines.append(f"-- rows, {res['rule']}")
            for r in res["rows"]:
                lines.append(f"  {r['i']:4d} {str(r['route']):9s} {r['who']:28s} {str(r['tag'] or ''):22s} "
                             f"{r['n']:8d} slot {r['slot']} {r['how']:9s} reused {r['reused']:8d} read {r['read']:8d}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace", nargs="?", help="a STRATA_SLOT_TRACE file (.jsonl or .jsonl.gz)")
    ap.add_argument("--bench", help="replay workstation_bench's order instead: a comma list of w1, w2a, w2s")
    ap.add_argument("--sequences", action="store_true",
                    help="replay the review's Claude-Code-shaped sequences instead (tools/cacheslots_sequences.py)")
    ap.add_argument("--workload-root", help="the tree the bench turns' tool results are read from (default: this one)")
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--rows", action="store_true", help="every request's slot, how, reused and read, per rule")
    a = ap.parse_args(argv)
    if sum(map(bool, (a.trace, a.bench, a.sequences))) != 1:
        ap.error("a trace file, --bench or --sequences, one of them")
    if a.sequences:
        import cacheslots_sequences as seq
        for name, make in seq.SEQUENCES.items():
            print(report(make(), a.slots, a.rows, f"sequence {name}: "))
    elif a.trace:
        print(report(read_trace(a.trace), a.slots, a.rows, f"{a.trace}: "))
    else:
        for mode in a.bench.split(","):
            print(report(bench_records(mode.strip(), a.workload_root), a.slots, a.rows, f"bench {mode.strip()}: "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
