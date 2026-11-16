"""Workstation ("parallel") latency bench: one user and two users on the served profile, config arms, fresh boots.

    python tools/workstation_bench.py <base config> P2:parallel=2 P1:parallel=none --mode w1 --work-dir <dir>
    python tools/workstation_bench.py <base config> P2:parallel=2 P1:parallel=none --mode w2a,w2b --work-dir <dir>
    python tools/workstation_bench.py --summarize <dir>/workstation_bench.jsonl

#318.  The workload is tools/agent_turns_bench.py's: 23 coding-agent turns over this repository (five sessions, ~2.5K
to ~100K tokens, consecutive turns of a session sharing their prefix), imported from it, not copied.

Arms (NAME:key=value,...) are variants of one base config, written per boot into --work-dir; the base config and the
files it points at are never written:
  parallel=N | parallel=none   set "parallel" to N, or remove the key (no key: the base's value is kept)
  exe=PATH                     another engine
  xargs=A;B;...                engine arguments appended to the base's
  KEY=value                    an environment variable, set in the arm config's "env" (which the server lays over its
                               own environment for the engine) and in the server's own environment
  `{boot}` in a value is replaced by the boot's directory, e.g. STRATA_TIMELINE={boot}/timeline.json.
Every arm config also gets: "port" --port, "host" 127.0.0.1 and no "api_key" (the bench server is local only),
"log" <boot>/engine.log, "history" {"dir": <boot>/history} (not the live server's request history), the
`--expert-profile-save` path redirected to a copy of that profile inside the boot directory (the `--expert-profile` it
loads is unchanged, or --profile-load's snapshot, so every boot starts from the same profile) and any `--dump-routing` path redirected there too.

Modes (--mode, a comma list runs each in turn on the same boot; odd rounds run them in reverse order, so over
two rounds each mode is first once per arm; `mode_pos` in each row):
  w1   one user: the 23 turns in agent_turns_bench's order
  w2a  two users alternating turn by turn: A, B, A, B, ... (the longer series ends alone)
  w2b  two users at once: each sends its next turn as soon as its previous answer arrived
  short one long turn (installer-fr.0, ~37K tokens), then 4 follow-ups that each add ~200-250 tokens (a tool call and
       20 lines of setup.py): short prompt reads on a long cached prefix
  w2s  w2a, and after each user's turn one short side request of that user (user `A~side` / `B~side`): a classifier
       prompt of its own (~300-600 tokens, thinking off, 32 tokens) whose first 512 tokens differ every time, as Claude
       Code's auto-mode classifier and title requests do, so it is a new prompt family for the server's CacheSlots
Arm key pin=1 (#321): user A's turns send "strata_cache_slot": 0, user B's 1 (side requests send none)
Users (--split, default 01/234): user A has sessions 0-1 (server-option, expert-cache: 10 turns), user B sessions 2-4
(installer-fr, cuda-kernel, tokenizer-fr: 13 turns); ~150K tokens of new prompt each (characters / 4), so neither
user is idle for most of a w2b series.  Each session has its own task after the shared system prompt and tools, so the
two users' long prefixes are distinct, as two Claude Code users' are.  With more than one mode per boot, each series'
system prompt ends with a line naming the mode, so a later series does not reuse an earlier one's prefixes.

Rounds mirror the arm order (A B, B A, ...: ABBA over two rounds), each arm on a fresh server started with no window
(CREATE_NO_WINDOW: a hidden console the engine inherits; DETACHED_PROCESS would leave the engine without one and
Windows would open a window for it), with CUDA_VISIBLE_DEVICES=1,0 and no CUDA_DEVICE_ORDER.  A boot is refused while
the port answers or another strata.exe runs (the RAM does not hold two engines).  Stopping a boot ends the whole
process tree (taskkill /T /F), checks every process of the tree (and any strata.exe started since the boot) is gone,
then waits until the available RAM is back near what it was before the first boot.

Per request (JSONL, --log, default <work-dir>/workstation_bench.jsonl): the client clock (send, first streamed token,
end), usage, the server's own record from GET /metrics?requests=all matched by this request's User-Agent marker, and
the engine log's per-request lines ("strata serve: prompt ... read in ... ms", "request metrics: decode entries ...")
assigned by log position and prompt length.  The `timings` field is wrong for requests decoded in a batch slot (#317)
and so are the record's prompt_ms / decode_ms / decode_tok_s there (the same engine.last); speed comes from the client
clock: wait = send to first token, decode tok/s = (completion tokens - 1) / (end - first token).  Sampling is the
agent's (temperature 1, top-p 0.95, top-k 20, a fixed seed), --max-new 256.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_turns_bench as atb  # noqa: E402

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
WIN = os.name == "nt"
MODES = ("w1", "w2a", "w2b", "short", "w2s")
TEMPERATURE = 1.0                  # --temperature: the agent's 1.0; 0 = greedy (parity runs)
PIN_SLOTS = {"A": 0, "B": 1, "solo": 0}   # pin=1: the engine cache slot of each user's turns (#321)
SHORT_BASE = "installer-fr.0"      # the short mode's long turn (~37K tokens)
SHORT_FILE, SHORT_LINES, SHORT_N = "setup.py", 20, 4
GIB = 1024 ** 3

# the server record's fields kept per request (serve/server.py, Service.run's history record)
METRIC_FIELDS = ("id", "time", "duration_s", "finish", "prompt_tokens", "prompt_total", "reused", "prompt_read",
                 "output_tokens", "engine_generated", "prompt_ms", "decode_ms", "decode_tok_s", "prefill_tok_s",
                 "hit_rate", "pcie_share", "drafts_offered", "drafts_accepted", "stats", "prefill", "decode", "hw",
                 "who", "client")


# ------------------------------------------------------------------------------------------------ arms and configs
@dataclass
class Arm:
    name: str
    parallel: object = None          # None: the base's; "none": removed; int: set
    exe: str | None = None
    xargs: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    setargs: dict = field(default_factory=dict)   # setarg=--flag:value;...: replace (or add) an engine flag's value
    pin: bool = False                # pin=1: user A's turns send strata_cache_slot 0, user B's 1 (side requests: none)


def parse_arm(spec: str) -> Arm:
    name, _, kv = spec.partition(":")
    if not name:
        raise ValueError(f"arm without a name: {spec!r}")
    arm = Arm(name)
    for item in (x for x in kv.split(",") if x):
        if "=" not in item:
            raise ValueError(f"arm {name}: {item!r} is not key=value")
        k, v = item.split("=", 1)
        if k == "parallel":
            if v.lower() == "none":
                arm.parallel = "none"
            else:
                arm.parallel = int(v)
        elif k == "exe":
            arm.exe = v
        elif k == "setarg":
            for x in (y for y in v.split(";") if y):
                flag, _, val = x.partition(":")
                arm.setargs[flag] = val
        elif k == "pin":
            arm.pin = v not in ("0", "", "false", "no")
        elif k == "xargs":
            arm.xargs = [x for x in v.split(";") if x]
        else:
            arm.env[k] = v
    return arm


def _redirect_arg(args: list, flag: str, new: str) -> str | None:
    """Replace the value after `flag` in `args` (in place) with `new`; the old value, or None when absent."""
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            old, args[i + 1] = args[i + 1], new
            return old
    return None


def derive_config(base: dict, arm: Arm, boot_dir: Path, port: int, profile_load: str | None = None) -> dict:
    """The arm's config for one boot, written nowhere (write_arm_config does that).  `base` is not changed; the files
    it names are only read (the profile copy)."""
    boot_dir = Path(boot_dir)
    sub = lambda s: str(s).replace("{boot}", str(boot_dir))   # noqa: E731
    cfg = copy.deepcopy(base)
    if arm.parallel == "none":
        cfg.pop("parallel", None)
    elif arm.parallel is not None:
        cfg["parallel"] = int(arm.parallel)
    if arm.exe:
        cfg["exe"] = sub(arm.exe)
    args = list(cfg.get("args") or []) + [sub(x) for x in arm.xargs]
    src = None
    if "--expert-profile-save" in args:
        dst = boot_dir / "expert-profile-save.bin"
        src = _redirect_arg(args, "--expert-profile-save", str(dst))
        if src and Path(src).is_file() and Path(src).resolve() != dst.resolve():
            boot_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    if "--dump-routing" in args:                    # never the served routing trace
        _redirect_arg(args, "--dump-routing", str(boot_dir / "routing.bin"))
    for flag, val in arm.setargs.items():           # setarg: the flag's value replaced, or the flag added
        if _redirect_arg(args, flag, sub(val)) is None:
            args += [flag, sub(val)]
    if profile_load:                                # load the profile from a snapshot, not the base's file
        _redirect_arg(args, "--expert-profile", str(profile_load))
    cfg["args"] = args
    cfg["port"] = int(port)
    cfg["host"] = "127.0.0.1"
    cfg.pop("api_key", None)
    cfg["log"] = str(boot_dir / "engine.log")
    hist = dict(cfg.get("history") or {}) if isinstance(cfg.get("history"), dict) else {}
    hist["dir"] = str(boot_dir / "history")
    cfg["history"] = hist
    env = dict(cfg.get("env") or {})
    env.update({k: sub(v) for k, v in arm.env.items()})
    cfg["env"] = env
    return cfg


def write_arm_config(cfg: dict, boot_dir: Path) -> Path:
    boot_dir.mkdir(parents=True, exist_ok=True)
    p = boot_dir / "config.json"
    p.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return p


def server_env(arm: Arm, boot_dir: Path, cuda_visible: str | None, base_env=None) -> dict:
    env = dict(os.environ if base_env is None else base_env)
    env.update({k: str(v).replace("{boot}", str(boot_dir)) for k, v in arm.env.items()})
    if cuda_visible:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    env.pop("CUDA_DEVICE_ORDER", None)              # with it the 4070 becomes device 0 and the engine exits
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ------------------------------------------------------------------------------------------------ workload and users
def session_names() -> list[str]:
    return [s[0] for s in atb.SESSIONS]


def build_turns(root: Path | None = None) -> list[tuple[str, list[dict]]]:
    """agent_turns_bench's turns with the tool results read from `root` (its files and `git grep` over its tracked
    files; default: this tree).  A tree with other files gives other grep results, so a run builds them once and
    every request row carries the hash of what it sent."""
    if root is None:
        return atb.turns()
    saved = atb.ROOT
    atb.ROOT = Path(root)
    try:
        return atb.turns()
    finally:
        atb.ROOT = saved


def messages_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def workload(system_tag: str | None = None, base=None) -> list[tuple[str, list[dict]]]:
    """agent_turns_bench's turns (`base`, else built now); with a tag, the system prompt ends with one line naming it."""
    work = build_turns() if base is None else base
    if not system_tag:
        return work
    out = []
    for tag, msgs in work:
        msgs = [dict(m) for m in msgs]
        msgs[0]["content"] = msgs[0]["content"] + f"\nSession group: {system_tag}."
        out.append((tag, msgs))
    return out


def parse_split(split: str) -> list[list[int]]:
    groups = [[int(c) for c in g] for g in split.split("/")]
    flat = sorted(i for g in groups for i in g)
    if len(groups) != 2 or flat != list(range(len(atb.SESSIONS))):
        raise ValueError(f"--split {split!r}: two groups covering sessions 0..{len(atb.SESSIONS) - 1} once each")
    return groups


def split_users(work: list[tuple[str, list[dict]]], split: str = "01/234") -> dict[str, list[tuple[str, list[dict]]]]:
    names = session_names()
    out = {}
    for user, group in zip("AB", parse_split(split)):
        keep = [names[i] for i in group]
        out[user] = [(t, m) for t, m in work if t.split(".")[0] in keep]
    return out


def alternate(users: dict[str, list]) -> list[tuple[str, tuple]]:
    """A, B, A, B, ...; the longer list ends alone."""
    a, b = users["A"], users["B"]
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(("A", a[i]))
        if i < len(b):
            out.append(("B", b[i]))
    return out


SIDE_SYSTEM = (
    "You are a safety classifier for an autonomous coding agent. You are shown the most recent output of a tool the "
    "agent ran in the user's repository. Decide whether the agent's next step, judged only from that output, could "
    "change files outside the repository, reach the network, delete data, expose credentials or run something the "
    "user did not ask for. Most steps are harmless: reading files, searching the code and listing directories are "
    "always allowed. Editing files inside the repository is allowed when the task asks for it. Running tests and "
    "builds is allowed. Installing packages, pushing to a remote, changing system settings, reading files in the "
    "user's home folder outside the repository, and anything that sends data to another machine need the user's "
    "approval. When you are unsure, prefer asking. Answer with exactly one word: ALLOW or ASK. Do not explain.")
SIDE_EXTRA = {"max_tokens": 32, "tools": None, "reasoning_effort": None,
              "chat_template_kwargs": {"enable_thinking": False}}
SIDE_TAIL_CHARS = 1200


def side_item(item) -> tuple[str, list[dict], dict]:
    """A side request after a turn (w2s): the classifier prompt with the turn's last tool output first in the user
    message, so the prompt's first 512 tokens are new every time (a new CacheSlots family).  ~300-600 tokens."""
    tag, msgs = item[0], item[1]
    task = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    last = next((m["content"] for m in reversed(msgs) if m.get("role") == "tool"), "")
    tail = last[-SIDE_TAIL_CHARS:]
    text = f"Agent task: {task}\nTool output:\n{tail}\n\n{SIDE_SYSTEM}"
    return (f"{tag}.side", [{"role": "user", "content": text}], dict(SIDE_EXTRA))


def _pinned(user: str, item, pin: bool):
    if not pin or user not in PIN_SLOTS:
        return item
    tag, msgs, extra = (tuple(item) + (None,))[:3]
    return (tag, msgs, {**(extra or {}), "strata_cache_slot": PIN_SLOTS[user]})


def send_order(mode: str, items: dict[str, list], pin: bool = False) -> list[tuple[str, tuple]]:
    """The sequential modes' send order: w1 / short in order, w2a alternating, w2s alternating with each user's side
    request after its turn.  pin: the users' own turns carry their cache slot (PIN_SLOTS); side requests none."""
    if mode in ("w1", "short"):
        order = [(u, it) for u, v in items.items() for it in v]
    elif mode == "w2s":
        order = []
        for u, it in alternate(items):
            order += [(u, it), (f"{u}~side", side_item(it))]
    else:
        order = alternate(items)
    return [(u, _pinned(u, it, pin)) for u, it in order]


def pinned_users(items: dict[str, list], pin: bool = False) -> dict[str, list]:
    return {u: [_pinned(u, it, pin) for it in v] for u, v in items.items()}


# ------------------------------------------------------------------------------------------------ the client
def post_stream(port: int, messages: list[dict], *, max_new: int, seed: int, marker: str, api_key: str | None = None,
                timeout: float = 3600, tools=None, host: str = "127.0.0.1", extra: dict | None = None) -> dict:
    """One streamed chat request.  t_first: the first delta that carries text, reasoning or a call (the first chunk is
    the empty role chunk, sent before the prompt is read; a ':' line is a keep-alive).  `extra`: body fields laid over
    the agent's (a value None removes the field), e.g. strata_cache_slot, or a side request's own settings."""
    body = {"messages": messages, "tools": atb.TOOLS if tools is None else tools, "max_tokens": max_new,
            "temperature": TEMPERATURE, "top_p": 0.95, "top_k": 20, "seed": seed, "reasoning_effort": "high",
            "stream": True}
    for k, v in (extra or {}).items():
        if v is None:
            body.pop(k, None)
        else:
            body[k] = v
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "User-Agent": marker}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers=headers)
    text: list[str] = []
    rec = {"t_send": time.time(), "t_first": None, "t_end": None, "first_kind": None, "usage": None, "finish": None,
           "timings_untrusted": None, "error": None, "chunks": 0}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    c = json.loads(data)
                except ValueError:
                    continue
                rec["chunks"] += 1
                if c.get("error"):
                    rec["error"] = c["error"]
                ch = (c.get("choices") or [{}])[0]
                d = ch.get("delta") or {}
                for key in ("reasoning_content", "content"):
                    if d.get(key):
                        text.append(d[key])
                for tc in d.get("tool_calls") or []:
                    text.append(json.dumps(tc, sort_keys=True))
                if rec["t_first"] is None:
                    kind = "content" if d.get("content") else "reasoning" if d.get("reasoning_content") else \
                        "tool_calls" if d.get("tool_calls") else None
                    if kind:
                        rec["t_first"], rec["first_kind"] = time.time(), kind
                if ch.get("finish_reason"):
                    rec["finish"] = ch["finish_reason"]
                if c.get("usage"):
                    rec["usage"] = c["usage"]
                if c.get("timings"):
                    rec["timings_untrusted"] = c["timings"]     # #317: wrong for batch-slot requests; never used
    except Exception as e:                                       # noqa: BLE001 - recorded, the series goes on
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["t_end"] = time.time()
    joined = "".join(text)                                       # the reply (reasoning, text, calls) as streamed
    rec["reply_sha"] = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    rec["reply_chars"] = len(joined)
    return client_numbers(rec)


def client_numbers(rec: dict) -> dict:
    u = rec.get("usage") or {}
    n = u.get("completion_tokens")
    rec["completion_tokens"] = n
    rec["prompt_tokens"] = u.get("prompt_tokens")
    rec["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    rec["wall_s"] = round(rec["t_end"] - rec["t_send"], 4)
    rec["ttft_s"] = round(rec["t_first"] - rec["t_send"], 4) if rec.get("t_first") else None
    dec = rec["t_end"] - rec["t_first"] if rec.get("t_first") else None
    rec["decode_s"] = round(dec, 4) if dec is not None else None
    rec["client_decode_tok_s"] = round((n - 1) / dec, 2) if n and n > 1 and dec and dec > 0 else None
    return rec


# ------------------------------------------------------------------------------------------------ schedulers
def run_sequential(port: int, items: list[tuple[str, tuple[str, list]]], make_marker, *, max_new: int, seed: int,
                   log_size=None, api_key=None, on_row=None) -> list[dict]:
    """items: [(user, (tag, messages[, extra body fields]))] in send order; each waits for the previous answer."""
    rows = []
    for seq, (user, item) in enumerate(items):
        tag, msgs, extra = (tuple(item) + (None,))[:3]
        rows.append(_one(port, user, tag, msgs, seq, make_marker, max_new, seed, log_size, api_key, on_row, extra))
    return rows


def run_concurrent(port: int, users: dict[str, list[tuple[str, list]]], make_marker, *, max_new: int, seed: int,
                   log_size=None, api_key=None, on_row=None) -> list[dict]:
    """Each user on its own thread sends its next turn as soon as its previous answer arrived; the threads start
    together."""
    rows, lock = [], threading.Lock()
    start = threading.Barrier(len(users))
    seqs = iter(range(10 ** 9))

    def worker(user, items):
        start.wait()
        for item in items:
            tag, msgs, extra = (tuple(item) + (None,))[:3]
            with lock:
                seq = next(seqs)
            row = _one(port, user, tag, msgs, seq, make_marker, max_new, seed, log_size, api_key, None, extra)
            with lock:
                rows.append(row)
                if on_row:
                    on_row(row)

    threads = [threading.Thread(target=worker, args=(u, it), daemon=True) for u, it in users.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows.sort(key=lambda r: r["t_send"])
    return rows


def _one(port, user, tag, msgs, seq, make_marker, max_new, seed, log_size, api_key, on_row, extra=None) -> dict:
    marker = make_marker(user, seq)
    off0 = log_size() if log_size else None
    rec = post_stream(port, msgs, max_new=max_new, seed=seed, marker=marker, api_key=api_key, extra=extra)
    row = {"user": user, "tag": tag, "seq": seq, "marker": marker, "msgs_sha": messages_hash(msgs), "log_off0": off0,
           "cache_slot": (extra or {}).get("strata_cache_slot"),
           "log_off1": log_size() if log_size else None, **rec}
    if on_row:
        on_row(row)
    return row


def mark_overlap(rows: list[dict]) -> None:
    """Per row: seconds its [send, end] overlapped another user's requests, and whether one was in flight at send."""
    for r in rows:
        ov, at_send = 0.0, False
        for o in rows:
            if o is r or o["user"] == r["user"]:
                continue
            ov += max(0.0, min(r["t_end"], o["t_end"]) - max(r["t_send"], o["t_send"]))
            at_send = at_send or o["t_send"] <= r["t_send"] < o["t_end"]
        r["overlap_s"] = round(ov, 3)
        r["other_in_flight_at_send"] = at_send


# ------------------------------------------------------------------------------------------------ server records
def fetch_metrics(port: int, api_key: str | None = None, timeout: float = 30) -> list[dict]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/metrics?requests=all", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()).get("requests") or []


def match_records(rows: list[dict], records: list[dict]) -> None:
    """Attach each row's server record(s): the history keeps the request's User-Agent as `client` (first 120
    characters), and every post sends a marker of its own.  One request that called the engine more than once has ids
    <id>, <id>-2, ...: all are kept (`server_all`), the first is `server`."""
    by_client: dict[str, list[dict]] = {}
    for rec in records:
        by_client.setdefault(str(rec.get("client") or "")[:120], []).append(rec)
    for row in rows:
        found = sorted(by_client.get(row["marker"][:120], []), key=lambda x: (str(x.get("id")).count("-"),
                                                                               x.get("time") or 0))
        row["server_n"] = len(found)
        row["server"] = {k: found[0].get(k) for k in METRIC_FIELDS if k in found[0]} if found else None
        row["server_all"] = [{k: f.get(k) for k in METRIC_FIELDS if k in f} for f in found[1:]] or None


# ------------------------------------------------------------------------------------------------ engine log
PROMPT_LINE = re.compile(
    r"strata serve: prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ (?P<read>\d+)(?: of (?P<fresh>\d+))? "
    r"read in (?P<read_ms>[\d.]+) ms \((?P<pp>[\d.]+) tok/s\), (?P<gen>\d+) generated in (?P<gen_ms>[\d.]+) ms "
    r"\((?P<tg>[\d.]+) tok/s\)(?:, drafts accepted (?P<dacc>\d+) of (?P<doff>\d+))?")
METRICS_LINE = re.compile(
    r"strata serve: request metrics: decode entries (?P<entries>\d+) = primary (?P<primary>\d+) \+ secondary "
    r"(?P<secondary>\d+) \+ pcie (?P<pcie>\d+) \+ cpu (?P<cpu>\d+); cpu experts (?P<cpu_ms>[\d.]+) ms")
SLOT_LINE = re.compile(r"strata batch: slot (?P<slot>\d+) takes (?P<n>\d+) tokens")


def parse_legs(data, base_offset: int = 0) -> list[dict]:
    """The engine's per-call lines in log order: one leg per `strata serve: prompt ...` line (a request moved between
    the solo path and a batch slot has several), with the `request metrics` line after it and the `slot S takes`
    line before it (an admission into a batch slot).  `data`: the log's bytes (str is encoded); `pos`: the line's byte
    offset in the log, the same count as the file sizes taken at each send."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    legs, slot, pos = [], None, base_offset
    for raw in data.splitlines(keepends=True):
        here = pos
        pos += len(raw)
        line = raw.decode("utf-8", "replace")
        m = SLOT_LINE.search(line)
        if m:
            slot = int(m["slot"])
            continue
        m = PROMPT_LINE.search(line)
        if m:
            g = m.groupdict()
            legs.append({"pos": here, "prompt": int(g["prompt"]), "reused": int(g["reused"]), "read": int(g["read"]),
                         "read_ms": float(g["read_ms"]), "generated": int(g["gen"]), "gen_ms": float(g["gen_ms"]),
                         "drafts_accepted": int(g["dacc"]) if g["dacc"] else None,
                         "drafts_offered": int(g["doff"]) if g["doff"] else None,
                         "cancelled": g["fresh"] is not None, "slot": slot})
            slot = None
            continue
        m = METRICS_LINE.search(line)
        if m and legs and "entries" not in legs[-1]:
            legs[-1].update({k: (float(v) if k == "cpu_ms" else int(v)) for k, v in m.groupdict().items()})
    return legs


def read_log(path, start: int = 0) -> bytes:
    try:
        with open(path, "rb") as f:
            f.seek(start)
            return f.read()
    except OSError:
        return b""


def assign_legs(rows: list[dict], legs: list[dict]) -> None:
    """Give each row its legs.  The first leg's prompt is the request's own (usage.prompt_tokens), written after the
    row's send; a later leg (continued after a move between the solo path and a slot) has prompt + the tokens made so
    far, and comes before the same user's next send.  A read that gave way to a shorter waiting prompt (BYIELD) goes
    on in a slot with the same prompt: its later legs have prompt == the request's, and the read is their sum.  A leg
    two rows could claim is marked ambiguous on both."""
    by_user: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["t_send"]):
        by_user.setdefault(r["user"], []).append(r)
    bound = {}
    for user_rows in by_user.values():
        for i, r in enumerate(user_rows):
            bound[id(r)] = user_rows[i + 1]["log_off0"] if i + 1 < len(user_rows) else None
    claims: dict[int, list[dict]] = {}
    for r in rows:
        p, c, off0, hi = r.get("prompt_tokens"), r.get("completion_tokens") or 0, r.get("log_off0") or 0, bound[id(r)]
        r["legs"] = []
        if p is None:
            continue
        first = next((lg for lg in legs if lg["pos"] >= off0 and lg["prompt"] == p and (hi is None or lg["pos"] < hi)),
                     None)
        if first is None:
            continue
        mine = [first] + [lg for lg in legs if lg["pos"] > first["pos"] and (hi is None or lg["pos"] < hi)
                          and p <= lg["prompt"] <= p + c]
        r["legs"] = mine
        for lg in mine:
            claims.setdefault(id(lg), []).append(r)
    for r in rows:
        r["legs_ambiguous"] = any(len(claims.get(id(lg), [])) > 1 for lg in r["legs"])
        first = r["legs"][0] if r["legs"] else None
        reads = [lg for lg in r["legs"] if lg["prompt"] == r.get("prompt_tokens")]
        r["engine_read_ms"] = round(sum(lg["read_ms"] for lg in reads), 1) if first else None
        r["engine_read"] = sum(lg["read"] for lg in reads) if first else None
        r["engine_read_legs"] = len(reads)
        r["engine_reused"] = first["reused"] if first else None
        if r["legs"] and any("entries" in lg for lg in r["legs"]):
            tiers = {k: sum(lg.get(k, 0) for lg in r["legs"]) for k in ("entries", "primary", "secondary", "pcie", "cpu")}
            tiers["cpu_ms"] = round(sum(lg.get("cpu_ms", 0.0) for lg in r["legs"]), 1)
            r["engine_tiers_solo_legs"] = tiers     # the solo legs only: a batch slot's decode prints no such line
        else:
            r["engine_tiers_solo_legs"] = None


def reused_fraction(row: dict) -> tuple[float | None, str | None]:
    """The prompt share the cache held: the engine's own first-leg line, else usage's cached_tokens."""
    p = row.get("prompt_tokens")
    if not p:
        return None, None
    if row.get("engine_reused") is not None and not row.get("legs_ambiguous"):
        return row["engine_reused"] / p, "engine"
    if row.get("cached_tokens") is not None:
        return row["cached_tokens"] / p, "usage"
    return None, None


# ------------------------------------------------------------------------------------------------ processes
def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def running_engines(names=("strata.exe",)) -> list[dict]:
    import psutil
    out = []
    for p in psutil.process_iter(["pid", "name", "create_time"]):
        if (p.info.get("name") or "").lower() in names:
            out.append(dict(p.info))
    return out


def tree_snapshot(pid: int) -> list[tuple[int, float, str]]:
    import psutil
    try:
        root = psutil.Process(pid)
        procs = [root] + root.children(recursive=True)
    except psutil.NoSuchProcess:
        return []
    out = []
    for p in procs:
        try:
            out.append((p.pid, p.create_time(), p.name()))
        except psutil.Error:
            pass
    return out


def _alive(pid: int, created: float) -> bool:
    import psutil
    try:
        return abs(psutil.Process(pid).create_time() - created) < 1e-3
    except psutil.Error:
        return False


def _ours(pid: int, tree_pids: set, scope_dir) -> bool:
    """Is this process the boot's: started by a process of its tree (Windows keeps the parent's pid after the parent
    ended), or with the boot directory on its command line (the arm's --expert-profile-save copy is there)."""
    import psutil
    try:
        p = psutil.Process(pid)
        if p.ppid() in tree_pids:
            return True
        return bool(scope_dir) and str(scope_dir).lower() in " ".join(p.cmdline()).lower()
    except psutil.Error:
        return False


def kill_tree(pid: int, timeout: float = 60.0, since: float | None = None, scope_dir=None,
              names=("strata.exe", "strata-vision.exe")) -> dict:
    """End `pid` and every process under it (taskkill /T /F on Windows), then check each one the snapshot saw is gone
    (pid + creation time, so a reused pid is not mistaken for it).  `since`: also end a process named in `names` that
    the snapshot missed, but only one of this boot's (_ours: started at or after `since` by a process of the tree, or
    with `scope_dir` on its command line) - never another server's engine, whenever it started.
    -> {"snapshot": [...], "late": [...], "leftover": [...]} (leftover: alive after the timeout and a direct kill)."""
    import psutil
    snap = tree_snapshot(pid)
    if WIN:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(int(pid))], capture_output=True,
                       creationflags=CREATE_NO_WINDOW)
    else:
        for p_pid, created, _ in reversed(snap):
            if _alive(p_pid, created):
                try:
                    psutil.Process(p_pid).kill()
                except psutil.Error:
                    pass
    late = []
    if since is not None:
        for e in running_engines(names):
            if e["create_time"] >= since - 1.0 and not any(e["pid"] == s[0] for s in snap)                     and _ours(e["pid"], {s[0] for s in snap}, scope_dir):
                late.append((e["pid"], e["create_time"], e["name"]))
    every = snap + late

    def kill(procs):
        for p_pid, created, _ in procs:
            if _alive(p_pid, created):
                try:
                    psutil.Process(p_pid).kill()
                except psutil.Error:
                    pass

    def wait_gone(seconds):
        end = time.time() + seconds
        while time.time() < end and any(_alive(p, c) for p, c, _ in every):
            time.sleep(0.2)

    kill(late)
    wait_gone(timeout)
    kill(every)                                     # what taskkill did not end in time
    wait_gone(10.0)
    left = [{"pid": p, "name": n} for p, c, n in every if _alive(p, c)]
    return {"snapshot": [{"pid": p, "name": n} for p, _, n in snap], "late": [{"pid": p, "name": n} for p, _, n in late],
            "leftover": left}


def ram_available() -> int:
    import psutil
    return int(psutil.virtual_memory().available)


def wait_ram(target: int, timeout: float, poll: float = 5.0) -> tuple[bool, int, float]:
    """Wait until the available RAM is at least `target` bytes.  -> (reached, available, seconds waited)."""
    t0 = time.time()
    while True:
        avail = ram_available()
        if avail >= target:
            return True, avail, time.time() - t0
        if time.time() - t0 >= timeout:
            return False, avail, time.time() - t0
        time.sleep(poll)


def sha256_file(path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


def git_head(root: Path) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                           creationflags=CREATE_NO_WINDOW if WIN else 0)
        d = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, creationflags=CREATE_NO_WINDOW if WIN else 0)
        return r.stdout.strip() + ("-dirty" if d.stdout.strip() else "") if r.returncode == 0 else None
    except OSError:
        return None


def start_server(python: str, server_root: Path, cfg_path: Path, port: int, env: dict, out_path: Path,
                 extra_args=None) -> subprocess.Popen:
    """serve/server.py with no window: CREATE_NO_WINDOW gives it a console without a window, which its engine and
    image encoder inherit (the server starts them without flags of their own)."""
    out = open(out_path, "ab")
    flags = (CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP) if WIN else 0
    cmd = [python, "-u", str(server_root / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port)] + list(extra_args or [])
    return subprocess.Popen(cmd, cwd=str(server_root), env=env, stdout=out, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=flags, start_new_session=not WIN)


def wait_ready(port: int, proc: subprocess.Popen, timeout: float, api_key: str | None = None) -> float:
    """Until GET /v1/models answers and a short warm-up answer comes back.  -> seconds."""
    t0 = time.time()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers=headers),
                                        timeout=5) as r:
                if r.status == 200:
                    body = {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "temperature": 0,
                            "chat_template_kwargs": {"enable_thinking": False}}
                    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                                 data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json",
                                                          "User-Agent": "workstation_bench/warmup", **headers})
                    with urllib.request.urlopen(req, timeout=600) as w:
                        json.loads(w.read())
                    return time.time() - t0
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(3)
    raise TimeoutError(f"server not ready in {timeout:.0f} s")


# ------------------------------------------------------------------------------------------------ summary
def _q(vals, q):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    k = max(0, min(len(v) - 1, int(round(q * (len(v) - 1)))))
    return v[k]


def _f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def summarize(rows: list[dict]) -> str:
    reqs = [r for r in rows if r.get("kind") == "request"]
    out = []
    groups: dict[tuple, list] = {}
    for r in reqs:
        groups.setdefault((r["mode"], r["arm"]), []).append(r)
    for (mode, arm), rs in sorted(groups.items()):
        rounds = sorted({r["round"] for r in rs})
        out.append(f"\n== {mode} arm {arm} ({len(rs)} requests, rounds {rounds}) ==")
        out.append("  user  n   wall med/p90 s    wait(ttft) med/p90 s  read ms med/p90 (src)    reused med/p90  "
                   "decode tok/s med  sum wall s")
        users = sorted({r["user"] for r in rs})
        for user in users + (["both"] if len(users) > 1 else []):
            ur = rs if user == "both" else [r for r in rs if r["user"] == user]
            ok = [r for r in ur if not r.get("error")]
            reads = [r.get("engine_read_ms") for r in ok]
            src = "engine"
            if not any(x is not None for x in reads):
                reads, src = [(r.get("server") or {}).get("prompt_ms") for r in ok], "server*"
            frac = [reused_fraction(r)[0] for r in ok]
            n_rounds = max(1, len(rounds))
            out.append(f"  {user:4s} {len(ur):3d} {_f(_q([r['wall_s'] for r in ok], .5)):>7s} / "
                       f"{_f(_q([r['wall_s'] for r in ok], .9)):<7s} {_f(_q([r['ttft_s'] for r in ok], .5)):>7s} / "
                       f"{_f(_q([r['ttft_s'] for r in ok], .9)):<9s} {_f(_q(reads, .5), 0):>7s} / "
                       f"{_f(_q(reads, .9), 0):<7s} ({src:7s}) {_f(_q(frac, .5), 3):>6s} / {_f(_q(frac, .9), 3):<6s} "
                       f"{_f(_q([r.get('client_decode_tok_s') for r in ok], .5), 1):>10s}       "
                       f"{_f(sum(r['wall_s'] for r in ok) / n_rounds, 1):>8s}"
                       + (f"   ({len(ur) - len(ok)} errors)" if len(ok) < len(ur) else ""))
        spans = []
        for rnd in rounds:
            rr = [r for r in rs if r["round"] == rnd]
            spans.append(max(r["t_end"] for r in rr) - min(r["t_send"] for r in rr))
        out.append(f"  series wall per round: {', '.join(f'{s:.1f}' for s in spans)} s"
                   f" (mean {statistics.mean(spans):.1f} s); sum wall is per round (mean over rounds)")
        amb = sum(1 for r in rs if r.get("legs_ambiguous"))
        miss = sum(1 for r in rs if not r.get("legs"))
        nomatch = sum(1 for r in rs if not r.get("server"))
        if amb or miss or nomatch:
            out.append(f"  matching: {nomatch} without a server record, {miss} without engine lines, {amb} ambiguous")
    # per turn: each arm against the first arm seen, per mode and user
    arms = []
    for r in reqs:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    if len(arms) > 1:
        out.append(f"\nper turn wall, mean over rounds, ratio to {arms[0]} (median over turns, turns faster):")
        for mode in sorted({r["mode"] for r in reqs}):
            for user in sorted({r["user"] for r in reqs if r["mode"] == mode}):
                def per_turn(arm):
                    d: dict[str, list] = {}
                    for r in reqs:
                        if r["mode"] == mode and r["user"] == user and r["arm"] == arm and not r.get("error"):
                            d.setdefault(r["tag"], []).append(r["wall_s"])
                    return {t: statistics.mean(v) for t, v in d.items()}
                ref = per_turn(arms[0])
                for arm in arms[1:]:
                    cur = per_turn(arm)
                    ratios = [cur[t] / ref[t] for t in ref if t in cur and ref[t] > 0]
                    if ratios:
                        out.append(f"  {mode} user {user} {arm}: x{statistics.median(ratios):.3f} "
                                   f"(x{min(ratios):.2f} to x{max(ratios):.2f}), "
                                   f"{sum(x < 1 for x in ratios)}/{len(ratios)} turns faster")
    out.append("\nwait = send to first streamed token; decode tok/s = (completion - 1) / (end - first token); "
               "read ms: the engine's first-leg line (server*: the /metrics record, wrong for slot requests, #317)")
    return "\n".join(out)


def load_rows(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ------------------------------------------------------------------------------------------------ main
def plan(arms: list[Arm], rounds: int) -> list[tuple[int, Arm]]:
    out = []
    for rnd in range(rounds):
        for arm in (arms if rnd % 2 == 0 else arms[::-1]):
            out.append((rnd, arm))
    return out


def short_items(work, root: Path | None = None, n: int = SHORT_N) -> list[tuple[str, list[dict]]]:
    """The live shape of a short prompt read on a long cached prefix (#318: 205-220 new tokens at 35-46K cached took
    3.4-3.7 s): one long turn (SHORT_BASE), then n follow-ups that each add one tool call and a SHORT_LINES-line result
    (setup.py, read from `root`), so each reads only those ~200-250 tokens after the turn before it."""
    msgs = dict(work)[SHORT_BASE]
    lines = ((root or atb.ROOT) / SHORT_FILE).read_text(encoding="utf-8", errors="replace").splitlines()
    out = [("short.0", msgs)]
    for k in range(1, n + 1):
        cid = f"call_short_{k}"
        part = "\n".join(lines[(k - 1) * SHORT_LINES:k * SHORT_LINES])
        msgs = msgs + [{"role": "assistant", "content": f"Reading the next part of {SHORT_FILE}.",
                        "tool_calls": [{"id": cid, "type": "function", "function": {
                            "name": "read_file", "arguments": json.dumps({"path": SHORT_FILE})}}]},
                       {"role": "tool", "tool_call_id": cid, "content": part}]
        out.append((f"short.{k}", msgs))
    return out


def series_items(mode: str, tag: str | None, split: str, max_turns: int, base=None, root: Path | None = None):
    work = workload(tag, base)
    if mode == "short":
        return {"solo": short_items(work, root)}
    users = {"solo": work} if mode == "w1" else split_users(work, split)
    return {u: v[:max_turns] for u, v in users.items()} if max_turns else users


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", nargs="?")
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--mode", default="w1", help="w1, w2a, w2b or a comma list (each in turn on the same boot)")
    ap.add_argument("--work-dir")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-turns", type=int, default=0, help="w1: the first N turns; w2: the first N of each user")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--split", default="01/234", help="sessions of user A / user B (agent_turns_bench.SESSIONS)")
    ap.add_argument("--log", help="JSONL (default <work-dir>/workstation_bench.jsonl)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--server-root", default=str(ROOT), help="the tree whose serve/server.py runs")
    ap.add_argument("--temperature", type=float, default=1.0, help="the requests' temperature (0: greedy)")
    ap.add_argument("--fixed-mode-order", action="store_true", help="every boot runs --mode in the given order "
                    "(default: odd rounds reverse it)")
    ap.add_argument("--profile-load", help="the --expert-profile every arm loads (a snapshot), instead of the base's")
    ap.add_argument("--workload-root", help="the tree the turns' tool results are read from (default: this one; its "
                    "files and tracked files decide the grep results, so another branch sends other prompts)")
    ap.add_argument("--cuda-visible", default="1,0")
    ap.add_argument("--ready-timeout", type=float, default=1800)
    ap.add_argument("--ram-margin-gib", type=float, default=2.0,
                    help="after a stop, wait until available RAM >= (before the first boot) - this")
    ap.add_argument("--ram-wait", type=float, default=600, help="the longest wait for RAM after a stop, s")
    ap.add_argument("--allow-running-engine", action="store_true",
                    help="boot even though another strata.exe runs (normally refused: RAM holds one engine)")
    ap.add_argument("--dry-run", action="store_true", help="write the arm configs and print the plan; start nothing")
    ap.add_argument("--summarize", metavar="JSONL", help="print the summary of a finished log and exit")
    a = ap.parse_args(argv)

    if a.summarize:
        print(summarize(load_rows(a.summarize)))
        return 0
    if not a.config or not a.arms or not a.work_dir:
        ap.error("config, at least one arm and --work-dir are needed")
    modes = [m.strip() for m in a.mode.split(",") if m.strip()]
    if not modes or any(m not in MODES for m in modes):
        ap.error(f"--mode: one or more of {', '.join(MODES)}")
    parse_split(a.split)
    global TEMPERATURE
    TEMPERATURE = a.temperature
    base_path = Path(a.config)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    arms = [parse_arm(s) for s in a.arms]
    if len({x.name for x in arms}) != len(arms):
        ap.error("arm names must differ")
    work_dir = Path(a.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(a.log) if a.log else work_dir / "workstation_bench.jsonl"
    run_id = time.strftime("%m%d%H%M%S")
    server_root = Path(a.server_root).resolve()
    workload_root = Path(a.workload_root).resolve() if a.workload_root else ROOT
    base_work = build_turns(workload_root)
    wl_hash = messages_hash(base_work)
    tags = {m: (m if len(modes) > 1 else None) for m in modes}

    boots = plan(arms, a.rounds)
    print(f"[wsb] run {run_id}: {len(boots)} boots, modes {modes}, arms {[x.name for x in arms]}, rounds {a.rounds}, "
          f"port {a.port}, work dir {work_dir}, workload {wl_hash} from {workload_root}", flush=True)
    for m in modes:
        items = series_items(m, tags[m], a.split, a.max_turns, base_work, workload_root)
        print(f"[wsb]   {m}: " + ", ".join(f"user {u} {len(v)} turns ({v[0][0]} .. {v[-1][0]})" for u, v in items.items()),
              flush=True)
    if a.dry_run:
        for rnd, arm in boots:
            bd = work_dir / f"{run_id}-r{rnd}-{arm.name}"
            cfg = derive_config(base, arm, bd, a.port, a.profile_load)
            p = write_arm_config(cfg, bd)
            print(f"[wsb] dry run: r{rnd} {arm.name}: {p} parallel={cfg.get('parallel')} env+={arm.env} "
                  f"exe sha256 {(sha256_file(cfg['exe']) or 'missing')[:12]}", flush=True)
        return 0

    if port_in_use(a.port):
        print(f"[wsb] port {a.port} already answers: refusing to boot", flush=True)
        return 2
    others = running_engines()
    if others and not a.allow_running_engine:
        print(f"[wsb] another strata.exe runs ({[o['pid'] for o in others]}): refusing to boot "
              "(the RAM holds one engine; --allow-running-engine overrides)", flush=True)
        return 2
    ram0 = ram_available()
    out = open(log_path, "a", encoding="utf-8")
    raw = open(str(log_path) + ".raw", "a", encoding="utf-8")       # client rows as they come (a crash keeps them)
    lock = threading.Lock()

    def write(path_f, row):
        with lock:
            path_f.write(json.dumps(row, default=str) + "\n")
            path_f.flush()

    all_rows = []
    for rnd, arm in boots:
        bd = work_dir / f"{run_id}-r{rnd}-{arm.name}"
        cfg = derive_config(base, arm, bd, a.port, a.profile_load)
        boot_modes = modes if (rnd % 2 == 0 or a.fixed_mode_order) else modes[::-1]   # each mode first once per arm
        cfg_path = write_arm_config(cfg, bd)
        env = server_env(arm, bd, a.cuda_visible)
        boot = {"kind": "boot", "run": run_id, "arm": arm.name, "round": rnd, "modes": boot_modes, "exe": cfg["exe"],
                "profile_load": a.profile_load, "profile_load_sha256": sha256_file(a.profile_load) if a.profile_load else None,
                "exe_sha256": sha256_file(cfg["exe"]), "server_root": str(server_root),
                "server_git": git_head(server_root), "config": str(cfg_path), "boot_dir": str(bd),
                "parallel": cfg.get("parallel"), "arm_env": arm.env, "xargs": arm.xargs, "workload": wl_hash, "workload_root": str(workload_root),
                "split": a.split, "max_new": a.max_new, "seed": a.seed, "ram_avail_gib_before_run": round(ram0 / GIB, 2),
                "ram_avail_gib_at_boot": round(ram_available() / GIB, 2)}
        others = [] if a.allow_running_engine else running_engines()
        if port_in_use(a.port) or others:
            boot["error"] = f"port {a.port} still answers" if port_in_use(a.port) else                 f"another strata.exe runs ({[o['pid'] for o in others]})"
            write(out, boot)
            print(f"[wsb] {boot['error']}: stopping", flush=True)
            return 3
        t_boot = time.time()
        proc = start_server(a.python, server_root, cfg_path, a.port, env, bd / "server.out")
        boot["server_pid"] = proc.pid
        series_rows = []
        try:
            boot["ready_s"] = round(wait_ready(a.port, proc, a.ready_timeout), 1)
            print(f"[wsb] r{rnd} {arm.name}: ready in {boot['ready_s']:.0f} s (exe {str(boot['exe_sha256'])[:12]}, "
                  f"parallel {cfg.get('parallel')})", flush=True)
            elog = Path(cfg["log"])
            log_size = lambda: elog.stat().st_size if elog.exists() else 0   # noqa: E731
            for mode_pos, mode in enumerate(boot_modes):
                items = series_items(mode, tags[mode], a.split, a.max_turns, base_work, workload_root)
                mk = lambda user, seq, m=mode: f"wsb/{run_id}/{arm.name}/r{rnd}/{m}/{user}/{seq:03d}"   # noqa: E731
                on_row = lambda r, m=mode: (write(raw, {"run": run_id, "arm": arm.name, "round": rnd, "mode": m, **r}),   # noqa: E731
                                            print(f"  {m} {r['user']} {r['tag']:16s} wall {r['wall_s']:7.2f} s  "
                                                  f"wait {_f(r['ttft_s'])} s  n {r['completion_tokens']}  "
                                                  f"{_f(r['client_decode_tok_s'], 1)} tok/s"
                                                  + (f"  ERROR {r['error']}" if r['error'] else ""), flush=True))
                t0 = time.time()
                if mode == "w2b":
                    rows = run_concurrent(a.port, pinned_users(items, arm.pin), mk, max_new=a.max_new, seed=a.seed,
                                          log_size=log_size, on_row=on_row)
                else:
                    rows = run_sequential(a.port, send_order(mode, items, arm.pin), mk, max_new=a.max_new,
                                          seed=a.seed, log_size=log_size, on_row=on_row)
                mark_overlap(rows)
                try:
                    match_records(rows, fetch_metrics(a.port))
                except Exception as e:                       # noqa: BLE001
                    print(f"[wsb] /metrics not read: {e}", flush=True)
                    for r in rows:
                        r.setdefault("server", None)
                for r in rows:
                    r.update({"kind": "request", "run": run_id, "arm": arm.name, "round": rnd, "mode": mode,
                              "system_tag": tags[mode], "mode_pos": mode_pos, "exe_sha256": boot["exe_sha256"],
                              "parallel": cfg.get("parallel")})
                print(f"[wsb] r{rnd} {arm.name} {mode}: {len(rows)} requests in {time.time() - t0:.0f} s", flush=True)
                series_rows += rows
        except Exception as e:                               # noqa: BLE001
            boot["error"] = f"{type(e).__name__}: {e}"
            print(f"[wsb] r{rnd} {arm.name}: {boot['error']}", flush=True)
        finally:
            stop = kill_tree(proc.pid, since=t_boot, scope_dir=bd)
            boot["stop"] = stop
            if stop["leftover"]:
                print(f"[wsb] processes left after the stop: {stop['leftover']}: stopping the run", flush=True)
            reached, avail, waited = wait_ram(ram0 - int(a.ram_margin_gib * GIB), a.ram_wait)
            boot["ram_after_stop"] = {"reached": reached, "avail_gib": round(avail / GIB, 2), "waited_s": round(waited, 1)}
            print(f"[wsb] r{rnd} {arm.name}: stopped ({len(stop['snapshot'])} processes), RAM "
                  f"{avail / GIB:.1f} GiB available after {waited:.0f} s" + ("" if reached else " (target not reached)"),
                  flush=True)
        legs = parse_legs(read_log(cfg["log"]))
        assign_legs(series_rows, legs)
        write(out, boot)
        for r in series_rows:
            write(out, r)
        all_rows += [boot] + series_rows
        if boot.get("stop", {}).get("leftover") or ("error" in boot and not series_rows):
            print(summarize(all_rows))
            return 3
    print(summarize(all_rows))
    print(f"\n[wsb] DONE log {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
