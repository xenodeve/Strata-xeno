"""serve/server.py - plan v0.3 P8: OpenAI and Anthropic endpoints over any engine that maps token ids to tokens.

    python -m serve.server --engine mock --port 8095            (a scripted engine, for clients and tests)
    python -m serve.server --engine strata --config strata.json --port 8080   (the real engine, resident)

Endpoints: POST /v1/chat/completions (OpenAI, stream and non-stream), POST /v1/messages (Anthropic, stream and
non-stream), GET /v1/models, GET /models, GET /props, GET /slots, GET /health, GET /mcp. One sequence at a time behind a FIFO (plan: one resident sequence).
Tools from MCP servers (serve/mcp.py, `"mcp_servers"` in the config or --mcp-config) are offered only to requests that
ask for them with `"strata_mcp": true` - the web app does; other clients see exactly the API they always saw.
Images (optional, when the config has a "vision" entry): OpenAI image_url parts and Anthropic image blocks (base64
data, http(s) URLs or local file paths) go through `strata-vision` (the model's mmproj file) and reach the engine as
embeddings (`GENI`).  JPEG/PNG/BMP/GIF go straight in; WebP, TIFF, AVIF, ... (agents like omp send WebP) are
converted to PNG first with Pillow.
Requests whose prompt plus max tokens exceed the engine's context are REJECTED with 400, never truncated.
An unset (or 0, or -1) max tokens means "unlimited": whatever the prompt leaves of the context.

The engine boundary is `Engine.generate(prompt_ids, max_new, sampling, cancel) -> iterator of token ids`.
`StrataEngine` keeps one `strata --serve` process resident (weights, expert arena and VRAM tier load once) and
talks to it over stdin/stdout; `MockEngine` is a scripted stand-in that makes every API path testable without a GPU.
"""
from __future__ import annotations

import argparse
import contextlib
import collections
import base64
import hashlib
import hmac
import ipaddress
import codecs
import heapq
import itertools
import ctypes
import gzip
import json
import math
import os
import queue
import re
import select
import signal
import socket
import struct
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from http.client import HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Protocol
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))   # run as a script (run-<model>.bat) as well as a module
from serve.frontend import (THINK_END, ChatTemplate, Event, OutputParser, anthropic_to_messages,  # noqa: E402
                            forced_call, images_of, literal_tags, mark_think_literals, openai_to_messages,
                            tool_choice_of, unmark_think_literals)
from serve.loop_guard import LoopGuard
from serve import reasoning_carry
from serve import cjk_guard, forced_opening, gguf_info, think_budget  # noqa: E402  (xeno #49 S4, S7 follow-up, S3)
from serve import app_settings, strata_tools  # noqa: E402  (#339: the app's settings writers; Strata's own docs and settings for the chat)
from serve.timing_line import report as timing_report  # noqa: E402  (xeno #49 S5)
from serve import timeline  # noqa: E402  (#33: STRATA_TIMELINE, the server's lanes)
from serve.history import HistoryStore, chunk_stats, prompt_for_keep, request_meta, summary_record, window_rates  # noqa: E402  (xeno UI S3)
from serve import harness, mcp_admin  # noqa: E402
from serve import identity as identity_mod  # noqa: E402  (#310: named keys, roles, the approval of devices)
from serve import skills as skills_mod  # noqa: E402
from serve import agent as agent_mod, agent_prompt, agent_run, hooks as hooks_mod, permissions, runs as runs_mod, shell as shell_mod, subagent as subagent_mod, web as web_mod  # noqa: E402
from serve import checkpoints as checkpoints_mod, folders as folders_mod, memory as memory_mod  # noqa: E402
from serve.mcp import McpCancelled, McpHub, hub_from_config  # noqa: E402
from serve import runconfig  # noqa: E402
from serve import runner_hub  # noqa: E402  (#310 part 2: the runner channel)
from serve import page_routes  # noqa: E402  (#310 part 2: the page routes that act on a project's machine)
from serve import logit_bias as bias_api  # noqa: E402
from serve.winjob import contain  # noqa: E402
from serve.structured import StructuredOutputError, prepare_format, validated_json  # noqa: E402
from serve import responses as responses_api  # noqa: E402
from serve.responses import ResponsesError, error_body as responses_error_body  # noqa: E402

IM_END = "<|im_end|>"
IMAGE_PAD = "<|image_pad|>"


def popen(what: str, args: list, **kw):
    """#735: subprocess.Popen, but Windows' refusal to start a program (WinError 4551, or 1260 when a policy blocks it:
    Smart App Control on a clean Windows 11 blocks an exe that is not signed) is said in words instead of a bare
    OSError."""
    try:
        return subprocess.Popen(args, **kw)
    except OSError as e:
        if getattr(e, "winerror", None) in (4551, 1260):
            raise OSError(f"Windows blocked {what} ({args[0]}): Smart App Control, or an application-control policy, does "
                          "not let a program that is not signed run. Turn Smart App Control off (Windows Security > App "
                          "& browser control > Smart App Control settings), or - for the image encoder only - run setup "
                          "with --vision no (see docs/TROUBLESHOOTING.md)") from e
        raise


def listen_problem(host: str, port: int, e: OSError) -> str:
    """#769: why the server cannot listen.  Only an address that is taken says "already in use"; anything else (a port
    Windows keeps reserved, an address this PC does not have, no permission) says what the OS said."""
    import errno
    code = getattr(e, "winerror", None) or e.errno
    if code in (errno.EADDRINUSE, 10048):
        return (f"port {port} is already in use - is Strata (or another server) already running? "
                f"Close it, or start this one with a different --port")
    what = e.strerror or str(e)
    if code in (10013, errno.EACCES):
        hint = ("Windows may keep this port for itself (see `netsh int ipv4 show excludedportrange protocol=tcp`) "
                "or a firewall rule forbids it: start this one with a different --port")
    elif code in (10049, errno.EADDRNOTAVAIL):
        hint = f"{host} is not an address of this PC: check \"host\" in the model's strata-<model>.json or --host"
    else:
        hint = "check --host and --port"
    return f"cannot listen on {host}:{port}: {what} [{code}]. {hint}"


def combined_embeddings_path(vision_dir: Path, nbytes: int) -> Path:
    """#874: where one request's combined image-embeddings file goes.  Every request with images rewrites it (it is
    deleted after the request), which on a disk is steady heavy writing: on Linux it goes to /dev/shm when that has
    room (the engine reads it back at once), on Windows the file is opened short-lived (the cache is not forced out
    to the drive).  Anywhere else, or when /dev/shm is too small (Docker's default is 64 MB), it stays in the encoder's
    own directory, as before."""
    name = f"req-{uuid.uuid4().hex[:12]}.sve"
    shm = Path("/dev/shm")
    if os.name != "nt" and shm.is_dir() and os.access(shm, os.W_OK):
        try:
            if shutil.disk_usage(shm).free > 2 * nbytes + (16 << 20):
                return shm / f"strata-vision-{os.getpid()}-{name}"
        except OSError:
            pass
    return Path(vision_dir) / name


def write_temporary(path: Path, parts: list) -> None:
    """#874: write `parts` (files) one after the other into `path`; on Windows with FILE_ATTRIBUTE_TEMPORARY
    (os.O_SHORT_LIVED), so the file normally never reaches the drive before it is deleted."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0) | getattr(os, "O_SHORT_LIVED", 0)
    with os.fdopen(os.open(path, flags, 0o600), "wb") as f:
        for part in parts:
            f.write(Path(part).read_bytes())
# #458: the trailing effort turn ("effort_position": "end"); low is the template's own sentence, medium (which the
# template says nothing for: no xhigh sentence is medium there) says it, since the xhigh one stays at the top
EFFORT_TURN = "<|im_start|>system\n{}<|im_end|>\n"
EFFORT_TEXT = {"low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                      "conclusion without unnecessary elaboration.",
               "medium": "Reasoning effort is set to medium. Think as much as the task needs, without unnecessary "
                         "elaboration.",
               # the template's own xhigh sentence (chat_template.jinja), which #728's recovery swaps for the low one
               "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                        "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
                        "clarity in the final answer."}
VISION_START = "<|vision_start|>"
# #606: a reply that repeats one token this many times in a row is ended there ("length"): a model in a loop, or a
# broken state that answers one token forever (an issue saw 36,689 tokens of "!"). The config's "repeat_stop_tokens"
# sets it; 0 turns it off.  It is reported as the loop guard's stop (serve/loop_guard.py): stop_detail "loop", counted
# in /status loops_stopped.  It catches what the guard cannot see: a token that never becomes reply text (a special
# token, a tool call's arguments) and a repeated token longer than the guard's 8-character period.  Its cost: a reply
# that really wants one token 256+ times in a row (a long divider, padding) is cut there; 0 or a larger N allows it.
REPEAT_STOP_TOKENS = 256
ANSWER_RESERVE_MIN = 512      # #984: the tokens kept for the answer when a thinking budget is cut down to fit max_tokens
# #123: what closes the thinking when it reaches reasoning_budget_tokens (the model's own end-of-thinking tag after it)
# #1053 (opt-in "reasoning_close_retry": true): a reply that ends on its stop token still inside <think>, with no answer
# and no call, is continued once with the thinking closed (the same way the budget's wrap-up closes it)
REASONING_CLOSE = "\n</think>\n\n"
REASONING_WRAP_UP = "\n\nI have thought about this long enough; time to give my answer.\n</think>\n\n"
# #728: opt-in handling of reasoning that repeats whole passages (which the single-token guard above cannot see).
# "reasoning_loop_recovery": "stop" ends the reply there; "recover" (or true) goes on from the same output with the
# low-effort instruction in place of the xhigh one.  Both off by default.
HIGH_EFFORT = EFFORT_TEXT["xhigh"]
LOW_EFFORT = EFFORT_TEXT["low"]
LOOP_CHECK_EVERY = 512           # output tokens between two looks at the reasoning (at a clean parser boundary)
LOOP_COVERAGE = 0.25             # the share of the last 2,000 words inside 12-word passages seen three times
LOOP_HISTORY_WORDS = 30000       # how far back the passages are counted (bounds the cost of a look)
LOOP_TAIL_CHARS = 2048           # chars at the end of the reasoning also checked for a short period (#1753)
LOOP_TAIL_PERIOD = 64            # a period at most this long in that tail reads as a fully repeated window


def reasoning_repeat_coverage(text):
    """Coverage of recent words by 12-word passages seen at least three times.

    Only reasoning is supplied. The history (the last LOOP_HISTORY_WORDS words) detects repeated verification passes
    separated by long code drafts; the recent window excludes old repetitions.  #1753: a loop can also be one long
    word (a 24k-digit string cycling an 11-digit pattern): the word split makes it a single word and no word count
    grows, and #606's token run never grows either (the tokenizer has no multi-digit tokens), so before the word
    count the last LOOP_TAIL_CHARS chars are checked for a period of at most LOOP_TAIL_PERIOD, and a periodic tail
    reads as full coverage: it is all of the recent output, repeated.
    """
    tail = text[-LOOP_TAIL_CHARS:].lower()
    if len(tail) == LOOP_TAIL_CHARS and any(tail[p:] == tail[:-p] for p in range(1, LOOP_TAIL_PERIOD + 1)):
        return 1.0
    words = re.findall(r"\w+|[^\w\s]", text.lower())[-LOOP_HISTORY_WORDS:]
    width, window = 12, 2000
    if len(words) < window:
        return 0.0
    grams = [tuple(words[i:i + width]) for i in range(len(words) - width + 1)]
    counts = collections.Counter(grams)
    start = max(0, len(words) - window)
    covered = set()
    for i in range(start, len(grams)):
        if counts[grams[i]] >= 3:
            covered.update(range(i, i + width))
    return len(covered) / (len(words) - start)


def focused_recovery_prompt(tok, ids, generated):
    """The prompt's ids with the xhigh effort sentence of the first system message replaced by the low one, then
    every token generated so far.  A splice of token ids: the prompt is never decoded and encoded again, so a literal
    `</think>` or vision marker in it (#537, #554) stays what it was.  None when the sentence is not there as whole
    tokens in the first system message (a user's text never counts).  No answer or end-of-thinking is inserted."""
    end = tok.encode(IM_END, parse_special=True)
    if len(end) != 1 or end[0] not in ids:
        return None
    head = ids[:ids.index(end[0])]
    if not tok.decode(head).startswith("<|im_start|>system\n"):
        return None
    old, new = tok.encode(HIGH_EFFORT, parse_special=False), tok.encode(LOW_EFFORT, parse_special=False)
    for i in range(len(head) - len(old) + 1):
        if head[i:i + len(old)] == old:
            return ids[:i] + new + ids[i + len(old):] + list(generated)
    return None


LOOPBACK_NAMES = ("localhost", "127.0.0.1", "::1")
QUEUE_BEAT_S = 10.0         # a request waiting for the single turn sends a keep-alive this often (#1619)
KEEPALIVE_S = 10.0          # a held-back tool argument leaves the stream quiet: ping at least this often
CTX_SLACK = 8               # `strata --serve` rejects prompt + max_new + 8 > context: keep the same margin here
# The live tok/s is a rate over a window, not a mean since the first token: a mean reads ~1/elapsed at the first
# token (the Monitor showed five-digit numbers) and then undershoots for the first second of every answer.
RATE_WINDOW_S = 2.0
RATE_MIN_SPAN_S = 0.25      # younger than this there is no rate yet: the mean so far, with the span floored here
# #481: a running request whose engine prints nothing (no T, PP or any other line) for this long has lost step with the
# server (the engine's main thread waits, untimed, for its next command): the engine is ended and the request fails;
# the next request starts it again.  The config's "engine_silence_s" sets it (0: wait forever, as before).
ENGINE_SILENCE_S = 300.0
# Disk sessions: the longest allowance an engine may announce (`SWAIT <phase> <s>`) for one step that blocks in a
# single call (a file flush, the host->device transfer); the engine's own bound is the same (session_phase_limit_s).
SESSION_WAIT_MAX_S = 3600
# ... except while a prompt is read: a PP line comes once per chunk (up to 32768 tokens with --prefill auto, issue
# #282), and the slowest PCs read ~100 tok/s, so a first chunk can take minutes before the first line.  Until the first
# PP the wait adds the chunk's tokens at PP_FLOOR_TOK_S; after one, a chunk may take PP_SLACK x the last one's time.
PP_CHUNK_MAX = 32768
PP_FLOOR_TOK_S = 50.0
PP_SLACK = 3.0
# Reads that had no timeout (#1317).  A blocking read of the image encoder's pipe or of the engine's READY line that
# never returned held the request turn for good: the engine then sat idle in its command loop, so no watchdog could
# fire and every later request queued behind it (the "lost step" class of #481).  Generous on purpose: a cold model
# load, a slow disk or a slow CPU encode is not a hang.  Only a process that says nothing at all for this long is
# ended; the next request starts it again.
# STRATA_ENGINE_READY_S / STRATA_VISION_READY_S / STRATA_VISION_ENCODE_S override them (0 = wait for ever, as before).


def _timeout_env(name: str, default: float) -> float | None:
    try:
        v = float(os.environ.get(name, default))
    except ValueError:
        v = default
    return None if v <= 0 else v


# #1317 part 2: the quick watchdog for a FROZEN engine.  engine_silence_s (above) is generous on purpose: a slow PC reads a
# prompt chunk for minutes without a line.  But an engine that prints nothing for ENGINE_STALL_S and also uses no CPU time and
# moves no bytes of I/O in that time is not slow, it is idle or frozen (a deadlock in a CUDA call, a driver stall, a lost
# step), and waiting longer cannot help: it is ended, and the next request starts it again.  An engine that is silent but
# still working is never ended by this, only reported once (the recommendation: engine_silence_s, or this variable).
# STRATA_ENGINE_STALL_S overrides it (0 = off).  Needs psutil (setup installs it); without it nothing is ended.
ENGINE_STALL_S = _timeout_env("STRATA_ENGINE_STALL_S", 90.0)
STALL_CPU_EPS_S = 0.5       # CPU seconds the engine may use in the whole stall window and still count as frozen
STALL_IO_EPS_B = 1 << 20    # bytes read or written in that window


def engine_frozen(base: tuple[float, int], now: tuple[float, int]) -> bool:
    """#1317: True when two (cpu seconds, io bytes) samples of the engine show no work between them."""
    return (now[0] - base[0]) <= STALL_CPU_EPS_S and (now[1] - base[1]) <= STALL_IO_EPS_B


VISION_READY_S = _timeout_env("STRATA_VISION_READY_S", 300.0)
VISION_ENCODE_S = _timeout_env("STRATA_VISION_ENCODE_S", 300.0)
ENGINE_READY_S = _timeout_env("STRATA_ENGINE_READY_S", 900.0)


# ------------------------------------------------------------------------------------------------ engines
class Engine(Protocol):
    max_context: int
    def generate(self, ids: list[int], max_new: int, sampling: dict, cancel: threading.Event) -> Iterator[int]: ...


class MockEngine:
    """Replays a scripted completion (text) as token ids, one per step, then the end-of-turn token.  Given a list of
    scripts, each request gets the next one and the last one repeats (a tool call, then the answer after it)."""

    def __init__(self, tokenizer, script: str | list[str], max_context: int = 32768, delay_s: float = 0.0):
        self.tok, self.max_context, self.delay = tokenizer, max_context, delay_s
        end = tokenizer.encode(IM_END, parse_special=True)
        self.scripts = [tokenizer.encode(x, parse_special=True) + end for x in ([script] if isinstance(script, str)
                                                                                 else script)]
        self.script, self.turns = self.scripts[0], 0
        self.last_prompt: list[int] = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        self.last_embeddings = embeddings
        if len(self.scripts) > 1:
            self.script = self.scripts[min(self.turns, len(self.scripts) - 1)]
            self.turns += 1
        for t in self.script[:max_new]:
            if cancel.is_set():
                return
            if self.delay:
                time.sleep(self.delay)
            yield t


class EngineDied(RuntimeError):
    """The engine process ended in the middle of a request (issue #27: on Linux, the out-of-memory killer)."""


class EngineStarting(RuntimeError):
    """The engine is (re)starting and has not said READY yet (#344): no context size to plan a request with - a 503
    (529 on /v1/messages, xeno #49 S2), not a 400 about the prompt."""


class ModelBusy(RuntimeError):
    """Explicit model controls must not interrupt active or queued requests."""


class EngineSilent(EngineDied):
    """#481: the engine said nothing for too long during a request (or never acknowledged a STOP): the two sides lost
    step - the engine waiting for its next command, the server for this request's end - and the server ended it.  An
    EngineDied, so the request ends with an error and the next one starts the engine again."""


class SessionRefused(ValueError):
    """Disk sessions: the engine refused or failed a SAVE / RESTORE and is still in step (`SERR <kind> <published>
    <reason>`).  `kind` is the engine's category - invalid (400), storage (507), memory (503), io (500); `published`:
    a save whose new file already replaced the old one before a later step (the folder flush) failed."""

    STATUS = {"invalid": 400, "storage": 507, "memory": 503, "io": 500}

    def __init__(self, kind: str, message: str, published: bool = False):
        super().__init__(message)
        self.kind = kind if kind in self.STATUS else "io"
        self.published = published

    @property
    def status(self) -> int:
        return self.STATUS[self.kind]


class EngineStuck(RuntimeError):
    """The engine process did not end after QUIT, terminate and kill: the server keeps it (and says so) rather than
    reporting its GPU and RAM as given back."""


class GpuBusy(RuntimeError):
    """The model is unloaded and the GPU has less free VRAM than min_free_vram_mib: something else (a game, another
    model server) is using it, so the engine is not started into the little that is left."""


ENGINE_REQUEST = re.compile(
    # "+ 12288 of 98179 read": a request cancelled while its prompt was read (#471)
    r"prompt (?P<prompt>\d+) tokens = (?P<reused>\d+) reused \+ \d+(?: of \d+)? read in (?P<read>[\d.]+) ms "
    r"\((?P<pp>[\d.]+) tok/s\), "
    r"(?P<gen>\d+) generated in (?P<gen_ms>[\d.]+) ms \((?P<tg>[\d.]+) tok/s\)")


_echoing: set[str] = set()      # the logs echo_requests already follows (restart() runs StrataEngine.__init__ again)

DRAFT_HEAD_FAIL = "the draft head does not fit"
DRAFT_HEAD_HINT = ("a smaller draft vocabulary needs less VRAM: --draft-vocab cyrillic (English, code and the Cyrillic "
                   "script) or --draft-vocab en (English and code, ~215 MiB less than the default). Start once with "
                   "it - START-HERE.bat --draft-vocab en (Windows) or ./setup.sh --draft-vocab en - and the model "
                   "keeps it; or a smaller --context in setup.")


def start_failure_hint(log: str | None, offset: int) -> str:
    """#474: what to change when the engine stopped at the start because the MTP draft head did not fit the VRAM
    left: the engine's own `strata mtp:` lines after that failure (0.1.36+: what it needs, what is free, the smaller
    subsets), else the same advice in words for an older engine.  "" for any other failure: the log says why."""
    if not log:
        return ""
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            text = f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""
    if DRAFT_HEAD_FAIL not in text:
        return ""
    said = [x.strip()[len("strata mtp: "):] for x in text.splitlines()
            if x.strip().startswith("strata mtp: ") and ("draft head over" in x or "hint:" in x)]
    return ". mtp: " + DRAFT_HEAD_FAIL + ". " + (" ".join(said) if said else "Hint: " + DRAFT_HEAD_HINT)


def start_log_tail(log: str | None, offset: int, n: int = 20) -> str:
    """#496: the last n lines this start wrote to the engine log, for the error when the engine ended before READY -
    whatever the failure, the engine's own reason is in them (people posted the traceback without the log).  "" when
    there is no log or nothing in it from this start."""
    if not log:
        return ""
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            start = max(offset, f.tell() - 64 * 1024)   # enough for 20 lines, never an earlier start's
            f.seek(start)
            text = f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""
    lines = text.splitlines()[1 if start > offset else 0:]   # not a line cut in half
    lines = [x.rstrip() for x in lines if x.strip()][-n:]
    if not lines:
        return ""
    return "\nthe engine log's last lines:\n" + "\n".join("  " + x for x in lines)


def echo_requests(log_path: str, offset: int) -> None:
    """STRATA_REQUEST_LINES=1: one stdout line per finished request, from the engine's own summary in its log.

    The engine's stderr goes to the log file (the start narrator reads it), so a supervisor that only sees this
    process's output - a tray, llama-swap - has no per-request numbers. This re-states the engine's line with the
    total the two times make: `request prompt P cached C output O prompt_read R ms total S ms prefill X tok/s decode Y
    tok/s` (prompt_read: the time the engine spent reading the prompt's new tokens, not a time to first token).
    """
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            m = ENGINE_REQUEST.search(line)
            if m:
                read_ms, gen_ms = float(m["read"]), float(m["gen_ms"])
                print("[strata] request prompt %s cached %s output %s prompt_read %.0f ms total %.0f ms prefill %s "
                      "tok/s decode %s tok/s" % (m["prompt"], m["reused"], m["gen"], read_ms, read_ms + gen_ms, m["pp"],
                                                 m["tg"]), flush=True)


def experts_loading_words(args: list, size: str) -> str:
    """#505: what the start does with the experts, by the engine's flags (generate.cpp's option parsing): a RAM budget
    copies the hottest N GiB into RAM (--resident-budget-gib), the resident low-RAM mode the ones the GPU does not hold
    (--resident-experts), plain --mmap-experts reads them from the model files through the OS file cache (nothing is
    loaded into RAM up front); otherwise all of them go into RAM."""
    if "--resident-budget-gib" in args:
        try:
            n = f"up to {float(args[args.index('--resident-budget-gib') + 1]):g} GiB"
        except (IndexError, ValueError):
            n = "a RAM budget"
        return (f"loading the most-used experts into RAM ({n}; the rest are read from the model files as needed) "
                "and locking part of them for the GPU.")
    if "--resident-experts" in args:
        return (f"loading the experts the GPU does not hold into RAM (of {size}) and locking part of them for the "
                "GPU.")
    if "--mmap-experts" in args:
        return (f"mapping the experts from the model files ({size}, --mmap-experts): they are not loaded into RAM - "
                "the OS file cache reads them as the GPU's expert cache fills and as requests need them.")
    return f"loading the experts into RAM ({size}) and locking part of them for the GPU."


def narrate_start(log_path: str, offset: int, args: list, done: threading.Event, heartbeat=20.0) -> None:
    """While the engine starts, say in the server window what it is doing, from its log: the start reads tens of GB
    into RAM and locks part of it for the GPU, and on many PCs everything is slow or frozen for a minute or more -
    people closed the window thinking it had hung.  The warning comes at that step, not after it."""
    gb = 0.0
    if "--native" in args:                              # about the size of the experts it will read
        try:
            gb = os.path.getsize(args[args.index("--native") + 1]) / 1e9
        except (OSError, IndexError):
            pass
    size = f"about {gb:.0f} GB" if gb >= 1 else "tens of GB"
    loading = experts_loading_words(args, size)
    t0 = last = time.time()
    said = set()

    def say(key, text):
        nonlocal last
        if key not in said:
            said.add(key)
            last = time.time()
            print(text, flush=True)

    say("weights", "[strata] starting the engine: reading the model's weights ...")
    pos = offset
    while not done.wait(0.5):
        try:
            with open(log_path, "rb") as f:
                f.seek(pos)
                chunk = f.read()
        except OSError:
            chunk = b""
        if chunk.count(b"\n"):
            cut = chunk.rfind(b"\n") + 1
            pos += cut
            for line in chunk[:cut].decode("utf-8", "replace").splitlines():
                if "PLE on" in line or "expert arena:" in line or "experts via mmap" in line:   # #505: mapped
                    say("arena", f"[strata] {loading}\n"
                                 "         YOUR PC CAN BE SLOW OR STOP RESPONDING FOR 1-3 MINUTES NOW - this is normal.\n"
                                 "         Please wait and don't close this window; the browser opens when it is ready.")
                elif " loaded " in line and "GiB at" in line:
                    say("loaded", "[strata] experts loaded: " + line.split(" loaded ", 1)[1].strip() +
                        f" ({time.time() - t0:.0f} s so far)")
                elif "expert cache " in line and " slots, " in line and "auto" not in line:
                    n = line.split("expert cache ", 1)[1].split(";")[0].replace(" slots,", " experts,").strip()
                    say("cache", f"[strata] filling the GPU's expert cache ({n}) ...")
                elif "session is up" in line:
                    say("up", "[strata] almost ready ...")
        if time.time() - last > heartbeat:
            last = time.time()
            print(f"[strata] still starting ({time.time() - t0:.0f} s) - please wait ...", flush=True)


def cache_slot(request: dict) -> int:
    """An explicit independent conversation cache, not another execution lane."""
    slot = request.get("strata_cache_slot", 0)
    if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot <= 3:
        raise ValueError("strata_cache_slot must be an integer from 0 through 3")
    return slot


class CacheSlots:
    """The server's choice of the engine's cache slot (xeno #49 S7, on upstream PR #175): Claude Code cannot send
    strata_cache_slot, and its side requests (the auto-mode classifier, titles) start from their own prompt, so on
    one slot each of them wiped the main session's cache (#50). A prompt family is its first PREFIX tokens; a family
    keeps its slot, and a new family takes a free slot or the least recently used one.  The default rule
    (STRATA_CACHE_SLOT_RULE unset or "family"); #321's rules "identity" (IdentityCacheSlots) and "continuation"
    (ContinuationCacheSlots) are chosen by that variable."""
    PREFIX = 512

    def __init__(self, n: int):
        self.n = n
        self.families: "collections.OrderedDict[int, int]" = collections.OrderedDict()   # family -> slot, oldest first
        self.lock = threading.Lock()
        self.last: dict = {}                             # family -> its last prompt's ids (for shared())

    def _family(self, ids, who=None):
        """The prompt's family: its first PREFIX tokens (who is not part of it here)."""
        return hash(tuple(ids[:self.PREFIX]))

    def shared(self, ids, who=None):
        """How many leading tokens the prompt shares with the last prompt of its family; None for a new family."""
        prev = self.last.get(self._family(ids, who))
        if prev is None:
            return None
        n = min(len(prev), len(ids))
        return next((i for i in range(n) if prev[i] != ids[i]), n)

    def pick(self, ids, who=None) -> int:
        family = self._family(ids, who)
        with self.lock:
            self.last[family] = list(ids)
            if family in self.families:
                self.families.move_to_end(family)
                return self.families[family]
            used = set(self.families.values())
            free = [s for s in range(self.n) if s not in used]
            if not free:
                gone, slot = self.families.popitem(last=False)
                self.last.pop(gone, None)
            else:
                slot = free[0]
            self.families[family] = slot
            return slot

    def choose(self, ids, nearest=None, who=None):
        """-> (slot, shared, "family"): shared() and then pick(), as Service.with_slot calls them."""
        shared = self.shared(ids, who)
        return self.pick(ids, who), shared, "family"

    def record(self, slot, ids):
        """A client's own strata_cache_slot: this rule never sees it."""


class IdentityCacheSlots(CacheSlots):
    """#321, STRATA_CACHE_SLOT_RULE=identity: the family rule with the family keyed by who sends the request as well
    as by its first PREFIX tokens, so two users of one harness (the same system prompt and tools) are two families
    and never share a slot while a free one exists; with no free slot the least recently used family gives up its
    slot, as in the family rule.  One identity is exactly the family rule.  Who (Service.with_slot's `who`, from
    the HTTP handler's _slot_who): the key's name (team mode; outside it every request is "owner") and the device id
    (X-Strata-Device, else "addr:<address>"; Claude Code sends no device header, so its address decides).  The
    address is the socket peer, or the client a proxy on this PC named in X-Forwarded-For / Cf-Connecting-Ip /
    Forwarded (identity.device_addr): cloudflared names it; a reverse proxy that names no one makes every user
    127.0.0.1, and users behind one NAT address with one key are one identity - the family rule again."""

    def _family(self, ids, who=None):
        return hash((who, tuple(ids[:self.PREFIX])))


_SLOT_TRACE_LOCK = threading.Lock()


def slot_trace(path: str, req: dict, ids, who, route: str, info) -> None:
    """#321: STRATA_SLOT_TRACE=<file.jsonl> (or .jsonl.gz) - one line per request that reaches Service.with_slot, for
    tools/cacheslots_replay.py to replay every slot rule on a real client's stream: the wall time, the route, who
    sends it (the key's NAME, never the key; the device id; the address), the prompt's token ids in full, max_tokens,
    a client's own strata_cache_slot and the engine's cache_slots.  The file holds every prompt's tokens (about 6-7
    bytes a token as JSON, a third of that gzipped).  A write that fails is reported once a request, never raised."""
    rec = {"t": round(time.time(), 3), "route": route or None,
           "who": {k: who.get(k) for k in ("key", "device", "addr")} if isinstance(who, dict) else None,
           "max_tokens": req.get("max_tokens", req.get("max_completion_tokens")),
           "pinned": req.get("strata_cache_slot"), "cache_slots": (info or {}).get("cache_slots") if isinstance(
               info, dict) else None, "n": len(ids), "ids": list(ids)}
    line = json.dumps(rec, separators=(",", ":")) + "\n"
    try:
        with _SLOT_TRACE_LOCK:
            if path.endswith(".gz"):
                with gzip.open(path, "at", encoding="utf-8") as f:
                    f.write(line)
            else:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
    except OSError as e:
        print(f"[strata] STRATA_SLOT_TRACE {path}: {e}", flush=True)


def _shared_len(a, b, step=4096) -> int:
    """How many leading tokens two id lists share: compared a block at a time (list slices compare in C), then one
    by one inside the first block that differs."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i:i + step] == b[i:i + step]:
        i += step
    if i >= n:
        return n
    end = min(i + step, n)
    while i < end and a[i] == b[i]:
        i += 1
    return i


class ContinuationCacheSlots:
    """#321, STRATA_CACHE_SLOT_RULE=continuation (opt-in; the default is the family rule, CacheSlots).  The server's
    choice of the engine's cache slot by continuation.  Each request goes to the engine as cache_slot=N; the slot holds the KV of its last prompt, and a prompt that
    lands in a slot reuses what it shares with that slot's last prompt.  The table keeps each slot's last prompt ids
    and when it was last used, nothing per conversation:

    1. Continuation: the slot whose last prompt this one continues - they share at least PREFIX tokens and all of the
       last one but at most its last TAIL tokens; several such slots: the longest shared prefix.
    2. Else a slot never used yet, the lowest first.
    3. Else evict.  A short prompt (under SIDE_MAX tokens: a side request such as a title or a one-shot check)
       evicts the least recently used of the slots whose last prompt is short too, if there is one; a long prompt,
       or a short one with no short slot, evicts the least recently used slot of all.  So a side request takes
       another side request's slot, not a conversation's, and a new conversation takes the slot used longest ago,
       which is a finished session's rather than a live one's.  (Evicting the fewest tokens instead, the first form
       of this rule, kept finished long sessions forever and evicted the live short conversations: a replay of the
       bench's w2a read 20,509 more tokens than w1 with no side request at all, #321.)
       SIDE_MAX: the bench's side requests are ~590-600 tokens and Claude Code's titles a few hundred; the bench's
       shortest first turns are 2,018-2,085 tokens and a Claude Code first turn carries its ~20K-token head, so
       SIDE_MAX (1,024) sits between them; at 2,048 or 4,096 those first turns count as short and a side request
       evicts them (the w2s replay in the #321 report).  Claude Code's auto-mode classifier
       (~30K, its own long root) is a long prompt here: when it is not a continuation it evicts by age.

    A shared head alone is no continuation: two users of one harness share its system prompt and tools (the first
    512 tokens and far more), and under the family rule they shared one slot and re-read each other's conversation
    on every alternated turn (#321, w2a: 452/419 s against 270/283 s with a slot per user).

    TAIL: the server renders the previous turns as they were (the template keeps every assistant turn's thinking -
    serve/ never sets preserve_thinking false - and a reasoning carry adds the cut reply after the cut prompt), so a
    conversation's next prompt repeats its last one up to the generation prompt's few tokens
    (test_consecutive_turns_through_the_template_part_within_TAIL).  What does not repeat is a transcript that grows
    inside one message, the classifier's (the _ckpt_at comment in Service.with_slot): its last prompt's text after
    the transcript (the closing instructions).  Its size was not measured; 1,024 tokens is the allowance, and the
    STRATA_DEBUG line names the nearest slot's tail for every prompt that is not a continuation, so one live session
    measures it.  TAIL is also the window of a false continuation: two conversations whose own parts are both under
    TAIL tokens past a shared head (two users' first short turns) share a slot, re-reading at most TAIL tokens a
    turn, until one grows past it.  A prompt under PREFIX tokens is never a continuation, not even an exact repeat.

    The trade-off accepted: a rewind or a compaction - a prompt that parts from its own conversation mid-way, well
    before its last TAIL tokens - is not a continuation.  It takes a free slot or evicts per 3, and is read from 0
    there (or from what it shares with the evicted slot's prompt) instead of from the point where it parted; the
    conversation it left keeps its slot until evicted.

    A client's own strata_cache_slot is used as sent (Service.with_slot) and recorded here (record): its prompt
    becomes that slot's last one, so the rule knows the slot is taken."""
    PREFIX = 512
    TAIL = 1024
    SIDE_MAX = 1024

    def __init__(self, n: int):
        self.n = n
        self.lock = threading.Lock()
        self.table: list = [None] * n                    # slot -> {"ids": its last prompt, "used": tick}, or None
        self.tick = 0

    def _evict(self, ids) -> int:
        """Rule 3 (every slot is in use): a short prompt takes the least recently used short slot, if any; else the
        least recently used slot of all."""
        slots = range(self.n)
        if len(ids) < self.SIDE_MAX:
            short = [s for s in slots if len(self.table[s]["ids"]) < self.SIDE_MAX]
            slots = short or slots
        return min(slots, key=lambda s: self.table[s]["used"])

    def _scan(self, ids):
        """-> (continued slot or None, its shared length, nearest slot, its shared length): one pass over the table."""
        best = best_sh = near = near_sh = None
        for s, e in enumerate(self.table):
            if e is None:
                continue
            last = e["ids"]
            sh = _shared_len(last, ids)
            if near is None or sh > near_sh:
                near, near_sh = s, sh
            if sh >= self.PREFIX and sh >= len(last) - self.TAIL and (best is None or sh > best_sh):
                best, best_sh = s, sh
        return best, best_sh, near, near_sh

    def _put(self, slot, ids):
        self.tick += 1
        self.table[slot] = {"ids": list(ids), "used": self.tick}

    def choose(self, ids, nearest=None, who=None):
        """-> (slot, shared, how): how is "continues" (shared = the prefix shared with the slot's last prompt),
        "free" or "evicts" (shared None).  `nearest` (a dict) is filled with the slot sharing the most and its tail
        (its last prompt's tokens past the shared prefix), for the STRATA_DEBUG line."""
        ids = ids if isinstance(ids, list) else list(ids)   # _shared_len compares list slices (never equal to tuples)
        with self.lock:
            best, best_sh, near, near_sh = self._scan(ids)
            if nearest is not None and near is not None:
                nearest.update(slot=near, shared=near_sh, tail=len(self.table[near]["ids"]) - near_sh)
            if best is not None:
                slot, shared, how = best, best_sh, "continues"
            else:
                shared = None
                free = [s for s in range(self.n) if self.table[s] is None]
                if free:
                    slot, how = free[0], "free"
                else:
                    slot, how = self._evict(ids), "evicts"
            self._put(slot, ids)
            return slot, shared, how

    def record(self, slot, ids):
        """A request whose client chose its slot: that slot's last prompt is now this one."""
        with self.lock:
            self._put(slot, ids)

    def shared(self, ids, who=None):
        """How many leading tokens the prompt shares with the last prompt of the slot it continues; None if none."""
        ids = ids if isinstance(ids, list) else list(ids)
        with self.lock:
            best, best_sh, _, _ = self._scan(ids)
        return best_sh if best is not None else None

    def pick(self, ids, who=None) -> int:
        return self.choose(ids)[0]


class HybridCacheSlots:
    """#321 round 4, STRATA_CACHE_SLOT_RULE=hybrid (opt-in; the default is the family rule, CacheSlots).  Built from
    what a placement costs: a prompt placed in a slot reuses H, the prefix it shares with the slot's last prompt, now,
    and destroys that prompt's part past H (loss = len(last) - H), which its owner re-reads only if it comes back.
    The table keeps each slot's last prompt, when it was used, and whether a client pinned it; senders play no part.

    1. Join: the slot with the longest H among those with H >= PREFIX and H >= JOIN * len(last), i.e. a certain
       saving at least the possible loss (JOIN = 1/2, the break-even).  A continuation is the case loss ~ 0 ("continues";
       a shared head otherwise, "joins").  So another agent's auto-mode classifier request (28-31K of shared
       instructions, a few thousand tokens of its own transcript, in the #321 trace) shares one slot across agents and
       senders, while a long conversation (its own part past the shared head larger than the head: two users' mains
       on a 9-10K head, a general-purpose subagent sharing 5-7K with main) is never joined by another.  Two
       conversations both under twice their shared head may share a slot; each re-reads less than the head it reuses.
    2. Else the lowest never-used slot.
    3. Else evict: the least recently used slot whose loss is at most EVICT (1) times this prompt's own length - a
       request evicts only what costs no more to read again than itself (a 400-token side request takes another short
       slot, never an idle 100K main conversation); if no slot is that small, the least recently used slot (a finished
       session's, by age, rather than a live short one's).
    A slot a client pinned (strata_cache_slot, recorded) is joined only by an exact continuation of its prompt and
    evicted only when every slot is pinned.

    PREFIX (512, the family rule's head) only keeps a few template tokens from counting as a shared head.  The
    constants are ratios, not sizes, so they hold for other heads and other Claude Code versions; their sensitivity
    (x0.5 / x2) is in the #321 round 4 report.  Known cost: a new conversation after a compaction, or a new session,
    takes a free slot while one exists and reads its head again there (the family rule overwrote the old one, which
    was dead or another user's; it cannot tell which), at most once per never-used slot."""
    PREFIX = 512
    JOIN = 0.5
    EVICT = 1.0

    def __init__(self, n: int):
        self.n = n
        self.lock = threading.Lock()
        self.table: list = [None] * n           # slot -> {"ids", "used": tick, "pinned": bool}, or None
        self.tick = 0

    def _put(self, slot, ids, pinned=False):
        self.tick += 1
        self.table[slot] = {"ids": list(ids), "used": self.tick, "pinned": pinned}   # a copy, as the other rules keep

    def _place(self, ids):
        """-> (slot, H with its last prompt, how, len of its last prompt); the table is not changed."""
        sh = [_shared_len(e["ids"], ids) if e else 0 for e in self.table]
        join = [s for s, e in enumerate(self.table) if e and sh[s] >= self.PREFIX and sh[s] >= self.JOIN * len(e["ids"])
                and (not e["pinned"] or sh[s] >= len(e["ids"]))]
        if join:
            slot = max(join, key=lambda s: (sh[s], -s))
            how = "continues" if sh[slot] >= len(self.table[slot]["ids"]) else "joins"
        elif None in self.table:
            slot, how = self.table.index(None), "free"
        else:
            pool = [s for s in range(self.n) if not self.table[s]["pinned"]] or list(range(self.n))
            small = [s for s in pool if len(self.table[s]["ids"]) - sh[s] <= self.EVICT * len(ids)]
            slot, how = min(small or pool, key=lambda s: self.table[s]["used"]), "evicts"
        last = len(self.table[slot]["ids"]) if self.table[slot] else 0
        return slot, sh[slot], how, last

    def choose(self, ids, nearest=None, who=None):
        """-> (slot, shared, how): how is "continues", "joins", "free" or "evicts"; shared is the prefix shared with
        the slot's last prompt when at least PREFIX (the engine's ckpt_at), else None.  `nearest` (a dict) gets the
        length of the slot's last prompt, for the STRATA_DEBUG line."""
        ids = ids if isinstance(ids, list) else list(ids)
        with self.lock:
            slot, h, how, last = self._place(ids)
            self._put(slot, ids)
        if nearest is not None:
            nearest["last"] = last
        return slot, (h if h >= self.PREFIX else None), how

    def record(self, slot, ids):
        """A request whose client chose its slot: its prompt is that slot's last one, and the slot is pinned."""
        with self.lock:
            self._put(slot, ids, pinned=True)

    def shared(self, ids, who=None):
        ids = ids if isinstance(ids, list) else list(ids)
        with self.lock:
            _, h, how, _ = self._place(ids)
        return h if how in ("continues", "joins") else None

    def pick(self, ids, who=None) -> int:
        return self.choose(ids)[0]


def without_secondary(args: list[str]) -> list[str]:
    """#59: the engine's arguments with the secondary (4070) expert tier off - its size 0, no exclusive placement and
    no adaptive swaps there - so it serves on the primary GPU alone while the display needs that card's VRAM."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a == "--exclusive-secondary-experts":
            i += 1
            continue
        if a in ("--secondary-expert-mib", "--adapt-secondary") and i + 1 < len(args):
            out += [a, "0"]
            i += 2
            continue
        out.append(a)
        i += 1
    return out


# the engine's last words when the 4070 free floor ends it or refuses its start: the runner's monitor and start check
# (secondary_runner.cpp), the arena's open and fill (secondary_arena.cpp)
FLOOR_BREACH = ("display VRAM below configured free floor", "above the free floor", "display free floor")


class ConvCacheLog:
    """#596: the engine's conversation cache as its log tells it (the engine writes "strata serve: conversation
    cache: parked N tokens ...; parked=P bytes=B evictions=E" and "restored N tokens ...; parked=P bytes=B" to stderr,
    which is the log): read on from where it was last read, from the start of the engine's current run."""
    EVENT = re.compile(r"conversation cache: (parked|skipped|restored) (\d+) tokens.*?parked=(\d+) bytes=(\d+)"
                       r"(?: evictions=(\d+))?")
    DROPPED = re.compile(r"conversation cache: dropped \d+ superseded .*?parked=(\d+)")
    READ_MAX = 1 << 20                                  # at most the last MiB of new lines per read

    def __init__(self):
        self.key, self.pos = None, 0
        self.reset()

    def reset(self):
        self.state = {"parked": 0, "bytes": 0, "evictions": 0, "parks": 0, "restores": 0, "last_event": None,
                      "last_tokens": None, "last_at": None}

    def poll(self, path, start) -> dict:
        """The state after the log's new lines; `start` is where the engine's current run began in it."""
        if not path or start is None:
            return dict(self.state)
        if self.key != (path, start):                   # another start of the engine: its cache starts empty
            self.key, self.pos = (path, start), start
            self.reset()
        try:
            size = os.path.getsize(path)
            if size < self.pos:                         # the log was emptied or replaced
                self.pos = 0
            if size - self.pos > self.READ_MAX:
                self.pos = size - self.READ_MAX
            with open(path, "rb") as f:
                f.seek(self.pos)
                data = f.read(size - self.pos)
        except OSError:
            return dict(self.state)
        end = data.rfind(b"\n") + 1                     # whole lines only: the rest is read next time
        self.pos += end
        now = time.time()
        for line in data[:end].decode("utf-8", "replace").splitlines():
            if "conversation cache:" not in line:
                continue
            m = self.EVENT.search(line)
            if m:
                st = self.state
                st["parked"], st["bytes"] = int(m.group(3)), int(m.group(4))
                if m.group(5) is not None:
                    st["evictions"] = int(m.group(5))
                if m.group(1) != "skipped":
                    st["parks" if m.group(1) == "parked" else "restores"] += 1
                    st["last_event"], st["last_tokens"], st["last_at"] = m.group(1), int(m.group(2)), now
                continue
            m = self.DROPPED.search(line)
            if m:
                self.state["parked"] = int(m.group(1))
        return dict(self.state)


def conversation_cache_view(info: dict, hist: list, totals: dict, parked: dict) -> dict:
    """#596: the Monitor's Conversation cache card: the parked conversations (the engine's opt-in
    --conversation-cache-mib: budget, slots, what its log says) and how much of the prompts the cache gave back."""
    mib = info.get("conversation_cache_mib")
    last = hist[-1] if hist else None
    return {"enabled": isinstance(mib, int) and mib > 0, "budget_mib": mib if isinstance(mib, int) else None,
            "slots": info.get("conversation_cache_slots"), **parked,
            "requests": len(hist), "requests_reused": sum(1 for r in hist if (r.get("reused") or 0) > 0),
            "reused_tokens": totals.get("reused", 0), "prompt_tokens": totals.get("prompt_tokens", 0),
            "last_reused": last.get("reused") if last else None,
            "last_prompt": last.get("prompt_tokens") if last else None}


_BTRACE = bool(os.environ.get("STRATA_BATCH_TRACE"))


def btrace(*a):
    if _BTRACE:
        print("[batch-trace]", threading.get_ident() % 10000, *a, file=sys.stderr, flush=True)


EOS_IDS = {248044, 248046}   # <|endoftext|>, <|im_end|>: the engine's default --eos-ids


# The engine's own words for a verifier that cannot go on (its GPU waits were released, or an earlier window never
# finished, #267): the process may stay alive and even print DONE, but it must not take another request.
FATAL_PREFIXES = ("ERR verify: timed out at layer ", "ERR verify batch: timed out at layer ",
                  "ERR verify: an earlier window never finished", "ERR verify batch: an earlier window never finished")


class StrataEngine:
    """The resident engine: `strata --serve` reads `GEN <max_new> <ids>` lines and streams `T <id>` lines, then
    `DONE ...`.  Requests are serialized by the service's FIFO, so one pipe is enough.

    Per-request sampling rides the same line as engine-side keys between max_new and the ids
    (`temperature=F top_p=F top_k=N seed=N`, the engine's own spelling).  An absent temperature keeps the
    engine's default, which is greedy; `temperature=0` means the same thing, so it is not forwarded.
    """
    QUIET_S = 10          # how long a quiet engine waits for a line before a heartbeat and a liveness check
    silence_s = ENGINE_SILENCE_S         # #481: main() sets the config's engine_silence_s (survives restart())
    silent_note = None                   # #481: why the server ended a silent engine (death_note says it)
    gpu_busy = None                      # #1317: () -> bool, "the GPU is working" (the telemetry's reading); a GPU at work is not frozen
    last_err = None                      # #997: an ERR that was its last stdout line (death_note says it)
    batch = 0                            # --batch: the engine's batch slots (0: one request at a time)

    # `last`: the figures of the engine's last DONE line.  With batch slots several requests run at once, each on its
    # own thread, so each one reads its own request's figures (Service.run records them); one at a time, the one dict.
    @property
    def last(self):
        if self.batch:
            tl = self.__dict__.setdefault("_tl", threading.local())
            if not hasattr(tl, "last"):
                tl.last = {}
            return tl.last
        return self.__dict__.get("_last", {})

    @last.setter
    def last(self, value):
        self.__dict__["_last"] = value
        if self.batch:
            tl = self.__dict__.setdefault("_tl", threading.local())
            tl.last = value

    def __init__(self, exe: str, args: list[str], cwd: str | None = None, log: str | None = None,
                 env: dict | None = None, lazy: bool = False, cpus: list[int] | None = None):
        self.spawn = (exe, list(args), cwd, log, env, False, cpus)   # to start it again after it died (issue #27)
        self.session_save_reclaim = '--session-save-reclaim' in args
        paths = {k: v for k, v in zip(args, args[1:]) if k in ("--native", "--pack")}
        self.model_path = paths.get("--native") or paths.get("--pack", "pack/full")
        self.log_path = log
        self.proc, self.pump, self.log = None, None, None
        self.ended, self.unloaded = True, True
        self.max_context = int(args[args.index("--max-context") + 1]) if "--max-context" in args else 4096
        self.can_stop = False            # the engine honours a STOP line mid-request (READY <ctx> stop)
        self.last = {}
        self.info = {}                   # INFO key=value facts (engine 0.1.8+): kv, expert slots, ... (Monitor tab)
        self.prefill_tok_s_mean = None
        self.progress = None             # (read, total) prompt tokens while a prompt is read, from PP lines
        self.progress_ms = 0             # PP's own milliseconds since the prompt started (its third field)
        self.reused = 0                  # RESUME: prompt tokens not read again (a client takes them out of the work)
        self.silent_note = None
        self.last_err = None
        try:                            # a ready-made engine's BUILD.json says its version
            self.info["version"] = json.loads((Path(exe).parent / "BUILD.json").read_text()).get("version")
        except (OSError, ValueError):
            self.info["version"] = None
        if lazy:
            return
        self.unloaded = False            # `ended` stays True until READY (below): not alive while starting (#344)
        self.log = open(log, "a", encoding="utf-8") if log else subprocess.DEVNULL
        loading = threading.Event()                     # set once READY: the narrator below stops
        log_start = os.path.getsize(log) if log else 0  # where this start's lines begin (start_failure_hint)
        self.log_start = log_start                      # #596: the Monitor's conversation cache reads from here
        if log:
            threading.Thread(target=narrate_start, args=(log, os.path.getsize(log), args, loading),
                             daemon=True).start()
            # once per log: the follower keeps reading the same (appended) log across restarts and reloads
            if os.environ.get("STRATA_REQUEST_LINES") and os.path.abspath(log) not in _echoing:
                _echoing.add(os.path.abspath(log))
                threading.Thread(target=echo_requests, args=(log, os.path.getsize(log)), daemon=True).start()
        if log:                                          # #779: when an engine started (a crash report has the time)
            try:
                self.log.write(f"[strata] {time.strftime('%Y-%m-%d %H:%M:%S')} engine started: {exe} --serve "
                               f"{' '.join(args)}\n")
                self.log.flush()
            except OSError:
                pass
        self.max_context = 0                             # before the new process is visible: never its predecessor's
        pin = {}
        if cpus and hasattr(os, "sched_setaffinity"):    # replicas: this engine's own physical cores (serve/replicas.py)
            pin["preexec_fn"] = lambda: os.sched_setaffinity(0, cpus)
        self.proc = popen("the Strata engine", [exe, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=self.log, text=True, encoding="utf-8", bufsize=1, env=env, **pin)
        contain(self.proc)                               # ends with the server, however it ends (Windows)
        # the READY read has a timeout (#1317): the lines come from a thread, up to READY, so the rest of the stream
        # stays for _pump (one reader at a time)
        ready_q: queue.Queue = queue.Queue()
        threading.Thread(target=self._ready_pump, args=(self.proc, ready_q), daemon=True).start()
        ready_deadline = None if ENGINE_READY_S is None else time.monotonic() + ENGINE_READY_S
        timed_out = False
        while True:
            left = None if ready_deadline is None else ready_deadline - time.monotonic()
            if left is not None and left <= 0:
                timed_out = True
                break
            try:
                line = ready_q.get(timeout=left)
            except queue.Empty:
                continue
            if line is None:
                break
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = int(v) if v.lstrip("-").isdigit() else v
            if line.startswith("READY"):
                f = line.split()
                self.max_context = int(f[1])
                self.can_stop = "stop" in f[2:]
                break
        loading.set()
        if self.max_context <= 0:
            if timed_out:                               # still loading, or hung: end it, so the turn is not held for good
                try:
                    self.proc.kill()
                except OSError:
                    pass
            try:                                        # its pipes and our handle on its log (the log stays)
                self.proc.wait(timeout=5)
                self.proc.stdin.close()
                self.proc.stdout.close()
                if log:
                    self.log.close()
            except (OSError, subprocess.TimeoutExpired):
                pass
            why = (f"the engine did not report READY within {ENGINE_READY_S:.0f} s" if timed_out else
                   "the engine exited before it was ready")
            # EngineDied, not bare RuntimeError: a request waiting on this load must end with a 503
            # through the request path's handler - a RuntimeError there left the turn with no answer (#1527)
            raise EngineDied(why + (f" (see {log})" if log else "") +
                             start_failure_hint(log, log_start) + start_log_tail(log, log_start))
        self.known_ctx = self.max_context   # survives a failed restart: requests keep their limit and restart it
        # (from PR #41, midhatn) a locally built engine can sit next to another release's BUILD.json: engines that
        # report their own version (INFO engine=, 0.1.8+) win, the manifest stays the fallback for older ones
        if self.info.get("engine"):
            self.info["version"] = str(self.info["engine"])
        self.ended = False                              # READY: alive from here (restart() set it True, #344)
        # the engine's stdout on a thread, so a request can wait with a timeout (heartbeats, cancel checks)
        self.lines: queue.Queue = queue.Queue()
        # --batch: concurrent requests in the engine's batch slots (see generate_batched).  The engine says how many
        # it runs (INFO batch_slots=N: it may have fewer than asked, or none, when they do not fit)
        asked = next((int(args[args.index(k) + 1]) for k in ("--batch", "--slots") if k in args), 0)
        self.batch = int(self.info.get("batch_slots") or 0)
        if asked and self.batch != asked:
            print(f"[strata] parallel requests: {asked} asked, the engine runs {self.batch or 'one at a time'} "
                  "(its log says why)", flush=True)
        _bg = args[args.index("--batch-groups") + 1] if "--batch-groups" in args else "default"
        # no `--batch-groups` (the default on a layer split) or `auto`: the engine picks the groups and says so
        # (INFO batch_groups=G); a number is the group count asked for (1 = off)
        groups = int(self.info.get("batch_groups") or 1) if _bg in ("auto", "default") else int(_bg)
        groups = groups if self.batch and groups > 0 and self.batch % groups == 0 else 1
        gs = self.batch // groups if self.batch else 0
        # slots in the order that spreads requests over the pipeline's groups first: 0, gs, 2gs, .., 1, gs+1, ..
        self.slot_order = [g * gs + t for t in range(gs) for g in range(groups)]
        self.slot_group = [b // gs if gs else 0 for b in range(self.batch)]   # each slot's pipeline group
        self.slot_groups = groups
        # pipelined groups (the engine says how many it runs: INFO batch_groups; older engines: the args).  A pipelined
        # group's window spans its slots up to its last busy one, and the engine keeps no pipelined slot as a
        # conversation cache - see pick_slot
        groups_run = int(self.info.get("batch_groups") or groups)
        self.slot_gs = gs if groups > 1 and groups_run > 1 else 0
        self.slot_q = [queue.Queue() for _ in range(self.batch)]
        self.slot_busy = [False] * self.batch
        # #1603: who owns a busy flag.  slot_claims: slots a request of this server picked or reserved and has not
        # given back; slot_releasing: slots handed to the background BDONE wait.  A busy slot in neither has no owner
        # (its flag leaked) and would keep every later request waiting for a slot that nothing will free.
        self.slot_claims: set[int] = set()
        self.slot_releasing: set[int] = set()
        self._orphan_seen: dict[int, float] = {}
        # what each slot's sessions hold (prompt + every token a window fed), so a conversation's next turn goes to
        # the slot that has its start (the engine checks it again); when the slot was last used
        self.slot_held: list[list[int]] = [[] for _ in range(self.batch)]
        self.slot_used = [0.0] * self.batch
        self.slot_live: list[dict | None] = [None] * self.batch   # /metrics: the request in each slot
        # #1012: the admission state belongs to the server, not to one engine process.  restart() runs this again
        # while requests still wait on it (they hold these very objects), so it is made once and kept: new ones
        # would leave the waiters on a lock and a condition nobody notifies, with their counts missing from the
        # new lists.
        if "slot_cv" not in self.__dict__:
            self.slot_cv = threading.Condition()
            self.waiting = 0                            # requests waiting for the control lines (ctl)
            self.admitting = 0                          # #1603: ... and those that gave the lines back to wait for a slot
            self.wait_lens: list[list[int]] = []        # ... their prompt lengths (a long read gives way to short ones)
            self.ctl_epoch = 0                          # how often the control lines were taken
            self.ctl = threading.Lock()                 # one admission or solo request on the control lines at a time
        self.gen = self.__dict__.get("gen", 0) + 1      # which engine process this is (a request notes its own)
        self._ctl_erred = False                         # #1059: the control lines ended on an ERR
        self._yielded = None                            # (slot, tokens read): the last request on them gave way
        self.wlock = threading.Lock()                   # stdin writes from several request threads
        # each pump owns its process and queue, which a restart replaces (xeno #49 review, #344); close() joins it
        self.pump = threading.Thread(target=self._pump, args=(self.proc, self.lines), daemon=True)
        self.pump.start()

    def _pump(self, proc=None, lines=None):
        if proc is None:                                # no arguments (upstream's callers): this process's
            proc, lines = self.proc, self.lines
        slot_q = getattr(self, "slot_q", [])            # (none on an engine made without __init__)
        line = None
        for line in proc.stdout:
            # checked before batch routing: a fatal line is never a slot's own
            if line.startswith(FATAL_PREFIXES):
                # release_gpu_waits invalidates the verifier, even if the native
                # process stays alive and prints DONE afterwards.
                if self.proc is proc:
                    self.silent_note = "Unrecoverable native verification failure: " + line[4:].strip()
                    self.ended = True
                try:
                    proc.terminate()
                except OSError:
                    pass
                break                           # publish EOF, never the trailing DONE
            if line.startswith(("BT ", "BDONE ")) and slot_q:   # --batch: a batch slot's own lines
                try:
                    slot_q[int(line.split()[1])].put(line)
                    continue
                except (IndexError, ValueError):
                    pass
            lines.put(line)
        if self.proc is proc:                           # a pump left over from before restart() must not mark its
            self.ended = True                           # successor dead; its output closed: it is gone, even before
                                                        # the OS says so (set first: whoever reads None sees it)
            if line and line.startswith("ERR"):         # #997 #890: why it exited, though no request may read it
                self.last_err = line[4:].strip()        # (a failed batch window: every request reads its slot)
        lines.put(None)
        for q in slot_q:
            q.put(None)

    def _log_tail(self) -> str:
        """The last 4 KiB of the engine's log ("" without one)."""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                return f.read().decode("utf-8", "replace")
        except (OSError, TypeError):
            return ""

    def death_note(self) -> str:
        """Why the engine most likely ended, from the end of its log: its own watchdog (issue #29), else RAM."""
        if getattr(self, "silent_note", None):          # #481: the server ended it, not the OS or the engine itself
            return self.silent_note
        tail = wide = ""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 262144))
                wide = f.read().decode("utf-8", "replace")
            tail = wide[-4096:]
        except (OSError, TypeError):
            pass
        # #1705: STRATA_VERIFY_TRACE dumps are ~30 KB and push the watchdog's line out of a 4 KB tail
        for line in reversed(wide.splitlines()):
            if "issue #29" in line:
                return ("The engine stopped itself because it had stopped making progress - a hang it caught. Its log "
                        "line: " + line.strip() + " - please report it at github.com/Niko1221/Strata/issues.")
        proc = self.proc                                 # None while a restart has not started the next one yet
        rc = proc.poll() if proc is not None else None
        if rc is not None and rc >= 0 and self.last_err:  # its own last words on stdout: they say why
            return (f"The engine exited (code {rc}) after it reported: {self.last_err} - please report it at "
                    "github.com/Niko1221/Strata/issues with the log.")
        last = next((x.strip() for x in reversed(tail.splitlines()) if x.strip().startswith(("strata", "ERR"))), "")
        if rc is not None and rc >= 0 and last:          # it ended by itself: its own last words say why (#215)
            return (f"The engine exited (code {rc}). Its last log line: {last} - if that does not explain it, please "
                    "report it at github.com/Niko1221/Strata/issues with the log.")
        return ("The usual cause is running out of RAM: Linux then ends the biggest program (check: sudo dmesg | "
                "grep -i -E 'killed process|out of memory'); Windows slows down instead. Close other programs or use a "
                "smaller model (Q2_0 / IQ2_XS).")

    def alive(self) -> bool:
        proc = self.proc                                 # read once: restart() / close() may set it to None meanwhile
        return proc is not None and not getattr(self, "ended", False) and proc.poll() is None

    def exit_code(self):
        if self.proc is None:
            return None
        try:
            return self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return None

    def unload(self):
        """Release GPU/RAM between requests, retaining the spawn config for automatic reloading."""
        self.close()
        self.unloaded = True

    def floor_breach(self) -> bool:
        """#59: the engine's last words (the end of its log) are a 4070 free-floor terminate or start refusal."""
        last = [x for x in self._log_tail().splitlines() if x.strip()][-5:]
        return any(m in x for x in last for m in FLOOR_BREACH)

    def secondary_need_mib(self) -> int | None:
        """The 4070 VRAM the configured tier takes plus its floor (what the display must have free to restore it)."""
        args = getattr(self, "full_spawn", self.spawn)[1]
        try:
            tier = int(args[args.index("--secondary-expert-mib") + 1])
            floor = int(args[args.index("--secondary-free-floor-mib") + 1]) if "--secondary-free-floor-mib" in args else 0
        except (ValueError, IndexError):
            return None
        return tier + floor + 512 if tier > 0 else None

    RESTART_RETRY_S = 15.0   # between the tries of restart(): a dying engine's VRAM may take a while to come back

    @staticmethod
    def _ready_pump(proc, out: queue.Queue) -> None:
        """The engine's first lines up to READY, for a read that can time out; None when its output closed first."""
        done = False
        try:
            for line in proc.stdout:
                out.put(line)
                if line.startswith("READY"):
                    done = True
                    return
        except (OSError, ValueError):                   # the pipe closed under us (the engine was killed)
            pass
        finally:
            if not done:
                out.put(None)

    def restart(self, tries: int = 3, full: bool | None = None):
        """Start the engine again after it died; the new process has its own line queue.  close() ends and waits for
        the old process first (EngineStuck when it cannot be ended).  A start that still exits before READY - a dead
        engine's VRAM can take a while to come back, notably on ROCm - is retried (PR #637).  #59: after a 4070
        free-floor terminate (or a start the floor refuses) it comes back without the 4070 tier (`degraded`);
        `full=True` restores the configured command; if that start fails, it comes back degraded again rather than
        leaving no engine."""
        info = dict(self.info)
        self.starting = True                     # prepare() answers 503 "starting" meanwhile (#344)
        try:
            self.close()                         # #333: QUIT first, so a running engine frees its memory
            base = getattr(self, "full_spawn", self.spawn)
            self.full_spawn = base               # __init__ stores the args it starts with: keep the configured ones
            exe, args, cwd, log, env, *rest = base      # upstream's spawn also keeps lazy and the CPU set
            want_full = (not self.floor_breach()) if full is None else full
            if args == without_secondary(args):
                want_full = True                 # no tier configured: nothing to drop
            for i in range(tries):
                # #344: not alive until READY - __init__ sets max_context to 0 and blocks until the engine says
                # READY, and a request that saw alive() in that window skipped load() and failed with "context
                # (0)".  __init__ clears `ended` itself once READY (before its pump thread can set it again).
                self.ended = True
                try:
                    self.__init__(exe, args if want_full else without_secondary(args), cwd, log, env, *rest)
                    break
                except RuntimeError:
                    if want_full and (full is True or self.floor_breach()):
                        print("[strata] " + ("the display needs the 4070's VRAM" if self.floor_breach() else
                                             "the start with the 4070 tier failed (see the log)") +
                              ": starting without the 4070 tier", flush=True)
                        self.close()             # the failed start's process, pipes and log handle
                        self.__init__(exe, without_secondary(args), cwd, log, env, *rest)
                        want_full = False
                        break
                    try:
                        if self.proc is not None:
                            self.proc.wait(timeout=60)
                    except (subprocess.TimeoutExpired, OSError):
                        pass
                    if i == tries - 1:
                        raise
                    print(f"[strata] the engine did not start (try {i + 1} of {tries}); again in "
                          f"{self.RESTART_RETRY_S:g} s", flush=True)
                    time.sleep(self.RESTART_RETRY_S)
            self.full_spawn = base
            self.degraded = not want_full
        finally:
            self.starting = False
            # a lazy first start never reached the post-READY __init__ where slot_cv is made (#1527)
            if "slot_cv" in self.__dict__:
                with self.slot_cv:               # #1012: wake the requests that waited through it: they go on with the
                    self.slot_cv.notify_all()    # new engine, or (it did not start) end with a clean EngineDied
        self.info = {**info, **self.info}

    def _parse_done(self, line):
        f = line.split()
        self.last = {"generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                     "decode_ms": float(f[4]), "finish": f[5]}
        if len(f) >= 9:                                   # the conversation cache's fields (engine 0.1.3+)
            self.last.update(drafts_accepted=int(f[6]), drafts_offered=int(f[7]), reused=int(f[8]))
        if len(f) >= 11:                                  # decode hit rate fields
            self.last.update(hits=int(f[9]), lookups=int(f[10]))
        if len(f) >= 14:                                  # the expert tiers (engine 0.1.31+): RAM / file blobs, file MB
            self.last.update(ram_blobs=int(f[11]), file_blobs=int(f[12]), file_mb=float(f[13]))
        if len(f) >= 15:                                  # #471 (engine 0.1.36+): the prompt tokens actually read
            self.last.update(prompt_read=int(f[14]))
        if len(f) >= 16:                                  # #588 (engine 0.1.39+): routed experts read over PCIe
            self.last.update(offloaded=int(f[15]))
        if getattr(self, "_stats", None):                 # the STATS line just before it (xeno UI S4)
            self.last["stats"], self._stats = self._stats, None
        if getattr(self, "_pp", None):                    # and the PP lines of its prompt read
            self.last["prefill_points"], self._pp = self._pp, []

    def _parse_stats(self, line):
        """`STATS key=value ...` (xeno UI S4): this request's decode counters, printed by the engine just before its
        DONE. Numbers only; a malformed line is skipped, never fatal."""
        stats = {}
        for kv in line.split()[1:]:
            k, _, v = kv.partition("=")
            try:
                stats[k] = int(v) if v.lstrip("-").isdigit() else float(v)
            except ValueError:
                return
        self._stats = stats or None

    def vram(self, reserve_mib: int | None, timeout: float = 120.0) -> dict:
        """#533: `VRAM <reserve_mib>` between requests (the caller holds the service's FIFO): the engine shrinks its
        expert cache until that much VRAM is free, or grows it back when more is free; None: back to the reserve it
        started with.  -> the engine's figures (expert_slots, expert_cache_mib, vram_free_mib, ...); raises ValueError
        with the engine's reason (e.g. it was not started with --vram-elastic), EngineDied when it ended."""
        if not self.alive():
            raise EngineDied("the engine is not running")
        self.proc.stdin.write("VRAM" + ("" if reserve_mib is None else f" {int(reserve_mib)}") + "\n")
        self.proc.stdin.flush()
        deadline = time.time() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                raise EngineDied("the engine did not answer the VRAM command") from None
            if line is None:
                raise EngineDied("the engine ended")
            line = line.strip()
            if line.startswith("ERR"):
                raise ValueError(line[4:].strip() or "the engine refused the VRAM command")
            if line.startswith("VRAM "):
                out = {}
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    out[k] = int(v) if v.lstrip("-").isdigit() else v
                self.info.update({k: out[k] for k in ("expert_slots", "vram_free_mib") if k in out})
                self.info["vram"] = out
                return out
            if time.time() > deadline:
                raise EngineDied("the engine did not answer the VRAM command")

    @staticmethod
    def sampling_keys(sampling: dict) -> str:
        keys = f" cache_slot={cache_slot(sampling)}"
        t = sampling.get("temperature")
        if isinstance(t, (int, float)) and float(t) > 0.0:
            keys += f" temperature={float(t)!r}"
        tp = sampling.get("top_p")
        if isinstance(tp, (int, float)) and float(tp) < 1.0:
            keys += f" top_p={float(tp)!r}"
        tk = sampling.get("top_k")
        if isinstance(tk, int) and not isinstance(tk, bool) and tk >= 0:
            # the engine's sampled path keeps at most 64 candidates: 0 ("off") and wider lists get all 64
            keys += f" top_k={tk if 1 <= tk <= 64 else 64}"
        mp = sampling.get("min_p")
        if isinstance(mp, (int, float)) and 0.0 < float(mp) <= 1.0:
            keys += f" min_p={float(mp)!r}"
        rp = sampling.get("repetition_penalty")
        rp_on = isinstance(rp, (int, float)) and float(rp) != 1.0
        pf = sampling.get("frequency_penalty")
        pf_on = isinstance(pf, (int, float)) and float(pf) != 0.0
        pp = sampling.get("presence_penalty")
        pp_on = isinstance(pp, (int, float)) and float(pp) != 0.0
        if rp_on:
            keys += f" penalty_repeat={float(rp)!r}"
        if pf_on:
            keys += f" penalty_freq={float(pf)!r}"
        if pp_on:
            keys += f" penalty_present={float(pp)!r}"
        if rp_on or pf_on or pp_on:
            # a penalty without a window counts over nothing: the engine's default is the last 64 tokens
            pln = sampling.get("penalty_last_n")
            if isinstance(pln, int) and not isinstance(pln, bool) and pln > 0:
                keys += f" penalty_last_n={pln}"
            else:
                keys += " penalty_last_n=64"
        seed = sampling.get("seed")
        if isinstance(seed, int) and seed > 0:
            keys += f" seed={seed}"
        # setup's calibration (tools/calibrate.py): engine settings for this request only, measured without a restart
        tune = sampling.get("strata_tune")
        if isinstance(tune, dict):
            for k in ("pcie_frac", "spec_min_p"):
                v = tune.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= float(v) <= 1.0:
                    keys += f" {k}={float(v)!r}"
            # --aux-cpus for this request: 1 puts the engine's threads that are neither pool workers nor the host on the
            # spare CPUs, 0 leaves them where they were created (true/false or 1/0; the engine started without a spare CPU
            # ignores a 1).  Anything else is not sent.
            ac = tune.get("aux_cpus")
            if isinstance(ac, bool) or (isinstance(ac, (int, float)) and ac in (0, 1)):
                keys += f" aux_cpus={int(ac)}"
        ck = sampling.get("_ckpt_at")                   # xeno #49 S7 follow-up: an extra prompt checkpoint
        if isinstance(ck, int) and not isinstance(ck, bool) and ck > 0:
            keys += f" ckpt_at={ck}"
        # "strata_checkpoint": false - a one-shot call (a classification, a probe) whose turn no later request
        # extends: no conversation checkpoint for it (#830).  It still reuses a cached prefix.  Absent = as before.
        if sampling.get("strata_checkpoint") is False:
            keys += " ckpt=0"
        # "strata_prefix" (resolved by Service.resolve_prefix to {"tokens": N}): the first N prompt tokens are a shared
        # prefix - the engine pins the checkpoint there.  Absent or unresolved = as before.
        prefix = sampling.get("strata_prefix")
        if isinstance(prefix, dict):
            pn = prefix.get("tokens")
            if isinstance(pn, int) and not isinstance(pn, bool) and pn > 0:
                keys += f" pin={pn}"
        return (keys + StrataEngine.ban_key(sampling) + StrataEngine.projection_key(sampling)
                + bias_api.engine_key(sampling.get("logit_bias")))

    @staticmethod
    def ban_key(sampling: dict) -> str:
        """` ban=1` (xeno #49 S4): the engine's --ban-ids list (the CJK guard's Han ids) for this request - on the GEN
        line and on an image request's GENI line alike."""
        return " ban=1" if sampling.get("_ban") is True else ""

    @staticmethod
    def projection_key(sampling: dict) -> str:
        """`cvec=0|1`: the experimental-speed-projection control vector for this request, when the engine was
        started with one (--control-vector-scaled; an engine without one ignores the key).  Absent = on."""
        on = sampling.get("experimental_speed_projection")
        return f" cvec={int(on)}" if isinstance(on, bool) else ""

    def _send(self, text: str):
        btrace("send>", text[:60])
        try:
            with self.wlock:
                self.proc.stdin.write(text + "\n")
                self.proc.stdin.flush()
        except (OSError, AttributeError):                # the pipe is gone (or a restart's close() took the process):
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None

    def _control(self, cancel, on_token, stop_when=None):
        """Reads the control lines of the request on them (GEN / BGEN), yielding None heartbeats.  Calls on_token(id)
        for each `T`; returns ("done", None) at DONE, or ("badm", continues) at BADM (after DONE).  `stop_when()`
        true sends STOP once (the request is then read to its DONE)."""
        stopped = False
        beat = time.monotonic()
        while True:
            try:
                # A client can leave before the first PP/token.  Poll its Event as often as _take_control does;
                # the ten-second HTTP heartbeat is independent of how promptly STOP reaches the engine.
                line = self.lines.get(timeout=0.5)
            except queue.Empty:
                if cancel.is_set() and not stopped:
                    self._send("STOP")
                    stopped = True
                if time.monotonic() - beat >= 10.0:
                    beat = time.monotonic()
                    yield None
                continue
            beat = time.monotonic()
            if line is None:
                raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
            if not line.startswith(("PP ", "INFO")):
                btrace("ctl<", self._ctl_mode, line.strip()[:60])
            if line.startswith("T "):
                on_token(int(line[2:]))
                if not stopped and (cancel.is_set() or (stop_when is not None and stop_when())):
                    self._send("STOP")
                    stopped = True
                yield False                              # a token is pending (not a heartbeat)
            elif line.startswith("PP "):
                f = line.split()
                if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                    self.progress = (int(f[1]), int(f[2]))
                    self.progress_ms = float(f[3]) if len(f) >= 4 else 0.0
                    self.prefill_tok_s_mean = float(f[4]) if len(f) >= 5 else None
                if not stopped and cancel.is_set():
                    # #879: a Stop during the prompt read went out only after the first token (or after 10 s of
                    # silence, which a stream of PP lines never allows): the engine read the whole prompt first
                    self._send("STOP")
                    stopped = True
                yield None
            elif line.startswith("DONE"):
                self._parse_done(line)
                self._last_done = line
            elif line.startswith("BADM "):
                f = line.split()
                self._ctl_result = ("badm", len(f) >= 3 and f[2] == "1")
                return
            elif line.startswith("YIELDED "):            # the read gave way (BYIELD): <slot> <tokens read>
                f = line.split()
                if len(f) >= 3 and f[1].lstrip("-").isdigit() and f[2].isdigit():
                    self._yielded = (int(f[1]), int(f[2]))
            elif line.startswith("ERR"):
                self._ctl_erred = True                   # #1059: the engine refused it and is idle: no DONE follows
                raise ValueError(line[4:].strip())
            if line.startswith("DONE") and self._ctl_mode == "solo":
                self._ctl_result = ("done", None)
                return

    def _drain_control(self, until: str, timeout: float = 300.0, born: int | None = None):
        """After a consumer left early: read the control lines up to the next `until` line (DONE or BADM) so the next
        request does not read this one's leftovers.  Returns that line (None: the engine ended or never answered).
        `born`: the engine process (self.gen) the request started on; once that one is gone there is nothing to
        drain - its end line was read already (by the request's own EngineDied), and waiting for another one held
        a request that had failed for the whole 300 s (#1012), or read the lines of the engine that replaced it."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if born is not None and (self.gen != born or not self.alive()):
                return None
            try:
                line = self.lines.get(timeout=max(0.1, min(1.0, end - time.monotonic())))
            except queue.Empty:
                continue
            if line is None:
                return None
            if line.startswith("DONE"):
                self._parse_done(line)
            if line.startswith(until) or line.startswith("ERR"):
                return line
        return None

    def _release_slot_when_done(self, slot: int, stream: list[int] | None = None):
        """A slot whose consumer left (a stop token, a stop string, a disconnect): BSTOP it and free it once the engine
        says BDONE (in the background).  `stream`: the prompt and every token of it so far - with the tokens still to
        come before BDONE, all but the last are what the slot holds then (the next turn of its conversation)."""
        if slot >= len(self.slot_q):                    # the engine restarted with fewer slots: this one is gone
            return
        try:
            self._send(f"BSTOP {slot}")
        except EngineDied:
            pass
        q, busy, held = self.slot_q[slot], self.slot_busy, self.slot_held   # this process's: restart() replaces them
        releasing = self.slot_releasing
        releasing.add(slot)

        def wait():
            end = time.monotonic() + 600.0
            tail = list(stream or [])
            while time.monotonic() < end:
                try:
                    line = q.get(timeout=5.0)
                except queue.Empty:
                    continue
                if line is None:
                    tail = []
                    break
                if line.startswith("BT "):
                    try:
                        tail.append(int(line.split()[2]))
                    except (IndexError, ValueError):
                        tail = []
                if line.startswith("BDONE "):
                    break
            held[slot] = tail[:-1] if stream and tail else []
            with self.slot_cv:
                busy[slot] = False
                releasing.discard(slot)
                self.slot_cv.notify_all()
        threading.Thread(target=wait, daemon=True).start()

    ORPHAN_SLOT_S = 3.0   # #1603: a busy flag nobody owns this long is a leak and is given back

    def _reap_orphan_slots(self) -> list[int]:
        """#1603: free the slots whose busy flag has no owner (no request holds or reserves them and no BDONE wait is
        running for them).  The caller holds slot_cv.  A flag that leaked kept every later request waiting for a slot
        that nothing would free - with nothing running, nothing queued and nothing for the frozen-engine check to see.
        A slot is only given back after it has looked ownerless for ORPHAN_SLOT_S, so a flag set a moment before its
        owner registers is never taken."""
        now = time.monotonic()
        freed = []
        for b, busy in enumerate(self.slot_busy):
            if busy and b not in self.slot_claims and b not in self.slot_releasing:
                first = self._orphan_seen.setdefault(b, now)
                if now - first >= self.ORPHAN_SLOT_S:
                    self.slot_busy[b] = False
                    self._orphan_seen.pop(b, None)
                    freed.append(b)
            else:
                self._orphan_seen.pop(b, None)
        if freed:
            print(f"[strata] parallel: slot(s) {', '.join(map(str, freed))} were marked busy with no request using them "
                  "(issue #1603); freed, the waiting request goes on", flush=True)
            self.slot_cv.notify_all()
        return freed

    YIELDS_MAX = 2   # #656: how often one request's prompt read gives way to a shorter waiting one

    def _take_control(self, cancel, plen: int, after_epoch: int | None = None):
        """Waits for the control lines (one prompt read at a time), yielding None heartbeats; False when cancelled.
        `after_epoch`: a request whose read gave way lets the requests waiting then go first."""
        entry = [plen]
        with self.slot_cv:
            self.waiting += 1
            self.wait_lens.append(entry)
        beat = time.monotonic()
        try:
            while True:
                if not self.alive() and not getattr(self, "starting", False):
                    # #1012: the engine is gone and nobody is starting it again: this request was not sent, so it
                    # ends now (a clean 503) instead of waiting for control lines nothing will answer on.  While a
                    # restart is under way it keeps waiting and goes on with the new engine.
                    raise EngineDied("the engine stopped while this request waited; it was not sent")
                with self.slot_cv:
                    turn = after_epoch is None or self.ctl_epoch > after_epoch or self.waiting <= 1
                if turn and self.ctl.acquire(timeout=0.5):
                    if self.alive():
                        break
                    self.ctl.release()                  # it died (or is being restarted) while this one was getting
                    turn = False                        # the lines: not the engine this request was meant for
                if not turn:
                    with self.slot_cv:
                        self.slot_cv.wait(timeout=0.5)
                if cancel.is_set():
                    return False
                if time.monotonic() - beat >= 10.0:     # a heartbeat every 10 s, as the engine's own wait
                    beat = time.monotonic()
                    yield None
        finally:
            with self.slot_cv:
                self.waiting = max(0, self.waiting - 1)
                if entry in self.wait_lens:
                    self.wait_lens.remove(entry)
        with self.slot_cv:
            self.ctl_epoch += 1
            self.slot_cv.notify_all()
        return True

    SOLO_AGAIN_MAX = 2      # how often a request left alone in a slot goes back to the solo path
    SOLO_AGAIN_MIN_LEFT = 32  # ... only with at least this many tokens still allowed (max_tokens)

    def _may_go_solo(self, left: int, times: int, embeddings) -> bool:
        """A request decoding in a slot that is alone now (no other slot busy, nobody waiting) goes back to the solo
        path with its MTP drafts: the engine continues it from the slot's sessions (its slot cache; INFO
        slot_cache=1).  STRATA_PARALLEL_SOLO=0 keeps it in the slot."""
        if (embeddings or times >= self.SOLO_AGAIN_MAX or left < self.SOLO_AGAIN_MIN_LEFT or
                not (self.info or {}).get("slot_cache") or os.environ.get("STRATA_PARALLEL_SOLO") == "0"):
            return False
        with self.slot_cv:
            return sum(1 for b in self.slot_busy if b) == 1 and self.waiting == 0

    def _shorter_waiting(self, plen: int) -> bool:
        """#656: someone waits for the control lines with a prompt under half this one's: worth giving way to."""
        with self.slot_cv:
            return any(e[0] * 2 <= plen for e in self.wait_lens)

    def generate_batched(self, ids, max_new, sampling, cancel, embeddings=None):
        """--batch.  Alone (no slot busy, nobody waiting): the solo path (GEN, with drafts: the fastest stream)
        - and when another request arrives meanwhile, this one is STOPped and continues in a batch slot (BGEN with
        its prompt + what it generated: the engine reuses that prefix).  Otherwise: BGEN into a free slot, then the
        slot's own BT lines until BDONE.  Several requests run at once; the control lines (prompt reading, admission)
        are taken one request at a time - and a long prompt read gives way at a chunk boundary to a waiting request
        with a much shorter prompt (#656: `BYIELD <slot>`; the part read waits in a slot and the read goes on after).
        A consumer that stops early leaves the engine in step: the solo request is STOPped and read to its DONE, an
        admission to its BADM, a slot is BSTOPped and freed at its BDONE."""
        if not self.alive():
            raise EngineDied("the engine is unavailable; this request was not sent")
        self.progress, self.progress_ms, self.reused = None, 0, 0
        born = self.gen                                 # the engine process this request is sent to
        keys = self.sampling_keys(sampling or {})
        out: list[int] = []
        pending: list[int] = []
        prompt, left = list(ids), int(max_new)
        reused0 = None      # the first read's reused tokens: a continued leg's count includes this request's own output
        ok = yield from self._take_control(cancel, len(prompt))
        if not ok:
            return
        holding, born = True, self.gen                  # (the engine it now has the control lines of)
        btrace("ctl acquired")
        slot, gen0, reserved = None, None, None
        phase = "none"            # solo -> (admit -> slot) ; "done" once the engine has finished with this request
        stop_sent = False
        yields, solo_again = 0, 0
        admit_wait, admit_since, admit_said = False, 0.0, False   # #1603: waiting for a slot, since when, said
        try:
            while True:   # a request in a slot that is left alone goes back to the solo path
                if not holding:
                    ok = yield from self._take_control(cancel, len(prompt))
                    if not ok:
                        return
                    holding, born = True, self.gen                  # (the engine it now has the control lines of)
                with self.slot_cv:
                    alone = not any(self.slot_busy) and self.waiting == 0
                if alone and left > 1:
                    head = f"GENI {left}{keys} {embeddings}" if embeddings else f"GEN {left}{keys}"
                    self._send(f"{head} {','.join(str(int(t)) for t in prompt)}")
                    phase = "solo"
                    self._ctl_mode, self._ctl_result, self._yielded = "solo", None, None
                    self._ctl_erred = False
                    def others():
                        with self.slot_cv:
                            return self.waiting > 0
                    for x in self._control(cancel, pending.append, stop_when=others):
                        while pending:
                            t = pending.pop(0)
                            out.append(t)
                            yield t
                        if x is None:                       # a heartbeat (False: a token, flushed above)
                            if (reserved is None and not out and not embeddings and yields < self.YIELDS_MAX and
                                    self._shorter_waiting(len(prompt))):
                                with self.slot_cv:          # the slot the part read will wait in
                                    reserved = self.pick_slot(prompt)
                                    if reserved is not None:
                                        self.slot_busy[reserved] = True
                                        self.slot_claims.add(reserved)
                                if reserved is not None:
                                    self._send(f"BYIELD {reserved}")
                            yield None
                    phase = "none"                          # its DONE is read
                    while pending:
                        t = pending.pop(0)
                        out.append(t)
                        yield t
                    if self._yielded is not None and reserved is not None and self._yielded[0] == reserved:
                        slot, reserved = reserved, None     # gave way: the read goes on in that slot (below)
                    elif reserved is not None:              # it did not give way: the slot is free again
                        with self.slot_cv:
                            self.slot_busy[reserved] = False
                            self.slot_claims.discard(reserved)
                            self.slot_cv.notify_all()
                        reserved = None
                    if slot is None:
                        finish = (self.last or {}).get("finish") if isinstance(self.last, dict) else None
                        left = int(max_new) - len(out)
                        if cancel.is_set() or left <= 0 or finish in ("stop", "length") or (out and out[-1] in EOS_IDS):
                            return
                        if reused0 is None:
                            reused0 = (self.last or {}).get("reused")
                        prompt = list(ids) + out            # promoted: it continues in a batch slot from here
                while True:
                    if slot is None:
                        # a free slot (they free themselves at BDONE, which needs no control lines): the one that holds
                        # the start of this prompt (its conversation's last turn), else the one used longest ago.
                        # With none free the control lines are given back while waiting: a slot can be held by a read
                        # that gave way (BYIELD) and needs them to go on, so waiting with them is a deadlock - long,
                        # long, short at 0.25 s steps under parallel: 2 (ENGINE_REVIEW finding 1, reproduced in 0.1.41).
                        while True:
                            with self.slot_cv:
                                slot = self.pick_slot(prompt)
                                if slot is not None:
                                    self.slot_busy[slot] = True
                                    self.slot_claims.add(slot)
                                    if admit_wait:
                                        admit_wait = False
                                        self.admitting = max(0, self.admitting - 1)
                                    break
                                if not admit_wait:          # #1603: counted while it waits for a slot (it holds no
                                    admit_wait, admit_since = True, time.monotonic()   # control lines then)
                                    self.admitting += 1
                            if holding:
                                self.ctl.release()
                                holding = False
                            with self.slot_cv:
                                if self.pick_slot(prompt) is None:
                                    if (ENGINE_STALL_S or 0) > 0 and time.monotonic() - admit_since >= ENGINE_STALL_S:
                                        # the stall check of #1317 only watches a request on the control lines; one
                                        # waiting here is covered by looking for a slot nobody owns, and by saying so
                                        if not admit_said:
                                            admit_said = True
                                            busy = [b for b, v in enumerate(self.slot_busy) if v]
                                            print(f"[strata] parallel: a request has waited {time.monotonic() - admit_since:.0f} s "
                                                  f"for a free slot (busy: {busy}, in use: {sorted(self.slot_claims)}, "
                                                  f"ending: {sorted(self.slot_releasing)}) (issue #1603)", flush=True)
                                    self._reap_orphan_slots()
                                if self.pick_slot(prompt) is None:
                                    self.slot_cv.wait(timeout=1.0)
                            if cancel.is_set():
                                return
                            if not holding:
                                ok = yield from self._take_control(cancel, len(prompt))
                                if not ok:
                                    return
                                holding, born = True, self.gen
                    if self._yielded is not None:           # it gave way: the others waiting then go first
                        self.slot_held[slot] = list(prompt[:self._yielded[1]])
                        self._yielded = None
                        yields += 1
                        self.ctl.release()
                        holding = False
                        with self.slot_cv:
                            epoch = self.ctl_epoch
                        ok = yield from self._take_control(cancel, len(prompt), after_epoch=epoch)
                        if not ok:
                            return
                        holding, born = True, self.gen                  # (the engine it now has the control lines of)
                    while not self.slot_q[slot].empty():
                        self.slot_q[slot].get_nowait()
                    head = f"BGENI {slot} {left}{keys} {embeddings}" if embeddings else f"BGEN {slot} {left}{keys}"
                    live = {"slot": slot, "state": "reading", "prompt_tokens": len(prompt), "generated": 0,
                            "started": time.time(), "first_token": None,
                            "who": ((sampling or {}).get("_meta") or {}).get("who") or "owner"}   # #310
                    self.slot_live[slot] = live
                    self._send(f"{head} {','.join(str(int(t)) for t in prompt)}")
                    self.slot_held[slot] = []               # the admission overwrites what the slot held
                    phase = "admit"
                    self._ctl_mode, self._ctl_result, self._yielded = "batch", None, None
                    self._ctl_erred = False
                    asked = False
                    for x in self._control(cancel, pending.append):
                        while pending:
                            t = pending.pop(0)
                            out.append(t)
                            yield t
                        if x is None:
                            if (not asked and not embeddings and yields < self.YIELDS_MAX and
                                    self._shorter_waiting(len(prompt))):
                                self._send(f"BYIELD {slot}")
                                asked = True
                            yield None
                    cont = bool(self._ctl_result and self._ctl_result[1])
                    phase = "slot" if cont else "none"
                    while pending:
                        t = pending.pop(0)
                        out.append(t)
                        yield t
                    if not cont and self._yielded is not None and self._yielded[0] == slot and not cancel.is_set():
                        continue                            # gave way: again once the shorter request is in
                    break
                self.ctl.release()
                holding = False
                if not cont:
                    return
                live.update(state="decoding", first_token=time.time(), generated=len(out))
                gen0 = len(out) - 1                         # the admission's own token: the slot feeds it first
                going_solo = False
                while True:
                    try:
                        line = self.slot_q[slot].get(timeout=10.0)
                    except queue.Empty:
                        if cancel.is_set() and not stop_sent:
                            self._send(f"BSTOP {slot}")
                            stop_sent = True
                        yield None
                        continue
                    if line is None:
                        phase = "none"
                        raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
                    if line.startswith("BT "):
                        t = int(line.split()[2])
                        out.append(t)
                        live["generated"] = len(out)
                        if cancel.is_set():
                            if not stop_sent:
                                self._send(f"BSTOP {slot}")
                                stop_sent = True
                            continue
                        yield t
                        if not going_solo and not stop_sent and self._may_go_solo(int(max_new) - len(out), solo_again,
                                                                                  embeddings):
                            self._send(f"BSTOP {slot}")         # alone now: on with the drafts (below)
                            going_solo = True
                    elif line.startswith("BDONE "):
                        phase = "none"
                        f = line.split()
                        if len(f) >= 5 and isinstance(self.last, dict):
                            self.last = {**self.last, "finish": f[3], "decode_ms": float(f[4]),
                                         "generated": int(f[2]) if f[2].isdigit() else self.last.get("generated")}
                        # what the slot's sessions hold now: the prompt and every token fed (all but the last one)
                        self.slot_held[slot] = list(prompt) + out[gen0:-1] if len(out) > gen0 else []
                        if (going_solo and f[3:4] == ["cancel"] and not cancel.is_set() and len(out) < int(max_new)
                                and not (out and out[-1] in EOS_IDS)):
                            # the solo path continues it: the engine copies the slot's sessions back (all but the
                            # last token are in them) and decodes with MTP drafts again
                            self.slot_live[slot] = None
                            self.slot_used[slot] = time.time()
                            with self.slot_cv:
                                self.slot_busy[slot] = False
                                self.slot_claims.discard(slot)
                                self.slot_cv.notify_all()
                            slot, gen0 = None, None
                            if reused0 is None:
                                reused0 = (self.last or {}).get("reused")
                            prompt, left = list(ids) + out, int(max_new) - len(out)
                            solo_again += 1
                            btrace("back to the solo path")
                            break
                        return
        finally:
            # a consumer that left early (or an error): keep the engine and this server in step
            btrace("finally phase", phase, "slot", slot, "holding", holding)
            try:
                if phase in ("solo", "admit") and (self.gen != born or not self.alive()):
                    phase = "none"                      # #1012: its engine is gone: nothing to stop or drain
                if phase in ("solo", "admit") and self._ctl_erred:
                    phase = "none"                      # #1059: it ended on the engine's ERR: no DONE to wait 300 s for
                if phase == "solo":
                    self._send("STOP")
                    self._drain_control("DONE", born=born)
                elif phase == "admit":
                    line = self._drain_control("BADM", born=born)
                    if line and line.startswith("BADM ") and line.split()[2:3] == ["1"]:
                        phase = "slot"
            except EngineDied:
                pass
            if reused0 is not None and isinstance(self.last, dict):
                self.last = {**self.last, "reused": reused0}
            if holding:
                self.ctl.release()
            if admit_wait:
                with self.slot_cv:
                    self.admitting = max(0, self.admitting - 1)
            if reserved is not None:
                with self.slot_cv:
                    self.slot_busy[reserved] = False
                    self.slot_claims.discard(reserved)
                    self.slot_cv.notify_all()
            if slot is not None:
                self.slot_live[slot] = None
                self.slot_used[slot] = time.time()
                if phase == "slot":
                    self.slot_held[slot] = []
                    stream = list(prompt) + out[gen0:] if gen0 is not None and len(out) > gen0 else None
                    self._release_slot_when_done(slot, stream)    # freed at its BDONE
                    self.slot_claims.discard(slot)                # (the BDONE wait owns it now)
                else:
                    with self.slot_cv:
                        self.slot_busy[slot] = False
                        self.slot_claims.discard(slot)
                        self.slot_cv.notify_all()

    slot_groups = 1         # --batch-groups (set when the engine starts)
    SLOT_PREFIX_MIN = 512   # a held prefix shorter than this is not worth unbalancing the pipeline's groups

    def pick_slot(self, prompt: list[int]) -> int | None:
        """A free slot for `prompt` (the caller holds slot_cv): the one whose held tokens are the longest start of the
        prompt (the engine then reads only the rest), else an empty one, else the one used longest ago - so the
        conversations other slots hold stay for their next turns.  None: every slot is busy."""
        free = [b for b in self.slot_order if not self.slot_busy[b]]
        if not free:
            return None
        gs = getattr(self, "slot_gs", 0)
        if gs:
            # pipelined groups: the engine keeps no pipelined slot as a conversation cache, so nothing is worth keeping
            # free for a next turn.  A group's window spans its slots up to its last busy one, so first fill a hole
            # below a busy slot of the same group (it costs no extra row), then follow the spreading order
            # (reported on #793: least-recently-used picks left leading holes, [_ B], that cost a whole row each)
            def fills_hole(b):
                g0 = b - b % gs
                return any(self.slot_busy[x] for x in range(b + 1, g0 + gs))
            return min(free, key=lambda b: (not fills_hole(b), self.slot_order.index(b)))
        def held_prefix(b):
            h = self.slot_held[b]
            return len(h) if h and len(h) < len(prompt) and prompt[:len(h)] == h else 0
        best = max(free, key=held_prefix)
        # without --batch-groups a held prefix of any length is taken, as before
        if held_prefix(best) >= (self.SLOT_PREFIX_MIN if self.slot_groups > 1 else 1):
            return best
        # with --batch-groups: the group with the fewest busy slots first, so the requests spread over the
        # pipeline's stages (two requests in one group leave the other stage idle); then empty, then least recent
        busy_in = {}
        for b in range(self.batch):
            if self.slot_busy[b]:
                busy_in[self.slot_group[b]] = busy_in.get(self.slot_group[b], 0) + 1
        return min(free, key=lambda b: (busy_in.get(self.slot_group[b], 0) if self.slot_groups > 1 else 0,
                                        bool(self.slot_held[b]), self.slot_used[b]))

    def slots_view(self) -> list[dict]:
        """/metrics: each batch slot - idle (with the tokens it holds for a next turn), reading or decoding."""
        now, view = time.time(), []
        for b in range(self.batch):
            r = self.slot_live[b]
            if r is None:
                view.append({"slot": b, "state": "idle", "held_tokens": len(self.slot_held[b])})
                continue
            ft = r.get("first_token")
            view.append({"slot": b, "state": r["state"], "prompt_tokens": r["prompt_tokens"],
                         "generated": r["generated"], "elapsed_s": round(now - r["started"], 1),
                         "tok_s": round(r["generated"] / max(1e-6, now - ft), 1) if ft else None,
                         "who": r.get("who")})                  # #310: whose request the slot runs
        return view

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        """Yields token ids, and None as a heartbeat every 10 s while the engine is quiet (reading a long prompt):
        the HTTP layer turns it into an SSE comment, which keeps clients' watchdogs calm and notices a client that
        has gone.  A consumer that stops early (or `cancel`) makes the engine STOP, so it does not run to max_new."""
        if not self.alive():
            raise EngineDied("the engine is unavailable; this request was not sent")
        bias_api.validate_request(sampling or {}, self, None)
        if getattr(self, "batch", 0):
            yield from self.generate_batched(ids, max_new, sampling, cancel, embeddings)
            return
        self.progress, self.progress_ms, self.reused = None, 0, 0
        self._stats = None          # xeno UI S4: this request's STATS line, once the engine sends it
        self._pp = []               # ... and its PP lines (position, ms): the prefill speed per chunk
        self.drained = []           # xeno #49 review: tokens the engine committed after an early stop (read in the drain)
        self.prefill_tok_s_mean = None
        # an image request takes the same sampling keys as text (#75: it used to decode greedily whatever was asked)
        head = f"GENI {int(max_new)}{self.sampling_keys(sampling or {})} {embeddings}" if embeddings else \
            f"GEN {int(max_new)}{self.sampling_keys(sampling or {})}"
        try:
            self.proc.stdin.write(f"{head} {','.join(str(int(t)) for t in ids)}\n")
            self.proc.stdin.flush()
        except OSError:                                  # the pipe is gone: the engine died (not the client)
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None
        done = False
        # #481: how long the engine may stay silent from here.  Until the first line: the request's first prompt
        # chunk at the slowest prompt reading on top of silence_s; a PP line resets it to its own chunk's time.
        silence = float(self.silence_s or 0)
        allow = silence + min(len(ids), PP_CHUNK_MAX) / PP_FLOOR_TOK_S if silence > 0 else 0.0
        heard, read_to = time.monotonic(), 0
        beat = heard
        stall_s = ENGINE_STALL_S or 0.0
        stall_state: dict = {}
        try:
            while True:
                wait = self._wait_s(allow, heard)
                if wait <= 0 and self.proc.poll() is None:   # xeno #103: one that exited is #48's EngineDied, below
                    done = True                       # no STOP and no drain: nothing is listening
                    raise self._silent(f"the engine said nothing for {time.monotonic() - heard:.0f} s during "
                                       "the request")
                try:
                    line = self.lines.get(timeout=min(wait, 0.5))
                except queue.Empty:
                    if self.proc.poll() is not None:     # xeno #48: it exited but its output never closed (a
                        done = self.ended = True         # crashed CUDA process can keep the pipe): fail, not wait
                        raise EngineDied(f"the engine stopped unexpectedly (exit code {self.proc.poll()})")
                    if cancel.is_set():
                        return
                    if stall_s > 0:
                        try:
                            self._stall_check(heard, stall_state, stall_s)
                        except EngineSilent:
                            done = True                   # as above: nothing is listening
                            raise
                    if allow > 0 and time.monotonic() - heard >= allow:
                        continue                          # enforce the expired deadline before another heartbeat
                    if time.monotonic() - beat >= self.QUIET_S:
                        beat = time.monotonic()
                        yield None                        # polling cancellation does not add HTTP heartbeats
                    continue
                if line is None:
                    done = True
                    raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
                heard = time.monotonic()                  # any line is output: T, PP, RESUME, INFO ...
                beat = heard
                if stall_state:
                    stall_state.clear()
                if line.startswith("T "):
                    allow = silence
                    if cancel.is_set():
                        return
                    yield int(line[2:])
                elif line.startswith("PP "):
                    f = line.split()
                    if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                        self.progress = (int(f[1]), int(f[2]))             # prompt progress, one per chunk: also a heartbeat (the
                        self.progress_ms = float(f[3]) if len(f) >= 4 else 0.0
                        try:                                              # xeno UI S4: (position, ms since the read began), per chunk
                            self._pp.append((int(f[1]), float(f[3]))) if len(f) >= 4 else None
                        except ValueError:
                            pass
                        self.prefill_tok_s_mean = float(f[4]) if len(f) >= 5 else None
                        rate, chunk = self.prefill_tok_s_mean or 0.0, int(f[1]) - read_to
                        read_to = int(f[1])
                        if silence > 0 and rate > 0 and chunk > 0:   # #481: the next chunk, as long as this one
                            allow = max(silence, PP_SLACK * chunk / rate)
                    if cancel.is_set():                   # lines reset the 10 s wait, so without this a long prompt
                        return                            # would send no keep-alives at all)
                    yield None
                elif line.startswith("STATS "):
                    self._parse_stats(line)
                elif line.startswith("RESUME "):          # the reused tokens: the first chunk starts after them
                    try:
                        read_to = self.reused = int(line.split()[1])
                    except (IndexError, ValueError):
                        pass
                elif line.startswith("DONE"):
                    self._parse_done(line)
                    done = True
                    return
                elif line.startswith("ERR"):
                    done = True
                    raise ValueError(line[4:].strip())
        finally:
            if not done:                                  # the consumer stopped early: stop the engine, drain to DONE
                if self.can_stop:
                    try:
                        self.proc.stdin.write("STOP\n")
                        self.proc.stdin.flush()
                    except OSError:
                        pass
                # #481: never an untimed wait here - it holds the request FIFO, and an engine that lost step never
                # answers.  An engine that honours STOP gets the current allowance in all (a STOP during a prompt
                # chunk is seen after it); an older one runs on to max_new, so each line only has to come in time.
                heard = time.monotonic()
                while True:
                    try:
                        line = self.lines.get(timeout=self._wait_s(allow, heard))
                    except queue.Empty:
                        if self.proc.poll() is not None:     # xeno #48 in the drain: exited, output never closed
                            self.ended, self.last = True, {}
                            break
                        if self._wait_s(allow, heard) <= 0:
                            self.last = {}                    # no DONE: the previous request's figures are not ours
                            raise self._silent("the engine did not finish the request after it was stopped (STOP) "
                                               f"within {allow:.0f} s") from None
                        continue
                    if line is None or line.startswith("ERR"):
                        self.last = {}                        # no DONE: the previous request's figures are not ours
                        break
                    if line.startswith("T "):
                        self.drained.append(int(line[2:]))
                    elif line.startswith("STATS "):
                        self._parse_stats(line)
                    elif line.startswith("DONE"):
                        self._parse_done(line)
                        break
                    if not self.can_stop:
                        heard = time.monotonic()

    def _wait_s(self, allow: float, heard: float) -> float:
        """xeno #103: the next line's wait - the QUIET_S heartbeat, cut to what is left of the #481 allowance
        (0 = none left; `allow` 0 = no limit)."""
        if allow <= 0:
            return float(self.QUIET_S)
        return max(0.0, min(float(self.QUIET_S), allow - (time.monotonic() - heard)))

    def _activity(self) -> tuple[float, int] | None:
        """#1317: (CPU seconds, I/O bytes) the engine process has used so far, or None when they cannot be read."""
        try:
            import psutil
            pr = psutil.Process(self.proc.pid)
            t = pr.cpu_times()
            io = pr.io_counters()
            return (t.user + t.system, io.read_bytes + io.write_bytes)
        except Exception:  # noqa: BLE001 - psutil missing, the process gone, no permission: no reading
            return None

    def _stall_check(self, heard: float, state: dict, stall_s: float) -> None:
        """#1317: called while the engine is silent.  Takes a baseline soon after the last line, and at stall_s of silence
        compares: no CPU and no I/O in all that time -> the engine is ended (EngineSilent); work going on -> one warning."""
        age = time.monotonic() - heard
        if state.get("base") is None:
            state["base"] = self._activity()
            return
        if age < state.get("next", stall_s) or state["base"] is None:
            return
        now = self._activity()
        try:
            gpu_working = bool(self.gpu_busy()) if self.gpu_busy is not None else False
        except Exception:  # noqa: BLE001 - a telemetry hiccup must not decide anything
            gpu_working = False
        if now is not None and not gpu_working and engine_frozen(state["base"], now):
            raise self._silent(f"the engine said nothing for {age:.0f} s and used no CPU or disk in that time (frozen; "
                               "STRATA_ENGINE_STALL_S sets this, 0 = off)")
        if now is not None:   # working: look again after another window from here, and say so once
            state["base"], state["next"] = now, age + stall_s
            if not state.get("warned"):
                state["warned"] = True
                print(f"[strata] the engine has said nothing for {age:.0f} s but is still working; it is not ended (the "
                      "limit is engine_silence_s). If it is stuck, \"engine_silence_s\" can be set lower.", flush=True)
        else:
            state["next"] = float("inf")   # no reading (psutil missing): nothing to decide on

    def _silent(self, what: str) -> EngineSilent:
        """#481: end an engine that lost step with the server (its main thread waits for a command the server never
        sends), so the next request starts it again: killed now, its GPU and RAM go back with the process."""
        self.silent_note = ("The engine and the server lost step (issue #481; a very slow PC can raise "
                            "\"engine_silence_s\" in the config, 0 = wait forever). If it happens again, please add "
                            "the end of the engine log to github.com/Niko1221/Strata/issues/481.")
        self.ended = True                               # not alive from now: the next request restarts it
        proc = self.proc
        try:
            proc.kill()
            proc.wait(timeout=20)                       # restart() -> close() handles one that is still exiting
        except (OSError, AttributeError, subprocess.TimeoutExpired):
            pass
        return EngineSilent(f"{what}; the server ended the engine")

    def session_file(self, action: str, path: str) -> dict:
        """Disk sessions: `SAVE <path>` / `RESTORE <path>` between requests (the caller holds the service FIFO).
        -> {"tokens", "bytes", "ms"}.  SessionRefused (kind, published) when the engine refused or failed the file and
        is still in step (`SERR <kind> <0|1> <reason>`).  Any line this exchange does not allow - an unknown or
        malformed one, a bare ERR, an answer with negative counts or a non-finite time - means the two sides lost
        step: the engine is ended (EngineDied) and the next request starts it again.  EngineDied also when it ended
        by itself (a restore transfer that failed after the device state changed ends it on purpose: FATAL), and
        EngineSilent (#481) when it printed nothing valid for engine_silence_s - a SESSION progress line (one per
        block of the file moved, at most 16 MiB) counts only when its numbers are.  `SWAIT <phase> <seconds>` announces
        a step that blocks in one call (a flush, the device transfer): until the next line the wait is that step's
        explicit allowance (at most SESSION_WAIT_MAX_S) when it is longer than engine_silence_s - no more, so a step
        that never ends is still ended."""
        if action not in ("save", "restore") or any(c in path for c in "\r\n\0"):
            raise ValueError("invalid session command")
        try:
            self.proc.stdin.write(f"{'SAVE' if action == 'save' else 'RESTORE'} {path}\n")
            self.proc.stdin.flush()
        except OSError:
            raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})") from None
        silence = float(self.silence_s or 0)
        heard = time.monotonic()
        want = "SAVED" if action == "save" else "RESTORED"

        def out_of_step(line: str) -> EngineDied:
            self.ended = True                               # not alive from now: the next request restarts it
            try:
                self.proc.kill()
                self.proc.wait(timeout=20)
            except (OSError, AttributeError, subprocess.TimeoutExpired):
                pass
            return EngineDied(f"the engine answered a session {action} with {line.strip()[:200]!r}; "
                              "the server ended the engine")

        def count(s: str) -> int:
            if not s.isdigit():                             # no sign, no blank, no fraction
                raise ValueError
            return int(s)

        last_done = -1
        allow = 0.0                                         # the announced blocking step's allowance, seconds
        while True:
            limit = max(silence, allow)
            left = limit - (time.monotonic() - heard) if silence > 0 else None
            try:
                if left is not None and left <= 0:
                    raise queue.Empty
                line = self.lines.get(timeout=left)
            except queue.Empty:
                raise self._silent(f"the engine said nothing for {limit:.0f} s during a session {action}"
                                   + (f" (in a blocking step allowed {allow:.0f} s)" if allow > silence else "")) from None
            if line is None:
                raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
            f = line.split()
            head = f[0] if f else ""
            if head == "SESSION":                          # a large file moving: progress, and a heartbeat
                try:
                    if len(f) != 3:
                        raise ValueError
                    done, total = count(f[1]), count(f[2])
                    if done > total or done < last_done:
                        raise ValueError
                except ValueError:
                    raise out_of_step(line) from None
                last_done = done
                heard, allow = time.monotonic(), 0.0
                continue
            if head == "SWAIT":                            # a blocking step starts: its bounded allowance
                try:
                    if len(f) != 3 or not f[1].isalpha() or not f[1].isascii():
                        raise ValueError
                    seconds = count(f[2])
                    if not 0 < seconds <= SESSION_WAIT_MAX_S:
                        raise ValueError
                except ValueError:
                    raise out_of_step(line) from None
                heard, allow = time.monotonic(), float(seconds)
                continue
            if head == "FATAL":
                raise EngineDied(line[6:].strip() or "the engine ended during a session restore")
            if head == "SERR":
                parts = line.rstrip("\r\n").split(" ", 3)
                if len(parts) != 4 or parts[1] not in SessionRefused.STATUS or parts[2] not in ("0", "1"):
                    raise out_of_step(line)
                raise SessionRefused(parts[1], parts[3].strip(), parts[2] == "1")
            if head == want:
                try:
                    if len(f) != 4:
                        raise ValueError
                    tokens, size, ms = count(f[1]), count(f[2]), float(f[3])
                    if not math.isfinite(ms) or ms < 0:
                        raise ValueError
                except ValueError:
                    raise out_of_step(line) from None
                return {"tokens": tokens, "bytes": size, "ms": ms}
            if head == "INFO" or head == "WARN":          # engine log lines may interleave; they are not answers
                heard = time.monotonic()
                continue
            raise out_of_step(line)

    def close(self):
        """End the engine process: QUIT first (the engine frees its memory itself - unpinning tens of GB can take
        a while), then terminate, then kill, each given 20 s.  Raises EngineStuck when it still runs after all three."""
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                try:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                    self.proc.stdin.close()  # Windows' detached stdin reader must see EOF before shutdown
                    self.proc.wait(timeout=20)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    self.proc.terminate()
                    self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                raise EngineStuck("Strata is still releasing GPU/RAM; retry unloading after it exits") from None
        except OSError:
            pass
        finally:
            if self.proc.poll() is not None:
                if self.pump is not None and self.pump.ident is not None:   # a restart can stop before starting it
                    self.pump.join(timeout=2)
                try:
                    self.proc.stdin.close()             # buffered data may flush again after the engine has exited
                except OSError:
                    pass
                # xeno #48: an engine can exit with its stdout never at end-of-file; its pump then sits in a read that
                # holds the pipe's lock, and closing it would wait for ever (restart() calls this since the 0.1.34
                # merge).  That pipe is left to the daemon pump, as the fork's restart() did before #333.
                if self.pump is None or not self.pump.is_alive():
                    self.proc.stdout.close()
                if self.log not in (None, subprocess.DEVNULL):
                    self.log.close()
                self.proc = None
                self.ended = True
                self.progress, self.last = None, {}
                self.progress_ms, self.reused = 0, 0


IMAGE_URL_MAX = 32 << 20        # an image URL is read up to this (the web app attaches pictures of up to 20 MB)
GUEST_IMAGE_URLS = 16           # #311: the image URLs one request of a team guest may name (each is held in memory)


def network_path(path: str) -> bool:
    r"""A path to another computer: \\host\share or //host/share (either slash), \\?\UNC\host\..., \\.\UNC\host\...
    or the NT form \??\UNC\host\...  Windows connects to that host at the first look at the path (os.path.isfile is
    enough) and signs in with the user's NTLM credentials.  \\?\ and \??\ name local files too; no client needs them."""
    p = path.replace("/", "\\")
    return p.startswith("\\\\") or p.startswith("\\??\\")


def vision_start_error(line: str, log=None) -> str:
    """The message when the image encoder answered something other than READY.  #1445: a card in the Exclusive_Process
    compute mode (or one another process holds) refuses the encoder's CUDA context with "busy or unavailable"; that
    cause is named instead of leaving the user the first line of a traceback.  `log` is the encoder's stderr file, whose
    tail is searched too (the CUDA error is printed there)."""
    text = line.strip()
    tail = ""
    try:
        name = getattr(log, "name", None)
        if name:
            with open(name, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 16384))
                tail = f.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        pass
    if "busy or unavailable" in (text + tail).lower():
        return ("the vision encoder did not start: the GPU refused a second CUDA context (\"CUDA-capable device(s) "
                "is/are busy or unavailable\"). This is usually the compute mode Exclusive_Process: only one process "
                "may use the card, and the engine already holds it. Check with: nvidia-smi --query-gpu=compute_mode "
                "--format=csv; set it back with: sudo nvidia-smi -c DEFAULT (or run the encoder on another card). "
                "First line from the encoder: " + (text or "(none)"))
    return "the vision encoder did not start: " + text


class Vision:
    """The resident image encoder: `strata-vision` (llama.cpp mtmd + the mmproj file) reads `ENC <image> <out>`
    lines and writes each image's embeddings; results are cached by the image's hash, so a conversation that
    sends the same picture again (every turn, with most clients) encodes it once."""

    @staticmethod
    def work_dir() -> Path:
        """#480: the directory for the encoder's files.  The engine reads the embeddings file named in a `GENI` line
        and the encoder the output named in an `ENC <image> <out>` line, both split at spaces, so a space in the path
        (a %TEMP% under a user name with a space) broke every image.  The system temp directory is used while its path
        has no space, then the working directory, then the system drive's root; the plain temp directory is the last
        resort (the encoder's own files are then named relative to it, see `_start`)."""
        for parent in (None, Path.cwd(), Path(os.environ.get("SystemDrive", "C:") + os.sep)):
            try:
                d = Path(tempfile.mkdtemp(prefix="strata-vision-", dir=parent))
            except OSError:
                continue                                       # not writable, or no such drive
            if " " not in str(d):
                return d
            try:
                d.rmdir()                                      # just created, so still empty
            except OSError:
                pass
        return Path(tempfile.mkdtemp(prefix="strata-vision-"))

    def __init__(self, cfg: dict, log=None, env: dict | None = None, lazy: bool = False):
        absolute = lambda p: os.path.abspath(p) if os.path.dirname(str(p)) else p   # the encoder runs in its own dir (#480)  # noqa: E731
        args = [absolute(cfg["exe"]), "--mmproj", absolute(cfg["mmproj"]), "--model", absolute(cfg["model"])]
        if cfg.get("gpu"):
            args.append("--gpu")
        if cfg.get("threads"):
            args += ["--threads", str(cfg["threads"])]
        if cfg.get("max_tokens"):
            args += ["--max-tokens", str(cfg["max_tokens"])]
        if cfg.get("min_tokens"):                       # #767: mtmd's image_min_tokens (a hand-edited key)
            args += ["--min-tokens", str(cfg["min_tokens"])]
        self.dir = self.work_dir()
        self.spawn = (args, log, env)                   # to start it again after an unload
        self.proc = None
        self.stopped = True
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[Path, int]] = {}
        if not lazy:
            self._start()

    def _start(self):
        args, log, env = self.spawn
        # cwd is the encoder's directory: the ENC lines name the files relative to it (#480: no space in the line)
        self.proc = popen("the image encoder", args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=log or subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1, env=env,
                          cwd=self.dir)
        contain(self.proc)
        try:
            line = self._readline(VISION_READY_S, "the vision encoder did not start")
        except RuntimeError as e:                       # silent for VISION_READY_S: it was ended, and it is not left half-started
            self.proc, self.stopped = None, True
            raise RuntimeError(str(e)) from None
        if not line.startswith("READY"):
            proc, self.proc = self.proc, None
            self.stopped = True
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            raise RuntimeError(vision_start_error(line, self.spawn[1]))
        self.stopped = False

    def _readline(self, timeout: float, what: str) -> str:
        """One line from the encoder, waiting at most `timeout` s (#1317).  On a timeout the encoder is killed, so the
        next request starts a fresh one, and the read raises RuntimeError instead of holding the request turn."""
        proc = self.proc
        got: queue.Queue = queue.Queue()

        def read():
            try:
                got.put(proc.stdout.readline())
            except (OSError, ValueError):               # killed under the read
                got.put("")

        threading.Thread(target=read, daemon=True).start()
        try:
            return got.get(timeout=timeout)
        except queue.Empty:
            try:
                proc.kill()
            except OSError:
                pass
            raise RuntimeError(f"{what}: it said nothing for {timeout:.0f} s") from None

    def alive(self) -> bool:
        return self.proc is not None and not self.stopped and self.proc.poll() is None

    def unload(self):
        """Stop the encoder process (its VRAM or RAM goes back); the encoded images stay cached on disk."""
        self.close()
        self.stopped = True

    def restart(self):
        """Start the encoder again after an unload (or if it died); the cache of encoded images is kept."""
        if self.proc is not None:
            try:
                self.proc.kill()
            except OSError:
                pass
        self.proc = None
        self.stopped = True
        self._start()

    @staticmethod
    def load(source: str) -> bytes:
        if source.startswith("data:"):
            try:
                return base64.b64decode(source.split(",", 1)[1])
            except (IndexError, ValueError) as e:                # no comma, or not base64
                raise ValueError(f"the image's data: URL could not be read ({e})") from None
        if source.startswith(("http://", "https://")):
            return Vision.download(source)
        path = source[7:] if source.startswith("file://") else source
        # refused before anything looks at the path: file://host/share/x.png is \\host\share\x.png ("file://C:/x.png",
        # a drive, is a client's spelling of file:///C:/x.png)
        host = re.split(r"[/\\]", path, maxsplit=1)[0] if source.startswith("file://") else ""
        if network_path(path) or host and not re.fullmatch(r"[A-Za-z]:|localhost", host, re.IGNORECASE):
            raise ValueError("an image file must be on this computer: network paths (\\\\host\\share, //host/share, "
                             "file://host/...) are not read")
        if path and os.path.isfile(path):
            try:
                return Path(path).read_bytes()
            except OSError as e:
                raise ValueError(f"the image could not be read from {path} ({e})") from None
        raise ValueError("an image must be a data: URL, an http(s) URL or a local file path")

    @staticmethod
    def download(url: str, guard=None) -> bytes:
        """An image URL's bytes, at most IMAGE_URL_MAX of them: the whole response was read, so a huge or endless one
        filled the memory.  One that cannot be read is a ValueError (a 400 that says so), not a dropped connection.
        `guard(ip, port)` (#311: a team guest's URL, web.public_not_vpn): fetched by web.Fetcher, which checks every
        address it connects to - after the name is looked up, and again on every redirect - and the refusal says only
        that the address is not public, before anything is sent to it."""
        too_big = f"the image URL's file is over {IMAGE_URL_MAX >> 20} MiB"
        if guard is not None:
            try:
                page = web_mod.Fetcher(guard=guard, timeout=60).get(url, headers={"Accept": "image/*,*/*;q=0.5", "Accept-Encoding": "identity"},
                                                                    limit=IMAGE_URL_MAX)
            except web_mod.NotPublic:
                raise ValueError("an image URL must be an address on the public internet: send the image as a data: "
                                 "URL") from None
            except web_mod.Refused as e:
                raise ValueError(f"the image URL could not be read: {e}") from None
            if page.cut:
                raise ValueError(too_big if len(page.body) >= IMAGE_URL_MAX else "the image URL could not be read whole (too slow, or the "
                                 "connection ended)")
            if not 200 <= page.status < 300:
                raise ValueError(f"the image URL could not be read: HTTP {page.status}")
            return page.body
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "strata"}), timeout=60) as r:
                size = r.headers.get("Content-Length") or ""
                if size.isdigit() and int(size) > IMAGE_URL_MAX:
                    raise ValueError(too_big)
                data = r.read(IMAGE_URL_MAX + 1)
        except (OSError, HTTPException) as e:
            raise ValueError(f"the image URL could not be read: {e}") from None
        if len(data) > IMAGE_URL_MAX:
            raise ValueError(too_big)
        return data

    @staticmethod
    def _exif_rotated(data: bytes) -> bool:
        """A JPEG whose EXIF Orientation says it is stored turned (phones do this for portrait photos): the decoder
        (stb_image) ignores the tag.  False without Pillow, or when the tag is missing, 1 or unreadable (#1229)."""
        try:
            import io
            from PIL import Image
            with Image.open(io.BytesIO(data)) as im:
                return im.getexif().get(0x0112, 1) not in (None, 0, 1)
        except Exception:  # noqa: BLE001 - any failure leaves the image as it was sent
            return False

    @staticmethod
    def normalize(data: bytes) -> bytes:
        """The formats strata-vision's decoder (stb_image) reads pass through; anything else is converted to PNG.
        A JPEG with an EXIF Orientation other than 1 is turned upright first (needs Pillow)."""
        jpeg = data[:3] == b"\xff\xd8\xff"
        if (jpeg and not Vision._exif_rotated(data)) or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:2] == b"BM" or \
                data[:6] in (b"GIF87a", b"GIF89a"):
            return data
        try:
            import io
            from PIL import Image
        except ImportError:
            raise ValueError("this image format needs Pillow (python -m pip install pillow); JPEG, PNG, BMP and "
                             "GIF work without it") from None
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im)             # the camera's orientation tag, applied (#1229)
        except Exception as e:
            raise ValueError(f"the image could not be read ({e})") from None
        if im.mode in ("RGBA", "LA", "P") and "transparency" in im.info or im.mode in ("RGBA", "LA"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))   # transparent areas become white, not black
            bg.paste(im, mask=im.split()[-1])
            im = bg
        elif im.mode != "RGB":
            im = im.convert("RGB")
        out = io.BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()

    def encode_all(self, sources: list) -> list[tuple[Path, int]]:
        """encode() for every image of one request: the files of this request's earlier images are never evicted to
        make room for a later one (#1072: with more than 64 unique images the first was deleted before the request
        had read it)."""
        out: list[tuple[Path, int]] = []
        for src in sources:
            out.append(self.encode(src, keep=[p for p, _ in out]))
        return out

    def encode(self, source: str | bytes, keep=()) -> tuple[Path, int]:
        """-> (embeddings file, number of image tokens).  `source`: what load() reads, or the image's bytes.
        `keep`: files the caller still needs (never evicted by this call)."""
        data = self.normalize(source if isinstance(source, bytes) else self.load(source))
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            if key in self.cache:
                self.cache[key] = self.cache.pop(key)                  # most recently used last
                return self.cache[key]
            img, out = self.dir / f"{key}.img", self.dir / f"{key}.sve"
            img.write_bytes(data)
            try:
                self.proc.stdin.write(f"ENC {img.name} {out.name}\n")      # relative to the encoder's cwd (#480)
                self.proc.stdin.flush()
                try:
                    line = self._readline(VISION_ENCODE_S, "the image could not be read").strip()
                except RuntimeError as e:                 # silent for VISION_ENCODE_S: the encoder was ended (#1317)
                    self.stopped = True                   # alive() is False, so the next request starts it again
                    raise ValueError(str(e)) from None
            finally:                                                   # #352: also when the encoder's pipe is gone
                img.unlink(missing_ok=True)
            if not line.startswith("OK"):
                raise ValueError("the image could not be read: " + (line[4:] if line.startswith("ERR") else
                                                                     "the vision encoder stopped"))
            self.cache[key] = (out, int(line.split()[1]))
            held = {str(p) for p in keep} | {str(out)}
            for old in list(self.cache):                               # oldest first, but never one still in use
                if len(self.cache) <= 64:
                    break
                if str(self.cache[old][0]) not in held:
                    self.cache.pop(old)[0].unlink(missing_ok=True)
            return self.cache[key]

    def close(self):
        proc, self.proc = self.proc, None
        if proc is None:
            self.stopped = True
            return
        try:
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self.stopped = True

    def shutdown(self):
        """#914: close() for good: the server ends, so the encoder's directory (one ~10 MB .sve per image, on a tmpfs
        /tmp that is RAM) goes with it instead of piling up with every start.  unload() keeps it for the restart."""
        self.close()
        shutil.rmtree(self.dir, ignore_errors=True)


def gpu_list(cfg: dict) -> list[int]:
    """The config's "gpu": one card (2), or several for a layer split ([0, 2] or "0,2"), numbered as nvidia-smi
    numbers them; [] when it names none."""
    g = cfg.get("gpu")
    if g is None or g == "":
        return []
    items = g if isinstance(g, (list, tuple)) else str(g).split(",")
    return [int(str(x).strip()) for x in items if str(x).strip() != ""]


def monitor_gpus(cfg: dict, env=None, count=None) -> list[int]:
    """The cards the Monitor reads: the config's "gpu" when it names any; else the launcher's CUDA_VISIBLE_DEVICES
    (numbers, taken as NVML numbers them: right where CUDA's order matches the PCI order, and when it does not the
    set of two cards is still both); else every card NVML sees. A config
    with no "gpu" key (D2x) used to leave only card 0 on the Monitor."""
    named = gpu_list(cfg)
    if named:
        return named
    if count is None:
        from serve.telemetry import nvml_device_count as count
    n = count()
    if n <= 0:
        return []
    raw = ((os.environ if env is None else env).get("CUDA_VISIBLE_DEVICES") or "").strip()
    try:
        ids = sorted({int(x) for x in raw.split(",") if x.strip() != ""})
    except ValueError:                                  # UUIDs: not mappable here, show every card
        ids = []
    return [i for i in ids if 0 <= i < n] or list(range(n))


def engine_binary(exe: str, cfg: dict | None = None) -> str:
    """The file that holds the engine's own option strings, for a feature check that reads it.  The Intel launcher
    (sycl/serve/strata-sycl.sh) is a shell script that runs its repo's STRATA_SYCL_BIN (default build-sycl-aot/strata)
    in a container, so the check reads that binary, not the script.  Any other `exe` is the engine itself."""
    p = Path(exe)
    if p.name == "strata-sycl.sh":
        rel = ((cfg or {}).get("env") or {}).get("STRATA_SYCL_BIN") or os.environ.get("STRATA_SYCL_BIN") \
            or "build-sycl-aot/strata"
        b = p.resolve().parents[2] / rel
        if b.is_file():
            return str(b)
    return exe


def effort_end_args(cfg: dict, exe: str, tok) -> list[str] | None:
    """#458 (opt-in): the engine arguments for "effort_position": "end" - the id of "system" as --tail-role-token, so
    the engine checkpoints in front of the trailing effort turn - or None when the config leaves it at the top (the
    default) or the engine is too old for it (said once; the prompt stays the default one).  ValueError for a value
    other than "start" / "end"."""
    pos = cfg.get("effort_position", "start")
    if pos not in ("start", "end"):
        raise ValueError(f'"effort_position" must be "start" (the default) or "end", not {pos!r}')
    if pos == "start":
        return None
    try:
        with open(engine_binary(exe, cfg), "rb") as f:
            known = b"--tail-role-token" in f.read()
    except OSError:
        known = False
    role = tok.encode("system", parse_special=True)
    if not known or len(role) != 1:
        print("[strata] effort_position \"end\" needs engine 0.1.39 or newer (--tail-role-token): the reasoning "
              "effort stays at the top of the prompt", flush=True)
        return None
    print("[strata] effort_position end: a request's non-default reasoning effort goes right before the answer, so "
          "changing it keeps the cached conversation", flush=True)
    return ["--tail-role-token", str(role[0])]


def engine_silence_s(cfg: dict) -> float:
    """#481: the config's "engine_silence_s" - seconds an engine may print nothing during a request before the server
    ends it (default ENGINE_SILENCE_S; 0 = wait forever).  ValueError for anything but a number >= 0."""
    v = cfg.get("engine_silence_s")
    if v is None:
        return ENGINE_SILENCE_S
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
        raise ValueError(f'"engine_silence_s" must be a number of seconds >= 0 (0 = no limit), not {v!r}')
    return float(v)


def layer_split_value(cfg: dict) -> str:
    """#644: the config's "layer_split" as the engine's --layer-split value: "auto" (the default), or the FIRST LAYER of
    each later GPU's share - one rising number per GPU after the first, e.g. "24,36,42" for 4 GPUs (layers 0-23 on the
    first card, 24-35, 36-41, 42 to the end) - as a string or a JSON list ([24, 36, 42]).  ValueError with the format
    and an example for anything else, such as per-card layer counts ("24,16,12,12")."""
    v = cfg.get("layer_split")
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "auto")):
        return "auto"
    n = len(gpu_list(cfg))
    items = v if isinstance(v, (list, tuple)) else str(v).split(",") if isinstance(v, (str, int)) else None
    vals = []
    for x in items or []:
        try:
            if isinstance(x, bool) or isinstance(x, float):
                raise ValueError
            vals.append(int(str(x).strip()))
        except ValueError:
            vals = None
            break
    want = max(n - 1, 1)
    example = ",".join(str(round(48 * (i + 1) / (want + 1))) for i in range(want))
    hint = (f'"layer_split" is the first layer of each later GPU, one rising number per GPU after the first '
            f'({want} for {n} GPUs), not a count of layers per card - e.g. "{example}" (or [{example.replace(",", ", ")}])'
            f' for an even share of 48 layers, or "auto" (the default) to place them by each card\'s free VRAM')
    if not vals:
        raise ValueError(f"{hint}; got {v!r}")
    if n > 1 and len(vals) != n - 1:
        raise ValueError(f"{hint}; got {len(vals)} number(s) ({v!r}) for {n} GPUs")
    if vals[0] < 2 or any(b <= a for a, b in zip(vals, vals[1:])):
        raise ValueError(f"{hint}; got {v!r}, which does not rise from 2 or more")
    return ",".join(str(x) for x in vals)


def layer_split_of(cfg: dict) -> bool:
    """Several GPUs in the config are a layer split, unless the args put the later card(s) to another use: with
    --peer-device the second card is an expert-cache tier, and the engine refuses that beside --layer-split (#665)."""
    return len(gpu_list(cfg)) > 1 and "--peer-device" not in cfg["args"]


def engine_args(cfg: dict) -> list[str]:
    """The engine's arguments: the config's, and with several GPUs the layer split across them ("layer_split" in the
    config: "auto" by default, or the first layer of each later GPU's share, e.g. "18" or "16,32"; see
    layer_split_value)."""
    args = list(cfg["args"])
    # #1322: a config with a "vision" section but without --vision in its args (written by an older setup run, or edited by
    # hand) advertised images and then refused every picture ("this engine was started without --vision").  The section
    # says images are wanted: start the engine with them, and keep the encoder's VRAM free as setup does for a GPU encoder
    # (only when the config has no reserve of its own).
    if isinstance(cfg.get("vision"), dict) and "--vision" not in args:
        args.append("--vision")
        note = "added --vision"
        if cfg["vision"].get("gpu") and "--vram-reserve-mib" not in args:
            args += ["--vram-reserve-mib", "700"]
            note += " and --vram-reserve-mib 700"
        print(f'[strata] the config has a "vision" section but no --vision in its args: {note} (run setup again with '
              "--vision to write them)", flush=True)
    if layer_split_of(cfg) and "--layer-split" not in args:
        args += ["--layer-split", layer_split_value(cfg)]
    # opt-in: an auto split runs on the first card alone when it holds every profiled expert and the KV
    if layer_split_of(cfg) and cfg.get("split_skip_if_fits") and "--split-skip-if-fits" not in args:
        args.append("--split-skip-if-fits")
    # #533 (opt-in): "vram_elastic": true - the expert cache in segments, so POST /v1/vram can give VRAM back to other
    # programs and take it back; "vram_segment_mib" sets the segment size (the engine's default: 512)
    if cfg.get("vram_elastic") is True and "--vram-elastic" not in args:
        args.append("--vram-elastic")
        seg = cfg.get("vram_segment_mib")
        if isinstance(seg, int) and not isinstance(seg, bool) and seg > 0 and "--vram-segment-mib" not in args:
            args += ["--vram-segment-mib", str(seg)]
    args += parallel_args(cfg, args)
    return learned_profile_args(cfg, args)


PARALLEL_MAX = 8      # the engine's batch window holds at most 8 rows (kVerifyMaxT); --batch-mtp waves more through it


def parallel_args(cfg: dict, args: list[str]) -> list[str]:
    """#465 (opt-in): "parallel": N in the config - up to N requests decode together in the engine's batch slots
    (--batch N); more wait their turn.  1 or absent: one request at a time, as before.  A value the engine cannot use
    is said and passed on: the engine warns and adjusts (recommend, never force)."""
    if "--batch" in args or "--slots" in args:
        return []
    n = cfg.get("parallel")
    if n is None or n is False:
        return []
    if isinstance(n, bool) or not isinstance(n, int):
        print(f'[strata] "parallel" must be a whole number of requests (2..{PARALLEL_MAX}), not {n!r}: ignored',
              flush=True)
        return []
    if n <= 1:
        return []
    if n > PARALLEL_MAX and "--batch-mtp" not in args and os.environ.get("STRATA_BATCH_MTP", "0") in ("", "0"):
        print(f'[strata] "parallel": {n} - the engine runs at most {PARALLEL_MAX} requests together; it will use '
              f"{PARALLEL_MAX}", flush=True)
    return ["--batch", str(n)]


def profile_shape(path: str) -> tuple[int, int] | None:
    """An expert profile's (layers, experts per layer), from its header (tools/make_profile.py's format), or None
    when the file is missing, is not one or is shorter than the pairs its header promises."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            size = os.fstat(f.fileno()).st_size
    except OSError:
        return None
    if len(head) < 24 or head[:4] != b"STRP":
        return None
    _, nl, ne, _, n = struct.unpack("<5I", head[4:])
    return (nl, ne) if size >= 24 + 4 * n else None


def learned_profile_args(cfg: dict, args: list[str]) -> list[str]:
    """#477 (opt-in): "expert_profile_save": "<path>" in the config has the engine (0.1.36+) save what its adaptive
    tier learned there - on QUIT and every "expert_profile_save_every" minutes (10 by default, 0 = at QUIT only) -
    and the next start begins from it instead of the config's --expert-profile, when it is a profile of the same
    model (its header's layers and experts match); otherwise from the config's own, as before.  A relative path is
    the engine's (the config's "cwd").  Without the key, the arguments are the config's, unchanged."""
    save = cfg.get("expert_profile_save")
    if not isinstance(save, str) or not save.strip() or "--expert-profile-save" in args:
        return args
    args = args + ["--expert-profile-save", save]
    every = cfg.get("expert_profile_save_every")
    if isinstance(every, (int, float)) and not isinstance(every, bool) and every >= 0:
        args += ["--expert-profile-save-every", str(every)]
    if "--expert-profile" in args[:-1]:
        i = args.index("--expert-profile") + 1
        here = cfg.get("cwd") or "."
        learned = profile_shape(save if os.path.isabs(save) else os.path.join(here, save))
        base = profile_shape(args[i] if os.path.isabs(args[i]) else os.path.join(here, args[i]))
        if learned is not None and learned == base:
            args[i] = save
    return args


def hip_visible(cfg: dict) -> list[int]:
    """AMD: the devices the engine should see, as the HIP runtime numbers them (HIP_VISIBLE_DEVICES).

    On Linux setup's KFD order is HIP's order, so the config's "gpu" is it.  On Windows setup finds the cards in the
    display-adapter order, and an integrated Radeon that HIP also enumerates takes ordinal 0 and pushes the discrete
    card to 1 (#325): setup records the ordinal `strata-device --list-devices` gave the card as "hip_ordinal", which
    wins for a one-card config.  Without it (a config from before), the config's "gpu"."""
    ordinal = cfg.get("hip_ordinal")
    if ordinal is not None and str(ordinal).strip() != "" and len(gpu_list(cfg)) <= 1:
        try:
            return [int(str(ordinal).strip())]
        except ValueError:
            pass
    return ordered_gpus(cfg)


def gpu_speed_scores(indices: list[int]) -> dict[int, float] | None:
    """Multiprocessors x max clock for each NVIDIA card, numbered as nvidia-smi numbers them (the CUDA driver API,
    which every driver ships: no toolkit needed); None when it cannot say for every one of them."""
    code = (
        "import ctypes,sys,os\n"
        "os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'\n"
        "lib=ctypes.CDLL('nvcuda.dll' if os.name=='nt' else 'libcuda.so.1')\n"
        "assert lib.cuInit(0)==0\n"
        "n=ctypes.c_int()\n"
        "assert lib.cuDeviceGetCount(ctypes.byref(n))==0\n"
        "for i in range(n.value):\n"
        "    d=ctypes.c_int()\n"
        "    assert lib.cuDeviceGet(ctypes.byref(d),i)==0\n"
        "    sm=ctypes.c_int(); mhz=ctypes.c_int()\n"
        "    lib.cuDeviceGetAttribute(ctypes.byref(sm),16,d)\n"
        "    lib.cuDeviceGetAttribute(ctypes.byref(mhz),13,d)\n"
        "    print(i,sm.value,mhz.value)\n")
    try:
        env = dict(os.environ)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env.pop("CUDA_VISIBLE_DEVICES", None)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, env=env)
        got = {}
        for line in r.stdout.splitlines():
            i, sm, khz = (int(x) for x in line.split())
            got[i] = float(sm) * float(khz)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return got if all(i in got and got[i] > 0 for i in indices) else None


def hip_speed_scores(indices: list[int], root: str = "/sys/class/kfd/kfd/topology/nodes") -> dict[int, float] | None:
    """AMD on Linux: SIMDs x max engine clock of each card, from the KFD topology (the order setup and the engine use
    for "gpu" there: the GPU nodes in node order).  None when the files are not there (Windows, no ROCm driver)."""
    try:
        nodes = sorted((int(n) for n in os.listdir(root) if n.isdigit()))
    except OSError:
        return None
    cards = []
    for n in nodes:
        try:
            props = dict(line.split() for line in open(os.path.join(root, str(n), "properties")) if len(line.split()) == 2)
            simd, clk = int(props.get("simd_count", 0)), int(props.get("max_engine_clk_fcompute", 0))
        except (OSError, ValueError):
            continue
        if simd > 0:
            cards.append(float(simd) * float(clk or 1))
    got = dict(enumerate(cards))
    return got if all(i in got for i in indices) else None


def gpu_vram_mib(indices: list[int], hip: bool = False) -> dict[int, int] | None:
    """Total VRAM of each card in MiB (NVML / amdgpu sysfs); None when any of them cannot be read."""
    try:
        from serve.telemetry import gpu_reader
    except ImportError:
        return None
    out = {}
    for i in indices:
        g = gpu_reader(i, amd=hip)
        if not g.ok():
            return None
        total = g.read().get("mem_total")
        if total is None:
            return None
        out[i] = int(total) >> 20
    return out


def ordered_gpus(cfg: dict, scores=None, vrams=None) -> list[int]:
    """#1352: the cards of a layer split in the order the engine stages them.  With "layer_split": "auto" (the
    default) the faster card goes LAST - the last stage runs the head, the draft layer and the verify, and a prompt
    chunk waits on it (the reporter's 4070 Ti SUPER + 5060 Ti: a 6K prompt took 50 s one way round and 15 s the
    other); "faster" is multiprocessors x max clock.  Equal cards - scores within 5% - keep the config's order, and
    so does anything we cannot measure.  "gpu_order": "as_given" keeps the config's order whatever the cards.  A
    manual "layer_split" ("24") also keeps it: the user placed the layers.  #1576: the reorder is skipped when it
    would land the last stage on a card with less total VRAM than the one the config put there - that stage's head,
    draft and verify have a floor the smaller card cannot afford.  scores/vrams: {index: value} (tests); None asks
    the driver - injected scores without injected vrams means no VRAM check (a test seam is a test seam)."""
    gl = gpu_list(cfg)
    if len(gl) < 2 or cfg.get("gpu_order") == "as_given":
        return gl
    args = cfg.get("args") or []
    if "--peer-device" in args or "--layer-split" in args or layer_split_value(cfg) != "auto":
        return gl
    hip = cfg.get("backend") == "hip"
    sc = scores if scores is not None else (hip_speed_scores(gl) if hip else gpu_speed_scores(gl))
    if not sc or any(i not in sc for i in gl):
        return gl
    if max(sc[i] for i in gl) < 1.05 * min(sc[i] for i in gl):
        return gl     # #1760: cards within 5% of each other are the same card (two 5070 Ti read 1.2% apart on the clock): keep the order
    out = sorted(gl, key=lambda i: sc[i])
    if out != gl:
        vram = vrams if vrams is not None else (None if scores is not None else gpu_vram_mib(gl, hip))
        if vram is not None and all(i in vram for i in gl) and vram[out[-1]] < vram[gl[-1]]:
            print(f"[strata] layer split: keeping card order {','.join(map(str, gl))} - the faster card has "
                  f"less VRAM than the one the split ends on ({vram[out[-1]]} vs {vram[gl[-1]]} MiB, #1576)",
                  flush=True)
            return gl
        print(f"[strata] layer split: card order {','.join(map(str, out))} (the faster card last; "
              f'"gpu_order": "as_given" keeps {",".join(map(str, gl))}, #1352)', flush=True)
    return out


def child_env(cfg: dict) -> dict:
    """The engine's environment: the CUDA libraries setup installed (pip's nvidia packages, or the toolkit that
    compiled it) first on the library search path."""
    env = dict(os.environ)
    if hip_visible(cfg) and cfg.get("backend") == "hip":   # AMD: numbered as HIP numbers them (hip_visible)
        env["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in hip_visible(cfg))
    elif gpu_list(cfg):                              # issue #51: the GPU(s) to run on, numbered as nvidia-smi does; CUDA's
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # own default order (fastest first) can number the cards otherwise
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in ordered_gpus(cfg))
    for k, v in (cfg.get("env") or {}).items():      # engine settings the config carries (AMD: the GEMM tuning table)
        env[str(k)] = str(v)
    dirs = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    if dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    if cfg.get("backend") == "hip":
        hip_env_guard(env, cfg)
    return env


def _dir_writable(p: str) -> bool:
    try:
        with tempfile.TemporaryFile(dir=p):
            return True
    except OSError:
        return False


def hip_env_guard(env: dict, cfg: dict) -> list[str]:
    """#654: two things in an AMD user's environment that crash the HIP runtime before it prints a line.  (1) HIP_PATH,
    HIP_DEVICE_LIB_PATH or LLVM_PATH naming a folder that is gone (a ROCm build that was deleted): the bundled runtime
    follows them.  They are dropped, and only when the folder does not exist.  (2) A TEMP / TMP / TMPDIR the engine cannot
    create a file in: the runtime's comgr JIT builds its blit kernels through temporary files and the first call
    crashes in amdhip64 (0xC0000005).  A writable folder next to the engine's log takes its place.  Returns what it
    changed (said once, on stderr); a healthy environment is left exactly as it was."""
    said: list[str] = []
    for k in ("HIP_PATH", "HIP_DEVICE_LIB_PATH", "LLVM_PATH"):
        v = env.get(k)
        if v and not Path(v).exists():
            del env[k]
            said.append(f"{k}={v} does not exist: not passed to the engine")
    bad = [k for k in ("TEMP", "TMP", "TMPDIR") if env.get(k) and not _dir_writable(env[k])]
    if bad:
        base = Path(cfg.get("log") or cfg.get("cwd") or ROOT).resolve()
        for d in (base.parent / "tmp", ROOT / "tmp"):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            if _dir_writable(str(d)):
                for k in bad:
                    said.append(f"{k}={env[k]} is not writable: using {d}")
                    env[k] = str(d)
                break
        else:
            said.append(f"{', '.join(bad)} not writable (AMD's runtime needs a writable temporary folder)")
    for line in said:
        print(f"strata serve: AMD: {line}", file=sys.stderr, flush=True)
    return said


def vision_env(cfg: dict, env: dict) -> dict:
    """The image encoder's environment: the engine's, unless the config's vision section names its own "cuda_device"
    (numbered like nvidia-smi) - then the encoder runs on that card alone, so a spare GPU can hold it while the engine
    keeps all of its own cards' VRAM (#408, Efs-O).  Without it, nothing changes."""
    dev = (cfg.get("vision") or {}).get("cuda_device")
    if dev is None:
        return env
    env = dict(env)
    if cfg.get("backend") == "hip":
        env["HIP_VISIBLE_DEVICES"] = str(dev)
    else:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(dev)
    return env


class ByteTokenizer:
    """Tiny stand-in tokenizer for tests without the pack: one id per UTF-8 byte, specials as ids >= 256."""
    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]

    ALWAYS = ()                                     # specials matched without parse_special (type 4, as <think>)

    @property
    def control_tokens(self):
        return [s for s in self.SPECIALS if s not in self.ALWAYS]

    def encode(self, text, parse_special=False, plain=()):
        out, i = [], 0
        while i < len(text):
            for k, s in enumerate(self.SPECIALS):
                if (parse_special or s in self.ALWAYS) and text.startswith(s, i) and not any(
                        a <= i < b for a, b in plain):
                    out.append(256 + k)
                    i += len(s)
                    break
            else:
                out.extend(text[i].encode("utf-8"))
                i += 1
        return out

    def decode(self, ids, errors="replace"):
        raw = bytearray()
        for t in ids:
            raw += self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])
        return raw.decode("utf-8", errors=errors)


# ------------------------------------------------------------------------------------------------ core
class Detokenizer:
    """Incremental decode: each token's bytes go through an incremental UTF-8 decoder, which emits the complete
    characters and holds a multi-byte character split across tokens until it is complete (invalid bytes become
    U+FFFD, as a whole decode with errors="replace" makes them).  Constant time per token - the old re-decode of
    every generated id cost 2 ms per token after 8K tokens and 4 ms after 16K (perf-review F-1).  A tokenizer
    without `token_bytes` (the tests' byte tokenizer) keeps the re-decode."""

    def __init__(self, tok):
        self.tok, self.ids, self.sent = tok, [], 0
        self.inc = codecs.getincrementaldecoder("utf-8")(errors="replace") if hasattr(tok, "token_bytes") else None

    def pending(self) -> bool:
        """A character is split across the tokens so far: its first bytes are held."""
        if self.inc is not None:
            return bool(self.inc.getstate()[0])
        return self.tok.decode(self.ids).endswith("\ufffd")

    def push(self, t: int) -> str:
        if self.inc is not None:
            return self.inc.decode(self.tok.token_bytes(t))
        self.ids.append(t)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta


class RequestGate:
    """One request on the engine at a time (xeno #49 S6). Waiting requests go in (priority, arrival) order: Claude
    Code's main turn streams (priority 0), its title and auto-mode classifier requests do not (1), and they used to
    hold the only slot while the main turn queued. The running request is never pre-empted. Used as a plain lock
    (`with gate:`) it is priority 1, first come first served."""

    def __init__(self):
        self._cv = threading.Condition()
        self._busy = False
        self._waiting: list = []
        self._arrival = itertools.count()

    def acquire(self, priority: int = 1, blocking: bool = True, timeout: float | None = None) -> bool:
        """`blocking=False` (upstream #208's idle unload, which assumed a Lock): take the gate only when it is free
        and no request waits for it; False otherwise, and nothing is queued.  `timeout` (upstream's callers that
        assumed a Lock: /v1/vram, #1619's queue keep-alive): False when the turn has not come within that many
        seconds, and the place in the queue is given up (None or a negative value: wait for good)."""
        if not blocking:
            with self._cv:
                if self._busy or self._waiting:
                    return False
                self._busy = True
                return True
        ticket = (priority, next(self._arrival))
        deadline = None if timeout is None or timeout < 0 else time.monotonic() + timeout
        with self._cv:
            heapq.heappush(self._waiting, ticket)
            while self._busy or self._waiting[0] != ticket:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    self._waiting.remove(ticket)
                    heapq.heapify(self._waiting)
                    self._cv.notify_all()               # the one behind it may be first now
                    return False
                self._cv.wait(left)
            heapq.heappop(self._waiting)
            self._busy = True
        return True

    def release(self):
        with self._cv:
            self._busy = False
            self._cv.notify_all()

    def locked(self) -> bool:
        """threading.Lock's question (upstream's callers and tests assumed a Lock): a request holds the gate."""
        with self._cv:
            return self._busy

    @contextlib.contextmanager
    def slot(self, priority: int = 1):
        self.acquire(priority)
        try:
            yield
        finally:
            self.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, *exc):
        self.release()


OPENAI_MAX_STOP = 4                                  # OpenAI's limit on `stop`


def stop_strings(req: dict) -> list[str]:
    """The request's stop strings: OpenAI's `stop` (a string or a list of up to 4) or Anthropic's `stop_sequences`
    (a list).  Empty strings are left out.  A bad value is a ValueError, so a 400 before anything is generated."""
    req = req or {}
    if req.get("stop") is not None:
        stop = req["stop"]
        stop = [stop] if isinstance(stop, str) else stop
        if not isinstance(stop, list) or not all(isinstance(x, str) for x in stop):
            raise ValueError("stop must be a string or a list of strings")
        if len(stop) > OPENAI_MAX_STOP:
            raise ValueError(f"stop takes at most {OPENAI_MAX_STOP} strings")
    elif req.get("stop_sequences") is not None:
        stop = req["stop_sequences"]
        if not isinstance(stop, list) or not all(isinstance(x, str) for x in stop):
            raise ValueError("stop_sequences must be a list of strings")
    else:
        return []
    return [x for x in stop if x]


class StopMatcher:
    """Cuts the answer's text at the first stop string.  Text that could still be the start of a stop string is held
    back (only that tail, never more), so a stop string split across tokens or chunks is still found and never sent."""

    def __init__(self, stops: list[str]):
        self.stops, self.held, self.hit = stops, "", None

    def push(self, text: str) -> str:
        """-> the text that is safe to send now; after a match, self.hit is the stop string and nothing more comes."""
        if self.hit is not None:
            return ""
        buf = self.held + text
        first = None
        for s in self.stops:
            i = buf.find(s)
            if i >= 0 and (first is None or i < first[0]):
                first = (i, s)
        if first is not None:
            self.held, self.hit = "", first[1]
            return buf[:first[0]]
        keep = 0                                      # the longest end of buf that a stop string starts with
        for s in self.stops:
            for k in range(min(len(s) - 1, len(buf)), keep, -1):
                if buf.endswith(s[:k]):
                    keep = k
                    break
        self.held = buf[len(buf) - keep:] if keep else ""
        return buf[:len(buf) - keep]

    def flush(self) -> str:
        held, self.held = self.held, ""
        return held



# Windows names that open a device whatever their extension or folder (Microsoft, "Naming Files, Paths, and Namespaces")
_WIN_DEVICES = {"con", "prn", "aux", "nul", "conin$", "conout$", *(f"com{i}" for i in "123456789\u00b9\u00b2\u00b3"),
                *(f"lpt{i}" for i in "123456789\u00b9\u00b2\u00b3")}


def slot_filename_problem(name) -> str | None:
    """/slots/0 {"filename": NAME}: why NAME is not one plain file name inside the slot save path, on Linux and on
    Windows alike (None when it is).  No separator or drive (`D:x`), no stream (`a:b`), no device (`NUL.txt`), no
    trailing dot or space (Windows drops them), no control character, no leading dot (temporary files use it)."""
    if not isinstance(name, str) or not name:
        return "must be a non-empty string"
    if len(name.encode("utf-8", "surrogatepass")) > 200:
        return "is too long (at most 200 bytes)"
    if any(ord(c) < 32 or ord(c) == 127 for c in name):
        return "must not contain control characters"
    if any(c in name for c in '/\\:*?"<>|'):
        return "must be a plain file name (no / \\ : * ? \" < > |)"
    if name.startswith("."):
        return "must not start with a dot"
    if name.endswith((".", " ")):
        return "must not end with a dot or a space"
    if name.split(".")[0].rstrip(" ").lower() in _WIN_DEVICES:
        return "names a Windows device"
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return "is not valid Unicode"
    return None


def slot_save_dir(value, base: str | None = None) -> str:
    """The slot save path as one absolute, normalized directory (relative to `base`, else the server's working
    directory), created private to this user when missing.  ValueError for anything else."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("slot_save_path must be a directory path (a string)")
    if any(ord(c) < 32 for c in value):
        raise ValueError("slot_save_path must not contain control characters")
    path = os.path.abspath(os.path.join(base, value) if base and not os.path.isabs(value) else value)
    os.makedirs(path, mode=0o700, exist_ok=True)
    if not os.path.isdir(path):
        raise ValueError(f"slot_save_path {path} is not a directory")
    return path

class Service:
    def __init__(self, engine: Engine, tokenizer, template: ChatTemplate, model_name: str = "qwen3.8-flash-next",
                 vision: Vision | None = None, sampling_defaults: dict | None = None,
                 fit_max_tokens: bool = False):
        self.engine, self.tok, self.template, self.model, self.vision = engine, tokenizer, template, model_name, vision
        self.efforts = template.efforts() if hasattr(template, "efforts") else {"levels": [], "default": None, "off": False}
        self.literals = literal_tags(getattr(tokenizer, "control_tokens", ()))   # texts that stay text inside a message
        self.fit_max_tokens = fit_max_tokens          # --fit-max-tokens: clamp the output cap instead of 400
        self.aliases: list[str] = []                  # #297: other names of the model (the config's `aliases`)
        self.reasoning_close_retry = False            # #1053 (opt-in): close the thinking once when a reply ends inside it
        self.codex_compaction_cache = False           # #924 (opt-in): a Codex compaction is rendered with its conversation's tools
        self.codex_thread_titles = False              # #923 (opt-in): answer Codex's thread-title turns without the engine
        self.sampling_defaults = dict(sampling_defaults or {})   # the run config's `sampling` block
        self.restarting = False         # xeno #49 review: the engine is loading again (requests get 529 meanwhile)
        self.cjk_ban = False            # xeno #49 S4: the engine was started with the Han ban list (--ban-ids)
        self.shared = {}                              # the web app's Chat settings for every client (POST /settings)
        self.shared_path = None                       # where they are kept between starts (next to the config)
        self.config_effort = None                     # #1641: the config's "reasoning_effort": the level for requests with none
        self.keep_awake = None                        # #1727 (opt-in): serve.keepawake.KeepAwake, the config's "prevent_sleep"
        self.fifo = RequestGate()
        self.slot_save_path = None                    # --slot-save-path: /slots/0?action=save|restore (off when None)
        self.embeddings = threading.local()           # the current request's image embeddings file (GENI)
        self.api_key = ""                              # when set, /v1/* needs it (Bearer or x-api-key)
        # #310: named keys with the roles owner and guest, and the approval of new devices (serve/identity.py); main()
        # gives it the keys file beside the run config.  Team mode is that file existing; without it every gate below
        # behaves as before (the one api_key is the owner's)
        self.keys = identity_mod.KeyStore(None)
        self.trust_loopback = True                     # #310: a request from this PC to this PC needs no device approval
        self.peer_override = None                      # #310 tests: the client address the handler sees instead of the socket's
        # #458 (opt-in, the config's "effort_position": "end"): a non-default reasoning effort goes in a short system
        # turn before the answer instead of the top of the prompt, so switching it keeps the cached conversation
        self.effort_end = False
        self.conv_log = ConvCacheLog()                  # #596: the parked conversations, from the engine's log
        self.config_path = None                         # #564: the run config the web page's Settings view edits
        self.config_lock = threading.Lock()
        # #321: browser pages of these origins may call /v1/* (CORS; "*" = any page - only with an api_key that
        # matters); empty = no CORS headers at all, as before
        self.cors_origins: list[str] = []
        # #321: origins that count as Strata's own page for /settings and MCP tools, e.g. the web app reached through a
        # reverse proxy or tunnel whose Host differs ("https://strata.example.com"); never a wildcard
        self.trusted_origins: list[str] = []
        # DNS rebinding: extra Host names this server answers to (the config's allowed_hosts, $STRATA_ALLOWED_HOSTS;
        # "*" = any), and every name it answers to, which serve() works out from the address it listens on
        self.allowed_hosts: list[str] = []
        self.host_names: set[str] = set(LOOPBACK_NAMES)
        self.status = {"busy": False, "queued": 0, "loops_stopped": 0, "thinking_cycles_closed": 0}  # GET /status: what the model is doing right now
        self.rate = collections.deque(maxlen=32)        # (time, generated) samples for the live tok/s window
        # #310: who waits for the engine, in arrival order ({"who", "since"}), one entry per status["queued"] count
        # (_queue_enter / _queue_leave change both together)
        self.queue_who: list[dict] = []
        # "parallel" (the engine's batch slots): every running request's own status and rate window; self.status
        # then says busy while any runs and shows the newest one
        self.live_reqs: dict[int, tuple[dict, collections.deque]] = {}
        # #332: the API request monitor (/api-monitor) keeps the last 100 requests' prompts and answers in memory,
        # so it is off unless the config's "api_monitor" (or --api-monitor) turns it on
        self.api_monitor = False
        self.api_requests = collections.deque(maxlen=100)  # bounded I/O in memory; no headers or API keys
        self.request_trace = threading.local()
        self.history = collections.deque(maxlen=500)    # the last finished requests, newest last (GET /metrics)
        self.hstore = HistoryStore(Path(tempfile.gettempdir()) / "strata-history-off", enabled=False)  # main() turns it on
        self.checkpoints = checkpoints_mod.Checkpoints(Path(tempfile.gettempdir()) / "strata-checkpoints")      # the way back for files the tools change; main() puts it in the user's data folder
        self.model_info = None                          # the model's name and quantization, from its GGUF headers (main() fills it)
        self.keep_prompts = 0                           # POST /metrics/keep: the next N requests keep their full prompt (Q8)
        self.keep_lock = threading.Lock()
        self.ui = "next"                                # which web app "/" serves: the new one; config "ui": "classic" brings the old one back
        # since the server started (the Monitor's totals, issue #35)
        self.totals = {"since": time.time(), "requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0,
                       "prompt_ms": 0.0, "decode_ms": 0.0,
                       "drafts_offered": 0, "drafts_accepted": 0}   # #457: the MTP drafts, summed where reported
        self.last_timings = None                         # the last finished request's, llama.cpp's names (/v1/status)
        from serve.prometheus import Latencies
        self.latencies = Latencies()                     # GET /metrics in Prometheus text: the latency histograms
        self.last_request_at = None                      # when a request last started or finished
        self.started_at = time.time()
        self.status_lock = threading.Lock()
        self.mcp = None                                  # serve/mcp.py's McpHub when MCP servers are configured
        self.config_path = None                          # the run config (--config) and the --mcp-config file: where serve/mcp_admin.py
        self.mcp_config_path = None                      # reads and writes the servers the web app sets up
        self.importer = None                             # serve/harness.py's Importer: the other coding apps' skills and MCP servers (#94)
        self.strata = None                               # serve/strata_tools.py's StrataServer: the chat's StrataDocs / StrataSettings / StrataSettingsChange (#339)
        self.agent = None                                # serve/agent.py's AgentServer: the chat's coding tools (Read, Write, Edit, Bash, ...); None: switched off
        self.broker = agent_run.Broker()                 # the questions the coding tools have asked the page and not had answered yet
        self.runs = runs_mod.RunStore()                  # the answers that use the coding tools: they go on when the page that asked is refreshed or closed (serve/runs.py)
        self.runner_hub = runner_hub.RunnerHub()         # #310 part 2: the runners on the engineers' own PCs (serve/runner_hub.py)
        # sharing the GPU with other programs (all off by default): unload the engine after this many idle seconds,
        # only start it again when this much VRAM is free, and run this command first (e.g. to unload another
        # server's model); the next request after an unload starts the engine again
        self.idle_unload_s = 0
        self.min_free_vram_mib = 0
        self.before_load = None
        self.vram_reserve = None                         # #533: the last POST /v1/vram reserve (None: the start's)
        self.reasoning_loop_recovery = False
        self.reasoning_budget_tokens = 0                 # #123: the config's default thinking budget (0: none)
        self.reasoning_carry = "off"                     # #200: the config's reasoning_carry (off / history / resume)
        self.carry = reasoning_carry.Carry()             # #200: the latest reply cut while still thinking
        self.repeat_stop_tokens = REPEAT_STOP_TOKENS     # #606: one token this many times in a row ends a reply (0: off)
        self.anthropic_think_unasked = True               # #278: "anthropic_thinking": "on_request" -> False
        self.tool_call_recovery = False                   # opt-in: tool calls in forms next to the template's
        self.stop_ids = set(tokenizer.encode(IM_END, parse_special=True) +
                            tokenizer.encode("<|endoftext|>", parse_special=True))
        self.replays = collections.OrderedDict()        # xeno: replay_key -> the last greedy non-stream answer

    REPLAY_KEYS = ("temperature", "top_p", "top_k", "min_p", "stop_sequences", "_opening", "_think_budget")

    def side_request(self, system: str, user: str, max_tokens: int = 64) -> str:
        """The model's short answer to one question outside any chat (auto mode's judge, serve/judge.py).  Like Claude Code's own side requests
        it is not streamed, has no tools and thinks little, and the opening its prompt asks for is written for the model (serve/forced_opening.py)."""
        req = {"model": self.model, "system": system, "messages": [{"role": "user", "content": user}], "max_tokens": max_tokens, "stream": False,
               "thinking": {"type": "disabled"}}
        messages, tools, kw = anthropic_to_messages(req, vision=False)
        think_budget.side_effort(req, kw)
        ids, thinking, max_new = self.prepare(messages, None, kw, max_tokens, 1)
        req = self.with_slot(req, ids, route="judge")
        opening = forced_opening.required(messages, thinking)
        if opening:
            req = {**req, "_opening": opening}
        body = anthropic_collect(anthropic_events(self, req, ids, thinking, None, max_new, threading.Event()))
        return "".join(b.get("text", "") for b in body.get("content", []) if isinstance(b, dict) and b.get("type") == "text")

    def memory_blocks(self, folders: list[str]) -> list[dict]:
        """The notes the chat is handed: the project's own instruction files, and what other apps wrote down that the user switched on (serve/memory.py)."""
        imp = getattr(self, "importer", None)
        home = str(imp.home) if imp is not None else str(harness.home_dir())
        try:
            on = memory_mod.settings(harness._cfg(self))["on"]
        except Exception:  # noqa: BLE001 - a config that cannot be read means no other app's notes
            on = []
        return memory_mod.collect(home, folders, on)

    def start_agent_run(self, sa: dict, messages: list, who: str = "owner", runner=None, guest: bool = False, local: bool = False,
                        my_mcp: str | bool | None = None):
        """One request's use of the coding tools: the folder, mode, rules and chat the page named (anything odd is ignored), the rules for the
        AI as a prompt, and auto mode's judge (serve/agent_run.py).  `who`: the identity whose request it is; only it answers what the run asks (#310).
        `runner` (#310 part 2): the serve/runner_hub.py Runner of a project on the engineer's own PC.  Its `run.start` job (the facts job) gives
        the real folders, the shell, the platform, git and the notes, and the server touches none of its own disk, hooks or checkpoints for
        the run; web access, helpers, vision and the judge stay the server's.  RunnerUnavailable when the job fails (before any run is made).
        `guest`: a team guest's run - its WebFetch (it runs here, for the run and its helpers) reaches only the public internet, never this
        PC, a private network, Tailscale or Radmin's 26.0.0.0/8 (web.public_not_vpn, as #311's image URLs; final review I-1).
        `local`: the page is open on this PC itself (its rules then say the central PC is the user's own, #339).
        `my_mcp`: the request offers MCP groups of the user's PC (`my_<name>`; a str names that PC), in a My PC run or through
        #338's `strata_mcp_runner` in a Server PC chat, so the rules say where they run; None: a My PC run's own MCP groups."""
        mode = permissions.mode_for(sa.get("mode"), guest=guest, on_runner=runner is not None)      # #329: a guest's YOLO only on their own PC
        rules = lambda v: [x for x in v if isinstance(x, str) and 0 < len(x) <= 500][:200] if isinstance(v, list) else []      # noqa: E731
        sid = sa.get("session") if isinstance(sa.get("session"), str) and 0 < len(sa["session"]) <= 80 else "default"
        goal = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                goal = c if isinstance(c, str) else "".join(p.get("text", "") for p in c if isinstance(p, dict)) if isinstance(c, list) else ""
                break

        def read_conf() -> dict:
            try:
                return harness._cfg(self)
            except Exception:  # noqa: BLE001 - a config that cannot be read means the defaults: no web access, no hooks
                return {}
        sh = getattr(self.agent, "shell", None)
        cp = sa.get("checkpoint")
        if runner is None:
            folder, dirs = folders_mod.resolve(sa.get("cwd"), sa.get("dirs"), agent_mod.MAX_DIRS)     # the real ones, once each, not the main one; none without a main one
            shell, facts, conf = shell_mod.describe(sh) if sh else None, None, None
            protected = [p for p in (self.config_path, self.mcp_config_path) if p]                 # hooks, web access and keys live there: never changed unasked
        else:
            conf = read_conf()
            path = lambda v: isinstance(v, str) and bool(v.strip()) and "\0" not in v and len(v) <= 4096    # noqa: E731 - the runner finds the real ones
            raw_dirs = sa.get("dirs") if isinstance(sa.get("dirs"), list) else []
            facts = runner_hub.run_facts(runner.request("run.start", {
                "cwd": sa.get("cwd") if path(sa.get("cwd")) else None, "dirs": [d for d in raw_dirs[:200] if path(d)], "mode": mode,
                "allow": rules(sa.get("allow")), "deny": rules(sa.get("deny")), "session": sid,
                "checkpoint": cp if isinstance(cp, str) and len(cp) <= 200 and sid != "default" else None,
                "memory_on": memory_mod.settings(conf)["on"]}), runner.info.get("machine") or "?", agent_mod.MAX_DIRS, memory_mod.MAX_TOTAL)
            folder, dirs, shell = facts["cwd"], facts["dirs"], facts["shell"]
            protected = []                                              # the runner keeps its own files from the tools; the server's are not there
        policy = permissions.Policy(cwd=folder, mode=mode, allow=rules(sa.get("allow")), deny=rules(sa.get("deny")), dirs=dirs, protected=protected)
        run = agent_run.AgentRun(policy, sid, self.broker, goal, self.side_request, threading.Event(), shell=shell, who=who)
        run.ctx.guest = guest                                           # #339: the server's settings are the owner's: a guest reads only the docs
        if facts is None:
            if isinstance(cp, str) and sid != "default":
                run.ctx.checkpoint = self.checkpoints.scope(sid, cp)              # this prompt's checkpoint: the files the tools change are kept as they were
            git = folders_mod.in_git(folder)                            # the folder or one of the folders above it holds .git
            conf = read_conf()
        else:                                                           # the runner keeps the checkpoint and found .git
            run.ctx.run_key, run.coding, run.runner, git = facts["run"], "my_agent", runner, facts["git"]
        web = web_mod.settings(conf)
        if guest and not (isinstance(conf, dict) and isinstance(conf.get("web"), dict) and conf["web"].get("on") is True):
            web = {**web, "on": False}                                  # #337: on by default for the owner; a team guest only when the owner switched it on outright
        run.hidden = set() if web["on"] else set(getattr(self.agent, "web_tools", ()))          # web access off ("on": false): the model is not even offered the tools
        if web["on"]:                                                   # a guest's fetches leave from this PC: not into the team's own network
            run.ctx.web = web_mod.Web(web, fetcher=web_mod.Fetcher(guard=web_mod.public_not_vpn) if guest else None)
        helpers = getattr(self.agent, "helper_tools", ())
        if subagent_mod.settings(conf)["on"] and helpers:                                         # helpers are off until the user switches them on, for the same reason
            run.ctx.spawn = lambda kind, prompt, description: subagent_mod.execute(self, run, kind, prompt, description, run_with_mcp)
        else:
            run.hidden |= set(helpers)
        if guest:
            run.hidden |= set(strata_tools.OWNER_TOOLS)
        strata = [t["name"] for t in getattr(self.strata, "tools", None) or [] if t["name"] not in run.hidden]      # #339: Strata's own tools, in the rules' list
        if facts is None:
            tools = [t["name"] for t in self.agent.tools if (t["name"] != "ExitPlanMode" or mode == "plan") and t["name"] not in run.hidden] + strata
            run.prompt = agent_prompt.build(folder, shell, mode, time.strftime("%Y-%m-%d"), sys.platform, git, None, tools, dirs, self.memory_blocks([folder, *dirs] if folder else []),
                                            same_pc=local, my_tools=my_mcp or False)
        else:                                                           # the runner's coding tools, then the server's own that run here
            ours = {t["name"] for t in getattr(self.agent, "tools", None) or []}
            names = [n for n in facts["tools"] if n not in runner_hub.SERVER_TOOLS and runner_hub.plain_name_ok(n)] + [n for n in runner_hub.SERVER_TOOLS if n in ours]
            tools = [n for n in names if (n != "ExitPlanMode" or mode == "plan") and n not in run.hidden] + strata
            mine = runner.servers() if hasattr(runner, "servers") else {}
            run.prompt = agent_prompt.build(folder, shell, mode, time.strftime("%Y-%m-%d"), facts["platform"], git, None, tools, dirs, facts["memory"],
                                            machine=runner.info.get("machine") or "?",
                                            my_tools=any(getattr(g, "group_kind", "") == "mcp" for g in mine.values()) if my_mcp is None else my_mcp)     # #339: where each tool runs
        run.ctx.vision = self.vision is not None and not self._vision_down()          # Read can give an image to the model only when the vision encoder is there
        if facts is not None:                                           # no hooks for a project on the engineer's PC (v1): the server's run on the server
            return run
        defined, _problems = hooks_mod.load(conf)
        hooks = hooks_mod.Runner(defined, sh, folder, sid, run._emit, lambda: run.cancel.is_set(), begin=run._emit)
        if hooks:
            run.ctx.hooks = run.hooks = hooks
            said = hooks.prompt(goal)                                    # the user's own prompt hooks: what they print goes to the model with the rules
            if said:
                run.prompt += "\n\n" + said
        return run

    def replay_key(self, req: dict, ids, max_new):
        """The key of a request whose answer can be given again, so Claude Code's repeats of one classifier request
        cost nothing: only the classifier's fast stage (a required opening, serve/forced_opening.py) and only
        greedy (no temperature, or 0 - the engine then writes the same tokens for the same prompt).  Any other
        repeated request is generated again: a benchmark that repeats a prompt must never time a replay."""
        if not req.get("_opening"):
            return None
        eff = {**self.sampling_defaults, **self.shared, **{k: v for k, v in req.items() if v is not None}}
        try:
            if float(eff.get("temperature") or 0) > 0:
                return None
        except (TypeError, ValueError):
            return None
        return (tuple(ids), max_new, json.dumps({k: eff.get(k) for k in self.REPLAY_KEYS}, sort_keys=True))

    def remember_reply(self, key, body: dict) -> None:
        if body.get("stop_reason") in ("end_turn", "max_tokens", "stop_sequence"):
            self.replays[key] = body
            self.replays.move_to_end(key)
            while len(self.replays) > 4:
                self.replays.popitem(last=False)

    def loaded(self) -> bool:
        return not hasattr(self.engine, "alive") or self.engine.alive()

    def key_required(self) -> bool:
        """Does a request need a key: the api_key is set, or the server is in team mode (#310: a keys file, even an
        empty or broken one - it must not turn into an open server)."""
        return bool(self.api_key) or self.keys.team_mode()

    def slot_action(self, slot: str, action: str, filename, who: str = "owner") -> tuple[int, dict]:
        """llama-server's POST /slots/{id}?action=save|restore {"filename": ...}: the conversation the engine holds,
        to or from a file in slot_save_path.  -> (HTTP status, body)."""
        def error(code, message):
            return code, {"error": {"code": code, "message": message,
                                    "type": "invalid_request_error" if code < 500 else "server_error"}}
        if not self.slot_save_path or not hasattr(self.engine, "session_file"):
            return error(501, "slot save/restore is disabled (start the server with --slot-save-path DIR)")
        if getattr(self.engine, "batch", 0):
            # #465 parallel requests do not hold the FIFO and share the engine's control lines: a session file
            # could interleave with a request being admitted, so the two are not combined (the engine refuses too)
            return error(501, "slot save/restore is not available with parallel requests (\"parallel\" / --batch)")
        if slot != "0":
            return error(400, "this server has one slot: 0")
        if action not in ("save", "restore"):
            return error(400, "action must be save or restore")
        why = slot_filename_problem(filename)
        if why:
            return error(400, f"filename {why}")
        path = os.path.join(self.slot_save_path, filename)
        if action == "restore":
            try:
                os.lstat(path)                              # the engine opens it without following a link
            except OSError:
                return error(404, f"no saved session named {filename}")
        self.last_request_at = time.time()                  # the idle unload counts this as activity
        with self.status_lock:
            place = self._queue_enter(who)
        waiting = True
        try:
            with self.fifo:
                with self.status_lock:
                    self._queue_leave(place)
                waiting = False
                if not self.loaded():
                    return error(503, "the model is not loaded")
                with self.status_lock:
                    self.status.update(busy=True, phase="saving the session" if action == "save"
                                       else "restoring a session", started=time.time(), first_token=None,
                                       prompt_tokens=None, generated=None, max_tokens=None, who=who)
                try:
                    if action == 'save' and getattr(self.engine, 'session_save_reclaim', False):
                        from serve.session_save_retry import session_file
                        r = session_file(self, action, path, SessionRefused)
                    else:
                        r = self.engine.session_file(action, path)
                except SessionRefused as e:
                    body = error(e.status, str(e))
                    body[1]["error"]["kind"] = e.kind
                    if e.published:
                        body[1]["error"]["published"] = True
                    return body
                except ValueError as e:                     # the request itself (an invalid command)
                    return error(400, str(e))
                finally:
                    with self.status_lock:
                        self.status["busy"] = False
                    self.last_request_at = time.time()
        finally:
            if waiting:                                     # never reached the FIFO (an exception while waiting)
                with self.status_lock:
                    self._queue_leave(place)
        if action == "save":
            return 200, {"id_slot": 0, "filename": filename, "n_saved": r["tokens"], "n_written": r["bytes"],
                         "timings": {"save_ms": r["ms"]}}
        return 200, {"id_slot": 0, "filename": filename, "n_restored": r["tokens"], "n_read": r["bytes"],
                     "timings": {"restore_ms": r["ms"]}}

    def set_aliases(self, aliases) -> None:
        """#297: the config's `aliases` - a list of names (or one comma-separated string), like llama-server's --alias.
        ValueError for anything else."""
        if aliases is None:
            aliases = []
        if isinstance(aliases, str):
            aliases = aliases.split(",")
        if not isinstance(aliases, list) or not all(isinstance(x, str) for x in aliases):
            raise ValueError(f"aliases={aliases!r}: expected a list of model names")
        names = []
        for x in (x.strip() for x in aliases):
            if x and x != self.model and x not in names:
                names.append(x)
        self.aliases = names

    def model_names(self) -> list[str]:
        return [self.model, *self.aliases]

    def model_for(self, req) -> str:
        """The name to answer with: the request's own when it is the model's name or an alias (#297), else the model's.
        Other names are still served, as before."""
        asked = req.get("model") if isinstance(req, dict) else None
        return asked if isinstance(asked, str) and asked in self.aliases else self.model

    def reported_ctx(self) -> int:
        """The context the endpoints report (/v1/status, /v1/models, /props, /health, /slots, /metrics): the engine's,
        or the last one it reported while it restarts (max_context is 0 until READY, and a client that sizes its
        prompt from it would send nothing, #351).  Reporting only: requests are checked in prepare()."""
        return int(getattr(self.engine, "max_context", 0) or getattr(self.engine, "known_ctx", 0) or 0)

    def reasoning_budget(self, req) -> int | None:
        """#123: the most tokens this request may think, or None: the request's `reasoning_budget_tokens`, else the
        config's.  0 (or less) means no budget, so a request can turn a configured one off.  ValueError (a 400) for
        anything that is not a whole number."""
        value = (req or {}).get("reasoning_budget_tokens") if isinstance(req, dict) else None
        if value is None:
            value = self.reasoning_budget_tokens
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"reasoning_budget_tokens={value!r}: expected a whole number of tokens (0: no budget)")
        return value if value > 0 else None

    def _vision_down(self) -> bool:
        return self.vision is not None and hasattr(self.vision, "alive") and not self.vision.alive()

    def free_vram_mib(self, index: int | None = None) -> int | None:
        """Free VRAM on the engine's (first) GPU, or device `index` (#59), from NVML (AMD backend: amdgpu's sysfs files,
        #301); None when it can't be read (then nothing is refused)."""
        try:
            dev = int(getattr(self, "gpu_index", 0) or 0) if index is None else index
            if getattr(self, "backend", None) == "hip":
                from serve.telemetry import free_vram_mib
                return free_vram_mib(dev, amd=True)
            from serve.telemetry import _Nvml
            nv = _Nvml(dev)
            if not nv.ok():
                return None
            m = nv.Mem()
            if nv.lib.nvmlDeviceGetMemoryInfo(nv.dev, ctypes.byref(m)) != 0:
                return None
            return int(m.free >> 20)
        except Exception:
            return None

    def ensure_loaded(self):
        """Start the engine if it is not running (unloaded, or it died - issue #27), after the before_load hook and
        the free-VRAM check.  The caller holds self.fifo."""
        if self.loaded() and not self._vision_down():
            return
        if self.before_load:
            cmd = self.before_load
            print(f"[strata] before loading: {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}", flush=True)
            try:
                subprocess.run(cmd, shell=isinstance(cmd, str), timeout=120, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.SubprocessError) as e:
                print(f"[strata] the before_load command failed ({e}); loading anyway", flush=True)
        if self.min_free_vram_mib:
            free = self.free_vram_mib()
            deadline = time.time() + 15                 # memory another process just gave back can take a moment
            while free is not None and free < self.min_free_vram_mib and time.time() < deadline:
                time.sleep(0.5)
                free = self.free_vram_mib()
            if free is not None and free < self.min_free_vram_mib:
                raise GpuBusy(f"the GPU is in use by another program: {free} MiB of VRAM free, the model needs "
                              f"{self.min_free_vram_mib} (min_free_vram_mib) - it stays unloaded until that is free")
        vision_started = False
        if self._vision_down():                         # first, as at a start: a GPU encoder takes its VRAM before
            print("[strata] starting the vision encoder ...", flush=True)   # the engine sizes its cache
            try:
                self.vision.restart()
            except Exception as e:
                raise EngineDied(f"the vision encoder failed to start: {e}") from e
            vision_started = True
        if self.loaded():
            return
        if getattr(self.engine, "unloaded", False):
            print("[strata] loading the model again (it was unloaded) ...", flush=True)
        else:
            code = self.engine.exit_code() if hasattr(self.engine, "exit_code") else None
            print(f"[strata] the engine had stopped (exit code {code}); starting it again "
                  "(a minute or two) ...", flush=True)
        self.restarting = True
        try:
            self.engine.restart()
        except Exception as e:
            if vision_started:
                self.vision.unload()
            # xeno #49 review: a 529, not a dropped connection (EngineStuck keeps its own answer: a restart that could
            # not end the engine)
            if isinstance(e, (RuntimeError, OSError)) and not isinstance(e, (EngineDied, EngineStuck)):
                raise EngineDied(f"the engine could not be started again: {e}") from e
            raise
        finally:
            self.restarting = False
        print("[strata] the engine is running again", flush=True)
        if self.vram_reserve is not None and hasattr(self.engine, "vram"):   # #533: the reserve asked for last
            try:
                self.engine.vram(self.vram_reserve)
            except (ValueError, EngineDied) as e:
                print(f"[strata] the VRAM reserve ({self.vram_reserve} MiB) was not applied: {e}", flush=True)

    vram_wait_s = 300.0                                  # #533: how long POST /v1/vram waits for a running request

    def vram(self, reserve_mib: int | None) -> dict:
        """#533, POST /v1/vram: keep `reserve_mib` of VRAM free for other programs (None: the reserve the engine
        started with), applied between requests - a request that is running finishes first (up to wait_s).  Only an
        engine started with --vram-elastic (the config's "vram_elastic": true) can do it; it never resizes on its own.
        An unloaded engine applies it when it loads.  -> {"status": ..., and the engine's figures}."""
        if not hasattr(self.engine, "vram"):
            raise ValueError("this engine cannot resize its VRAM use")
        if not self.fifo.acquire(timeout=self.vram_wait_s):
            raise ModelBusy("a request is still running; try again when it has finished")
        # parallel requests do not hold the fifo: the engine's control lines (a prompt being admitted) are taken too,
        # and the engine refuses a resize while a slot decodes
        ctl = getattr(self.engine, "ctl", None) if getattr(self.engine, "batch", 0) else None
        if ctl is not None and not ctl.acquire(timeout=self.vram_wait_s):
            self.fifo.release()
            raise ModelBusy("a request is still running; try again when it has finished")
        try:
            if not self.loaded():
                self.vram_reserve = reserve_mib
                return {"status": "not loaded", "reserve_mib": reserve_mib,
                        "note": "applied when the model loads"}
            out = self.engine.vram(reserve_mib)
            self.vram_reserve = reserve_mib
            print(f"[strata] VRAM: {out.get('vram_free_mib')} MiB free, expert cache {out.get('expert_cache_mib')} of "
                  f"{out.get('expert_cache_full_mib')} MiB ({out.get('expert_slots')} experts)", flush=True)
            return {"status": "ok", **out}
        finally:
            if ctl is not None:
                ctl.release()
            self.fifo.release()

    def _say_died(self, e: Exception) -> None:
        """The server window's line for an engine that died (or was ended, #481) in the middle of a request."""
        note = self.engine.death_note() if hasattr(self.engine, "death_note") else ""
        log = getattr(self.engine, "log_path", None)
        print(f"[strata] {e}. {note} The next request starts the engine again."
              f"{' Its log: ' + log if log else ''}", flush=True)

    def load(self, wait_for_restart: bool = False, who: str = "owner"):
        """POST /load and every generation request: start the engine now if it is unloaded (raises GpuBusy).  While
        another request (re)loads it: EngineDied at once (xeno #49 S2, the Anthropic API's retryable 529), or, with
        `wait_for_restart` (the OpenAI API), a turn in the queue until it is loaded (#208, #332).  That wait is this
        method's only: a restart that begins after it returned (the #59 watch's) still ends the request at once, from
        run()'s `restarting` check or prepare()'s EngineStarting (529 on /v1/messages, 503 elsewhere; merge of 0.1.34)."""
        # a request is on its way: the idle thread must not unload between this and the request's own start
        self.last_request_at = time.time()
        if self.loaded() and not self._vision_down():
            return
        if self.restarting and not wait_for_restart:   # xeno #49 S2: a request while the engine loads again is a
            raise EngineDied("the engine is loading again after it stopped")   # 529 at once
        trace = getattr(self.request_trace, "record", None)
        waiting = time.perf_counter()
        with self.status_lock:                          # #1603: waiting for the fifo to start the engine is queued, not nothing
            place = self._queue_enter(who)
        counted = True
        try:
            with self.fifo:
                with self.status_lock:
                    self._queue_leave(place)
                counted = False
                loading = time.perf_counter()
                if trace is not None:
                    with self.status_lock:
                        trace["queue_s"] += round(loading - waiting, 3)
                        trace["state"] = "loading"
                try:
                    self.ensure_loaded()
                finally:
                    if trace is not None:
                        with self.status_lock:
                            trace["load_s"] += round(time.perf_counter() - loading, 3)
                            trace["state"] = "queued"
        finally:
            if counted:
                with self.status_lock:
                    self._queue_leave(place)

    def _queue_enter(self, who: str) -> dict:
        """#310: one more request waits for the engine (the caller holds status_lock): status["queued"] counts it and
        queue_who names it, in arrival order.  Returns its place, for _queue_leave."""
        self.status["queued"] += 1
        place = {"who": who, "since": time.time()}
        self.queue_who.append(place)
        return place

    def _queue_leave(self, place: dict) -> None:
        """It waits no more (the caller holds status_lock): its turn came, it was cancelled or it failed."""
        self.status["queued"] -= 1
        for i, x in enumerate(self.queue_who):
            if x is place:
                del self.queue_who[i]
                break

    def unload(self, idle_for: float | None = None) -> str:
        """Stop the engine between requests: "unloaded", "not loaded", "busy" (a request is running or waiting, or
        with idle_for: one ran more recently than that) or "unsupported"."""
        if not hasattr(self.engine, "unload"):
            return "unsupported"
        if not self.fifo.acquire(blocking=False):
            return "busy"
        try:
            if not self.engine.alive():
                return "not loaded"
            with self.status_lock:
                if self.status.get("busy") or self.status.get("queued"):
                    return "busy"
            if idle_for is not None and time.time() - (self.last_request_at or self.started_at) < idle_for:
                return "busy"
            self.engine.unload()
            if self.vision is not None and hasattr(self.vision, "unload"):
                self.vision.unload()
            print(f"[strata] model unloaded{f' after {idle_for:.0f} s idle' if idle_for else ''}; "
                  "the next request loads it again", flush=True)
            return "unloaded"
        finally:
            self.fifo.release()

    def display_free_mib(self) -> int | None:
        """#59: free VRAM on the secondary GPU (the second of CUDA_VISIBLE_DEVICES, the 4070 here), from NVML."""
        vis = [v for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
        if len(vis) < 2 or not vis[1].strip().isdigit():
            return None
        return self.free_vram_mib(int(vis[1]))

    def watch_once(self, upgrade_idle_s: float = 300):
        """#59: between requests, start an engine that died by itself again (no request has to pay for it), and give a
        degraded engine (no 4070 tier) its tier back once the server has been idle and the display has the room."""
        eng = self.engine
        if getattr(eng, "unloaded", False) or self.restarting or not hasattr(eng, "restart"):
            return
        if not self.fifo.acquire(blocking=False):      # a request holds the engine: it handles a death itself
            return
        try:
            if not self.loaded():
                try:
                    self.ensure_loaded()
                except (EngineDied, GpuBusy) as e:
                    print(f"[strata] the engine could not be started again yet: {e}", flush=True)
            elif (getattr(eng, "degraded", False) and
                  time.time() - (self.last_request_at or self.started_at) >= upgrade_idle_s):
                need = eng.secondary_need_mib() if hasattr(eng, "secondary_need_mib") else None
                free = self.display_free_mib() if need is not None else None
                if free is not None and free >= need:
                    print(f"[strata] the display has room again ({free} MiB free): restoring the 4070 tier ...",
                          flush=True)
                    self.restarting = True
                    try:
                        eng.restart(full=True)
                    except (EngineDied, RuntimeError, OSError) as e:
                        print(f"[strata] restoring the 4070 tier failed ({e}); the next check tries again", flush=True)
                    finally:
                        self.restarting = False
        finally:
            self.fifo.release()

    def start_engine_watch(self, every_s: float = 5.0):
        """#59: watch_once every few seconds on a daemon thread (a real engine only: it can restart)."""
        if not hasattr(self.engine, "restart"):
            return

        def loop():
            while True:
                time.sleep(every_s)
                try:
                    self.watch_once()
                except Exception as e:                  # the watch must never end the server
                    print(f"[strata] engine watch: {e}", flush=True)
        threading.Thread(target=loop, daemon=True).start()

    def start_idle_unload(self):
        if not self.idle_unload_s or not hasattr(self.engine, "unload"):
            return
        print(f"[strata] the model unloads after {self.idle_unload_s} s without requests", flush=True)

        def loop():
            while True:
                time.sleep(max(1.0, min(30.0, self.idle_unload_s / 4)))
                try:
                    self.unload(idle_for=self.idle_unload_s)
                except EngineStuck as e:                # tried again at the next turn; the thread keeps running
                    print(f"[strata] idle unload: {e}", flush=True)
        threading.Thread(target=loop, daemon=True).start()

    def set_shared(self, defaults) -> dict:
        """The Chat settings every client gets for what it leaves out; {} / None = clients use their own again."""
        self.shared = clean_shared_defaults(defaults)
        if self.shared_path:
            try:
                if self.shared:
                    Path(self.shared_path).write_text(json.dumps(self.shared, indent=1), encoding="utf-8")
                else:
                    Path(self.shared_path).unlink(missing_ok=True)
            except OSError as e:
                print(f"[strata] could not save the shared settings: {e}", flush=True)
        return self.shared

    SLOT_RULES = {"family": CacheSlots, "identity": IdentityCacheSlots, "continuation": ContinuationCacheSlots,
                  "hybrid": HybridCacheSlots}

    def with_slot(self, req: dict, ids, who: dict | None = None, route: str = "") -> dict:
        """The request with the cache slot of its prompt (xeno #49 S7), when the engine has several slots (INFO
        cache_slots) and the client chose none.  The rule is STRATA_CACHE_SLOT_RULE (#321): unset or "family" the
        prompt-family rule (CacheSlots), "identity" the family per sender (IdentityCacheSlots), "continuation" the
        slot the prompt continues (ContinuationCacheSlots), "hybrid" by what a placement saves and destroys
        (HybridCacheSlots); those two also record a client's own strata_cache_slot.  `who`
        ({"key", "device", "addr"}, the HTTP handler's _slot_who; None for the server's own requests) and `route`
        ("openai", "anthropic", "judge") go to STRATA_SLOT_TRACE as well."""
        trace = os.environ.get("STRATA_SLOT_TRACE")
        if trace:
            slot_trace(trace, req, ids, who, route, getattr(self.engine, "info", None))
        try:
            n = int((getattr(self.engine, "info", {}) or {}).get("cache_slots", 1))
        except (TypeError, ValueError):
            n = 1
        if n <= 1:
            return req
        name = os.environ.get("STRATA_CACHE_SLOT_RULE", "").strip().lower()
        kind = self.SLOT_RULES.get(name, CacheSlots)
        family = kind in (CacheSlots, IdentityCacheSlots)
        if getattr(self, "slots", None) is None or self.slots.n != n or type(self.slots) is not kind:
            self.slots = kind(n)
            # #321 review: say which rule runs (once, here where it is made); a misspelt name is not silently family
            said = f"[strata] cache-slot rule: {name or 'family'} ({n} slots)"
            if name and name not in self.SLOT_RULES:
                said = f"[strata] cache-slot rule: unknown STRATA_CACHE_SLOT_RULE={name!r}, running family ({n} slots)"
            print(said, flush=True)
        if "strata_cache_slot" in req:
            pinned = req["strata_cache_slot"]           # a bad one is the request's 400 (cache_slot), not ours
            if isinstance(pinned, int) and not isinstance(pinned, bool) and 0 <= pinned < n:
                self.slots.record(pinned, ids)
            return req
        key = (who.get("key"), who.get("device")) if isinstance(who, dict) else None
        nearest = {}
        slot, shared, how = self.slots.choose(ids, nearest, who=key)
        if os.environ.get("STRATA_DEBUG"):              # where the prompt leaves its slot's last one, and its turns
            start = self.tok.encode("<|im_start|>", parse_special=True)
            turns = [i for i, t in enumerate(ids) if len(start) == 1 and t == start[0]]
            if family:
                said = f"slot {slot}: {len(ids)} tokens, shares {shared} with the family's last"
                if kind is IdentityCacheSlots:
                    said += f" (sender {key})"
            elif kind is HybridCacheSlots:              # what it reuses of the slot's last prompt, and that prompt
                said = f"slot {slot} ({how}): {len(ids)} tokens, shares {shared} of its last {nearest.get('last', 0)}"
            else:                                       # the nearest slot's tail: what TAIL would have to cover
                near = (f"; nearest: slot {nearest['slot']} shares {nearest['shared']}, tail {nearest['tail']}"
                        if nearest and how != "continues" else "")
                said = f"slot {slot} ({how}): {len(ids)} tokens, shares {shared} with its last{near}"
            print(f"[strata] {said}, turns at {turns[:3]}{'...' if len(turns) > 6 else ''}"
                  f"{turns[-3:] if len(turns) > 3 else ''}", flush=True)
        req = {**req, "strata_cache_slot": slot}
        if shared:                                      # the engine keeps a checkpoint there (GEN key ckpt_at): the
            req["_ckpt_at"] = shared                    # family's next prompt shares it (a transcript that grows
        return req                                      # inside one message has no turn start to resume from)

    def cjk(self, req: dict, messages) -> dict:
        """The request with the Han ban on when the engine has the list and the prompt wants it (xeno #49 S4)."""
        return {**req, "_ban": True} if self.cjk_ban and cjk_guard.wanted(messages) else req

    def with_shared(self, req: dict, api: str) -> dict:
        """The request with the shared thinking level and max tokens filled in where it has none of its own."""
        s = self.shared
        effort = s.get("reasoning_effort") or self.config_effort   # the Chat settings first, then the config's
        if not s and not effort:
            return req
        req = dict(req)
        if "max_tokens" in s and not req.get("max_tokens") and not req.get("max_completion_tokens"):
            req["max_tokens"] = s["max_tokens"]
        if effort:
            if api == "openai":
                ctk = req.get("chat_template_kwargs") if isinstance(req.get("chat_template_kwargs"), dict) else {}
                if not req.get("reasoning_effort") and not req.get("reasoning") and \
                        "enable_thinking" not in ctk and "reasoning_effort" not in ctk:
                    req["reasoning_effort"] = effort
            elif not req.get("thinking") and not req.get("output_config"):
                if effort == "none":
                    req["thinking"] = {"type": "disabled"}
                else:
                    req["output_config"] = {"effort": effort}
        return req

    def meta_for(self, dialect, messages, tools, user_agent, who: str = "owner") -> dict:
        """The request's history meta; while "keep the next N prompts" is on it also carries the full prompt (`_prompt`,
        which run() moves to the detail file and never into the summary row).  `who`: the identity that sent it (#310),
        which run() also shows as the engine's holder or a waiting request."""
        meta = request_meta(dialect, messages, tools, user_agent, who)
        with self.keep_lock:
            if self.keep_prompts > 0:
                self.keep_prompts -= 1
                meta["_prompt"] = prompt_for_keep(messages)
        return meta

    def start_telemetry(self):
        """The hardware sampler behind GET /metrics (serve/telemetry.py), recording this server's tok/s too."""
        if getattr(self, "telemetry", None) is None:
            from serve.telemetry import Telemetry
            self.telemetry = Telemetry(extra=lambda: {"tok_s": self._tok_s(), "tok_s_mean": self._tok_s_mean(),
                                                    "prefill_tok_s_mean": self._prefill_tok_s_mean()},
                                       gpu_index=int(getattr(self, "gpu_index", 0) or 0),
                                       gpu_indices=getattr(self, "gpu_indices", None),
                                       busy_fn=lambda: bool(self.status.get("busy")),      # 5 Hz while a request runs
                                       model_path=getattr(self.engine, "model_path", None),
                                       amd=getattr(self, "backend", None) == "hip",
                                       # xeno #237: the engine's working set and faults; the per-second record goes to
                                       # the request history, which decides whether to keep it (history.telemetry)
                                       pid_fn=lambda: getattr(getattr(self.engine, "proc", None), "pid", None),
                                       sink=lambda p: self.hstore.append_telemetry(p))
            if getattr(self, "engine", None) is not None and hasattr(self.engine, "gpu_busy"):
                # #1317: the quick frozen-engine check does not end an engine whose GPU is busy (a long prompt chunk on a
                # slow card keeps the GPU at work while the host thread sleeps)
                self.engine.gpu_busy = lambda: (self.telemetry.snapshot()["now"].get("gpu_util") or 0) >= 10

    def _tok_s(self):
        """tok/s over the last RATE_WINDOW_S seconds.  Returns 0.0 while nothing is generating.  With parallel requests:
        the sum of theirs."""
        with self.status_lock:
            many = [(dict(st), list(rt)) for st, rt in self.live_reqs.values()]
        if many:
            return sum(self._rate_of(st, rt) for st, rt in many)
        with self.status_lock:
            s = dict(self.status)
            rate = list(self.rate)
        return self._rate_of(s, rate)

    @staticmethod
    def _rate_of(s, rate):
        if not s.get("busy") or not s.get("first_token"):
            return 0.0
        now = time.time()
        newest = rate[-1] if rate else None
        oldest = next(((t, g) for t, g in rate if now - t <= RATE_WINDOW_S), None)
        if newest and oldest and newest[0] - oldest[0] >= RATE_MIN_SPAN_S:
            return max(0.0, (newest[1] - oldest[1]) / (newest[0] - oldest[0]))
        return s["generated"] / max(RATE_MIN_SPAN_S, now - s["first_token"])

    def _tok_s_mean(self):
        """The whole-request mean since the first token (the old formula), kept so the two can be compared."""
        with self.status_lock:
            many = [dict(st) for st, _ in self.live_reqs.values()] or [dict(self.status)]
        return sum(s["generated"] / max(1e-6, time.time() - s["first_token"]) for s in many
                   if s.get("busy") and s.get("first_token"))

    def _prefill_tok_s_mean(self):
        """Engine-reported mean over newly read tokens, excluding the cached prefix."""
        with self.status_lock:
            reading = self.status.get("busy") and self.status.get("first_token") is None
        return getattr(self.engine, "prefill_tok_s_mean", None) if reading else 0.0

    def begin_request(self, path, req, who: str = "owner"):
        """#332: a monitor record for this request, or None when the monitor is off (nothing is kept then).  `who`: the
        identity that sent it (#310)."""
        if not self.api_monitor:
            return None
        raw = json.dumps(req, ensure_ascii=False, indent=2)
        record = {"id": uuid.uuid4().hex[:12], "path": path, "model": req.get("model") or self.model,
                  "started_at": time.time(), "state": "queued", "stream": bool(req.get("stream")),
                  "response_format": (req.get("response_format") or {}).get("type")
                  if isinstance(req.get("response_format"), dict) else None,
                  "input": raw[:262144], "input_truncated": len(raw) > 262144,
                  "output": "", "reasoning": "", "output_truncated": False, "reasoning_truncated": False,
                  "queue_s": 0.0, "load_s": 0.0, "first_token_s": None, "who": who,
                  "_clock": time.perf_counter()}
        with self.status_lock:
            self.api_requests.append(record)
        self.request_trace.record = record
        return record

    def request_records(self, request_id=None, viewer: identity_mod.Identity = identity_mod.OWNER):
        """#332's records as `viewer` may see them (#310): another person's request is not found by id, and in the
        list it shows who, state, queue place and timings only (identity.visible_record)."""
        with self.status_lock:
            records = list(self.api_requests)
            if request_id:
                record = next((r for r in records if r["id"] == request_id), None)
                if record is None or not self.may_read(record, viewer):
                    return None
                return {k: v for k, v in record.items() if not k.startswith("_")}
            return [identity_mod.visible_record(
                    {k: v for k, v in r.items()
                     if k not in ("input", "output", "reasoning", "response") and not k.startswith("_")}
                    | {"wallclock_s": r.get("wallclock_s", round(time.perf_counter() - r["_clock"], 3))}, viewer)
                    for r in reversed(records)]

    @staticmethod
    def may_read(record: dict, viewer: identity_mod.Identity) -> bool:
        """#310: may `viewer` read what a request asked and answered - the owner, or the identity that sent it."""
        return viewer.is_owner or record.get("who") == viewer.name

    def metrics(self, all_requests=False, viewer: identity_mod.Identity = identity_mod.OWNER) -> dict:
        """GET /metrics: what the Monitor tab shows - the engine's facts, what it is doing, the last requests, and
        the hardware (with a minute of history per series).  #310: the requests as `viewer` may see them; who holds
        the engine and who waits for it, by name."""
        with self.status_lock:
            s = dict(self.status)
            if self.live_reqs:                          # parallel requests: the newest one's status
                s = {**s, **dict(list(self.live_reqs.values())[-1][0]), "queued": s.get("queued", 0)}
            hist = list(self.history)
            totals = dict(self.totals)
            waiting = [dict(x) for x in self.queue_who]
        now = time.time()
        progress = getattr(self.engine, "progress", None)
        if s.get("busy") and s.get("first_token") is None:
            state = "reading"
        elif s.get("busy"):
            state = "generating"
        elif not self.loaded():
            state = "unloaded"
        else:
            state = "idle"
        live = {"state": state, "queued": s.get("queued", 0), "phase": s.get("phase") if s.get("busy") else None,
                "prompt_tokens": s.get("prompt_tokens") if s.get("busy") else None,
                "prompt_read": None, "prompt_total": None, "generated": s.get("generated") if s.get("busy") else None,
                "max_tokens": s.get("max_tokens") if s.get("busy") else None,
                "elapsed_s": round(now - s["started"], 1) if s.get("busy") and s.get("started") else None,
                "tok_s": round(self._tok_s(), 1) if state == "generating" else None,
                "tok_s_mean": round(self._tok_s_mean(), 1) if state == "generating" else None,
                "prefill_tok_s_mean": getattr(self.engine, "prefill_tok_s_mean", None) if s.get("busy") else None,
                "tok_s_window_s": RATE_WINDOW_S if state == "generating" else None,
                # #310: the identity whose request has the engine (with "parallel": each busy slot's "who"), and the
                # ones waiting for it, in arrival order (in both modes; "waiting" is #465's count, with "parallel" only)
                "holder": s.get("who") if s.get("busy") else None, "waiting_who": waiting}
        if state == "reading" and progress:
            live["prompt_read"], live["prompt_total"] = progress
        par = int(getattr(self.engine, "batch", 0) or 0)
        if par:                                         # #465: the requests running together, slot by slot
            with self.status_lock:
                running = len(self.live_reqs)
            slots = self.engine.slots_view() if hasattr(self.engine, "slots_view") else []
            # the requests in flight that are not in a slot: the one alone on the solo path, or waiting their turn
            in_slots = sum(1 for x in slots if x["state"] != "idle")
            # "holder" is each busy slot's "who" here (#310)
            live.update(parallel=par, running=running, slots=slots, outside_slots=max(0, running - in_slots),
                        waiting=int(getattr(self.engine, "waiting", 0) or 0),
                        admitting=int(getattr(self.engine, "admitting", 0) or 0), holder=None)
            if running and state == "idle":
                live["state"] = "generating"
        if hasattr(self.engine, "replica_view"):        # opt-in replicas: each one's state and requests routed to it
            live["replicas"] = self.engine.replica_view()
        engine = {"model": self.model, "max_context": self.reported_ctx(), "images": self.vision is not None,
                  "efforts": (["none"] if self.efforts["off"] else []) + self.efforts["levels"],      # what the model's template accepts
                  "effort_default": self.efforts["default"],
                  **dict(getattr(self.engine, "info", {}) or {})}
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "history": {}, "static": {}}
        parked = self.conv_log.poll(getattr(self.engine, "log_path", None), getattr(self.engine, "log_start", None))
        return {"engine": engine, "live": live,
                "requests": [identity_mod.visible_record(r, viewer) for r in hist[::-1][:None if all_requests else 12]],
                "conversation_cache": conversation_cache_view(engine, hist, totals, parked),
                "requests_kept": len(hist), "totals": totals, "hardware": tel["now"],
                "hardware_static":
                tel["static"], "history": tel["history"], "time": now, "keep_prompts_left": self.keep_prompts,
                "model_info": self.model_info}

    def prometheus(self) -> str:
        """GET /metrics/prometheus: the totals and what is running in Prometheus' text format, under llama-server's
        metric names (llamacpp:*) so its dashboards work unchanged; the drafts and requests as strata:*."""
        with self.status_lock:
            s, t = dict(self.status), dict(self.totals)
            running = len(self.live_reqs) or int(bool(s.get("busy")))
        prompt_n = t["prompt_tokens"] - t["reused"]               # what was read, as llama.cpp counts it
        prompt_s, decode_s = t["prompt_ms"] / 1000, t["decode_ms"] / 1000
        rows = [
            ("llamacpp:prompt_tokens_total", "counter", "Prompt tokens read", prompt_n),
            ("llamacpp:prompt_tokens_cached_total", "counter", "Prompt tokens the conversation cache already held",
             t["reused"]),
            ("llamacpp:prompt_seconds_total", "counter", "Prompt read time", prompt_s),
            ("llamacpp:tokens_predicted_total", "counter", "Tokens generated", t["output_tokens"]),
            ("llamacpp:tokens_predicted_seconds_total", "counter", "Generation time", decode_s),
            ("llamacpp:prompt_tokens_seconds", "gauge", "Average prompt read throughput in tokens/s",
             prompt_n / prompt_s if prompt_s else 0),
            ("llamacpp:predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s",
             t["output_tokens"] / decode_s if decode_s else 0),
            ("llamacpp:requests_processing", "gauge", "Requests running", running),
            ("llamacpp:requests_deferred", "gauge", "Requests waiting their turn", s.get("queued") or 0),
            ("strata:requests_total", "counter", "Requests finished", t["requests"]),
            ("strata:drafts_offered_total", "counter", "Speculative draft tokens offered", t["drafts_offered"]),
            ("strata:drafts_accepted_total", "counter", "Speculative draft tokens accepted", t["drafts_accepted"]),
            ("strata:model_loaded", "gauge", "1 while the model is loaded", int(self.loaded())),
        ]
        return "".join(f"# HELP {name} {help_}\n# TYPE {name} {kind}\n{name} {value}\n"
                       for name, kind, help_, value in rows)

    def v1_status(self) -> dict:
        """GET /v1/status: what this server is and does, for a client that would rather ask than guess (a front-end
        that polls its OpenAI-compatible server's status, collabosm's for one): the model and its window, images,
        the APIs, what is running, and the last request's timings in llama.cpp's names.  /metrics has the rest."""
        with self.status_lock:
            s, totals = dict(self.status), dict(self.totals)
            last_t, last_at = (dict(self.last_timings) if self.last_timings else None), self.last_request_at
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "static": {}}
        hw, static = tel.get("now") or {}, tel.get("static") or {}

        def scaled(v, unit, digits=0):
            return round(v / unit, digits) if isinstance(v, (int, float)) else None

        busy, ctx = bool(s.get("busy")), self.reported_ctx()
        images = self.vision is not None
        return {
            "service": "strata", "model": self.model,
            "loaded": self.loaded(), "auto_load": hasattr(self.engine, "restart"),
            "structured_output": {"formats": ["json_object", "json_schema"], "method": "prompt_and_validate",
                                  "constrained_decoding": False, "stream_buffered": True},
            "engine": (getattr(self.engine, "info", {}) or {}).get("version"),
            "started": int(self.started_at), "uptime_s": int(time.time() - self.started_at),
            "cache_max_tokens": ctx,
            "context": {"native": ctx, "max_positions": ctx},
            # one request at a time (more wait their turn), or "parallel": N batch slots (#465)
            "concurrency": {"serving": max(1, int(getattr(self.engine, "batch", 0) or 0)),
                            "requested": max(1, int(getattr(self.engine, "batch", 0) or 0))},
            "dialects": ["/v1/chat/completions", "/v1/messages", "/v1/responses"],
            "vision": {"enabled": images, "available": images, "error": None},
            "activity": {"requests": totals["requests"] + int(busy), "in_flight": int(busy) + int(s.get("queued") or 0),
                         "last_request_at": int(last_at) if last_at else None},
            "last_timings": last_t,
            "vram": dict((getattr(self.engine, "info", {}) or {}).get("vram") or {},
                         elastic=bool((getattr(self.engine, "info", {}) or {}).get("vram_elastic"))),
            "machine": {
                "at": int(time.time()),
                "gpu": {"name": static.get("gpu_name"), "used_mib": scaled(hw.get("gpu_mem_used"), 2 ** 20),
                        "total_mib": scaled(hw.get("gpu_mem_total"), 2 ** 20), "util_pct": hw.get("gpu_util"),
                        "temp_c": hw.get("gpu_temp"), "power_w": hw.get("gpu_power")} if static.get("gpu_name") else None,
                "ram": {"used_gib": scaled(hw.get("ram_used"), 2 ** 30, 1),
                        "total_gib": scaled(hw.get("ram_total"), 2 ** 30, 1)} if hw.get("ram_total") else None}}

    def prepare_carried(self, req, messages, tools, kwargs, max_new, priority, force=None):
        """#200: prepare() with the request's (else the config's) reasoning_carry -> (messages rendered, req, ids,
        thinking, max_new).  "history" renders the kept cut reply's messages, its thinking and the new message;
        "resume" generates on from the cut prompt + what it wrote, inside the open thinking (the new message is not
        shown; the request's own render is skipped); either only when the request is that conversation plus one new
        user message (serve/reasoning_carry.py).  Every carrying request's reply settles its conversation in run()
        ("_carry").  Not for a request without thinking, or one running the web app's tools / MCP."""
        mode = reasoning_carry.mode_of(req, self.reasoning_carry)
        if mode == "off" or kwargs.get("enable_thinking", True) is False or req.get("strata_agent") \
                or req.get("strata_mcp"):
            return (messages, req, *self.prepare(messages, tools, kwargs, max_new, priority, force=force, req=req))
        key = reasoning_carry.key_of(messages)
        rendered, resume = messages, None
        if mode == "history":
            carried = self.carry.history(messages)
            if carried is not None:
                rendered, n = carried
                # the cut reply is an assistant turn with no text: #843's filter (ChatTemplate.render) keeps it
                rendered = rendered[:-2] + [{**rendered[-2], "_carried": True}, rendered[-1]]
                print(f"[strata] reasoning carry (history): the cut reply's thinking ({n} tokens) goes back before "
                      "the new message", flush=True)
        elif not images_of(messages):
            resume = self.carry.resume(messages)
        if resume is not None:
            self.embeddings.path = None
            ids, thinking, max_new = resume, True, self._fit(resume, max_new)
            print(f"[strata] reasoning carry (resume): generating on from the cut reply ({len(ids)} tokens); the new "
                  "message is not shown to the model", flush=True)
        else:
            ids, thinking, max_new = self.prepare(rendered, tools, kwargs, max_new, priority, force=force, req=req)
        return rendered, {**req, "_carry": {"key": key, "rendered": rendered, "ids": ids}}, ids, thinking, max_new

    def _fit(self, ids, max_new):
        """The max_new a prompt of `ids` gets: "unlimited" (<= 0) is the rest of the context; never truncated unless
        the config's fit_max_tokens (#545); EngineStarting while the engine has no context size (#344)."""
        ctx = self.engine.max_context
        if ctx <= 0:
            if getattr(self.engine, "starting", False):   # #344: (re)starting, not a prompt that is too long
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
            # a failed restart leaves max_context 0: check against the last known context, so the request reaches
            # run() - which starts the engine again - instead of failing with "exceeds the context (0)" forever
            ctx = getattr(self.engine, "known_ctx", 0)
            if ctx <= 0:
                raise EngineStarting("the engine is starting (a minute or two); try again shortly")
        room = ctx - CTX_SLACK - len(ids)
        if max_new <= 0 or (self.fit_max_tokens and room < 1):
            if room < 1:
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) leaves no room to answer in the context "
                                 f"({ctx}); requests are never truncated")
            return room
        if max_new > room:
            if not self.fit_max_tokens:
                raise ValueError("request exceeds the context window: "
                                 f"prompt ({len(ids)} tokens) + max tokens ({max_new}) exceeds the context "
                                 f"({ctx}); requests are never truncated. Send a smaller "
                                 f"max_tokens (at most {max(0, room)} here), or add \"fit_max_tokens\": true to the "
                                 "model's strata-<model>.json to shorten it to the room left (#545)")
            return max(1, room)             # --fit-max-tokens: a shorter completion beats a 400
        return max_new

    def render_prompt(self, messages, tools, kwargs) -> str:
        """The template rendered.  #458: with `effort_end`, a request with a non-default effort (low, medium or no
        thinking) is rendered as a default one up to the answer - the same prompt start, so the conversation cache
        keeps it - and its effort follows in a short system turn right before the answer (thinking off: the template's
        empty thinking block).  The engine (--tail-role-token) checkpoints in front of that turn."""
        effort = kwargs.get("reasoning_effort")
        off = kwargs.get("enable_thinking") is False
        if not self.effort_end or (not off and effort in (None, "", "xhigh", "high")):
            return self.template.render(messages, tools=tools, **kwargs)
        base = {k: v for k, v in kwargs.items() if k not in ("reasoning_effort", "enable_thinking")}
        history = self.template.render(messages, tools=tools, add_generation_prompt=False, **base)
        full = self.template.render(messages, tools=tools, add_generation_prompt=True, **kwargs)
        mine = self.template.render(messages, tools=tools, add_generation_prompt=False, **kwargs)
        if not full.startswith(mine):                  # a template this cannot take apart: rendered as asked
            return full
        tail = "" if off else EFFORT_TURN.format(EFFORT_TEXT.get(effort, EFFORT_TEXT["medium"]))
        return history + tail + full[len(mine):]

    def encode_prompt(self, messages, tools, kwargs) -> list[int]:
        """The request's prompt: the template rendered and tokenized.  #537: a <think> / </think> written inside a
        message's text is encoded as the text it is, not as the model's reasoning markers (the template's own are).
        A control token's text (<|im_start|>, <|im_end|>, <|endoftext|>, ...) inside a message is text as well: only
        the control tokens the template writes are control tokens."""
        marked, marked_tools, changed = mark_think_literals(messages, tools, self.literals)
        prompt = self.render_prompt(marked, marked_tools, kwargs)
        if not changed:
            return self.tok.encode(prompt, parse_special=True)
        prompt, plain = unmark_think_literals(prompt, self.literals)
        return self.tok.encode(prompt, parse_special=True, plain=plain)

    def _note_unreadable_tool_images(self, messages):
        """A picture a tool returned (Claude Code's Read of an image file) that this server cannot read - it has no
        image encoder, or the encoder refuses that picture - becomes a short note in its place instead of a 400.
        Clients resend the whole history every turn, so one unreadable tool picture would fail every later request of
        the conversation; a picture the user sends keeps the 400.  A picture that encodes stays in the cache, so the
        prompt's own encode below finds it."""
        fetched = {}                     # a URL's bytes, so prepare() does not download the same picture again
        for m in messages:
            content = m.get("content")
            if m.get("role") != "tool" or not isinstance(content, list):
                continue
            for n, item in enumerate(content):
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                why = "this server has no image encoder" if self.vision is None else None
                if why is None:
                    src = item["source"]
                    try:
                        if src.startswith(("http://", "https://")):
                            src = fetched[item["source"]] = Vision.download(src)   # outside the FIFO, as in prepare()
                        with self.fifo:          # the encoder takes its turn with the requests (see below)
                            self.vision.encode(src)
                    except (ValueError, OSError) as e:
                        why = str(e)
                if why is not None:
                    print(f"[strata] a picture a tool returned was left out of the prompt: {why}", flush=True)
                    content[n] = {"type": "text", "text": f"[image omitted: {why}]"}
        return fetched

    def prepare(self, messages, tools, kwargs, max_new=None, priority: int = 1, force=None, req=None,
                count_only: bool = False):
        """-> (ids, thinking, max_new). An unset or non-positive max_new (some clients send -1) means "unlimited":
        the rest of the context.  `force` (forced_call): without thinking the reply starts with it, so it ends the
        prompt; with thinking, Service.run writes it once the thinking is over.  `priority`: the request's place in
        the queue for the image encoder (xeno #49 S6).  `count_only` (POST /v1/messages/count_tokens): no room check
        and no embeddings file - the ids alone (images counted as read, xeno #46)."""
        fetched = self._note_unreadable_tool_images(messages)
        t_tl = timeline.now_us()
        ids = self.encode_prompt(messages, tools, kwargs)
        timeline.complete("template+tokenize", t_tl, timeline.now_us(), len(ids))
        if force and kwargs.get("enable_thinking", True) is False:
            ids = ids + self.tok.encode(force, parse_special=True)
        self.embeddings.path = None
        images = images_of(messages)
        if images:
            if self.vision is None:
                raise ValueError("this server was started without the vision encoder (run setup again and choose "
                                 "'vision'), so it cannot read images")
            pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
            start = self.tok.encode(VISION_START, parse_special=True)[0]
            # An image URL is downloaded first, outside the FIFO: under it, a slow server held every other request
            # for as long as urlopen waited.
            images = [fetched[src] if src in fetched else Vision.download(src) if src.startswith(("http://", "https://"))
                      else src for src in images]
            # Encode only while the engine is idle: the engine and the image encoder (a separate process) must not
            # run on the GPU at the same time - an encode during a running request left that request stuck at
            # "reading the prompt" with CPU and GPU busy, for good (reproduced).  So encoding takes its turn in the
            # same FIFO as the requests.
            with self.fifo.slot(priority):              # xeno #49 review: at the request's own priority (S6)
                encoded = (self.vision.encode_all(images) if hasattr(self.vision, "encode_all")
                           else [self.vision.encode(src) for src in images])
            # one <|image_pad|> per image -> one per image token.  Only the markers the template writes for an image
            # (right after <|vision_start|>) are images: the same text inside a message (an agent reading these docs,
            # #150) is kept as plain text, or it took an image's place and the counts no longer matched.
            literal = self.tok.encode(IMAGE_PAD, parse_special=False)
            out, k = [], 0
            for j, t in enumerate(ids):
                if t == pad and j > 0 and ids[j - 1] == start:
                    # more pairs than images: text parts that cut both markers apart (the literal marks keep whole ones text)
                    if k == len(encoded):
                        raise ValueError("the prompt and its images do not match")
                    out += [pad] * encoded[k][1]
                    k += 1
                elif t == pad:
                    out += literal
                else:
                    out.append(t)
            if k != len(encoded):
                raise ValueError("the prompt and its images do not match")
            ids = out
        if count_only:                       # count only (POST /v1/messages/count_tokens): no room check
            return ids, kwargs.get("enable_thinking", True) is not False, None
        max_new = self._fit(ids, 0 if max_new is None else max_new)
        if images:
            # The request's images in one file for GENI (~10 MB a picture), written once nothing above refuses the
            # request: one refused after it (the engine starting, no room) left it in the vision directory for good,
            # one more for every retry of a 503.  run() deletes it; drop_embeddings() if run() never starts.
            combined = combined_embeddings_path(self.vision.dir, sum(Path(p).stat().st_size for p, _ in encoded))
            self.embeddings.path = combined             # first, so a half-written one is found as well
            try:
                write_temporary(combined, [p for p, _ in encoded])
            except Exception:                           # xeno: a failed staging leaves no partial file behind
                combined.unlink(missing_ok=True)
                self.embeddings.path = None
                raise
        if req is not None and req.get("strata_prefix") is not None:
            req["strata_prefix"] = self.resolve_prefix(req["strata_prefix"], messages, tools, kwargs, ids, bool(images))
        return ids, kwargs.get("enable_thinking", True) is not False, max_new

    def _engine(self, ids, max_new, sampling, cancel, emb):
        return self.engine.generate(ids, max_new, sampling, cancel, embeddings=emb) if emb else \
            self.engine.generate(ids, max_new, sampling, cancel)

    def _generate(self, ids, max_new, sampling, cancel, emb, state):
        """The engine's tokens, with the thinking cut.  A budget comes from two sources, and the smaller cuts: xeno #49 S3's
        `_think_budget` (Anthropic's thinking.budget_tokens and side requests, serve/think_budget.py; closed with its
        CLOSE) and #123's reasoning_budget_tokens (state["budget"], opt-in; closed with REASONING_WRAP_UP).  Once that
        many reasoning tokens are out (state["thought"]) while state["thinking"] holds, at a clean point
        (state["clean"](): no tag held back, no character split across tokens - #123), stop the engine, yield the
        close's tokens as if the model wrote them and continue from (prompt + written + close), which reuses the
        engine's cached prefix.  No room left to answer after the close: the request ends there ("length", #123).
        #199: the loop guard's cycle rule cuts the same way, at any length (state["close"], set by run()), so a
        thinking request is watched even without a budget.  #728 "recover" (state["recover"], set by run()) cuts the same
        way and goes on from focused_recovery_prompt (the low-effort instruction) with its sampling.  A forced tool call (upstream's tool_choice, state["force"])
        is opened by the same cut: after the close when the thinking is cut, or - state["opens"], set by run() - where
        the thinking ended, after the blank line the template puts before a call.  #1053 (opt-in, state["retry"], set
        by run()): a reply that ends on its stop token inside the thinking with no answer is continued once, the
        thinking closed (REASONING_CLOSE).  One cut per request: state["cut"] says it was made."""
        opening = (sampling or {}).get("_opening")      # xeno (serve/forced_opening.py): written for the model,
        if opening:                                     # which continues from it
            head = self.tok.encode(opening)
            yield from head
            ids, max_new = list(ids) + head, (max(1, max_new - len(head)) if max_new else max_new)
        budget, close_text = state.get("budget"), REASONING_WRAP_UP          # #123
        own = (sampling or {}).get("_think_budget")                            # xeno #49 S3
        if own and (not budget or own <= budget):
            budget, close_text = own, think_budget.CLOSE
        gen = self._engine(ids, max_new, sampling, cancel, emb)
        if not budget and not state["thinking"]:        # nothing to cut: no budget, and not thinking (#199)
            yield from gen
            return
        written = []
        try:
            for t in gen:
                yield t
                if t is not None:
                    written.append(t)
                    due = state["close"] or (budget and state["thought"] >= budget)   # #199: or a cycle
                    if state["thinking"] and due and state["clean"]():
                        break
                    if state.get("opens") or state.get("retry") or state.get("recover"):   # forced call / #1053 / #728
                        if t in self.stop_ids:          # at the end of the turn (#537): the stop token is not written
                            written.pop()
                        break
            else:
                return
        finally:
            gen.close()
        state["cut"] = True
        if cancel.is_set():
            return
        # the engine committed a few tokens past the cut before it read STOP: they are part of the answer (and of
        # its live session, whose whole length must be a prefix of the next prompt for the cache to be reused)
        for t in list(getattr(self.engine, "drained", None) or []):
            written.append(t)
            yield t
        recover = state.get("recover")
        if recover is not None:                         # #728 "recover": the same output, the low-effort instruction
            prompt = focused_recovery_prompt(self.tok, list(ids), written)
            if prompt is None or (max_new and max_new - len(written) < 1):
                return
            print(f"[strata] repeated reasoning detected at {recover['n']} tokens (coverage={recover['coverage']:.3f}); "
                  "resuming the same output with the low-effort instruction (reasoning_loop_recovery)", flush=True)
            state["recovered"] = True
            leg1 = getattr(self.engine, "last", None)
            first = dict(leg1 or {})
            rest = max(1, max_new - len(written)) if max_new else max_new
            try:
                yield from self._engine(prompt, rest, recover["sampling"], cancel, emb)
            finally:
                self._merge_legs(first, 0, leg1)
            return
        # run() has parsed those tokens: they may hold the model's own </think>, and then there is nothing left to
        # close (the close would be answer text, with a second </think>)
        force, state["force"] = state.get("force"), None
        retry = state.get("retry")
        cutting = state["thinking"] and not state.get("opens") and not retry   # the budget's or the cycle's close
        if retry:
            text = REASONING_CLOSE + (force or "")
        elif cutting:
            text = close_text + (force or "")
        elif force:
            tail = state.get("tail", "")
            text = "\n" * (2 - (len(tail) - len(tail.rstrip("\n")))) + force
        else:
            text = ""
        close = self.tok.encode(text, parse_special=True) if text else []
        if max_new and max_new - len(written) - len(close) < 1:
            return                                      # #123: no room left to answer: "length", as without a budget
        leg1 = getattr(self.engine, "last", None)
        first = dict(leg1 or {})
        if retry:
            print("[strata] the reply ended inside its thinking with no answer: closing the thinking once and "
                  "continuing (reasoning_close_retry)", flush=True)
        if cutting:
            why = (f"thinking cycle ({state['close']})" if state["close"] else
                   f"thinking budget reached ({state['thought']} of {budget} tokens)")
            print(f"[strata] {why}: closing the thinking block", flush=True)
            if state["close"]:
                with self.status_lock:
                    self.status["thinking_cycles_closed"] += 1   # #199: counted when the close is written
        yield from close
        rest = max(1, max_new - len(written) - len(close)) if max_new else max_new
        try:
            yield from self._engine(list(ids) + written + close, rest, sampling, cancel, emb)
        finally:
            self._merge_legs(first, len(close), leg1)

    def _merge_legs(self, first: dict, n_close: int, leg1=None):
        """One request's DONE figures from its two engine calls (xeno #49 review): the prompt read is the first
        call's; re-reading the written tokens + CLOSE is part of making the output, so it counts as decode time.
        A second call that ended without its own DONE (`engine.last` is still the first call's object, `leg1`) adds
        nothing: the first DONE is not counted twice (upstream #615's rule)."""
        if leg1 is not None and getattr(self.engine, "last", None) is leg1:
            return
        second = dict(getattr(self.engine, "last", None) or {})
        if not first or not second:
            return
        merged = dict(first)
        merged["decode_ms"] = first.get("decode_ms", 0.0) + second.get("prompt_ms", 0.0) + second.get("decode_ms", 0.0)
        merged["generated"] = first.get("generated", 0) + n_close + second.get("generated", 0)
        # the expert-tier counters (engine 0.1.31+, the DONE line's RAM / file blobs and file MB) are per call too
        for k in ("drafts_accepted", "drafts_offered", "hits", "lookups", "offloaded", "ram_blobs", "file_blobs",
                  "file_mb"):
            if k in first or k in second:
                merged[k] = first.get(k, 0) + second.get(k, 0)
        self.engine.last = merged

    PREFIX_END = "\u0001strata-prefix-end\u0001"          # marks where a message's shared part ends in the rendered prompt

    def resolve_prefix(self, spec, messages, tools, kwargs, ids, has_images=False) -> dict:
        """R1: `"strata_prefix"` marks the start of the prompt as a shared read-only prefix: a long document that many
        short questions follow.  Three ways to say where it ends: `{"messages": K}` (the first K messages), `{"message":
        I, "chars": N}` (the first N characters of message I's text: a document and its question in one message) or
        `{"tokens": N}`.  The engine reads the prompt in two parts there, keeps the checkpoint at the boundary pinned
        (never evicted) and does not park the branch it leaves, so N questions cost one prefix.  Returns
        {"tokens": N} (what the engine's `pin=N` takes: always a prefix of the prompt's ids, the tokenizer's merge across
        the boundary included), or {} when the prefix cannot be marked (said in the log: the request is never refused for
        it).  A malformed field is a 400."""
        keys = ("messages", "message", "chars", "tokens")
        if (not isinstance(spec, dict) or not spec or any(k not in keys for k in spec) or
                set(spec) not in ({"messages"}, {"tokens"}, {"message", "chars"})):
            raise ValueError('strata_prefix: {"messages": K} (the first K messages), {"message": I, "chars": N} (the first '
                             'N characters of message I) or {"tokens": N}')
        for k, v in spec.items():
            if not isinstance(v, int) or isinstance(v, bool) or v < (0 if k == "message" else 1):
                raise ValueError(f"strata_prefix.{k}: a whole number ({'0' if k == 'message' else '1'} or more)")
        if has_images:
            print("[strata] strata_prefix ignored: the request has images", flush=True)
            return {}
        what = ",".join(f"{k}={v}" for k, v in spec.items())

        def lcp(a, b):
            n = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                n += 1
            return n

        def head_ids(msgs):
            """The ids the rendered `msgs` start with (a message's trailing marker cut at PREFIX_END when it is there)."""
            marked, marked_tools, changed = mark_think_literals(msgs, tools, self.literals)
            effort = kwargs.get("reasoning_effort")
            off = kwargs.get("enable_thinking") is False        # render_prompt's rule for #458's effort_end
            kw = ({k: v for k, v in kwargs.items() if k not in ("reasoning_effort", "enable_thinking")}
                  if self.effort_end and (off or effort not in (None, "", "xhigh", "high")) else dict(kwargs))
            kw["add_generation_prompt"] = False
            text = self.template.render(marked, tools=marked_tools, **kw)
            cut = text.find(self.PREFIX_END)
            if cut >= 0:
                text = text[:cut]
            if changed:
                text, plain = unmark_think_literals(text, self.literals)
                return self.tok.encode(text, parse_special=True, plain=plain)
            return self.tok.encode(text, parse_special=True)

        if "tokens" in spec:
            n = spec["tokens"]
        elif "messages" in spec:
            k = spec["messages"]
            if k >= len(messages):
                print(f"[strata] strata_prefix.messages={k} ignored: the request has {len(messages)} message(s) and the "
                      "prefix must leave at least the last one", flush=True)
                return {}
            head = head_ids(messages[:k])
            n = lcp(head, ids)
        else:
            i, chars = spec["message"], spec["chars"]
            text = messages[i].get("content") if 0 <= i < len(messages) else None
            if not isinstance(text, str) or chars > len(text):
                print(f"[strata] strata_prefix ignored: message {i} has no text of {chars} characters", flush=True)
                return {}
            cut = messages[:i] + [{**messages[i], "content": text[:chars] + self.PREFIX_END}]
            head = head_ids(cut)
            n = lcp(head, ids)
        if n < 1 or n >= len(ids) - 1:
            print(f"[strata] strata_prefix {what} ignored: it leaves no usable prefix of the {len(ids)}-token prompt "
                  f"(common start {n})", flush=True)
            return {}
        return {"tokens": n}

    def drop_embeddings(self) -> None:
        """Delete the combined image file prepare() wrote when no run() took it over (a run deletes its own as it
        ends): a request can end before its answer starts - its client gone, anything raised after prepare() - and
        then nothing else would.  The handler calls this when a request is done."""
        path, self.embeddings.path = getattr(self.embeddings, "path", None), None
        if path is not None:
            Path(path).unlink(missing_ok=True)

    def _loop_stopped(self) -> str:
        """A reply ended as degenerate output (the loop guard, or #606's one-token run): counted in /status, and its
        stop_detail."""
        with self.status_lock:
            self.status["loops_stopped"] += 1
            self.status["last_stop_reason"] = "loop"
        return "loop"

    def _note(self, n, evs, st=None, rate=None):
        with self.status_lock:
            s = self.status if st is None else st
            s["generated"] = n
            if s.get("first_token") is None:
                s["first_token"] = time.time()
            (self.rate if rate is None else rate).append((time.time(), n))   # the live rate's window (RATE_WINDOW_S)
            for ev in evs:
                if ev.kind == "reasoning":
                    s["phase"] = "thinking"
                elif ev.kind == "content":
                    s["phase"] = "answering"
                elif ev.kind == "tool_start":
                    s["phase"], s["tool"] = f"writing a tool call: {ev.call.name}", ev.call.name
                elif ev.kind == "tool_call":
                    s["phase"] = "tool call complete"
                s["tail"] = ((s.get("tail") or "") + (ev.text or ""))[-600:]

    def _progress(self, last_print, every=1.0, st=None):
        """A progress line in the server window every `every` seconds while a request runs."""
        now = time.time()
        if now - last_print < every:
            return last_print
        with self.status_lock:
            s = dict(self.status if st is None else st)
        el = now - s.get("started", now)
        if s.get("first_token") is None:
            pr = getattr(self.engine, "progress", None)   # (position reached, prompt tokens): a reused prefix counts
            done = f"{pr[0]:,} of {pr[1]:,}" if pr and pr[1] else f"{s.get('prompt_tokens', 0):,}"   # as read (#29)
            print(f"[strata] reading the prompt: {done} tokens, {el:.0f} s so far", flush=True)
        else:
            rate = s["generated"] / max(1e-6, now - s["first_token"])
            # #984: while a thinking budget applies, the thinking line shows it, not the request's max_tokens
            limit = (f"budget {s['thinking_budget']} (max {s.get('max_tokens')})"
                     if s["phase"] == "thinking" and s.get("thinking_budget") else f"max {s.get('max_tokens')}")
            print(f"[strata] {s['phase']}: {s['generated']} of {limit} tokens, {rate:.1f} tok/s, "
                  f"{el:.0f} s", flush=True)
        return now

    def run(self, ids, thinking, tools, max_new, sampling, cancel, force=None) -> Iterator[tuple[str, object]]:
        """The request's events (see _run); with "prevent_sleep" in the config, the PC stays awake while it runs or waits."""
        ka = self.keep_awake
        if ka is not None:
            ka.acquire()
        try:
            yield from self._run(ids, thinking, tools, max_new, sampling, cancel, force)
        finally:
            if ka is not None:
                ka.release()

    def _run(self, ids, thinking, tools, max_new, sampling, cancel, force=None) -> Iterator[tuple[str, object]]:
        """Yields ("event", Event) as text arrives, then ("done", {"finish": .., "completion_tokens": ..}).
        `force` (forced_call): the opening of the call the reply must make - see prepare()."""
        defaults = {**self.sampling_defaults, **self.shared}   # the config's, then the Chat settings shared with apps
        if defaults:                   # the request's own fields win (explicit 0 stays greedy)
            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}
            sampling = {**defaults, **req_values}
        # #123: read after the merge, so a budget shared through POST /settings is seen like the other keys
        budget = self.reasoning_budget(sampling) if thinking else None
        if budget and max_new and max_new > 0:
            # #984: a budget at or near max_tokens never fires before the reply is cut off while still thinking, so
            # it has no answer.  Close the thinking early enough to leave room for one.
            reserve = min(max(ANSWER_RESERVE_MIN, max_new // 4), max_new // 2)
            if budget > max_new - reserve:
                print(f"[strata] thinking capped at {max_new - reserve} of max_tokens {max_new} to leave room for "
                      f"the answer (budget {budget})", flush=True)
                budget = max(1, max_new - reserve)
        parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True, recover=self.tool_call_recovery)
        guard = LoopGuard()
        detok, n, finish = Detokenizer(self.tok), 0, "length"
        carry = (sampling or {}).get("_carry")         # #200: this request settles its conversation
        reasoning_text = []                             # #200: this reply's thinking, if it is cut
        run_tok, run_len, repeated = None, 0, False     # #606: the current run of one repeated token
        thinking_n = 0                                  # tokens written while thinking (Responses' reasoning_tokens)
        stop_list = stop_strings(sampling)
        stops = StopMatcher(stop_list) if stop_list else None

        def cut(evs):
            """The events with the answer's text cut at a stop string (OpenAI stop, Anthropic stop_sequences)."""
            if stops is None:
                return evs
            out = []
            for ev in evs:
                if stops.hit is not None:
                    break                               # nothing after the stop string is sent
                if ev.kind == "content":
                    text = stops.push(ev.text or "")
                    if text:
                        out.append(Event("content", text))
                    continue
                held = stops.flush()                    # a stop string does not run across a tool call or thinking
                if held:
                    out.append(Event("content", held))
                out.append(ev)
            return out
        opening = []                                    # without thinking the prompt ends with the forced opening:
        if force and not thinking:                      # the parser reads it as if the model had written it
            opening, force = parser.feed(force), None
        answered = False                                # #1053: content or a call came out
        timings, before = None, None                    # this request's timings; the engine's `last` before it
        stop_detail = None
        # #728 (opt-in "reasoning_loop_recovery"): reasoning that repeats whole passages.  "stop" ends the reply here;
        # "recover" goes on once from the same output with the low-effort instruction (_generate's state["recover"])
        recovery_count, loop_text, repeat_coverage = 0, "", 0.0
        looped, next_loop_check = False, LOOP_CHECK_EVERY
        raw_ids = []                                    # every generated id (STRATA_DEBUG: dump raw model text)
        tok_times = []                                  # when each token reached the server
        emb, self.embeddings.path = getattr(self.embeddings, "path", None), None   # this run's to delete now
        if self.restarting:            # xeno #49 S2: the native API's answer while the model loads - retry later
            raise EngineDied("the engine is loading again after it stopped")
        # Identity token: only a DONE line replaces engine.last, so a request that died, errored or was
        # disconnected must not have the PREVIOUS request's decode figures recorded as its own.
        engine_last0 = getattr(self.engine, "last", None)
        trace = getattr(self.request_trace, "record", None)
        # #310: the identity whose request this is (its history meta's "who"): the engine's holder while it runs
        who = ((sampling or {}).get("_meta") or {}).get("who") or "owner"
        waiting = time.perf_counter()
        # #465: with "parallel" (the engine's batch slots) requests run at once: each keeps its own status and rate
        # window (self.status says busy while any runs); one at a time they are self.status / self.rate, as before
        par = bool(getattr(self.engine, "batch", 0))
        st = {} if par else self.status
        rate = collections.deque(maxlen=32) if par else self.rate
        with self.status_lock:
            place = self._queue_enter(who)
        t_queue = timeline.now_us()
        priority = 0 if (sampling or {}).get("stream") else 1   # xeno #49 S6: main turn first
        try:
            # --batch: the engine runs several requests at once (StrataEngine.generate_batched orders them)
            turn = contextlib.ExitStack()
            if not getattr(self.engine, "batch", 0):
                # #1619: a request waiting for the single turn sends a keep-alive every QUEUE_BEAT_S, so a client with
                # a stream idle timeout does not abort a healthy queued request (and its retry read the prompt again);
                # a client that left while queued gives its place up instead of taking the turn only to cancel there
                got = False
                try:
                    while not got:
                        got = self.fifo.acquire(priority, timeout=QUEUE_BEAT_S)
                        if got:
                            turn.callback(self.fifo.release)
                        elif cancel.is_set():
                            break
                        else:
                            yield "ping", None
                except BaseException:                    # the consumer closed us at a ping: no longer queued
                    with self.status_lock:
                        self._queue_leave(place)
                    raise
                if not got:                              # cancelled while queued
                    with self.status_lock:
                        self._queue_leave(place)
                    yield "done", {"finish": "cancel", "completion_tokens": 0, "reused": 0, "timings": None,
                                   "reasoning_tokens": 0, "reasoning_recoveries": 0}
                    return
            with turn:
                try:
                    timeline.complete("queue wait", t_queue, timeline.now_us())
                    with self.status_lock:
                        if trace is not None:
                            trace["queue_s"] += round(time.perf_counter() - waiting, 3)
                            trace["state"] = "generating"
                        if not par:
                            self._queue_leave(place)
                    # issue #27: it died in an earlier request (or was unloaded) - start it again instead of failing
                    # (parallel requests do not hold the fifo: one of them starts it, the others wait for that).
                    # #1603: a parallel request stays counted as queued until it is registered as running below -
                    # waiting here for the fifo (an image encode, a /v1/vram resize) left it neither - and one on a
                    # loaded engine has nothing to start, so it does not wait for the fifo at all
                    try:
                        if not par:
                            self.ensure_loaded()
                        elif not (self.loaded() and not self._vision_down()):
                            with self.fifo:
                                self.ensure_loaded()
                    except BaseException:
                        if par:
                            with self.status_lock:
                                self._queue_leave(place)
                        raise
                    with self.status_lock:
                        if par:
                            self._queue_leave(place)
                        st.update(busy=True, phase="reading the prompt", prompt_tokens=len(ids),
                                  generated=0, started=time.time(), first_token=None, tool=None, tail="",
                                  max_tokens=max_new, reasoning_recoveries=0, thinking_budget=budget, who=who)
                        if par:
                            self.live_reqs[id(st)] = (st, rate)
                            self.status.update(st)
                        self.last_request_at = time.time()
                        rate.clear()                    # the previous request's samples must not leak into this one
                    before = getattr(self.engine, "last", None)
                    last_print = time.time()
                    for ev in opening:
                        yield "event", ev
                    t_engine = timeline.now_us()
                    # still inside the thinking block, and the reasoning tokens so far: the thinking budget's cut
                    # (_generate); `clean` is asked only at the budget (#123: no tag held back, no split character).
                    # "force" / "opens" / "tail": a forced tool call opened after the thinking (upstream's tool_choice)
                    state = {"thinking": thinking, "budget": budget, "thought": 0, "close": None,
                             "clean": lambda: not parser.buf and not detok.pending(),
                             "force": force, "opens": False, "tail": ""}
                    if hasattr(self.engine, "last"):
                        self.engine.last = {}              # this request's DONE only, never the previous one's
                    gen = self._generate(ids, max_new, sampling, cancel, emb, state)
                    leaving = False                         # #481: the client went away (GeneratorExit below)
                    try:
                        # A value the frontend holds back (a tool call's array or object argument is sent whole,
                        # once complete) can keep the stream silent for minutes while tokens are generated; a
                        # client's idle timeout then ends the request.  Send the same keep-alive the engine's
                        # heartbeat sends when nothing has gone out for KEEPALIVE_S.
                        last_out = time.perf_counter()
                        for t in gen:
                            if t is None:                   # heartbeat while the engine is quiet
                                last_print = self._progress(last_print, st=st)
                                last_out = time.perf_counter()
                                yield "ping", None
                                continue
                            n += 1
                            tok_times.append(time.monotonic())      # xeno UI S3: the decode speed over a sliding window
                            if n == 1:
                                timeline.instant("first token", len(ids))
                            if trace is not None and trace["first_token_s"] is None:
                                with self.status_lock:
                                    trace["first_token_s"] = round(time.perf_counter() - trace["_clock"], 3)
                            if t in self.stop_ids:
                                if not state.get("cut") and state["force"] and parser.state == "reasoning"                                         and parser.buf == THINK_END:
                                    state["opens"] = True   # #537: the held </think> was the end: the call opens there
                                    continue                # (_generate drops the stop token and writes the opening)
                                if (self.reasoning_close_retry and thinking and not state.get("cut") and not answered
                                        and parser.state == "reasoning" and parser.buf != THINK_END
                                        and (stops is None or stops.hit is None) and not cancel.is_set()
                                        and max_new - n - len(self.tok.encode(REASONING_CLOSE, parse_special=True)) >= 1):
                                    # #1053: the model wrote its reasoning and stopped before </think>: the client
                                    # would get an empty answer.  Close the thinking once and let it answer (_generate)
                                    state["retry"] = True
                                    continue
                                finish = "stop"
                                raw_ids.append(t)
                                break
                            raw_ids.append(t)
                            thinking_n += parser.state in ("reasoning", "rcall")
                            run_len = run_len + 1 if t == run_tok else 1
                            run_tok = t
                            if self.repeat_stop_tokens and run_len >= self.repeat_stop_tokens:
                                repeated = True             # #606: a degenerate output, not an answer: end it here
                                break
                            piece = detok.push(t)
                            state["tail"] = (state["tail"] + piece)[-2:]
                            evs = cut(parser.feed(piece))
                            state["thinking"] = parser.state == "reasoning"   # not from events: '</think>' emits none
                            if state["thinking"]:
                                state["thought"] += 1
                            self._note(n, evs, st, rate)
                            last_print = self._progress(last_print, st=st)
                            for ev in evs:
                                if carry is not None and ev.kind == "reasoning":
                                    reasoning_text.append(ev.text)   # #200: what a cut reply carries
                                if self.reasoning_loop_recovery and ev.kind == "reasoning":
                                    loop_text += ev.text or ""
                                if ev.kind in ("content", "tool_start", "tool_call"):
                                    answered = True         # #1053: content or a call came out
                                if ev.kind in ("reasoning", "content") and guard.feed(
                                        ev.text, in_think=ev.kind == "reasoning"):
                                    cancel.set()
                                    finish = "length"
                                    stop_detail = self._loop_stopped()
                                    print(f"[strata] loop guard stopped generation: {guard.reason}", flush=True)
                                    break
                                yield "event", ev
                            if guard.reason:
                                break
                            if evs:
                                last_out = time.perf_counter()
                            elif time.perf_counter() - last_out >= KEEPALIVE_S:
                                last_out = time.perf_counter()
                                yield "ping", None
                            if stops is not None and stops.hit is not None:
                                finish = "stop"             # gen.close() below STOPs the engine, as for a stop token
                                break
                            if state.get("recovered") and not recovery_count:   # #728: _generate went on with it
                                recovery_count = 1
                                with self.status_lock:
                                    st["reasoning_recoveries"] = recovery_count
                                if trace is not None:
                                    trace["reasoning_recoveries"] = recovery_count
                                    trace["reasoning_repeat_coverage"] = round(repeat_coverage, 3)
                            if (self.reasoning_loop_recovery and not state.get("cut") and not state.get("recover")
                                    and parser.state == "reasoning" and n >= next_loop_check and not parser.buf
                                    and not parser.pending and not detok.pending()):
                                next_loop_check = n + LOOP_CHECK_EVERY
                                repeat_coverage = reasoning_repeat_coverage(loop_text)
                                if repeat_coverage >= LOOP_COVERAGE:
                                    if self.reasoning_loop_recovery == "stop":
                                        looped = True       # #728: end the reply here, as for a repeated token
                                        break
                                    # "recover": not for a request with pictures, nor without the template's own xhigh
                                    # sentence to swap.  Only the two settings that keep the same words coming are
                                    # raised (temperature to at least 1.0, presence penalty to at least 1.5): a
                                    # client's top_p and top_k are never touched, and its own dict is copied, not
                                    # changed.  _generate cuts at the next token and goes on from the new prompt.
                                    if not emb and focused_recovery_prompt(self.tok, ids, raw_ids) is not None:
                                        want = (max(1.0, float((sampling or {}).get("temperature") or 0)),
                                                max(1.5, float((sampling or {}).get("presence_penalty") or 0)))
                                        same = want == tuple(float((sampling or {}).get(k) or 0)
                                                             for k in ("temperature", "presence_penalty"))
                                        state["recover"] = {"sampling": sampling if same else {
                                            **(sampling or {}), "temperature": want[0], "presence_penalty": want[1]},
                                            "n": n, "coverage": repeat_coverage}
                            # #199: the thinking is cycling - _generate closes it at the next clean point (once)
                            if guard.close_reason and state["thinking"] and not state["close"]:
                                state["close"] = guard.close_reason
                            if state["force"]:              # a forced call: _generate opens it once the thinking ends
                                if any(ev.kind in ("tool_start", "tool_call") for ev in evs):
                                    state["force"] = None   # the model is writing a call of its own
                                elif parser.state == "content" and not parser.buf and not detok.pending():
                                    state["opens"] = True
                        # the loop guard cancels the engine itself; only a client cancel is one
                        if cancel.is_set() and not guard.reason:
                            finish = "stop" if getattr(cancel, "server_stop", False) else "cancel"
                        elif looped:
                            print(f"[strata] the thinking repeated the same passages (coverage={repeat_coverage:.3f}) "
                                  f"at {n} tokens: ended as \"length\" (reasoning_loop_recovery: \"stop\" in "
                                  "strata-<model>.json; remove it to turn this off)", flush=True)
                        elif repeated:
                            stop_detail = self._loop_stopped()   # #606: reported as the loop guard's stop
                            print(f"[strata] the reply repeated one token ({self.tok.decode([run_tok])!r}) "
                                  f"{run_len} times in a row: ended as \"length\" (repeat_stop_tokens in "
                                  "strata-<model>.json; 0 turns this off). If a new request with a short prompt does "
                                  "the same, restart the server and report it (#606)", flush=True)
                        # #200: a reply of its own prompt settles its conversation: cut at max_tokens while still
                        # thinking (not by a loop stop or a close) - kept; any other end - forgotten
                        if carry is not None and carry["ids"] is ids:
                            cut_reply = finish == "length" and state["thinking"] and stop_detail is None \
                                and not state["close"] and not looped and not repeated
                            self.carry.settle(carry["key"], (carry["rendered"], ids, raw_ids,
                                                             "".join(reasoning_text)) if cut_reply else None)
                    except EngineDied as e:
                        finish = "error"
                        self._say_died(e)
                        raise
                    except ValueError as e:                 # the engine's ERR line (it may have ended after it)
                        finish = "error"
                        print(f"[strata] the engine reported an error: {e}", flush=True)
                        raise
                    except GeneratorExit:                   # the client went away: an engine that never acknowledges
                        leaving = True                      # the STOP below is ended, but no error replaces this
                        raise
                    finally:
                        try:
                            gen.close()                 # STOP+drain to THIS request's DONE while still holding the
                            #                             fifo, so a stop-token break can't leave the shared engine
                            #                             queue mid-drain for the next request to read as its own DONE
                        except EngineSilent as e:       # #481: the STOP was never acknowledged: the engine is ended
                            finish = "error"
                            self._say_died(e)
                            if not leaving and not cancel.is_set():
                                raise
                        finally:
                            timeline.complete("engine request", t_engine, timeline.now_us(), len(ids), n)
                except GeneratorExit:                   # the client disconnected mid-stream
                    finish = "disconnect"
                    raise
                finally:
                    # #266: settle this request's status, history and totals while still holding the fifo: once
                    # it is released the next request sets its own status, which this must not record or clear
                    to_disk = None
                    with self.status_lock:
                        if st.get("busy"):
                            # only this request's DONE counts: same object means no DONE arrived (death, error,
                            # disconnect); the thinking budget's two engine calls are one DONE (_merge_legs)
                            last = dict(getattr(self.engine, "last", {}) or {}) \
                                if getattr(self.engine, "last", None) is not engine_last0 else {}
                            started = st.get("started", time.time())
                            cvec = (getattr(self.engine, "info", {}) or {}).get("cvec", 0)
                            loaded = str(cvec) not in ("0", "", "None")
                            hit_rate = round(last["hits"] / last["lookups"], 3) if last.get("lookups") else None
                            # #588: the share the GPU read over PCIe (--pcie-frac) or another GPU computed - not in
                            # the hit rate's lookups, which is the VRAM share of the rest (None: an older engine)
                            routed = (last.get("lookups") or 0) + (last.get("offloaded") or 0)
                            pcie_share = round(last["offloaded"] / routed, 3) \
                                if last.get("offloaded") is not None and routed else None
                            seen = prompt_tokens_seen(len(ids), last)   # #471: < len(ids) when cancelled mid-read
                            meta = dict((sampling or {}).get("_meta") or request_meta("", [], None, None, who))
                            # one request can call run() several times (an MCP tool round, the thinking budget's second call):
                            # the first keeps the request's id, the later ones get -2, -3 ... so no row or detail overwrites another
                            shared = (sampling or {}).get("_meta")
                            if shared is not None:
                                calls = shared["calls"] = shared.get("calls", 0) + 1
                                if calls > 1:
                                    meta["id"] = f"{shared['id']}-{calls}"
                            meta.pop("calls", None)
                            prompt = meta.pop("_prompt", None)                  # Q8: only when asked for; the detail file, never the row
                            if shared is not None:
                                shared.pop("_prompt", None)                   # once per request: the later rounds of an agent run do not write it again
                            chunks = chunk_stats(last.get("prefill_points") or [], last.get("reused") or 0)
                            decode = window_rates(tok_times)
                            rec = summary_record({
                                **meta,
                                "decode": {k: v for k, v in decode.items() if k != "series"} if decode else None,
                                "prompt_kept": prompt is not None,
                                "prefill": {k: v for k, v in chunks.items() if k != "items"} if chunks else None,
                                "projection": (sampling or {}).get("experimental_speed_projection") is not False
                                if loaded else None,
                                "time": started, "duration_s": round(time.time() - started, 1), "finish": finish,
                                "prompt_tokens": seen, "reused": last.get("reused"), "output_tokens": n,
                                # the request's whole prompt, and the tokens read of it (None: an older engine)
                                "prompt_total": len(ids), "prompt_read": last.get("prompt_read"),
                                "reasoning_recoveries": recovery_count,
                                "reasoning_repeat_coverage": round(repeat_coverage, 3),
                                "engine_generated": last.get("generated"), "stats": last.get("stats"),
                                "prompt_ms": last.get("prompt_ms"), "decode_ms": last.get("decode_ms"),
                                "decode_tok_s": round(last["generated"] / (last["decode_ms"] / 1000), 1)
                                if n and last.get("generated") and last.get("decode_ms") else None,
                                "cjk_chars": cjk_guard.count_han(self.tok.decode(raw_ids)) if raw_ids else 0,   # #49 S4
                                "hit_rate": hit_rate, "pcie_share": pcie_share, "ram_blobs": last.get("ram_blobs"),
                                "file_blobs": last.get("file_blobs"), "file_mb": last.get("file_mb"),
                                # #457: the speculative drafts from the DONE line (None: the engine did not say)
                                "drafts_offered": last.get("drafts_offered"),
                                "drafts_accepted": last.get("drafts_accepted")})
                            tel = getattr(self, "telemetry", None)       # xeno #237: the hardware over this request
                            hw, hw_series = tel.request_hw(started, time.time()) if tel is not None else (None, [])
                            rec["hw"] = hw
                            self.history.append(rec)
                            to_disk = (rec, {"who": rec.get("who"),     # #310: whose request (the route checks its summary's)
                                             "prefill_chunks": chunks["items"] if chunks else [],
                                             "decode_series": decode["series"] if decode else [],
                                             "stats": last.get("stats"),
                                             "hw_series": hw_series,
                                             **({"prompt": prompt} if prompt is not None else {})})   # written after the lock is let go
                            t = self.totals
                            t["requests"] += 1
                            t["prompt_tokens"] += seen
                            t["reused"] += last.get("reused") or 0
                            t["output_tokens"] += n
                            t["prompt_ms"] += last.get("prompt_ms") or 0.0
                            t["decode_ms"] += last.get("decode_ms") or 0.0
                            t["drafts_offered"] += last.get("drafts_offered") or 0
                            t["drafts_accepted"] += last.get("drafts_accepted") or 0
                            ft0 = st.get("first_token")             # this request's own (#465: one per request)
                            self.latencies.observe(ft0 - started if ft0 else None, n,
                                                   (last.get("decode_ms") or 0.0) / 1000, time.time() - started)
                            fresh = getattr(self.engine, "last", None)
                            if fresh is not None and fresh is not before:      # the engine's clock for THIS request
                                timings = request_timings(seen, n, last)
                                self.last_timings = dict(timings, at=int(time.time())) if timings else None
                            self.last_request_at = time.time()
                            now = time.time()
                            el = now - st.get("started", now)
                            ft = st.get("first_token")
                            tok_rate = n / max(1e-6, now - ft) if ft else 0.0
                            hit_msg = f", expert cache {hit_rate*100:.1f}% hit" if hit_rate is not None else ""
                            if hit_msg and pcie_share:
                                hit_msg += f" (+{pcie_share*100:.1f}% of the routed experts over PCIe)"
                            print(f"[strata] done: {n} tokens in {el:.0f} s ({tok_rate:.1f} tok/s) "
                                  f"({finish}, cancel={cancel.is_set()}){hit_msg}", flush=True)
                            if finish != "error" and last.get("prompt_ms") is not None:   # xeno #49 S5
                                for line in timing_report(last, len(ids), el):
                                    print(line, flush=True)
                            if finish == "length" and parser.state in ("reasoning", "rcall"):   # #530
                                print("[strata] the reply reached max tokens while still thinking, so it has no "
                                      "answer: a thinking budget (reasoning_budget_tokens, in the request or in "
                                      "strata-<model>.json for every request) leaves room to answer", flush=True)
                            if os.environ.get("STRATA_DEBUG") and raw_ids:   # xeno: ascii(), never a UnicodeEncodeError
                                print(f"[strata] raw: {ascii(self.tok.decode(raw_ids))}", flush=True)
                        st["busy"] = False
                        self.status["last_stop_reason"] = stop_detail or finish
                        st.pop("tail", None)                     # #212: the answer's end is not kept once it is done
                        st.pop("tool", None)
                        if par:
                            self.live_reqs.pop(id(st), None)
                            if self.live_reqs:              # the newest request still running
                                self.status.update(list(self.live_reqs.values())[-1][0])
                            else:
                                self.status.update(busy=False)
                                self.status.pop("tail", None)
                                self.status.pop("tool", None)
                    if to_disk is not None:                     # on disk, kept: outside status_lock (it is /status's and /metrics's lock),
                        try:                                    # and a full disk must not fail the request
                            self.hstore.append(to_disk[0])
                            self.hstore.write_detail(to_disk[0]["id"], to_disk[1])
                        except OSError as e:
                            print(f"[strata] history not saved: {e}", flush=True)
        finally:
            if emb:
                Path(emb).unlink(missing_ok=True)
        for ev in cut(parser.finish(finish)):
            yield "event", ev
        if parser.rescued or parser.refused:
            self.totals["tool_calls_from_reasoning"] = self.totals.get("tool_calls_from_reasoning", 0) + parser.rescued
            print(f"[strata] tool calls inside the thinking: {parser.rescued} read as calls, {parser.refused} kept as "
                  f"reasoning (quoted, or the turn did not end on a stop)", flush=True)
        if stops is not None and stops.hit is None and stops.held:
            yield "event", Event("content", stops.flush())     # the held tail was not a stop string after all
        if stops is not None and stops.hit is not None:
            finish = "stop"
            with self.status_lock:
                self.status["last_stop_reason"] = "stop_sequence"
        done = {"finish": finish, "completion_tokens": n, "reused": (timings or {}).get("cache_n", 0),
                "timings": timings, "reasoning_tokens": thinking_n,
                "reasoning_recoveries": recovery_count}
        if stops is not None and stops.hit is not None:
            done["stop_sequence"] = stops.hit
        if stop_detail:
            done["stop_detail"] = stop_detail
        yield "done", done

def prompt_tokens_seen(prompt_tokens: int, last: dict) -> int:
    """#471: the prompt tokens a request got through - all of them, unless the engine's DONE line says a cancel stopped
    its prompt read part-way (then the reused ones plus those read).  /metrics' history and totals count these, so a
    cancelled read is neither recorded as the whole prompt nor given a rate from tokens it never read.  An engine
    before 0.1.36 does not say (no `prompt_read`): the whole prompt, as before."""
    read = last.get("prompt_read")
    if read is None or last.get("finish") != "cancel":
        return prompt_tokens
    return min(prompt_tokens, int(last.get("reused") or 0) + int(read))


def request_timings(prompt_tokens: int, generated: int, last: dict) -> dict | None:
    """One request's `timings` in llama.cpp's names (what its clients show as speed), from the engine's own clock
    (StrataEngine.last): prompt_n is what was read, cache_n what the conversation cache already held.  None when the
    engine keeps no clock (MockEngine)."""
    if last.get("prompt_ms") is None:
        return None
    cache_n = int(last.get("reused") or 0)
    prompt_n, prompt_ms, decode_ms = max(0, prompt_tokens - cache_n), float(last["prompt_ms"]), float(last.get("decode_ms") or 0)
    decoded = int(last.get("generated") or generated)          # the engine's count gives its rate, as /metrics does
    return {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": round(prompt_ms, 1),
            "prompt_per_token_ms": round(prompt_ms / prompt_n, 3) if prompt_n else None,
            "prompt_per_second": round(prompt_n / (prompt_ms / 1000), 1) if prompt_n and prompt_ms > 0 else None,
            "predicted_n": generated, "predicted_ms": round(decode_ms, 1),
            "predicted_per_token_ms": round(decode_ms / decoded, 3) if decoded else None,
            "predicted_per_second": round(decoded / (decode_ms / 1000), 1) if decoded and decode_ms > 0 else None,
            # the speculative drafts, as llama.cpp names them (from PR #83, @mikicvi): only when the engine reported them
            **({"draft_n": int(last["drafts_offered"]), "draft_n_accepted": int(last["drafts_accepted"])}
               if last.get("drafts_offered") is not None else {})}


def request_stats(segments: list[dict]) -> dict:
    """Keep input/cache accounting at the original API boundary and total native work across continuations."""
    if not segments:
        return {}
    result = dict(segments[-1])
    for key in ("reused", "prompt_read"):
        if key in segments[0]:
            result[key] = segments[0][key]
        else:
            result.pop(key, None)
    for key in ("prompt_ms", "decode_ms", "generated", "drafts_offered", "drafts_accepted",
                "hits", "lookups", "offloaded", "ram_blobs", "file_blobs", "file_mb"):
        if any(segment.get(key) is not None for segment in segments):
            result[key] = sum(segment.get(key) or 0 for segment in segments)
    return result


def _debug_req(api, req, messages, tools, max_new, thinking, prompt_tokens):
    """One compact line per request while diagnosing blank/empty turns. Set STRATA_DEBUG=1 to enable."""
    if not os.environ.get("STRATA_DEBUG"):
        return
    last = messages[-1] if messages else {}
    body = last.get("content")
    if isinstance(body, list):
        body = " ".join(p.get("text", "") for p in body if isinstance(p, dict))
    preview = (str(body or "")[:80]).replace("\n", " ")
    print(f"[strata] req {api}: msgs={len(messages)} tools={len(tools or [])} "
          f"max_tokens_raw={req.get('max_tokens')!r}/{req.get('max_completion_tokens')!r} "
          f"max_new={max_new} thinking={thinking} stream={bool(req.get('stream'))} "
          f"prompt_tokens={prompt_tokens} last={last.get('role')!r}:{preview!r}", flush=True)


# ------------------------------------------------------------------------------------------------ MCP tool loop
AGENT_ROUNDS = 100      # tool rounds in one answer when the chat has the coding tools
def run_with_mcp(svc: Service, hub, messages, tools, kw, ids, thinking, max_new, max_req, sampling, cancel,
                 mcp_names, agent_run=None, rounds_limit=None):
    """Service.run with the MCP tools executed here: the model writes a call to an MCP tool, the server runs it, adds
    the call and its result to the conversation and lets the model continue - up to `max_rounds` times.  Yields what
    Service.run yields (text, thinking, the request's own tool calls) plus ("mcp", {...}) for the tool activity, and
    one ("done", ...) at the very end with the output tokens of every round.

    `mcp_names`: the MCP tools this request offered; any other call is one of the request's own tools and ends the
    turn as always (the client answers it).  MCP calls written in the same answer are then not run (their results
    could not reach the model before the client's)."""
    max_rounds = max(int(hub.settings["max_rounds"]), AGENT_ROUNDS) if agent_run is not None else int(hub.settings["max_rounds"])      # coding takes many rounds
    if rounds_limit is not None:                                  # a helper (serve/subagent.py) stops sooner
        max_rounds = rounds_limit
    total, rounds, done = 0, 0, None
    dec_n, dec_ms = 0, 0.0                                       # the tokens decoded and the time the engine spent decoding them, over every round
    messages = list(messages)
    if agent_run is not None:
        for e in agent_run.drain():                      # what the prompt hooks did, before the first word
            yield "mcp", e
    while True:
        text, reasoning, calls, own_calls = [], [], [], 0
        for kind, x in svc.run(ids, thinking, tools, max_new, sampling, cancel):
            if kind == "done":
                done = x
                continue
            if kind == "event":
                ev: Event = x
                if ev.call is not None and ev.call.name in mcp_names:
                    if ev.kind == "tool_start":          # the model has started writing a call: say so at once
                        yield "mcp", {"event": "start", "id": ev.call.id, "name": ev.call.name}
                    elif ev.kind == "tool_call":
                        calls.append(ev.call)
                    continue                             # its argument pieces are not streamed to the client
                if ev.kind == "tool_call":
                    own_calls += 1
                elif ev.kind == "content":
                    text.append(ev.text)
                elif ev.kind == "reasoning":
                    reasoning.append(ev.text)
            yield kind, x
        total += done["completion_tokens"]
        tm = done.get("timings") or {}
        if tm.get("predicted_per_second") and tm.get("predicted_ms"):
            dec_n += round(tm["predicted_per_second"] * tm["predicted_ms"] / 1000)
            dec_ms += tm["predicted_ms"]
        run_them = calls and not own_calls and done["finish"] == "stop" and not cancel.is_set()
        if run_them and rounds >= max_rounds:
            yield "mcp", {"event": "limit", "max_rounds": max_rounds}
            if agent_run is not None:
                agent_run.limit_hit = True
            run_them = False
        if not run_them and agent_run is not None and not calls and not own_calls and done["finish"] == "stop" and not cancel.is_set():
            said_now = agent_run.take_steers()                    # the answer was about to end, and the user had sent something meanwhile: it is read now, in the same run
            if said_now:
                messages.append({"role": "assistant", "content": "".join(text).strip(), **({"reasoning_content": "".join(reasoning).strip()} if reasoning else {})})
                for said in said_now:
                    messages.append({"role": "user", "content": said})
                    yield "mcp", {"event": "steered", "text": said}
                ids, thinking, max_new = svc.prepare(messages, tools, kw, max_req, 0 if sampling.get("stream") else 1)
                continue
        if not run_them:
            for c in calls:                              # announced, never run: close them in the client's view
                yield "mcp", {"event": "result", "id": c.id, "ok": False, "skipped": True, "text": "not run",
                              "chars": 0, "truncated": False, "ms": 0}
            break
        rounds += 1
        results = []
        for c in calls:
            s, tool = hub.routes().get(c.name, (None, c.name))
            remote = s is not None and getattr(s, "machine", None) == "my"      # #310 part 2: a group of the engineer's own PC
            yield "mcp", {"event": "call", "id": c.id, "name": c.name, "server": s.name if s else None,
                          "tool": tool, "arguments": c.arguments, "round": rounds, **({"machine": "my"} if remote else {})}
            states: queue.SimpleQueue = queue.SimpleQueue()      # the runner group's progress words, from the worker thread
            said = "sending"
            if remote:
                yield "mcp", {"event": "remote", "id": c.id, "state": said}
            # The call runs on a thread while this generator keeps yielding heartbeats: they reach the client as
            # keep-alives, which is how a client that went away (the web app's Stop) is noticed during a slow tool.
            box = {}
            if agent_run is not None:
                agent_run.current = c.id

            def work(c=c, box=box, remote=remote, states=states):
                try:
                    ctx = agent_run.ctx if agent_run is not None else None
                    box["r"] = hub.call(c.name, c.arguments, cancel, ctx, progress=states.put) if remote \
                        else hub.call(c.name, c.arguments, cancel, ctx)
                except McpCancelled:
                    box["cancelled"] = True

            def progress(c=c, states=states):
                """The runner's `running` / `offline` as `remote` events, each once in a row (`sending` was said)."""
                nonlocal said
                while not states.empty():
                    word = states.get()
                    if word in ("running", "offline") and word != said:
                        said = word
                        yield "mcp", {"event": "remote", "id": c.id, "state": word}
            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            try:
                beat = 0
                while worker.is_alive():
                    worker.join(0.2)
                    if agent_run is not None:
                        for e in agent_run.drain():      # a question for the user, a todo list, ...: to the page at once
                            yield "mcp", e
                    yield from progress()
                    beat += 1
                    if worker.is_alive() and beat % 5 == 0:
                        yield "ping", None
                if agent_run is not None:
                    for e in agent_run.drain():
                        yield "mcp", e
                yield from progress()
            except GeneratorExit:
                cancel.set()                             # the client is gone: stop the tool too (a question that is open ends too)
                raise
            if "r" not in box:
                break
            r = box["r"]
            print(f"[strata] tool {c.name}: {'ok' if r['ok'] else 'error'}, {r['chars']:,} characters in "
                  f"{r['ms'] / 1000:.1f} s{' (truncated for the model)' if r['truncated'] else ''}", flush=True)
            results.append([{"type": "text", "text": r["text"]}, *({"type": "image", "source": u} for u in r["images"])] if r.get("images") else r["text"])      # an image goes to the model as an image
            yield "mcp", {"event": "result", "id": c.id, **{k: r[k] for k in ("ok", "text", "chars", "truncated", "ms")}}
        if cancel.is_set() or len(results) < len(calls):
            done = {**done, "finish": "cancel"}
            break
        messages.append({"role": "assistant", "content": "".join(text).strip(),
                         **({"reasoning_content": "".join(reasoning).strip()} if reasoning else {}),
                         "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in calls]})
        messages += [{"role": "tool", "content": r} for r in results]
        if agent_run is not None:
            for said in agent_run.take_steers():                  # what the user sent while the agent worked is read at this step
                messages.append({"role": "user", "content": said})
                yield "mcp", {"event": "steered", "text": said}
        ids, thinking, max_new = svc.prepare(messages, tools, kw, max_req, 0 if sampling.get("stream") else 1)
    if agent_run is not None and getattr(agent_run, "hooks", None) and not cancel.is_set() and done["finish"] != "cancel":
        agent_run.current = None
        worker = threading.Thread(target=agent_run.hooks.stop, args=("".join(text).strip(),), daemon=True)      # the model has finished: the user's stop hooks (their output is for the user)
        worker.start()
        try:
            beat = 0
            while worker.is_alive():
                worker.join(0.2)
                for e in agent_run.drain():
                    yield "mcp", e
                beat += 1
                if worker.is_alive() and beat % 5 == 0:
                    yield "ping", None
        except GeneratorExit:
            cancel.set()
            raise
        for e in agent_run.drain():
            yield "mcp", e
    if rounds and dec_ms > 0 and done.get("timings"):            # several rounds: the speed of the decoding is that of all of them, not of the last (a tool's time and the next prompt's reading are not decoding)
        done = {**done, "timings": {**done["timings"], "predicted_n": total, "predicted_ms": round(dec_ms, 1), "predicted_per_second": round(dec_n / (dec_ms / 1000), 1), "rounds": rounds + 1}}
    yield "done", {**done, "completion_tokens": total, "prompt_tokens": len(ids)}


# ------------------------------------------------------------------------------------------------ OpenAI
# The engine's last PP line stops short of the prompt: up to `--short-read` tokens (generate.cpp's `short_read`, 64 by
# default) are read through the verify windows instead of the batched path, so the batched path never reports the whole
# prompt.  A PP that close to the end says the prompt is read.  It matters because a prompt shorter than one `--prefill`
# chunk sends exactly one PP, that one: a client would see a bar that appears already full.  Such a line is dropped, and
# the answer's first token ends the progress for the client anyway.
PP_DONE_TAIL = 64


def prompt_progress(svc: Service):
    """The last PP line as llama.cpp's `prompt_progress`, so a client written for llama.cpp reads both servers: the
    same four fields, `time_ms` elapsed since prompt reading started, and `processed` counting from token 0 so that a
    client takes the reused prefix out with `cache` (llama.cpp's rule: the timed progress is
    `(processed-cache)/(total-cache)`).  None when there is nothing to say: no PP yet, or the line that says the prompt
    is read.  An engine too old to print RESUME gives `cache` 0, which only makes a reused prefix count as work."""
    engine = svc.engine
    if getattr(engine, "batch", 0):      # --batch: the engine's progress is one global line, not this request's
        return None
    progress = getattr(engine, "progress", None)
    if not progress:
        return None
    read, total = progress
    if total <= 0 or total - read <= PP_DONE_TAIL:
        return None
    return {"total": int(total), "cache": int(getattr(engine, "reused", 0) or 0), "processed": int(read),
            "time_ms": int(getattr(engine, "progress_ms", 0) or 0)}


def openai_chunks(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, run=None, force=None):
    """`run`: the events to send instead of Service.run's (run_with_mcp); its ("mcp", {...}) items become chunks with
    an empty delta and a `strata_mcp` field, which only the web app reads.  `force`: see Service.run."""
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
    model = svc.model_for(req)
    # llama.cpp's flag, and its default: a client that wants prompt progress asks for it, and nothing changes for one
    # that does not.  Strata already sends a keep-alive comment on every PP line, so this only fills that line in.
    want_progress = bool(req.get("return_progress"))

    def chunk(delta, finish=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield chunk({"role": "assistant", "content": ""})
    calls = 0
    streamed = {}                                  # tool call id -> index, for calls sent piece by piece
    finished = set()                               # ... and the ones whose final tool_call came (#211)
    for kind, x in run if run is not None else svc.run(ids, thinking, tools, max_new, req, cancel, force=force):
        if kind == "ping":
            # a PP line: the prompt is being read and there is nothing to say yet.  It used to be only the SSE comment;
            # now it carries the progress when the client asked, and stays the comment when there is nothing to report.
            pp = prompt_progress(svc) if want_progress else None
            yield None if pp is None else {**chunk({}), "prompt_progress": pp}
        elif kind == "mcp":
            c = chunk({})
            c["strata_mcp"] = x
            yield c
        elif kind == "event":
            ev: Event = x
            if ev.kind == "reasoning" and ev.text:
                yield chunk({"reasoning_content": ev.text})
            elif ev.kind == "content" and ev.text:
                yield chunk({"content": ev.text})
            elif ev.kind == "tool_start":
                streamed[ev.call.id] = calls
                calls += 1
                yield chunk({"tool_calls": [{"index": streamed[ev.call.id], "id": ev.call.id, "type": "function",
                                             "function": {"name": ev.call.name, "arguments": ""}}]})
            elif ev.kind == "tool_args":
                yield chunk({"tool_calls": [{"index": streamed[ev.call.id], "function": {"arguments": ev.text}}]})
            elif ev.kind == "tool_call" and ev.call.id in streamed:
                finished.add(ev.call.id)
            elif ev.kind == "tool_call":
                yield chunk({"tool_calls": [{"index": calls, "id": ev.call.id, "type": "function",
                                             "function": {"name": ev.call.name,
                                                          "arguments": json.dumps(ev.call.arguments, ensure_ascii=False)}}]})
                calls += 1
        else:
            # a streamed call without its final tool_call is one the output ended inside: not "tool_calls" (#211)
            whole = calls and streamed.keys() <= finished
            finish = "tool_calls" if whole and x["finish"] == "stop" else {"cancel": "stop"}.get(x["finish"], x["finish"])
            last = chunk({}, finish)
            pt = x.get("prompt_tokens", len(ids))     # after MCP rounds: the last round's prompt
            last["usage"] = {"prompt_tokens": pt, "completion_tokens": x["completion_tokens"],
                             "total_tokens": pt + x["completion_tokens"],
                             # the part of the prompt the conversation cache already held (OpenAI's field)
                             "prompt_tokens_details": {"cached_tokens": x.get("reused") or 0}}
            if x.get("timings"):
                last["timings"] = x["timings"]          # llama.cpp's field: the speed its clients show
            if x.get("stop_detail"):
                last["timings"] = {**(last.get("timings") or {}), "stop_reason": x["stop_detail"]}
            yield last


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def openai_collect(chunks) -> dict:
    content, reasoning, by_index, last, mcp = [], [], {}, None, []
    for c in chunks:
        if c is None:                              # a heartbeat
            continue
        if c.get("strata_mcp"):
            mcp.append(c["strata_mcp"])
        d = c["choices"][0]["delta"]
        content.append(d.get("content") or "")
        reasoning.append(d.get("reasoning_content") or "")
        for tc in d.get("tool_calls") or []:       # streamed calls arrive in pieces: merge them by index
            cur = by_index.setdefault(tc.get("index", len(by_index)), {"id": None, "type": "function",
                                                                        "function": {"name": "", "arguments": ""}})
            cur["id"] = tc.get("id") or cur["id"]
            fn = tc.get("function") or {}
            cur["function"]["name"] += fn.get("name") or ""
            cur["function"]["arguments"] += fn.get("arguments") or ""
        last = c
    calls = [by_index[i] for i in sorted(by_index)]
    if last["choices"][0]["finish_reason"] != "tool_calls":
        calls = [c for c in calls if _is_json(c["function"]["arguments"])]   # a call the output ended inside (#211)
    msg = {"role": "assistant", "content": "".join(content) or None}
    if "".join(reasoning):
        msg["reasoning_content"] = "".join(reasoning)
    if calls:
        msg["tool_calls"] = calls
    if mcp:
        msg["strata_mcp"] = mcp
    out = {"id": last["id"], "object": "chat.completion", "created": last["created"], "model": last["model"],
           "choices": [{"index": 0, "message": msg, "finish_reason": last["choices"][0]["finish_reason"]}],
           "usage": last["usage"]}
    if last.get("timings"):
        out["timings"] = last["timings"]
    return out


def structured_chunks(chunks, validator):
    """Buffer structured streams so a client never receives unvalidated content."""
    buffered = []
    heartbeat = time.monotonic()
    try:
        for chunk in chunks:
            progress = chunk is not None and "prompt_progress" in chunk
            if chunk is not None and not progress:
                buffered.append(chunk)      # a progress chunk carries no content: there is nothing in it to validate
            if chunk is None or progress or time.monotonic() - heartbeat >= 1:
                heartbeat = time.monotonic()
                yield chunk if progress else None
        result = openai_collect(buffered)
        choice = result["choices"][0]
        content = validated_json(choice["message"]["content"], validator, choice["finish_reason"])
        yield buffered[0]
        delta = {"content": content}
        if choice["message"].get("reasoning_content"):
            delta["reasoning_content"] = choice["message"]["reasoning_content"]
        yield {**buffered[0], "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        yield buffered[-1]
    finally:
        chunks.close()


# ------------------------------------------------------------------------------------------------ Anthropic
def anthropic_events(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, force=None):
    mid = "msg_" + uuid.uuid4().hex[:24]
    yield "message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": svc.model_for(req), "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": len(ids), "output_tokens": 0}}}
    index, open_kind, used_tool = -1, None, False

    def close():
        if open_kind == "thinking":
            yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                          "delta": {"type": "signature_delta", "signature": ""}}
        yield "content_block_stop", {"type": "content_block_stop", "index": index}

    streamed, finished = set(), set()              # calls sent piece by piece; those whose final tool_call came (#211)
    choice = req.get("tool_choice") if isinstance(req.get("tool_choice"), dict) else {}
    one_call, first_done = bool(choice.get("disable_parallel_tool_use")), False   # xeno #49 S2
    for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel, force=force):
        if kind == "ping":
            yield None
            continue
        if kind == "event":
            ev: Event = x
            if first_done:                          # disable_parallel_tool_use: nothing after the first call
                continue
            if one_call and ev.kind == "tool_call":
                first_done = True
                cancel.server_stop = True           # the server's own stop, not a client cancel (Service.run)
                cancel.set()
            if ev.kind == "tool_args":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "input_json_delta", "partial_json": ev.text}}
                continue
            if ev.kind == "tool_call" and ev.call.id in streamed:
                finished.add(ev.call.id)
                continue
            want = {"reasoning": "thinking", "content": "text", "tool_call": "tool_use", "tool_start": "tool_use"}[ev.kind]
            if ev.kind not in ("tool_call", "tool_start") and not ev.text:
                continue
            if ev.kind == "tool_start":
                streamed.add(ev.call.id)
                used_tool = True
            if open_kind != want or want == "tool_use":
                if open_kind is not None:
                    yield from close()
                index += 1
                open_kind = want
                block = {"thinking": {"type": "thinking", "thinking": "", "signature": ""},
                         "text": {"type": "text", "text": ""},
                         "tool_use": {"type": "tool_use", "id": ev.call.id if ev.call else "", "name":
                                      ev.call.name if ev.call else "", "input": {}}}[want]
                yield "content_block_start", {"type": "content_block_start", "index": index, "content_block": block}
            if want == "thinking":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "thinking_delta", "thinking": ev.text}}
            elif want == "text":
                yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                              "delta": {"type": "text_delta", "text": ev.text}}
            elif ev.kind == "tool_start":
                pass                                # its input follows as tool_args pieces
            else:
                used_tool = True
                yield "content_block_delta", {"type": "content_block_delta", "index": index, "delta": {
                    "type": "input_json_delta", "partial_json": json.dumps(ev.call.arguments, ensure_ascii=False)}}
        else:
            if open_kind is not None:
                yield from close()
            # a streamed call without its final tool_call is one the output ended inside: not "tool_use" (#211)
            stop = "stop_sequence" if x.get("stop_sequence") is not None else \
                   "tool_use" if used_tool and streamed <= finished and (x["finish"] == "stop" or first_done) else \
                   {"stop": "end_turn", "length": "max_tokens", "cancel": "end_turn"}[x["finish"]]
            # the final counts, Anthropic's way: input_tokens leaves out what the conversation cache already held,
            # which is cache_read_input_tokens (message_start could only say the whole prompt).  At least the last
            # prompt token is always read, and Claude Code takes an input_tokens of 0 as "not given" and keeps
            # message_start's whole prompt beside cache_read_input_tokens: the prompt counted twice (#1013)
            reused = max(0, min(x.get("reused") or 0, len(ids) - 1))
            yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": stop,
                                                                       "stop_sequence": x.get("stop_sequence")},
                                    "usage": {"input_tokens": len(ids) - reused, "cache_read_input_tokens": reused,
                                              "output_tokens": x["completion_tokens"]}}
            yield "message_stop", {"type": "message_stop"}


def sse_tracer():
    """STRATA_TRACE_SSE=<file> (xeno #49 S5): one JSON line per SSE event sent on /v1/messages - what the client got
    and when (seconds since the stream began), to diagnose what Claude Code shows. Unset: a no-op."""
    path = os.environ.get("STRATA_TRACE_SSE")
    if not path:
        return lambda name, e: None
    rid, t0 = uuid.uuid4().hex[:12], time.perf_counter()

    def trace(name, e):
        rec = {"rid": rid, "t": round(time.perf_counter() - t0, 3), "ev": name}
        if isinstance(e, dict):
            for k in ("index", "delta", "content_block", "usage"):
                if k in e:
                    rec[k] = e[k]
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as err:
            print(f"[strata] SSE trace write failed ({path}): {err}", flush=True)
    return trace


def anthropic_collect(events) -> dict:
    msg, blocks = None, []
    for item in events:
        if item is None:                           # a heartbeat
            continue
        name, e = item
        if name == "message_start":
            msg = e["message"]
        elif name == "content_block_start":
            blocks.append(dict(e["content_block"]))
        elif name == "content_block_delta":
            d, b = e["delta"], blocks[-1]
            if d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            elif d["type"] == "signature_delta":
                b["signature"] = d["signature"]
            else:                                  # input_json_delta pieces: parsed when complete
                b["_json"] = b.get("_json", "") + d["partial_json"]
        elif name == "content_block_stop" and blocks and "_json" in blocks[-1]:
            b = blocks[-1]
            try:
                b["input"] = json.loads(b.pop("_json") or "{}")
            except ValueError:                     # a call the output ended inside (#211): it has no input to give
                blocks.pop()
        elif name == "message_delta":
            msg["stop_reason"] = e["delta"]["stop_reason"]
            msg["stop_sequence"] = e["delta"].get("stop_sequence")
            msg["usage"].update(e["usage"])
    msg["content"] = blocks
    return msg


# ------------------------------------------------------------------------------------------------ HTTP
CHUNKED_BODY_MAX = 256 << 20                # #893: the most a Transfer-Encoding: chunked body may hold (read into memory)


def body_limit() -> int:
    """The most a request body may hold, in bytes (read into memory): 256 MiB, which is a million-token conversation
    with room to spare; STRATA_MAX_BODY_MIB changes it (0 or an unreadable value: the default).  A larger one is
    answered 413 before it is read (a Content-Length of 10 TB used to be read until the client gave up)."""
    try:
        mib = int(os.environ.get("STRATA_MAX_BODY_MIB", "0"))
    except ValueError:
        mib = 0
    return mib << 20 if mib > 0 else CHUNKED_BODY_MAX


class BadBody(Exception):
    """A request body that cannot be read (a malformed or oversized chunked body): the status and the sentence."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# #310: POST routes that only an owner key may use in team mode (with /slots/*), and what each one does
OWNER_POSTS = {"/settings": "settings can be changed", "/config": "the run config can be changed",
               "/unload": "the model can be loaded or unloaded", "/load": "the model can be loaded or unloaded",
               "/v1/unload": "the model can be loaded or unloaded", "/v1/load": "the model can be loaded or unloaded",
               "/v1/vram": "the VRAM reserve can be changed", "/metrics/keep": "prompts can be kept"}
NO_RUN = "there is no such run (it ended long ago, or the server was restarted)"
# the owner gate's words of the page routes on the workstation (serve/page_routes.py; with `runner=` they act on the
# caller's own PC instead, #310 part 2)
PAGE_GATES = {"/agent/folders": "this PC's folders can be listed", "/agent/git": "a folder's Git state can be read",
              "/agent/git/diff": "a folder's Git state can be read", "/agent/files": "the project's files can be read",
              "/agent/mention": "the project's files can be read", "/agent/memory": "the project's notes can be read",
              "/agent/memory/file": "the project's notes can be read", "/agent/rewind": "files can be put back",
              "/agent/checkpoints/forget": "files can be put back"}
# #310 part 2: the runner channel's bodies (a reply carries a tool's result: a Read image fits) and answers
RUNNER_HELLO_MAX, RUNNER_REPLY_MAX = 4 << 20, 32 << 20
NO_RUNNER = "there is no such runner (say hello again)"
RUNNER_REPLACED = {"error": {"type": "runner_replaced", "message": "a newer Strata Desktop runner on this PC took this "
                                                                   "runner's place; this one stops"}}
# #313: the runner's own requests keep their connection (HTTP/1.1, Content-Length): a new TCP connection per poll and
# per reply cost 10-14 ms median on this PC (bimodal 0.2-0.5 or 10-25 ms), and 1 RTT more (plus 1-2 for TLS) on a
# real link.  Every other route stays HTTP/1.0 with Connection: close (end_headers).
RUNNER_KEEP = {("POST", "/runner/hello"), ("GET", "/runner/next"), ("POST", "/runner/reply")}


def route_path(target: str) -> str:
    """The path do_GET / do_POST route a request target on: the classic app's /classic prefix taken off once, the
    query cut, trailing slashes dropped.  The keep-alive decision (#313) uses it too, so it keeps exactly the
    requests the runner's routes answer; a fragment or an absolute-form target ends in the generic 404 and closes."""
    if target.startswith("/classic/"):
        target = target[len("/classic"):]
    return target.split("?")[0].rstrip("/")
RUNNER_IDLE_S = 90.0            # a kept connection with no next request for this long is closed (the runner's own limit
                                # is shorter, serve/runner.py KEEP_IDLE_S, so it rarely sends on one the server closed)


def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection

        record = None                                       # #332: this request's monitor record, if kept
        watch_done = None                                   # #430 #431: stops this request's disconnect watcher
        body_read = False                                   # whether a handler took this request's body
        identity = None                                     # #310: who sent this request (_authorized sets it)
        device_id = ""                                      # #310: the device it came from, in team mode
        team = False                                        # #310: the server was in team mode when _authorized ran
        DRAIN_SECONDS = 5                                   # the longest an unread body is read and dropped

        def log_message(self, fmt, *args):
            pass

        def handle_one_request(self):
            # #313: a runner's kept connection brings every one of its requests to this one object: each starts clean
            self.answer_started = self.body_read = self.body_done = self.team = self.length_sent = False
            self.record = self.watch_done = self.identity = self.headers = None
            self.device_id, self.status_sent = "", 0
            self.protocol_version = "HTTP/1.0"
            super().handle_one_request()
            self._drain_body()
            if not self.close_connection:                    # kept (only the runner's routes are): until its next
                try:                                         # request, or RUNNER_IDLE_S without one
                    self.connection.settimeout(RUNNER_IDLE_S)
                    self.server.keep(self.connection, True)
                except (OSError, AttributeError):
                    self.close_connection = True

        def finish(self):
            keep = getattr(self.server, "keep", None)
            if keep is not None:
                keep(self.connection, False)
            super().finish()

        def send_response(self, code, message=None):
            self.answer_started = True                 # a malformed-request answer can only replace one not yet begun
            self.status_sent = code
            super().send_response(code, message)

        def send_header(self, keyword, value):
            if keyword.lower() == "content-length":
                self.length_sent = True
            super().send_header(keyword, value)

        def _keeps(self) -> bool:
            """#313: this answer leaves the connection open - a runner route asked over HTTP/1.1 (not `Connection:
            close`), its body read to the end (`body_done`: not `body_read`, which is set before a read that may
            fail half way) or none, a Content-Length, and no 5xx."""
            if self.protocol_version != "HTTP/1.1" or self.close_connection or not self.length_sent:
                return False
            if self.status_sent >= 500:
                return False
            headers = getattr(self, "headers", None)
            no_body = headers is not None and not self._chunked() and headers.get("Content-Length", "0").strip() == "0"
            return self.body_done or no_body

        def _chunked(self) -> bool:
            """#893: a body sent as Transfer-Encoding: chunked (a relay or proxy that does not buffer it).  By RFC 9112
            it wins over a Content-Length."""
            te = self.headers.get("Transfer-Encoding", "") or ""
            return te.split(",")[-1].strip().lower() == "chunked"

        def _read_chunked(self, limit: int, keep: bool = True, deadline: float | None = None) -> bytes:
            """Decode a chunked body: at most `limit` bytes (BadBody 413 beyond it - the size is checked before a chunk
            is read, so a huge announced chunk is never allocated), BadBody 400 for a malformed one.  keep=False reads
            and drops it (the drain); `deadline` (monotonic) ends the read early."""
            parts, total = [], 0
            while True:
                if deadline is not None:
                    wait = deadline - time.monotonic()
                    if wait <= 0:
                        return b"".join(parts)
                    self.connection.settimeout(wait)
                line = self.rfile.readline(1025)
                if not line.endswith(b"\n"):
                    raise BadBody(400, "malformed chunked request body")
                try:
                    size = int(line.split(b";", 1)[0].strip(), 16)
                except ValueError:
                    raise BadBody(400, "malformed chunk size in the request body") from None
                if size < 0:
                    raise BadBody(400, "malformed chunk size in the request body")
                if size == 0:
                    for _ in range(64):                       # the trailers, including the final blank line
                        trailer = self.rfile.readline(8193)
                        if trailer == b"\r\n":
                            return b"".join(parts)
                        if not trailer.endswith(b"\r\n"):
                            raise BadBody(400, "malformed or incomplete chunked request trailers")
                    raise BadBody(400, "too many chunked request trailers")
                total += size
                if total > limit:
                    self.close_connection = True
                    raise BadBody(413, f"the request body is larger than {limit >> 20} MiB")
                left = size
                while left > 0:
                    piece = self.rfile.read(min(left, 1 << 20))
                    if not piece:
                        raise BadBody(400, "the chunked request body ended early")
                    left -= len(piece)
                    if keep:
                        parts.append(piece)
                if self.rfile.read(2) != b"\r\n":
                    raise BadBody(400, "malformed chunked request body")

        def _body(self) -> bytes:
            self.body_read = True
            limit = body_limit()
            if self._chunked():
                return self._read_chunked(limit)
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                raise BadBody(400, "invalid Content-Length") from None
            if length < 0:
                raise BadBody(400, "invalid Content-Length")
            if length > limit:
                self.close_connection = True               # the body is not read: the connection ends with the answer
                raise BadBody(413, f"the request body is larger than {limit >> 20} MiB "
                                   "(STRATA_MAX_BODY_MIB raises the limit)")
            return self.rfile.read(length)

        def _drain_body(self):
            """An answer sent before the body was read (a 401, a 403, /load, a method with no handler) must not close
            the connection on unread bytes: the close then sends a reset, and a client that sends its body after the
            headers (http.client, urllib, requests) gets a connection error instead of the answer.  So the body is
            read and dropped here, once, after an answer, in pieces so that its size is never held in memory.  What
            has not arrived DRAIN_SECONDS later is left unread: the limit is time, so that a conversation of many
            megabytes, at any speed the client has, still gets its answer."""
            headers = getattr(self, "headers", None)
            if self.body_read or headers is None:
                return
            if self._chunked():
                try:
                    self._read_chunked(body_limit(), keep=False, deadline=time.monotonic() + self.DRAIN_SECONDS)
                except (BadBody, OSError):
                    self.close_connection = True
                return
            try:
                left = int(headers.get("Content-Length", 0))
            except ValueError:
                return
            if left <= 0:
                return
            deadline = time.monotonic() + self.DRAIN_SECONDS
            try:
                while left > 0 and (wait := deadline - time.monotonic()) > 0:
                    self.connection.settimeout(wait)         # a client that never sends what it announced
                    piece = self.rfile.read1(min(left, 1 << 20))   # one socket read: read() would wait for it all
                    if not piece:
                        break
                    left -= len(piece)
            except OSError:
                pass

        def parse_request(self):
            """Without an API key, every request (any method) first passes the Host check: DNS rebinding protection
            (host_allowed).  With a key (the api_key, or team mode's named keys, #310) a rebinding page cannot
            authenticate, so the check is skipped: tunnels and proxies that pass their own name on keep working."""
            try:
                self.connection.settimeout(None)             # #313: a kept connection's idle limit (RUNNER_IDLE_S) is
            except OSError:                                  # for the wait for a request, not for serving it
                pass
            if not super().parse_request():
                return False
            host = self.headers.get("Host")
            # svc.allowed_hosts again: a name the config adds after serve() worked out host_names (xeno #71)
            if svc.key_required() or host_allowed(host, svc.host_names | set(svc.allowed_hosts), "*" in svc.allowed_hosts):
                if self.request_version == "HTTP/1.1" and (self.command, route_path(self.path)) in RUNNER_KEEP:
                    self.protocol_version = "HTTP/1.1"       # #313: the runner's routes may keep the connection
                    self.close_connection = (self.headers.get("Connection") or "").strip().lower() == "close"
                return True
            print(f"[strata] refused a request for Host {host!r} from {self.client_address[0]}: not a name this server "
                  f"answers to (add it to \"allowed_hosts\" in the config or STRATA_ALLOWED_HOSTS, or set an API key)",
                  flush=True)
            self._json(403, {"error": {"type": "forbidden", "message":
                             f"Host {host!r} is not allowed (DNS rebinding protection). Reaching Strata under this "
                             f"name on purpose? Add it to \"allowed_hosts\" in the config (strata-<model>.json) or to "
                             f"the STRATA_ALLOWED_HOSTS environment variable, or set an API key (\"api_key\"), which "
                             f"turns this check off"}})
            return False

        def _foreign_page(self) -> bool:
            """Without an API key, a /v1 POST from a browser page of another site (any site can POST text/plain
            there without a CORS preflight) would burn GPU time: an Origin header must name an allowed page, and
            then the body must be JSON.  No Origin (curl, the SDKs, other servers): any content type, as before."""
            origin = self.headers.get("Origin")
            if svc.key_required() or not origin:
                return False
            if not origin_allowed(origin, self.headers.get("Host"), svc.host_names,
                                  [*svc.trusted_origins, *svc.cors_origins]):
                print(f"[strata] refused an API request from the web page {origin!r} (no API key; add its host to "
                      f"\"allowed_hosts\" or its origin to \"cors_origins\" in the config)", flush=True)
                self._json(403, {"error": {"type": "forbidden", "message":
                                 f"web pages of {origin} may not use this server without an API key; set \"api_key\", "
                                 f"or add the page's host to \"allowed_hosts\" (or its origin to \"cors_origins\") "
                                 f"in the config"}})
                return True
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return True
            return False

        def end_headers(self):
            # HTTP/1.0 without keep-alive: the server closes the connection after every response.  Say so, or a
            # client that pools connections (.NET's SocketsHttpHandler) can put its next request on this socket
            # before the close reaches it, and that request fails with "response ended prematurely" without the
            # server ever reading it (a .NET agent app, 2026-10-03: 25 of ~1000 requests, 1-24 ms after the previous answer).
            # The one exception is the runner's own routes over HTTP/1.1 (#313, `_keeps`).
            if not self._keeps():
                self.send_header("Connection", "close")
            super().end_headers()

        def _watch_client(self, cancel: threading.Event) -> None:
            """#430 #431: cancel the request as soon as its client hangs up.  A non-streamed request writes nothing
            until it ends, and a streamed one only a keep-alive per prompt chunk (and the first write after a hang-up
            usually still succeeds), so a dropped request kept the engine busy until its answer or the whole prompt
            was done.  Every 0.5 s: the socket readable with nothing to read (EOF) means the client closed it.  A
            request is fully read here, and no client sends its next request before this one's answer (a runner's kept
            connection, #313, neither), so no later bytes are expected - data is not a hang-up."""
            done = self.watch_done = threading.Event()
            sock = self.connection

            def watch():
                while not done.wait(0.5) and not cancel.is_set():
                    try:
                        readable, _, _ = select.select([sock], [], [], 0)
                        gone = bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
                    except (ConnectionError, TimeoutError):
                        gone = True
                    except (OSError, ValueError):            # the socket was closed here: the request has ended
                        return
                    if gone:
                        self._note(outcome="disconnected")
                        cancel.client_gone = True            # xeno: the answer is cut - never one to replay
                        cancel.set()
                        return

            threading.Thread(target=watch, daemon=True, name="strata-client-watch").start()

        def _note(self, **values):
            """#332: what the monitor shows about this request (nothing when the monitor is off)."""
            if self.record is not None:
                with svc.status_lock:
                    self.record.update(values)

        def _cors(self):
            """#321: CORS headers for an API path (/v1/*) and an origin the config lists in cors_origins - nothing
            otherwise, so a browser keeps every other page away from the API, /settings, /unload and the MCP tools."""
            if not svc.cors_origins or not self.path.split("?")[0].startswith("/v1/"):
                return
            origin = (self.headers.get("Origin") or "").rstrip("/")
            if "*" in svc.cors_origins:
                allow = "*"
            elif origin and origin in svc.cors_origins:
                allow = origin
                self.send_header("Vary", "Origin")
            else:
                return
            self.send_header("Access-Control-Allow-Origin", allow)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            # the headers the preflight asks for (SDKs add their own; "*" does not cover Authorization)
            asked = self.headers.get("Access-Control-Request-Headers")
            self.send_header("Access-Control-Allow-Headers",
                             asked or "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta")
            self.send_header("Access-Control-Max-Age", "600")

        def do_OPTIONS(self):
            # a CORS preflight: no credentials come with it, so no API key; the headers only for cors_origins
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _json(self, code, obj):
            if self.record is not None:
                with svc.status_lock:
                    self.record["http_status"] = code
                    raw = json.dumps(obj, ensure_ascii=False, indent=2)
                    self.record.update(response=raw[:262144], response_truncated=len(raw) > 262144)
                    for key in ("error", "usage", "timings"):
                        if key in obj:
                            self.record[key] = obj[key]
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            """Who sent this request (self.identity) and, in team mode, from which device (#310).  Outside team mode
            (no keys file) it is as before: the api_key, when set, is the door and every holder is the owner; with no
            key the owner is a request from this PC that names this PC (mcp_admin.may_edit), anyone else a guest."""
            self.team = svc.keys.team_mode()
            given = self._given_key()
            if not self.team:
                if svc.api_key and not key_matches(given, svc.api_key):
                    self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
                    return False
                self.identity = identity_mod.OWNER if self._edit_answer()[0] else identity_mod.LOCAL_GUEST
                return True
            ident = svc.keys.match(given)                    # team mode: a key always, even with no api_key set
            if ident is None:
                self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
                return False
            self.identity = ident
            peer, h = self._peer(), self.headers
            addr = identity_mod.device_addr(peer, mcp_admin.is_loopback(peer), h.get("Cf-Connecting-Ip"),
                                            self._all_lines("X-Forwarded-For"), self._all_lines("Forwarded"))
            self.device_id, label = identity_mod.device_of(h.get("X-Strata-Device"), h.get("X-Strata-Device-Name"),
                                                           h.get("User-Agent"), addr)
            if self._loopback_local():                       # the workstation itself: approved without asking
                return True
            # every key's first use from a new device waits for the owner, the owner's own key too (spec decision 1)
            state = svc.keys.device(ident, self.device_id, label, addr, time.time())
            if state == "refused":                           # #315: the key renamed between match() and the check -
                again = svc.keys.match(given)                # its devices went with it, so its new name decides (a
                if again is not None and again.name != ident.name:     # runner took the old name's "refused" as the
                    ident = self.identity = again                       # owner's word and stopped for good)
                    state = svc.keys.device(ident, self.device_id, label, addr, time.time())
            if state == "approved":
                return True
            if state == "pending":
                code = svc.keys.device_code(ident.name, self.device_id)
                self._json(403, {"error": {"type": "device_pending", "code": code or "----", "message":
                                 f"this device waits for the owner's approval to use the key {ident.name!r}; the owner "
                                 f"sees it with the code {code}" if code else
                                 "this device waits for the owner's approval, but the keys file on the server cannot be "
                                 "read or written, so the device is not recorded: the owner must fix the keys file"}})
            else:
                self._json(403, {"error": {"type": "device_refused", "message":
                                 f"this device may not use the key {ident.name!r} (the owner refused it, or the key has "
                                 f"too many devices); the owner can clear it"}})
            return False

        def _all_lines(self, name: str) -> str | None:
            """A list header as one value: every line it came on, in order, joined by commas (as RFC 9110 reads it), so
            the last entry is the nearest proxy's even when that proxy added a line of its own (final review M2)."""
            lines = self.headers.get_all(name)
            return ", ".join(lines) if lines else None

        def _given_key(self) -> str:
            auth = self.headers.get("Authorization", "")
            return auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")

        def _still_may_run(self, arun) -> bool:
            """#310: right before a coding run starts, after the request may have waited (a model load): its key is still
            there and still an owner's, and no revoke or demotion stopped the run meanwhile.  Otherwise the run's cancel
            is set and the request is refused (401 revoked, 403 no longer an owner).  A run on My PC (#310 part 2) needs no
            owner key: its key must still match, the run not be stopped, and its runner still be the caller's and online
            (else 409 runner_offline)."""
            if arun.runner is not None:                      # #310 part 2: a run on My PC - any identity whose key still
                ident = svc.keys.match(self._given_key()) if self.team else identity_mod.OWNER     # matches, while its runner is still there
                if ident is not None and not arun.stopped \
                        and svc.runner_hub.get_for(arun.runner.id, ident) is arun.runner and arun.runner.online():
                    return True
                arun.cancel.set()
                if ident is None:
                    self._json(401, {"error": {"type": "authentication_error", "message": "the key was revoked"}})
                elif arun.stopped:
                    self._json(403, {"error": {"type": "forbidden", "message": "the run was stopped (this key was changed meanwhile)"}})
                else:
                    self._json(409, self._runner_offline(arun.runner))
                return False
            if not self.team:
                return True
            ident = svc.keys.match(self._given_key())
            if ident is not None and ident.is_owner and not arun.stopped:
                return True
            arun.cancel.set()
            if ident is None:
                self._json(401, {"error": {"type": "authentication_error", "message": "the key was revoked"}})
            else:
                self._json(403, {"error": {"type": "forbidden", "message": "The coding tools can be used only with an "
                                                                          "owner key (this key was changed meanwhile)"}})
            return False

        def _runner_offline(self, runner) -> dict:
            """#310 part 2: the 409 of a chat whose runner is offline (or gone)."""
            m = runner.info.get("machine") or "?"
            return {"error": {"type": "runner_offline", "message": f"your PC ({m}) is offline: start the runner on it and send again"}}

        def _runner_param(self, q: dict, body, online: bool = True):
            """#310 part 2, task 5: the runner a page route names - `runner` in the query (`q`: parse_qs with blank values
            kept) or in the JSON body.  None: it names none (no key, or the body's "runner" is null), the route acts on
            the workstation.  The Runner when it is the caller's own (RunnerHub.get_for: the same identity exactly, the
            owner role too) and, with `online`, online.  False after answering: 404 runner_not_found for another
            identity's id, an unknown, blank or non-string one (the same answer, so ids cannot be probed); 409
            runner_offline.  Outside team mode a runner is the owner's: the owner gate comes first."""
            if "runner" in q:
                rid = q["runner"][0]
            elif isinstance(body, dict) and body.get("runner") is not None:
                rid = body["runner"]
            else:
                return None
            if not self.team and not self._owner_only("a runner can be used"):
                return False
            r = svc.runner_hub.get_for(rid, self._acting()) if isinstance(rid, str) and rid else None
            if r is None:
                self._json(404, {"error": {"type": "runner_not_found", "message": "there is no runner with that id for this key"}})
                return False
            if online and not r.online():
                self._json(409, self._runner_offline(r))
                return False
            return r

        def _page_job(self, runner, path: str, query: dict, body) -> None:
            """#310 part 2, task 5: a page route on the caller's own PC - one `page` job (Runner.request, at most
            REQUEST_TIMEOUT_S), its answer checked by page_routes.relay (409 offline, 504 no answer in time, 502 a
            status or body the page must not get).  The notes' "apps switched on" come from the server's run config,
            for the notes' routes only."""
            on = memory_mod.settings(harness._cfg(svc))["on"] if path in page_routes.MEMORY_PATHS else []
            res = runner.request("page", {"path": path, "query": query, "body": body, "memory_on": on})
            self._json(*page_routes.relay(res, runner.info.get("machine") or "?"))

        def _mcp_runner_groups(self, rid):
            """#338: `strata_mcp_runner` of a chat whose project is on the Server PC - this browser's runner, whose MCP
            servers the chat may use beside the server's (the tools run on the user's PC: a Playwright there reaches the
            site the user develops on that PC).  Only its MCP groups (`my_<name>`, named as GET /mcp?runner= names them,
            so the page's switches match): its coding tools and skills stay out, the coding tools of a server project are
            the server's.  Offered while the runner is offline too, as it last said hello (a call then fails with "your PC
            is offline"): the tool block does not change with the runner's state, so the cached prefix stays.  False after a 404
            runner_not_found for an id that is not the caller's own runner (RunnerHub.get_for: the same key name exactly,
            never another identity's, the owner role too) - or, outside team mode, a 403 for a caller who is not the
            owner (every runner there is the owner's)."""
            if not self.team and not self._owner_only("a runner can be used"):
                return False
            runner = svc.runner_hub.get_for(rid, self._acting()) if isinstance(rid, str) and rid else None
            if runner is None:
                self._json(404, {"error": {"type": "runner_not_found", "message": "there is no runner with that id for this key"}})
                return False
            return {n: g for n, g in runner.servers().items() if g.group_kind == "mcp"}

        @staticmethod
        def _my_mcp_offered(req: dict, groups: dict) -> str | bool:
            """#339: the name of the user's PC when this request offers MCP groups of it (`my_<name>`: the web app's MCP opt-in, not
            switched off for this chat), else False - for the rules' "Where things run".  From the groups the runner last said hello
            with, which #338 keeps offered while it is offline, so the rules do not change between the requests of one chat."""
            if req.get("strata_mcp") is not True:
                return False
            off = req.get("strata_mcp_off")
            off = {x for x in off if isinstance(x, str)} if isinstance(off, list) else set()
            mcp = [g for n, g in (groups or {}).items() if getattr(g, "group_kind", "") == "mcp" and n not in off and getattr(g, "tools", None)]
            if not mcp:
                return False
            r = getattr(mcp[0], "runner", None)
            return (r.info.get("machine") if r is not None else None) or True

        @staticmethod
        def _names_runner(raw: bytes) -> dict | None:
            """#338: a POST /mcp/config or /import body (already read, at most 1 MB) that names a runner (`"runner"` not
            null): that object; None for any other body, which takes the route's own way."""
            try:
                body = json.loads(raw) if raw else None
            except ValueError:
                return None
            return body if isinstance(body, dict) and body.get("runner") is not None else None

        def _runner_mcp(self, q: dict, body, kind: str, payload: dict) -> None:
            """#338: My PC's MCP servers (GET / POST /mcp/config, GET / POST /import with `runner`): one MCP job on
            the runner (serve/runner.py Executor._mcp), its answer relayed with the statuses those routes give.  An MCP
            server is a program that runs on that PC, so only the runner's own identity (RunnerHub.get_for: the same
            key name exactly - never the owner for a guest's PC: 404 as for an unknown id) may read how its servers
            start or change them.  The device check (X-Strata-Device must be the runner's: 403 runner_device) only stops
            a click in the wrong browser - the header can be copied by anyone who holds the key.  The real guard is on
            the PC: a server added, changed or switched on from here waits for the user's approval in Strata Desktop
            (serve/runner.py Executor._write_servers).  Then 409 runner_offline, 409 runner_outdated (an older Strata
            Desktop)."""
            runner = self._runner_param(q, body, online=False)
            if runner is False or runner is None:
                return
            if self._runner_device() != runner.device:
                return self._json(403, {"error": {"type": "runner_device", "message": (
                    f"the MCP servers of {runner.info.get('machine') or 'your PC'} start programs there, so they can be "
                    "changed only from that PC: open the web app in Strata Desktop on it")}})
            if not runner.online():
                return self._json(409, self._runner_offline(runner))
            res = runner.request(kind, payload)
            self._json(*page_routes.relay(res, runner.info.get("machine") or "?", page_routes.ADMIN_STATUSES))

        def _page_here(self, path: str, q: dict, body) -> tuple[int, dict]:
            """A page route on the workstation itself (page_routes.answer): its home and run config for the notes, the
            server's checkpoints for the way back."""
            home, on = "", []
            if path in page_routes.MEMORY_PATHS:
                imp = getattr(svc, "importer", None)
                home = str(imp.home) if imp is not None else str(harness.home_dir())
                on = memory_mod.settings(harness._cfg(svc))["on"]
            return page_routes.answer(path, q, body, home=home, memory_on=on, checkpoints=svc.checkpoints)

        def _peer(self) -> str:
            """The client's address as the socket has it (#310 tests put another one in svc.peer_override)."""
            return svc.peer_override or self.client_address[0]

        def _loopback_local(self) -> bool:
            """#310: a request from this PC that names this PC and that no proxy relayed - the only kind whose device is
            approved without asking.  A tunnel (cloudflared, tailscale serve) connects from 127.0.0.1 too, but names
            its public Host or adds a forwarding header."""
            return bool(svc.trust_loopback) and mcp_admin.is_loopback(self._peer()) \
                and mcp_admin.host_is_loopback(self.headers.get("Host", "")) \
                and not any(name in self.headers for name in identity_mod.FORWARDING)

        def _edit_answer(self) -> tuple[bool, str]:
            """(may this request change what runs on this PC, why not).  In team mode: an owner key (#310).  Outside
            it, as before: mcp_admin.may_edit."""
            if self.team:
                ok = self.identity is not None and self.identity.is_owner
                return ok, "" if ok else "only with an owner key"
            return mcp_admin.may_edit(bool(svc.api_key), self._peer(), self.headers.get("Host", ""))

        def _owner_only(self, what: str) -> bool:
            """True for an owner.  Otherwise a 403: in team mode `forbidden` ("<what> only with an owner key", #310);
            outside it the answer the route gave before, word for word."""
            ok, why = self._edit_answer()
            if ok:
                return True
            if self.team:
                self._json(403, {"error": {"type": "forbidden", "message": f"{what} only with an owner key"}})
            else:
                self._json(403, {"error": {"message": why}})
            return False

        def _mcp_allowed(self) -> set[str] | None:
            """#310 ruling R11: the server MCP servers this request may use - None for all of them (the owner, and every
            caller outside team mode, as before: the run config is not read).  A team guest gets the servers the owner
            marked for guests (the run config's "guest_mcp", read on each request like the other tool settings) and the
            skills, never the coding tools: an MCP server runs on this PC with its owner's rights."""
            if not self.team or (self.identity is not None and self.identity.is_owner):
                return None
            return mcp_admin.guest_allowed(harness._cfg(svc), svc.mcp)

        def _mcp_view(self) -> dict:
            """GET /mcp/config's answer for this caller (#79): a team guest's lists only her servers (R11), the team owner's
            says which servers guests may use."""
            may = self._edit_answer()
            marks = mcp_admin.guest_servers(harness._cfg(svc)) if self.team and may[0] else None
            return mcp_admin.view(svc, self._peer(), self.headers.get("Host", ""), may=may, allowed=self._mcp_allowed(),
                                  guests=marks)

        def _runs_gate(self, what: str) -> bool:
            """#310: the gate of a run's routes and of the answers to its prompts.  In team mode every key passes it: a
            run and a prompt belong to the identity that launched it, and the run store and the broker answer 404 for
            another's id, as for one that is not there, so ids cannot be probed.  Outside team mode `_owner_only`, as
            before."""
            return True if self.team else self._owner_only(what)

        def _slot_who(self) -> dict:
            """#321: who sends this request, for the cache-slot rule "identity" and STRATA_SLOT_TRACE: the key's name
            (never the key; "owner" outside team mode), the device id (X-Strata-Device, else "addr:<address>") and the
            address (the socket peer, or the client a proxy on this PC named: identity.device_addr)."""
            peer, h = self._peer(), self.headers
            addr = identity_mod.device_addr(peer, mcp_admin.is_loopback(peer), h.get("Cf-Connecting-Ip"),
                                            self._all_lines("X-Forwarded-For"), self._all_lines("Forwarded"))
            device, _ = identity_mod.device_of(h.get("X-Strata-Device"), None, None, addr)
            return {"key": self._acting().name, "device": device, "addr": addr}

        def _acting(self) -> identity_mod.Identity:
            """Who a run route acts as: the request's identity in team mode; outside it OWNER, whose every run is."""
            return self.identity if self.team else identity_mod.OWNER

        def _ui_prefix(self) -> bool:
            """xeno UI S1: /classic/... is the classic web app under a prefix (its relative URLs then land on the
            same routes), /next/... the new one (serve/ui/dist). True when this call answered."""
            p = self.path.split("?")[0]
            for name in ("classic", "next"):
                if p == "/" + name:                          # relative, so a path-prefixed proxy still works
                    self.send_response(301)
                    self.send_header("Location", name + "/")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return True
            if p.startswith("/classic/"):
                self.path = self.path[len("/classic"):]
                return False
            rel = None                                       # the path inside dist/, when this request is for the new app
            if p.startswith("/next/"):
                rel = p[len("/next/"):]
            elif getattr(svc, "ui", "next") == "next":       # the new app at / unless the config says "ui": "classic" (its assets at /assets/)
                if p == "/":
                    rel = ""
                elif p.startswith("/assets/"):
                    rel = p[1:]
            if rel is None:
                return False
            types = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
                     ".woff2": "font/woff2", ".svg": "image/svg+xml", ".png": "image/png", ".webp": "image/webp"}
            f, ctype, cache = None, "text/html; charset=utf-8", "no-cache"
            if rel == "":
                f = ROOT / "serve" / "ui" / "dist" / "index.html"
            elif rel.startswith("assets/") and "/" not in rel[7:] and "\\" not in rel and os.path.splitext(rel)[1] in types:
                f, ctype, cache = ROOT / "serve" / "ui" / "dist" / "assets" / rel[7:], types[os.path.splitext(rel)[1]], \
                    "public, max-age=31536000, immutable"       # hashed names never change
            if f is None or not f.is_file():
                self._json(404, {"error": {"message": "not found"}})
                return True
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            if f.suffix == ".html":                          # the app is never shown inside another page (clickjacking the buttons)
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True

        def do_GET(self):
            target = self.path                               # as it came: _ui_prefix may take /classic off
            if self._ui_prefix():
                return
            path = route_path(target)
            if path.startswith("/fonts/"):
                # the web app's font (Outfit, OFL: serve/web/fonts); the page falls back to the system font
                name = path[len("/fonts/"):]
                f = ROOT / "serve" / "web" / "fonts" / name
                if "/" in name or "\\" in name or not name.endswith(".woff2") or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "font/woff2")
                self.send_header("Cache-Control", "max-age=86400")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/web/"):
                # the web app's own files (serve/web): styles, script, icon sprite - same origin, no CDN
                name = path[len("/web/"):]
                types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                         ".svg": "image/svg+xml"}
                f = ROOT / "serve" / "web" / name
                ext = os.path.splitext(name)[1]
                if "/" in name or "\\" in name or ext not in types or not f.is_file():
                    self._json(404, {"error": {"message": "not found"}})
                    return
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", types[ext])
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/metrics/requests" or path.startswith("/metrics/requests/"):
                if self._authorized():                     # the history on disk (xeno UI S3): a page, or one request
                    q = parse_qs(urlsplit(self.path).query)
                    if path == "/metrics/requests":
                        try:
                            page, size = int(q.get("page", ["0"])[0]), int(q.get("size", ["50"])[0])
                        except ValueError:
                            return self._json(400, {"error": {"message": "page and size are numbers"}})
                        viewer = self._acting()              # #310: another's rows show no prompt or answer
                        got = svc.hstore.page(page, size)
                        got["items"] = [identity_mod.visible_record(r, viewer) for r in got["items"]]
                        return self._json(200, got)
                    rid = path[len("/metrics/requests/"):]
                    summary = svc.hstore.summary(rid)
                    if summary is None or not svc.may_read(summary, self._acting()):   # #310: another's, as one not there
                        return self._json(404, {"error": {"message": "no such request"}})
                    detail = svc.hstore.detail(rid)
                    return self._json(200, {"summary": summary, "detail": detail,
                                            "detail_state": "kept" if detail is not None else "deleted"})
                return
            if path == "/metrics/prometheus":
                if self._authorized():
                    body = svc.prometheus().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                return
            if path == "/metrics":
                from serve import prometheus
                if prometheus.wants_prometheus(self.headers.get("Accept", ""), urlsplit(self.path).query):
                    if self._authorized():   # a scraper: the text format, vLLM's names (serve/prometheus.py)
                        body = prometheus.render(svc.metrics(), svc.latencies.snapshot()).encode()
                        self.send_response(200)
                        self.send_header("Content-Type", prometheus.CONTENT_TYPE)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                    return
                if self._authorized():
                    # the last 12 requests; `?requests=all` every one kept (the Monitor's "Show all", issue #35)
                    self._json(200, svc.metrics(all_requests="requests=all" in self.path, viewer=self._acting()))
                return
            if path == "/api/requests" and svc.api_monitor:
                if self._authorized():
                    request_id = parse_qs(urlsplit(self.path).query).get("id", [None])[0]
                    records = svc.request_records(request_id, viewer=self._acting())
                    self._json(404 if records is None else 200,
                               {"error": {"message": "request no longer retained"}} if records is None else
                               records if request_id else {"requests": records, "retention": 100, "persistent": False,
                                                          "loaded": svc.loaded(), "auto_load": hasattr(svc.engine, "restart")})
                return
            if path == "/settings":
                if self._authorized():
                    self._json(200, {"shared": bool(svc.shared), "defaults": svc.shared})
                return
            if path == "/keys":                              # #310: the named keys and what was done to them, for the owner's Settings
                if self._authorized() and self._owner_only("keys can be managed"):
                    self._json(200, {"keys": svc.keys.view(), "log": svc.keys.log(100), "retired": svc.keys.retired(),
                                     "team": self.team,
                                     "keys_file": os.path.basename(svc.keys.path) if svc.keys.path else None})
                return
            if path == "/devices/pending":                   # #310: the devices waiting for approval that this caller may decide
                if self._authorized():                       # (a caller on a device that waits never gets here: the gate answers 403)
                    rows = svc.keys.pending() if self.team else []
                    self._json(200, {"devices": [r for r in rows if svc.keys.may_approve(self.identity, r["key"], self.device_id or None)]})
                return
            if path == "/config":                            # #564: the Settings view's keys of the run config
                if self._authorized():
                    self._config_get()
                return
            if path in ("/runner/next", "/runners"):         # #310 part 2: a runner's long poll, and who has one
                if self._authorized() and self._runner_gate():
                    if path == "/runners":
                        return self._json(200, {"runners": svc.runner_hub.view(self._acting())})
                    self._runner_next()
                return
            if path == "/mcp":
                # the MCP servers, their state and tools (the web app's switch and Monitor card)
                if self._authorized():                       # #310 R11: a team guest sees only the servers she may use
                    q = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                    if "runner" in q:                        # #310 part 2: the caller's own PC's groups, from its hello (no job)
                        r = self._runner_param(q, None, online=False)
                        if r is not False:
                            st = McpHub({}, builtins=r.servers()).status()   # its own routes: names as my_<x>__<tool>
                            self._json(200, {"servers": st["servers"], "tools": st["tools"], "skills": [dict(s) for s in r.skills],
                                             "machine": r.info.get("machine"), "online": r.online()})
                        return
                    self._json(200, mcp_admin.status_for(svc.mcp.status() if svc.mcp else {"servers": [], "tools": 0},
                                                         self._mcp_allowed()))
                return
            if path == "/agent":
                # the chat's coding tools: whether they are on, which shell commands run in, and whether this caller may use them (only from this PC, or with the key)
                if self._authorized():
                    ok, why = self._edit_answer()
                    q = parse_qs(urlsplit(self.path).query)
                    if "runner" in q:                        # #310 part 2: the coding tools of the caller's own PC, from its hello
                        if not self.team and not self._owner_only("the coding tools of a runner can be read"):
                            return
                        r = svc.runner_hub.get_for(q["runner"][0], self._acting())
                        if r is None:
                            return self._json(404, {"error": {"type": "runner_not_found", "message": "there is no runner with that id for this key"}})
                        online = r.online()
                        coding = r.servers().get("my_agent")             # the first coding group, without the tools the server runs
                        return self._json(200, {"available": True, "allowed": online, "reason": "" if online else "your PC is offline",
                                                "shell": r.info.get("shell"), "tools": [t["name"] for t in coding.tools] if coding else [],
                                                "machine": r.info.get("machine"), "online": online, "yolo": True})
                    sh = getattr(svc.agent, "shell", None)
                    self._json(200, {"available": svc.agent is not None, "allowed": ok, "reason": "" if ok else why, "shell": shell_mod.describe(sh) if sh else None,
                                     "tools": [t["name"] for t in svc.agent.tools] if svc.agent else [],
                                     "yolo": not (self.team and not (self.identity is not None and self.identity.is_owner))})     # #329: a guest's YOLO only on My PC
                return
            if path == "/agent/hooks":
                # the user's hooks (serve/hooks.py), read only: they are written in the run config; only who may use the coding tools sees the commands
                if self._authorized():
                    if not self._owner_only("the hooks can be read"):
                        return
                    self._json(200, {**hooks_mod.view(harness._cfg(svc)), "config_file": os.path.basename(svc.config_path) if svc.config_path else None,
                                     "shell": bool(getattr(svc.agent, "shell", None)), "editable": bool(svc.config_path)})
                return
            if path == "/agent/helpers":
                # sub-agents (serve/subagent.py): off by default; only who may use the coding tools sees the setting
                if self._authorized():
                    if not self._owner_only("the sub-agent settings can be read"):
                        return
                    self._json(200, app_settings.helpers_view(svc))
                return
            if path == "/agent/run":
                # a run that went on while the page was away (serve/runs.py): its events from number `from` on, as they come; only for who may use the coding tools
                if self._authorized():
                    if not self._runs_gate("a run can be read"):
                        return
                    origin = self.headers.get("Origin")
                    if origin and origin.split("://", 1)[-1] != self.headers.get("Host", ""):
                        return self._json(403, {"error": {"message": "a run can be read only from Strata's own page"}})
                    q = parse_qs(urlsplit(self.path).query)
                    run = svc.runs.get_for(q.get("id", [""])[0], self._acting())          # #310: another identity's run is not there
                    if run is None:
                        return self._json(404, {"error": {"message": NO_RUN}})
                    try:
                        start = int(q.get("from", ["0"])[0])
                    except ValueError:
                        return self._json(400, {"error": {"message": "from is a number"}})
                    self._sse(now=True)
                    self._follow(run, start)
                return
            if path == "/agent/web":
                # web access (serve/web.py): off by default; only who may use the coding tools sees the settings
                if self._authorized():
                    if not self._owner_only("the web access settings can be read"):
                        return
                    self._json(200, app_settings.web_view(svc))
                return
            if path in page_routes.GET_PATHS:
                # the folders of this PC (the project picker), a folder's Git state (the Chat's right panel), `@file`, and
                # the notes the chat reads for a project (serve/page_routes.py): on the workstation for who may use the
                # coding tools; with `runner=<id>` on the caller's own PC, through its runner (#310 part 2)
                if self._authorized():
                    query = urlsplit(self.path).query
                    runner = self._runner_param(parse_qs(query, keep_blank_values=True), None)
                    if runner is False:
                        return
                    q = parse_qs(query)
                    if runner is not None:
                        return self._page_job(runner, path, {k: v for k, v in q.items() if k != "runner"}, None)
                    if not self._owner_only(PAGE_GATES[path]):
                        return
                    self._json(*self._page_here(path, q, None))
                return
            if path == "/mcp/config":
                # the servers as set up (secrets masked), their state, the limits, and whether this caller may change them (#79)
                if self._authorized():
                    q = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                    if "runner" in q:                        # #338: the caller's own PC's, from its runner.json
                        return self._runner_mcp(q, None, "mcp.config", {})
                    self._json(200, self._mcp_view())
                return
            if path == "/import":
                # the skills and the MCP servers of the other coding apps on this PC, with what may be changed (#94)
                if self._authorized():
                    q = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                    if "runner" in q:                        # #338: the other apps' MCP servers on the caller's own PC
                        return self._runner_mcp(q, None, "mcp.import.list", {"rescan": q.get("rescan", [""])[0] == "1"})
                    may = self._edit_answer()                    # #310: in team mode only an owner makes the server rescan this PC's apps
                    self._json(200, harness.view(svc, self._peer(), self.headers.get("Host", ""),
                                                 rescan="rescan=1" in self.path and (not self.team or may[0]), may=may))
                return
            if path == "" or (path == "/api-monitor" and svc.api_monitor):
                body = (ROOT / "serve" / "web" / ("monitor.html" if path else "index.html")).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path in ("/health", "/api/health"):
                proc = getattr(svc.engine, "proc", None)
                # an engine unloaded on purpose (#208) is not a dead one; `not svc.loaded()` is true after a crash too
                alive = proc is None or proc.poll() is None or bool(getattr(svc.engine, "unloaded", False))
                self._json(200 if alive else 503, {"status": "ok" if alive else "engine_exited",
                                     "max_context": svc.reported_ctx(), "model": svc.model,
                                     "images": svc.vision is not None, "api_key": svc.key_required(),
                                     "loops_stopped": svc.status["loops_stopped"],
                                     "thinking_cycles_closed": svc.status["thinking_cycles_closed"],   # #199
                                     "loaded": svc.loaded(),
                                     "degraded": bool(getattr(svc.engine, "degraded", False)), "service": "strata"})
            elif path == "/status":
                if not self._authorized():                  # #212: it shows the end of the last answer
                    return
                with svc.status_lock:
                    s = dict(svc.status)
                now = time.time()
                if s.get("busy"):
                    s["elapsed_s"] = round(now - s["started"], 1)
                    if s.get("first_token"):
                        s["tokens_per_s"] = round(svc._tok_s(), 1)
                        s["tokens_per_s_mean"] = round(svc._tok_s_mean(), 1)
                for k in ("started", "first_token"):
                    s.pop(k, None)
                if "tail" in s and not svc.may_read(s, self._acting()):   # #310: the answer is its sender's and the owner's
                    s["tail"] = ""
                self._json(200, s)
            elif path in ("/v1/models", "/models"):
                if self._authorized():
                    loaded = svc.loaded()
                    model = {"id": svc.model, "object": "model", "status": {"value": "loaded"},
                             "meta": {"n_ctx": svc.reported_ctx()},
                             "architecture": {"input_modalities": ["text", "image"] if svc.vision is not None else ["text"],
                                              "output_modalities": ["text"]}}
                    if not loaded and (svc.idle_unload_s or getattr(svc.engine, "unloaded", False)):
                        model["status"] = {"value": "unloaded"}   # like llama-server's router: listed, loads on use
                        loaded = True
                    if svc.aliases:                       # #297: the aliases, and each one listed under its own id
                        model["aliases"] = list(svc.aliases)
                    data = [model, *({**model, "id": x, "alias_of": svc.model} for x in svc.aliases)]
                    self._json(200, {"object": "list", "data": data if loaded else []})
            elif path == "/props":
                if self._authorized():
                    self._props()
            elif path == "/slots":
                if self._authorized():
                    loaded = not hasattr(svc.engine, "alive") or svc.engine.alive()
                    with svc.status_lock:
                        busy = bool(svc.status.get("busy"))
                        # llama.cpp's name for what the slot's context holds: the running request's prompt size,
                        # kept after it ends (the conversation cache carries it on).  A front-end's context meter
                        # divides it by n_ctx; without the key it reads 0 % against Strata forever (#562).
                        in_use = int(svc.status.get("prompt_tokens") or 0)
                    if getattr(svc.engine, "batch", 0) and hasattr(svc.engine, "slots_view"):
                        # --batch: one entry per slot (the same view /metrics has), not one for the whole engine
                        n_ctx = svc.reported_ctx()
                        slots = [{"id": s["slot"], "n_ctx": n_ctx, "is_processing": s["state"] != "idle",
                                  "n_prompt_tokens": int(s.get("prompt_tokens") or s.get("held_tokens") or 0)}
                                 for s in svc.engine.slots_view()]
                    else:
                        slots = [{"id": 0, "n_ctx": svc.reported_ctx(), "is_processing": busy,
                                  "n_prompt_tokens": in_use}]
                    self._json(200, slots if loaded else [])
            elif path == "/v1/status":
                if self._authorized():
                    self._json(200, svc.v1_status())
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            with timeline.span("http request"):
                try:
                    self._do_post()
                except BadBody as e:                          # #893: a malformed or oversized chunked body
                    self._json(e.status, {"error": {"type": "invalid_request_error", "message": str(e)}})

        def _do_post(self):
            target = self.path
            if self.path.startswith("/classic/"):             # the classic app under its prefix (xeno UI S1)
                self.path = self.path[len("/classic"):]
            if not self._authorized():
                return
            path = route_path(target)                    # issue #55: Claude Code posts /v1/messages?beta=true
            if path.startswith("/v1/") and self._foreign_page():
                return
            # #310: what changes the shared engine, every client's settings or what is recorded is an owner key's in
            # team mode; outside it these stay open to every key holder, as before
            if self.team and (path in OWNER_POSTS or path.startswith("/slots/")) and \
                    not self._owner_only(OWNER_POSTS.get(path, "conversations can be saved or restored")):
                return
            if path == "/settings":
                self._settings()
                return
            if path in ("/runner/hello", "/runner/reply"):   # #310 part 2: a runner says hello, or answers a job
                if self._runner_gate():
                    self._runner_post(path)
                return
            if path in ("/keys", "/devices"):                # #310: named keys and device approval, managed live
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 10_000 else b""
                if not self._own_page("keys and devices can be managed"):
                    return
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                if not isinstance(body, dict):
                    return self._json(400, {"error": {"message": "send a JSON object"}})
                return self._keys_post(body) if path == "/keys" else self._devices_post(body)
            if path == "/mcp/config":                        # set up the MCP servers: a write that decides which programs Strata starts (#79)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 1_000_000 else b""
                if not self._own_page("MCP servers can be changed"):
                    return
                mine = self._names_runner(raw)                   # #338: the servers of the caller's own PC (runner.json)
                if mine is not None:
                    return self._runner_mcp({}, mine, "mcp.config.set", {"body": {k: v for k, v in mine.items() if k != "runner"}})
                if not self._owner_only("MCP servers can be changed"):
                    return
                if not svc.config_path:
                    return self._json(409, {"error": {"message": "this server was started without a run config file (--config), so there is nowhere to save the servers"}})
                if not 0 < n <= 1_000_000:
                    return self._json(413, {"error": {"message": "send a body of up to 1 MB"}})
                try:
                    body = json.loads(raw)
                except ValueError:
                    return self._json(400, {"error": {"message": "the body is not JSON", "fields": []}})
                code, out = mcp_admin.apply(svc, body)
                return self._json(code, out if code != 200 else self._mcp_view())
            if path in page_routes.POST_PATHS:               # the way back for the files the tools changed (serve/checkpoints.py, serve/page_routes.py)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 10_000 else b""
                if not self._own_page("files can be put back"):
                    return
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                runner = self._runner_param({}, body)          # #310 part 2: the body's "runner" - the files on the caller's own PC
                if runner is False:
                    return
                if runner is not None:
                    return self._page_job(runner, path, {}, {k: v for k, v in body.items() if k != "runner"})
                if not self._owner_only(PAGE_GATES[path]):
                    return
                return self._json(*self._page_here(path, {}, body))
            if path == "/agent/steer":                       # what the user sends while the agent works: the agent reads it at its next step (serve/runs.py, run_with_mcp)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 100_000 else b""
                if not self._own_page("A message can be sent to a run"):
                    return
                if not self._runs_gate("A message can be sent to a run"):
                    return
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                if not isinstance(body, dict) or not isinstance(body.get("id"), str) or not isinstance(body.get("text"), str) or not 0 < len(body["text"].strip()) <= 20_000:
                    return self._json(400, {"error": {"message": "send {\"id\": the run's id, \"text\": the message (1 to 20,000 characters)}"}})
                if self.team and svc.runs.get_for(body["id"], self.identity) is None:
                    return self._json(404, {"error": {"message": NO_RUN}})          # #310: another identity's run is not there
                return self._json(200, {"ok": True, "found": svc.runs.steer(body["id"], body["text"].strip(), self._acting())})
            if path == "/agent/cancel":                      # the user's Stop for an answer that runs on its own (serve/runs.py)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 10_000 else b""
                if not self._own_page("A run can be stopped"):
                    return
                if not self._runs_gate("A run can be stopped"):
                    return
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                if not isinstance(body, dict) or not isinstance(body.get("id"), str):
                    return self._json(400, {"error": {"message": "send {\"id\": the run's id}"}})
                if self.team and svc.runs.get_for(body["id"], self.identity) is None:
                    return self._json(404, {"error": {"message": NO_RUN}})          # #310: another identity's run is not there
                return self._json(200, {"ok": True, "found": svc.runs.cancel(body["id"], self._acting())})
            if path == "/agent/question":                    # the page's answer to a question the model asked with AskUserQuestion (serve/agent_run.py)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 50_000 else b""
                if not self._own_page("A question can be answered"):
                    return
                if not self._runs_gate("A question can be answered"):
                    return
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                ans = body.get("answers") if isinstance(body, dict) else None
                if not isinstance(body, dict) or not isinstance(body.get("id"), str) or not (ans is None or (isinstance(ans, dict) and len(ans) <= 4 and all(
                        isinstance(k, str) and len(k) <= 400 and isinstance(v, list) and 0 < len(v) <= 8 and all(isinstance(x, str) and len(x) <= 500 for x in v) for k, v in ans.items()))):
                    return self._json(400, {"error": {"message": "send {\"id\": the question's id, \"answers\": {question: [choices]} or null to skip}"}})
                if not svc.broker.respond(body["id"], ans, self._acting()):          # #310: only the run's holder answers it
                    return self._json(404, {"error": {"message": "no question with that id is waiting (it was answered, or the request ended)"}})
                return self._json(200, {"ok": True})
            if path == "/agent/permission":                  # the page's answer to a question of the coding tools (serve/agent_run.py)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 10_000 else b""
                if not self._own_page("A question can be answered"):
                    return
                if not self._runs_gate("A question can be answered"):
                    return
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                if not isinstance(body, dict) or not isinstance(body.get("id"), str) or body.get("decision") not in agent_run.ANSWERS:
                    return self._json(400, {"error": {"message": "send {\"id\": the question's id, \"decision\": \"allow\" | \"allow_chat\" | \"deny\"}"}})
                if not svc.broker.answer(body["id"], body["decision"], self._acting()):     # #310: only the run's holder answers it
                    return self._json(404, {"error": {"message": "no question with that id is waiting (it was answered, or the request ended)"}})
                return self._json(200, {"ok": True})
            if path == "/agent/helpers":                     # sub-agents on or off (serve/subagent.py)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 10_000 else b""
                if not self._own_page("sub-agents can be switched"):
                    return
                if not self._owner_only("sub-agents can be switched"):
                    return
                if not svc.config_path:
                    return self._json(409, {"error": {"message": harness.NO_FILE}})
                if not 0 < n <= 10_000:
                    return self._json(413, {"error": {"message": "send a body of up to 10 KB"}})
                try:
                    body = json.loads(raw)
                except ValueError:
                    return self._json(400, {"error": {"message": "the body is not JSON"}})
                return self._json(*app_settings.save_helpers(svc, body))       # the same writer as the chat's StrataSettingsChange (#339)
            if path == "/agent/web":                         # web access on or off, and the search provider (serve/web.py)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 100_000 else b""
                if not self._own_page("web access can be changed"):
                    return
                if not self._owner_only("web access can be changed"):
                    return
                if not svc.config_path:
                    return self._json(409, {"error": {"message": harness.NO_FILE}})
                if not 0 < n <= 100_000:
                    return self._json(413, {"error": {"message": "send a body of up to 100 KB"}})
                try:
                    body = json.loads(raw)
                except ValueError:
                    return self._json(400, {"error": {"message": "the body is not JSON"}})
                return self._json(*app_settings.save_web(svc, body))           # the same writer as the chat's StrataSettingsChange (#339)
            if path == "/agent/hooks":                       # which hooks are switched off (the hooks themselves are written in the run config only)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 100_000 else b""
                if not self._own_page("the hooks can be switched"):
                    return
                if not self._owner_only("the hooks can be switched"):
                    return
                if not svc.config_path:
                    return self._json(409, {"error": {"message": harness.NO_FILE}})
                if not 0 < n <= 100_000:
                    return self._json(413, {"error": {"message": "send a body of up to 100 KB"}})
                try:
                    body = json.loads(raw)
                except ValueError:
                    return self._json(400, {"error": {"message": "the body is not JSON"}})
                off, bad = hooks_mod.check_off(body.get("off") if isinstance(body, dict) and set(body) == {"off"} else None)
                if bad:
                    return self._json(400, {"error": {"message": bad}})
                known = {h["id"] for h in hooks_mod.view(harness._cfg(svc))["hooks"]}
                with mcp_admin._lock:
                    try:
                        mcp_admin.update_config(svc.config_path, lambda cfg: cfg.__setitem__("hooks_off", [x for x in off if x in known]))
                    except (OSError, ValueError, TypeError) as e:
                        return self._json(500, {"error": {"message": f"it could not be saved: {e}"}})
                return self._json(200, {**hooks_mod.view(harness._cfg(svc)), "config_file": os.path.basename(svc.config_path), "shell": bool(getattr(svc.agent, "shell", None)), "editable": True})
            if path == "/import":                            # the skill switches, importing one MCP server, a rescan: a write like /mcp/config (#94)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 1_000_000 else b""
                if not self._own_page("the import can be changed"):
                    return
                mine = self._names_runner(raw)                   # #338: an MCP server of another app on the caller's own PC
                if mine is not None:
                    return self._runner_mcp({}, mine, "mcp.import.add", {"body": {k: v for k, v in mine.items() if k != "runner"}})
                if not self._owner_only("the import can be changed"):
                    return
                if not svc.config_path:
                    return self._json(409, {"error": {"message": harness.NO_FILE}})
                if not 0 < n <= 1_000_000:
                    return self._json(413, {"error": {"message": "send a body of up to 1 MB"}})
                try:
                    body = json.loads(raw)
                except ValueError:
                    return self._json(400, {"error": {"message": "the body is not JSON", "fields": []}})
                code, out = harness.apply(svc, body)
                return self._json(code, out if code != 200 else harness.view(svc, self._peer(), self.headers.get("Host", ""), may=self._edit_answer()))
            if path == "/metrics/keep":                      # keep the full prompt of the next N requests (0: off)
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if not self._own_page("prompts can be kept"):      # xeno #71: a "simple" cross-site POST must not switch this on
                    return
                try:
                    n = json.loads(raw or b"{}").get("next")
                except ValueError:
                    n = None
                if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= 50:
                    return self._json(400, {"error": {"type": "invalid_request_error", "message": "next is a number from 0 to 50"}})
                with svc.keep_lock:
                    svc.keep_prompts = n
                return self._json(200, {"keep_prompts_left": n})
            if path == "/config":
                self._config_post()
                return
            if path in ("/unload", "/load") and not self._control_body():
                return
            if path in ("/unload", "/load") and self._foreign_origin():   # xeno: a foreign page is 403 whatever it sends
                self._json(403, {"error": {"message": "only from Strata's own page"}})
                return
            # JSON from Strata's own page only, as /settings: else a plain form POST from any site unloads the model
            if path in ("/unload", "/load") and not self._own_page("the model can be loaded or unloaded"):
                return
            # the same for the slot files: else any site could overwrite a saved conversation or replace the live one
            if path.startswith("/slots/") and not self._own_page("conversations can be saved or restored"):
                return
            if path == "/unload":                            # give the GPU back now (between requests)
                try:
                    r = svc.unload()
                except EngineStuck as e:
                    self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                    return
                self._json(409 if r == "busy" else 200, {"status": r})
                return
            if path == "/load":                              # load now, e.g. ahead of a request
                try:
                    svc.load(who=self._acting().name)
                    self._json(200, {"status": "loaded"})
                except GpuBusy as e:
                    self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                except EngineDied as e:                      # #56 review: while it restarts, a 529, not a dropped connection
                    self._json(529, {"type": "error", "error": {"type": "overloaded_error", "message": str(e)}})
                return
            try:
                req = json.loads(self._body() or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("send a JSON object")
                if path.startswith("/v1/responses/"):        # retrieve/delete/cancel/compact: nothing is stored
                    self._json(404, responses_error_body("this server keeps no responses (stateless); send the "
                                                         "whole conversation to POST /v1/responses", code="not_found"))
                    return
                if path in ("/v1/load", "/v1/unload"):
                    if not self._own_page("the model can be loaded or unloaded"):
                        return
                    if req.get("model") not in (None, svc.model):
                        self._json(404, {"error": {"message": "model not found"}})
                        return
                    if path == "/v1/unload":
                        result = svc.unload()
                        if result == "busy":
                            raise ModelBusy("a request is running or queued")
                    else:
                        if not svc.fifo.acquire(blocking=False):
                            raise ModelBusy("a request is running or queued")
                        try:
                            with svc.status_lock:
                                if svc.status.get("busy") or svc.status.get("queued"):
                                    raise ModelBusy("a request is running or queued")
                            svc.ensure_loaded()
                            svc.last_request_at = time.time()
                        finally:
                            svc.fifo.release()
                        result = "loaded"
                    self._json(200, {"status": result, **svc.v1_status()})
                    return
                if path == "/v1/vram":                       # #533: give VRAM back to other programs, or take it
                    if not self._own_page("the VRAM reserve can be changed"):
                        return
                    r = req.get("reserve_mib")
                    if r is not None and (isinstance(r, bool) or not isinstance(r, int) or r < 0):
                        raise ValueError("reserve_mib: a whole number of MiB (0 or more), or null for the start's")
                    try:
                        self._json(200, svc.vram(r))
                    except EngineDied as e:
                        self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                    return
                if path in ("/v1/chat/completions", "/v1/messages", "/v1/responses"):
                    slot = cache_slot(req)  # validate before opening an SSE response (and before loading)
                    if slot and int(getattr(svc.engine, "info", {}).get("cache_slots", 4)) == 1:
                        raise ValueError("cache slots require a single-GPU session")
                    # unloaded: _openai / _anthropic load first (or 503 while the GPU is busy)
                    self.record = svc.begin_request(path, req, who=self._acting().name)
                if path == "/v1/responses":
                    self._responses(req)
                elif path.startswith("/slots/"):
                    query = parse_qs(self.path.partition("?")[2])
                    try:
                        code, body = svc.slot_action(path[len("/slots/"):], (query.get("action") or [""])[0],
                                                     req.get("filename"), who=self._acting().name)
                    except EngineDied as e:                  # EngineSilent included: the engine was ended
                        code, body = 500, {"error": {"code": 500, "message": str(e) + "; the next request starts "
                                                     "the engine again", "type": "server_error"}}
                    self._json(code, body)
                elif path == "/v1/chat/completions":
                    self._openai(req)
                elif path == "/v1/messages":
                    self._anthropic(req)
                elif path == "/v1/messages/count_tokens":
                    self._count_tokens(req)
                else:
                    self._json(404, {"error": {"message": "not found"}})
            except ValueError as e:
                print(f"[strata] 400 invalid request: {e}", flush=True)
                if path == "/v1/responses":
                    self._json(400, responses_error_body(str(e)))
                else:
                    self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
            except ModelBusy as e:
                self._json(409, {"error": {"type": "model_busy", "message": str(e)}})
            except StructuredOutputError as e:
                self._json(502, {"error": {"type": "structured_output_failed", "code": "structured_output_failed",
                                          "message": str(e)}})
            except GpuBusy as e:
                self._json(503, {"error": {"type": "server_error", "message": str(e)}})
            except (TypeError, KeyError, AttributeError, IndexError, UnicodeError) as e:
                # a body that is JSON but the wrong shape ("messages": 5, a content part that is a number): a 400 that
                # says so, with the traceback in the log, instead of a dropped connection - unless the answer began
                if getattr(self, "answer_started", False):
                    raise
                import traceback
                print(f"[strata] 400 malformed request ({type(e).__name__}: {e}):\n" + traceback.format_exc(),
                      flush=True)
                body = {"error": {"type": "invalid_request_error",
                                  "message": f"the request is malformed ({type(e).__name__}: {e})"}}
                self._json(400, responses_error_body(body["error"]["message"]) if path == "/v1/responses" else body)
            except (EngineDied, EngineStarting) as e:        # before the answer started (not streamed)
                msg = f"{e}; the next request restarts it" if isinstance(e, EngineDied) else str(e)
                if self.path.startswith("/v1/messages"):     # xeno #49 S2: the native API's retryable error, also
                    self._json(529, {"type": "error", "error": {"type": "overloaded_error", "message": msg}})
                else:                                        # while it (re)starts (#344's context 0, merge of 0.1.34)
                    self._json(503, {"error": {"type": "server_error", "message": msg}})
            except EngineStuck as e:                         # an unload or restart that could not end the engine
                self._json(503, {"error": {"type": "server_error", "message": str(e)}})

            except OSError:
                self._note(outcome="disconnected")
                raise                                        # as before #332: the server's own handling
            finally:
                svc.drop_embeddings()                        # the images' file of a request that never got to run()
                if self.watch_done is not None:
                    self.watch_done.set()
                record = self.record
                if record is not None:
                    with svc.status_lock:
                        record["wallclock_s"] = round(time.perf_counter() - record["_clock"], 3)
                        record["finished_at"] = time.time()
                        record["state"] = "error" if record.get("error") else record.get("outcome", "completed")
                    svc.request_trace.record = None

        def _props(self):
            model = parse_qs(urlsplit(self.path).query).get("model", [svc.model])[0]
            if model not in svc.model_names():
                self._json(404, {"error": {"message": "model not found"}})
                return
            if not svc.loaded() and not getattr(svc.engine, "unloaded", False):
                self._json(503, {"error": {"message": "the engine is not running"}})
                return
            defaults = {**svc.sampling_defaults, **svc.shared}
            names = {"repetition_penalty": "repeat_penalty", "penalty_last_n": "repeat_last_n"}
            params = {names.get(k, k): v for k, v in defaults.items()
                      if k in ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
                               "presence_penalty", "frequency_penalty", "penalty_last_n")}
            params["n_predict"] = svc.shared.get("max_tokens", -1)
            props = {"default_generation_settings": {"n_ctx": svc.reported_ctx(), "params": params},
                     "total_slots": max(1, int(getattr(svc.engine, "batch", 0) or 0)),   # #1004: the batch slots
                     "model_alias": svc.model, "chat_template": svc.template.source,
                     "chat_template_caps": svc.template.caps,
                     "modalities": {"vision": svc.vision is not None}, "models_autoload": hasattr(svc.engine, "restart"),
                     "is_sleeping": not svc.loaded()}
            if getattr(svc.engine, "model_path", None):
                props["model_path"] = svc.engine.model_path
            version = getattr(svc.engine, "info", {}).get("version")
            if version:
                props["build_info"] = "Strata " + str(version)
            self._json(200, props)

        def _control_body(self) -> bool:
            """Consume the unused control body before replying/closing (Windows otherwise sends a TCP reset)."""
            if self._chunked():
                self.body_read = True
                try:
                    self.connection.settimeout(2.0)
                    self._read_chunked(65536, keep=False)
                except BadBody as e:
                    self._json(e.status, {"error": {"message": f"control request: {e}"}})
                    return False
                except OSError:
                    self._json(400, {"error": {"message": "incomplete control request body"}})
                    return False
                return True
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"error": {"message": "invalid Content-Length"}})
                return False
            if not 0 <= length <= 65536:
                self._json(413, {"error": {"message": "control request body is limited to 64 KiB"}})
                return False
            timeout = self.connection.gettimeout()
            try:
                self.connection.settimeout(2.0)
                complete = len(self.rfile.read(length)) == length
            except OSError:
                complete = False
            finally:
                self.connection.settimeout(timeout)
                self.body_read = True                    # nothing left for the drain to wait for (#594)
            if not complete:
                self._json(400, {"error": {"message": "incomplete control request body"}})
            return complete

        def _keys_post(self, body: dict):
            """#310: POST /keys - create, rename, change the role of or revoke a named key (an owner's).  The plain key of a
            new one is in this answer only: it is not printed, logged or kept (only its sha256 is).  Outside team mode
            the first create makes the keys file and so turns team mode on; nothing else writes it there."""
            if not self._owner_only("keys can be managed"):
                return
            action, name = body.get("action"), body.get("name")
            if action not in ("create", "rename", "role", "revoke") or not isinstance(name, str):
                return self._json(400, {"error": {"message": 'send {"action": "create", "name", "role"}, {"action": "rename", '
                                                             '"name", "to"}, {"action": "role", "name", "role"} or {"action": '
                                                             '"revoke", "name"}'}})
            if not svc.keys.path:
                return self._json(409, {"error": {"message": "this server was started without a run config (--config), so "
                                                             "there is no keys file to keep keys in"}})
            if action != "create" and name.strip().casefold() == "owner":
                return self._json(400, {"error": {"message": "the owner key is the run config's api_key: change or remove it "
                                                             "there; it cannot be renamed, made a guest or revoked here"}})
            if action != "create" and not self.team:        # no keys file: no key to change, and nothing is written
                return self._json(404, {"error": {"message": f"there is no key named {name!r}"}})
            if not self.team and not svc.api_key and not self._loopback_local():
                # the first key turns team mode on: with no API key it is made on this PC itself only, never through a
                # tunnel that names this PC (may_edit does not look at the forwarding headers)
                return self._json(403, {"error": {"type": "forbidden", "message": "the first key is made only on this PC "
                                                  "itself (open the page as localhost, not through a tunnel or proxy), or "
                                                  "with the API key"}})
            by, missing = self.identity, {"error": {"message": f"there is no key named {name!r}"}}
            try:
                if action == "create":
                    role = body.get("role")
                    key = svc.keys.create(name, role, by=by, owner_first=True)
                    return self._json(200, {"key": key, "name": name.strip(), "role": role})
                if action == "rename":
                    with svc.runner_hub.names:              # #315: no runner's hello between the key's rename and
                        renamed = svc.keys.rename(name, body.get("to"), by=by)      # its runners' move
                        if renamed:
                            new = body["to"].strip()
                            svc.runs.move(name, new)        # the holder keeps her runs and prompts; a later key named
                            svc.broker.move(name, new)      # `name` inherits none of them
                            svc.runner_hub.move(name, new)  # and her runners (their ids follow the name)
                    if not renamed:                         # answered after the lock: a slow client holds no hello
                        return self._json(404, missing)
                    return self._json(200, {"ok": True, "name": new})
                if action == "role":
                    role = body.get("role")
                    if not svc.keys.set_role(name, role, by=by):
                        return self._json(404, missing)
                    # the coding tools are an owner's (part 1): a key made a guest keeps no run going on this PC
                    return self._json(200, {"ok": True, **(self._stop_identity(name) if role != "owner" else {})})
                if not svc.keys.revoke(name, by=by):
                    return self._json(404, missing)
                stopped = self._stop_identity(name)
                svc.runner_hub.end_all(name)                # her runners end (a demotion keeps them: not in _stop_identity)
                svc.runs.move(name, identity_mod.REVOKED)   # a later key with this name inherits nothing
                svc.broker.move(name, identity_mod.REVOKED)
                return self._json(200, {"ok": True, **stopped})
            except identity_mod.KeysFileBad as e:
                return self._json(409, {"error": {"message": str(e)}})
            except ValueError as e:
                return self._json(400, {"error": {"message": str(e)}})
            except OSError as e:
                return self._json(500, {"error": {"message": f"the keys file could not be written: {e}"}})

        def _stop_identity(self, name: str) -> dict:
            """#310: every run of `name` that is going is cancelled, and every prompt it holds is closed as refused."""
            return {"runs_stopped": svc.runs.cancel_all(name), "prompts_closed": svc.broker.end_all(name)}

        def _devices_post(self, body: dict):
            """#310: POST /devices - approve, refuse or clear a device of a key.  `may_approve` decides (the owner; a
            key's holder from an approved device of that key), before anything is looked up, so the answer to anyone else
            is the same whether the key or device exists."""
            key, device, clear = body.get("key"), body.get("device"), body.get("action") == "clear"
            if not isinstance(key, str) or not isinstance(device, str) or not (
                    clear and "approve" not in body or "action" not in body and isinstance(body.get("approve"), bool)):
                return self._json(400, {"error": {"message": 'send {"key", "device", "approve": true | false} or {"key", '
                                                             '"device", "action": "clear"}'}})
            if not svc.keys.may_approve(self.identity, key, self.device_id or None):
                return self._json(403, {"error": {"type": "forbidden", "message": "a device is decided only by an owner key, "
                                                  "or by its own key from a device of that key that is already approved"}})
            missing = {"error": {"message": "that key has no such device"}}
            if not self.team:                                # no keys file: no devices are recorded, and nothing is written
                return self._json(404, missing)
            try:
                ok = svc.keys.clear_device(key, device, by=self.identity) if clear else \
                    svc.keys.decide_device(key, device, body["approve"], self.identity)
            except identity_mod.KeysFileBad as e:            # where the file lives is the owner's to know
                return self._json(409, {"error": {"message": str(e) if self.identity.is_owner else
                                                  "the keys file on the server does not parse, so nothing can be "
                                                  "changed now: tell the owner"}})
            except OSError as e:
                return self._json(500, {"error": {"message": f"the keys file could not be written: {e}"
                                                  if self.identity.is_owner else
                                                  "the keys file on the server could not be written: tell the owner"}})
            if ok:
                if clear or not body["approve"]:             # #310 part 2: a refused or cleared device keeps no runner
                    svc.runner_hub.end_device(key, device)
                return self._json(200, {"ok": True})
            status = next((d["status"] for r in svc.keys.view() if r["name"] == key for d in r["devices"]
                           if d["device"] == device), None)
            if status == "refused" and not self.identity.is_owner:
                return self._json(403, {"error": {"type": "forbidden", "message": "a refused device is let in again only "
                                                  "by an owner key"}})
            return self._json(404, missing)

        def _runner_gate(self) -> bool:
            """#310 part 2: the runner routes, after `_authorized`.  In team mode every key (the device gate has run);
            outside it the owner, as `may_edit` decides - a runner runs its holder's tools on her own PC.  No
            `_own_page`: a runner is no page."""
            return True if self.team else self._owner_only("a runner can connect")

        def _runner_device(self) -> str:
            """The device a runner's hello comes from: the gate's in team mode; outside it `_authorized` records none,
            so it is read here the same way (two PCs on a single-user server get two runners)."""
            if self.team:
                return self.device_id
            h = self.headers
            return identity_mod.device_of(h.get("X-Strata-Device"), h.get("X-Strata-Device-Name"), h.get("User-Agent"),
                                          self._peer())[0]

        def _runner_body(self, limit: int) -> dict | None:
            """A runner's POST body: JSON only, at most `limit` bytes, an object.  None after an answer (415, 413, 400)."""
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return None
            if self._chunked():
                self.body_read = True
                raw = self._read_chunked(limit)              # BadBody 413 / 400: do_POST answers it
            else:
                try:
                    n = int(self.headers.get("Content-Length", 0))
                except ValueError:
                    n = -1
                if n > limit:
                    self._json(413, {"error": {"message": f"send a body of up to {limit >> 20} MB"}})
                    return None
                if n < 0:
                    self._json(400, {"error": {"message": "invalid Content-Length"}})
                    return None
                self.body_read = True
                raw = self.rfile.read(n)
                if len(raw) != n:                            # the client stopped sending: nothing more on this socket
                    self.close_connection = True
            self.body_done = not self.close_connection       # #313: read to its end - the connection may be kept
            try:
                body = json.loads(raw)
            except ValueError:
                body = None
            if not isinstance(body, dict):
                self._json(400, {"error": {"message": "send a JSON object"}})
                return None
            return body

        def _runner_post(self, path: str):
            """#310 part 2: POST /runner/hello and /runner/reply (serve/runner_hub.py)."""
            body = self._runner_body(RUNNER_HELLO_MAX if path == "/runner/hello" else RUNNER_REPLY_MAX)
            if body is None:
                return
            who = self._acting()
            if path == "/runner/reply":
                if not all(isinstance(body.get(k), str) for k in ("id", "instance", "job")):
                    return self._json(400, {"error": {"message": 'send {"id", "instance", "job", "rid", ...}'}})
                if type(body.get("rid")) is not int or body["rid"] < 1:     # the reply's id in its job: a retry is applied once
                    return self._json(400, {"error": {"message": "rid numbers the job's replies: a whole number from 1"}})
                r = svc.runner_hub.get_for(body["id"], who)         # another identity's runner: as one not there
                if r is None:
                    return self._json(404, {"error": {"message": NO_RUNNER}})
                if r.instance != body["instance"]:
                    return self._json(409, RUNNER_REPLACED)
                if not r.deliver(body):
                    return self._json(404, {"error": {"message": "there is no such job (it finished, or was never sent)"}})
                return self._json(200, {"ok": True})
            protocol = body.get("protocol")                          # first: a runner of another version may send another shape
            if type(protocol) is not int or protocol != runner_hub.RUNNER_PROTOCOL:
                return self._json(409, {"error": {"type": "runner_protocol", "server_protocol": runner_hub.RUNNER_PROTOCOL,
                                                  "message": f"update Strata Desktop (the server speaks runner protocol "
                                                             f"{runner_hub.RUNNER_PROTOCOL})"}})
            got = runner_hub.check_hello(body)
            if isinstance(got, str):
                return self._json(400, {"error": {"message": got}})
            info, groups, skills = got
            with svc.runner_hub.names:                      # #315: wholly before or after a rename of the key
                if self.team:                               # the key's name now, not as it was when the request came
                    who = svc.keys.match(self._given_key())
                r = None if who is None else svc.runner_hub.hello(who.name, self._runner_device(), body["instance"], info, groups, skills)
            if r is None:                                   # revoked meanwhile (answered after the lock)
                return self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            poll = runner_hub.POLL_MAX_S
            return self._json(200, {"runner": r.id, "who": r.who, "protocol": runner_hub.RUNNER_PROTOCOL,
                                    "poll_s": int(poll) if float(poll).is_integer() else poll, "channel": r.channel})

        def _runner_next(self):
            """#310 part 2: GET /runner/next?id=&instance=&wait=&channel=&ack= - the messages for the runner, held at
            most POLL_MAX_S; `ack` (the highest `seq` received on `channel`, default 0) lets the acknowledged ones go
            (an ack that names another channel than the runner's is ignored).  A runner that ended while the poll was held is told why: replaced 409, revoked 401, its
            device refused or cleared 404 (a new hello meets the device gate), the server stopping 503."""
            q = parse_qs(urlsplit(self.path).query)
            asked = q.get("id", [""])[0]
            r = svc.runner_hub.get_for(asked, self._acting())
            if r is None:
                return self._json(404, {"error": {"message": NO_RUNNER}})
            if r.instance != q.get("instance", [""])[0]:
                return self._json(409, RUNNER_REPLACED)
            try:
                wait = float(q.get("wait", [str(runner_hub.POLL_MAX_S)])[0])
            except ValueError:
                wait = math.nan
            if not math.isfinite(wait):
                return self._json(400, {"error": {"message": "wait is a number of seconds"}})
            ack = q.get("ack", ["0"])[0]
            if not (ack.isascii() and ack.isdigit() and len(ack) <= 18):
                return self._json(400, {"error": {"message": "ack is the highest seq received, a whole number"}})
            gone = threading.Event()                 # the runner hung up: the held poll ends and takes nothing
            self._watch_client(gone)
            try:
                messages = r.take(min(max(wait, 0.0), runner_hub.POLL_MAX_S), gone=gone, ack=int(ack),
                                  channel=q.get("channel", [None])[0], as_id=asked)
            finally:
                self.watch_done.set()
            if gone.is_set():
                return                                       # nobody to answer: the messages stay for the next poll
            if r.ended is None and r.id != asked:            # #315: the key was renamed meanwhile - the runner says
                return self._json(404, {"error": {"message": NO_RUNNER}})      # hello again now and learns its new id
            if r.ended == "replaced":
                return self._json(409, RUNNER_REPLACED)
            if r.ended == "revoked":
                return self._json(401, {"error": {"type": "authentication_error", "message": "the key was revoked"}})
            if r.ended == "closed":
                return self._json(503, {"error": {"message": "the server is stopping"}})
            if r.ended is not None:
                return self._json(404, {"error": {"message": NO_RUNNER}})
            return self._json(200, {"messages": messages})

        def _own_page(self, what) -> bool:
            """Only JSON (a form or a "simple" cross-site request can't send it without a CORS preflight, which this
            server never grants) and no foreign Origin: a web page elsewhere must not change settings or run tools."""
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return False
            if self._foreign_origin():
                self._json(403, {"error": {"message": f"{what} only from Strata's own page (or an origin in the "
                                                      f"config's trusted_origins)"}})
                return False
            return True

        def _config_get(self):
            if not svc.config_path:
                self._json(404, {"error": {"message": "this server was started without a run config"}})
                return
            try:
                cfg = runconfig.load(svc.config_path)
            except (OSError, ValueError) as e:
                self._json(500, {"error": {"type": "server_error", "message": f"the run config cannot be read: {e}"}})
                return
            self._json(200, runconfig.view(cfg, svc.config_path))

        def _config_post(self):
            """#564: change a few documented keys of the run config - JSON from Strata's own page only, as
            /settings (the key is checked before); every other key of the file stays as it is."""
            body = self._body()
            if not self._own_page("the run config can be changed"):
                return
            if not svc.config_path:
                self._json(404, {"error": {"message": "this server was started without a run config"}})
                return
            try:
                req = json.loads(body or b"{}")
                with svc.config_lock:
                    cfg = runconfig.load(svc.config_path)
                    new, changed = runconfig.apply(cfg, req.get("set") if isinstance(req, dict) else None)
                    if changed:
                        bak = runconfig.save(svc.config_path, new)
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            except OSError as e:
                self._json(500, {"error": {"type": "server_error", "message": f"the run config cannot be written: {e}"}})
                return
            if changed:
                print(f"[strata] the Settings view changed {', '.join(changed)} in {Path(svc.config_path).name} "
                      f"(the earlier file: {bak.name}); used from the next start", flush=True)
            self._json(200, {**runconfig.view(new, svc.config_path), "changed": changed})
        def _foreign_origin(self) -> bool:
            """A web page of another origin sent this.  The Origin must be this server's own address (host and port),
            or an origin the config trusts (trusted_origins: the web app behind a reverse proxy or tunnel); no Origin
            is a client that is no browser.  Headers a proxy adds (X-Forwarded-*, CF-Ray, CF-Connecting-IP) prove
            nothing about the page that sent the request, so they open nothing."""
            origin = (self.headers.get("Origin") or "").rstrip("/")
            return bool(origin) and origin.split("://", 1)[-1] != self.headers.get("Host", "") and \
                origin not in svc.trusted_origins

        def _no_local_images(self, messages, foreign: bool = True):
            """A web page of another origin must not have a file on this computer read as an image.  Without an
            api_key, _foreign_page refuses such a page unless cors_origins lets it in - and a page cors_origins lets
            in (["*"]: every page) could name any file and, CORS allowing, read what the model says about it.  data:
            and http(s) images are the page's own to send; files come from Strata's own page, a trusted origin or a
            client that is no browser.  #310: in team mode only an owner key may name a file on this PC - a guest's
            files are on the guest's own PC, and the refusal comes before anything looks at the path, so it does not
            tell whether the path exists.  `foreign`: False checks only the team rule.  A guest's http(s) images go
            through _guest_image_urls (#311).  ValueError (a 400)."""
            guest = self.team and not (self.identity is not None and self.identity.is_owner)
            if any(not src.startswith(("data:", "http://", "https://")) for src in images_of(messages)):
                if foreign and self._foreign_origin():
                    raise ValueError("a web page of another origin cannot have a file on this computer read as an "
                                     "image: send it as a data: URL (or add the page's origin to the config's "
                                     "trusted_origins)")
                if guest:
                    raise ValueError("a file on the server's computer can be read as an image only with an owner key: "
                                     "send the image as a data: URL")
            if guest:
                self._guest_image_urls(messages)

        def _guest_image_urls(self, messages):
            """#311: a team guest's http(s) image is fetched here, only from an address on the public internet - not
            this PC, a private or link-local network, CGNAT 100.64/10 (Tailscale) or 26.0.0.0/8 (Radmin VPN) - checked
            on the address connected to and on every redirect (Vision.download with web.public_not_vpn).  Its item then
            carries the bytes as a data: URL, so nothing later (prepare, a tool round) holds the URL to fetch it again
            without the check.  Every role's images, as images_of reads them.  With no vision encoder nothing is
            fetched (prepare refuses the image, or notes a tool's, as for anyone), and a request names at most
            GUEST_IMAGE_URLS of them: each is held in memory (final review T5b-m5, m7).  ValueError (a 400)."""
            if svc.vision is None:
                return
            urls = [src for src in images_of(messages) if src.startswith(("http://", "https://"))]
            if len(urls) > GUEST_IMAGE_URLS:
                raise ValueError(f"a request with a guest key may name at most {GUEST_IMAGE_URLS} image URLs (this one "
                                 f"names {len(urls)}): send the others as data: URLs, or in another request")
            fetched: dict[str, str] = {}
            for m in messages:
                content = m.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    src = item.get("source") if isinstance(item, dict) and item.get("type") == "image" else None
                    if isinstance(src, str) and src.startswith(("http://", "https://")):
                        if src not in fetched:
                            data = Vision.download(src, guard=web_mod.public_not_vpn)
                            fetched[src] = "data:;base64," + base64.b64encode(data).decode("ascii")
                        item["source"] = fetched[src]

        def _settings(self):
            # They change what every client gets, so only the app's own page may set them
            body = self._body()
            if not self._own_page("settings can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                if not isinstance(req, dict) or "defaults" not in req:
                    self._json(400, {"error": {"type": "invalid_request_error",
                                              "message": 'the body must be {"defaults": {...}} to set them, '
                                                         'or {"defaults": null} to clear them'}})
                    return
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            self._json(*app_settings.save_shared(svc, req["defaults"]))     # the same writer as the chat's StrataSettingsChange (#339)

        def _sse(self, now: bool = False):
            self._note(http_status=200)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")   # nginx and similar proxies pass each event at once
            if now:
                self.send_header("X-Strata-Now", str(int(time.time() * 1000)))      # the server's clock, so that the page can set the events' times (`strata_t`) against its own
            self._cors()
            self.end_headers()

        def _follow(self, run, start: int = 0):
            """Send a run's stream from event `start` on, as it grows.  A page that goes away ends only this reading: the run goes on (serve/runs.py)."""
            try:
                for ev in run.follow(start):
                    self.wfile.write(b": keep-alive\n\n" if ev is None else ev)
                    self.wfile.flush()
            except OSError:
                pass

        def _capture(self, items, api):
            """Retain bounded input/output for the monitor without changing the API response (the items as they
            are when the monitor is off)."""
            if self.record is None:
                return items
            return self._captured(items, api)

        def _captured(self, items, api):
            try:
                for item in items:
                    if item is not None:
                        if api == "openai":
                            delta = item["choices"][0]["delta"]
                            content, reasoning = delta.get("content", ""), delta.get("reasoning_content", "")
                            usage, timings = item.get("usage"), item.get("timings")
                        elif api == "responses":
                            kind = item["type"]
                            content = item["delta"] if kind == "response.output_text.delta" else ""
                            reasoning = item["delta"] if kind == "response.reasoning_text.delta" else ""
                            usage, timings = (item.get("response") or {}).get("usage"), None
                        else:
                            _, event = item
                            delta = event.get("delta", {})
                            content, reasoning = delta.get("text", ""), delta.get("thinking", "")
                            usage, timings = event.get("usage"), None
                        with svc.status_lock:
                            for name, value in (("output", content), ("reasoning", reasoning)):
                                if value:
                                    combined = self.record[name] + value
                                    self.record[name] = combined[:262144]
                                    self.record[name + "_truncated"] |= len(combined) > 262144
                            if usage:
                                self.record["usage"] = usage
                            if timings:
                                self.record["timings"] = timings
                    yield item
            finally:
                items.close()

        def _openai(self, req):
            req = svc.with_shared(req, "openai")
            skill = req.get("strata_skill")                       # the web app's "/name": that skill is loaded here, not left for the model to ask for
            if isinstance(skill, str):                            # anything but a name is ignored
                if not self._own_page("A skill can be used"):
                    return
                hub = svc.mcp.servers.get("skills") if svc.mcp is not None else None
                got = hub.call("use_skill", {"name": skill}) if hub is not None else None
                if got is None or got.get("isError"):
                    return self._json(400, {"error": {"message": f"there is no skill named {skill!r} (it may be switched off in Settings > Import)"}})
                req = {**req, "messages": skills_mod.put_before_last_user(req.get("messages") or [], skills_mod.invoked(skill, got["content"][0]["text"]))}
            arun = None
            premine = None                                       # #338: the runner's MCP groups of a Server PC chat, once resolved
            sa = req.get("strata_agent")                         # the web app's coding tools (serve/agent.py): files and commands on this PC, with the user's say-so
            if isinstance(sa, dict):
                if not self._own_page("The coding tools can be used"):
                    return
                runner = None
                ok, why = self._edit_answer()
                if isinstance(sa.get("runner"), str) and (self.team or ok):      # #310 part 2: the project is on the caller's own PC
                    runner = svc.runner_hub.get_for(sa["runner"], self._acting())     # only her own runner, the owner role too
                    if runner is None:
                        return self._json(404, {"error": {"type": "runner_not_found", "message": "there is no runner with that id for this key"}})
                    if not runner.online():
                        return self._json(409, self._runner_offline(runner))
                elif not ok:
                    if self.team:                                 # #310: the coding tools run on this PC: an owner's
                        self._owner_only("The coding tools can be used")
                        return
                    return self._json(403, {"error": {"message": "The coding tools change files and run commands on this PC, so they work only from this PC itself (open the page as localhost) or with the API key. " + why}})
                if runner is None and (svc.agent is None or svc.mcp is None):
                    return self._json(409, {"error": {"message": "the coding tools are switched off on this server (\"agent\": false in the run config)"}})
                guest = self.team and not (self.identity is not None and self.identity.is_owner)     # its WebFetch: never the team's network (I-1)
                if runner is None:
                    local = mcp_admin.is_loopback(self._peer()) and mcp_admin.host_is_loopback(self.headers.get("Host", ""))      # #339: the page is on this PC
                    if "strata_mcp_runner" in req and req.get("strata_mcp") is True:
                        premine = self._mcp_runner_groups(req["strata_mcp_runner"])      # #338: resolved once, here, so the rules know them (#339)
                        if premine is False:
                            return
                    arun = svc.start_agent_run(sa, req.get("messages") or [], self._acting().name, guest=guest, local=local,
                                               my_mcp=self._my_mcp_offered(req, premine or {}))     # #310: the run and what it asks are this identity's
                else:
                    try:                                          # the facts job: up to REQUEST_TIMEOUT_S, on this request's own thread, no lock held
                        arun = svc.start_agent_run(sa, req.get("messages") or [], self._acting().name, runner=runner, guest=guest,
                                                   my_mcp=self._my_mcp_offered(req, runner.servers()))
                    except runner_hub.RunnerUnavailable as e:
                        return self._json(409, {"error": {"type": "runner_offline", "message": e.message[:300]}})
                    base = svc.mcp if svc.mcp is not None else McpHub({})
                    mine = runner.servers()               # my_agent, my_skills, my_<name>: after the server's groups
                    arun.hub = base.with_servers(mine, replace={"agent": runner_hub.agent_subset(svc.agent, runner_hub.SERVER_TOOLS)}
                                                 if svc.agent is not None else None)        # never two coding groups: the server's keeps WebFetch, WebSearch, Task
                    arun.coding = next((k for k, s in arun.hub.servers.items() if getattr(s, "runner", None) is runner      # the name its coding
                                        and getattr(s, "group_kind", "") == "coding"), "my_agent")             # group landed under in the view
                req = {**req, "messages": agent_prompt.with_system(req.get("messages") or [], arun.prompt)}
            bias_api.normalize(req.get("logit_bias"), len(getattr(svc.tok, "tokens", ())) or None)
            messages, tools, kw = openai_to_messages(req, vision=svc.vision is not None)       # #334: a scanned PDF page of an attached file is an image only with vision
            req = svc.cjk(req, messages)                                # xeno #49 S4
            if arun is not None:
                arun.kw, arun.sampling = kw, req                      # a helper (serve/subagent.py) runs with the same settings
            self._no_local_images(messages)
            if tool_choice_of(req.get("tool_choice"))[0] == "none":   # as the Responses route: no tools are offered
                tools = None
            force = forced_call(req.get("tool_choice"), tools)      # a bad value is a 400 before anything is sent
            messages, validator = prepare_format(req.get("response_format"), messages)
            if validator is not None and (tools or req.get("strata_mcp")):
                raise ValueError("structured response_format with tools/MCP is not supported")
            # an OpenAI request waits in the queue while another one loads the model (#208, #332) - in load() only, see
            # there; the Anthropic API answers 529 at once instead (xeno #49 S2: Claude Code retries it, rather than
            # wait minutes)
            svc.load(wait_for_restart=True, who=self._acting().name)
            max_req = max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 0)   # 0/-1: the rest
            bias_api.validate_request(req, svc.engine, len(getattr(svc.tok, "tokens", ())) or None)
            hub = arun.hub if arun is not None and arun.hub is not None else svc.mcp      # #310 part 2: a run on My PC has its own view
            if "strata_mcp_runner" in req and req.get("strata_mcp") is True and not (arun is not None and arun.runner is not None):
                mine = premine if premine is not None else self._mcp_runner_groups(req["strata_mcp_runner"])     # #338: My PC's MCP servers in a Server PC chat
                if mine is False:
                    return
                if mine:
                    hub = (hub if hub is not None else McpHub({})).with_servers(mine)
                    if arun is not None:
                        arun.hub = hub                           # its helpers have them too (serve/subagent.py)
            use_mcp = (req.get("strata_mcp") is True or arun is not None) and hub is not None      # the web app's opt-in (serve/mcp.py), or the coding tools
            own = {t.get("name") for t in tools or [] if isinstance(t, dict)}   # #592: a second line of defence
            if use_mcp:
                if not self._own_page("MCP tools can be used"):   # tools run with the user's rights on this PC
                    return
                hub.wait(10)                                      # servers still starting (only right after start)
                off = req.get("strata_mcp_off")                   # servers the page's list switched off for this chat; anything but a list of names is ignored
                skip = {x for x in off if isinstance(x, str)} if isinstance(off, list) and all(isinstance(x, str) for x in off) else set()
                subset = arun is not None and arun.runner is not None and svc.agent is not None    # #310 part 2: the view's "agent" is the server's WebFetch / WebSearch / Task
                if req.get("strata_mcp") is not True:             # (without server coding tools a config server may be called "agent": it is an MCP server then)
                    keep = {arun.coding, "strata"} | ({"agent"} if arun.runner is None or subset else set())      # #339: Strata's own docs and settings go with the coding tools
                    skip |= {n for n in hub.servers if n not in keep}      # the coding tools alone: no MCP server
                if arun is None:
                    skip.add("agent")                             # and the coding tools only when the request asks for them
                allowed = self._mcp_allowed()                     # #310 R11: a team guest, only the servers marked for guests
                if allowed is not None:                           # (the server's WebFetch / WebSearch / Task of a run on My PC stay)
                    skip |= {n for n in (svc.mcp.servers if svc.mcp is not None else ()) if n not in allowed and not (subset and n == "agent")}
                if hub is not svc.mcp:                            # #310 part 2: a run on My PC keeps the names it is offered now, for every
                    hub.pin(skip)                                 # call of this request and its helpers; a skipped server is never called
                hide = (arun.hidden if arun is not None else set()) | (set(strata_tools.OWNER_TOOLS) if allowed is not None or arun is None else set())    # #339: a guest, or a chat without the coding tools (no card to ask on), is not offered the server's settings
                extra = hub.template_tools(exclude=own | hide, skip_servers=skip)       # the request's own tools win a name clash; the coding tools that are switched off are not offered
                use_mcp = bool(extra)
                tools = (tools or []) + extra or None
            if force and use_mcp:
                raise ValueError("a forced tool_choice with MCP tools is not supported")
            svc.reasoning_budget(req)                         # a bad value is a 400 before anything is sent
            stop_strings(req)                                 # ... and so is a bad stop / stop_sequences
            check_request_sampling(req)                       # ... and a sampling field of the wrong type
            messages, req, ids, thinking, max_new = svc.prepare_carried(          # #200
                req, messages, tools, kw, max_new, 0 if req.get("stream") else 1, force=force)
            req = svc.with_slot(req, ids, self._slot_who(), "openai")   # xeno #49 S7, #321
            req = {**req, "_meta": svc.meta_for("openai", messages, tools, self.headers.get("User-Agent"),
                                               who=self._acting().name)}
            _debug_req("openai", req, messages, tools, max_new, thinking, len(ids))
            cancel = threading.Event()
            rid = sa.get("run") if isinstance(sa, dict) else None
            detached = arun is not None and bool(req.get("stream")) and isinstance(rid, str) and re.fullmatch(r"[A-Za-z0-9_-]{8,64}", rid) is not None
            if arun is not None:
                arun.bind(cancel)
                if not self._still_may_run(arun):            # #310: a revoke or demotion while this request waited
                    return
            if not detached:
                self._watch_client(cancel)                   # #430 #431 (a run that goes on when the page goes away is not cancelled by it: serve/runs.py)
            run = run_with_mcp(svc, hub, messages, tools, kw, ids, thinking, max_new, max_req, req, cancel,
                               {t["name"] for t in extra}, arun) if use_mcp else None
            chunks = openai_chunks(svc, req, ids, thinking, tools, max_new, cancel, run=run, force=force)
            if validator is not None:
                chunks = structured_chunks(chunks, validator)
            chunks = self._capture(chunks, "openai")
            if detached:
                def describe(e):
                    if isinstance(e, EngineDied):
                        return {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}}
                    return {"error": {"type": "server_error", "message": str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"}}
                buf = svc.runs.launch(rid, chunks, cancel, describe, owner=arun, who=arun.who)             # the answer is written on its own thread: a refresh of the page does not stop it
                if buf is None:
                    return self._json(409, {"error": {"message": "there is a run with that id already"}})
                self._sse()
                return self._follow(buf, 0)
            if not req.get("stream"):
                return self._json(200, openai_collect(chunks))
            self._sse()
            try:
                for c in chunks:
                    if c is None:
                        self.wfile.write(b": keep-alive\n\n")      # an SSE comment: clients ignore it
                    else:
                        self.wfile.write(b"data: " + json.dumps(c, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
            except OSError:
                self._note(outcome="disconnected")
                cancel.set()                                 # client went away: stop the engine
                chunks.close()
            except EngineDied as e:                          # mid-stream: say so, then end the stream properly
                err = {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}}
                self._note(error=err["error"])
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")
            except StructuredOutputError as e:
                err = {"error": {"type": "structured_output_failed", "code": "structured_output_failed", "message": str(e)}}
                self._note(error=err["error"])
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started: the
                err = {"error": {"type": "server_error", "message": str(e)}}   # headers are sent, so no 400 now
                self._note(error=err["error"])
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")

        def _responses_ready(self, req, events):
            """Send a finished Responses result that did not run on the engine."""
            if not req.get("stream"):
                return self._json(200, responses_api.collect(events))
            self._sse()
            try:
                for e in events:
                    self.wfile.write(f"event: {e['type']}\n".encode() + b"data: " +
                                     json.dumps(e, ensure_ascii=False).encode() + b"\n\n")
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")

        def _responses(self, req):
            """#451: POST /v1/responses - OpenAI's Responses API, stateless (serve/responses.py), on the chat path.
            Errors use the Responses format; once the stream has started they arrive as a response.failed event."""
            try:
                title = (responses_api.thread_title_events(req, svc.model_for(req))
                         if svc.codex_thread_titles else None)
            except ResponsesError as e:
                return self._json(e.status, e.body())
            if title is not None:
                return self._responses_ready(req, title)
            try:
                ids, thinking, tools, max_new, asm, validator, req = self._responses_prepare(req)
            except ResponsesError as e:
                return self._json(e.status, e.body())
            except ModelBusy as e:
                return self._json(409, responses_error_body(str(e), "server_error", code="model_busy"))
            except (GpuBusy, EngineStarting, EngineStuck, EngineDied) as e:
                return self._json(503, responses_error_body(str(e), "server_error", code="server_error"))
            cancel = threading.Event()
            self._watch_client(cancel)                       # #430 #431

            def check(text, finish):
                return validated_json(text, validator, finish)

            def events():
                yield from asm.start()
                done = None
                for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel):
                    if kind == "ping":
                        yield None
                    elif kind == "event":
                        yield from asm.feed(x)
                    elif kind == "done":
                        done = x
                if done is not None and done["finish"] != "cancel" and not cancel.is_set():
                    yield from asm.finish(done, check)
            items = self._capture(events(), "responses")
            if not req.get("stream"):
                try:
                    result = responses_api.collect(items)
                except StructuredOutputError as e:
                    return self._json(502, responses_error_body(str(e), "server_error",
                                                                code="structured_output_failed"))
                except EngineDied as e:
                    return self._json(503, responses_error_body(f"{e}; the next request restarts it", "server_error",
                                                                code="server_error"))
                except ValueError as e:                      # the engine's ERR line
                    return self._json(500, responses_error_body(str(e), "server_error", code="server_error"))
                return self._json(200, result)
            self._sse()
            last = time.monotonic()

            def send(e):
                self.wfile.write(f"event: {e['type']}\n".encode() + b"data: " +
                                 json.dumps(e, ensure_ascii=False).encode() + b"\n\n")

            try:
                for e in items:
                    if e is None:
                        # Codex's idle timeout (5 minutes by default) counts events, not SSE comments: while a long
                        # prompt is read, a response.in_progress every 15 s tells it the server is still working
                        if time.monotonic() - last >= 15:
                            send(asm.in_progress())
                            last = time.monotonic()
                        else:
                            self.wfile.write(b": keep-alive\n\n")
                    else:
                        send(e)
                        last = time.monotonic()
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")
                cancel.set()                                 # client went away: stop the engine
                items.close()
            except EngineDied as e:                          # mid-stream: response.failed, then the stream ends
                self._responses_failed(asm, f"{e}; the next request restarts it", "server_error", send)
            except StructuredOutputError as e:
                self._responses_failed(asm, str(e), "structured_output_failed", send)
            except ValueError as e:                          # the engine's ERR after the stream started
                self._responses_failed(asm, str(e), "server_error", send)

        def _responses_failed(self, asm, message, code, send):
            e = asm.failed(message, code)
            self._note(error=e["response"]["error"])
            try:
                send(e)
                self.wfile.flush()
            except OSError:
                pass

        def _responses_prepare(self, req):
            responses_api.check_request(req)
            messages = responses_api.input_messages(req)
            tools, names, skipped = responses_api.request_tools(req)
            kw = responses_api.template_kwargs(req, svc.shared)
            try:
                self._no_local_images(messages)              # as on the chat route: no file read for a foreign page
            except ValueError as e:
                raise ResponsesError(str(e), "input") from None
            try:
                messages, validator = prepare_format(responses_api.text_format(req), messages, with_tools=bool(tools))
            except ValueError as e:
                raise ResponsesError(str(e).replace("response_format", "text.format"), "text.format") from None
            noted = svc.__dict__.setdefault("hosted_tools_noted", set())   # said once per tool, not per request
            if set(skipped) - noted:
                print(f"[strata] /v1/responses: left out the hosted tools {', '.join(sorted(set(skipped) - noted))} "
                      f"(this server cannot run them)", flush=True)
                noted.update(skipped)
            if req.get("max_output_tokens") is None and "max_tokens" in svc.shared:   # the shared Chat settings
                req = {**req, "max_output_tokens": svc.shared["max_tokens"]}
            try:
                svc.reasoning_budget(req)                    # a bad value is a 400 before anything is sent
            except ValueError as e:
                raise ResponsesError(str(e), "reasoning_budget_tokens") from None
            try:
                stop_strings(req)
            except ValueError as e:
                raise ResponsesError(str(e), "stop") from None
            try:
                check_request_sampling(req)                   # the request is also the sampling dict (see below)
            except ValueError as e:                           # its message starts with the field's name: the param
                raise ResponsesError(str(e), str(e).partition(":")[0]) from None
            svc.load(who=self._acting().name)
            # #924 (opt-in): a Codex compaction request is rendered with its conversation's tools, so its prompt
            # starts as the cached one did; the parser and the response keep the request's own tools
            shown = responses_api.prompt_tools(req, tools) if svc.codex_compaction_cache else tools
            try:
                try:
                    ids, thinking, max_new = svc.prepare(messages, shown, kw, req.get("max_output_tokens") or 0, req=req)
                except ValueError:
                    if shown is tools:
                        raise
                    shown = tools                            # too long with the kept tools: the request as sent
                    ids, thinking, max_new = svc.prepare(messages, shown, kw, req.get("max_output_tokens") or 0, req=req)
            except ResponsesError:
                raise
            except ValueError as e:                          # too long for the context, an image without vision
                raise ResponsesError(str(e), "input", "context_length_exceeded" if "context" in str(e) else None) \
                    from None
            if svc.codex_compaction_cache:
                responses_api.prompt_made(req, shown)
            _debug_req("responses", req, messages, tools, max_new, thinking, len(ids))
            include = req.get("include") if isinstance(req.get("include"), list) else []
            asm = responses_api.Assembler(req, svc.model_for(req), len(ids), names,
                                          "reasoning.encrypted_content" in include, validator is not None)
            # the request is also the sampling dict, as on the chat path (temperature, top_p, top_k, seed, ...); its
            # history meta is the empty one run() makes, but with the identity that sent it (#310)
            req = {**req, "_meta": request_meta("", [], None, None, self._acting().name)}
            return ids, thinking, tools, max_new, asm, validator, req

        def _count_tokens(self, req):
            """Anthropic's token count, which Claude Code asks for its context figures: the prompt this server would
            read for the same request, rendered and tokenized - the model does not run.  Images count as the tokens
            their encoding takes and a PDF as its text (xeno #46, through prepare()); the low effort a side request
            gets (xeno #49 S3, think_budget.side_effort) is not applied - this is the main turn's prompt."""
            req = svc.with_shared(req, "anthropic")
            messages, tools, kw = anthropic_to_messages(req, svc.anthropic_think_unasked, vision=svc.vision is not None)
            self._no_local_images(messages, foreign=False)    # #310: counting reads the image too (team mode only)
            stop_strings(req)                                 # the same 400 as the request itself would get
            check_request_sampling(req)                       # ... likewise a sampling field of the wrong type
            try:
                ids, _, _ = svc.prepare(messages, tools, kw, None, count_only=True)   # images and PDFs counted as read (xeno #46)
            finally:
                staged = getattr(svc.embeddings, "path", None)        # an image request's embeddings, not needed
                if staged is not None:
                    Path(staged).unlink(missing_ok=True)
                svc.embeddings.path = None
            self._json(200, {"input_tokens": len(ids)})

        def _anthropic(self, req):
            svc.load(who=self._acting().name)
            req = svc.with_shared(req, "anthropic")
            messages, tools, kw = anthropic_to_messages(req, svc.anthropic_think_unasked, vision=svc.vision is not None)
            req = svc.cjk(req, messages)                                # xeno #49 S4
            self._no_local_images(messages)
            if tool_choice_of(req.get("tool_choice"))[0] == "none":   # Anthropic's {"type": "none"}: no tools offered
                tools = None
            force = forced_call(req.get("tool_choice"), tools)        # "any" / {"type": "tool", "name": N}
            max_new = int(req.get("max_tokens") or 0)                  # 0/-1: the rest of the context
            svc.reasoning_budget(req)                         # a bad value is a 400 before anything is sent
            think_budget.side_effort(req, kw)                         # xeno #49 S3: before the template renders
            stop_strings(req)                                 # ... and so is a bad stop / stop_sequences
            check_request_sampling(req)                       # ... and a sampling field of the wrong type
            messages, req, ids, thinking, max_new = svc.prepare_carried(          # #200
                req, messages, tools, kw, max_new, 0 if req.get("stream") else 1, force=force)
            req = svc.with_slot(req, ids, self._slot_who(), "anthropic")   # xeno #49 S7, #321
            req = {**req, "_meta": svc.meta_for("anthropic", messages, tools, self.headers.get("User-Agent"),
                                               who=self._acting().name)}
            budget = think_budget.for_anthropic(req, max_new)         # capped by the max_new the engine gets
            if budget and thinking:
                req = {**req, "_think_budget": budget}
            opening = forced_opening.required(messages, thinking)      # xeno: the classifier's <severity>
            if opening:
                req = {**req, "_opening": opening}
                print(f"[strata] the request requires its reply to begin with {opening}: written for the model",
                      flush=True)
            _debug_req("anthropic", req, messages, tools, max_new, thinking, len(ids))
            cancel = threading.Event()
            self._watch_client(cancel)                       # #430 #431
            if not req.get("stream"):
                key = svc.replay_key(req, ids, max_new)
                if key is not None and key in svc.replays:
                    print("[strata] the same greedy request again: its last answer, not generated again", flush=True)
                    return self._json(200, {**svc.replays[key], "id": f"msg_{uuid.uuid4().hex[:24]}"})
                events = self._capture(anthropic_events(svc, req, ids, thinking, tools, max_new, cancel, force=force), "anthropic")
                body = anthropic_collect(events)
                if key is not None and not getattr(cancel, "client_gone", False):   # never one a hang-up cut short
                    svc.remember_reply(key, body)
                return self._json(200, body)
            events = self._capture(anthropic_events(svc, req, ids, thinking, tools, max_new, cancel, force=force), "anthropic")
            self._sse()
            trace = sse_tracer()                             # xeno #49 S5: STRATA_TRACE_SSE=<file>
            try:
                for item in events:
                    if item is None:
                        self.wfile.write(b": keep-alive\n\n")
                    else:
                        name, e = item
                        self.wfile.write(f"event: {name}\n".encode() + b"data: " +
                                         json.dumps(e, ensure_ascii=False).encode() + b"\n\n")
                        trace(name, e)
                    self.wfile.flush()
            except OSError:
                self._note(outcome="disconnected")
                cancel.set()
                events.close()
            except EngineDied as e:                          # mid-stream: Anthropic's retryable error event (#49 S2)
                err = {"type": "error", "error": {"type": "overloaded_error",
                                                  "message": f"{e}; the next request restarts it"}}
                self._note(error=err["error"])
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                self._note(error=err["error"])
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")

    return Handler


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already serving, and requests then land on
    # either one (a forgotten second start of run-<model>.bat).  Without it the second start fails loudly instead.
    allow_reuse_address = os.name != "nt"
    # socketserver listens with a backlog of 5: a burst of 30-40 clients at once got "connection reset by peer" on the
    # first ones (measured on 4 x R9700, also with one request at a time); the requests wait in the server, not the kernel
    # (STRATA_HTTP_BACKLOG overrides it)
    request_queue_size = max(5, int(os.environ.get("STRATA_HTTP_BACKLOG") or 256))

    def __init__(self, *args, **kwargs):
        self._kept: set = set()                     # #313: the runner connections open between two requests
        self._kept_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def keep(self, sock, kept: bool) -> None:
        """#313: `sock` is a kept connection (True: from the end of its first kept answer) or is done (False).
        server_close shuts every one, whether it waits for its next request or is serving one (a stop ends that
        request too: the runner sees the connection end and connects again), so a stopped server answers no more
        requests on them."""
        with self._kept_lock:
            (self._kept.add if kept else self._kept.discard)(sock)

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):   # a client that hangs up needs no stack trace
            super().handle_error(request, client_address)

    def server_close(self):
        with self._kept_lock:
            kept, self._kept = list(self._kept), set()
        for sock in kept:                           # the runner sees the end at once and connects again
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        # the hardware sampler serve() started for this server stops with it (test isolation: every Service a test
        # started used to leave its sampler running, 21 of them after test_responses, which slowed test_parallel's timing)
        tel = getattr(getattr(self, "svc", None), "telemetry", None)
        if tel is not None and hasattr(tel, "close"):
            tel.close()
        super().server_close()


def warn_tight_ram(arena_mib) -> None:
    """The model's experts live in RAM (INFO arena_mib, engine 0.1.10+).  With less than ~6 GB left beside them for the
    system, the engine and this server, Linux ends the engine mid-answer when memory runs out (issue #27) and Windows
    pages to disk; say so at start instead of after a lost answer."""
    if not isinstance(arena_mib, int) or arena_mib <= 0:
        return
    try:
        import psutil
        total = psutil.virtual_memory().total
    except Exception:  # noqa: BLE001 - psutil is optional here
        return
    left = total / 2**30 - arena_mib / 1024
    if left < 6:
        print(f"[strata] WARNING: RAM is tight - the model's experts take {arena_mib / 1024:.1f} GB of this PC's "
              f"{total / 2**30:.0f} GB, leaving {left:.1f} GB for everything else. "
              + ("Linux may stop the engine in the middle of an answer. " if os.name != "nt" else
                 "Windows will slow down (paging to disk). ")
              + "Close other programs, or run START-HERE --setup and pick a smaller size (Q2_0 / IQ2_XS).", flush=True)


def warn_budget_over_ram(args) -> str | None:
    """#1080: an explicit `--resident-budget-gib N` bigger than the RAM this PC has free (or than its total beside
    ~6 GB for everything else) pushes experts into swap or the file tier: decode that is "dead slow" until the number is
    removed.  A warning at start, never a refusal (recommend, never force).  -> the sentence printed, or None."""
    try:
        budget = float(args[list(args).index("--resident-budget-gib") + 1])
    except (ValueError, IndexError):
        return None
    try:
        import psutil
        vm = psutil.virtual_memory()
        total, free = vm.total / 2**30, vm.available / 2**30
    except Exception:  # noqa: BLE001 - psutil is optional here
        return None
    if budget <= free and budget <= total - 6:
        return None
    msg = (f"WARNING: the RAM budget for experts is {budget:g} GiB (--resident-budget-gib), but this PC has "
           f"{free:.0f} GiB free of {total:.0f} GiB: what does not fit goes to swap or is read from disk, which can make "
           "decoding very slow. Lower it (setup sizes it for this PC) or remove the argument to let the engine choose.")
    print("[strata] " + msg, flush=True)
    return msg


DESKTOP_FREE_MIB = 2048          # #560 #516: below this, an AMD card that also drives a Linux desktop can run out
DESKTOP_RESERVE_MIB = 3072       # what kept KDE/Wayland alive beside a full expert cache in both reports


def linux_desktop(env=None) -> bool:
    """A graphical session on Linux (Wayland or X): its compositor, browser and apps take VRAM after the model has."""
    env = os.environ if env is None else env
    return os.name != "nt" and sys.platform != "darwin" and bool(env.get("WAYLAND_DISPLAY") or env.get("DISPLAY"))


def desktop_vram_note(backend, vram_free_mib, args: list, desktop: bool) -> str:
    """#560 #516: on Linux, when the desktop needs VRAM the AMD card does not have, amdgpu moves GPU memory (the expert
    cache, ~24 GB) to system RAM, which the experts already fill - the OOM killer then ends the compositor.  The
    default reserve (700 MiB) is sized for a card without a desktop.  A recommendation, nothing changes: "" when it
    does not apply."""
    if backend != "hip" or not desktop or not isinstance(vram_free_mib, int) or vram_free_mib >= DESKTOP_FREE_MIB:
        return ""
    try:
        reserve = int(args[args.index("--vram-reserve-mib") + 1]) if "--vram-reserve-mib" in args else 700
    except (ValueError, IndexError):
        reserve = 700
    if reserve >= DESKTOP_RESERVE_MIB:
        return ""
    return (f"[strata] note: {vram_free_mib} MiB of VRAM free with the model loaded. If this AMD card also drives your "
            "desktop and the desktop or apps crash after the start (the driver moves the expert cache to RAM and "
            "the OOM killer ends the session), keep more VRAM free: ./setup.sh --vram-reserve-mib "
            f"{DESKTOP_RESERVE_MIB} (remembered; the expert cache gets "
            f"{(DESKTOP_RESERVE_MIB - reserve) / 1024:.1f} GB less, a few % of speed)")


def lan_addresses() -> list[str]:
    """This PC's IPv4 addresses on its networks (what another device types in), without loopback/link-local."""
    import socket
    first, ips = None, set()
    try:                                                # the address of the default route; sends nothing (UDP)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            first = s.getsockname()[0]
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    ok = lambda ip: ip and not ip.startswith(("127.", "169.254.", "0."))
    return ([first] if ok(first) else []) + sorted(ip for ip in ips if ok(ip) and ip != first)


def loading_server(host="127.0.0.1", port=8095) -> ThreadingHTTPServer:
    """The port's answer while the model loads (xeno, as EXL3's a04b381): every POST is Anthropic's retryable 529
    overloaded_error and every GET a 503 "loading".  Without it the port was closed for the 1-2 minutes of a start,
    and Claude Code took each refused connection for a network fault and backed off (once to "retry in 29m")."""
    class Loading(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(503, {"status": "loading"})

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self._send(529, {"type": "error", "error": {"type": "overloaded_error",
                                                        "message": "the model is loading; retry shortly"}})

        def log_message(self, *args):
            pass

    httpd = Server((host, port), Loading)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def host_name(value) -> str:
    """The name in a Host header (or an origin's host[:port]): "Example.com:8080" -> "example.com",
    "[::1]:8095" -> "::1"; "" when it is malformed."""
    v = (value or "").strip().lower()
    if v.startswith("["):
        name, sep, rest = v[1:].partition("]")
        return name if sep and (not rest or (rest[:1] == ":" and rest[1:].isdigit())) else ""
    if v.count(":") == 1:
        v, port = v.split(":")
        if not port.isdigit():
            return ""
    elif ":" in v:                                       # a bare IPv6 address (a config entry)
        return v
    v = v.rstrip(".")
    return v if v and all(c.isalnum() or c in "-._" for c in v) else ""


def _is_ip(name: str) -> bool:
    import ipaddress
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


def _name_in(name: str, names) -> bool:
    """`name` is one of `names`, or below an entry that starts with a dot (".example.com")."""
    return name in names or any(n.startswith(".") and (name.endswith(n) or name == n[1:]) for n in names)


def allowed_hosts_of(value, env: str = "") -> list[str]:
    """The config's allowed_hosts (a name or a list) plus $STRATA_ALLOWED_HOSTS (comma-separated): host names, "*" or
    ".example.com" (it and every name below it).  A scheme, port or path is dropped ("https://a.example.com:8443/"
    -> "a.example.com"); a wrong entry stops the start (ValueError)."""
    items = [] if value in (None, "") else [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise ValueError("allowed_hosts: expected a host name or a list of them")
    out = []
    for raw in items + [x for x in env.split(",") if x.strip()]:
        x = raw.strip().lower()
        if x == "*":
            out.append(x)
            continue
        x = x.split("://", 1)[-1].split("/", 1)[0]
        dot = x.startswith(".")
        name = host_name(x[1:] if dot else x)
        if not name:
            raise ValueError(f"allowed_hosts: {raw!r} is not a host name like strata.example.com")
        out.append("." + name if dot else name)
    return list(dict.fromkeys(out))


def host_names_for(bind_host: str, allowed_hosts=(), trusted_origins=()) -> set[str]:
    """Every name this server answers to besides an IP address: localhost, the address it listens on, the config's
    allowed_hosts and trusted_origins' hosts, and - listening beyond this PC (0.0.0.0 or a LAN address) - this PC's
    name and LAN addresses (the LAN addresses matter for the Origin check, which takes no IP on trust)."""
    names = set(LOOPBACK_NAMES)
    bind = host_name(bind_host)
    if bind and bind not in ("0.0.0.0", "::"):
        names.add(bind)
    if bind not in LOOPBACK_NAMES:
        try:
            pc = socket.gethostname().lower()
            names.update((pc, pc + ".local"))
        except OSError:
            pass
        names.update(("host.docker.internal", *lan_addresses()))
    names.update(x for x in allowed_hosts if x != "*")
    names.update(filter(None, (host_name(o.split("://", 1)[-1]) for o in trusted_origins)))
    return names


def host_allowed(host, names, any_host=False) -> bool:
    """DNS rebinding: a web page of another site whose name its DNS points at 127.0.0.1 reaches this server as the
    same origin, so the browser lets it read every answer.  Its requests carry that site's name in Host, so only
    the names this server answers to pass.  An IP address passes (a page served from an IP is that IP's own page;
    rebinding needs a name), as does "*.localhost" (browsers never ask DNS for it) and a request without a Host
    header (HTTP/1.0 clients; browsers always send one)."""
    if any_host or not (host or "").strip():
        return True
    name = host_name(host)
    # xeno #71: also a name with no dot (a machine on the LAN: no public site has one) and an mDNS ".local" name,
    # which the fork's own check let through before upstream's check replaced it
    lan = bool(name) and ":" not in name and ("." not in name or name.endswith(".local"))
    return bool(name) and (_is_ip(name) or name == "localhost" or name.endswith(".localhost") or lan
                           or _name_in(name, names))


def key_list(value) -> list[str]:
    """#1344: the keys a server accepts.  --api-key / STRATA_API_KEY / "api_key" take llama.cpp's form, several keys
    separated by commas ("k1,k2"; spaces around each are dropped), or a list in the config ("api_key": ["k1", "k2"],
    the way to give a key that contains a comma).  A list is taken as it is."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(k) for k in value if str(k)]
    return [k.strip() for k in str(value).split(",") if k.strip()]


def api_key_of(value):
    """The key as a client can send it (#725).  An HTTP header loses the spaces and line ends around its value, so a
    key kept with them (a config file or an environment file with CRLF line ends, a quoted " key ") matched no
    request: every client got 401 with the right key.  A value that is only such characters raises ValueError, it
    never means "no key" (#213).  #1344: a config list of keys returns the list (each key stripped, none empty); a
    string stays a string (its commas separate keys, see key_list)."""
    if isinstance(value, (list, tuple)):
        keys = [str(k).strip() for k in value]
        if not keys or any(not k for k in keys):
            raise ValueError("an API key was given but it is empty")
        return keys
    key = "" if value is None else str(value)
    if key and not key_list(key):
        raise ValueError("an API key was given but it is empty")
    return key.strip()


def start_lines(svc, host: str, port: int, max_context: int, vision: bool, ips: list[str]) -> list[str]:
    """What main() prints once it serves: where, and whether a request needs a key - `key_required()`, so a team-mode
    server with no api_key does not say it is open (#310, final review M3).  `ips`: this PC's LAN addresses, asked only
    when `host` is not a loopback name (issue #26: reachable from other devices - say at which address)."""
    here = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    keyed = svc.key_required()
    lines = [f"ready: http://{here}:{port}/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, "
             f"context {max_context} tokens{', images on' if vision else ''}{', API key required' if keyed else ''})",
             f"       open http://{here}:{port}/ in a browser to chat; close this window to stop the model"]
    if host not in ("127.0.0.1", "localhost", "::1"):
        lines += [f"       from other devices: http://{ip}:{port}/   (API: http://{ip}:{port}/v1)" for ip in ips]
        if not ips:
            lines.append("       from other devices: http://<this PC's IP address>:" + str(port) + "/")
        if not keyed:
            lines.append("       WARNING: no API key - anyone on your network can use this model. Add \"api_key\": "
                         "\"...\" to the config (clients send it as their API key; the web page asks for it)")
    return lines


def keys_file_of(cfg: dict, config_path) -> str | None:
    """#310: where the named keys are kept: the run config's "keys_file" (a path relative to the config's folder), by
    default strata-keys.json beside the run config.  No run config, no keys file (and no team mode)."""
    if not config_path:
        return None
    name = cfg.get("keys_file", "strata-keys.json")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("keys_file: the path of the keys file, e.g. \"strata-keys.json\"")
    return os.path.join(os.path.dirname(os.path.abspath(config_path)), os.path.expanduser(name.strip()))


def key_matches(given: str, key) -> bool:
    """given: a header's value as http.server read it, each byte one character.  A key with characters outside ASCII
    arrives as UTF-8 from most clients and as Latin-1 from some; the right key passes in both forms (#725: the
    bytes were re-encoded before, so a UTF-8 key never matched).  Constant-time (#213).  #1344: `key` may name several
    keys (see key_list); the given one must match any, and every key is compared whatever the outcome."""
    raw = given.encode("latin-1", "replace")
    ok = False
    for k in key_list(key):
        ok |= hmac.compare_digest(raw, k.encode())
        try:
            ok |= hmac.compare_digest(raw, k.encode("latin-1"))
        except UnicodeEncodeError:
            pass
    return ok


def origin_allowed(origin: str, host, names, origins=()) -> bool:
    """A browser page's Origin that may use the model without an API key: this server's own page (the Origin is the
    request's own Host), a page on one of the names this server answers to (any port), or an origin the config lists
    (trusted_origins, cors_origins).  Unlike the Host check no IP passes on trust: a page served from any other IP is
    another site.  "null" (a sandboxed frame, a file:// page) does not pass: any site can send it.  An origin of another
    scheme (chrome-extension://, moz-extension://, an Electron app's app://) does: no web site can send one."""
    origin = (origin or "").strip().rstrip("/")
    if origin in origins or "*" in origins:             # cors_origins ["*"]: the config lets every page in
        return True
    scheme, sep, rest = origin.lower().partition("://")
    if not sep or not scheme:
        return False
    if scheme not in ("http", "https"):
        return True
    if host and rest == host.strip().lower():
        return True
    name = host_name(rest)
    return bool(name) and (name in LOOPBACK_NAMES or name.endswith(".localhost") or _name_in(name, names))


def serve(svc: Service, host="127.0.0.1", port=8095) -> ThreadingHTTPServer:
    svc.host_names = host_names_for(host, svc.allowed_hosts, svc.trusted_origins)
    svc.start_telemetry()
    httpd = Server((host, port), make_handler(svc))
    httpd.svc = svc
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def ui_choice(cfg: dict) -> str:
    """Which web app `/` serves: the new one, unless the run config says `"ui": "classic"` (the classic app is at /classic/ either way)."""
    return "classic" if (cfg or {}).get("ui") == "classic" else "next"
def origins_of(value, key: str, wildcard: bool) -> list[str]:
    """#321: a config's origin list ("https://chat.example.com" or a list of them) - scheme://host[:port], no path;
    "*" only where `wildcard` allows it.  A wrong entry stops the start rather than opening less or more than meant."""
    if value is None or value == "" or value == []:
        return []
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise SystemExit(f"[strata] {key}: expected an origin or a list of origins")
    out = []
    for x in items:
        x = x.strip().rstrip("/")
        if x == "*" and wildcard:
            out.append(x)
            continue
        scheme, sep, rest = x.partition("://")
        if scheme not in ("http", "https") or not sep or not rest or "/" in rest or "*" in rest:
            raise SystemExit(f"[strata] {key}: {x!r} is not an origin like https://chat.example.com"
                             + ("" if wildcard else " (no wildcards here)"))
        out.append(x)
    return out


from serve.app_settings import SHARED_KEYS, clean_shared_defaults  # noqa: E402,F401  (#339: moved there, beside the writer that uses it)


# xeno #49 S8 / story 20: coding presets, chosen in the run config ("sampling": {"preset": "balanced"}) and never
# switched automatically. None of them is measured here yet. reasoning = the sampling docs/DETAILS.md shows for this
# model; balanced = the Qwen3 family's thinking-mode recommendation (VENDOR); deterministic = greedy.
SAMPLING_PRESETS = {
    "deterministic": {"temperature": 0.0},
    "balanced": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "reasoning": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
}


def lift_reasoning_effort_arg(cfg: dict) -> None:
    """#1641: "--reasoning-effort high" in a config's "args" is meant for the thinking level, which is not an engine
    option (the engine stopped at the unknown flag).  It is the config's "reasoning_effort" (none, low, medium or high:
    the level for requests that name none), so the flag is taken out of the engine's arguments and becomes that key
    unless the config already has one; said at startup."""
    args = cfg.get("args")
    if not isinstance(args, list):
        return
    out, value, i = [], None, 0
    while i < len(args):
        a = args[i]
        if a == "--reasoning-effort" and i + 1 < len(args):
            value, i = args[i + 1], i + 2
        elif isinstance(a, str) and a.startswith("--reasoning-effort="):
            value, i = a.partition("=")[2], i + 1
        else:
            out.append(a)
            i += 1
    if value is None:
        return
    cfg["args"] = out
    if cfg.get("reasoning_effort") is None:
        cfg["reasoning_effort"] = value
    print("[strata] '--reasoning-effort' in the config's args is not an engine option: taken out of them. The thinking "
          "level is the config's 'reasoning_effort' key (outside 'args'); it is used as " +
          f"{cfg['reasoning_effort']!r}", flush=True)


# #1819: llama.cpp clients call the repetition penalty and its window repeat_penalty / repeat_last_n; they are the
# same settings as repetition_penalty / penalty_last_n (the /props answer already reports them under those names)
SAMPLING_ALIASES = {"repeat_penalty": "repetition_penalty", "repeat_last_n": "penalty_last_n"}


def sampling_defaults_from_config(cfg: dict) -> dict:
    """The run config's optional `sampling` block: defaults for the sampling fields a request leaves out, so
    a plain client gets configured sampling instead of greedy.  Supported: temperature, top_p, top_k, min_p,
    presence_penalty, repetition_penalty (also written repeat_penalty), frequency_penalty, penalty_last_n (also
    repeat_last_n), seed.  The request's own fields
    always win - an explicit temperature=0 still means greedy, a field set to null falls back to the default.
    A bad value refuses to start the server (a typo'd config should not quietly change sampling); unknown keys
    are named at startup and ignored."""
    out = {}
    block = dict(cfg.get("sampling") or {})
    preset = block.pop("preset", None)
    if preset is not None:
        if preset not in SAMPLING_PRESETS:
            raise SystemExit(f"[strata] config sampling.preset={preset!r}: expected one of "
                             f"{', '.join(SAMPLING_PRESETS)}")
        block = {**SAMPLING_PRESETS[preset], **block}         # the block's own keys win
    for alias, canon in SAMPLING_ALIASES.items():      # #1819: llama.cpp's names for the same two fields
        if block.get(alias) is not None:
            if block.get(canon) is not None:
                raise SystemExit(f"[strata] config sampling: both {alias!r} and {canon!r} are set; they are the same "
                                 f"setting, keep one")
            block[canon] = block.pop(alias)
    for key, value in block.items():
        if value is None:
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "temperature":
            if not number or value < 0:
                raise SystemExit(f"[strata] config sampling.temperature={value!r}: expected a number >= 0 (0 = greedy)")
            out[key] = float(value)
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise SystemExit(f"[strata] config sampling.top_p={value!r}: expected 0 < top_p <= 1")
            out[key] = float(value)
        elif key == "min_p":
            if not number or not 0 <= value <= 1:
                raise SystemExit(f"[strata] config sampling.min_p={value!r}: expected 0 <= min_p <= 1")
            out[key] = float(value)
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= 64:
                raise SystemExit(f"[strata] config sampling.top_k={value!r}: the sampled path takes an integer 1..64")
            out[key] = int(value)
        elif key == "presence_penalty":
            if not number or value < 0:
                raise SystemExit(f"[strata] config sampling.presence_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "frequency_penalty":
            if not number or value < 0:
                raise SystemExit(f"[strata] config sampling.frequency_penalty={value!r}: expected a number >= 0")
            out[key] = float(value)
        elif key == "repetition_penalty":
            if not number or value <= 0:
                raise SystemExit(f"[strata] config sampling.repetition_penalty={value!r}: expected a number > 0 (1 = off)")
            out[key] = float(value)
        elif key == "penalty_last_n":
            if not number or value != int(value) or value < 0:
                raise SystemExit(f"[strata] config sampling.penalty_last_n={value!r}: expected a non-negative integer")
            out[key] = int(value)
        elif key == "seed":
            if not number or value != int(value) or value <= 0:
                raise SystemExit(f"[strata] config sampling.seed={value!r}: expected a positive integer")
            out[key] = int(value)
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise SystemExit(f"[strata] config sampling.experimental_speed_projection={value!r}: expected true or "
                                 "false (the default for requests that leave it out, when the engine has the vector)")
            out[key] = value
        else:
            print(f"[strata] config sampling.{key}={value!r}: unknown key, ignored", flush=True)
    return out


def start_replicas(cfg: dict, groups: list[list[int]], exe: str, effort_end: list[str], lazy: bool):
    """Opt-in data-parallel replicas (serve/replicas.py): one StrataEngine per card group, each with its own cards, log
    and CPU cores, started together; the ReplicaEngine in front spreads the requests."""
    from serve import replicas as rp
    n = len(groups)
    cpus = rp.cpu_sets(n, cfg.get("replica_cpus") is not False)
    print(f"[strata] {n} replicas on the cards {' | '.join(','.join(map(str, g)) for g in groups)}"
          f"{' (own CPU cores each)' if cpus else ''}: requests go to the least busy one, a conversation back to its own",
          flush=True)
    engines: list = [None] * n
    errors: list = []

    def one(i: int):
        rc = rp.replica_cfg(cfg, i, groups[i])
        try:
            engines[i] = StrataEngine(exe, engine_args(rc) + effort_end, cwd=rc.get("cwd"), log=rc.get("log"),
                                      env=child_env(rc), lazy=lazy, cpus=cpus[i] if cpus else None)
        except BaseException as e:        # noqa: BLE001
            errors.append((i, e))
    rc0 = rp.replica_cfg(cfg, 0, groups[0])
    warn_budget_over_ram(engine_args(rc0) if "args" in rc0 else [])
    if os.environ.get("STRATA_REPLICA_START", "") == "serial":
        for i in range(n):
            one(i)
    else:
        ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    if errors:
        for e in engines:
            if e is not None:
                try:
                    e.close()
                except Exception:        # noqa: BLE001
                    pass
        i, e = errors[0]
        raise SystemExit(f"[strata] replica {i} did not start: {e}")
    load = int(cfg.get("replica_load_tokens") or rp.LOAD_TOKENS)
    return rp.ReplicaEngine(engines, load_tokens=load)


def check_request_sampling(req) -> None:
    """A request's sampling field of the wrong JSON type ("temperature": "0.7", "top_k": 40.0, "seed": true) is a 400
    that names the field, as the same value is through POST /settings or the run config.  Unchecked, the value
    replaced the configured default in run()'s merge and sampling_keys dropped it, so the reply was greedy.  Only
    types are checked: a number (int or float, never bool) for temperature, top_p, min_p and the three penalties; an
    integer (never bool or float) for top_k, penalty_last_n and seed.  Ranges behave as before (a temperature <= 0 is
    greedy, top_k 0 or above 64 is clamped, a seed <= 0 is unset), and an absent or null field is not checked."""
    if isinstance(req, dict):      # #1819: llama.cpp's names; the OpenAI-style field wins when both are sent
        if req.get("repeat_penalty") is not None and req.get("repetition_penalty") is None:
            req["repetition_penalty"] = req["repeat_penalty"]
        rl = req.get("repeat_last_n")
        if req.get("penalty_last_n") is None and isinstance(rl, int) and not isinstance(rl, bool) and rl > 0:
            req["penalty_last_n"] = rl      # 0 and -1 (off / the whole context) keep the engine's default window
    for key, integer in (("temperature", False), ("top_p", False), ("min_p", False), ("presence_penalty", False),
                         ("frequency_penalty", False), ("repetition_penalty", False), ("top_k", True),
                         ("penalty_last_n", True), ("seed", True)):
        value = (req or {}).get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():      # some clients send numbers as strings: read and used
            try:
                num = float(value)
                num = int(num) if integer and num == int(num) else (num if not integer else None)
                if integer and num is None:
                    raise ValueError
                if not integer and re.fullmatch(r"\s*[+-]?\d+\s*", value):
                    num = int(value)
            except (ValueError, OverflowError):
                raise ValueError(f"{key}: {'an integer' if integer else 'a number'}") from None
            if num != num or num in (float("inf"), float("-inf")):
                raise ValueError(f"{key}: {'an integer' if integer else 'a number'}")
            req[key] = value = num
        if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
            raise ValueError(f"{key}: {'an integer' if integer else 'a number'}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):          # xeno (2026-09-30): a log line must never fail a request
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    if os.environ.get("STRATA_TIMELINE"):   # #33: the engine writes the file, the server the one beside it
        timeline.configure(os.environ["STRATA_TIMELINE"] + ".server.json")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["mock", "strata"], default="mock")
    ap.add_argument("--config", help="strata engine config (JSON: exe, args, cwd, tokenizer, model_name), "
                                     "written by setup.py")
    ap.add_argument("--host", default=None,
                    help="the address to listen on: 127.0.0.1 = this PC only (the default), 0.0.0.0 = also other devices "
                         "on your network (set an API key); also \"host\" in the config")
    ap.add_argument("--script", action="append",
                    help="the mock engine's answer (default: a short greeting); given more than once, requests get "
                         "them in turn and the last one repeats")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--gpu", help="the GPU to run on, as nvidia-smi numbers them, or several for a layer split "
                                  "(\"0,2\"; also \"gpu\" in the config)")
    ap.add_argument("--tokenizer", default=str(ROOT / "pack/full/tokenizer"),
                    help="pack tokenizer directory (falls back to a byte tokenizer if absent)")
    ap.add_argument("--open", action="store_true", help="open the local page in the browser once the model is ready")
    ap.add_argument("--fit-max-tokens", action="store_true",
                    help="clamp max_tokens to the remaining context instead of rejecting the request "
                         "(default: reject with 400, like llama.cpp; also \"fit_max_tokens\": true in the config)")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""),
                    help="require this key on /v1/* (Authorization: Bearer ... or x-api-key); also $STRATA_API_KEY. "
                         "Several keys: separate them with commas (key1,key2), as llama.cpp does")
    ap.add_argument("--mcp-config", help="a JSON file with MCP servers in Claude Desktop's format ({\"mcpServers\": "
                                         "{...}}); the web app's chat can use their tools (also \"mcp_servers\" in "
                                         "the config)")
    ap.add_argument("--lazy", action="store_true", help="start the API unloaded (the engine and the vision encoder start with the first request)")
    ap.add_argument("--api-monitor", action="store_true",
                    help="the API request monitor at /api-monitor: keeps the last 100 requests' prompts and answers in "
                         "memory (also \"api_monitor\": true in the config; off by default)")
    ap.add_argument("--idle-unload", type=float, default=None, metavar="SECONDS",
                    help="unload the model after this many seconds without requests, so other programs (games, other "
                         "model servers) can use the VRAM; the next request loads it again (also \"idle_unload_s\" "
                         "in the config; default: never)")
    ap.add_argument("--min-free-vram-mib", type=int, default=None,
                    help="load an unloaded model only when this much VRAM is free, else answer 503 (also "
                         "\"min_free_vram_mib\" in the config; default: always load)")
    ap.add_argument("--before-load", help="a command run before the model is loaded again (e.g. to unload another "
                                          "server's model; also \"before_load\" in the config, a string or a list)")
    ap.add_argument("--slot-save-path", default=None, metavar="DIR",
                    help="enable POST /slots/0?action=save|restore {\"filename\": NAME} (llama-server's API): the "
                         "conversation the engine holds, to or from DIR/NAME (also \"slot_save_path\" in the config)")
    ap.add_argument("--replicas", type=int, default=None, metavar="N",
                    help="data-parallel replicas (opt-in): cut the \"gpu\" list into N equal groups and run one engine on "
                         "each, requests spread over them (also \"replicas\" in the config: a number, or "
                         "[{\"gpus\": [0, 1]}, {\"gpus\": [2, 3]}])")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig")) if a.config else {}   # Notepad adds a BOM
    lift_reasoning_effort_arg(cfg)
    if a.replicas is not None:
        cfg["replicas"] = a.replicas
    if a.gpu is not None:
        cfg["gpu"] = int(a.gpu) if a.gpu.strip().isdigit() else a.gpu
    a.host = a.host or cfg.get("host") or "127.0.0.1"   # issue #26: the run scripts pass no --host, the config can
    try:                                                # before the minutes of loading: is the port free?
        Server((a.host, a.port), BaseHTTPRequestHandler).server_close()
    except OSError as e:
        ap.error(listen_problem(a.host, a.port, e))
    if cfg.get("tokenizer"):
        a.tokenizer = cfg["tokenizer"]
    tok = ByteTokenizer()
    tpath = Path(a.tokenizer)
    if a.engine == "strata" and not (tpath / "vocab.json").exists():
        ap.error(f"the model's tokenizer is missing ({tpath / 'vocab.json'}); run setup again")
    if (tpath / "vocab.json").exists():
        import strata_tokenizer as ST
        vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for t, i in vocab.items():
            tokens[i] = t
        merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
        types = json.loads((tpath / "token_type.json").read_text())
        tok = ST.Tokenizer(tokens, merges, types)
    importer = harness.Importer()                        # the skills (and the MCP servers, to import by a click) of the other coding apps on this PC (#94)
    importer.rescan(cfg)
    agent_server = None if cfg.get("agent") is False else agent_mod.AgentServer()      # the chat's coding tools; "agent": false in the run config switches them off
    if agent_server is not None:
        shell_mod.install(agent_server, shell_mod.find_shell())
        web_mod.install(agent_server)                         # WebFetch and WebSearch: offered to a chat only when the user has switched web access on
        subagent_mod.install(agent_server)                    # Task: the same, for helpers
    strata_server = strata_tools.StrataServer()           # #339: Strata's own docs and settings for the chat (its Service is bound below)
    hub = hub_from_config(cfg, a.mcp_config, builtins={**importer.builtins(), "strata": strata_server, **({"agent": agent_server} if agent_server else {})})     # before the minutes of loading: a bad entry stops here
    placeholder = None
    if a.engine == "strata":
        if not cfg:
            ap.error("--engine strata needs --config")
        vision = None
        env = child_env(cfg)
        sampling_defaults = sampling_defaults_from_config(cfg)
        if sampling_defaults:
            pretty = ", ".join(f"{k}={v}" for k, v in sampling_defaults.items())
            print(f"[strata] sampling defaults from the config: {pretty}", flush=True)
        lazy = a.lazy or cfg.get("lazy_load") is True
        if cfg.get("vision"):
            print("loading the vision encoder ..." if not lazy else
                  "vision encoder unloaded; it starts with the model ...", flush=True)
            # relative paths are the config's cwd's, as for the engine below
            vcfg = {k: (os.path.abspath(os.path.join(cfg.get("cwd") or ".", v))
                        if k in ("exe", "mmproj", "model") and isinstance(v, str) and not os.path.isabs(v) else v)
                    for k, v in cfg["vision"].items()}
            vision = Vision(vcfg, log=open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None,
                            env=vision_env(cfg, env), lazy=lazy)
        args = engine_args(cfg)
        if cfg.get("cjk_guard"):                        # xeno #49 S4: an engine without --ban-ids would not start
            # next to the run config: one per server, not purged with %TEMP% before a restart() reads it again
            ban_path = Path(a.config).with_suffix(".cjk-ban-ids.txt")
            ids = cjk_guard.ban_ids(lambda i: tok.decode([i]), len(tokens))
            ban_path.write_text("\n".join(map(str, ids)) + "\n", encoding="ascii")
            args += ["--ban-ids", str(ban_path)]
            print(f"[strata] CJK guard: {len(ids)} Han token ids banned unless a prompt has or names Chinese",
                  flush=True)
        placeholder = loading_server(a.host, a.port)    # xeno: 529 while loading, never a refused connection
        print("model unloaded; the first request loads it ..." if lazy else
              "loading the model (the first start takes a minute or two) ...", flush=True)
        warn_budget_over_ram(engine_args(cfg) if "args" in cfg else [])      # #1080
        if layer_split_of(cfg):
            try:
                split = layer_split_value(cfg)          # #644: before the (minutes-long) start
            except ValueError as e:
                raise SystemExit(f"[strata] config {e}")
            print(f"[strata] layer split across GPUs {gpu_list(cfg)} ({split})", flush=True)
        elif len(gpu_list(cfg)) > 1:
            print(f"[strata] GPUs {gpu_list(cfg)}: the later card(s) serve as the peer expert tier (--peer-device)",
                  flush=True)
        # a relative "exe" is the config's cwd's: Windows' CreateProcess resolves "engine/strata.exe" against nothing
        # it is told about (WinError 2), so it is made absolute here
        exe = cfg["exe"] if os.path.isabs(cfg["exe"]) else os.path.abspath(os.path.join(cfg.get("cwd") or ".", cfg["exe"]))
        try:
            silence = engine_silence_s(cfg)             # #481: checked before the (minutes-long) start
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        try:
            effort_end = effort_end_args(cfg, exe, tok)  # #458
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        from serve import replicas as rp
        try:
            groups = rp.gpu_groups(cfg)
        except rp.ReplicaConfigError as e:
            raise SystemExit(f"[strata] config {e}")
        if groups:
            engine = start_replicas(cfg, groups, exe, effort_end or [], lazy)
        else:                                           # args: engine_args(cfg) and the fork's --ban-ids (#49 S4)
            engine = StrataEngine(exe, args + (effort_end or []), cwd=cfg.get("cwd"), log=cfg.get("log"), env=env,
                                  lazy=lazy)
        engine.silence_s = silence                      # an attribute of its own: restart() keeps it
        warn_tight_ram(engine.info.get("arena_mib"))
        note = desktop_vram_note(cfg.get("backend"), engine.info.get("vram_free_mib"), engine.spawn[1],
                                 linux_desktop()) if not groups else None
        if note:                                        # #560 #516: before --open starts a browser on that card
            print(note, flush=True)
    else:
        effort_end = None
        engine, vision, sampling_defaults = MockEngine(tok, a.script or [
            "Thinking about it.</think>\n\nHello from the mock engine."]), None, {}
    # the model's own chat template (exported with its tokenizer), else the original model's
    tpl = tpath / "chat_template.jinja"
    svc = Service(engine, tok, ChatTemplate(tpl if tpl.exists() else ROOT / "serve/chat_template.jinja"),
                  model_name=cfg.get("model_name", "qwen3.8-flash-next"), vision=vision,
                  sampling_defaults=sampling_defaults,
                  fit_max_tokens=a.fit_max_tokens or cfg.get("fit_max_tokens") is True)
    svc.reasoning_close_retry = cfg.get("reasoning_close_retry") is True    # #1053: opt-in, off by default
    if cfg.get("prevent_sleep") is not None and not isinstance(cfg["prevent_sleep"], bool):
        raise SystemExit(f"[strata] config \"prevent_sleep\" must be true or false, not {cfg['prevent_sleep']!r}")
    from serve.keepawake import KeepAwake
    svc.keep_awake = KeepAwake.create(cfg.get("prevent_sleep") is True)      # #1727: opt-in, Windows only
    if cfg.get("reasoning_effort") is not None:         # #1641: the thinking level of requests that name none
        try:
            svc.config_effort = clean_shared_defaults({"reasoning_effort": cfg["reasoning_effort"]}).get("reasoning_effort")
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        if svc.config_effort:
            print(f"[strata] thinking level: {svc.config_effort} (reasoning_effort in the config; a request, or the "
                  "Chat settings shared with other apps, can set its own)", flush=True)
    svc.codex_thread_titles = cfg.get("codex_thread_titles") is True    # #923: opt-in, off by default
    svc.codex_compaction_cache = cfg.get("codex_compaction_cache") is True   # #924: opt-in, off by default
    try:
        svc.set_aliases(cfg.get("aliases"))             # #297: other names the model answers to
    except ValueError as e:
        raise SystemExit(f"[strata] config {e}")
    if svc.aliases:
        print(f"[strata] model aliases: {', '.join(svc.aliases)}", flush=True)
    if ("STRATA_API_KEY" in os.environ and not os.environ["STRATA_API_KEY"].strip()) or             any(x == "--api-key" and i + 1 < len(sys.argv) and not sys.argv[i + 1].strip() or x.strip() == "--api-key="
                for i, x in enumerate(sys.argv)):
        # #213: an empty key would switch authentication off without a word
        print("[strata] an API key was given but it is empty: set a key, or leave --api-key / STRATA_API_KEY out",
              file=sys.stderr)
        return 2
    try:
        svc.api_key = api_key_of(a.api_key or cfg.get("api_key", ""))
    except ValueError as e:
        print(f'[strata] {e}: set a key, or leave --api-key / STRATA_API_KEY / "api_key" out', file=sys.stderr)
        return 2
    if not svc.api_key and "api_key" in cfg:             # #569: written, but empty: the server has none (a warning, not a stop)
        print("[strata] the api_key in the config is empty: this server has no API key (anyone who can reach it can "
              "use it); set one, or leave api_key out", file=sys.stderr, flush=True)
    try:                                                 # #310: the named keys; the file existing is team mode
        svc.keys = identity_mod.KeyStore(keys_file_of(cfg, a.config), svc.api_key)
    except ValueError as e:
        raise SystemExit(f"[strata] config {e}")
    if svc.keys.team_mode():
        print(f"[strata] team keys: {svc.keys.path} ({len(svc.keys.view())} named); a key used from a new device waits "
              "for the owner's approval", flush=True)
    svc.cjk_ban = a.engine == "strata" and bool(cfg.get("cjk_guard"))   # xeno #49 S4: --ban-ids was passed
    svc.cors_origins = origins_of(cfg.get("cors_origins"), "cors_origins", wildcard=True)
    svc.trusted_origins = origins_of(cfg.get("trusted_origins"), "trusted_origins", wildcard=False)
    try:
        svc.allowed_hosts = allowed_hosts_of(cfg.get("allowed_hosts"), os.environ.get("STRATA_ALLOWED_HOSTS", ""))
    except ValueError as e:
        raise SystemExit(f"[strata] config {e}")
    if svc.allowed_hosts:
        print("[strata] Host check off: any name reaches this server (allowed_hosts \"*\")" if "*" in svc.allowed_hosts
              else f"[strata] also answers to the host names {', '.join(svc.allowed_hosts)} (allowed_hosts)", flush=True)
    svc.api_monitor = a.api_monitor or cfg.get("api_monitor") is True
    if svc.api_monitor:                                  # #310: a team-mode server needs a key even with no api_key
        print("[strata] API request monitor on (/api-monitor): the last 100 requests' prompts and answers are kept in "
              "memory" + ("" if svc.key_required() else "; anyone who can reach this server can read them (no API key)"),
              flush=True)
    if svc.cors_origins:
        print(f"[strata] CORS on /v1/* for {', '.join(svc.cors_origins)}"
              + ("" if svc.key_required() or "*" not in svc.cors_origins else
                 " - WARNING: any web page may use the model (no API key)"), flush=True)
    svc.idle_unload_s = a.idle_unload if a.idle_unload is not None else float(cfg.get("idle_unload_s") or 0)
    svc.min_free_vram_mib = a.min_free_vram_mib if a.min_free_vram_mib is not None else \
        int(cfg.get("min_free_vram_mib") or 0)
    svc.before_load = a.before_load or cfg.get("before_load") or None
    hist = cfg.get("history") if isinstance(cfg.get("history"), dict) else {}      # {"enabled", "dir", "detail_cap_gb",
    from serve.history import TELEMETRY_DAYS, default_dir                          #  "telemetry", "telemetry_days"}
    svc.hstore = HistoryStore(hist.get("dir") or default_dir(), enabled=hist.get("enabled", True) is not False,
                              detail_cap_bytes=int(float(hist.get("detail_cap_gb", 2)) * 2**30),
                              telemetry=hist.get("telemetry", True) is not False,             # xeno #237
                              telemetry_days=hist.get("telemetry_days", TELEMETRY_DAYS))
    svc.checkpoints = checkpoints_mod.Checkpoints()          # the way back for files the tools change: in the user's data folder, the oldest trimmed at start
    threading.Thread(target=svc.checkpoints.trim, daemon=True).start()
    def _model_info(files=gguf_info.files_from_args(list(cfg.get("args") or []))):
        svc.model_info = gguf_info.model_info(files)        # reads headers only (~0.1 s); a model with no GGUF gives None
    threading.Thread(target=_model_info, daemon=True).start()
    svc.ui = ui_choice(cfg)                                            # which web app "/" serves (xeno UI)
    svc.config_path, svc.mcp_config_path = a.config, a.mcp_config      # where the web app saves and reads the MCP servers (#79)
    # (xeno #71's own allowed_hosts parse is upstream's allowed_hosts_of above since 0.1.38)
    # --slot-save-path is relative to the working directory the server was started in; the config's slot_save_path
    # to the config's "cwd" (the engine's folder) when it has one.  Either way the engine gets an absolute path.
    if a.slot_save_path or cfg.get("slot_save_path"):
        try:
            svc.slot_save_path = slot_save_dir(a.slot_save_path) if a.slot_save_path else \
                slot_save_dir(cfg["slot_save_path"], cfg.get("cwd"))
        except (ValueError, OSError) as e:
            raise SystemExit(f"[strata] {e}")
        print(f"[strata] slot save/restore on: {svc.slot_save_path} (session files are kept until deleted: "
              "about 1.2 GB per 63K-token conversation; restore only files this server wrote)", flush=True)
    mode = str(cfg.get("anthropic_thinking") or "model")   # #278: "on_request" = only when the request asks
    if mode not in ("model", "on_request"):
        raise SystemExit(f"[strata] config anthropic_thinking must be \"model\" or \"on_request\", not {mode!r}")
    svc.anthropic_think_unasked = mode == "model"
    if cfg.get("reasoning_carry") is not None:          # #200: carry a thinking cut at max_tokens to the next turn
        try:
            svc.reasoning_carry = reasoning_carry.mode_of({}, cfg["reasoning_carry"])
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        print(f"[strata] reasoning carry: {svc.reasoning_carry}", flush=True)
    recovery = cfg.get("tool_call_recovery", False)     # opt-in: see OutputParser's recover
    if not isinstance(recovery, bool):
        raise SystemExit(f"[strata] config \"tool_call_recovery\" must be true or false, not {recovery!r}")
    svc.tool_call_recovery = recovery
    rs = cfg.get("repeat_stop_tokens", REPEAT_STOP_TOKENS)  # #606
    if isinstance(rs, bool) or not isinstance(rs, int) or rs < 0:
        raise SystemExit(f"[strata] config \"repeat_stop_tokens\" must be a whole number >= 0 (0 = off), not {rs!r}")
    svc.repeat_stop_tokens = rs
    svc.effort_end = bool(effort_end)                   # #458: "effort_position": "end" with an engine that has it
    if cfg.get("reasoning_budget_tokens") is not None:  # #123: a default thinking budget for every request
        try:
            svc.reasoning_budget_tokens = cfg["reasoning_budget_tokens"]
            budget = svc.reasoning_budget({})
        except ValueError as e:
            raise SystemExit(f"[strata] config {e}")
        if budget:
            print(f"[strata] thinking budget: {budget} tokens (reasoning_budget_tokens; a request can set its own)",
                  flush=True)
    recovery = cfg.get("reasoning_loop_recovery", False)   # #728: false (default) | "stop" | "recover" (true)
    if recovery is True:
        recovery = "recover"
    if recovery is not False and recovery not in ("stop", "recover"):
        raise SystemExit("[strata] config \"reasoning_loop_recovery\" must be false, \"stop\" or \"recover\", "
                         f"not {recovery!r}")
    svc.reasoning_loop_recovery = recovery
    svc.backend = cfg.get("backend")                    # "hip": the AMD cards' readings come from sysfs (#301)
    svc.gpu_indices = gpu_list(cfg) if svc.backend == "hip" else monitor_gpus(cfg)      # every card the engine can see (issue #112; UI S0: no "gpu" key); NVML sees no AMD card
    svc.gpu_index = (svc.gpu_indices or [0])[0]         # the Monitor reads the card the engine runs on (issue #51)
    if a.engine == "strata" and groups:                 # replicas: every card of every group
        svc.gpu_indices = [c for g in groups for c in g]
        svc.gpu_index = svc.gpu_indices[0]
    if a.config:
        svc.config_path = a.config                      # #564: the web page's Settings view
    if a.config:                                        # the Chat settings shared with other apps, from last time
        svc.shared_path = str(Path(a.config).with_suffix("")) + ".shared-settings.json"
        try:
            svc.shared = clean_shared_defaults(json.loads(Path(svc.shared_path).read_text(encoding="utf-8")))
            if svc.shared:
                print("[strata] other apps use the Chat settings: " +
                      ", ".join(f"{k}={v}" for k, v in svc.shared.items()), flush=True)
        except (OSError, ValueError):
            svc.shared = {}
    svc.importer = importer
    svc.strata, strata_server.svc = strata_server, svc
    skills_in_use = len(importer.skills._skills)
    if skills_in_use:
        print(f"[strata] {skills_in_use} skill{'s' * (skills_in_use != 1)} imported from your other coding apps for the web app's chat "
              "(switch them off in Settings > Import)", flush=True)
    if hub is not None:
        import atexit
        svc.mcp = hub
        svc.agent = agent_server
        own = [n for n in hub.servers if n != "skills"]
        if own:
            print(f"[strata] starting {len(own)} MCP server{'s' * (len(own) != 1)} for the web app's chat: {', '.join(own)}", flush=True)
        hub.start()
        atexit.register(hub.close)                      # the servers Strata started end with it
    if placeholder is not None:
        placeholder.shutdown()
        placeholder.server_close()
    httpd = serve(svc, host=a.host, port=a.port)
    svc.start_idle_unload()
    svc.start_engine_watch()                            # #59: a dead engine comes back without waiting for a request
    lan = a.host not in ("127.0.0.1", "localhost", "::1")
    for line in start_lines(svc, a.host, a.port, engine.max_context, bool(vision), lan_addresses() if lan else []):
        print(line, flush=True)
    if lan:
        if os.name == "nt":
            print("       nothing arrives? Windows Firewall blocks it until allowed: accept its prompt for Python, or run "
                  "in an admin PowerShell:\n         New-NetFirewallRule -DisplayName \"Strata " + str(a.port) + "\" "
                  "-Direction Inbound -Protocol TCP -LocalPort " + str(a.port) + " -Action Allow -Profile Private\n"
                  "       (and set this network to Private in Windows' network settings)", flush=True)
    if a.open and cfg.get("open_browser") is not False:   # #609: the config's "open_browser": false wins (an older
        import webbrowser                                  # run-<model>.bat still passes --open)
        webbrowser.open(f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '') else a.host}:{a.port}/")
    # #96: docker stop sends SIGTERM, which Python ignores by default, so the container's PID 1 would be killed after
    # the grace period with the engine still running. SIGTERM takes Ctrl+C's path below (QUIT to the engine).
    # SIGINT keeps Python's own handler, so Ctrl+C and a second Ctrl+C work as before.
    def on_sigterm(signum, frame):
        raise KeyboardInterrupt
    try:
        signal.signal(signal.SIGTERM, on_sigterm)
    except (ValueError, OSError, AttributeError):         # not the main thread
        pass
    try:
        while True:
            time.sleep(1)                               # Windows never delivers Ctrl+C to an untimed Event.wait()
    except KeyboardInterrupt:
        print("\n[strata] stopping (Ctrl+C again to end the engine at once) ...", flush=True)
        closers = [svc.runner_hub.close, httpd.shutdown, getattr(engine, "close", None), vision.shutdown if vision else None,
                   hub.close if hub is not None else None]
        for close in filter(None, closers):
            try:
                close()
            except KeyboardInterrupt:                   # a second Ctrl+C: don't wait for the engine to free its memory
                if getattr(engine, "proc", None):
                    engine.proc.kill()
        print("[strata] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
