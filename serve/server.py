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
import collections
import contextlib
import base64
import hashlib
import hmac
import ipaddress
import codecs
import heapq
import itertools
import ctypes
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Protocol
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))   # run as a script (run-<model>.bat) as well as a module
from serve.frontend import (ChatTemplate, Event, OutputParser, anthropic_to_messages,  # noqa: E402
                            images_of, openai_to_messages)
from serve.loop_guard import LoopGuard
from serve import cjk_guard, forced_opening, gguf_info, think_budget  # noqa: E402  (xeno #49 S4, S7 follow-up, S3)
from serve.timing_line import report as timing_report  # noqa: E402  (xeno #49 S5)
from serve import timeline  # noqa: E402  (#33: STRATA_TIMELINE, the server's lanes)
from serve.history import HistoryStore, chunk_stats, prompt_for_keep, request_meta, summary_record, window_rates  # noqa: E402  (xeno UI S3)
from serve import harness, mcp_admin  # noqa: E402
from serve import skills as skills_mod  # noqa: E402
from serve import agent as agent_mod, agent_prompt, agent_run, permissions, shell as shell_mod  # noqa: E402
from serve import checkpoints as checkpoints_mod, files as files_mod, folders as folders_mod, gitview, memory as memory_mod  # noqa: E402
from serve.mcp import McpCancelled, hub_from_config  # noqa: E402
from serve.winjob import contain  # noqa: E402

IM_END = "<|im_end|>"
IMAGE_PAD = "<|image_pad|>"
VISION_START = "<|vision_start|>"
CTX_SLACK = 8               # `strata --serve` rejects prompt + max_new + 8 > context: keep the same margin here
# The live tok/s is a rate over a window, not a mean since the first token: a mean reads ~1/elapsed at the first
# token (the Monitor showed five-digit numbers) and then undershoots for the first second of every answer.
RATE_WINDOW_S = 2.0
RATE_MIN_SPAN_S = 0.25      # younger than this there is no rate yet: the mean so far, with the span floored here


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


class GpuBusy(RuntimeError):
    """The model is unloaded and the GPU has less free VRAM than min_free_vram_mib: something else (a game, another
    model server) is using it, so the engine is not started into the little that is left."""


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
                if "PLE on" in line or "expert arena:" in line:
                    say("arena", f"[strata] loading the experts into RAM ({size}) and locking part of them for the GPU.\n"
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
    keeps its slot, and a new family takes a free slot or the least recently used one."""
    PREFIX = 512

    def __init__(self, n: int):
        self.n = n
        self.families: "collections.OrderedDict[int, int]" = collections.OrderedDict()   # family -> slot, oldest first
        self.lock = threading.Lock()
        self.last: dict = {}                             # family -> its last prompt's ids (for shared())

    def shared(self, ids):
        """How many leading tokens the prompt shares with the last prompt of its family; None for a new family."""
        prev = self.last.get(hash(tuple(ids[:self.PREFIX])))
        if prev is None:
            return None
        n = min(len(prev), len(ids))
        return next((i for i in range(n) if prev[i] != ids[i]), n)

    def pick(self, ids) -> int:
        family = hash(tuple(ids[:self.PREFIX]))
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


class StrataEngine:
    """The resident engine: `strata --serve` reads `GEN <max_new> <ids>` lines and streams `T <id>` lines, then
    `DONE ...`.  Requests are serialized by the service's FIFO, so one pipe is enough.

    Per-request sampling rides the same line as engine-side keys between max_new and the ids
    (`temperature=F top_p=F top_k=N seed=N`, the engine's own spelling).  An absent temperature keeps the
    engine's default, which is greedy; `temperature=0` means the same thing, so it is not forwarded.
    """
    QUIET_S = 10          # how long a quiet engine waits for a line before a heartbeat and a liveness check

    def __init__(self, exe: str, args: list[str], cwd: str | None = None, log: str | None = None,
                 env: dict | None = None):
        self.spawn = (exe, list(args), cwd, log, env)   # to start it again after it died (issue #27)
        paths = {k: v for k, v in zip(args, args[1:]) if k in ("--native", "--pack")}
        self.model_path = paths.get("--native") or paths.get("--pack", "pack/full")
        self.log_path = log
        self.log = open(log, "a", encoding="utf-8") if log else subprocess.DEVNULL
        loading = threading.Event()                     # set once READY: the narrator below stops
        if log:
            threading.Thread(target=narrate_start, args=(log, os.path.getsize(log), args, loading),
                             daemon=True).start()
        self.proc = subprocess.Popen([exe, "--serve", *args], cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, encoding="utf-8", bufsize=1, env=env)
        contain(self.proc)                               # ends with the server, however it ends (Windows)
        self.max_context = 0
        self.unloaded = False            # stopped on purpose (idle unload, POST /unload), not crashed
        self.can_stop = False            # the engine honours a STOP line mid-request (READY <ctx> stop)
        self.last = {}
        self.info = {}                   # INFO key=value facts (engine 0.1.8+): kv, expert slots, ... (Monitor tab)
        self.prefill_tok_s_mean = None
        self.progress = None             # (read, total) prompt tokens while a prompt is read, from PP lines
        try:                             # a ready-made engine's BUILD.json says its version
            self.info["version"] = json.loads((Path(exe).parent / "BUILD.json").read_text()).get("version")
        except (OSError, ValueError):
            self.info["version"] = None
        for line in self.proc.stdout:
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
            raise RuntimeError("the engine exited before it was ready" + (f" (see {log})" if log else ""))
        # (from PR #41, midhatn) a locally built engine can sit next to another release's BUILD.json: engines that
        # report their own version (INFO engine=, 0.1.8+) win, the manifest stays the fallback for older ones
        if self.info.get("engine"):
            self.info["version"] = str(self.info["engine"])
        # the engine's stdout on a thread, so a request can wait with a timeout (heartbeats, cancel checks)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self.lines), daemon=True).start()

    def _pump(self, proc, lines):
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)
        if self.proc is proc:                           # xeno #49 review: a pump left over from before restart()
            self.ended = True                           # its output closed: it is gone, even before the OS says so

    def death_note(self) -> str:
        """Why the engine most likely ended, from the end of its log: its own watchdog (issue #29), else RAM."""
        tail = ""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                tail = f.read().decode("utf-8", "replace")
        except (OSError, TypeError):
            pass
        for line in reversed(tail.splitlines()):
            if "issue #29" in line:
                return ("The engine stopped itself because it had stopped making progress - a hang it caught. Its log "
                        "line: " + line.strip() + " - please report it at github.com/Niko1221/Strata/issues.")
        rc = self.proc.poll()
        last = next((x.strip() for x in reversed(tail.splitlines()) if x.strip().startswith(("strata", "ERR"))), "")
        if rc is not None and rc >= 0 and last:          # it ended by itself: its own last words say why (#215)
            return (f"The engine exited (code {rc}). Its last log line: {last} - if that does not explain it, please "
                    "report it at github.com/Niko1221/Strata/issues with the log.")
        return ("The usual cause is running out of RAM: Linux then ends the biggest program (check: sudo dmesg | "
                "grep -i -E 'killed process|out of memory'); Windows slows down instead. Close other programs or use a "
                "smaller model (Q2_0 / IQ2_XS).")

    def alive(self) -> bool:
        return not getattr(self, "ended", False) and self.proc.poll() is None

    def exit_code(self):
        try:
            return self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return None

    def unload(self):
        """Stop the engine process so its VRAM and RAM go back to the system (idle unload, POST /unload); the next
        request starts it again with restart().  Only between requests: the caller holds the service's fifo."""
        try:
            try:                                        # QUIT first, as close() does: the engine frees its memory
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=20)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                self.proc.terminate()
                self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=20)
        except OSError:
            pass
        self.ended = True
        self.unloaded = True

    def restart(self):
        """Start the engine again (the same command) after it died; the new process has its own line queue."""
        try:
            self.proc.kill()
        except OSError:
            pass
        info = dict(self.info)
        self.ended = False
        self.__init__(*self.spawn)
        self.info = {**info, **self.info}

    def _parse_done(self, line):
        f = line.split()
        self.last = {"generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                     "decode_ms": float(f[4]), "finish": f[5]}
        if len(f) >= 9:                                   # the conversation cache's fields (engine 0.1.3+)
            self.last.update(drafts_accepted=int(f[6]), drafts_offered=int(f[7]), reused=int(f[8]))
        if len(f) >= 11:                                  # decode hit rate fields
            self.last.update(hits=int(f[9]), lookups=int(f[10]))
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
        ck = sampling.get("_ckpt_at")                   # xeno #49 S7 follow-up: an extra prompt checkpoint
        if isinstance(ck, int) and not isinstance(ck, bool) and ck > 0:
            keys += f" ckpt_at={ck}"
        return keys + StrataEngine.ban_key(sampling) + StrataEngine.projection_key(sampling)

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

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        """Yields token ids, and None as a heartbeat every 10 s while the engine is quiet (reading a long prompt):
        the HTTP layer turns it into an SSE comment, which keeps clients' watchdogs calm and notices a client that
        has gone.  A consumer that stops early (or `cancel`) makes the engine STOP, so it does not run to max_new."""
        self.progress = None
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
        try:
            while True:
                try:
                    line = self.lines.get(timeout=self.QUIET_S)
                except queue.Empty:
                    if self.proc.poll() is not None:     # xeno #48: it exited but its output never closed (a
                        done = self.ended = True         # crashed CUDA process can keep the pipe): fail, not wait
                        raise EngineDied(f"the engine stopped unexpectedly (exit code {self.proc.poll()})")
                    if cancel.is_set():
                        return
                    yield None
                    continue
                if line is None:
                    done = True
                    raise EngineDied(f"the engine stopped unexpectedly (exit code {self.exit_code()})")
                if line.startswith("T "):
                    if cancel.is_set():
                        return
                    yield int(line[2:])
                elif line.startswith("PP "):
                    f = line.split()
                    if len(f) >= 3 and f[1].isdigit() and f[2].isdigit():
                        self.progress = (int(f[1]), int(f[2]))             # prompt progress, one per chunk: also a heartbeat (the
                        try:                                              # xeno UI S4: (position, ms since the read began), per chunk
                            self._pp.append((int(f[1]), float(f[3]))) if len(f) >= 4 else None
                        except ValueError:
                            pass
                        self.prefill_tok_s_mean = float(f[4]) if len(f) >= 5 else None
                    if cancel.is_set():                   # lines reset the 10 s wait, so without this a long prompt
                        return                            # would send no keep-alives at all)
                    yield None
                elif line.startswith("STATS "):
                    self._parse_stats(line)
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
                while True:
                    try:
                        line = self.lines.get(timeout=self.QUIET_S)
                    except queue.Empty:
                        if self.proc.poll() is not None:     # xeno #48 in the drain: exited, output never closed
                            self.ended, self.last = True, {}
                            break
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

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


class Vision:
    """The resident image encoder: `strata-vision` (llama.cpp mtmd + the mmproj file) reads `ENC <image> <out>`
    lines and writes each image's embeddings; results are cached by the image's hash, so a conversation that
    sends the same picture again (every turn, with most clients) encodes it once."""

    def __init__(self, cfg: dict, log=None, env: dict | None = None):
        args = [cfg["exe"], "--mmproj", cfg["mmproj"], "--model", cfg["model"]]
        if cfg.get("gpu"):
            args.append("--gpu")
        if cfg.get("threads"):
            args += ["--threads", str(cfg["threads"])]
        if cfg.get("max_tokens"):
            args += ["--max-tokens", str(cfg["max_tokens"])]
        self.dir = Path(tempfile.mkdtemp(prefix="strata-vision-"))
        self.spawn = (args, log, env)                   # to start it again after an unload
        self.stopped = False
        self._start()
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[Path, int]] = {}

    def _start(self):
        args, log, env = self.spawn
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log or subprocess.DEVNULL,
                                     text=True, encoding="utf-8", bufsize=1, env=env)
        contain(self.proc)
        line = self.proc.stdout.readline()
        if not line.startswith("READY"):
            raise RuntimeError("the vision encoder did not start: " + line.strip())
        self.stopped = False

    def alive(self) -> bool:
        return not self.stopped and self.proc.poll() is None

    def unload(self):
        """Stop the encoder process (its VRAM or RAM goes back); the encoded images stay cached on disk."""
        self.close()
        self.stopped = True

    def restart(self):
        """Start the encoder again after an unload (or if it died); the cache of encoded images is kept."""
        try:
            self.proc.kill()
        except OSError:
            pass
        self._start()

    @staticmethod
    def load(source: str) -> bytes:
        if source.startswith("data:"):
            return base64.b64decode(source.split(",", 1)[1])
        if source.startswith(("http://", "https://")):
            req = urllib.request.Request(source, headers={"User-Agent": "strata"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        path = source[7:] if source.startswith("file://") else source
        if path and os.path.isfile(path):
            return Path(path).read_bytes()
        raise ValueError("an image must be a data: URL, an http(s) URL or a local file path")

    @staticmethod
    def normalize(data: bytes) -> bytes:
        """The formats strata-vision's decoder (stb_image) reads pass through; anything else is converted to PNG."""
        if data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n" or data[:2] == b"BM" or \
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

    def encode(self, source: str) -> tuple[Path, int]:
        """-> (embeddings file, number of image tokens)."""
        data = self.normalize(self.load(source))
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            img, out = self.dir / f"{key}.img", self.dir / f"{key}.sve"
            img.write_bytes(data)
            self.proc.stdin.write(f"ENC {img} {out}\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline().strip()
            img.unlink(missing_ok=True)
            if not line.startswith("OK"):
                raise ValueError("the image could not be read: " + (line[4:] if line.startswith("ERR") else
                                                                     "the vision encoder stopped"))
            self.cache[key] = (out, int(line.split()[1]))
            if len(self.cache) > 64:                                   # oldest first
                old = next(iter(self.cache))
                self.cache.pop(old)[0].unlink(missing_ok=True)
            return self.cache[key]

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


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


def engine_args(cfg: dict) -> list[str]:
    """The engine's arguments: the config's, and with several GPUs the layer split across them ("layer_split" in the
    config: "auto" by default, or the first layer of each later GPU's share, e.g. "18" or "16,32")."""
    args = list(cfg["args"])
    if len(gpu_list(cfg)) > 1 and "--layer-split" not in args:
        args += ["--layer-split", str(cfg.get("layer_split") or "auto")]
    return args


def child_env(cfg: dict) -> dict:
    """The engine's environment: the CUDA libraries setup installed (pip's nvidia packages, or the toolkit that
    compiled it) first on the library search path."""
    env = dict(os.environ)
    if gpu_list(cfg) and cfg.get("backend") == "hip":   # AMD: numbered as HIP numbers them (setup's KFD order)
        env["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_list(cfg))
    elif gpu_list(cfg):                              # issue #51: the GPU(s) to run on, numbered as nvidia-smi does; CUDA's
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # own default order (fastest first) can number the cards otherwise
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_list(cfg))
    for k, v in (cfg.get("env") or {}).items():      # engine settings the config carries (AMD: the GEMM tuning table)
        env[str(k)] = str(v)
    dirs = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    if dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    return env


class ByteTokenizer:
    """Tiny stand-in tokenizer for tests without the pack: one id per UTF-8 byte, specials as ids >= 256."""
    SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]

    def encode(self, text, parse_special=False):
        out, i = [], 0
        while i < len(text):
            for k, s in enumerate(self.SPECIALS):
                if parse_special and text.startswith(s, i):
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

    def push(self, t: int) -> str:
        if self.inc is not None:
            return self.inc.decode(self.tok.token_bytes(t))
        self.ids.append(t)
        text = self.tok.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta, self.sent = text[self.sent:], len(text)
        return delta


class StopSequenceFilter:
    """Hold only the suffix that might become a stop sequence on the next delta."""

    def __init__(self, sequences):
        if not isinstance(sequences, list) or any(not isinstance(s, str) or not s for s in sequences):
            raise ValueError("stop_sequences must be a list of non-empty strings")
        self.sequences = sequences
        self.pending = ""

    def feed(self, text):
        self.pending += text
        hits = [(self.pending.find(s), -len(s), s) for s in self.sequences if s in self.pending]
        if hits:
            pos, _, sequence = min(hits)
            visible = self.pending[:pos]
            self.pending = ""
            return visible, sequence
        hold = max((n for s in self.sequences for n in range(1, min(len(s), len(self.pending)) + 1)
                    if self.pending.endswith(s[:n])), default=0)
        visible = self.pending[:-hold] if hold else self.pending
        self.pending = self.pending[-hold:] if hold else ""
        return visible, None

    def finish(self):
        visible, self.pending = self.pending, ""
        return visible


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

    def acquire(self, priority: int = 1, blocking: bool = True) -> bool:
        """`blocking=False` (upstream #208's idle unload, which assumed a Lock): take the gate only when it is free
        and no request waits for it; False otherwise, and nothing is queued."""
        if not blocking:
            with self._cv:
                if self._busy or self._waiting:
                    return False
                self._busy = True
                return True
        ticket = (priority, next(self._arrival))
        with self._cv:
            heapq.heappush(self._waiting, ticket)
            while self._busy or self._waiting[0] != ticket:
                self._cv.wait()
            heapq.heappop(self._waiting)
            self._busy = True
        return True

    def release(self):
        with self._cv:
            self._busy = False
            self._cv.notify_all()

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


class Service:
    def __init__(self, engine: Engine, tokenizer, template: ChatTemplate, model_name: str = "qwen3.8-flash-next",
                 vision: Vision | None = None, sampling_defaults: dict | None = None,
                 fit_max_tokens: bool = False):
        self.engine, self.tok, self.template, self.model, self.vision = engine, tokenizer, template, model_name, vision
        self.efforts = template.efforts() if hasattr(template, "efforts") else {"levels": [], "default": None, "off": False}
        self.fit_max_tokens = fit_max_tokens          # --fit-max-tokens: clamp the output cap instead of 400
        self.sampling_defaults = dict(sampling_defaults or {})   # the run config's `sampling` block
        self.restarting = False         # xeno #49 review: the engine is loading again (requests get 529 meanwhile)
        self.cjk_ban = False            # xeno #49 S4: the engine was started with the Han ban list (--ban-ids)
        self.shared = {}                              # the web app's Chat settings for every client (POST /settings)
        self.shared_path = None                       # where they are kept between starts (next to the config)
        self.fifo = RequestGate()
        self.embeddings = threading.local()           # the current request's image embeddings file (GENI)
        self.api_key = ""                              # when set, /v1/* needs it (Bearer or x-api-key)
        self.allowed_hosts: set = set()                # names besides this PC's own that may be used as Host when there is no key (config "allowed_hosts")
        self.status = {"busy": False, "queued": 0, "loops_stopped": 0}  # GET /status: what the model is doing right now
        self.rate = collections.deque(maxlen=32)        # (time, generated) samples for the live tok/s window
        self.history = collections.deque(maxlen=500)    # the last finished requests, newest last (GET /metrics)
        self.hstore = HistoryStore(Path(tempfile.gettempdir()) / "strata-history-off", enabled=False)  # main() turns it on
        self.checkpoints = checkpoints_mod.Checkpoints(Path(tempfile.gettempdir()) / "strata-checkpoints")      # the way back for files the tools change; main() puts it in the user's data folder
        self.model_info = None                          # the model's name and quantization, from its GGUF headers (main() fills it)
        self.keep_prompts = 0                           # POST /metrics/keep: the next N requests keep their full prompt (Q8)
        self.keep_lock = threading.Lock()
        self.ui = "next"                                # which web app "/" serves: the new one; config "ui": "classic" brings the old one back
        # since the server started (the Monitor's totals, issue #35)
        self.totals = {"since": time.time(), "requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0,
                       "prompt_ms": 0.0, "decode_ms": 0.0}
        self.last_timings = None                         # the last finished request's, llama.cpp's names (/v1/status)
        self.last_request_at = None                      # when a request last started or finished
        self.started_at = time.time()
        self.status_lock = threading.Lock()
        self.mcp = None                                  # serve/mcp.py's McpHub when MCP servers are configured
        self.config_path = None                          # the run config (--config) and the --mcp-config file: where serve/mcp_admin.py
        self.mcp_config_path = None                      # reads and writes the servers the web app sets up
        self.importer = None                             # serve/harness.py's Importer: the other coding apps' skills and MCP servers (#94)
        self.agent = None                                # serve/agent.py's AgentServer: the chat's coding tools (Read, Write, Edit, Bash, ...); None: switched off
        self.broker = agent_run.Broker()                 # the questions the coding tools have asked the page and not had answered yet
        # sharing the GPU with other programs (all off by default): unload the engine after this many idle seconds,
        # only start it again when this much VRAM is free, and run this command first (e.g. to unload another
        # server's model); the next request after an unload starts the engine again
        self.idle_unload_s = 0
        self.min_free_vram_mib = 0
        self.before_load = None
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
        req = self.with_slot(req, ids)
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

    def start_agent_run(self, sa: dict, messages: list):
        """One request's use of the coding tools: the folder, mode, rules and chat the page named (anything odd is ignored), the rules for the
        AI as a prompt, and auto mode's judge (serve/agent_run.py)."""
        raw = sa.get("cwd")
        folder = os.path.realpath(raw) if isinstance(raw, str) and raw.strip() and "\0" not in raw and os.path.isdir(raw) else None
        dirs: list[str] = []                                            # the project's other folders: the real ones, once each, not the main one; none without a main one
        for d in sa.get("dirs") if folder and isinstance(sa.get("dirs"), list) else []:
            if isinstance(d, str) and d.strip() and "\0" not in d and os.path.isdir(d):
                r = os.path.realpath(d)
                if os.path.normcase(r) != os.path.normcase(folder) and os.path.normcase(r) not in [os.path.normcase(x) for x in dirs]:
                    dirs.append(r)
        dirs = dirs[:agent_mod.MAX_DIRS]
        mode = sa.get("mode") if sa.get("mode") in ("auto", "plan") else None
        rules = lambda v: [x for x in v if isinstance(x, str) and 0 < len(x) <= 500][:200] if isinstance(v, list) else []      # noqa: E731
        sid = sa.get("session") if isinstance(sa.get("session"), str) and 0 < len(sa["session"]) <= 80 else "default"
        goal = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                goal = c if isinstance(c, str) else "".join(p.get("text", "") for p in c if isinstance(p, dict)) if isinstance(c, list) else ""
                break
        sh = getattr(self.agent, "shell", None)
        policy = permissions.Policy(cwd=folder, mode=mode, allow=rules(sa.get("allow")), deny=rules(sa.get("deny")), dirs=dirs)
        run = agent_run.AgentRun(policy, sid, self.broker, goal, self.side_request, threading.Event(), shell=shell_mod.describe(sh) if sh else None)
        cp = sa.get("checkpoint")
        if isinstance(cp, str) and sid != "default":
            run.ctx.checkpoint = self.checkpoints.scope(sid, cp)                  # this prompt's checkpoint: the files the tools change are kept as they were
        git, d = False, folder
        for _ in range(6):                                              # the folder or one of the folders above it holds .git
            if not d:
                break
            if os.path.exists(os.path.join(d, ".git")):
                git = True
                break
            parent = os.path.dirname(d)
            d = parent if parent != d else None
        tools = [t["name"] for t in self.agent.tools if t["name"] != "ExitPlanMode" or mode == "plan"]
        run.prompt = agent_prompt.build(folder, shell_mod.describe(sh) if sh else None, mode, time.strftime("%Y-%m-%d"), sys.platform, git, None, tools, dirs, self.memory_blocks([folder, *dirs] if folder else []))
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

    def _vision_down(self) -> bool:
        return self.vision is not None and hasattr(self.vision, "alive") and not self.vision.alive()

    def free_vram_mib(self) -> int | None:
        """Free VRAM on the engine's (first) GPU, from NVML; None when it can't be read (then nothing is refused)."""
        try:
            from serve.telemetry import _Nvml
            nv = _Nvml(int(getattr(self, "gpu_index", 0) or 0))
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
        if self._vision_down():                         # first, as at a start: a GPU encoder takes its VRAM before
            print("[strata] starting the vision encoder again ...", flush=True)   # the engine sizes its cache
            self.vision.restart()
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
        except EngineDied:
            raise
        except (RuntimeError, OSError) as e:          # xeno #49 review: a 529, not a dropped connection
            raise EngineDied(f"the engine could not be started again: {e}") from e
        finally:
            self.restarting = False
        print("[strata] the engine is running again", flush=True)

    def load(self):
        """POST /load and every generation request: start the engine now if it is unloaded (raises GpuBusy)."""
        # a request is on its way: the idle thread must not unload between this and the request's own start
        self.last_request_at = time.time()
        if self.loaded() and not self._vision_down():
            return
        if self.restarting:            # xeno #49 S2: a request while the engine loads again is a 529 at once
            raise EngineDied("the engine is loading again after it stopped")
        with self.fifo:
            self.ensure_loaded()

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

    def start_idle_unload(self):
        if not self.idle_unload_s or not hasattr(self.engine, "unload"):
            return
        print(f"[strata] the model unloads after {self.idle_unload_s} s without requests", flush=True)

        def loop():
            while True:
                time.sleep(max(1.0, min(30.0, self.idle_unload_s / 4)))
                self.unload(idle_for=self.idle_unload_s)
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

    def with_slot(self, req: dict, ids) -> dict:
        """The request with the cache slot of its prompt family (xeno #49 S7), when the engine has several slots
        (INFO cache_slots) and the client chose none."""
        if "strata_cache_slot" in req:
            return req
        try:
            n = int((getattr(self.engine, "info", {}) or {}).get("cache_slots", 1))
        except (TypeError, ValueError):
            n = 1
        if n <= 1:
            return req
        if getattr(self, "slots", None) is None or self.slots.n != n:
            self.slots = CacheSlots(n)
        shared = self.slots.shared(ids)
        slot = self.slots.pick(ids)
        if os.environ.get("STRATA_DEBUG"):              # where the prompt leaves its family's last one, and its turns
            start = self.tok.encode("<|im_start|>", parse_special=True)
            turns = [i for i, t in enumerate(ids) if len(start) == 1 and t == start[0]]
            print(f"[strata] slot {slot}: {len(ids)} tokens, shares {shared} with the family's last, "
                  f"turns at {turns[:3]}{'...' if len(turns) > 6 else ''}{turns[-3:] if len(turns) > 3 else ''}",
                  flush=True)
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
        if not s:
            return req
        req = dict(req)
        if "max_tokens" in s and not req.get("max_tokens") and not req.get("max_completion_tokens"):
            req["max_tokens"] = s["max_tokens"]
        effort = s.get("reasoning_effort")
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

    def meta_for(self, dialect, messages, tools, user_agent) -> dict:
        """The request's history meta; while "keep the next N prompts" is on it also carries the full prompt (`_prompt`,
        which run() moves to the detail file and never into the summary row)."""
        meta = request_meta(dialect, messages, tools, user_agent)
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
                                       model_path=getattr(self.engine, "model_path", None))

    def _tok_s(self):
        """tok/s over the last RATE_WINDOW_S seconds.  Returns 0.0 while nothing is generating."""
        with self.status_lock:
            s = dict(self.status)
            rate = list(self.rate)
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
            s = dict(self.status)
        if not s.get("busy") or not s.get("first_token"):
            return 0.0
        return s["generated"] / max(1e-6, time.time() - s["first_token"])

    def _prefill_tok_s_mean(self):
        """Engine-reported mean over newly read tokens, excluding the cached prefix."""
        with self.status_lock:
            reading = self.status.get("busy") and self.status.get("first_token") is None
        return getattr(self.engine, "prefill_tok_s_mean", None) if reading else 0.0

    def metrics(self, all_requests=False) -> dict:
        """GET /metrics: what the Monitor tab shows - the engine's facts, what it is doing, the last requests, and
        the hardware (with a minute of history per series)."""
        with self.status_lock:
            s = dict(self.status)
            hist = list(self.history)
            totals = dict(self.totals)
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
                "tok_s_window_s": RATE_WINDOW_S if state == "generating" else None}
        if state == "reading" and progress:
            live["prompt_read"], live["prompt_total"] = progress
        engine = {"model": self.model, "max_context": self.engine.max_context, "images": self.vision is not None,
                  "efforts": (["none"] if self.efforts["off"] else []) + self.efforts["levels"],      # what the model's template accepts
                  "effort_default": self.efforts["default"],
                  **dict(getattr(self.engine, "info", {}) or {})}
        tel = self.telemetry.snapshot() if getattr(self, "telemetry", None) else {"now": {}, "history": {}, "static": {}}
        return {"engine": engine, "live": live, "requests": hist[::-1][:None if all_requests else 12],
                "requests_kept": len(hist), "totals": totals, "hardware": tel["now"],
                "hardware_static":
                tel["static"], "history": tel["history"], "time": now, "keep_prompts_left": self.keep_prompts,
                "model_info": self.model_info}

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

        busy, ctx = bool(s.get("busy")), self.engine.max_context
        images = self.vision is not None
        return {
            "service": "strata", "model": self.model,
            "engine": (getattr(self.engine, "info", {}) or {}).get("version"),
            "started": int(self.started_at), "uptime_s": int(time.time() - self.started_at),
            "cache_max_tokens": ctx,
            "context": {"native": ctx, "max_positions": ctx},
            "concurrency": {"serving": 1, "requested": 1},       # one request at a time; more wait their turn
            "dialects": ["/v1/chat/completions", "/v1/messages"],
            "vision": {"enabled": images, "available": images, "error": None},
            "activity": {"requests": totals["requests"] + int(busy), "in_flight": int(busy) + int(s.get("queued") or 0),
                         "last_request_at": int(last_at) if last_at else None},
            "last_timings": last_t,
            "machine": {
                "at": int(time.time()),
                "gpu": {"name": static.get("gpu_name"), "used_mib": scaled(hw.get("gpu_mem_used"), 2 ** 20),
                        "total_mib": scaled(hw.get("gpu_mem_total"), 2 ** 20), "util_pct": hw.get("gpu_util"),
                        "temp_c": hw.get("gpu_temp"), "power_w": hw.get("gpu_power")} if static.get("gpu_name") else None,
                "ram": {"used_gib": scaled(hw.get("ram_used"), 2 ** 30, 1),
                        "total_gib": scaled(hw.get("ram_total"), 2 ** 30, 1)} if hw.get("ram_total") else None}}

    def prepare(self, messages, tools, kwargs, max_new=None, priority: int = 1):
        """-> (ids, thinking, max_new). An unset or non-positive max_new (some clients send -1) means "unlimited":
        the rest of the context."""
        t_tl = timeline.now_us()
        prompt = self.template.render(messages, tools=tools, **kwargs)
        ids = self.tok.encode(prompt, parse_special=True)
        timeline.complete("template+tokenize", t_tl, timeline.now_us(), len(ids))
        self.embeddings.path = None
        images = images_of(messages)
        if images:
            if self.vision is None:
                raise ValueError("this server was started without the vision encoder (run setup again and choose "
                                 "'vision'), so it cannot read images")
            pad = self.tok.encode(IMAGE_PAD, parse_special=True)[0]
            start = self.tok.encode(VISION_START, parse_special=True)[0]
            # Encode only while the engine is idle: the engine and the image encoder (a separate process) must not
            # run on the GPU at the same time - an encode during a running request left that request stuck at
            # "reading the prompt" with CPU and GPU busy, for good (reproduced).  So encoding takes its turn in the
            # same FIFO as the requests.
            with self.fifo.slot(priority):              # xeno #49 review: at the request's own priority (S6)
                encoded = [self.vision.encode(src) for src in images]
            # one <|image_pad|> per image -> one per image token.  Only the markers the template writes for an image
            # (right after <|vision_start|>) are images: the same text inside a message (an agent reading these docs,
            # #150) is kept as plain text, or it took an image's place and the counts no longer matched.
            literal = self.tok.encode(IMAGE_PAD, parse_special=False)
            out, k = [], 0
            for j, t in enumerate(ids):
                if t == pad and j > 0 and ids[j - 1] == start and k < len(encoded):
                    out += [pad] * encoded[k][1]
                    k += 1
                elif t == pad:
                    out += literal
                else:
                    out.append(t)
            if k != len(encoded):
                raise ValueError("the prompt and its images do not match")
            ids = out
            combined = self.vision.dir / f"req-{uuid.uuid4().hex[:12]}.sve"
            try:
                with open(combined, "wb") as f:
                    for path, _ in encoded:
                        f.write(path.read_bytes())
            except Exception:
                combined.unlink(missing_ok=True)
                raise
            self.embeddings.path = combined
        if max_new is None:                  # count only (POST /v1/messages/count_tokens): no room check
            return ids, kwargs.get("enable_thinking", True) is not False, None
        room = self.engine.max_context - CTX_SLACK - len(ids)
        if max_new <= 0 or (self.fit_max_tokens and room < 1):
            if room < 1:
                raise ValueError(f"prompt ({len(ids)} tokens) leaves no room to answer in the context "
                                 f"({self.engine.max_context}); requests are never truncated")
            max_new = room
        elif max_new > room:
            if not self.fit_max_tokens:
                raise ValueError(f"prompt ({len(ids)} tokens) + max tokens ({max_new}) exceeds the context "
                                 f"({self.engine.max_context}); requests are never truncated")
            max_new = max(1, room)          # --fit-max-tokens: a shorter completion beats a 400
        return ids, kwargs.get("enable_thinking", True) is not False, max_new

    def _engine(self, ids, max_new, sampling, cancel, emb):
        return self.engine.generate(ids, max_new, sampling, cancel, embeddings=emb) if emb else \
            self.engine.generate(ids, max_new, sampling, cancel)

    def _generate(self, ids, max_new, sampling, cancel, emb, state):
        """The engine's tokens, with the thinking budget (xeno #49 S3, serve/think_budget.py): once `budget` tokens
        are out while state["thinking"] holds, stop the engine, yield CLOSE's tokens as if the model wrote them and
        continue from (prompt + written + CLOSE), which reuses the engine's cached prefix."""
        opening = (sampling or {}).get("_opening")      # xeno (serve/forced_opening.py): written for the model,
        if opening:                                     # which continues from it
            head = self.tok.encode(opening)
            yield from head
            ids, max_new = list(ids) + head, (max(1, max_new - len(head)) if max_new else max_new)
        budget = (sampling or {}).get("_think_budget")
        gen = self._engine(ids, max_new, sampling, cancel, emb)
        if not budget:
            yield from gen
            return
        written = []
        try:
            for t in gen:
                yield t
                if t is not None:
                    written.append(t)
                    if len(written) >= budget and state["thinking"]:
                        break
            else:
                return
        finally:
            gen.close()
        if cancel.is_set():
            return
        # the engine committed a few tokens past the cut before it read STOP: they are part of the answer (and of
        # its live session, whose whole length must be a prefix of the next prompt for the cache to be reused)
        for t in list(getattr(self.engine, "drained", None) or []):
            written.append(t)
            yield t
        first = dict(getattr(self.engine, "last", None) or {})
        close = self.tok.encode(think_budget.CLOSE, parse_special=True)
        print(f"[strata] thinking budget spent ({budget} tokens): closing the thinking block", flush=True)
        yield from close
        rest = max(1, max_new - len(written) - len(close)) if max_new else max_new
        try:
            yield from self._engine(list(ids) + written + close, rest, sampling, cancel, emb)
        finally:
            self._merge_legs(first, len(close))

    def _merge_legs(self, first: dict, n_close: int):
        """One request's DONE figures from its two engine calls (xeno #49 review): the prompt read is the first
        call's; re-reading the written tokens + CLOSE is part of making the output, so it counts as decode time."""
        second = dict(getattr(self.engine, "last", None) or {})
        if not first or not second:
            return
        merged = dict(first)
        merged["decode_ms"] = first.get("decode_ms", 0.0) + second.get("prompt_ms", 0.0) + second.get("decode_ms", 0.0)
        merged["generated"] = first.get("generated", 0) + n_close + second.get("generated", 0)
        for k in ("drafts_accepted", "drafts_offered", "hits", "lookups"):
            if k in first or k in second:
                merged[k] = first.get(k, 0) + second.get(k, 0)
        self.engine.last = merged

    def _note(self, n, evs):
        with self.status_lock:
            s = self.status
            s["generated"] = n
            if s.get("first_token") is None:
                s["first_token"] = time.time()
            self.rate.append((time.time(), n))          # the live rate's window over the last RATE_WINDOW_S
            for ev in evs:
                if ev.kind == "reasoning":
                    s["phase"] = "thinking"
                elif ev.kind == "content":
                    s["phase"] = "answering"
                elif ev.kind == "tool_start":
                    s["phase"], s["tool"] = f"writing a tool call: {ev.call.name}", ev.call.name
                elif ev.kind == "tool_call":
                    s["phase"] = "tool call complete"
                s["tail"] = (s["tail"] + (ev.text or ""))[-600:]

    def _progress(self, last_print, every=1.0):
        """A progress line in the server window every `every` seconds while a request runs."""
        now = time.time()
        if now - last_print < every:
            return last_print
        with self.status_lock:
            s = dict(self.status)
        el = now - s.get("started", now)
        if s.get("first_token") is None:
            pr = getattr(self.engine, "progress", None)   # (position reached, prompt tokens): a reused prefix counts
            done = f"{pr[0]:,} of {pr[1]:,}" if pr and pr[1] else f"{s.get('prompt_tokens', 0):,}"   # as read (#29)
            print(f"[strata] reading the prompt: {done} tokens, {el:.0f} s so far", flush=True)
        else:
            rate = s["generated"] / max(1e-6, now - s["first_token"])
            print(f"[strata] {s['phase']}: {s['generated']} of max {s.get('max_tokens')} tokens, {rate:.1f} tok/s, "
                  f"{el:.0f} s", flush=True)
        return now

    def run(self, ids, thinking, tools, max_new, sampling, cancel) -> Iterator[tuple[str, object]]:
        """Yields ("event", Event) as text arrives, then ("done", {"finish": .., "completion_tokens": ..})."""
        defaults = {**self.sampling_defaults, **self.shared}   # the config's, then the Chat settings shared with apps
        if defaults:                   # the request's own fields win (explicit 0 stays greedy)
            req_values = {k: v for k, v in (sampling or {}).items() if v is not None}
            sampling = {**defaults, **req_values}
        parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True)
        guard = LoopGuard()
        stop_filter = StopSequenceFilter(sampling["stop_sequences"]) if sampling.get("stop_sequences") else None
        matched_sequence = None
        detok, n, finish = Detokenizer(self.tok), 0, "length"
        timings, before = None, None                    # this request's timings; the engine's `last` before it
        stop_detail = None
        raw_ids = []                                    # every generated id (STRATA_DEBUG: dump raw model text)
        tok_times = []                                  # when each token reached the server
        emb = getattr(self.embeddings, "path", None)
        if self.restarting:            # xeno #49 S2: the native API's answer while the model loads - retry later
            raise EngineDied("the engine is loading again after it stopped")
        # Identity token: only a DONE line replaces engine.last, so a request that died, errored or was
        # disconnected must not have the PREVIOUS request's decode figures recorded as its own.
        engine_last0 = getattr(self.engine, "last", None)
        with self.status_lock:
            self.status["queued"] += 1
        t_queue = timeline.now_us()
        try:
            with self.fifo.slot(0 if (sampling or {}).get("stream") else 1):   # xeno #49 S6: main turn first
                timeline.complete("queue wait", t_queue, timeline.now_us())
                with self.status_lock:
                    self.status["queued"] -= 1
                # issue #27: it died in an earlier request (or was unloaded) - start it again instead of failing
                self.ensure_loaded()
                with self.status_lock:
                    self.status.update(busy=True, phase="reading the prompt", prompt_tokens=len(ids), generated=0,
                                       started=time.time(), first_token=None, tool=None, tail="", max_tokens=max_new)
                    self.last_request_at = time.time()
                    self.rate.clear()               # the previous request's samples must not leak into this one
                before = getattr(self.engine, "last", None)
                last_print = time.time()
                t_engine = timeline.now_us()
                state = {"thinking": thinking}         # still inside the thinking block (the budget's cut)
                if hasattr(self.engine, "last"):
                    self.engine.last = {}              # this request's DONE only, never the previous one's
                gen = self._generate(ids, max_new, sampling, cancel, emb, state)
                try:
                    for t in gen:
                        if t is None:                   # heartbeat while the engine is quiet
                            last_print = self._progress(last_print)
                            yield "ping", None
                            continue
                        n += 1
                        tok_times.append(time.monotonic())      # xeno UI S3: the decode speed over a sliding window
                        if n == 1:
                            timeline.instant("first token", len(ids))
                        if t in self.stop_ids:
                            finish = "stop"
                            raw_ids.append(t)
                            break
                        raw_ids.append(t)
                        evs = parser.feed(detok.push(t))
                        state["thinking"] = parser.state == "reasoning"   # not from events: '</think>' emits none
                        self._note(n, evs)
                        last_print = self._progress(last_print)
                        for ev in evs:
                            if ev.kind in ("reasoning", "content") and guard.feed(
                                    ev.text, in_think=ev.kind == "reasoning"):
                                cancel.set()
                                finish = "length"
                                stop_detail = "loop"
                                with self.status_lock:
                                    self.status["loops_stopped"] += 1
                                    self.status["last_stop_reason"] = "loop"
                                print(f"[strata] loop guard stopped generation: {guard.reason}", flush=True)
                                break
                            if ev.kind != "content" and stop_filter:
                                tail = stop_filter.finish()
                                if tail:
                                    yield "event", Event("content", text=tail)
                            if ev.kind == "content" and stop_filter:
                                visible, matched_sequence = stop_filter.feed(ev.text)
                                if visible:
                                    yield "event", Event("content", text=visible)
                                if matched_sequence:
                                    cancel.set()
                                    finish = "stop"
                                    break
                                continue
                            yield "event", ev
                        if guard.reason or matched_sequence:
                            break
                    # the loop guard and a stop sequence cancel the engine themselves; only a client cancel is one
                    if cancel.is_set() and not matched_sequence and not guard.reason:
                        finish = "stop" if getattr(cancel, "server_stop", False) else "cancel"
                except EngineDied as e:
                    finish = "error"
                    note = self.engine.death_note() if hasattr(self.engine, "death_note") else ""
                    print(f"[strata] {e}. {note} The next request starts the engine again."
                          f"{' Its log: ' + self.engine.log_path if getattr(self.engine, 'log_path', None) else ''}",
                          flush=True)
                    raise
                except ValueError as e:                 # the engine's ERR line (it may have ended after it)
                    finish = "error"
                    print(f"[strata] the engine reported an error: {e}", flush=True)
                    raise
                finally:
                    gen.close()                         # STOP+drain to THIS request's DONE while still holding the
                    #                                     fifo, so a stop-token break can't leave the shared engine
                    #                                     queue mid-drain for the next request to read as its own DONE
                    timeline.complete("engine request", t_engine, timeline.now_us(), len(ids), n)
        except GeneratorExit:                           # the client disconnected mid-stream
            finish = "disconnect"
            raise
        finally:
            if emb:
                Path(emb).unlink(missing_ok=True)
            to_disk = None
            with self.status_lock:
                if self.status.get("busy"):
                    # only this request's DONE counts: same object means no DONE arrived (death, error, disconnect)
                    last = dict(getattr(self.engine, "last", {}) or {}) \
                        if getattr(self.engine, "last", None) is not engine_last0 else {}
                    started = self.status.get("started", time.time())
                    loaded = str((getattr(self.engine, "info", {}) or {}).get("cvec", 0)) not in ("0", "", "None")
                    hit_rate = round(last["hits"] / last["lookups"], 3) if last.get("lookups") else None
                    meta = dict((sampling or {}).get("_meta") or request_meta("", [], None, None))
                    # one request can call run() several times (an MCP tool round, the thinking budget's second call):
                    # the first keeps the request's id, the later ones get -2, -3 ... so no row or detail overwrites another
                    shared = (sampling or {}).get("_meta")
                    if shared is not None:
                        calls = shared["calls"] = shared.get("calls", 0) + 1
                        if calls > 1:
                            meta["id"] = f"{shared['id']}-{calls}"
                    meta.pop("calls", None)
                    prompt = meta.pop("_prompt", None)                  # Q8: only when asked for; the detail file, never the row
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
                        "prompt_tokens": len(ids), "reused": last.get("reused"), "output_tokens": n,
                        "engine_generated": last.get("generated"), "stats": last.get("stats"),
                        "prompt_ms": last.get("prompt_ms"), "decode_ms": last.get("decode_ms"),
                        "decode_tok_s": round(last["generated"] / (last["decode_ms"] / 1000), 1)
                        if n and last.get("generated") and last.get("decode_ms") else None,
                        "cjk_chars": cjk_guard.count_han(self.tok.decode(raw_ids)) if raw_ids else 0,   # #49 S4
                        "hit_rate": hit_rate})
                    self.history.append(rec)
                    to_disk = (rec, {"prefill_chunks": chunks["items"] if chunks else [],
                                     "decode_series": decode["series"] if decode else [],
                                     "stats": last.get("stats"),
                                     **({"prompt": prompt} if prompt is not None else {})})   # written after the lock is let go
                    t = self.totals
                    t["requests"] += 1
                    t["prompt_tokens"] += len(ids)
                    t["reused"] += last.get("reused") or 0
                    t["output_tokens"] += n
                    t["prompt_ms"] += last.get("prompt_ms") or 0.0
                    t["decode_ms"] += last.get("decode_ms") or 0.0
                    fresh = getattr(self.engine, "last", None)
                    if fresh is not None and fresh is not before:      # the engine's clock for THIS request
                        timings = request_timings(len(ids), n, last)
                        self.last_timings = dict(timings, at=int(time.time())) if timings else None
                    self.last_request_at = time.time()
                    now = time.time()
                    el = now - self.status.get("started", now)
                    ft = self.status.get("first_token")
                    rate = n / max(1e-6, now - ft) if ft else 0.0
                    hit_msg = f", expert cache {hit_rate*100:.1f}% hit" if hit_rate is not None else ""
                    print(f"[strata] done: {n} tokens in {el:.0f} s ({rate:.1f} tok/s) "
                          f"({finish}, cancel={cancel.is_set()}){hit_msg}", flush=True)
                    if finish != "error" and last.get("prompt_ms") is not None:   # xeno #49 S5
                        for line in timing_report(last, len(ids), el):
                            print(line, flush=True)
                    if os.environ.get("STRATA_DEBUG") and raw_ids:
                        print(f"[strata] raw: {ascii(self.tok.decode(raw_ids))}", flush=True)   # xeno: never a UnicodeEncodeError
                self.status["busy"] = False
                self.status["last_stop_reason"] = stop_detail or finish
                self.status.pop("tail", None)            # #212: the answer's end is not kept once it is done
                self.status.pop("tool", None)
            if to_disk is not None:                     # on disk, kept: outside status_lock (it is /status's and /metrics's lock),
                try:                                    # and a full disk must not fail the request
                    self.hstore.append(to_disk[0])
                    self.hstore.write_detail(to_disk[0]["id"], to_disk[1])
                except OSError as e:
                    print(f"[strata] history not saved: {e}", flush=True)
        if not matched_sequence:
            for ev in parser.finish():
                if ev.kind == "content" and stop_filter:
                    visible, matched_sequence = stop_filter.feed(ev.text)
                    if visible:
                        yield "event", Event("content", text=visible)
                    if matched_sequence:
                        break
                else:
                    if stop_filter:
                        tail = stop_filter.finish()
                        if tail:
                            yield "event", Event("content", text=tail)
                    yield "event", ev
            if stop_filter and not matched_sequence:
                tail = stop_filter.finish()
                if tail:
                    yield "event", Event("content", text=tail)
        if matched_sequence:
            finish = "stop"
            with self.status_lock:
                self.status["last_stop_reason"] = "stop_sequence"
        done = {"finish": finish, "completion_tokens": n, "reused": (timings or {}).get("cache_n", 0), "timings": timings}
        if matched_sequence:
            done["stop_sequence"] = matched_sequence
        if stop_detail:
            done["stop_detail"] = stop_detail
        yield "done", done


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
                 mcp_names, agent_run=None):
    """Service.run with the MCP tools executed here: the model writes a call to an MCP tool, the server runs it, adds
    the call and its result to the conversation and lets the model continue - up to `max_rounds` times.  Yields what
    Service.run yields (text, thinking, the request's own tool calls) plus ("mcp", {...}) for the tool activity, and
    one ("done", ...) at the very end with the output tokens of every round.

    `mcp_names`: the MCP tools this request offered; any other call is one of the request's own tools and ends the
    turn as always (the client answers it).  MCP calls written in the same answer are then not run (their results
    could not reach the model before the client's)."""
    max_rounds = max(int(hub.settings["max_rounds"]), AGENT_ROUNDS) if agent_run is not None else int(hub.settings["max_rounds"])      # coding takes many rounds
    total, rounds, done = 0, 0, None
    messages = list(messages)
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
        run_them = calls and not own_calls and done["finish"] == "stop" and not cancel.is_set()
        if run_them and rounds >= max_rounds:
            yield "mcp", {"event": "limit", "max_rounds": max_rounds}
            run_them = False
        if not run_them:
            for c in calls:                              # announced, never run: close them in the client's view
                yield "mcp", {"event": "result", "id": c.id, "ok": False, "skipped": True, "text": "not run",
                              "chars": 0, "truncated": False, "ms": 0}
            break
        rounds += 1
        results = []
        for c in calls:
            s, tool = hub.routes().get(c.name, (None, c.name))
            yield "mcp", {"event": "call", "id": c.id, "name": c.name, "server": s.name if s else None,
                          "tool": tool, "arguments": c.arguments, "round": rounds}
            # The call runs on a thread while this generator keeps yielding heartbeats: they reach the client as
            # keep-alives, which is how a client that went away (the web app's Stop) is noticed during a slow tool.
            box = {}
            if agent_run is not None:
                agent_run.current = c.id

            def work(c=c, box=box):
                try:
                    box["r"] = hub.call(c.name, c.arguments, cancel, agent_run.ctx if agent_run is not None else None)
                except McpCancelled:
                    box["cancelled"] = True
            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            try:
                beat = 0
                while worker.is_alive():
                    worker.join(0.2)
                    if agent_run is not None:
                        for e in agent_run.drain():      # a question for the user, a todo list, ...: to the page at once
                            yield "mcp", e
                    beat += 1
                    if worker.is_alive() and beat % 5 == 0:
                        yield "ping", None
                if agent_run is not None:
                    for e in agent_run.drain():
                        yield "mcp", e
            except GeneratorExit:
                cancel.set()                             # the client is gone: stop the tool too (a question that is open ends too)
                raise
            if "r" not in box:
                break
            r = box["r"]
            print(f"[strata] tool {c.name}: {'ok' if r['ok'] else 'error'}, {r['chars']:,} characters in "
                  f"{r['ms'] / 1000:.1f} s{' (truncated for the model)' if r['truncated'] else ''}", flush=True)
            results.append(r["text"])
            yield "mcp", {"event": "result", "id": c.id, **{k: r[k] for k in ("ok", "text", "chars", "truncated", "ms")}}
        if cancel.is_set() or len(results) < len(calls):
            done = {**done, "finish": "cancel"}
            break
        messages.append({"role": "assistant", "content": "".join(text).strip(),
                         **({"reasoning_content": "".join(reasoning).strip()} if reasoning else {}),
                         "tool_calls": [{"function": {"name": c.name, "arguments": c.arguments}} for c in calls]})
        messages += [{"role": "tool", "content": r} for r in results]
        ids, thinking, max_new = svc.prepare(messages, tools, kw, max_req, 0 if sampling.get("stream") else 1)
    yield "done", {**done, "completion_tokens": total, "prompt_tokens": len(ids)}


# ------------------------------------------------------------------------------------------------ OpenAI
def openai_chunks(svc: Service, req: dict, ids, thinking, tools, max_new, cancel, run=None):
    """`run`: the events to send instead of Service.run's (run_with_mcp); its ("mcp", {...}) items become chunks with
    an empty delta and a `strata_mcp` field, which only the web app reads."""
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

    def chunk(delta, finish=None):
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": svc.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    yield chunk({"role": "assistant", "content": ""})
    calls = 0
    streamed = {}                                  # tool call id -> index, for calls sent piece by piece
    for kind, x in run if run is not None else svc.run(ids, thinking, tools, max_new, req, cancel):
        if kind == "ping":
            yield None
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
                continue
            elif ev.kind == "tool_call":
                yield chunk({"tool_calls": [{"index": calls, "id": ev.call.id, "type": "function",
                                             "function": {"name": ev.call.name,
                                                          "arguments": json.dumps(ev.call.arguments, ensure_ascii=False)}}]})
                calls += 1
        else:
            finish = "tool_calls" if calls and x["finish"] == "stop" else {"cancel": "stop"}.get(x["finish"], x["finish"])
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


# ------------------------------------------------------------------------------------------------ Anthropic
def anthropic_events(svc: Service, req: dict, ids, thinking, tools, max_new, cancel):
    mid = "msg_" + uuid.uuid4().hex[:24]
    yield "message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": svc.model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": len(ids), "output_tokens": 0}}}
    index, open_kind, used_tool = -1, None, False

    def close():
        if open_kind == "thinking":
            yield "content_block_delta", {"type": "content_block_delta", "index": index,
                                          "delta": {"type": "signature_delta", "signature": ""}}
        yield "content_block_stop", {"type": "content_block_stop", "index": index}

    streamed = set()
    choice = req.get("tool_choice") if isinstance(req.get("tool_choice"), dict) else {}
    one_call, first_done = bool(choice.get("disable_parallel_tool_use")), False   # xeno #49 S2
    for kind, x in svc.run(ids, thinking, tools, max_new, req, cancel):
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
            stop = "stop_sequence" if x.get("stop_sequence") else \
                   "tool_use" if used_tool and (x["finish"] == "stop" or first_done) else \
                   {"stop": "end_turn", "length": "max_tokens", "cancel": "end_turn"}[x["finish"]]
            # the final counts, Anthropic's way: input_tokens leaves out what the conversation cache already held,
            # which is cache_read_input_tokens (message_start could only say the whole prompt)
            reused = min(x.get("reused") or 0, len(ids))
            yield "message_delta", {"type": "message_delta",
                                    "delta": {"stop_reason": stop, "stop_sequence": x.get("stop_sequence")},
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
            b["input"] = json.loads(b.pop("_json") or "{}")
        elif name == "message_delta":
            msg["stop_reason"] = e["delta"]["stop_reason"]
            msg["stop_sequence"] = e["delta"]["stop_sequence"]
            msg["usage"].update(e["usage"])
    msg["content"] = blocks
    return msg


# ------------------------------------------------------------------------------------------------ HTTP
def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"                       # SSE ends by closing the connection

        def log_message(self, fmt, *args):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if not svc.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            given = auth[7:].strip() if auth.lower().startswith("bearer ") else self.headers.get("x-api-key", "")
            if hmac.compare_digest(given.encode(), svc.api_key.encode()):   # #213: constant-time
                return True
            self._json(401, {"error": {"type": "authentication_error", "message": "missing or wrong API key"}})
            return False

        def _host_ok(self) -> bool:
            """With no API key, only this PC's own names as Host (see host_allowed); with one, the key is the gate."""
            if svc.api_key or host_allowed(self.headers.get("Host", ""), svc.allowed_hosts):
                return True
            self._json(421, {"error": {"message": "this name does not reach this server: use the PC's address, or list the name in the run config's allowed_hosts"}})
            return False

        def _foreign_origin(self) -> bool:
            """A browser sends Origin on a cross-site POST; one that is not this server's own page is refused (403)."""
            origin = self.headers.get("Origin")
            if origin and origin.split("://", 1)[-1] != self.headers.get("Host", ""):
                self._json(403, {"error": {"message": "only from Strata's own page"}})
                return True
            return False

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
            if not self._host_ok():
                return
            if self._ui_prefix():
                return
            path = self.path.split("?")[0].rstrip("/")
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
                        return self._json(200, svc.hstore.page(page, size))
                    rid = path[len("/metrics/requests/"):]
                    summary = svc.hstore.summary(rid)
                    if summary is None:
                        return self._json(404, {"error": {"message": "no such request"}})
                    detail = svc.hstore.detail(rid)
                    return self._json(200, {"summary": summary, "detail": detail,
                                            "detail_state": "kept" if detail is not None else "deleted"})
                return
            if path == "/metrics":
                if self._authorized():
                    # the last 12 requests; `?requests=all` every one kept (the Monitor's "Show all", issue #35)
                    self._json(200, svc.metrics(all_requests="requests=all" in self.path))
                return
            if path == "/settings":
                if self._authorized():
                    self._json(200, {"shared": bool(svc.shared), "defaults": svc.shared})
                return
            if path == "/mcp":
                # the MCP servers, their state and tools (the web app's switch and Monitor card)
                if self._authorized():
                    self._json(200, svc.mcp.status() if svc.mcp else {"servers": [], "tools": 0})
                return
            if path == "/agent":
                # the chat's coding tools: whether they are on, which shell commands run in, and whether this caller may use them (only from this PC, or with the key)
                if self._authorized():
                    ok, why = mcp_admin.may_edit(bool(svc.api_key), self.client_address[0], self.headers.get("Host", ""))
                    sh = getattr(svc.agent, "shell", None)
                    self._json(200, {"available": svc.agent is not None, "allowed": ok, "reason": "" if ok else why, "shell": shell_mod.describe(sh) if sh else None,
                                     "tools": [t["name"] for t in svc.agent.tools] if svc.agent else []})
                return
            if path == "/agent/folders":
                # the folders of this PC, to choose a project's folder from (only the names of folders, and only for who may use the coding tools)
                if self._authorized():
                    ok, why = mcp_admin.may_edit(bool(svc.api_key), self.client_address[0], self.headers.get("Host", ""))
                    if not ok:
                        return self._json(403, {"error": {"message": why}})
                    seen = folders_mod.look(parse_qs(urlsplit(self.path).query).get("path", [""])[0])
                    # a path that is no folder is an answer, not an error (a 404 would be logged as one in the browser's console)
                    self._json(200, seen if seen is not None else {"ok": False, "error": "That is not a folder on this PC"})
                return
            if path in ("/agent/git", "/agent/git/diff"):
                # the Git state of a folder, read only, for the Chat's right panel (serve/gitview.py); for who may use the coding tools
                if self._authorized():
                    ok, why = mcp_admin.may_edit(bool(svc.api_key), self.client_address[0], self.headers.get("Host", ""))
                    if not ok:
                        return self._json(403, {"error": {"message": why}})
                    q = parse_qs(urlsplit(self.path).query)
                    folder = q.get("path", [""])[0]
                    if path == "/agent/git":
                        return self._json(200, gitview.info(folder))
                    return self._json(200, gitview.diff(folder, q.get("file", [""])[0], q.get("staged", ["0"])[0] == "1", q.get("untracked", ["0"])[0] == "1"))
                return
            if path in ("/agent/files", "/agent/mention"):
                # `@file` in the prompt: the files of the project's folders that go with typed letters, and the text of one that was mentioned (serve/files.py); for who may use the coding tools
                if self._authorized():
                    ok, why = mcp_admin.may_edit(bool(svc.api_key), self.client_address[0], self.headers.get("Host", ""))
                    if not ok:
                        return self._json(403, {"error": {"message": why}})
                    q = parse_qs(urlsplit(self.path).query)
                    folders = [f for f in [q.get("path", [""])[0], *q.get("dirs", [])] if f.strip() and "\0" not in f][:agent_mod.MAX_DIRS + 1]
                    if path == "/agent/files":
                        return self._json(200, {"files": files_mod.find(folders, q.get("q", [""])[0])})
                    return self._json(200, files_mod.read(folders, q.get("rel", [""])[0]))
                return
            if path in ("/agent/memory", "/agent/memory/file"):
                # the notes the chat reads for a project: the project's own files and what the user's other apps wrote down (serve/memory.py); for who may use the coding tools
                if self._authorized():
                    ok, why = mcp_admin.may_edit(bool(svc.api_key), self.client_address[0], self.headers.get("Host", ""))
                    if not ok:
                        return self._json(403, {"error": {"message": why}})
                    q = parse_qs(urlsplit(self.path).query)
                    folders = [f for f in [q.get("path", [""])[0], *q.get("dirs", [])] if f.strip() and "\0" not in f][:agent_mod.MAX_DIRS + 1]
                    imp = getattr(svc, "importer", None)
                    home = str(imp.home) if imp is not None else str(harness.home_dir())
                    on = memory_mod.settings(harness._cfg(svc))["on"]
                    sources = memory_mod.discover(home, folders, on)
                    if path == "/agent/memory":
                        return self._json(200, {"max": memory_mod.MAX_TOTAL, "sources": [{k: s[k] for k in ("id", "app", "label", "kind", "shown", "bytes", "on")} for s in sources]})
                    src = next((s for s in sources if s["id"] == q.get("id", [""])[0]), None)
                    try:
                        n = int(q.get("n", ["0"])[0])
                    except ValueError:
                        n = -1
                    if src is None or not 0 <= n < len(src["files"]):
                        return self._json(404, {"ok": False, "error": "no such file"})
                    got = memory_mod._read(src["files"][n], 200_000)
                    if got is None:
                        return self._json(200, {"ok": False, "error": "cannot be read"})
                    return self._json(200, {"ok": True, "name": os.path.basename(src["files"][n]), "text": got[0], "cut": got[1]})
                return
            if path == "/mcp/config":
                # the servers as set up (secrets masked), their state, the limits, and whether this caller may change them (#79)
                if self._authorized():
                    self._json(200, mcp_admin.view(svc, self.client_address[0], self.headers.get("Host", "")))
                return
            if path == "/import":
                # the skills and the MCP servers of the other coding apps on this PC, with what may be changed (#94)
                if self._authorized():
                    self._json(200, harness.view(svc, self.client_address[0], self.headers.get("Host", ""), rescan="rescan=1" in self.path))
                return
            if path == "":
                body = (ROOT / "serve" / "web" / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/health":
                proc = getattr(svc.engine, "proc", None)
                # an engine unloaded on purpose (#208) is not a dead one; `not svc.loaded()` is true after a crash too
                alive = proc is None or proc.poll() is None or bool(getattr(svc.engine, "unloaded", False))
                self._json(200 if alive else 503, {"status": "ok" if alive else "engine_exited",
                                     "max_context": svc.engine.max_context, "model": svc.model,
                                     "images": svc.vision is not None, "api_key": bool(svc.api_key),
                                     "loops_stopped": svc.status["loops_stopped"], "loaded": svc.loaded()})
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
                self._json(200, s)
            elif path in ("/v1/models", "/models"):
                if self._authorized():
                    loaded = svc.loaded()
                    model = {"id": svc.model, "object": "model", "status": {"value": "loaded"},
                             "meta": {"n_ctx": svc.engine.max_context},
                             "architecture": {"input_modalities": ["text", "image"] if svc.vision is not None else ["text"],
                                              "output_modalities": ["text"]}}
                    if not loaded and (svc.idle_unload_s or getattr(svc.engine, "unloaded", False)):
                        model["status"] = {"value": "unloaded"}   # like llama-server's router: listed, loads on use
                        loaded = True
                    self._json(200, {"object": "list", "data": [model] if loaded else []})
            elif path == "/props":
                if self._authorized():
                    self._props()
            elif path == "/slots":
                if self._authorized():
                    loaded = not hasattr(svc.engine, "alive") or svc.engine.alive()
                    with svc.status_lock:
                        busy = bool(svc.status.get("busy"))
                    slot = {"id": 0, "n_ctx": svc.engine.max_context, "is_processing": busy}
                    self._json(200, [slot] if loaded else [])
            elif path == "/v1/status":
                if self._authorized():
                    self._json(200, svc.v1_status())
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            with timeline.span("http request"):
                self._do_post()

        def _do_post(self):
            if not self._host_ok():
                return
            if self.path.startswith("/classic/"):             # the classic app under its prefix (xeno UI S1)
                self.path = self.path[len("/classic"):]
            if not self._authorized():
                return
            path = self.path.split("?")[0].rstrip("/")   # issue #55: Claude Code posts /v1/messages?beta=true
            if path == "/settings":
                self._settings()
                return
            if path == "/mcp/config":                        # set up the MCP servers: a write that decides which programs Strata starts (#79)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 1_000_000 else b""
                if not self._own_page("MCP servers can be changed"):
                    return
                ok, why = mcp_admin.may_edit(bool(svc.api_key), self.client_address[0], self.headers.get("Host", ""))
                if not ok:
                    return self._json(403, {"error": {"message": why}})
                if not svc.config_path:
                    return self._json(409, {"error": {"message": "this server was started without a run config file (--config), so there is nowhere to save the servers"}})
                if not 0 < n <= 1_000_000:
                    return self._json(413, {"error": {"message": "send a body of up to 1 MB"}})
                try:
                    body = json.loads(raw)
                except ValueError:
                    return self._json(400, {"error": {"message": "the body is not JSON", "fields": []}})
                code, out = mcp_admin.apply(svc, body)
                return self._json(code, out if code != 200 else mcp_admin.view(svc, self.client_address[0], self.headers.get("Host", "")))
            if path in ("/agent/rewind", "/agent/checkpoints/forget"):         # the way back for the files the tools changed (serve/checkpoints.py)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 10_000 else b""
                if not self._own_page("files can be put back"):
                    return
                ok, why = mcp_admin.may_edit(bool(svc.api_key), self.client_address[0], self.headers.get("Host", ""))
                if not ok:
                    return self._json(403, {"error": {"message": why}})
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                if not isinstance(body, dict) or not isinstance(body.get("session"), str):
                    return self._json(400, {"error": {"message": "send {\"session\": the chat's id, \"checkpoint\": the prompt's id}"}})
                if path == "/agent/checkpoints/forget":
                    svc.checkpoints.forget(body["session"])
                    return self._json(200, {"ok": True})
                if not isinstance(body.get("checkpoint"), str):
                    return self._json(400, {"error": {"message": "send the checkpoint, the prompt's id"}})
                if body.get("apply") is True:
                    return self._json(200, svc.checkpoints.apply(body["session"], body["checkpoint"], body.get("include_changed") is True))
                return self._json(200, svc.checkpoints.preview(body["session"], body["checkpoint"]))
            if path == "/agent/permission":                  # the page's answer to a question of the coding tools (serve/agent_run.py)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 10_000 else b""
                if not self._own_page("A question can be answered"):
                    return
                ok, why = mcp_admin.may_edit(bool(svc.api_key), self.client_address[0], self.headers.get("Host", ""))
                if not ok:
                    return self._json(403, {"error": {"message": why}})
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
                if not isinstance(body, dict) or not isinstance(body.get("id"), str) or body.get("decision") not in agent_run.ANSWERS:
                    return self._json(400, {"error": {"message": "send {\"id\": the question's id, \"decision\": \"allow\" | \"allow_chat\" | \"deny\"}"}})
                if not svc.broker.answer(body["id"], body["decision"]):
                    return self._json(404, {"error": {"message": "no question with that id is waiting (it was answered, or the request ended)"}})
                return self._json(200, {"ok": True})
            if path == "/import":                            # the skill switches, importing one MCP server, a rescan: a write like /mcp/config (#94)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n) if 0 <= n <= 1_000_000 else b""
                if not self._own_page("the import can be changed"):
                    return
                ok, why = mcp_admin.may_edit(bool(svc.api_key), self.client_address[0], self.headers.get("Host", ""))
                if not ok:
                    return self._json(403, {"error": {"message": why}})
                if not svc.config_path:
                    return self._json(409, {"error": {"message": harness.NO_FILE}})
                if not 0 < n <= 1_000_000:
                    return self._json(413, {"error": {"message": "send a body of up to 1 MB"}})
                try:
                    body = json.loads(raw)
                except ValueError:
                    return self._json(400, {"error": {"message": "the body is not JSON", "fields": []}})
                code, out = harness.apply(svc, body)
                return self._json(code, out if code != 200 else harness.view(svc, self.client_address[0], self.headers.get("Host", "")))
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
            if path in ("/unload", "/load") and self._foreign_origin():
                return
            if path == "/unload":                            # give the GPU back now (between requests)
                r = svc.unload()
                self._json(409 if r == "busy" else 200, {"status": r})
                return
            if path == "/load":                              # load now, e.g. ahead of a request
                try:
                    svc.load()
                    self._json(200, {"status": "loaded"})
                except GpuBusy as e:
                    self._json(503, {"error": {"type": "server_error", "message": str(e)}})
                except EngineDied as e:                      # #56 review: while it restarts, a 529, not a dropped connection
                    self._json(529, {"type": "error", "error": {"type": "overloaded_error", "message": str(e)}})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if path in ("/v1/chat/completions", "/v1/messages"):
                    slot = cache_slot(req)  # validate before opening an SSE response
                    if slot and int(getattr(svc.engine, "info", {}).get("cache_slots", 4)) == 1:
                        raise ValueError("cache slots require a single-GPU session")
                    svc.load()                               # unloaded: load first (or 503 while the GPU is busy)
                if path == "/v1/chat/completions":
                    self._openai(req)
                elif path == "/v1/messages":
                    self._anthropic(req)
                elif self.path.rstrip("/") == "/v1/messages/count_tokens":
                    messages, tools, kw = anthropic_to_messages(req, vision=svc.vision is not None)
                    try:
                        ids, _, _ = svc.prepare(messages, tools, kw, None)
                    finally:
                        path = getattr(svc.embeddings, "path", None)
                        if path is not None:
                            Path(path).unlink(missing_ok=True)
                        svc.embeddings.path = None
                    self._json(200, {"input_tokens": len(ids)})
                else:
                    self._json(404, {"error": {"message": "not found"}})
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
            except GpuBusy as e:
                self._json(503, {"error": {"type": "server_error", "message": str(e)}})
            except EngineDied as e:                          # before the answer started (not streamed)
                if self.path.startswith("/v1/messages"):     # xeno #49 S2: the native API's retryable error
                    self._json(529, {"type": "error", "error": {"type": "overloaded_error",
                                                                "message": f"{e}; the next request restarts it"}})
                else:
                    self._json(503, {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}})

        def _props(self):
            model = parse_qs(urlsplit(self.path).query).get("model", [svc.model])[0]
            if model != svc.model:
                self._json(404, {"error": {"message": "model not found"}})
                return
            if hasattr(svc.engine, "alive") and not svc.engine.alive():
                self._json(503, {"error": {"message": "the engine is not running"}})
                return
            defaults = {**svc.sampling_defaults, **svc.shared}
            names = {"repetition_penalty": "repeat_penalty", "penalty_last_n": "repeat_last_n"}
            params = {names.get(k, k): v for k, v in defaults.items()
                      if k in ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
                               "presence_penalty", "frequency_penalty", "penalty_last_n")}
            params["n_predict"] = svc.shared.get("max_tokens", -1)
            props = {"default_generation_settings": {"n_ctx": svc.engine.max_context, "params": params},
                     "total_slots": 1, "model_alias": svc.model, "chat_template": svc.template.source,
                     "modalities": {"vision": svc.vision is not None}, "models_autoload": False,
                     "is_sleeping": False}
            if getattr(svc.engine, "model_path", None):
                props["model_path"] = svc.engine.model_path
            version = getattr(svc.engine, "info", {}).get("version")
            if version:
                props["build_info"] = "Strata " + str(version)
            self._json(200, props)

        def _props(self):
            model = parse_qs(urlsplit(self.path).query).get("model", [svc.model])[0]
            if model != svc.model:
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
            props = {"default_generation_settings": {"n_ctx": svc.engine.max_context, "params": params},
                     "total_slots": 1, "model_alias": svc.model, "chat_template": svc.template.source,
                     "modalities": {"vision": svc.vision is not None}, "models_autoload": False,
                     "is_sleeping": not svc.loaded()}
            if getattr(svc.engine, "model_path", None):
                props["model_path"] = svc.engine.model_path
            version = getattr(svc.engine, "info", {}).get("version")
            if version:
                props["build_info"] = "Strata " + str(version)
            self._json(200, props)

        def _own_page(self, what) -> bool:
            """Only JSON (a form or a "simple" cross-site request can't send it without a CORS preflight, which this
            server never grants) and no foreign Origin: a web page elsewhere must not change settings or run tools."""
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                self._json(415, {"error": {"message": "send application/json"}})
                return False
            origin = self.headers.get("Origin")
            if origin and origin.split("://", 1)[-1] != self.headers.get("Host", ""):
                self._json(403, {"error": {"message": f"{what} only from Strata's own page"}})
                return False
            return True

        def _settings(self):
            # They change what every client gets, so only the app's own page may set them
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not self._own_page("settings can be changed"):
                return
            try:
                req = json.loads(body or b"{}")
                shared = svc.set_shared(req.get("defaults") if isinstance(req, dict) else None)
            except ValueError as e:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(e)}})
                return
            print("[strata] other apps now use the Chat settings: " + ", ".join(f"{k}={v}" for k, v in shared.items())
                  if shared else "[strata] other apps use their own settings again", flush=True)
            self._json(200, {"shared": bool(shared), "defaults": shared})

        def _sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

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
            sa = req.get("strata_agent")                         # the web app's coding tools (serve/agent.py): files and commands on this PC, with the user's say-so
            if isinstance(sa, dict):
                if not self._own_page("The coding tools can be used"):
                    return
                ok, why = mcp_admin.may_edit(bool(svc.api_key), self.client_address[0], self.headers.get("Host", ""))
                if not ok:
                    return self._json(403, {"error": {"message": "The coding tools change files and run commands on this PC, so they work only from this PC itself (open the page as localhost) or with the API key. " + why}})
                if svc.agent is None or svc.mcp is None:
                    return self._json(409, {"error": {"message": "the coding tools are switched off on this server (\"agent\": false in the run config)"}})
                arun = svc.start_agent_run(sa, req.get("messages") or [])
                req = {**req, "messages": agent_prompt.with_system(req.get("messages") or [], arun.prompt)}
            messages, tools, kw = openai_to_messages(req)
            req = svc.cjk(req, messages)                                # xeno #49 S4
            max_req = max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 0)   # 0/-1: the rest
            use_mcp = (req.get("strata_mcp") is True or arun is not None) and svc.mcp is not None      # the web app's opt-in (serve/mcp.py), or the coding tools
            own = {t.get("name") for t in tools or []}
            if use_mcp:
                if not self._own_page("MCP tools can be used"):   # tools run with the user's rights on this PC
                    return
                svc.mcp.wait(10)                                  # servers still starting (only right after start)
                off = req.get("strata_mcp_off")                   # servers the page's list switched off for this chat; anything but a list of names is ignored
                skip = {x for x in off if isinstance(x, str)} if isinstance(off, list) and all(isinstance(x, str) for x in off) else set()
                if req.get("strata_mcp") is not True:
                    skip |= {n for n in svc.mcp.servers if n != "agent"}      # the coding tools alone: no MCP server
                if arun is None:
                    skip.add("agent")                             # and the coding tools only when the request asks for them
                extra = svc.mcp.template_tools(exclude=own, skip_servers=skip)       # the request's own tools win a name clash
                use_mcp = bool(extra)
                tools = (tools or []) + extra or None
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, 0 if req.get("stream") else 1)
            req = svc.with_slot(req, ids)                               # xeno #49 S7
            req = {**req, "_meta": svc.meta_for("openai", messages, tools, self.headers.get("User-Agent"))}
            _debug_req("openai", req, messages, tools, max_new, thinking, len(ids))
            cancel = threading.Event()
            if arun is not None:
                arun.bind(cancel)
            run = run_with_mcp(svc, svc.mcp, messages, tools, kw, ids, thinking, max_new, max_req, req, cancel,
                               {t["name"] for t in extra}, arun) if use_mcp else None
            chunks = openai_chunks(svc, req, ids, thinking, tools, max_new, cancel, run=run)
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
                cancel.set()                                 # client went away: stop the engine
                chunks.close()
            except EngineDied as e:                          # mid-stream: say so, then end the stream properly
                err = {"error": {"type": "server_error", "message": f"{e}; the next request restarts it"}}
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started: the
                err = {"error": {"type": "server_error", "message": str(e)}}   # headers are sent, so no 400 now
                self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\ndata: [DONE]\n\n")

        def _anthropic(self, req):
            req = svc.with_shared(req, "anthropic")
            messages, tools, kw = anthropic_to_messages(req, vision=svc.vision is not None)
            req = svc.cjk(req, messages)                                # xeno #49 S4
            max_new = int(req.get("max_tokens") or 0)                  # 0/-1: the rest of the context
            think_budget.side_effort(req, kw)                         # xeno #49 S3: before the template renders
            ids, thinking, max_new = svc.prepare(messages, tools, kw, max_new, 0 if req.get("stream") else 1)
            req = svc.with_slot(req, ids)                               # xeno #49 S7
            req = {**req, "_meta": svc.meta_for("anthropic", messages, tools, self.headers.get("User-Agent"))}
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
            if not req.get("stream"):
                key = svc.replay_key(req, ids, max_new)
                if key is not None and key in svc.replays:
                    print("[strata] the same greedy request again: its last answer, not generated again", flush=True)
                    return self._json(200, {**svc.replays[key], "id": f"msg_{uuid.uuid4().hex[:24]}"})
                body = anthropic_collect(anthropic_events(svc, req, ids, thinking, tools, max_new, cancel))
                if key is not None:
                    svc.remember_reply(key, body)
                return self._json(200, body)
            events = anthropic_events(svc, req, ids, thinking, tools, max_new, cancel)
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
                cancel.set()
                events.close()
            except EngineDied as e:                          # mid-stream: Anthropic's retryable error event (#49 S2)
                err = {"type": "error", "error": {"type": "overloaded_error",
                                                  "message": f"{e}; the next request restarts it"}}
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")
            except ValueError as e:                          # the engine's ERR after the stream started
                err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                self.wfile.write(b"event: error\ndata: " + json.dumps(err).encode() + b"\n\n")

    return Handler


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second server bind a port that is already serving, and requests then land on
    # either one (a forgotten second start of run-<model>.bat).  Without it the second start fails loudly instead.
    allow_reuse_address = os.name != "nt"

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):   # a client that hangs up needs no stack trace
            super().handle_error(request, client_address)


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


def serve(svc: Service, host="127.0.0.1", port=8095) -> ThreadingHTTPServer:
    svc.start_telemetry()
    httpd = Server((host, port), make_handler(svc))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def host_allowed(host: str, allowed=()) -> bool:
    """Whether the Host a request names is this PC (xeno #71). A web page on another site whose name has been re-pointed at
    this PC (DNS rebinding) is then same-origin with the server, and its requests carry that site's name as Host. An IP
    address cannot be re-pointed, `localhost` and a name with no dot (a machine on the LAN) cannot be a public site, `.local`
    is mDNS; any other name has to be in the run config's `allowed_hosts`. No Host at all is no browser."""
    h = (host or "").strip().lower()
    if not h:
        return True
    if h.startswith("["):
        name = h[1:h.find("]")] if "]" in h else h
    else:
        name = h.rsplit(":", 1)[0] if h.count(":") == 1 else h
    name = name.rstrip(".")
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        pass
    return name == "localhost" or name.endswith(".localhost") or "." not in name or name.endswith(".local") or name in allowed


def ui_choice(cfg: dict) -> str:
    """Which web app `/` serves: the new one, unless the run config says `"ui": "classic"` (the classic app is at /classic/ either way)."""
    return "classic" if (cfg or {}).get("ui") == "classic" else "next"


SHARED_KEYS = ("reasoning_effort", "temperature", "top_p", "top_k", "seed", "max_tokens", "experimental_speed_projection")


def clean_shared_defaults(d) -> dict:
    """The Chat settings other apps get (POST /settings): only known keys, each checked; ValueError names a bad one."""
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError("defaults must be an object")
    out = {}
    for key, value in d.items():
        if value is None or value == "":
            continue
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if key == "reasoning_effort":
            if value not in ("none", "low", "medium", "high", "xhigh"):
                raise ValueError("reasoning_effort: none, low, medium, high or xhigh")
        elif key == "temperature":
            if not number or not 0 <= value <= 2:
                raise ValueError("temperature: 0..2")
        elif key == "top_p":
            if not number or not 0 < value <= 1:
                raise ValueError("top_p: 0 < top_p <= 1")
        elif key == "top_k":
            if not number or value != int(value) or not 1 <= value <= 64:
                raise ValueError("top_k: an integer 1..64")
            value = int(value)
        elif key in ("seed", "max_tokens"):
            if not number or value != int(value) or value <= 0:
                raise ValueError(f"{key}: a positive integer")
            value = int(value)
        elif key == "experimental_speed_projection":
            if not isinstance(value, bool):
                raise ValueError("experimental_speed_projection: true or false")
        else:
            raise ValueError(f"unknown setting {key!r}")
        out[key] = float(value) if key in ("temperature", "top_p") else value
    return out


# xeno #49 S8 / story 20: coding presets, chosen in the run config ("sampling": {"preset": "balanced"}) and never
# switched automatically. None of them is measured here yet. reasoning = the sampling docs/DETAILS.md shows for this
# model; balanced = the Qwen3 family's thinking-mode recommendation (VENDOR); deterministic = greedy.
SAMPLING_PRESETS = {
    "deterministic": {"temperature": 0.0},
    "balanced": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "reasoning": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
}


def sampling_defaults_from_config(cfg: dict) -> dict:
    """The run config's optional `sampling` block: defaults for the sampling fields a request leaves out, so
    a plain client gets configured sampling instead of greedy.  Supported: temperature, top_p, top_k, min_p,
    presence_penalty, repetition_penalty, frequency_penalty, penalty_last_n, seed.  The request's own fields
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
                    help="require this key on /v1/* (Authorization: Bearer ... or x-api-key); also $STRATA_API_KEY")
    ap.add_argument("--mcp-config", help="a JSON file with MCP servers in Claude Desktop's format ({\"mcpServers\": "
                                         "{...}}); the web app's chat can use their tools (also \"mcp_servers\" in "
                                         "the config)")
    ap.add_argument("--idle-unload", type=float, default=None, metavar="SECONDS",
                    help="unload the model after this many seconds without requests, so other programs (games, other "
                         "model servers) can use the VRAM; the next request loads it again (also \"idle_unload_s\" "
                         "in the config; default: never)")
    ap.add_argument("--min-free-vram-mib", type=int, default=None,
                    help="load an unloaded model only when this much VRAM is free, else answer 503 (also "
                         "\"min_free_vram_mib\" in the config; default: always load)")
    ap.add_argument("--before-load", help="a command run before the model is loaded again (e.g. to unload another "
                                          "server's model; also \"before_load\" in the config, a string or a list)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig")) if a.config else {}   # Notepad adds a BOM
    if a.gpu is not None:
        cfg["gpu"] = int(a.gpu) if a.gpu.strip().isdigit() else a.gpu
    a.host = a.host or cfg.get("host") or "127.0.0.1"   # issue #26: the run scripts pass no --host, the config can
    try:                                                # before the minutes of loading: is the port free?
        Server((a.host, a.port), BaseHTTPRequestHandler).server_close()
    except OSError:
        ap.error(f"port {a.port} is already in use - is Strata (or another server) already running? "
                 f"Close it, or start this one with a different --port")
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
    hub = hub_from_config(cfg, a.mcp_config, builtins={**importer.builtins(), **({"agent": agent_server} if agent_server else {})})     # before the minutes of loading: a bad entry stops here
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
        if cfg.get("vision"):
            print("loading the vision encoder ...", flush=True)
            vision = Vision(cfg["vision"], log=open(cfg["log"], "a", encoding="utf-8") if cfg.get("log") else None,
                            env=env)
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
        print("loading the model (the first start takes a minute or two) ...", flush=True)
        if len(gpu_list(cfg)) > 1:
            print(f"[strata] layer split across GPUs {gpu_list(cfg)} ({cfg.get('layer_split') or 'auto'})", flush=True)
        engine = StrataEngine(cfg["exe"], args, cwd=cfg.get("cwd"), log=cfg.get("log"), env=env)
        warn_tight_ram(engine.info.get("arena_mib"))
    else:
        engine, vision, sampling_defaults = MockEngine(tok, a.script or [
            "Thinking about it.</think>\n\nHello from the mock engine."]), None, {}
    # the model's own chat template (exported with its tokenizer), else the original model's
    tpl = tpath / "chat_template.jinja"
    svc = Service(engine, tok, ChatTemplate(tpl if tpl.exists() else ROOT / "serve/chat_template.jinja"),
                  model_name=cfg.get("model_name", "qwen3.8-flash-next"), vision=vision,
                  sampling_defaults=sampling_defaults,
                  fit_max_tokens=a.fit_max_tokens or cfg.get("fit_max_tokens") is True)
    if ("STRATA_API_KEY" in os.environ and not os.environ["STRATA_API_KEY"].strip()) or             any(x == "--api-key" and i + 1 < len(sys.argv) and not sys.argv[i + 1].strip() or x.strip() == "--api-key="
                for i, x in enumerate(sys.argv)):
        # #213: an empty key would switch authentication off without a word
        print("[strata] an API key was given but it is empty: set a key, or leave --api-key / STRATA_API_KEY out",
              file=sys.stderr)
        return 2
    svc.api_key = a.api_key or cfg.get("api_key", "")
    svc.cjk_ban = a.engine == "strata" and bool(cfg.get("cjk_guard"))   # xeno #49 S4: --ban-ids was passed
    svc.idle_unload_s = a.idle_unload if a.idle_unload is not None else float(cfg.get("idle_unload_s") or 0)
    svc.min_free_vram_mib = a.min_free_vram_mib if a.min_free_vram_mib is not None else \
        int(cfg.get("min_free_vram_mib") or 0)
    svc.before_load = a.before_load or cfg.get("before_load") or None
    hist = cfg.get("history") if isinstance(cfg.get("history"), dict) else {}      # {"enabled", "dir", "detail_cap_gb"}
    from serve.history import default_dir
    svc.hstore = HistoryStore(hist.get("dir") or default_dir(), enabled=hist.get("enabled", True) is not False,
                              detail_cap_bytes=int(float(hist.get("detail_cap_gb", 2)) * 2**30))
    svc.checkpoints = checkpoints_mod.Checkpoints()          # the way back for files the tools change: in the user's data folder, the oldest trimmed at start
    threading.Thread(target=svc.checkpoints.trim, daemon=True).start()
    def _model_info(files=gguf_info.files_from_args(list(cfg.get("args") or []))):
        svc.model_info = gguf_info.model_info(files)        # reads headers only (~0.1 s); a model with no GGUF gives None
    threading.Thread(target=_model_info, daemon=True).start()
    svc.ui = ui_choice(cfg)                                            # which web app "/" serves (xeno UI)
    svc.config_path, svc.mcp_config_path = a.config, a.mcp_config      # where the web app saves and reads the MCP servers (#79)
    svc.allowed_hosts = {str(h).strip().lower().rstrip(".") for h in (cfg.get("allowed_hosts") or []) if str(h).strip()}   # xeno #71
    svc.gpu_indices = monitor_gpus(cfg)               # every card the engine can see (issue #112; UI S0: no "gpu" key)
    svc.gpu_index = (svc.gpu_indices or [0])[0]         # the Monitor reads the card the engine runs on (issue #51)
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
    here = "127.0.0.1" if a.host in ("0.0.0.0", "", "::") else a.host
    print(f"ready: http://{here}:{a.port}/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, "
          f"context {engine.max_context} tokens{', images on' if vision else ''}"
          f"{', API key required' if svc.api_key else ''})", flush=True)
    print(f"       open http://{here}:{a.port}/ in a browser to chat; close this window to stop the model", flush=True)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        # issue #26: reachable from other devices - say at which address, and what can still block it
        ips = lan_addresses()
        for ip in ips:
            print(f"       from other devices: http://{ip}:{a.port}/   (API: http://{ip}:{a.port}/v1)", flush=True)
        if not ips:
            print("       from other devices: http://<this PC's IP address>:" + str(a.port) + "/", flush=True)
        if not svc.api_key:
            print("       WARNING: no API key - anyone on your network can use this model. Add \"api_key\": \"...\" "
                  "to the config (clients send it as their API key; the web page asks for it)", flush=True)
        if os.name == "nt":
            print("       nothing arrives? Windows Firewall blocks it until allowed: accept its prompt for Python, or run "
                  "in an admin PowerShell:\n         New-NetFirewallRule -DisplayName \"Strata " + str(a.port) + "\" "
                  "-Direction Inbound -Protocol TCP -LocalPort " + str(a.port) + " -Action Allow -Profile Private\n"
                  "       (and set this network to Private in Windows' network settings)", flush=True)
    if a.open:
        import webbrowser
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
        closers = [httpd.shutdown, getattr(engine, "close", None), vision.close if vision else None,
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
