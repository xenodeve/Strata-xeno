"""serve/runner.py - the runner: Strata Desktop's background part on an engineer's own PC (#310 part 2, task 6).

    python -m serve.runner --server https://strata.example --key <key> [--device-id ID] [--config PATH]

It runs the chat's coding tools, the PC's own MCP servers and skills, and the web app's page routes (folders, Git,
`@file`, notes, rewind) on this PC for a project that lives here, while the model runs on the server.  It only makes
outbound requests to the server (serve/runner_hub.py is the other side):

    POST /runner/hello   this PC's facts and tool groups -> the runner id (bound to the key's identity) and the channel
    GET  /runner/next    a long poll for jobs, answers and cancels; `channel=` and `ack=` (the highest seq received)
    POST /runner/reply   a job's result, or an ask / question / emit in the middle of one; `rid` numbers them per job

Delivery is at least once: a message whose seq was already handled is ignored, and a hello that answers another
channel (the server restarted) starts the ack and the handled seqs again - the jobs of the old channel ended on the
server, so their links are cancelled.  A poll answered 404 (the server restarted, or the key was renamed) says hello
again; the id is sha256(identity, device), so it stays the same across restarts.

Files, in `runner_dir()` (%LOCALAPPDATA%\\StrataRunner on Windows, ~/.strata-runner elsewhere): `runner.json` (the run
config: `mcp_servers` / `mcpServers` and the `import.skills` switches, in the server's format; never the server's
config; its MCP servers are also set from the web app, #338: `MCP_JOBS`), `device.json` (`{"id", "name"}`, made once) and `checkpoints/` (the way back for the files the tools change).

The key may come from STRATA_RUNNER_KEY (it is then taken out of the environment, so no command or MCP server the
runner starts inherits it) and is never printed.  Every child process starts with no window (CREATE_NO_WINDOW).
Exit codes: 0 on Ctrl+C, 2 for a bad command line or runner.json, 3 when the server refuses the runner (the key was
revoked, the device refused, a newer runner on this PC, another protocol).  Standard library only.
"""
from __future__ import annotations

import argparse
import collections
import copy
import http.client
import json
import os
import queue
import re
import secrets
import socket
import ssl
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlsplit

from serve import agent as agent_mod, checkpoints as checkpoints_mod, folders as folders_mod, harness
from serve import mcp_admin, memory as memory_mod, page_routes, permissions, runner_hub, shell as shell_mod
from serve.mcp import McpCancelled, McpError, McpServer, hub_from_config

RETRY_MAX_S = 30.0              # the longest wait between tries after a network error (1, 2, 4 ... 30 s)
PENDING_RETRY_S = 2.0           # how often a runner whose device waits for the owner's approval says hello again: it
                                # is online only at its next hello after the approval, so this is the most the user
                                # waits after approving (#310 part 3 task 6, approval right after a pending hello:
                                # 9.78 / 9.98 / 10.0 s at 10 s, 1.91 / 1.88 / 2.03 s at 2 s)
PENDING_FAST_S = 120.0          # ... for this long; after it, every PENDING_SLOW_S (an approval nobody gives soon)
PENDING_SLOW_S = 10.0
MAX_JOBS = 16                   # jobs run at once; the others wait their turn
RUNS_KEPT = 50                  # run keys (one per message of a chat) whose policy and checkpoint are kept
REPLY_MAX = 32 << 20            # the largest reply the server takes (serve/server.py RUNNER_REPLY_MAX): a larger one is
                                # not sent (a 413 from the server may come as a reset while it is still being sent)
REPLY_RETRY_S = 60.0            # how long a reply is tried again after network errors before it is given up
STOP_LOOK_S = 0.1               # how often a reply waiting for the new hello after a 404 looks at the stop (the hello
                                # itself wakes it at once, #315)
STARTING_POLL_S = 1.0           # the poll's wait while an MCP server still starts, so its tools are announced soon
KEEP_IDLE_S = 60.0              # #313: a kept connection idle longer than this is not used again (the server closes its
                                # side after serve/server.py RUNNER_IDLE_S, 90 s; a proxy may close sooner)
KEEP_MAX = 4                    # #313: idle connections kept for hellos and replies (replies may go out at once from
                                # several jobs; more than this are opened as needed and closed after)
CODING = "agent"                # the coding group's name on the runner (the server calls it my_agent)
DEVICE_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")
KEY = re.compile(r"[!-~]{1,4096}")              # printable ASCII, no whitespace: what a header can carry
EXIT_REFUSED = 3


def _say(text: str) -> None:
    """What the runner tells the developer (its console, or the desktop app's log).  Tests replace it."""
    print(text, flush=True)


# ------------------------------------------------------------------------------------------------ files
def runner_dir() -> Path:
    """%LOCALAPPDATA%\\StrataRunner on Windows, ~/.strata-runner elsewhere."""
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "StrataRunner"
    return Path.home() / ".strata-runner"


def machine_name() -> str:
    """This PC's name, 1 to 80 characters.  Not platform.node(): on Windows it starts `cmd /c ver`, a console window."""
    try:
        name = socket.gethostname()
    except OSError:
        name = ""
    name = (name or os.environ.get("COMPUTERNAME") or "").strip()
    return name[:80] or "this PC"


def load_device(d: Path) -> dict:
    """This PC's device for the server: {"id": 22 characters of [A-Za-z0-9_-], "name": the machine's name}, made once
    and kept in `d`/device.json (a broken file is made again)."""
    p = Path(d) / "device.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("id"), str) and DEVICE_ID.fullmatch(data["id"]) \
                and isinstance(data.get("name"), str) and data["name"].strip():
            return {"id": data["id"], "name": data["name"][:80]}
    except (OSError, ValueError):
        pass
    dev = {"id": secrets.token_urlsafe(16), "name": machine_name()}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dev), encoding="utf-8")
    os.replace(tmp, p)
    return dev


def new_instance() -> str:
    """This process's own random id (the hello's `instance`: 8 to 64 of [A-Za-z0-9_-])."""
    return uuid.uuid4().hex


# ------------------------------------------------------------------------------------------------ the hello's limits
def _cut(v, most: int) -> str:
    return v[:most] if isinstance(v, str) else ""


def fit_info(info: dict) -> dict:
    """machine, platform, shell, home within serve/runner_hub.py check_hello's limits."""
    return {"machine": _cut(info.get("machine"), 80).strip() or "this PC",
            "platform": _cut(info.get("platform"), 40).strip() or "unknown",
            "shell": _cut(info["shell"], 400) if isinstance(info.get("shell"), str) else None,
            "home": _cut(info.get("home"), 1000)}


def fit_groups(groups: list[dict]) -> list[dict]:
    """The tool groups within check_hello's limits: at most MAX_GROUPS, a name of 1 to 64 characters (a longer one is
    left out, with a line saying so), status at most 40 and error at most 4000 characters, at most MAX_TOOLS tools,
    each named 1 to 128 characters (others left out), a description cut at 100,000 and an inputSchema that is an
    object (else none).  A coding tool whose name is not plain (`__`, `my_`) is left out, as the server would."""
    out = []
    for g in groups:
        name = g.get("name")
        if not isinstance(name, str) or not 0 < len(name) <= 64 or not name.strip():
            _say(f"[runner] the tool group {str(name)[:80]!r} is left out: its name must be 1 to 64 characters")
            continue
        if len(out) >= runner_hub.MAX_GROUPS:
            _say(f"[runner] the tool group {name!r} is left out: at most {runner_hub.MAX_GROUPS} groups")
            continue
        tools = []
        for t in g.get("tools") or []:
            tn = t.get("name") if isinstance(t, dict) else None
            if not isinstance(tn, str) or not 0 < len(tn) <= 128 or not tn.strip():
                continue
            if g["kind"] == "coding" and not runner_hub.plain_name_ok(tn):
                continue
            if len(tools) >= runner_hub.MAX_TOOLS:
                break
            tool = {"name": tn, "description": _cut(t.get("description") or t.get("title"), 100_000)}
            if isinstance(t.get("inputSchema"), dict):
                tool["inputSchema"] = t["inputSchema"]
            tools.append(tool)
        status, error = g.get("status"), g.get("error")
        out.append({"name": name, "kind": g["kind"], "status": status[:40] if isinstance(status, str) else None,
                    "error": str(error)[:4000] if error else None, "tools": tools})
    return out


def fit_skills(skills: list[dict]) -> list[dict]:
    """The skills in use within check_hello's limits: at most MAX_SKILLS, a name of 1 to 200 characters, a description
    cut at 10,000, the app (the skill's harness) at 80."""
    out = []
    for s in skills:
        name = s.get("name")
        if not isinstance(name, str) or not 0 < len(name) <= 200 or not name.strip():
            continue
        if len(out) >= runner_hub.MAX_SKILLS:
            break
        app = s.get("app", s.get("harness"))
        out.append({"name": name, "description": _cut(s.get("description"), 10_000),
                    "app": app[:80] if isinstance(app, str) else None})
    return out


# ------------------------------------------------------------------------------------------------ the server
class Refused(Exception):
    """The server will not have this runner: `kind` is "key" (401: a wrong or revoked key), "device_refused",
    "runner_replaced", "runner_protocol" or "refused" (another answer that trying again does not change)."""

    def __init__(self, kind: str, message: str, status: int = 0):
        super().__init__(message)
        self.kind, self.message, self.status = kind, message, status


class Pending(Exception):
    """This PC's device waits for the owner's approval; `code` is what the owner sees."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class Gone(Exception):
    """404: the server does not know this runner (it restarted, or the key was renamed) or the job (it finished)."""


class NetworkError(Exception):
    """The server could not be reached, or answered 5xx: try again after a while."""


def _header(text: str) -> str:
    """A header value http.client can send: printable ASCII only."""
    return "".join(c if 32 <= ord(c) < 127 else "?" for c in text)


def device_label(device: dict, app: str = "Strata runner") -> str:
    """The X-Strata-Device-Name the owner sees for this PC: "<app> on <machine>".  Strata Desktop passes its own app
    name (#310 part 3), so its page, its server test and its runner name one device the same way."""
    return _header(f"{app} on {device.get('name') or machine_name()}")


# #313: what a request on a kept connection meets when the server or a proxy closed it while it was kept: a reset or
# an abort, a FIN where the answer should be, and over TLS a FIN without close_notify (SSLEOFError) or a close_notify
# (SSLZeroReturnError).  A timeout is not one of them: a request that waited that long is not repeated.
STALE = (ConnectionError, http.client.RemoteDisconnected, ssl.SSLEOFError, ssl.SSLZeroReturnError)


class Client:
    """The runner's side of the channel.  Holds the delivery state of the current registration: `channel`, `ack`
    (the highest seq received on it) and `handled` (the seqs handled on it).  `label`: the device's name for the
    server (default `device_label(device)`, "Strata runner on <machine>")."""

    def __init__(self, server: str, key: str, device: dict, label: str | None = None):
        u = urlsplit(server.strip())
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ValueError(f"--server must be an http:// or https:// address, not {server!r}")
        self.url = f"{u.scheme}://{u.netloc}{u.path.rstrip('/')}"
        self._https, self._host, self._port = u.scheme == "https", u.hostname, u.port
        self._prefix = u.path.rstrip("/")
        self._headers = {"Authorization": "Bearer " + key, "X-Strata-Device": device["id"],
                         "X-Strata-Device-Name": device_label(device) if label is None else _header(label),
                         "User-Agent": "strata-runner/1"}
        self.channel: str | None = None
        self.ack = 0
        self.handled: set[int] = set()
        self._ssl = ssl.create_default_context() if self._https else None
        self._poll_lock = threading.Lock()
        self._poll_conn: http.client.HTTPConnection | None = None      # the GET /runner/next in flight
        self._polls_ended = False
        # #313: kept connections (HTTP/1.1): the poll's own, so a held poll never waits in front of a reply, and up to
        # KEEP_MAX for hellos and replies; each with the time it was last used
        self._keep_lock = threading.Lock()
        self._poll_kept: tuple[http.client.HTTPConnection, float] | None = None
        self._kept: list[tuple[http.client.HTTPConnection, float]] = []
        self._closed = False

    def close(self) -> None:
        """Close the kept connections.  A request after this still works, on a connection of its own."""
        with self._keep_lock:
            self._closed = True
            conns = [c for c, _ in self._kept] + ([self._poll_kept[0]] if self._poll_kept else [])
            self._kept, self._poll_kept = [], None
        for c in conns:
            c.close()

    def _drop_kept(self) -> None:
        """A kept connection was found closed by the server: so were the other idle ones, most likely (they all go to
        the same server, and one that stops closes every connection it kept), so none of them is tried."""
        with self._keep_lock:
            conns = [c for c, _ in self._kept] + ([self._poll_kept[0]] if self._poll_kept else [])
            self._kept, self._poll_kept = [], None
        for c in conns:
            c.close()

    def _take(self, poll: bool, timeout: float, fresh: bool = False) -> http.client.HTTPConnection:
        """A kept connection used within KEEP_IDLE_S (the poll's own for a poll), else - or with `fresh` - a new one:
        its socket is opened by the request."""
        now, conn, stale = time.monotonic(), None, []
        with self._keep_lock:
            if fresh:
                pass
            elif poll:
                if self._poll_kept is not None:
                    (conn, used), self._poll_kept = self._poll_kept, None
                    if now - used >= KEEP_IDLE_S:
                        conn, stale = None, [conn]
            else:
                while self._kept and conn is None:
                    c, used = self._kept.pop()
                    if now - used < KEEP_IDLE_S:
                        conn = c
                    else:
                        stale.append(c)
        for c in stale:
            c.close()
        if conn is None:
            conn = (http.client.HTTPSConnection(self._host, self._port, timeout=timeout, context=self._ssl)
                    if self._https else http.client.HTTPConnection(self._host, self._port, timeout=timeout))
        conn.timeout = timeout                          # http.client sets a socket's timeout only when it connects
        if conn.sock is not None:
            conn.sock.settimeout(timeout)
        return conn

    def _give_back(self, conn: http.client.HTTPConnection, poll: bool) -> None:
        """Keep `conn` for the next request, unless the server closes it (its answer said so) or enough are kept."""
        if conn.sock is not None:
            with self._keep_lock:
                if not self._closed and poll and self._poll_kept is None:
                    self._poll_kept = (conn, time.monotonic())
                    return
                if not self._closed and not poll and len(self._kept) < KEEP_MAX:
                    self._kept.append((conn, time.monotonic()))
                    return
        conn.close()

    def abort_poll(self) -> None:
        """End the poll in flight now, and every later poll at once, each as an empty poll (Strata Desktop's stop, #310
        part 3: its thread must not wait out a poll the server holds for up to POLL_MAX_S).  Replies are left alone,
        so cancelled jobs still send their results.  The loop then ends as on stop (`polls_ended`); the caller sets
        stop first.  `python -m serve.runner` never calls it."""
        with self._poll_lock:
            self._polls_ended = True
            conn = self._poll_conn
        sock = getattr(conn, "sock", None)
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)                     # enough elsewhere; on Windows a blocked read stays...
        except OSError:
            pass
        try:                                                    # ...until its handle is closed.  sock.close() would
            socket.close(sock.detach())                         # wait for the response's file to let go of it, so
        except OSError:                                         # the handle is taken from the object and closed:
            pass                                                # the read fails at once (WinError 10038)

    @property
    def polls_ended(self) -> bool:
        """abort_poll was called: RunnerLoop.run ends (it would otherwise go round with nothing to wait on)."""
        return self._polls_ended

    def _track_poll(self, conn: http.client.HTTPConnection) -> None:
        """Register the poll's connection, connect it, and check once more: an abort_poll that came in between (when
        the socket did not exist yet to be shut down) ends the poll here.  NetworkError when the polls ended."""
        with self._poll_lock:
            if self._polls_ended:
                raise NetworkError("the runner stopped")
            self._poll_conn = conn
        if conn.sock is None:                           # a kept connection is open already (#313)
            conn.connect()
        with self._poll_lock:
            if self._polls_ended:
                raise NetworkError("the runner stopped")

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float = 60.0,
                 most: int | None = None, poll: bool = False) -> tuple[int, object]:
        """(status, JSON body).  A body over `most` bytes is not sent: it is answered 413 here, as the server would.
        `poll`: the connection can be cut by `abort_poll`.  The connection is kept for the next request (#313); one
        that the server or a proxy closed while it was kept fails at once (STALE), the other kept ones are dropped
        with it, and the request goes once more on a new connection (safe for every request here: a reply's `rid`,
        the poll's ack and the hello make a repeat harmless)."""
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            if most is not None and len(data) > most:
                return 413, {"error": {"message": f"send a body of up to {most >> 20} MB"}}
        headers = dict(self._headers)
        if data is not None:
            headers["Content-Type"] = "application/json"
        for again in (False, True):
            conn = self._take(poll, timeout, fresh=again)
            kept = conn.sock is not None
            try:
                if poll:
                    self._track_poll(conn)
                conn.request(method, self._prefix + path, body=data, headers=headers)
                r = conn.getresponse()
                raw = r.read()
            except NetworkError:                        # abort_poll came first: the connection is not used again
                conn.close()
                raise
            except (OSError, http.client.HTTPException) as e:
                conn.close()
                if kept and not again and isinstance(e, STALE) and not (poll and self._polls_ended):
                    self._drop_kept()
                    continue                            # closed while kept: once more, on a new connection
                raise NetworkError(self.scrub(f"{type(e).__name__}: {e}")) from None
            except ValueError as e:                     # e.g. a header value http.client refuses: its text holds the
                conn.close()                            # value, so not shown
                raise NetworkError(f"{type(e).__name__} while sending the request") from None
            finally:
                if poll:
                    with self._poll_lock:
                        if self._poll_conn is conn:
                            self._poll_conn = None
            self._give_back(conn, poll)
            break
        try:
            got = json.loads(raw) if raw else None
        except ValueError:
            got = None
        return r.status, got

    @staticmethod
    def _check(code: int, body) -> dict:
        """The answer's body, or the exception its status means."""
        err = body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else {}
        kind, message = err.get("type"), str(err.get("message") or "")
        if code == 200:
            return body if isinstance(body, dict) else {}
        if code == 401:
            raise Refused("key", message or "the key is wrong or was revoked")
        if code == 403 and kind == "device_pending":
            raise Pending(str(err.get("code") or "----"))
        if code == 403 and kind == "device_refused":
            raise Refused("device_refused", message)
        if code == 409 and kind in ("runner_replaced", "runner_protocol"):
            raise Refused(kind, message)
        if code == 404:
            raise Gone(message)
        if code >= 500 or code in (408, 429):           # 408 / 429 (a proxy's timeout, a rate limit): try again
            raise NetworkError(f"the server answered {code}: {message}")
        raise Refused("refused", f"the server answered {code}: {message}", code)

    def scrub(self, text: str) -> str:
        """`text` with the key blanked out, for anything the runner prints."""
        key = self._headers["Authorization"][7:]
        return text.replace(key, "***") if key else text

    def on_hello(self, body: dict) -> bool:
        """A hello's answer: another channel than before starts the ack and the handled seqs again.  True then."""
        channel = body.get("channel")
        if channel == self.channel:
            return False
        self.channel, self.ack, self.handled = channel, 0, set()
        return True

    def hello(self, info: dict) -> dict:
        """POST /runner/hello {"protocol", **info} -> {"runner", "who", "protocol", "poll_s", "channel"}."""
        code, body = self._request("POST", "/runner/hello", {"protocol": runner_hub.RUNNER_PROTOCOL, **info})
        got = self._check(code, body)
        if not isinstance(got.get("runner"), str):
            raise NetworkError("the server's hello answer has no runner id")
        return got

    def next(self, rid: str, instance: str, wait: float) -> list[dict]:
        """GET /runner/next: the messages for this runner (held at most `wait` seconds), with this channel's ack."""
        q = f"id={quote(rid)}&instance={quote(instance)}&wait={wait:g}&channel={quote(self.channel or '')}&ack={self.ack}"
        if self._polls_ended:
            return []
        try:
            code, body = self._request("GET", "/runner/next?" + q, timeout=wait + 30, poll=True)
        except NetworkError:
            if self._polls_ended:                       # cut by abort_poll: an empty poll, not a failure
                return []
            raise
        msgs = self._check(code, body).get("messages")
        return [m for m in msgs if isinstance(m, dict)] if isinstance(msgs, list) else []

    def reply(self, rid: str, instance: str, msg: dict) -> None:
        """POST /runner/reply {"id", "instance", **msg} (msg has "job" and "rid")."""
        code, body = self._request("POST", "/runner/reply", {"id": rid, "instance": instance, **msg}, most=REPLY_MAX)
        self._check(code, body)


# ------------------------------------------------------------------------------------------------ one job
class JobLink:
    """One job's way back to the server while it runs: `send` posts a reply (it gets "job" and the next "rid", from 1;
    the posting callable gives False when the server did not take it),
    `ask` / `question` send the card and wait for the answer whose `to` is that reply's rid, `emit` sends tool activity,
    and `cancel` is set when the server cancels the job (or the runner stops)."""

    def __init__(self, job: str, send: Callable[[dict], None]):
        self.job = job
        self._post = send
        self.cancel = threading.Event()
        self.inbox: queue.Queue = queue.Queue()
        self._rid = 0
        self._lock = threading.Lock()
        self._answers: dict[int, object] = {}

    def send(self, more: dict) -> int:
        """Post a reply -> its rid.  A reply the server did not take (`send` gave False: the job is over there) ends
        the job here too: `cancel` is set, so an ask or question waiting for its answer returns "cancelled"."""
        with self._lock:
            self._rid += 1
            rid = self._rid
        if self._post({"job": self.job, "rid": rid, **more}) is False:
            self.cancel.set()
        return rid

    def deliver(self, msg: dict) -> None:
        """An answer (or a wake-up) from the poll thread."""
        self.inbox.put(msg)

    def _wait(self, rid: int):
        while True:
            if rid in self._answers:
                return self._answers.pop(rid)
            if self.cancel.is_set():
                return "cancelled"
            try:
                msg = self.inbox.get(timeout=0.25)
            except queue.Empty:
                continue
            to = msg.get("to")
            if type(to) is int and "answer" in msg:
                self._answers[to] = msg["answer"]

    def ask(self, card: dict) -> str:
        """A permission card: "allow" | "allow_chat" | "deny" | "blocked: ..." | "cancelled"."""
        got = self._wait(self.send({"ask": card}))
        return got if isinstance(got, str) else "deny"

    def question(self, questions: list):
        """AskUserQuestion: {"answers": ...} or "cancelled"."""
        return self._wait(self.send({"question": questions}))

    def emit(self, event) -> None:
        if isinstance(event, dict):
            self.send({"emit": event})


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _unsent(result, status: int):
    """What is sent instead of a job's result the server refused (413: over its size limit; 400), in the result's own
    shape, so the call on the server ends with text instead of waiting."""
    why = ("the result was too large to send to the server" if status == 413
           else f"the server refused the result ({status or 'refused'})")
    text = f"{why}; nothing of it reached the model. Try a smaller request (a part of the file, fewer lines)."
    if isinstance(result, dict) and "content" in result:
        return _err(text)
    if isinstance(result, dict) and "status" in result:
        return {"status": 400, "body": {"error": {"message": text}}}
    return {"error": "failed", "message": text}


def _rules(v) -> list[str]:
    return [x for x in v if isinstance(x, str) and 0 < len(x) <= 500][:200] if isinstance(v, list) else []


def _session(v) -> str:
    return v if isinstance(v, str) and 0 < len(v) <= 80 else "default"


# ------------------------------------------------------------------------------------------------ this PC's MCP servers (#338)
MCP_JOBS = ("mcp.config", "mcp.config.set", "mcp.import.list", "mcp.import.add")
PENDING_KEY = "mcp_pending"     # runner.json: {name: {"entry", "at"}} - a server the web app added, changed or switched on,
                                # which waits for the user's approval in Strata Desktop on this PC before it runs


def pending_in(cfg) -> dict:
    """runner.json's servers waiting for approval: {name: {"entry": {...}, "at": seconds}} (anything else is left out)."""
    block = cfg.get(PENDING_KEY) if isinstance(cfg, dict) else None
    if not isinstance(block, dict):
        return {}
    return {n: {"entry": v["entry"], "at": float(v.get("at") or 0)} for n, v in block.items()
            if isinstance(n, str) and isinstance(v, dict) and isinstance(v.get("entry"), dict)}


def command_line(entry: dict) -> str:
    """How a server starts, in one line (the address of one that is reached by URL), for the approval and its notice."""
    if entry.get("url"):
        return str(entry["url"])
    return " ".join([str(entry.get("command") or "")] + [str(a) for a in entry.get("args") or []]).strip()


class _Refused(Exception):
    def __init__(self, status: int, body: dict):
        super().__init__(status)
        self.status, self.body = status, body


def _bad(field: str, message: str, server: str | None = None) -> dict:
    """A refused change in the server route's shape (serve/mcp_admin.py apply): a 400 with the field errors."""
    return {"status": 400, "body": {"error": {"message": f"{message} ({server or 'settings'}: {field})",
                                              "fields": [{"server": server, "field": field, "message": message}]}}}


class _Admin:
    """This runner's own run config as serve/mcp_admin.py `view` and serve/harness.py `view` read a server's (`svc`):
    `config_path` is runner.json, no --mcp-config, no API key (the server decided who may ask), the PC's importer and
    hub.  Read only: the runner writes runner.json itself (`Executor._write_servers`), so that a change from the web app
    waits for the approval on this PC."""

    api_key = None
    mcp_config_path = None

    def __init__(self, executor: "Executor"):
        self._ex = executor

    @property
    def config_path(self) -> str:
        return str(self._ex.config_path)

    @property
    def importer(self):
        return self._ex.importer

    @property
    def mcp(self):
        return self._ex.hub


# ------------------------------------------------------------------------------------------------ the tools of this PC
class Executor:
    """What runs on this PC: its own coding tools (shell.install only: no web, no Task - those run on the server), its
    MCP servers and skills (from `config_path`), its checkpoints (`data_dir`/checkpoints), and the page routes."""

    def __init__(self, home: Path, config_path: Path, data_dir: Path):
        self.home, self.config_path, self.data_dir = Path(home), Path(config_path), Path(data_dir)
        self.cfg = self._read_config()
        self.machine = machine_name()
        self.agent = agent_mod.AgentServer()
        self.agent.shell = None
        shell_mod.install(self.agent, shell_mod.find_shell())
        self.importer = harness.Importer(home=self.home)
        self.importer.rescan(self.cfg)
        self.on_change: Callable[[dict], None] | None = None   # #338: RunnerLoop's: the web app changed the MCP servers
        self.on_groups: Callable[[], None] | None = None       # #338: RunnerLoop's: the groups may differ (say hello)
        self._last_tools: dict[str, list] = {}                  # #338: each MCP server's tools when it was last ready
        self.hub = None
        self._take_hub(hub_from_config(self.cfg, None, builtins=self.importer.builtins()))
        if self.hub is not None:
            self._start_servers(self.hub, list(self.hub.servers.values()))
        self.admin = _Admin(self)
        self.checkpoints = checkpoints_mod.Checkpoints(self.data_dir / "checkpoints")
        self.protected = [str(self.config_path), str(self.config_path) + ".bak-mcp",       # #338: the web app's first change keeps
                          str(self.data_dir / "device.json")]                               # a copy beside it (its secrets too)
        self.runs: collections.OrderedDict[str, tuple] = collections.OrderedDict()     # run key -> (Policy, Scope | None)
        self.lock = threading.Lock()

    def _read_config(self) -> dict:
        try:
            with open(self.config_path, encoding="utf-8-sig") as f:
                cfg = json.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as e:
            raise SystemExit(f"[runner] {self.config_path} cannot be read: {e}") from None
        if not isinstance(cfg, dict):
            raise SystemExit(f"[runner] {self.config_path} must be a JSON object")
        return cfg

    def _take_hub(self, hub) -> None:
        """`hub` (not started yet) becomes this PC's: a server named as the coding tools is left out."""
        if hub is not None and CODING in hub.servers:
            _say(f"[runner] the MCP server {CODING!r} in {self.config_path} is left out: that name is the coding tools'")
            hub.servers.pop(CODING)
        self.hub = hub

    def _start_servers(self, hub, servers: list) -> None:
        """Start `servers` of `hub`, each on its own thread (as McpHub.start); each one that ends starting (ready or
        failed) tells the loop, which says hello if the groups differ - no polling (#338)."""
        def run(srv):
            try:
                srv.start()
            finally:
                self._groups_moved()
        hub.threads = [threading.Thread(target=run, args=(srv,), daemon=True, name=f"mcp-{srv.name}") for srv in servers]
        for t in hub.threads:
            t.start()

    def _groups_moved(self) -> None:
        if self.on_groups is not None:
            try:
                self.on_groups()
            except Exception as e:  # noqa: BLE001 - the next poll says hello anyway
                _say(f"[runner] announcing the MCP servers raised {type(e).__name__}")

    def _reload(self) -> None:
        """#338: the hub follows runner.json after a change, restarting only what changed: a server whose entry is the
        same keeps running (its sessions and the calls in flight go on), a removed or changed one is closed, a new or
        changed one is started."""
        try:
            new = hub_from_config(self.cfg, None, builtins=self.importer.builtins())
        except SystemExit as e:                         # checked before it was written: a file edited by hand
            raise ValueError(str(e)) from None
        old, kept = self.hub, set()
        if new is not None and old is not None:
            for name, srv in list(new.servers.items()):
                was = old.servers.get(name)
                if isinstance(srv, McpServer) and isinstance(was, McpServer) and was.cfg == srv.cfg:
                    was.settings = new.settings
                    new.servers[name] = was
                    kept.add(name)
        self._take_hub(new)
        if old is not None:
            for name, was in old.servers.items():
                if isinstance(was, McpServer) and name not in kept:
                    was.close()
        if new is not None:
            self._start_servers(new, [srv for n, srv in new.servers.items() if n not in kept])
        self._groups_moved()

    def close(self) -> None:
        """Background commands and MCP servers end with the runner."""
        self.agent.close()
        if self.hub is not None:
            self.hub.close()

    # -------------------------------------------------------------------------------------------- the hello
    def info(self) -> dict:
        sh = self.agent.shell
        return fit_info({"machine": self.machine, "platform": sys.platform,
                         "shell": shell_mod.describe(sh) if sh else None, "home": str(self.importer.home)})

    def groups(self) -> list[dict]:
        """The coding tools, then the skills (when some are in use), then each MCP server, as their state is now."""
        raw = [{"name": CODING, "kind": "coding", "status": "ready", "error": None,
                "tools": [t for t in self.agent.tools if t["name"] not in runner_hub.SERVER_TOOLS]}]
        servers = dict(self.hub.servers) if self.hub is not None else {}
        for name in [n for n in self._last_tools if n not in servers]:
            self._last_tools.pop(name, None)
        for name, s in servers.items():
            kind = "skills" if name == "skills" and getattr(s, "kind", "") == "builtin" else "mcp"
            if kind == "skills" and not s.tools:
                continue
            tools = list(s.tools)
            if kind == "mcp" and tools and s.status == "ready":
                self._last_tools[name] = tools
            elif kind == "mcp" and not tools and s.status in ("idle", "starting"):
                tools = self._last_tools.get(name, [])  # #338: one that restarts keeps its tools offered (a call waits)
            raw.append({"name": name, "kind": kind, "status": s.status, "error": s.error, "tools": tools})
        return fit_groups(raw)

    def skills(self) -> list[dict]:
        return fit_skills(harness.select_skills(self.importer.found, harness.skill_settings(self.cfg)))

    def starting(self) -> bool:
        """An MCP server is still starting: its tools are not known yet."""
        return self.hub is not None and any(getattr(s, "status", "") in ("idle", "starting") for s in self.hub.servers.values())

    # -------------------------------------------------------------------------------------------- jobs
    def handle(self, msg: dict, link: JobLink):
        """A job's result: run.start -> the facts, tool -> an MCP result, page -> {"status", "body"}."""
        kind = msg.get("kind")
        if kind == "run.start":
            return self._run_start(msg)
        if kind == "tool":
            return self._tool(msg, link)
        if kind == "page":
            return self._page(msg)
        if kind in MCP_JOBS:
            return self._mcp(kind, msg)
        return {"error": "unknown", "message": f"this runner does not know the job {str(kind)[:40]!r}: update Strata Desktop"}

    def _run_start(self, msg: dict) -> dict:
        """A message of a chat begins: the real folders, the shell, git, the notes; its policy and checkpoint are kept
        under a new run key (the tools' session state is the chat's, by `session`)."""
        folder, dirs = folders_mod.resolve(msg.get("cwd"), msg.get("dirs"), agent_mod.MAX_DIRS)
        mode = permissions.mode_for(msg.get("mode"), on_runner=True)        # this PC is the user's own: the server already gated a guest's YOLO (#329)
        session = _session(msg.get("session"))
        policy = permissions.Policy(cwd=folder, mode=mode, allow=_rules(msg.get("allow")), deny=_rules(msg.get("deny")),
                                    dirs=dirs, protected=list(self.protected))
        cp = msg.get("checkpoint")
        scope = self.checkpoints.scope(session, cp) if isinstance(cp, str) and session != "default" else None
        run = "run-" + uuid.uuid4().hex
        with self.lock:
            self.runs[run] = (policy, scope)
            while len(self.runs) > RUNS_KEPT:
                self.runs.popitem(last=False)
        on = [x for x in msg.get("memory_on") or [] if isinstance(x, str)] if isinstance(msg.get("memory_on"), list) else []
        blocks = []
        if folder:
            try:
                got = memory_mod.collect(str(self.home), [folder, *dirs], on)
            except Exception as e:  # noqa: BLE001 - notes that cannot be read are no reason not to start
                _say(f"[runner] the project's notes could not be read: {type(e).__name__}: {e}")
                got = []
            for b in got[:200]:
                blocks.append({**{k: str(b[k])[:100] for k in ("id", "kind", "label") if b.get(k) is not None},
                               "name": str(b.get("name") or "notes")[:500], "text": b.get("text") or "",
                               "cut": b.get("cut") is True})
        sh = self.agent.shell
        return {"run": run, "cwd": folder, "dirs": dirs, "shell": shell_mod.describe(sh) if sh else None,
                "platform": sys.platform, "git": folders_mod.in_git(folder), "memory": blocks,
                "tools": [t["name"] for t in self.agent.tools if t["name"] not in runner_hub.SERVER_TOOLS]}

    def _tool(self, msg: dict, link: JobLink) -> dict:
        group, tool = msg.get("group"), msg.get("tool")
        args = msg.get("arguments") if isinstance(msg.get("arguments"), dict) else {}
        if not isinstance(tool, str):
            return _err("the job names no tool")
        if group == CODING:
            run = msg.get("run")
            with self.lock:
                entry = self.runs.get(run) if isinstance(run, str) else None
                if entry is not None:
                    self.runs.move_to_end(run)
            if entry is None:
                return _err(f"{tool} was not run: this chat's message is not known on the PC (its runner restarted, or "
                            f"many other messages came since). Tell the user to send the message again.")
            policy, scope = entry
            helper = msg.get("helper") is True
            question = link.question
            if helper:                                  # a helper: its own mode, and nobody to answer it questions
                policy = copy.copy(policy)
                policy.mode = permissions.mode_for(msg.get("mode"), on_runner=True)
                # allow / deny stay the run's lists: "allow for this chat" holds for the helper's next calls too,
                # as on the server (serve/subagent.py)
                policy.dirs, policy.protected = list(policy.dirs), list(policy.protected)
                question = lambda qs: {"answers": None}          # noqa: E731
            ctx = agent_mod.AgentContext(policy=policy, session=_session(msg.get("session")), ask=link.ask,
                                         cancel=link.cancel, emit=link.emit, question=question, checkpoint=scope,
                                         vision=msg.get("vision") is True, in_helper=helper, run_key=run)
            return self.agent.call(tool, args, 0, link.cancel, ctx)
        server = self.hub.servers.get(group) if self.hub is not None and isinstance(group, str) else None
        if server is None:
            return _err(f"there is no tool group {str(group)[:80]!r} on this PC")
        t = msg.get("timeout_s")
        timeout = float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) and t > 0 \
            else float(self.hub.settings["timeout_s"])
        try:
            res = server.call(tool, args, timeout, link.cancel)
        except McpCancelled:
            return _err("The request was cancelled.")
        except McpError as e:
            return _err(str(e))
        return res if isinstance(res, dict) else _err(f"{tool} gave no result")

    # -------------------------------------------------------------------------------------------- MCP servers (#338)
    def _mcp(self, kind: str, msg: dict) -> dict:
        """The web app's MCP editor for this PC, in a page job's shape ({"status", "body"}).  `mcp.config` and
        `mcp.config.set` are GET and POST /mcp/config on runner.json; `mcp.import.list` and `mcp.import.add` are GET and
        POST /import, for the MCP servers of the other apps on this PC only (their skills and notes are not changed
        here).  The checks and the masks are serve/mcp_admin.py's and serve/harness.py's own.

        A server the web app adds, changes or switches on does not run: it waits in runner.json's `mcp_pending` until
        the user approves it in Strata Desktop on this PC (`mcp_decide`) - a program the server could otherwise plant
        here.  Until then a changed one keeps running as it was approved.  Switching one off or deleting it is done
        at once."""
        body = msg.get("body")
        if kind == "mcp.config":
            return {"status": 200, "body": self._mcp_view()}
        if kind == "mcp.import.list":
            return {"status": 200, "body": self._import_view(rescan=msg.get("rescan") is True)}
        if kind == "mcp.config.set":
            if not isinstance(body, dict) or not isinstance(body.get("servers"), dict):
                return _bad("servers", 'send {"servers": {name: {...}}}')
            if CODING in body["servers"]:
                return _bad("name", f"{CODING!r} is the coding tools' name on this PC: give the server another name", CODING)
            if mcp_admin.GUEST_KEY in body:
                return _bad(mcp_admin.GUEST_KEY, "guests never use the MCP servers of your PC")
            status, out = self._write_servers(body["servers"], body.get("settings", {}))
            return {"status": status, "body": out if status != 200 else self._mcp_view()}
        if not isinstance(body, dict) or "import_mcp" not in body or any(k not in ("import_mcp", "rescan") for k in body):
            return _bad("import_mcp", 'send {"import_mcp": {"harness", "name"}}: only an MCP server is imported here')
        want = body["import_mcp"]
        if not (isinstance(want, dict) and isinstance(want.get("harness"), str) and isinstance(want.get("name"), str)
                and want["name"]):
            return _bad("import_mcp", "import_mcp is {\"harness\": ..., \"name\": ...}")
        if body.get("rescan") is True:
            self.importer.rescan(self.cfg)
        approved, pend = self._config_now()
        own = {**approved, **{n: p["entry"] for n, p in pend.items()}}
        try:
            servers, _name = harness.import_server(own, want["harness"], want["name"], self.importer.found,
                                                   self.importer.environ)
        except harness.ImportRefused as e:
            return {"status": {"missing": 404, "unsupported": 422, "already": 409}[e.code],
                    "body": {"error": {"message": str(e), "code": e.code}}}
        status, out = self._write_servers(servers, {})
        return {"status": status, "body": out if status != 200 else self._import_view()}

    def _config_now(self) -> tuple[dict, dict]:
        """(the approved MCP servers, the ones waiting for approval) as runner.json has them now."""
        try:
            cfg = self._read_config()
        except SystemExit:
            return {}, {}
        return mcp_admin._servers_in(cfg), pending_in(cfg)

    def _write_servers(self, sent: dict, settings_sent) -> tuple[int, dict]:
        """Write the list the web app sent (`sent`: every server, by name, as POST /mcp/config takes it) into
        runner.json, under mcp_admin's lock: each entry checked and its masked secrets put back (mcp_admin), the other
        keys kept, atomically.  An entry that would start a program it did not start before - a new one, a changed one,
        one switched on - goes to `mcp_pending` instead; the approved entry (if any) stays.  Then the hub follows
        (`_reload`), and the loop hears what changed.  (status, body): 400 with the field errors, as the route gives."""
        if len(sent) > mcp_admin.MAX_SERVERS:
            return 400, {"error": {"message": f"at most {mcp_admin.MAX_SERVERS} servers", "fields": []}}
        seen: dict = {}

        def mutate(cfg: dict) -> None:
            approved, pend = mcp_admin._servers_in(cfg), pending_in(cfg)
            errors, final, waiting = [], {}, {}
            for name, entry in sent.items():
                base = pend[name]["entry"] if name in pend else approved.get(name)
                if isinstance(entry, dict) and isinstance(name, str):
                    entry = mcp_admin.merge_masked(entry, base if isinstance(base, dict) else None)
                entry, errs = mcp_admin.validate_entry(name, entry)
                errors += [{"server": name, **e} for e in errs]
                if errs:
                    continue
                old = approved.get(name) if isinstance(approved.get(name), dict) else None
                if entry.get("disabled") is True or entry == old:
                    final[name] = entry                 # off, or as approved: nothing new runs
                    continue
                if old is not None:
                    final[name] = old                   # it keeps running as approved until the user decides
                was = pend.get(name)
                waiting[name] = {"entry": entry, "at": was["at"] if was and was["entry"] == entry else time.time()}
            settings, errs = mcp_admin.validate_settings(settings_sent if settings_sent is not None else {})
            errors += [{"server": None, **e} for e in errs]
            if errors:
                first = errors[0]
                raise _Refused(400, {"error": {"message": f"{first['message']} ({first['server'] or 'settings'}: "
                                                          f"{first['field']})", "fields": errors}})
            seen.update(before=(approved, pend), after=(final, waiting))
            key = "mcpServers" if "mcpServers" in cfg and "mcp_servers" not in cfg else "mcp_servers"
            cfg.pop("mcpServers" if key == "mcp_servers" else "mcp_servers", None)
            cfg[key] = final
            if settings:
                cfg["mcp"] = {**(cfg["mcp"] if isinstance(cfg.get("mcp"), dict) else {}), **settings}
            if waiting:
                cfg[PENDING_KEY] = waiting
            else:
                cfg.pop(PENDING_KEY, None)

        with mcp_admin._lock:
            self._make_config()
            try:
                mcp_admin.update_config(self.config_path, mutate)
            except _Refused as e:
                return e.status, e.body
            except (OSError, ValueError, TypeError) as e:
                return 500, {"error": {"message": f"the servers could not be saved: {e}"}}
            self.cfg = self._read_config()
            try:
                self._reload()
            except ValueError as e:
                return 500, {"error": {"message": f"the servers could not be started: {e}"}}
        self._changed(*seen["before"], *seen["after"])
        return 200, {}

    def mcp_pending(self) -> list[dict]:
        """The servers waiting for the user's approval on this PC, for Strata Desktop's window: the name, how it starts
        (command, args, cwd, URL - in full), the names of its environment variables and headers (never their values),
        whether it replaces an approved one, and when it was asked."""
        approved, pend = self._config_now()
        out = []
        for name, item in sorted(pend.items()):
            e = item["entry"]
            out.append({"name": name, "line": command_line(e), "command": e.get("command"), "args": list(e.get("args") or []),
                        "cwd": e.get("cwd"), "url": e.get("url"), "env": sorted(e.get("env") or {}),
                        "headers": sorted(e.get("headers") or {}), "replaces": name in approved, "at": item["at"]})
        return out

    def mcp_decide(self, name: str, approve: bool) -> bool:
        """The user's answer in Strata Desktop for a waiting server: approved, it becomes the server of that name and
        starts (only it); rejected, it is dropped (an approved one of that name keeps running).  False when nothing of
        that name waits."""
        found = []

        def mutate(cfg: dict) -> None:
            pend = pending_in(cfg)
            item = pend.pop(name, None)
            if item is None:
                return
            found.append(item)
            if approve:
                key = "mcpServers" if "mcpServers" in cfg and "mcp_servers" not in cfg else "mcp_servers"
                block = cfg.get(key) if isinstance(cfg.get(key), dict) else {}
                block[name] = item["entry"]
                cfg[key] = block
            if pend:
                cfg[PENDING_KEY] = pend
            else:
                cfg.pop(PENDING_KEY, None)

        with mcp_admin._lock:
            if not self.config_path.exists():
                return False
            mcp_admin.update_config(self.config_path, mutate)
            if not found:
                return False
            self.cfg = self._read_config()
            self._reload()
        _say(f"[runner] the MCP server {name!r} was {'approved' if approve else 'rejected'} on this PC")
        return True

    def _mcp_view(self) -> dict:
        """GET /mcp/config's answer for this PC, with each server that waits for approval as a row of its own entry
        (`pending`: true, state "pending"), so a save from the page keeps it waiting."""
        v = mcp_admin.view(self.admin, "", "", may=(True, ""))
        _approved, pend = self._config_now()
        rows = v["servers"]
        for name, item in pend.items():
            e = item["entry"]
            was = next((r for r in rows if r["name"] == name), None)
            row = {"name": name, "kind": mcp_admin._kind(e), **mcp_admin.public_entry(e), "source": "config", "editable": True,
                   "overridden": False, "disabled": False, "pending": True, "status": "pending", "error": None,
                   "tools": was["tools"] if was else [], "info": {}}
            if was is not None:
                rows[rows.index(was)] = row
            else:
                rows.append(row)
        return {**v, "machine": self.machine}

    def _import_view(self, rescan: bool = False) -> dict:
        v = harness.view(self.admin, "", "", rescan=rescan, may=(True, ""))
        return {**{k: v[k] for k in ("available", "editable", "reason", "config_file", "harnesses", "mcp") if k in v},
                "machine": self.machine}

    def _make_config(self) -> None:
        """runner.json for the first change, when there is none yet (the runner runs without one)."""
        if self.config_path.exists():
            return
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.config_path.with_name(self.config_path.name + ".new")
        tmp.write_text("{}\n", encoding="utf-8")
        os.replace(tmp, self.config_path)

    def _changed(self, approved: dict, pend: dict, final: dict, waiting: dict) -> None:
        """Tell the loop what the web app changed (read under the lock): {"added", "removed", "changed"} of the approved
        servers, and "pending" [{"name", "line"}]: the ones that newly wait for approval, with how they start."""
        if self.on_change is None:
            return
        change = {"added": sorted(n for n in final if n not in approved),
                  "removed": sorted(n for n in approved if n not in final),
                  "changed": sorted(n for n in final if n in approved and final[n] != approved[n]),
                  "pending": [{"name": n, "line": command_line(w["entry"])} for n, w in sorted(waiting.items())
                              if pend.get(n, {}).get("entry") != w["entry"]]}
        try:
            self.on_change(change)
        except Exception as e:  # noqa: BLE001 - the change is made; telling of it must not undo the answer
            _say(f"[runner] telling of the MCP change raised {type(e).__name__}")

    def _page(self, msg: dict) -> dict:
        q = msg.get("query") if isinstance(msg.get("query"), dict) else {}
        query = {k: [x for x in v if isinstance(x, str)] for k, v in q.items() if isinstance(k, str) and isinstance(v, list)}
        on = [x for x in msg.get("memory_on") or [] if isinstance(x, str)] if isinstance(msg.get("memory_on"), list) else []
        status, body = page_routes.answer(str(msg.get("path")), query, msg.get("body"), home=str(self.importer.home),
                                          memory_on=on, checkpoints=self.checkpoints)
        return {"status": status, "body": body}


# ------------------------------------------------------------------------------------------------ the loop
class RunnerLoop:
    """One poll thread (hello, poll, hand out), one thread per job (at most MAX_JOBS at once, the others wait)."""

    def __init__(self, client: Client, executor, *, stop: threading.Event | None = None,
                 on_state: Callable[[str, dict], None] | None = None):
        """`on_state(event, data)` (Strata Desktop's tray, #310 part 3; None for `python -m serve.runner`): "online"
        {"id", "who"} after each hello that succeeds and when a poll works again after network errors; "pending"
        {"code"} on each Pending; "offline" {"message"} (scrubbed) on the first network error of a run of them;
        "refused" {"kind", "message"} before run returns EXIT_REFUSED; "stopped" {} when run returns 0; "mcp_changed"
        {"added", "removed", "changed"} (lists of server names) when the web app changed this PC's MCP servers (#338).
        It is called on the poll thread (on a job's thread for "mcp_changed"); an exception from it is logged and
        ignored."""
        self.client, self.executor = client, executor
        self.on_state = on_state
        self.stop = stop or threading.Event()
        self.instance = new_instance()
        self.id: str | None = None
        self.who: str | None = None
        self.poll_s = runner_hub.POLL_MAX_S
        self.links: dict[str, JobLink] = {}
        self.slots = threading.BoundedSemaphore(MAX_JOBS)
        self._lock = threading.Lock()
        self._sent_groups: str | None = None
        self.connected = threading.Event()              # a hello was answered, and no poll said 404 since
        self._threads: set[threading.Thread] = set()    # the job threads still running
        self._printed: str | None = None
        self._said_hello = threading.Condition()        # notified after each answered hello: a reply that met 404
                                                        # goes again at once under the new id (#315)
        self._hello_lock = threading.Lock()             # #338: a hello from the poll thread or after an MCP change
        if hasattr(executor, "on_change"):
            executor.on_change = self._mcp_changed
        if hasattr(executor, "on_groups"):
            executor.on_groups = self._announce

    # -------------------------------------------------------------------------------------------- hello
    def _info(self) -> tuple[dict, str]:
        groups, skills = self.executor.groups(), self.executor.skills()
        sig = json.dumps([groups, skills], sort_keys=True, default=str)
        return {"instance": self.instance, **self.executor.info(), "groups": groups, "skills": skills}, sig

    def _hello(self) -> None:
        with self._hello_lock:
            self._hello_now()

    def _hello_now(self) -> None:
        info, sig = self._info()
        body = self.client.hello(info)
        if self.client.on_hello(body):
            self._channel_changed()
        with self._said_hello:
            self.id, self.who, self._sent_groups = body["runner"], body.get("who"), sig
            self.connected.set()
            self._said_hello.notify_all()
        p = body.get("poll_s")
        self.poll_s = float(p) if isinstance(p, (int, float)) and not isinstance(p, bool) and p > 0 else runner_hub.POLL_MAX_S
        if self._printed != self.id:
            self._printed = self.id
            _say(f"[runner] connected to {self.client.url} as {self.who}; this PC's runner id is {self.id} - paste it "
                 f"in the web app under Settings > This browser's runner")
        self._state("online", {"id": self.id, "who": self.who})

    def _state(self, event: str, data: dict) -> None:
        if self.on_state is None:
            return
        try:
            self.on_state(event, data)
        except Exception as e:  # noqa: BLE001 - the app's view of the runner must never stop the runner
            _say(f"[runner] the state callback raised {type(e).__name__} on {event!r}; the runner goes on")

    def _mcp_changed(self, change: dict) -> None:
        """#338: the web app changed this PC's MCP servers (on a job's thread): the state callback hears "mcp_changed"
        {"added", "removed", "changed", "pending"} - Strata Desktop's notice, and its approval window for "pending"."""
        self._state("mcp_changed", change)

    def _announce(self) -> None:
        """#338: the groups may have changed (a change was written, a server ended starting): say hello now if they
        differ from the last hello - the poll in flight may be held POLL_MAX_S.  On its own thread: never the caller's."""
        threading.Thread(target=self._announce_now, daemon=True, name="runner-announce").start()

    def _announce_now(self) -> None:
        try:
            if self.connected.is_set() and not self.stop.is_set() and self._groups_changed():
                self._hello()
        except Exception as e:  # noqa: BLE001 - the poll thread says hello at its next round anyway
            _say(f"[runner] the hello after the MCP change failed ({type(e).__name__}); the next poll says it")

    def _channel_changed(self) -> None:
        """A new registration on the server: the jobs of the old one ended there, so their links are cancelled."""
        with self._lock:
            links = list(self.links.values())
        for link in links:
            link.cancel.set()

    def _groups_changed(self) -> bool:
        return self._info()[1] != self._sent_groups

    # -------------------------------------------------------------------------------------------- polling
    def run(self) -> int:
        """Until `stop`: 0; EXIT_REFUSED when the server refuses this runner."""
        failures, gones, pending, pending_since = 0, 0, None, None
        while not self.stop.is_set() and not getattr(self.client, "polls_ended", False):    # abort_poll: no spin
            said_hello = False
            try:
                if not self.connected.is_set() or self._groups_changed():
                    self._hello()
                    pending, pending_since, said_hello = None, None, True
                wait = min(self.poll_s, STARTING_POLL_S) if self.executor.starting() else self.poll_s
                msgs = self.client.next(self.id, self.instance, wait)
                if failures:
                    _say("[runner] the server answers again")
                    if not said_hello:                  # the hello just said "online"
                        self._state("online", {"id": self.id, "who": self.who})
                failures = gones = 0
                self._dispatch(msgs)
            except Gone:                                # the server restarted, or the key was renamed: hello again
                self.connected.clear()
                gones += 1
                if gones > 1:                           # a 404 again right after a hello: never spin
                    self.stop.wait(min(RETRY_MAX_S, 2.0 ** (gones - 2)))
            except Pending as e:
                self.connected.clear()
                now = time.monotonic()
                if pending_since is None:
                    pending_since = now
                if pending != e.code:
                    pending = e.code
                    _say(f"[runner] this PC waits for the owner's approval: the owner sees it with the code {e.code}. "
                         f"Trying again every {PENDING_RETRY_S:g} s (every {PENDING_SLOW_S:g} s after "
                         f"{PENDING_FAST_S / 60:g} min).")
                self._state("pending", {"code": e.code})      # each time: a network error may have come in between
                self.stop.wait(PENDING_RETRY_S if now - pending_since < PENDING_FAST_S else PENDING_SLOW_S)
            except Refused as e:
                why = {"key": "the key was refused (wrong or revoked)",
                       "device_refused": "the owner refused this PC as a device",
                       "runner_replaced": "a newer runner on this PC took this one's place",
                       "runner_protocol": "this runner and the server speak different versions"}.get(e.kind, "the server refused the runner")
                _say(f"[runner] stopped: {why}" + (f" ({e.message})" if e.message else ""))
                self._end()
                self._state("refused", {"kind": e.kind, "message": self.client.scrub(e.message)})
                return EXIT_REFUSED
            except NetworkError as e:
                delay = min(RETRY_MAX_S, 2.0 ** failures)
                message = self.client.scrub(str(e))
                if failures == 0:
                    _say(f"[runner] the server cannot be reached ({message}); trying again")
                if failures == 0 or said_hello:         # a hello in this round said "online" while failures went on
                    self._state("offline", {"message": message})
                failures += 1
                self.stop.wait(delay)
            except Exception as e:  # noqa: BLE001 - a bug here must not end the runner silently, nor print the key
                delay = min(RETRY_MAX_S, 2.0 ** failures)
                _say(f"[runner] an unexpected {type(e).__name__} in the poll thread; trying again in {delay:g} s")
                failures += 1
                self.connected.clear()
                self.stop.wait(delay)
        self._end()
        self._state("stopped", {})
        return 0

    def _end(self) -> None:
        with self._lock:
            links = list(self.links.values())
        for link in links:
            link.cancel.set()

    def finish(self, timeout: float) -> bool:
        """The runner stops: every job is cancelled (a command's process tree is ended), and its thread gets up to
        `timeout` seconds to end before the executor closes - so nothing started here keeps running on the PC.
        True when every job thread ended (`main` does not need the answer; Strata Desktop's stop reports it)."""
        self.stop.set()
        self._end()
        end = time.monotonic() + timeout
        with self._lock:
            threads = list(self._threads)
        for t in threads:
            t.join(max(0.0, end - time.monotonic()))
        close = getattr(self.client, "close", None)     # the kept connections (#313); a late reply opens its own
        if close is not None:
            close()
        with self._lock:
            return not any(t.is_alive() for t in self._threads)

    def _dispatch(self, msgs: list[dict]) -> None:
        """Hand the poll's messages out: a job starts, an answer reaches its job's link, a cancel sets it.  A seq
        already handled is ignored (delivery is at least once); the ack is the highest seq received."""
        c = self.client
        top = c.ack
        for m in msgs:
            seq = m.get("seq")
            if type(seq) is int:
                if seq <= c.ack or seq in c.handled:
                    continue
                c.handled.add(seq)
                top = max(top, seq)
            job = m.get("job")
            if not isinstance(job, str):
                continue
            if "kind" in m:
                self._start(m)
                continue
            with self._lock:
                link = self.links.get(job)
            if link is None:
                continue
            if m.get("cancel"):
                link.cancel.set()
                link.deliver({})
            elif "answer" in m:
                link.deliver(m)
        c.ack = top
        if len(c.handled) > 4096:                       # every seq up to the ack was handled: keep the set small
            c.handled = {s for s in c.handled if s > c.ack}

    def _start(self, msg: dict) -> None:
        link = JobLink(msg["job"], self._post)
        with self._lock:
            self.links[msg["job"]] = link
        t = threading.Thread(target=self._job, args=(msg, link), daemon=True, name="runner-job")
        with self._lock:
            self._threads.add(t)
        t.start()

    def _job(self, msg: dict, link: JobLink) -> None:
        with self.slots:
            t0 = time.perf_counter()
            try:
                if link.cancel.is_set():
                    res = _err("The request was cancelled.") if msg.get("kind") == "tool" \
                        else {"error": "cancelled", "message": "cancelled"}
                else:
                    res = self.executor.handle(msg, link)
            except Exception as e:  # noqa: BLE001 - whatever broke, the server gets an answer it can read
                text = f"the runner on this PC could not finish the job: {type(e).__name__}: {e}"
                kind = msg.get("kind")
                res = _err(text) if kind == "tool" else {"status": 400, "body": {"error": {"message": text}}} \
                    if kind == "page" else {"error": "failed", "message": text}
            finally:
                with self._lock:
                    self.links.pop(msg["job"], None)
            ms = int((time.perf_counter() - t0) * 1000)
        try:
            link.send({"result": res, "exec_ms": ms})   # outside the slot: a reply that waits holds no job's turn
        finally:
            with self._lock:
                self._threads.discard(threading.current_thread())

    def _post(self, msg: dict) -> bool:
        """POST a reply; network errors are tried again (the same rid: the server applies it once) for up to
        REPLY_RETRY_S.  A result the server refuses (413, 400) is replaced once by a short error in its own shape
        (`_unsent`), so the call there ends with text.  Any other refusal ends it (the poll thread handles it), and so does 404 - the job finished on the
        server - unless the runner's id changed meanwhile (a key rename: the job goes on under the new id), when the
        reply is sent again under the new one once the poll thread said hello (its next poll meets the 404: within a
        poll's wait).  True when the server took it."""
        end, delay = time.monotonic() + REPLY_RETRY_S, 0.5
        fell_back = False
        while True:
            used = self.id
            try:
                self.client.reply(used or "", self.instance, msg)
                return True
            except Refused as e:
                if e.kind == "refused" and "result" in msg and not fell_back:
                    fell_back = True                    # the server took nothing, so the same rid is free
                    msg = {"job": msg["job"], "rid": msg["rid"], "result": _unsent(msg["result"], e.status)}
                    continue
                return False
            except Pending:
                return False
            except Gone:
                wait_end = time.monotonic() + self.poll_s + 5
                with self._said_hello:                  # woken by the hello itself, not a 0.1 s look (#315)
                    while not (self.connected.is_set() and self.id != used):
                        left = wait_end - time.monotonic()
                        if left <= 0 or self.stop.is_set():
                            return False
                        self._said_hello.wait(min(left, STOP_LOOK_S))
            except NetworkError:
                if time.monotonic() > end or self.stop.is_set():
                    return False
                self.stop.wait(delay)
                delay = min(delay * 2, 8.0)


# ------------------------------------------------------------------------------------------------ the program
def main(argv: list[str] | None = None, stop: threading.Event | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m serve.runner",
                                 description="Strata Desktop's runner: the chat's coding tools on this PC, for a Strata "
                                             "server elsewhere.")
    ap.add_argument("--server", required=True, help="the Strata server's address, e.g. https://strata.example")
    ap.add_argument("--key", help="your key for the server (default: the environment variable STRATA_RUNNER_KEY)")
    ap.add_argument("--device-id", help="this PC's device id (default: the one made once in device.json)")
    ap.add_argument("--config", help="the runner's run config (default: runner.json in the runner's folder)")
    a = ap.parse_args(argv)
    key = (a.key or os.environ.get("STRATA_RUNNER_KEY") or "").strip()     # a key pasted with its line end works
    os.environ.pop("STRATA_RUNNER_KEY", None)           # no command or MCP server this runner starts inherits it
    if not key:
        _say("[runner] give your key with --key or the environment variable STRATA_RUNNER_KEY")
        return 2
    if not KEY.fullmatch(key):                          # the key itself is never shown
        _say("[runner] the key must be printable characters with no spaces or line breaks inside it")
        return 2
    if a.device_id is not None and not DEVICE_ID.fullmatch(a.device_id):
        _say("[runner] --device-id must be 8 to 64 of A-Z, a-z, 0-9, _ and -")
        return 2
    d = runner_dir()
    device = load_device(d)
    if a.device_id:
        device = {"id": a.device_id, "name": device["name"]}
    try:
        client = Client(a.server, key, device)
        executor = Executor(harness.home_dir(), Path(a.config) if a.config else d / "runner.json", d)
    except (ValueError, SystemExit) as e:
        _say(str(e) if str(e).startswith("[") else f"[runner] {e}")
        return 2
    stop = stop or threading.Event()
    loop = RunnerLoop(client, executor, stop=stop)
    codes: list[int] = []
    t = threading.Thread(target=lambda: codes.append(loop.run()), daemon=True, name="runner-poll")
    t.start()
    try:
        while t.is_alive():
            t.join(0.5)                                 # short waits: Ctrl+C reaches the main thread
    except KeyboardInterrupt:
        stop.set()
        _say("[runner] stopping")
        codes = [0]
    finally:
        loop.finish(5.0)                                # its jobs end (a running command is stopped) first
        executor.close()
    return codes[0] if codes else 1                     # the poll thread ended without saying why


if __name__ == "__main__":
    sys.exit(main())
