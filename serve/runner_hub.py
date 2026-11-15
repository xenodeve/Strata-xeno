"""serve/runner_hub.py - the server side of the runner channel (#310 part 2).

A runner is Strata Desktop's background part on an engineer's own PC.  It only makes outbound requests to the server:

    POST /runner/hello   its protocol, machine facts and tool groups  -> its runner id (bound to the key's identity)
    GET  /runner/next    a long poll (at most POLL_MAX_S) for the messages the server has for it
    POST /runner/reply   a job's result, or a message in the middle of a job

The hub keeps one `Runner` per (identity, device).  Its id is `runner_id(who, device)`, stable across runner and
server restarts, and knowing it grants nothing: every lookup is `RunnerHub.get_for(id, identity)`, which matches
`who == identity.name` exactly - the owner role too, since nobody runs commands on another person's PC.

Presence comes from time stamps, with no thread of its own: a runner is online while a poll is held, and for
OFFLINE_S after a hello or the end of its last poll.  A second process on the same PC (a new `instance` under the
same id) replaces the first: the old one's open jobs fail with "your PC's runner restarted", its held poll is woken,
and its next poll is refused (409 `runner_replaced` at the route).

Jobs (Task 2).  `Runner.submit(kind, payload)` opens a `Job` and queues its first message; the runner's replies for it
reach `job.inbox`.  A tool call goes through a `RunnerGroup` (one per tool group the runner offered, named for the
server side by `Runner.servers()`), which the hub calls like any built-in server; `Runner.request` waits for a
`run.start` or `page` job's result.

Delivery is at least once, and the runner makes it exactly once.  Every message the server sends a runner (a job, an
answer to its ask or question, a cancel) carries `seq`, numbered per `Runner` object from 1, and stays in the outbox
until the runner acknowledges it: each poll sends `ack=` the highest `seq` it has received, and each poll answers every
message above that at once.  So a poll result lost on the way back (a hang-up in the 0.5 s before the server notices,
a peer that vanished without FIN or RST) is simply sent again, and the runner ignores a `seq` it already handled.
`seq` starts again with every new `Runner` object, which the hello answer's `channel` names: a runner that is answered
another channel than before starts its ack and its handled list again.  The constants are read at call time, so tests
may patch them.
"""
from __future__ import annotations

import hashlib
import inspect
import queue
import re
import threading
import time
import uuid
from typing import Any, Callable

from serve import timeline
from serve.mcp import result_text

RUNNER_PROTOCOL = 1             # a runner that speaks another one is refused at hello (409 runner_protocol)
POLL_MAX_S = 25.0               # the longest GET /runner/next is held: well inside Cloudflare's 125 s idle limit
OFFLINE_S = 60.0                # no poll held and none ended for this long: the runner is offline
CANCEL_GRACE_S = 10.0           # after a cancel, how long a job waits for the runner's own result (Task 2)
REQUEST_TIMEOUT_S = 30.0        # a run.start or page job's answer (Task 2)

REPLACED = "your PC's runner restarted"
REVOKED = "the key was revoked"
DEVICE = "the owner refused or cleared this PC's device"
CLOSED = "the server is stopping"
ENDED = {"replaced": REPLACED, "revoked": REVOKED, "device": DEVICE, "closed": CLOSED}
TICK_S = 0.25                   # how often a waiting call looks at its inbox, the cancel and the runner's presence
REPLY_KEYS = ("result", "exec_ms", "ask", "question", "emit")   # what a runner's reply may carry for its job


INSTANCE = re.compile(r"[A-Za-z0-9_-]{8,64}")     # a runner process's own random id
KINDS = ("coding", "mcp", "skills")
MAX_GROUPS, MAX_TOOLS, MAX_SKILLS = 64, 1000, 2000


def _text(value, most: int, empty: bool = True) -> bool:
    return isinstance(value, str) and len(value) <= most and (empty or bool(value.strip()))


def check_hello(body: dict) -> tuple[dict, list, list] | str:
    """(info, groups, skills) of a POST /runner/hello body (its protocol is checked before), copied field by field so
    nothing else is kept, or what is wrong with it."""
    if not isinstance(body.get("instance"), str) or not INSTANCE.fullmatch(body["instance"]):
        return "instance: 8 to 64 of A-Z, a-z, 0-9, _ and -"
    if not _text(body.get("machine"), 80, empty=False):
        return "machine: the PC's name, 1 to 80 characters"
    if not _text(body.get("platform"), 40, empty=False):
        return "platform: sys.platform of the runner's PC"
    if body.get("shell") is not None and not _text(body["shell"], 400):
        return "shell: a description of the runner's shell, or null"
    if not _text(body.get("home"), 1000):
        return "home: the home folder of the runner's user"
    groups, skills = body.get("groups"), body.get("skills")
    if not isinstance(groups, list) or len(groups) > MAX_GROUPS:
        return f"groups: a list of at most {MAX_GROUPS}"
    if not isinstance(skills, list) or len(skills) > MAX_SKILLS:
        return f"skills: a list of at most {MAX_SKILLS}"
    out_groups = []
    for g in groups:
        if not isinstance(g, dict) or not _text(g.get("name"), 64, empty=False) or g.get("kind") not in KINDS:
            return 'groups: each one {"name", "kind": "coding" | "mcp" | "skills", "status", "error", "tools"}'
        if (g.get("status") is not None and not _text(g["status"], 40)
                or g.get("error") is not None and not _text(g["error"], 4000)):
            return "groups: status and error are text or null"
        tools = g.get("tools") or []
        if not isinstance(tools, list) or len(tools) > MAX_TOOLS:
            return f"groups: tools is a list of at most {MAX_TOOLS}"
        out_tools = []
        for t in tools:
            if (not isinstance(t, dict) or not _text(t.get("name"), 128, empty=False)
                    or t.get("description") is not None and not _text(t["description"], 100_000)
                    or t.get("inputSchema") is not None and not isinstance(t["inputSchema"], dict)):
                return 'groups: each tool {"name", "description", "inputSchema"}'
            out_tools.append({"name": t["name"], "description": t.get("description") or "",
                              "inputSchema": t.get("inputSchema") or {"type": "object"}})
        out_groups.append({"name": g["name"], "kind": g["kind"], "status": g.get("status"), "error": g.get("error"),
                           "tools": out_tools})
    out_skills = []
    for k in skills:
        if (not isinstance(k, dict) or not _text(k.get("name"), 200, empty=False)
                or k.get("description") is not None and not _text(k["description"], 10_000)
                or k.get("app") is not None and not _text(k["app"], 80)):
            return 'skills: each one {"name", "description", "app"}'
        out_skills.append({"name": k["name"], "description": k.get("description") or "", "app": k.get("app")})
    info = {"machine": body["machine"], "platform": body["platform"], "shell": body.get("shell"), "home": body["home"]}
    return info, out_groups, out_skills


def runner_id(who: str, device: str) -> str:
    """The runner of `who` on `device`: "r" + 16 hex digits of sha256("<who>\\n<device>")."""
    return "r" + hashlib.sha256(f"{who}\n{device}".encode("utf-8")).hexdigest()[:16]


class Runner:
    """One runner process as the server knows it.  `ended` is None while it may poll, else why it may not:
    "replaced" (a newer instance took its id), "revoked" (its key), "device" (its device refused or cleared) or
    "closed" (the server stops)."""

    def __init__(self, who: str, device: str, instance: str, info: dict, groups: list, skills: list):
        self.id = runner_id(who, device)
        self.who = who
        self.device = device
        self.instance = instance
        self.info = info                        # machine, platform, shell, home
        self.groups = groups                    # [{name, kind, status, error, tools: [{name, description, inputSchema}]}]
        self.skills = skills                    # [{name, description, app}]
        self.since = time.time()
        self.channel = uuid.uuid4().hex[:16]    # names this registration: `seq` starts again with every new one
        self.jobs: dict[str, Any] = {}          # open jobs by id: t_picked, inbox, fail(text)
        self.ended: str | None = None
        self.cond = threading.Condition()
        self._out: list[dict] = []              # the messages the runner has not acknowledged yet, in `seq` order
        self._seq = 0                           # the last `seq` given out
        self._rids: dict[str, set] = {}         # the reply ids (`rid`) already taken, per open job
        self._held = 0                          # polls held now (at most one waits: the newest wins)
        self._poll = 0                          # the newest poll's number: an older one stops waiting
        self._seen = time.monotonic()           # the last hello, or the start or end of a poll

    def online(self, now: float | None = None) -> bool:
        """A poll is held, or one started or ended (or a hello came) within OFFLINE_S.  `now` is time.monotonic()."""
        now = time.monotonic() if now is None else now
        with self.cond:
            return self.ended is None and (self._held > 0 or now - self._seen < OFFLINE_S)

    def touch(self) -> None:
        with self.cond:
            self._seen = time.monotonic()

    def _queue(self, msg: dict) -> None:
        """Under `cond`: give the message its `seq` and queue it."""
        self._seq += 1
        self._out.append({**msg, "seq": self._seq})
        self.cond.notify_all()

    def push(self, msg: dict) -> bool:
        """Queue a message for the next poll (and wake a held one); False once the runner has ended."""
        with self.cond:
            if self.ended is not None:
                return False
            self._queue(msg)
            return True

    def take(self, wait: float, gone: threading.Event | None = None, ack: int = 0,
             channel: str | None = None) -> list[dict]:
        """The messages for this runner.  First every one up to `ack` (the highest `seq` the runner says it received)
        leaves the outbox - unless the poll names another `channel` (its ack counts another registration's seqs), or
        the ack is above every seq given out; then the ones left are answered at once, or the poll waits at most `wait` seconds for a new
        one.  [] at the end of the wait, or at once when the runner ends meanwhile (the route then tells the poll
        why).  Being answered removes nothing: a message stays, and is answered again, until a later poll
        acknowledges it.  A message that starts a job (it has a "kind") marks that job picked up the first time it is
        handed out (`t_picked`, time.perf_counter(): the timeline's clock).

        The newest poll wins: a poll still held when a newer one arrives returns [] at once, so a stale connection (the
        runner reconnected after its network dropped) never waits beside the live one.  `gone` is set when the poll's
        client hung up (the route's watcher): the poll then returns [] at once."""
        deadline = time.monotonic() + max(0.0, wait)
        with self.cond:
            if 0 < ack <= self._seq and channel in (None, self.channel):    # another channel's ack: ignored
                self._out = [m for m in self._out if m["seq"] > ack]
            self._poll += 1
            mine = self._poll
            self.cond.notify_all()                  # an older held poll sees it is no longer the newest
            self._held += 1
            self._seen = time.monotonic()
            dead = False
            try:
                while not self._out and self.ended is None and self._poll == mine:
                    if gone is not None and gone.is_set():
                        break
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    self.cond.wait(left if gone is None else min(left, 0.25))
                dead = gone is not None and gone.is_set()
                if self.ended is not None or self._poll != mine or dead:
                    return []
                out = list(self._out)
                now = time.perf_counter()
                for msg in out:
                    if "kind" in msg:
                        job = self.jobs.get(msg.get("job"))
                        if job is not None and getattr(job, "t_picked", None) is None:
                            job.t_picked = now
                return out
            finally:
                self._held -= 1
                if not dead:                        # a hang-up is no sign of life
                    self._seen = time.monotonic()

    def deliver(self, msg: dict) -> bool:
        """A message from the runner for one of its jobs (POST /runner/reply); False for a job that is unknown or
        finished, or once the runner has ended.  Only `job`, `rid` and REPLY_KEYS reach the job (the server's own
        signals cannot be sent from outside), and a reply whose `rid` (a whole number per job, from the runner) was
        taken already is answered True but applied once: a retried ask opens one card."""
        with self.cond:
            job = self.jobs.get(msg.get("job")) if self.ended is None else None
            if job is None:
                return False
            rid = msg.get("rid")
            if type(rid) is int:
                seen = self._rids.setdefault(msg["job"], set())
                if rid in seen:
                    return True
                seen.add(rid)
        job.inbox.put({k: msg[k] for k in ("job", "rid", *REPLY_KEYS) if k in msg})
        return True

    def submit(self, kind: str, payload: dict) -> "Job":
        """Open a job and queue its first message `{"job", "kind", **payload, "seq"}`.  A runner that has ended fails
        the job at once with its reason (it is never registered)."""
        job = Job(kind)
        with self.cond:
            if self.ended is None:
                self.jobs[job.id] = job
                self._queue({"job": job.id, "kind": kind, **{k: v for k, v in payload.items() if k not in ("job", "kind")}})
                return job
            why = self.ended
        job.fail(ENDED.get(why, why))
        return job

    def withdraw(self, job: "Job") -> bool:
        """Take back a job no poll has handed out yet (its runner went offline, or the call was cancelled before
        pick-up): True, and it is never handed out; False when a poll took it already - the runner may be running it.
        One step under the lock, so a poll cannot hand it out in between."""
        with self.cond:
            if job.t_picked is not None:
                return False
            self.jobs.pop(job.id, None)
            self._rids.pop(job.id, None)
            self._out = [m for m in self._out if m.get("job") != job.id]
            return True

    def finish(self, job: "Job") -> None:
        """A job is over on the server: replies for it are refused from now on, and its messages leave the outbox -
        all but a cancel, which is still sent until the runner acknowledges it (so the command on the PC is stopped
        even when the server stopped waiting for it)."""
        with self.cond:
            self.jobs.pop(job.id, None)
            self._rids.pop(job.id, None)
            self._out = [m for m in self._out if m.get("job") != job.id or m.get("cancel")]
        job.done.set()

    def fail_all(self, text: str) -> int:
        """End every open job with `text` (what the model reads), and drop the messages queued for them, so that a
        job the model was told did not run is never handed out later.  How many there were."""
        with self.cond:
            jobs, self.jobs = self.jobs, {}
            self._rids = {}
            self._out = [m for m in self._out if m.get("job") not in jobs]
        for job in jobs.values():
            job.fail(text)
        return len(jobs)

    def end(self, why: str, text: str) -> int:
        """Stop this runner for good: its held poll wakes, its next one is refused, its open jobs fail with `text`."""
        with self.cond:
            if self.ended is None:
                self.ended = why
            self.cond.notify_all()
        return self.fail_all(text)

    def request(self, kind: str, payload: dict, timeout: float | None = None) -> dict:
        """A `run.start` or `page` job: the runner's `result` (a dict), or {"error": "offline" | "timeout",
        "message"} - offline when the runner is (or goes) offline or ends, timeout after `timeout` seconds
        (REQUEST_TIMEOUT_S when None).  Other replies are ignored, and do not put off the checks.  A job the runner
        took and did not finish gets a cancel."""
        timeout = REQUEST_TIMEOUT_S if timeout is None else timeout
        m = self.info.get("machine") or "?"
        job = self.submit(kind, payload)
        end = time.perf_counter() + timeout
        exec_ms, out = -1, None
        while out is None:
            try:
                msg = job.inbox.get(timeout=min(TICK_S, max(0.0, end - time.perf_counter())) or 0.001)
            except queue.Empty:
                msg = None
            if msg is not None and "failed" in msg:
                out = {"error": "offline", "message": f"your PC ({m}) is not connected: {msg['failed']}"}
            elif msg is not None and "result" in msg:
                res = msg["result"]
                out = res if isinstance(res, dict) else {"error": "offline", "message": "your PC's runner sent no result"}
                exec_ms = msg["exec_ms"] if type(msg.get("exec_ms")) is int else -1
            elif self.ended is not None:                # ended between setting `ended` and failing its jobs
                out = {"error": "offline", "message": f"your PC ({m}) is not connected: {ENDED.get(self.ended)}"}
            elif not self.online():
                if not self.withdraw(job):
                    self.push({"job": job.id, "cancel": True})
                out = {"error": "offline", "message": f"your PC ({m}) is offline"}
            elif time.perf_counter() >= end:
                if not self.withdraw(job):              # it is running there: stop it, nobody waits for it now
                    self.push({"job": job.id, "cancel": True})
                out = {"error": "timeout", "message": f"your PC ({m}) did not answer within {timeout:g} s"}
        self._end_job(job, out, exec_ms, 0)
        return out

    def _end_job(self, job: "Job", result, exec_ms: int, chars: int) -> None:
        """The end of every job, on the calling thread: the outbox and the job's state, then the timeline's
        `runner.call` (queued -> the end; a = the runner's exec_ms, b = the result's characters) and `runner.call wait`
        (queued -> picked up, or -> the end for a job no poll took)."""
        t_end = time.perf_counter()
        job.result = result
        self.finish(job)
        timeline.complete("runner.call", job.t_queued * 1e6, t_end * 1e6, exec_ms, chars)
        timeline.complete("runner.call wait", job.t_queued * 1e6, (job.t_picked or t_end) * 1e6)

    def servers(self) -> dict[str, "RunnerGroup"]:
        """The runner's tool groups as the hub's servers, by their names on the server: the coding group `my_agent`,
        the skills `my_skills`, each MCP server `my_<name>`; a name taken already gets `_2`, `_3`, ...  Only the first
        coding and the first skills group are offered (a request never carries two coding groups)."""
        out: dict[str, RunnerGroup] = {}
        firsts = [next((g for g in self.groups if g["kind"] == k), None) for k in ("coding", "skills")]
        mcps = [g for g in self.groups if g["kind"] == "mcp"]
        for g in [x for x in firsts if x is not None] + mcps:       # my_agent and my_skills are theirs, whatever the order
            base = {"coding": "my_agent", "skills": "my_skills"}.get(g["kind"]) or f"my_{g['name']}"
            name, n = base, 2
            while name in out:
                name, n = f"{base}_{n}", n + 1
            out[name] = RunnerGroup(self, g, name)
        return out

    def public(self, full: bool) -> dict:
        """GET /runners' entry: everything but the instance for its own identity and the owner; `who` and `online`
        for anyone else."""
        if not full:
            return {"who": self.who, "online": self.online()}
        info = self.info
        return {"id": self.id, "who": self.who, "device": self.device, "online": self.online(), "since": self.since,
                "machine": info.get("machine"), "platform": info.get("platform"), "shell": info.get("shell"),
                "home": info.get("home"),
                "groups": [{"name": g.get("name"), "kind": g.get("kind"), "status": g.get("status"),
                            "error": g.get("error"),
                            "tools": [{"name": t.get("name"), "description": t.get("description")}
                                      for t in g.get("tools") or []]} for g in self.groups],
                "skills": [dict(s) for s in self.skills]}


class Job:
    """One job on a runner: a tool call, a `run.start` or a `page` request.  `t_queued` / `t_picked` are
    time.perf_counter() (the timeline's clock); `inbox` gets the runner's replies (and {"failed": text} when the runner
    ends); `done` is set and `result` holds the outcome when the job is over on the server."""

    def __init__(self, kind: str):
        self.id = uuid.uuid4().hex
        self.kind = kind
        self.t_queued = time.perf_counter()
        self.t_picked: float | None = None
        self.inbox: queue.Queue = queue.Queue()
        self.done = threading.Event()
        self.result = None
        self.failed: str | None = None          # why its runner ended, once it did

    def fail(self, text: str) -> None:
        """Its runner ended (`text`: why, one of the hub's reasons): the waiting call wakes at once."""
        self.failed = text
        self.inbox.put({"job": self.id, "failed": text})


def _error(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _readable(res) -> bool:
    """A tool result the server can read: {"content": [blocks (objects)], ...}; a resource block's resource an object."""
    if not isinstance(res, dict) or not isinstance(res.get("content", []), list):
        return False
    return all(isinstance(b, dict) and (b.get("type") != "resource" or isinstance(b.get("resource", {}), dict))
               for b in res.get("content", []))


def _takes_abort(fn) -> bool:
    """`fn` (ctx.ask / ctx.question) accepts `abort=` (AgentRun's do; a plain lambda does not)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == "abort" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


class RunnerGroup:
    """One tool group of a runner, as a built-in server of the hub (duck-typed like McpServer: name, kind, status,
    error, info, tools, call, start, close).  `call` sends a tool job and waits for its result on the hub's worker
    thread, serving the runner's calls back (ask, question, emit) on the way."""

    kind = "runner"
    machine = "my"
    wants_progress = True

    def __init__(self, runner: Runner, group: dict, name: str):
        self.runner = runner
        self.name = name                        # the name on the server: my_agent, my_skills, my_<x>
        self.local = group["name"]              # the name on the runner: what its jobs say in "group"
        self.group_kind = group["kind"]
        self.status = group.get("status") or "ready"
        self.error = group.get("error")
        self.tools = group.get("tools") or []
        self.info = {"machine": runner.info.get("machine"), "group": self.local}
        if group["kind"] == "coding":           # the coding tools: the request's context, Claude Code's plain names,
            self.wants_context = True           # and no entry in GET /mcp (they have their own switch)
            self.plain_names = True
            self.hidden = True

    def start(self):
        pass

    def close(self):
        pass

    def _ended_text(self, tool: str, why: str, sent: bool) -> str:
        """What the model reads when the runner ended during the call (`why`: the reason the job failed with;
        `sent`: a poll took the job)."""
        m = self.runner.info.get("machine") or "?"
        if why == REVOKED:
            return f"the key of this chat was revoked, so {tool} was not finished."
        if why == REPLACED:
            if not sent:
                return f"your PC's runner restarted before {tool} was sent, so it was not run."
            return f"your PC's runner restarted while {tool} was running; whether it finished is unknown."
        if why == DEVICE:
            return f"the owner refused or cleared your PC ({m}) as a device, so {tool} was not finished."
        if why == CLOSED:
            return f"the server is stopping, so {tool} was not finished."
        return f"your PC ({m}) is not connected ({why}), so {tool} was not finished."

    def _card(self, fn, arg, job: Job, cancel: threading.Event | None):
        """Serve one card (ctx.ask / ctx.question) on the call's thread, while a watcher closes it at once - `abort`,
        with `why` - when the runner ends, goes offline or the call is cancelled: an open card never hides them."""
        r = self.runner
        abort, done = threading.Event(), threading.Event()

        def watch():
            while not done.wait(TICK_S):
                why = (r.ended or "ended" if r.ended is not None or job.failed is not None
                       else "offline" if not r.online()
                       else "cancelled" if cancel is not None and cancel.is_set() else None)
                if why is not None:
                    abort.why = why
                    abort.set()
                    return
        threading.Thread(target=watch, daemon=True, name="runner-card-watch").start()
        try:
            return fn(arg, abort=abort) if _takes_abort(fn) else fn(arg)
        finally:
            done.set()

    def call(self, tool: str, arguments: dict, timeout: float, cancel: threading.Event | None, ctx=None,
             progress: Callable[[str], None] | None = None) -> dict:
        """Run `tool` on the runner -> an MCP result {"content", "isError"}.  The coding group has no overall
        deadline (a command's timeout is its own parameter, as on the server); an MCP or skills call waits
        `timeout` + 30 s.  `progress` hears "sending", "running" once a poll took the job, and "offline".  Whatever
        happens, the call ends with a result the model reads, the job is finished, and a job the runner took and did
        not answer gets a cancel."""
        r = self.runner
        m = r.info.get("machine") or "?"

        def say(word):
            try:
                if progress is not None:
                    progress(word)
            except Exception:  # noqa: BLE001 - the page's status words are not worth the call
                pass
        job = r.submit("tool", {"group": self.local, "tool": tool, "arguments": arguments,
                                "run": getattr(ctx, "run_key", None), "session": getattr(ctx, "session", None),
                                "mode": getattr(getattr(ctx, "policy", None), "mode", None),
                                "helper": getattr(ctx, "in_helper", False), "vision": getattr(ctx, "vision", False),
                                "timeout_s": timeout})
        end = None if self.group_kind == "coding" else time.perf_counter() + timeout + 30
        running, answered, cancelled_at = False, False, None
        exec_ms, result = -1, None
        try:
            say("sending")
            while result is None:
                try:
                    msg = job.inbox.get(timeout=TICK_S)
                except queue.Empty:
                    msg = None
                if not running and job.t_picked is not None:
                    running = True
                    say("running")
                if msg is not None:
                    if "failed" in msg:
                        result = _error(self._ended_text(tool, msg["failed"], job.t_picked is not None))
                    elif "result" in msg:
                        answered = True
                        res = msg["result"]
                        result = res if _readable(res) else _error(f"your PC's runner sent a result for {tool} that "
                                                                   f"the server could not read.")
                        exec_ms = msg["exec_ms"] if type(msg.get("exec_ms")) is int else -1
                    elif "ask" in msg:          # a permission card: the user (or auto mode's judge) on the server
                        card = msg["ask"]
                        ok = isinstance(card, dict) and ctx is not None
                        answer = self._card(ctx.ask, card, job, cancel) if ok else "deny"
                        r.push({"job": job.id, "answer": answer, "to": msg.get("rid")})
                    elif "question" in msg:     # AskUserQuestion
                        qs = msg["question"]
                        ok = isinstance(qs, list) and ctx is not None
                        answer = self._card(ctx.question, qs, job, cancel) if ok else {"answers": None}
                        r.push({"job": job.id, "answer": answer, "to": msg.get("rid")})
                    elif "emit" in msg:         # tool activity for the page (todos, a shell's output)
                        if ctx is not None and isinstance(msg["emit"], dict):
                            ctx.emit(msg["emit"])
                    if result is not None:
                        break
                now = time.perf_counter()               # a busy inbox does not put off the cancel or the presence
                if cancel is not None and cancel.is_set():
                    if cancelled_at is None:
                        if r.withdraw(job):     # never handed out: nothing runs on the PC
                            result = _error("The request was cancelled.")
                            break
                        cancelled_at = now
                        r.push({"job": job.id, "cancel": True})
                    elif now - cancelled_at > CANCEL_GRACE_S:
                        result = _error("The request was cancelled.")
                        break
                if r.ended is not None:                 # ended between setting `ended` and failing its jobs
                    result = _error(self._ended_text(tool, ENDED.get(r.ended, r.ended), job.t_picked is not None))
                elif not r.online():
                    say("offline")
                    if r.withdraw(job):
                        result = _error(f"your PC ({m}) is offline, so {tool} was not run. Tell the user; do not "
                                        f"retry it until they say their PC is back.")
                    else:
                        result = _error(f"your PC ({m}) went offline while {tool} was running; whether it finished "
                                        f"is unknown. Tell the user, and check the state before you repeat it.")
                elif end is not None and now > end:
                    result = _error(f"{tool} did not finish on your PC ({m}) within {timeout + 30:g} s.")
        except Exception as e:  # noqa: BLE001 - whatever broke here, the model gets text and the PC a cancel
            result = _error(f"the server could not finish {tool} on your PC ({m}): {type(e).__name__}: {e}")
        finally:
            if result is None:                          # not an Exception (the thread is stopping): still end the job
                result = _error(f"{tool} on your PC ({m}) was stopped on the server.")
            if not answered and cancelled_at is None and r.ended is None and not r.withdraw(job):
                r.push({"job": job.id, "cancel": True})     # it may be running there, and nobody waits for it now
            try:
                chars = len(result_text(result))
            except Exception:  # noqa: BLE001
                chars = -1
            r._end_job(job, result, exec_ms, chars)
        return result


class RunnerHub:
    """Every runner the server knows, by id."""

    def __init__(self):
        self._lock = threading.Lock()
        self._runners: dict[str, Runner] = {}
        self._closed = False

    def hello(self, who: str, device: str, instance: str, info: dict, groups: list, skills: list) -> Runner:
        """Register a runner.  The same id and instance: its facts and lists are updated, nothing else changes.  The
        same id with a new instance: a new process on that PC - the old one is ended as replaced."""
        rid = runner_id(who, device)
        with self._lock:
            old = self._runners.get(rid)
            if old is not None and old.instance == instance and old.ended is None:
                with old.cond:
                    old.info, old.groups, old.skills = info, groups, skills
                    old._seen = time.monotonic()
                return old
            new = Runner(who, device, instance, info, groups, skills)
            if self._closed:
                new.ended = "closed"
            self._runners[rid] = new
        if old is not None:
            old.end("replaced", REPLACED)
        return new

    def get_for(self, rid, viewer) -> Runner | None:
        """The runner `rid` if it is `viewer`'s own (who == viewer.name, the owner role included), else None - the
        same as for an id that does not exist."""
        if not isinstance(rid, str):
            return None
        with self._lock:
            r = self._runners.get(rid)
        return r if r is not None and r.who == viewer.name else None

    def move(self, old: str, new: str) -> int:
        """A key renamed from `old` to `new`: its runners become `new`'s and take the ids of the new name (their
        processes and jobs go on; the old id answers 404, so the runner says hello again and learns the new one).
        How many."""
        displaced = []
        with self._lock:
            mine = [r for r in self._runners.values() if r.who == old]
            for r in mine:
                del self._runners[r.id]
            for r in mine:
                r.who, r.id = new, runner_id(new, r.device)
                if (other := self._runners.get(r.id)) is not None:
                    displaced.append(other)
                self._runners[r.id] = r
        for r in displaced:
            r.end("replaced", REPLACED)
        return len(mine)

    def _end(self, match, why: str, text: str) -> int:
        with self._lock:
            gone = [r for r in self._runners.values() if match(r)]
            for r in gone:
                del self._runners[r.id]
        for r in gone:
            r.end(why, text)
        return len(gone)

    def end_all(self, who: str) -> int:
        """A revoked key: every runner of `who` ends, its open jobs fail.  How many."""
        return self._end(lambda r: r.who == who, "revoked", REVOKED)

    def end_device(self, who: str, device: str) -> int:
        """A device of `who` refused or cleared: the runner on it ends.  How many (0 or 1)."""
        return self._end(lambda r: r.who == who and r.device == device, "device", DEVICE)

    def view(self, viewer) -> list[dict]:
        """GET /runners: `viewer`'s own runners in full, and the others in full for the owner role, else as who/online."""
        with self._lock:
            rs = list(self._runners.values())
        return [r.public(viewer.is_owner or r.who == viewer.name) for r in rs]

    def close(self) -> None:
        """The server stops: every held poll is woken at once (so shutting the HTTP server down never waits
        POLL_MAX_S), and no runner polls again."""
        with self._lock:
            self._closed = True
            rs = list(self._runners.values())
        for r in rs:
            r.end("closed", CLOSED)
