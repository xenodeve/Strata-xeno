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

The job side (Task 2) is duck-typed here: `Runner.jobs` maps a job id to an object with `t_picked` (None until a
poll hands its first message out), `inbox` (a queue the replies are put in) and `fail(text)`.  `push(msg)` queues a
message for the next poll.  The constants are read at call time, so tests may patch them.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from typing import Any

RUNNER_PROTOCOL = 1             # a runner that speaks another one is refused at hello (409 runner_protocol)
POLL_MAX_S = 25.0               # the longest GET /runner/next is held: well inside Cloudflare's 125 s idle limit
OFFLINE_S = 60.0                # no poll held and none ended for this long: the runner is offline
CANCEL_GRACE_S = 10.0           # after a cancel, how long a job waits for the runner's own result (Task 2)
REQUEST_TIMEOUT_S = 30.0        # a run.start or page job's answer (Task 2)

REPLACED = "your PC's runner restarted"
REVOKED = "the key was revoked"
DEVICE = "the owner refused or cleared this PC's device"
CLOSED = "the server is stopping"


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
        if g.get("status") is not None and not _text(g["status"], 40) or                 g.get("error") is not None and not _text(g["error"], 4000):
            return "groups: status and error are text or null"
        tools = g.get("tools") or []
        if not isinstance(tools, list) or len(tools) > MAX_TOOLS:
            return f"groups: tools is a list of at most {MAX_TOOLS}"
        out_tools = []
        for t in tools:
            if not isinstance(t, dict) or not _text(t.get("name"), 128, empty=False) or                     t.get("description") is not None and not _text(t["description"], 100_000) or                     t.get("inputSchema") is not None and not isinstance(t["inputSchema"], dict):
                return 'groups: each tool {"name", "description", "inputSchema"}'
            out_tools.append({"name": t["name"], "description": t.get("description") or "",
                              "inputSchema": t.get("inputSchema") or {"type": "object"}})
        out_groups.append({"name": g["name"], "kind": g["kind"], "status": g.get("status"), "error": g.get("error"),
                           "tools": out_tools})
    out_skills = []
    for k in skills:
        if not isinstance(k, dict) or not _text(k.get("name"), 200, empty=False) or                 k.get("description") is not None and not _text(k["description"], 10_000) or                 k.get("app") is not None and not _text(k["app"], 80):
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
        self.jobs: dict[str, Any] = {}          # open jobs by id (Task 2): t_picked, inbox, fail(text)
        self.ended: str | None = None
        self.cond = threading.Condition()
        self._out: list[dict] = []              # messages for the next poll
        self._held = 0                          # polls held now
        self._seen = time.monotonic()           # the last hello, or the start or end of a poll

    def online(self, now: float | None = None) -> bool:
        """A poll is held, or one started or ended (or a hello came) within OFFLINE_S.  `now` is time.monotonic()."""
        now = time.monotonic() if now is None else now
        with self.cond:
            return self.ended is None and (self._held > 0 or now - self._seen < OFFLINE_S)

    def touch(self) -> None:
        with self.cond:
            self._seen = time.monotonic()

    def push(self, msg: dict) -> bool:
        """Queue a message for the next poll (and wake a held one); False once the runner has ended."""
        with self.cond:
            if self.ended is not None:
                return False
            self._out.append(msg)
            self.cond.notify_all()
            return True

    def take(self, wait: float) -> list[dict]:
        """The messages queued for this runner, waiting at most `wait` seconds for the first one; [] at the end of the
        wait, or at once when the runner ends meanwhile (the route then tells the poll why).  A message that starts a
        job (it has a "kind") marks that job picked up (`t_picked`, time.monotonic())."""
        deadline = time.monotonic() + max(0.0, wait)
        with self.cond:
            self._held += 1
            self._seen = time.monotonic()
            try:
                while not self._out and self.ended is None:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    self.cond.wait(left)
                if self.ended is not None:
                    return []
                out, self._out = self._out, []
                now = time.monotonic()
                for msg in out:
                    if "kind" in msg:
                        job = self.jobs.get(msg.get("job"))
                        if job is not None and getattr(job, "t_picked", None) is None:
                            job.t_picked = now
                return out
            finally:
                self._held -= 1
                self._seen = time.monotonic()

    def deliver(self, msg: dict) -> bool:
        """A message from the runner for one of its jobs (POST /runner/reply); False for a job that is unknown or
        finished, or once the runner has ended."""
        with self.cond:
            job = self.jobs.get(msg.get("job")) if self.ended is None else None
        if job is None:
            return False
        job.inbox.put(msg)
        return True

    def fail_all(self, text: str) -> int:
        """End every open job with `text` (what the model reads), and drop the messages queued for them, so that a
        job the model was told did not run is never handed out later.  How many there were."""
        with self.cond:
            jobs, self.jobs = self.jobs, {}
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
