"""serve/agent_run.py - one chat request's use of the coding tools: asking the user through the page, and auto mode.

The tool runs on a worker thread (serve/server.py run_with_mcp) and calls `ctx.ask(card)` when serve/permissions.py says "ask".  `ask` puts a
`permission` event on a queue - the request's stream sends it to the page - and waits until the page answers (POST /agent/permission, which
reaches `Broker.answer`), the request is cancelled, or the time is up (a refusal).  In mode "auto" a call that may be judged goes to the
judge first (serve/judge.py); a clear verdict settles it, an unclear one asks the user.

#310: a question belongs to the identity whose run asked it (`AgentRun.who`, kept in the slot) and only that identity answers it.  The owner role
does not answer for another - an approval is the run holder's.  An answer from anyone else is refused as for an id that was not asked, and the
question stays open for its holder.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
import weakref
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from serve.identity import Identity

from serve import agent, judge, permissions

ANSWERS = ("allow", "allow_chat", "deny")


class Broker:
    """The questions that wait for the page: id -> slot.  Shared by every request; an id is answered once."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pending: dict[str, dict] = {}
        self.runs: weakref.WeakSet = weakref.WeakSet()      # the AgentRuns that ask through this broker (helpers too), while alive

    def track(self, run: AgentRun) -> None:
        with self.lock:
            self.runs.add(run)

    def move(self, old: str, new: str) -> int:
        """#310: a key renamed from `old` to `new`: every open prompt and every live run of `old` (so what it asks next
        too) is `new`'s.  A later key named `old` inherits nothing.  How many open prompts moved."""
        with self.lock:
            for run in list(self.runs):
                if run.who == old:
                    run.who = new
            slots = [s for s in self.pending.values() if s.get("who") == old]
            for s in slots:
                s["who"] = new
        return len(slots)

    def end_all(self, who: str) -> int:
        """#310: a key that is revoked (or made a guest): every live run of `who` is cancelled and every prompt it holds
        is closed as refused (a permission as "deny", a question as skipped).  How many prompts were closed."""
        with self.lock:
            runs = [run for run in list(self.runs) if run.who == who]
            gone = [rid for rid, s in self.pending.items() if s.get("who") == who]
            slots = [self.pending.pop(rid) for rid in gone]
        for run in runs:
            run.cancel.set()
        for s in slots:
            s["answer"] = "deny" if s.get("kind") == "permission" else {"answers": None}
            s["event"].set()
        return len(slots)

    def open(self, rid: str, kind: str = "permission", who: str = "owner") -> dict:
        slot = {"event": threading.Event(), "answer": None, "kind": kind, "who": who}
        with self.lock:
            self.pending[rid] = slot
        return slot

    def close(self, rid: str) -> None:
        with self.lock:
            self.pending.pop(rid, None)

    def answer(self, rid, decision, by: Identity) -> bool:
        """The page's answer; False for an id that was not asked (or is answered already), one that `by` did not ask, or an answer that is not one."""
        if decision not in ANSWERS:
            return False
        with self.lock:
            slot = self.pending.get(rid) if isinstance(rid, str) else None
            if slot is None or slot.get("kind") != "permission" or slot.get("who") != by.name:
                return False
            self.pending.pop(rid, None)
        slot["answer"] = decision
        slot["event"].set()
        return True

    def respond(self, rid, answers, by: Identity) -> bool:
        """The page's answer to a question the model asked (AskUserQuestion): {question: [labels or the user's own words]}, or None when the user skipped it. False for an id that
        was not asked, is answered already, is a permission question, or that `by` did not ask."""
        with self.lock:
            slot = self.pending.get(rid) if isinstance(rid, str) else None
            if slot is None or slot.get("kind") != "question" or slot.get("who") != by.name:
                return False
            self.pending.pop(rid, None)
        slot["answer"] = {"answers": answers}
        slot["event"].set()
        return True


class AgentRun:
    def __init__(self, policy: permissions.Policy, session: str, broker: Broker, goal: str, side: Callable[[str, str], str] | None,
                 cancel: threading.Event, timeout: float = 600.0, shell: str | None = None, who: str = "owner"):
        self.policy, self.broker, self.goal, self.side, self.cancel, self.timeout, self.shell = policy, broker, goal, side, cancel, timeout, shell
        self.who = who                                         # #310: the identity whose request this is: only it answers what the run asks
        self.events: queue.Queue = queue.Queue()
        self.current: str | None = None                      # the id of the model's call that is running (for the page to tie a card to it)
        self.prompt = ""                                       # the rules for the AI (serve/agent_prompt.py), set by the server
        self.hidden: set = set()                               # coding tools that are not offered to the model in this request (web access, until it is switched on)
        self.hooks = None                                      # serve/hooks.py Runner, when the user has hooks
        self.kw, self.sampling = {}, {}                        # the request's template settings and sampling: a helper (serve/subagent.py) runs with the same
        self.limit_hit = False                                 # the tool loop stopped at its limit of rounds
        self.steers: queue.Queue = queue.Queue()               # what the user sent while the agent worked: read by the model at the next step (serve/server.py run_with_mcp)
        self.ctx = agent.AgentContext(policy=policy, session=session, ask=self.ask, cancel=cancel, emit=self._emit, question=self.question)
        broker.track(self)                                     # #310: a rename moves it, a revoke stops it

    def steer(self, text: str) -> None:
        self.steers.put(text)

    def take_steers(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self.steers.get_nowait())
            except queue.Empty:
                return out

    def bind(self, cancel: threading.Event) -> None:
        """The request's own cancel event (made after the run, once the prompt is ready)."""
        self.cancel = self.ctx.cancel = cancel

    def _emit(self, event: dict) -> None:
        self.events.put({"call_id": self.current, **event})                # an event that names its call (or none: a prompt or stop hook) keeps it

    def drain(self) -> list[dict]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def _judge(self, req: dict, rid: str) -> str | None:
        """"allow", a "blocked:..." answer, or None (ask the user)."""
        self._emit({"event": "judging", "id": rid, "tool": req["tool"]})
        try:
            system, user = judge.prompt(req, self.goal, self.policy.cwd, self.shell, self.policy.dirs)
            kind, severity = judge.verdict(self.side(system, user))
        except Exception:  # noqa: BLE001 - a judge that cannot answer is no judge: the user is asked
            self._emit({"event": "judged", "id": rid, "verdict": "unavailable"})
            return None
        self._emit({"event": "judged", "id": rid, "verdict": kind, "severity": severity})
        if kind == "allow":
            return "allow"
        if kind == "block":
            return f"blocked: auto mode judged this call too risky (severity {severity} of 5)"
        return None

    def question(self, questions: list) -> dict | str:
        """The model's questions to the user: a `question` event, and the answer ({"answers": {...} or None}), or "cancelled", or {"answers": None} when nobody answered in time."""
        rid = uuid.uuid4().hex[:16]
        slot = self.broker.open(rid, "question", self.who)
        self._emit({"event": "question", "id": rid, "questions": questions})
        end = time.monotonic() + self.timeout
        try:
            while not slot["event"].wait(0.25):
                if self.cancel.is_set():
                    return "cancelled"
                if time.monotonic() > end:
                    self._emit({"event": "question_answered", "id": rid, "answers": None})
                    return {"answers": None}
            self._emit({"event": "question_answered", "id": rid, "answers": (slot["answer"] or {}).get("answers")})        # a page that reads the run again later learns that it was answered
            return slot["answer"]
        finally:
            self.broker.close(rid)

    def ask(self, req: dict) -> str:
        rid = uuid.uuid4().hex[:16]
        if self.policy.mode == "auto" and req.get("judgeable") and not req.get("danger") and self.side is not None:
            settled = self._judge(req, rid)
            if settled:
                return settled
        slot = self.broker.open(rid, "permission", self.who)
        self._emit({"event": "permission", "id": rid, **{k: req.get(k) for k in ("tool", "arguments", "why", "danger", "rule")}})
        end = time.monotonic() + self.timeout
        try:
            while not slot["event"].wait(0.25):
                if self.cancel.is_set():
                    return "cancelled"
                if time.monotonic() > end:
                    self._emit({"event": "answered", "id": rid, "answer": "deny"})
                    return "deny"
            self._emit({"event": "answered", "id": rid, "answer": slot["answer"]})              # a page that reads the run again later learns that it was answered
            return slot["answer"]
        finally:
            self.broker.close(rid)
