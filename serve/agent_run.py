"""serve/agent_run.py - one chat request's use of the coding tools: asking the user through the page, and auto mode.

The tool runs on a worker thread (serve/server.py run_with_mcp) and calls `ctx.ask(card)` when serve/permissions.py says "ask".  `ask` puts a
`permission` event on a queue - the request's stream sends it to the page - and waits until the page answers (POST /agent/permission, which
reaches `Broker.answer`), the request is cancelled, or the time is up (a refusal).  In mode "auto" a call that may be judged goes to the
judge first (serve/judge.py); a clear verdict settles it, an unclear one asks the user.
"""
from __future__ import annotations

import queue
import threading
import time
import uuid
from typing import Callable

from serve import agent, judge, permissions

ANSWERS = ("allow", "allow_chat", "deny")


class Broker:
    """The questions that wait for the page: id -> slot.  Shared by every request; an id is answered once."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pending: dict[str, dict] = {}

    def open(self, rid: str) -> dict:
        slot = {"event": threading.Event(), "answer": None}
        with self.lock:
            self.pending[rid] = slot
        return slot

    def close(self, rid: str) -> None:
        with self.lock:
            self.pending.pop(rid, None)

    def answer(self, rid, decision) -> bool:
        """The page's answer; False for an id that was not asked (or is answered already) or an answer that is not one."""
        if decision not in ANSWERS:
            return False
        with self.lock:
            slot = self.pending.pop(rid, None) if isinstance(rid, str) else None
        if slot is None:
            return False
        slot["answer"] = decision
        slot["event"].set()
        return True


class AgentRun:
    def __init__(self, policy: permissions.Policy, session: str, broker: Broker, goal: str, side: Callable[[str, str], str] | None,
                 cancel: threading.Event, timeout: float = 600.0, shell: str | None = None):
        self.policy, self.broker, self.goal, self.side, self.cancel, self.timeout, self.shell = policy, broker, goal, side, cancel, timeout, shell
        self.events: queue.Queue = queue.Queue()
        self.current: str | None = None                      # the id of the model's call that is running (for the page to tie a card to it)
        self.prompt = ""                                       # the rules for the AI (serve/agent_prompt.py), set by the server
        self.ctx = agent.AgentContext(policy=policy, session=session, ask=self.ask, cancel=cancel, emit=self._emit)

    def bind(self, cancel: threading.Event) -> None:
        """The request's own cancel event (made after the run, once the prompt is ready)."""
        self.cancel = self.ctx.cancel = cancel

    def _emit(self, event: dict) -> None:
        self.events.put({**event, "call_id": self.current})

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
            system, user = judge.prompt(req, self.goal, self.policy.cwd, self.shell)
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

    def ask(self, req: dict) -> str:
        rid = uuid.uuid4().hex[:16]
        if self.policy.mode == "auto" and req.get("judgeable") and not req.get("danger") and self.side is not None:
            settled = self._judge(req, rid)
            if settled:
                return settled
        slot = self.broker.open(rid)
        self._emit({"event": "permission", "id": rid, **{k: req.get(k) for k in ("tool", "arguments", "why", "danger", "rule")}})
        end = time.monotonic() + self.timeout
        try:
            while not slot["event"].wait(0.25):
                if self.cancel.is_set():
                    return "cancelled"
                if time.monotonic() > end:
                    return "deny"
            return slot["answer"]
        finally:
            self.broker.close(rid)
