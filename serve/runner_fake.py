"""serve/runner_fake.py - a scripted runner over real HTTP, for the runner tests (#310 part 2), as
serve/mcp_fake_server.py is for MCP.  It says hello, then polls GET /runner/next on a thread of its own and hands each
message to `on_job(msg, reply)` on a thread of the message's own (so jobs run side by side, and a job that waits for
an answer does not stop the polling); `reply(more)` posts {"id", "instance", "job": msg["job"], **more} to
POST /runner/reply and returns (status, body).

    fake = FakeRunner(port, key, device, groups, on_job)
    ...
    fake.stop()

Polling ends on `stop()` or on any answer but 200 (401 revoked, 404, 409 replaced): `ended` then holds (status,
body).  Standard library only.
"""
from __future__ import annotations

import http.client
import json
import socket
import threading
import time
import uuid
from typing import Callable

from serve import runner_hub


class FakeRunner:
    def __init__(self, port: int, key: str | None, device: str, groups: list,
                 on_job: Callable[[dict, Callable[[dict], tuple]], None], *, skills: list | None = None,
                 instance: str | None = None, machine: str = "fake-pc", host: str = "127.0.0.1",
                 wait: float | None = None, protocol: int = runner_hub.RUNNER_PROTOCOL):
        self.port, self.key, self.device, self.host = port, key, device, host
        self.groups, self.skills, self.on_job = groups, list(skills or []), on_job
        self.instance = instance or uuid.uuid4().hex
        self.machine, self.wait, self.protocol = machine, wait, protocol
        self.stopped = threading.Event()
        self.ended: tuple[int, object] | None = None      # the answer that ended the polling
        self.replies: list[tuple[dict, int]] = []         # every reply posted, with its status
        self._conn: http.client.HTTPConnection | None = None
        self._lock = threading.Lock()
        code, body = self.hello()
        if code != 200:
            raise RuntimeError(f"hello answered {code}: {body}")
        self.id, self.who = body["runner"], body["who"]
        self.thread = threading.Thread(target=self._loop, daemon=True, name="fake-runner")
        self.thread.start()

    # ------------------------------------------------------------------------------------------------ HTTP
    def _request(self, method: str, path: str, body: dict | None = None, held: bool = False) -> tuple[int, object]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=runner_hub.POLL_MAX_S + 30)
        headers = {"X-Strata-Device": self.device, "X-Strata-Device-Name": "fake runner"}
        if self.key is not None:
            headers["Authorization"] = "Bearer " + self.key
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if held:
            with self._lock:
                self._conn = conn
        try:
            conn.request(method, path, body=data, headers=headers)
            r = conn.getresponse()
            raw = r.read()
            try:
                return r.status, json.loads(raw) if raw else None
            except ValueError:
                return r.status, raw
        finally:
            if held:
                with self._lock:
                    self._conn = None
            conn.close()

    def hello(self) -> tuple[int, object]:
        """POST /runner/hello with this runner's facts (again: the lists only change)."""
        return self._request("POST", "/runner/hello", {
            "protocol": self.protocol, "instance": self.instance, "machine": self.machine, "platform": "win32",
            "shell": "Git Bash", "home": "C:\\Users\\fake", "groups": self.groups, "skills": self.skills})

    def reply_for(self, job) -> Callable[[dict], tuple]:
        def reply(more: dict) -> tuple:
            body = {"id": self.id, "instance": self.instance, "job": job, **more}
            code, got = self._request("POST", "/runner/reply", body)
            self.replies.append((body, code))
            return code, got
        return reply

    # ------------------------------------------------------------------------------------------------ polling
    def _loop(self):
        while not self.stopped.is_set():
            wait = runner_hub.POLL_MAX_S if self.wait is None else self.wait
            try:
                code, body = self._request("GET", f"/runner/next?id={self.id}&instance={self.instance}&wait={wait}",
                                           held=True)
            except (OSError, http.client.HTTPException, ValueError):   # a stop() closes the held poll's socket
                if self.stopped.is_set():
                    return
                time.sleep(0.05)
                continue
            if code != 200:
                self.ended = (code, body)
                return
            for msg in body.get("messages", []):
                threading.Thread(target=self.on_job, args=(msg, self.reply_for(msg.get("job"))), daemon=True,
                                 name="fake-runner-job").start()

    def stop(self) -> None:
        """End the polling at once (the held poll's socket is shut) and wait for the thread."""
        self.stopped.set()
        with self._lock:
            conn = self._conn
        if conn is not None and conn.sock is not None:
            try:                                          # close the handle itself: on Windows a shutdown does not
                conn.sock.shutdown(socket.SHUT_RDWR)      # wake a recv held in another thread, a closesocket does
                socket.socket(fileno=conn.sock.detach()).close()
            except OSError:
                pass
        self.thread.join(5)
