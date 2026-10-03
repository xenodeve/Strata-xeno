"""api/index.py - the live demo of the Strata-xeno web app, for the showcase site (website/), as a Vercel Python function.

It runs the repo's own mock server (serve/ui/dev/mock_server.py: the real routes, the real pages, the real coding tools, a scripted model, no GPU)
on a loopback port inside the function, and passes requests to it. What the demo is NOT: a model. The answers are the mock server's fixtures, the
hardware page shows the cloud machine the function runs on, and the history lives in the function's temp folder (it goes when the instance does).

This file is a public front door to a server that was written for one person's PC, so it is a short allowlist, not a pass-through:
  - GET only the app's own files and the read-only status pages; never the pages that list folders, files, git or notes of the machine.
  - POST only the chat and the answers to its permission / question / steer / cancel cards; never a setting, a load or a rewind.
  - The chat request's working folder is replaced with a temp folder of this file's own, whatever the page sent.
"""
from __future__ import annotations

import http.client
import importlib.util
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
SANDBOX = Path(tempfile.gettempdir()) / "strata-demo-work"
HOME = Path(tempfile.gettempdir()) / "strata-demo-home"
MAX_BODY = 256 * 1024
MAX_CHATS = 6

GET_EXACT = {"", "/", "/next", "/health", "/api/health", "/status", "/props", "/v1/models", "/v1/status", "/metrics", "/metrics/requests",
             "/settings", "/agent", "/agent/run", "/agent/helpers", "/agent/hooks", "/agent/web", "/mcp", "/mcp/config", "/import"}
GET_PREFIX = ("/next/", "/assets/", "/fonts/", "/metrics/requests/")
POST_EXACT = {"/v1/chat/completions", "/agent/steer", "/agent/cancel", "/agent/question", "/agent/permission"}
HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization", "proxy-authenticate", "host", "content-length",
       "accept-encoding"}

_lock = threading.Lock()
_port: int | None = None
_chats = threading.Semaphore(MAX_CHATS)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _backend() -> int:
    """The port of the mock server, started on the first request of this instance."""
    global _port
    with _lock:
        if _port is not None:
            return _port
        SANDBOX.mkdir(parents=True, exist_ok=True)
        HOME.mkdir(parents=True, exist_ok=True)
        if (ROOT / "README.md").is_file():
            shutil.copyfile(ROOT / "README.md", SANDBOX / "README.md")   # the scripted "Read README.md" has a file to read
        os.chdir(SANDBOX)
        os.environ.update({"STRATA_MOCK_AGENT": "1", "STRATA_HOME": str(HOME), "HOME": str(HOME), "USERPROFILE": str(HOME),
                           "STRATA_MOCK_PREFILL_TPS": os.environ.get("STRATA_MOCK_PREFILL_TPS", "40000"),
                           "STRATA_MOCK_THINK_MS": os.environ.get("STRATA_MOCK_THINK_MS", "6")})
        path = ROOT / "serve" / "ui" / "dev" / "mock_server.py"
        spec = importlib.util.spec_from_file_location("strata_mock_server", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["strata_mock_server"] = mod
        spec.loader.exec_module(mod)
        port = _free_port()
        threading.Thread(target=mod.main, args=(port,), daemon=True, name="strata-mock").start()
        for _ in range(200):                                                  # up to 10 s
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("the demo server did not start")
        _port = port
        return port


def _allowed(method: str, path: str) -> bool:
    if "\\" in path or ".." in path.split("/") or "//" in path or "\0" in path:
        return False
    p = path.rstrip("/") if path != "/" else path
    if method in ("GET", "HEAD"):
        return p in GET_EXACT or path.startswith(GET_PREFIX)
    if method == "POST":
        return p in POST_EXACT
    return False


def _sandboxed(body: bytes, path: str) -> bytes:
    """A chat request names a working folder; it is always the demo's own temp folder (and no other folders)."""
    if path.rstrip("/") != "/v1/chat/completions":
        return body
    try:
        req = json.loads(body)
    except ValueError:
        return body
    sa = req.get("strata_agent") if isinstance(req, dict) else None
    if isinstance(sa, dict):
        sa["cwd"] = str(SANDBOX)
        sa.pop("dirs", None)
        return json.dumps(req).encode()
    return body


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.api_route("/{full:path}", methods=["GET", "HEAD", "POST", "OPTIONS"])
async def front(full: str, request: Request):
    path = "/" + full
    method = request.method
    if method == "OPTIONS":
        return Response(status_code=204, headers={"Allow": "GET, HEAD, POST"})
    if not _allowed(method, path):
        return Response(json.dumps({"error": {"message": "not part of the live demo"}}), status_code=403, media_type="application/json")
    body = b""
    if method == "POST":
        body = await request.body()
        if len(body) > MAX_BODY:
            return Response(json.dumps({"error": {"message": "too large for the demo"}}), status_code=413, media_type="application/json")
        body = _sandboxed(body, path)
    query = request.url.query
    target = path + ("?" + query if query else "")
    chat = path.rstrip("/") == "/v1/chat/completions"
    if chat and not _chats.acquire(blocking=False):
        return Response(json.dumps({"error": {"message": "the demo is busy, try again in a moment"}}), status_code=429, media_type="application/json",
                        headers={"Retry-After": "3"})
    try:
        port = _backend()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        headers["Host"] = f"127.0.0.1:{port}"
        if method == "POST":
            headers["Content-Length"] = str(len(body))
        conn.request(method, target, body=body or None, headers=headers)
        resp = conn.getresponse()
    except Exception:
        if chat:
            _chats.release()
        raise
    out = {k: v for k, v in resp.getheaders() if k.lower() not in HOP | {"x-frame-options", "content-security-policy", "content-encoding"}}
    out["Content-Security-Policy"] = "frame-ancestors 'self'"
    out["X-Robots-Tag"] = "noindex"

    def chunks():
        try:
            while True:
                data = resp.read1(8192) if hasattr(resp, "read1") else resp.read(8192)
                if not data:
                    break
                yield data
        finally:
            conn.close()
            if chat:
                _chats.release()

    if method == "HEAD":
        conn.close()
        if chat:
            _chats.release()
        return Response(status_code=resp.status, headers=out)
    return StreamingResponse(chunks(), status_code=resp.status, headers=out)
