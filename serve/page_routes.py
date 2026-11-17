"""The web app's routes that act on a project's own machine (#310 part 2, task 5): the folder picker, the Git panel,
`@file`, the project's notes and the way back for the files the tools changed.

    GET  /agent/folders, /agent/git, /agent/git/diff, /agent/files, /agent/mention, /agent/memory, /agent/memory/file
    POST /agent/rewind, /agent/checkpoints/forget

`answer` is their bodies, moved here word for word from serve/server.py, so the same code answers on both machines:
the server calls it for the workstation (owner-only, as before), and a runner (Strata Desktop) calls it for a `page`
job on the engineer's PC.  It is pure: no Service, no handler, no config read - the caller passes `home`, the notes'
"apps switched on" (the server's run config, in each job) and the checkpoint store of its machine.

`relay` turns what `Runner.request` gave for a `page` job into the page's answer, checked first: only the statuses
`answer` gives, a JSON object, and at most MAX_ANSWER characters - a runner's answer never reaches the page as a 401
or a 5 MB body.  Standard library only (the runner imports it too)."""
from __future__ import annotations

import json
import os

from serve import agent as agent_mod, files as files_mod, folders as folders_mod, gitview, memory as memory_mod

GET_PATHS = ("/agent/folders", "/agent/git", "/agent/git/diff", "/agent/files", "/agent/mention", "/agent/memory",
             "/agent/memory/file")
POST_PATHS = ("/agent/rewind", "/agent/checkpoints/forget")
PATHS = GET_PATHS + POST_PATHS
MEMORY_PATHS = ("/agent/memory", "/agent/memory/file")
MEMORY_FILE_MAX = 200_000           # characters of one note's file the page is shown
STATUSES = (200, 400, 404)          # the only statuses `answer` gives: a runner's other ones are not passed on
ADMIN_STATUSES = (200, 400, 404, 409, 422, 500)     # #338: the MCP jobs' (serve/mcp_admin.py and serve/harness.py apply)
MAX_ANSWER = 2_000_000              # characters of a runner's answer (as JSON): above every answer the routes give
                                    # (the largest, a note's file, is 200,000 characters: 1.2 M at JSON's worst escaping)
NOT_FOUND = (404, {"error": {"message": "not found"}})


def _folders(q: dict) -> list[str]:
    """The project's folders the page named: `path` then each `dirs`, blank and NUL ones left out, at most MAX_DIRS + 1."""
    return [f for f in [q.get("path", [""])[0], *q.get("dirs", [])] if f.strip() and "\0" not in f][:agent_mod.MAX_DIRS + 1]


def answer(path: str, query: dict[str, list[str]], body: dict | None, *, home: str, memory_on: list[str],
           checkpoints) -> tuple[int, dict]:
    """(status, JSON body) of one of the nine routes, on this machine.  `query`: the GET query as parse_qs gives it;
    `body`: the POST body (None, or anything json.loads gave); `home`: the user's home (the other apps' notes);
    `memory_on`: the notes' sources switched on; `checkpoints`: this machine's checkpoints.Checkpoints."""
    q = query
    if path == "/agent/folders":
        # the folders of this PC, to choose a project's folder from (only the names of folders)
        seen = folders_mod.look(q.get("path", [""])[0])
        # a path that is no folder is an answer, not an error (a 404 would be logged as one in the browser's console)
        return 200, (seen if seen is not None else {"ok": False, "error": "That is not a folder on this PC"})
    if path in ("/agent/git", "/agent/git/diff"):
        # the Git state of a folder, read only, for the Chat's right panel (serve/gitview.py)
        folder = q.get("path", [""])[0]
        if path == "/agent/git":
            return 200, gitview.info(folder)
        return 200, gitview.diff(folder, q.get("file", [""])[0], q.get("staged", ["0"])[0] == "1", q.get("untracked", ["0"])[0] == "1")
    if path in ("/agent/files", "/agent/mention"):
        # `@file` in the prompt: the files of the project's folders that go with typed letters, and the text of one
        # that was mentioned (serve/files.py)
        folders = _folders(q)
        if path == "/agent/files":
            return 200, {"files": files_mod.find(folders, q.get("q", [""])[0])}
        return 200, files_mod.read(folders, q.get("rel", [""])[0])
    if path in MEMORY_PATHS:
        # the notes the chat reads for a project: the project's own files and what the user's other apps wrote down
        # (serve/memory.py)
        sources = memory_mod.discover(home, _folders(q), memory_on)
        if path == "/agent/memory":
            return 200, {"max": memory_mod.MAX_TOTAL, "sources": [{k: s[k] for k in ("id", "app", "label", "kind", "shown", "bytes", "on")} for s in sources]}
        src = next((s for s in sources if s["id"] == q.get("id", [""])[0]), None)
        try:
            n = int(q.get("n", ["0"])[0])
        except ValueError:
            n = -1
        if src is None or not 0 <= n < len(src["files"]):
            return 404, {"ok": False, "error": "no such file"}
        got = memory_mod._read(src["files"][n], MEMORY_FILE_MAX)
        if got is None:
            return 200, {"ok": False, "error": "cannot be read"}
        return 200, {"ok": True, "name": os.path.basename(src["files"][n]), "text": got[0], "cut": got[1]}
    if path in POST_PATHS:
        # the way back for the files the tools changed (serve/checkpoints.py)
        if not isinstance(body, dict) or not isinstance(body.get("session"), str):
            return 400, {"error": {"message": "send {\"session\": the chat's id, \"checkpoint\": the prompt's id}"}}
        if path == "/agent/checkpoints/forget":
            checkpoints.forget(body["session"])
            return 200, {"ok": True}
        if not isinstance(body.get("checkpoint"), str):
            return 400, {"error": {"message": "send the checkpoint, the prompt's id"}}
        if body.get("apply") is True:
            return 200, checkpoints.apply(body["session"], body["checkpoint"], body.get("include_changed") is True)
        return 200, checkpoints.preview(body["session"], body["checkpoint"])
    return NOT_FOUND


def relay(res, machine: str, statuses: tuple = STATUSES) -> tuple[int, dict]:
    """The page's answer for a `page` job (or, with `statuses` ADMIN_STATUSES, an MCP job of #338): `res` is what
    Runner.request gave - the runner's result {"status", "body"}, or its own {"error": "offline" | "timeout",
    "message"}.  Offline: 409 runner_offline; no answer in time: 504 runner_timeout (messages cut at 300 characters); a
    runner that does not know the job (an older Strata Desktop answers {"error": "unknown"}): 409 runner_outdated; a
    result that is not one of `statuses` with a JSON object, or one larger than MAX_ANSWER characters: 502
    runner_answer."""
    if isinstance(res, dict) and "status" not in res and res.get("error") in ("offline", "timeout", "unknown"):
        msg = res.get("message") if isinstance(res.get("message"), str) else f"your PC ({machine}) did not answer"
        kind, code = {"offline": ("runner_offline", 409), "timeout": ("runner_timeout", 504),
                      "unknown": ("runner_outdated", 409)}[res["error"]]
        return code, {"error": {"type": kind, "message": msg[:300]}}
    unreadable = (502, {"error": {"type": "runner_answer",
                                  "message": f"your PC's runner ({machine}) sent an answer the server could not read"}})
    if not isinstance(res, dict) or type(res.get("status")) is not int or res["status"] not in statuses \
            or not isinstance(res.get("body"), dict):
        return unreadable
    try:
        size = len(json.dumps(res["body"], ensure_ascii=False))
    except (TypeError, ValueError):
        return unreadable
    if size > MAX_ANSWER:
        return 502, {"error": {"type": "runner_answer", "message": f"your PC's runner ({machine}) sent an answer too large "
                                                                   f"for the page ({size:,} characters; at most {MAX_ANSWER:,})"}}
    return res["status"], res["body"]
