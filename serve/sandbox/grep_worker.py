"""serve/sandbox/grep_worker.py - a pattern that can backtrack runs here, in a process of its own, because `re` has no time limit and holds the GIL.

`(a|aa)+$` ran past 15 s on 60 characters: in the server's process that stops every request, here it costs one kill. Standard library only: the file runs under
`python -I`, and `search.py` takes `decode` and `hits_in` from it, so a child's hits and the server's are the same. A job on stdin, the hits as JSON on stdout."""
import codecs
import json
import os
import re
import sys

MAX_LINE = 240                         # characters of a matching line shown


def decode(raw: bytes) -> str | None:
    """A file's text, or None for a binary one. UTF-16 with a mark is text that a zero byte would otherwise call binary, and Windows tools write it."""
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            return None
    if b"\0" in raw[:8192]:
        return None
    return raw.decode("utf-8-sig", "replace")


def hits_in(rel: str, text: str, rx: re.Pattern, room: int) -> list[dict]:
    """At most `room` matching lines of one file, one hit for a line."""
    hits: list[dict] = []
    line, upto, last = 1, 0, 0
    for m in rx.finditer(text):
        line += text.count("\n", upto, m.start())
        upto = m.start()
        if line == last:
            continue
        last = line
        start = text.rfind("\n", 0, m.start()) + 1
        end = text.find("\n", m.start())
        hits.append({"path": rel, "line": line, "text": text[start:end if end >= 0 else len(text)].rstrip("\r")[:MAX_LINE]})
        if len(hits) >= room:
            break
    return hits


def run(job: dict) -> dict:
    rx = re.compile(job["pattern"], re.M)
    root = os.path.realpath(job["root"])
    hits: list[dict] = []
    binary = 0
    for rel in job["files"]:
        path = os.path.realpath(os.path.join(root, rel))
        if not path.startswith(root + os.sep):                                # the box can swap a file for a link while this runs
            continue
        try:
            with open(path, "rb") as f:
                text = decode(f.read(job["max_file"] + 1))
        except OSError:
            continue
        if text is None:
            binary += 1
            continue
        hits += hits_in(rel, text, rx, job["max_hits"] - len(hits))
        if len(hits) >= job["max_hits"]:
            return {"hits": hits, "truncated": True, "binary": binary}
    return {"hits": hits, "truncated": False, "binary": binary}


if __name__ == "__main__":
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    json.dump(run(json.load(sys.stdin)), sys.stdout)
