"""serve/ui_hash.py - is the committed serve/ui/dist built from the committed source? (xeno UI S1)

The runtime needs no Node, so dist/ is committed. `bun run build` writes dist/source-hash.txt with this module; the
test (serve/test_ui.py) recomputes it and fails when the source moved and dist did not.

    python serve/ui_hash.py --write        after a build
    python serve/ui_hash.py                print the hash
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

UI = Path(__file__).resolve().parent / "ui"
SOURCES = ("src", "index.html", "package.json", "bun.lock", "vite.config.ts", "tsconfig.json")


def source_hash(ui: Path = UI) -> str:
    h = hashlib.sha256()
    files: list[Path] = []
    for s in SOURCES:
        p = ui / s
        files += sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p] if p.is_file() else []
    for f in sorted(files):
        h.update(f.relative_to(ui).as_posix().encode() + b"\0")
        h.update(f.read_bytes().replace(b"\r\n", b"\n") + b"\0")      # CRLF and LF checkouts hash alike
    return h.hexdigest()


if __name__ == "__main__":
    digest = source_hash()
    if "--write" in sys.argv:
        (UI / "dist" / "source-hash.txt").write_text(digest + "\n", encoding="utf-8", newline="\n")
    print(digest)
