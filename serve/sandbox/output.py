"""serve/sandbox/output.py - what a result of the sandbox looks like to the model (the design's 4.6, invariant I5).

Every byte printed here costs forced tokens, so there are two budgets: one result, and one turn. Output past the first is saved to a file in the copy (in the folder that is never part of a change) and the text says
how to read on. The first line is always the same: it says the result comes from the copy and not from the real project, and it stays even when nothing else fits. The wording is the design's."""
from __future__ import annotations

from serve.sandbox.workspace import RESERVED, WorkspaceError

HEADER = "[sandbox: สำเนา ไม่ใช่ของจริง]"
OUT_OF_BUDGET = "[หมดงบการอ่าน]"
CONFLICT = "[เปลี่ยนในของจริงด้วย: {path}]"
TOUCHED = "แก้แล้ว {n} ไฟล์ในสำเนา ยังไม่แตะ project จริง"

RESULT_BYTES = 1024                    # the design's 1,024: the result as a whole, header and all
TURN_BYTES = 16 * 1024                 # the design leaves this open; a start, to be set from what a real turn reads
OUTPUT_DIR = f"{RESERVED}/output"
TAIL_SHARE = 2 / 3                     # of what fits of long output: the end of a run is where the failure and the count are


def header(exit_code: int | None = None, seconds: float | None = None) -> str:
    parts = []
    if exit_code is not None:
        parts.append(f"exit {exit_code}")
    if seconds is not None:
        parts.append(f"{seconds:.1f} s")
    return f"{HEADER} {', '.join(parts)}" if parts else HEADER


def conflict_lines(paths: list[str]) -> str:
    return "\n".join(CONFLICT.format(path=p) for p in paths)


class Output:
    def __init__(self, workspace, per_result: int = RESULT_BYTES, per_turn: int = TURN_BYTES):
        self.ws, self.per_result, self.per_turn = workspace, per_result, per_turn
        self.left = per_turn
        self._saved = 0

    def new_turn(self) -> None:
        self.left = self.per_turn

    def _save(self, raw: bytes) -> str | None:
        self._saved += 1
        rel = f"{OUTPUT_DIR}/{self._saved}.txt"
        try:
            self.ws.write_bytes(rel, raw)
        except (OSError, WorkspaceError):
            return None
        return rel

    def render(self, body: str | bytes, exit_code: int | None = None, seconds: float | None = None, changed: int | None = None) -> str:
        """The text the model gets for one result: at most the result's budget and what is left of the turn's, in bytes. `changed`: files the copy differs by, for the design's last line."""
        text = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
        raw = text.encode("utf-8", "replace")
        top = header(exit_code, seconds) + "\n"
        end = TOUCHED.format(n=changed) + "\n" if changed else ""
        whole = top + text + ("\n" if end and text and not text.endswith("\n") else "") + end
        budget = min(self.per_result, self.left)
        if len(whole.encode("utf-8", "replace")) > budget:
            return self._cut(raw, top, end, budget)
        self.left -= len(whole.encode("utf-8", "replace"))
        return whole

    def _cut(self, raw: bytes, top: str, end: str, budget: int) -> str:
        saved = self._save(raw)
        where = f'sandbox_read("{saved}", offset={{offset}})' if saved else "it was not saved: the scratch quota is reached"
        probe = f"\n[cut {len(raw):,} of {len(raw):,} bytes. More: {where.format(offset=len(raw))}]\n"
        room = budget - len(top.encode()) - len(end.encode()) - len(probe.encode()) - 1            # 1: the line end `end` may need in front of it
        if room < 16:                                                         # not enough left to be worth a line of output
            note = f"{OUT_OF_BUDGET} {where.format(offset=0)}" if saved else OUT_OF_BUDGET
            return top + note + "\n"
        tail = int(room * TAIL_SHARE)
        first = raw[:room - tail].decode("utf-8", "ignore")
        last = raw[-tail:].decode("utf-8", "ignore") if tail else ""
        offset = len(first.encode())
        note = f"\n[cut {len(raw) - offset - len(last.encode()):,} of {len(raw):,} bytes. More: {where.format(offset=offset)}]\n"
        text = top + first + note + last
        if end:
            text += "" if text.endswith("\n") else "\n"
            text += end
        self.left -= len(text.encode())
        return text
