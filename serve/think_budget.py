"""Thinking budget (xeno, #49 S3; ported from our EXL3 server's think_budget.py, 2026-09-26).

Anthropic's `thinking.budget_tokens` only picked an effort level here, so a turn could think through all of
max_tokens and write nothing. When a request's budget is spent while the model is still thinking, the server
closes the thinking block itself: it stops the engine, adds CLOSE as if the model had written it, and asks the
engine to go on from there (the prompt, what was written, CLOSE), which reuses the cached prefix.

A non-streamed request is a side request: Claude Code's main turn always streams, and the auto-mode classifier
does not (on the EXL3 server it held the only slot 30-130 s while the main turn queued). Side requests think at
the low effort with a small budget. STRATA_SIDE_BUDGET sets it (tokens; 0 = leave them alone).
"""
import os

ROOM_TO_ACT = 2048     # tokens left after the budget for the answer or tool call
SIDE_ROOM = 256        # a classifier answers in a few dozen tokens
CLOSE = "\n\nI have thought enough; now I act on the plan.\n</think>\n\n"


def resolve(max_tokens, requested, room=ROOM_TO_ACT):
    """The budget for one request, or None (no cut); capped so `room` tokens remain inside max_tokens."""
    if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
        return None
    if not max_tokens:
        return requested
    cap = int(max_tokens) - room
    return min(requested, cap) if cap > 0 else None


def side_budget():
    try:
        return max(0, int(os.environ.get("STRATA_SIDE_BUDGET", "1024")))
    except ValueError:
        return 1024


def for_anthropic(req: dict, kwargs: dict, max_new):
    """The budget for an Anthropic request, or None. A side request also gets the low effort in `kwargs` (the
    template's), unless its thinking is off."""
    thinking = req.get("thinking") if isinstance(req.get("thinking"), dict) else {}
    requested = thinking.get("budget_tokens") if thinking.get("type") == "enabled" else None
    side = side_budget()
    if req.get("stream") or not side:
        return resolve(max_new, requested)
    if kwargs.get("enable_thinking") is not False:
        kwargs["reasoning_effort"] = "low"
    return resolve(max_new, min(requested, side) if isinstance(requested, int) and requested > 0 else side,
                   room=SIDE_ROOM)
