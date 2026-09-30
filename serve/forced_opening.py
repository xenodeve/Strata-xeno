"""serve/forced_opening.py (xeno #49 S7 follow-up) - the opening a request says its reply MUST begin with.

Claude Code's auto-mode classifier asks its fast stage for a reply whose "ENTIRE response MUST begin with <block>",
reads it with `<block>(yes|no)`, and gives it 64 tokens without thinking.  Qwen wrote prose there and was cut, so the
stage failed every time.  When the request's last message (a user turn, thinking off) states such an opening, the
server writes it for the model and the model continues from it.  The rule in the system prompt alone does not count:
the classifier's slow stage shares that system prompt and must think before its <block>.
"""
from __future__ import annotations

import re

MUST = re.compile(r"response MUST begin with (<[A-Za-z_][\w-]*>)")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def required(messages, thinking: bool):
    """The opening the last user message requires (e.g. "<block>"), or None."""
    if thinking or not messages or messages[-1].get("role") != "user":
        return None
    m = MUST.search(_text(messages[-1].get("content")))
    return m.group(1) if m else None
