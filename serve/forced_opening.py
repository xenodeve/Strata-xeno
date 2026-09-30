"""serve/forced_opening.py (xeno #49 S7 follow-up) - the opening a request says its reply must begin with.

Claude Code's auto-mode classifier gives its fast stage 64 tokens without thinking and ends the request with
"Respond with <severity>N</severity> ONLY. ... No other text." (live, 2026-09-30).  Qwen wrote prose there ("The
agent's most recent action is ...") and was cut, so the fast stage failed every time and was sent again.  When the
request's last message (a user turn, thinking off) states such an opening, the server writes it for the model and
the model continues from it.  The same words in the system prompt alone do not count: the classifier's slow stage
shares that system prompt and must think before it answers.
"""
from __future__ import annotations

import re

RULES = (
    re.compile(r"response MUST begin with (<[A-Za-z_][\w-]*>)"),        # "Your ENTIRE response MUST begin with <block>"
    re.compile(r"Respond with (<[A-Za-z_][\w-]*>)[^\n]*?\bONLY\b"),     # "Respond with <severity>N</severity> ONLY"
)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def required(messages, thinking: bool):
    """The opening the last user message requires (e.g. "<severity>"), or None."""
    if thinking or not messages or messages[-1].get("role") != "user":
        return None
    text = _text(messages[-1].get("content"))
    for rule in RULES:
        m = rule.search(text)
        if m:
            return m.group(1)
    return None
