"""#200: carry an unfinished thinking across a reply cut at max_tokens.

A reply that ran out of tokens while still thinking (no `</think>` written) loses that thinking on the next turn
when the client drops the cut assistant message - deepseek-harness does: its next request ("continue") was 7 tokens
longer, not 32,768, and the model planned the same 80K characters again from zero.  After every reply of a request
that asked for a carry, the server settles its conversation: a reply cut while thinking is kept (keyed by the client's
messages), any other end forgets it.  When the next request is that conversation plus one new user message (the cut
reply dropped, or sent back as an empty assistant message), the kept reply is carried:

- "history": the messages the cut reply was rendered from, then the cut reply as an assistant message with its
  reasoning_content (the chat template renders it, preserve_thinking), then the new message - so a chain of cuts
  keeps every thinking of it;
- "resume": generation goes on from the cut prompt + the tokens written, inside the open thinking - the new
  message is not shown to the model (meant for a bare "continue").

A client's own assistant message (content, reasoning or tool calls) is never touched.
"""
import hashlib
import json
import threading
from collections import OrderedDict

MODES = ("off", "history", "resume")
KEEP = 8                    # conversations kept at once (the main one, its subagents); the oldest goes first


def mode_of(req, default):
    """The request's reasoning_carry, else the config's; ValueError (a 400) for an unknown one."""
    mode = req.get("reasoning_carry", default) if isinstance(req, dict) else default
    if mode not in MODES:
        raise ValueError(f"reasoning_carry={mode!r}: expected one of {', '.join(MODES)}")
    return mode


def key_of(messages):
    """A conversation's key: its template messages, canonical."""
    return hashlib.sha1(json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _empty_assistant(m):
    return (m.get("role") == "assistant" and not m.get("content") and not m.get("reasoning_content")
            and not m.get("tool_calls"))


class Carry:
    def __init__(self, keep=KEEP):
        self.entries = OrderedDict()                # key -> the cut reply: rendered messages, prompt, written, reasoning
        self.keep = keep
        self.lock = threading.Lock()

    def settle(self, key, cut=None):
        """After a reply: `cut` = (rendered messages, prompt ids, written ids, reasoning) when it ran out of tokens
        while thinking - kept for this conversation; None (any other end) - this conversation's is forgotten."""
        with self.lock:
            if cut is None:
                self.entries.pop(key, None)
                return
            rendered, prompt, written, reasoning = cut
            self.entries[key] = {"rendered": list(rendered), "prompt": list(prompt), "written": list(written),
                                 "reasoning": reasoning}
            self.entries.move_to_end(key)
            while len(self.entries) > self.keep:
                self.entries.popitem(last=False)

    def _take(self, messages):
        """The kept cut reply this request continues (used once), or None."""
        if not messages or messages[-1].get("role") != "user":
            return None
        before = messages[:-1]
        if before and _empty_assistant(before[-1]):
            before = before[:-1]
        with self.lock:
            return self.entries.pop(key_of(before), None)

    def history(self, messages):
        """The messages to render: the cut reply's own, its thinking, the new message - or None."""
        e = self._take(messages)
        if e is None:
            return None
        reply = {"role": "assistant", "content": "", "reasoning_content": e["reasoning"]}
        return e["rendered"] + [reply, messages[-1]], len(e["written"])

    def resume(self, messages):
        """The ids to generate on from (the cut prompt + what it wrote), or None."""
        e = self._take(messages)
        return None if e is None else e["prompt"] + e["written"]
