"""#200: carry an unfinished thinking across a reply cut at max_tokens.

A reply that ran out of tokens while still thinking (no `</think>` written) loses that thinking on the next turn
when the client drops the cut assistant message - deepseek-harness does: its next request ("continue") was 7 tokens
longer, not 32,768, and the model planned the same 80K characters again from zero.  The server remembers the cut
reply (the latest one) and, when the next request is the same conversation plus one new user message, carries it:

- "history": the cut reply is put back as an assistant message with its reasoning_content - the chat template
  renders it (preserve_thinking), the model reads its own thinking, then the new message;
- "resume": generation goes on from the cut prompt + the tokens written, inside the open thinking - the new
  message is not shown to the model (meant for a bare "continue").

A finished thinking is never remembered, and a request whose assistant message carries anything is not touched.
"""
import copy

MODES = ("off", "history", "resume")


def mode_of(req, default):
    """The request's reasoning_carry, else the config's; ValueError (a 400) for an unknown one."""
    mode = (req or {}).get("reasoning_carry", default) if isinstance(req, dict) else default
    if mode not in MODES:
        raise ValueError(f"reasoning_carry={mode!r}: expected one of {', '.join(MODES)}")
    return mode


def _empty_assistant(m):
    return (m.get("role") == "assistant" and not m.get("content") and not m.get("reasoning_content")
            and not m.get("tool_calls"))


class Carry:
    def __init__(self):
        self.entry = None           # the latest cut reply: messages, prompt ids, written ids, reasoning text

    def remember(self, messages, prompt_ids, written_ids, reasoning):
        self.entry = {"messages": copy.deepcopy(messages), "prompt": list(prompt_ids), "written": list(written_ids),
                      "reasoning": reasoning}

    def forget(self):
        self.entry = None

    def _match(self, messages):
        """-> (entry, the index of the new user message, whether an empty assistant sits before it) or None."""
        e = self.entry
        if e is None:
            return None
        n = len(e["messages"])
        if len(messages) <= n or messages[:n] != e["messages"]:
            return None
        rest = messages[n:]
        if len(rest) == 1 and rest[0].get("role") == "user":
            return e, n, False
        if len(rest) == 2 and _empty_assistant(rest[0]) and rest[1].get("role") == "user":
            return e, n + 1, True
        return None

    def history(self, messages):
        """The messages with the cut reply's thinking before the new user message, or None (used once)."""
        m = self._match(messages)
        if m is None:
            return None
        e, at, filled = m
        self.entry = None
        reply = {"role": "assistant", "content": "", "reasoning_content": e["reasoning"]}
        return list(messages[:at - 1]) + [reply] + list(messages[at:]) if filled else \
            list(messages[:at]) + [reply] + list(messages[at:])

    def resume(self, messages):
        """The ids to generate on from (the cut prompt + what it wrote), or None (used once)."""
        m = self._match(messages)
        if m is None:
            return None
        self.entry = None
        return m[0]["prompt"] + m[0]["written"]
