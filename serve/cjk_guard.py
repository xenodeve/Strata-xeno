"""Ban Han (Chinese) tokens when the prompt carries none (xeno, #49 S4; ported from our EXL3 server's #77).

Over 43 Claude Code streams of a 2026-09-05 quality bench, a 4.0bpw EXL3 quant dropped 14 Han characters into
3 streams, every one inside a Thai sentence. It is sampling drift, not a language choice: where the next Thai
token is diffuse, a Chinese token with the same meaning sits inside top-k and wins. A prompt line cannot reach
that point; a token ban can.

The rule: if the prompt has no Han character and does not name Chinese/China (จีน, china, chinese, mandarin),
the model may not emit one (thinking included). A prompt that has one - pasted text, a request for Chinese, a
tool result - lifts the ban for that request. STRATA_ALLOW_CJK=1 lifts it for the server.

The engine holds the id list (loaded once at start, `--ban-ids FILE`); the server switches it on per request
with the GEN key `ban=1`, and only when the run config asks for the guard ("cjk_guard": true), because an
engine without the flag would not start. A Han character assembled from byte-level pieces, none of which decodes
to Han on its own, is not covered; count_han on the completions is the instrument that shows whether that
matters.
"""
import os
import re

ENV = "STRATA_ALLOW_CJK"
# CJK Unified Ideographs, Extension A, Compatibility Ideographs, Extensions B-H. Not kana, not Hangul, not CJK
# punctuation: those never appeared in a leak.
HAN = re.compile(r"[一-鿿㐀-䶿豈-﫿\U00020000-\U0003134f]")
# Talking ABOUT Chinese without typing any lifts the ban too. Word-bounded in English so "machinations" and
# "chinatown" do not; Thai has no word boundary, so จีน is a plain substring.
MENTION = re.compile(r"จีน|\b(?:china|chinese|mandarin)\b", re.IGNORECASE)


def count_han(text):
    return len(HAN.findall(text or ""))


def _texts(messages):
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            yield c
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    yield part["text"]


def wanted(messages):
    """Ban for this request? Not with STRATA_ALLOW_CJK=1, and not when the prompt has Han or names Chinese."""
    if os.environ.get(ENV, "").strip().lower() in ("1", "true", "yes"):
        return False
    return not any(HAN.search(t) or MENTION.search(t) for t in _texts(messages))


def ban_ids(decode, vocab_size):
    """Every id below vocab_size whose decoded piece (`decode(id)`) contains a Han character."""
    out = []
    for i in range(vocab_size):
        try:
            piece = decode(i)
        except Exception:
            continue
        if piece and HAN.search(piece):
            out.append(i)
    return out
