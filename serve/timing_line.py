"""The end-of-request timing in llama-server's shape (xeno, #49 S5; the EXL3 server's live_timing.report).

Our tools and our eyes already read llama-server's `prompt eval time` / `eval time` / `total time` block, so
the Strata server prints the same, from the engine's DONE line: prompt_ms covers only the tokens it read
(the rest came from the conversation cache: `[N cached]`), decode_ms the generated ones.
"""


def report(last: dict, prompt_tokens: int, wall_s: float) -> list[str]:
    """The block's lines for one finished request; `last` is StrataEngine.last (the DONE fields)."""
    cached = int(last.get("reused") or 0)
    pn = max(int(prompt_tokens) - cached, 0)
    gn = int(last.get("generated") or 0)
    pms, gms = float(last.get("prompt_ms") or 0.0), float(last.get("decode_ms") or 0.0)
    lines = [
        f"prompt eval time = {pms:10.2f} ms / {pn:5d} tokens ({pms / pn if pn else 0.0:8.2f} ms per token, "
        f"{pn / pms * 1000 if pms else 0.0:8.2f} tokens per second)" + (f"  [{cached} cached]" if cached else ""),
        f"       eval time = {gms:10.2f} ms / {gn:5d} tokens ({gms / gn if gn else 0.0:8.2f} ms per token, "
        f"{gn / gms * 1000 if gms else 0.0:8.2f} tokens per second)",
        f"      total time = {wall_s * 1000:10.2f} ms / {pn + gn:5d} tokens",
    ]
    offered = int(last.get("drafts_offered") or 0)
    if offered:
        acc = int(last.get("drafts_accepted") or 0)
        lines.append(f"draft acceptance = {acc / offered:.5f} ({acc:5d} accepted / {offered:5d} generated)")
    return lines
