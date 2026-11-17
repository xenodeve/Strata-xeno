"""Claude-Code-shaped request sequences for tools/cacheslots_replay.py (#321), from the review of the continuation rule.

One sender, synthetic token ids; the token shapes are assumptions, not measurements: a 20,000-token main head (system
prompt and tools), a 28,000-token classifier root (its instructions), a 12,000-token subagent head.  Each sequence is a
list of replay records: {"ids", "who", "role", "tag", "pinned"}.  Roles: main, subagent, classifier, side.

    python tools/cacheslots_replay.py --sequences            # every sequence, every rule
"""
from __future__ import annotations

WHO = {"key": "owner", "device": "addr:127.0.0.1", "addr": "127.0.0.1"}


def toks(tag: int, n: int) -> list[int]:
    return [tag * 1_000_000 + i for i in range(n)]


MAIN_HEAD = toks(1, 20000)
CLS_ROOT = toks(2, 28000)
SUB_HEAD = toks(3, 12000)


def rec(tag: str, ids: list[int], role: str, pinned=None) -> dict:
    return {"ids": ids, "who": WHO, "role": role, "tag": tag, "pinned": pinned, "route": "seq"}


def classifier(closing: int, rounds: int = 4) -> list[dict]:
    """Main turns, each followed by a classifier round whose transcript grows inside one message, then a closing."""
    out, main = [], MAIN_HEAD + toks(10, 3000)
    for k in range(rounds):
        main = main + toks(20 + k, 2000)
        out.append(rec(f"m{k}", main, "main"))
        out.append(rec(f"c{k}", CLS_ROOT + toks(50, 1000 * (k + 1)) + toks(90, closing), "classifier"))
    return out


def idle_main_side() -> list[dict]:
    """I1 with side requests: main idles through a subagent run with classifier rounds and short side requests (new
    heads), then a second subagent of the same type starts, then main returns."""
    main = MAIN_HEAD + toks(10, 80000)
    out = [rec("main", main, "main"), rec("title", toks(70, 300), "side"),
           rec("cls", CLS_ROOT + toks(51, 500) + toks(90, 60), "classifier")]
    sub = SUB_HEAD + toks(30, 2000)
    out.append(rec("sub1", sub, "subagent"))
    for k in range(3):
        sub = sub + toks(31 + k, 1500)
        out.append(rec(f"sub1.{k}", sub, "subagent"))
        out.append(rec(f"cls.{k}", CLS_ROOT + toks(51, 500) + toks(60, 300 * (k + 1)) + toks(90, 60), "classifier"))
        out.append(rec(f"side.{k}", toks(80 + k, 400), "side"))
    out.append(rec("sub2", SUB_HEAD + toks(40, 2000), "subagent"))
    out.append(rec("main'", main + toks(99, 1500), "main"))
    return out


def idle_main(launch_between: bool) -> list[dict]:
    """I1 without side requests: sub1 runs, (main launches sub2 | sub1 and sub2 were launched together), sub2 runs,
    main returns."""
    main = MAIN_HEAD + toks(10, 80000)
    out = [rec("main", main, "main"), rec("title", toks(70, 300), "side"),
           rec("cls", CLS_ROOT + toks(51, 500) + toks(90, 60), "classifier")]
    sub = SUB_HEAD + toks(30, 2000)
    out.append(rec("sub1", sub, "subagent"))
    for k in range(3):
        sub = sub + toks(31 + k, 1500)
        out.append(rec(f"sub1.{k}", sub, "subagent"))
        out.append(rec(f"cls.{k}", CLS_ROOT + toks(51, 500) + toks(60, 300 * (k + 1)) + toks(90, 60), "classifier"))
    if launch_between:
        main = main + toks(98, 800)
        out.append(rec("main.task", main, "main"))
    sub2 = SUB_HEAD + toks(40, 2000)
    out.append(rec("sub2", sub2, "subagent"))
    for k in range(2):
        sub2 = sub2 + toks(41 + k, 1500)
        out.append(rec(f"sub2.{k}", sub2, "subagent"))
        out.append(rec(f"cls2.{k}", CLS_ROOT + toks(51, 500) + toks(60, 1200 + 300 * (k + 1)) + toks(90, 60),
                       "classifier"))
    out.append(rec("main'", main + toks(99, 1500), "main"))
    return out


def parallel_subagents() -> list[dict]:
    main = MAIN_HEAD + toks(10, 80000)
    out = [rec("main", main, "main")]
    subs = [SUB_HEAD + toks(100 + j, 2000) for j in range(3)]
    for r in range(3):
        for j in range(3):
            subs[j] = subs[j] + toks(200 + 10 * j + r, 1200)
            out.append(rec(f"s{j}.{r}", subs[j], "subagent"))
    out.append(rec("main'", main + toks(99, 1500), "main"))
    return out


def one_shots() -> list[dict]:
    """I3: long one-shot requests that share main's head."""
    main = MAIN_HEAD + toks(10, 80000)
    return ([rec("main", main, "main")] + [rec(f"o{k}", MAIN_HEAD + toks(300 + k, 3000), "side") for k in range(5)]
            + [rec("main'", main + toks(99, 500), "main")])


def compaction() -> list[dict]:
    """I3: the summary request continues main; the compacted conversation shares only the head."""
    main = MAIN_HEAD + toks(10, 80000)
    return [rec("main", main, "main"), rec("cls", CLS_ROOT + toks(51, 500) + toks(90, 60), "classifier"),
            rec("summ", main + toks(400, 900), "main"), rec("compacted", MAIN_HEAD + toks(401, 4000), "main"),
            rec("comp.1", MAIN_HEAD + toks(401, 4000) + toks(402, 1500), "main")]


def fork() -> list[dict]:
    """I2: a fork subagent inherits main's conversation and grows it in main's slot; main returns."""
    main = MAIN_HEAD + toks(10, 80000)
    f = main + toks(500, 300)
    out = [rec("main", main, "main"), rec("cls", CLS_ROOT + toks(51, 500) + toks(90, 60), "classifier"),
           rec("fork0", f, "subagent")]
    for k in range(4):
        f = f + toks(501 + k, 1500)
        out.append(rec(f"fork{k + 1}", f, "subagent"))
    out.append(rec("main'", main + toks(510, 600), "main"))
    return out


def pinned_newcomers() -> list[dict]:
    out, pa = [], MAIN_HEAD + toks(600, 3000)
    for k in range(5):
        pa = pa + toks(610 + k, 1500)
        out.append(rec(f"P{k}", pa, "main", pinned=0))
        out.append(rec(f"u{k}", MAIN_HEAD + toks(700 + k, 3000), "main"))
    return out


SEQUENCES = {
    "classifier-60": lambda: classifier(60),
    "classifier-1025": lambda: classifier(1025),
    "I1-idle-main-side": idle_main_side,
    "I1-idle-main-task": lambda: idle_main(True),
    "I1-idle-main-together": lambda: idle_main(False),
    "parallel-subagents": parallel_subagents,
    "I3-one-shots": one_shots,
    "I3-compaction": compaction,
    "I2-fork": fork,
    "pinned-newcomers": pinned_newcomers,
}
