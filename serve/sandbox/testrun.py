"""serve/sandbox/testrun.py - what a test run said, and what that means against the same tests on the base (the design's 4.6, and its most important line: whether this change broke a test or it was failing before).

pytest and unittest are read test by test; any other command by its exit code, which can still tell "this change broke it" from "it was failing before". A run that could not start or was stopped is said to be that, never that the code is wrong
(the design's Q4). Nothing here says a change is right: a pass names the command that passed (design case 47)."""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

MAX_LISTED = 8                         # failing tests named in a result; the rest are counted
SHELL_WORDS = {"|", "||", "&&", "&", ";", ">", ">>", "<"}
PYTEST_FLAGS = ("-rfEp", "-p", "no:cacheprovider")      # a line for each test, and no cache folder left in the copy

_PYTEST_ROW = re.compile(r"^(PASSED|FAILED|ERROR) (.+?)(?: - (.*))?$", re.M)
_PYTEST_END = re.compile(r"^[= ]*(.*?\d+ (?:passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?).*?|no tests ran.*?) in [\d.]+s", re.M)
_UNIT_RAN = re.compile(r"^Ran (\d+) tests? in ", re.M)
_UNIT_BLOCK = re.compile(r"^(?:FAIL|ERROR): (.+?)\n-+\n(.*?)(?=^={20,}$|^-{20,}$|\Z)", re.M | re.S)
_TEST_FILE = re.compile(r"(^|/)(tests?|__tests__|spec)/|(^|/)(test_[^/]*|[^/]*_test\.\w+|[^/]*\.(test|spec)\.\w+|conftest\.py)$", re.I)
_BLOCKED = ((re.compile(r"No module named '?([\w.]+)", re.I), "ไม่มี library {0} ใน box"),
            (re.compile(r"gaierror|getaddrinfo failed|Network is unreachable|WinError 100(?:13|51|60|61|65)|Connection refused|Name or service not known", re.I), "ไม่มี network ใน box"),
            (re.compile(r"\bCUDA\b|cuda\.is_available|nvcuda|No GPU", re.I), "ไม่มี GPU ใน box"))

PASSES = ("passed", "fixed", "same", "flaky")           # what lets a change go on to `sandbox_apply` without being called untested


def touches_tests(paths: list[str]) -> list[str]:
    """Which of these are test files: a change to them together with a pass may be a change to the exam (design 4.7, case 31)."""
    return [p for p in paths if _TEST_FILE.search(p)]


def blocked_by(text: str) -> str | None:
    """What the box lacks, when the output says so: then the run says "could not run here", not "wrong"."""
    for rx, words in _BLOCKED:
        m = rx.search(text)
        if m:
            return words.format(*m.groups())
    return None


def split_command(command: str) -> list[str]:
    """Words of a command, by the shell's quoting and without its meaning: a backslash stays, because a Windows path holds one. The box has no shell for it, so `|` or `&&` is refused and not passed on as a word."""
    lex = shlex.shlex(command, posix=True)
    lex.whitespace_split, lex.commenters, lex.escape = True, "", ""
    words = list(lex)
    if not words:
        raise ValueError("there is no command")
    if SHELL_WORDS & set(words):
        raise ValueError("the box runs one program with no shell, so | && ; > do not work there: write it as Python with `code`")
    return words


def prepared(argv: list[str]) -> tuple[list[str], str]:
    """The command as it runs in the box, and how its output reads: "pytest", "unittest" or "other"."""
    if argv[0] == "pytest":
        argv = ["python", "-m", *argv]
    if argv[:3] == ["python", "-m", "pytest"]:
        return argv + [f for f in PYTEST_FLAGS if f not in argv], "pytest"
    if argv[:3] == ["python", "-m", "unittest"]:
        return argv, "unittest"
    return argv, "other"


def again(kind: str, ids: list[str]) -> list[str] | None:
    """Only these tests, as a command; None for a command that cannot name them."""
    if kind == "pytest":
        return ["python", "-m", "pytest", "-q", *PYTEST_FLAGS, *ids]
    if kind == "unittest":
        return ["python", "-m", "unittest", *ids]
    return None


@dataclass
class Outcome:
    exit_code: int | None
    kind: str
    passed: set[str] | int = 0             # the ids when the tool names them, else only a count
    failed: dict[str, str] = field(default_factory=dict)       # id -> the first line of why
    read: bool = True                      # the output was read test by test
    none: bool = False                     # it ran and found no test
    killed: str | None = None
    missing: str | None = None             # what the box lacks, if the output says
    text: str = ""

    @property
    def count(self) -> int:
        return len(self.passed) if isinstance(self.passed, set) else self.passed


def _pytest(out: Outcome) -> None:
    for m in _PYTEST_ROW.finditer(out.text):
        kind, name, why = m.groups()
        if kind == "PASSED":
            out.passed.add(name)
        else:
            out.failed[name] = (why or "").strip()
    end = _PYTEST_END.findall(out.text)
    if not end or (out.exit_code not in (0, 1, 5) and not out.failed):    # no summary, or a run that ended some other way: the exit code says more than the rows
        out.read = False
        return
    out.none = out.exit_code == 5
    count = re.search(r"(\d+) passed", end[-1])
    if count and not out.passed:                                          # the flags were overridden: the count stands in
        out.passed = int(count[1])


def _unittest(out: Outcome) -> None:
    ran = _UNIT_RAN.findall(out.text)
    if not ran:
        out.read = False
        return
    for name, block in _UNIT_BLOCK.findall(out.text):
        lines = [x.strip() for x in block.strip().splitlines() if x.strip()]
        inner = re.search(r"\((.+)\)$", name.strip())                     # `test_a (module.Class.test_a)`: what can be run again by name
        out.failed[inner[1] if inner else name.strip()] = lines[-1] if lines else ""
    out.passed = max(0, int(ran[-1]) - len(out.failed) - sum(int(x) for x in re.findall(r"skipped=(\d+)", out.text)))
    out.none = int(ran[-1]) == 0


def parse(kind: str, exit_code: int | None, text: str, killed: str | None = None) -> Outcome:
    text = text.replace("\r\n", "\n")
    out = Outcome(exit_code, kind, passed=set(), killed=killed, text=text, missing=blocked_by(text))
    if killed:
        out.read = False
    elif kind == "pytest":
        _pytest(out)
    elif kind == "unittest":
        _unittest(out)
    else:
        out.read = False
    return out


@dataclass
class Compared:
    verdict: str                           # one of PASSES, or "broke", "new", "nothing", "stopped"
    line: str                              # the design's "tests: ..." line
    fresh: list[str]                       # tests that fail now and did not before
    rows: list[str]                        # the design's FAILED lines


def _counts(out: Outcome) -> str:
    if not out.read:
        return f"exit {out.exit_code}"
    words = [f"{out.count} ผ่าน"]
    if out.failed:
        words.append(f"{len(out.failed)} ตก")
    return ", ".join(words)


def _before(base: Outcome | None) -> str:
    if base is None:
        return "ก่อนแก้รันไม่ได้"
    if base.killed:
        return f"ก่อนแก้ถูกหยุด: {base.killed}"
    return f"ก่อนแก้ {_counts(base)}"


def _rows(failed: dict[str, str], fresh: list[str]) -> list[str]:
    names = fresh + [n for n in failed if n not in fresh]
    rows = [f"FAILED {n}  {failed[n]}".rstrip()[:200] for n in names[:MAX_LISTED]]
    if len(names) > MAX_LISTED:
        rows.append(f"และตกอีก {len(names) - MAX_LISTED} test")
    return rows


def _existed(name: str, base: Outcome) -> bool:
    """Whether the base had this test. A failure to collect names a file: it existed when tests of that file did."""
    ids = base.passed | set(base.failed)
    return name in ids or any(x.startswith(name + "::") for x in ids)


def compare(now: Outcome, base: Outcome | None, flaky: set[str] = frozenset()) -> Compared:
    """The run on the copy against the same command on the base. `flaky`: tests that failed and then passed when run again."""
    if now.killed:
        return Compared("stopped", f"tests: ถูกหยุดก่อนจบ: {now.killed} ({_before(base)})", [], [])
    if now.none:
        return Compared("nothing", f"tests: ไม่มี test ให้รัน ({_before(base)})", [], [])
    if not now.read:                                                          # the exit code is all there is
        if flaky:
            return Compared("flaky", f"exit {now.exit_code} ({_before(base)}) ← ไม่แน่นอน: ตกแล้วผ่านเมื่อรันซ้ำ", [], [])
        if now.exit_code == 0:
            return Compared("fixed" if base is not None and base.exit_code else "passed", f"exit 0 ({_before(base)})", [], [])
        if base is None or base.exit_code == 0 or base.killed:
            return Compared("broke", f"exit {now.exit_code} ({_before(base)}) ← พังเพราะการแก้นี้", [], [])
        return Compared("same", f"exit {now.exit_code} ({_before(base)}) ← ตกอยู่ก่อนแล้ว ไม่ใช่เพราะการแก้นี้", [], [])
    known = base is not None and base.read and not base.killed
    fresh = [n for n in now.failed if n not in flaky and not (known and n in base.failed)]
    new_only = known and isinstance(base.passed, set) and not any(_existed(n, base) for n in fresh)
    rows = _rows({n: why for n, why in now.failed.items() if n not in flaky}, fresh)
    line = f"tests: {_counts(now)} ({_before(base)})"
    if fresh:
        if not known:
            return Compared("broke", line + " ← เทียบไม่ได้ ถือว่าตกเพราะการแก้นี้", fresh, rows)
        return Compared("new" if new_only else "broke", line + (" ← ตกใน test ใหม่ของการแก้นี้" if new_only else " ← พังเพราะการแก้นี้"), fresh, rows)
    if flaky:
        return Compared("flaky", line + f" ← ไม่แน่นอน: {len(flaky)} test ตกแล้วผ่านเมื่อรันซ้ำ", [], rows + [f"ไม่แน่นอน {n}" for n in sorted(flaky)[:MAX_LISTED]])
    if now.failed:
        return Compared("same", line + " ← ตกอยู่ก่อนแล้ว ไม่ใช่เพราะการแก้นี้", [], rows)
    fixed = len(base.failed) if known else 0
    return Compared("fixed" if fixed else "passed", line + (f" ← แก้ได้ {fixed} test" if fixed else ""), [], [])
