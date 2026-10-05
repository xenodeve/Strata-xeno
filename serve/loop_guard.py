"""Detect degenerate text loops during Strata serving.

Adapted from the EXL3 server guard (issues #76 and #86). Service.run feeds
reasoning/content text and cancels the engine request after a match. Short Thai
loops trip within 64 Thai-script characters; general loops use 512 characters.
"""
import re
from collections import Counter

WINDOW = 512
MAX_DISTINCT = 2
MAX_PERIOD = 8
THAI = re.compile(r"[ก-๛]")   # Thai block: consonants, vowels, tone marks, digits, symbols
THAI_MICRO_WINDOW = 64        # issue #86: a Thai-script micro-loop trips here, not at 512
THAI_MICRO_DISTINCT = 2       # same bar as the main rule; normal Thai prose is far above it
NON_THAI_RESET = 16           # ASCII dividers never run this long; a language switch does
THINK_WINDOW = 4096       # the sentence-loop rule, thinking only (19:03: a 3-sentence cycle, 127,996 tokens)
THINK_UNIT = 64
THINK_REPEATS = 8
# markup drafted inside thinking repeats legitimately (both-r2: six section headers share a
# 64-character opening); only prose units count
PROSE_UNIT = re.compile(r"^[^<>{}|`]+$")
# #199: a thinking that re-plans the same answer in cycles too long for THINK_WINDOW (12 cycles of ~8,400 characters,
# 32,768 tokens and no answer).  Prose lines (PROSE_UNIT, and no ';' or ' = ': not code) of CYCLE_LINE+ characters
# outside code fences; a line counts again only CYCLE_GAP+ characters after it last came (a short re-check of a list
# is not a cycle; a short cycle is THINK_WINDOW's), and once CYCLE_LINES distinct ones have each come CYCLE_REPEATS
# times the thinking is cycling - the server then closes it (not a stop)
CYCLE_LINE = 40
CYCLE_REPEATS = 3
CYCLE_LINES = 4
CYCLE_GAP = 2048
CYCLE_MAX_LINE = 512   # a longer line is not a plan line: not collected


def _plan_line(line):
    return (CYCLE_LINE <= len(line) <= CYCLE_MAX_LINE and PROSE_UNIT.match(line) is not None and ";" not in line
            and " = " not in line)


class LoopGuard:
    def __init__(self, window = WINDOW, max_distinct = MAX_DISTINCT, max_period = MAX_PERIOD,
                 think_window = THINK_WINDOW, think_unit = THINK_UNIT, think_repeats = THINK_REPEATS,
                 thai_micro_window = THAI_MICRO_WINDOW, thai_micro_distinct = THAI_MICRO_DISTINCT):
        self.window = window
        self.max_distinct = max_distinct
        self.max_period = max_period
        self.think_window = think_window
        self.think_unit = think_unit
        self.think_repeats = think_repeats
        self.thai_micro_window = thai_micro_window
        self.thai_micro_distinct = thai_micro_distinct
        self.tail = ""
        self.think_tail = ""
        self.thai_tail = ""       # issue #86: trailing Thai-block characters only
        self.non_thai_run = 0     # issue #86: consecutive non-Thai, non-space characters
        self.n_chars = 0
        self.reason = None
        self.close_reason = None  # #199: the thinking is cycling - close it (the request goes on)
        self.think_line = ""      # the thinking's unfinished line
        self.in_fence = False
        self.think_pos = 0        # characters of the thinking's finished lines
        self.line_counts = Counter()
        self.line_at = {}         # where each counted line last came
        self.cycling_lines = 0

    def _think_lines(self, chunk):
        """#199: count the thinking's long prose lines; set close_reason in the cycle that repeats a set of them."""
        if self.close_reason:
            return
        if "\n" not in chunk:                           # most tokens: no line ends here
            if len(self.think_line) <= CYCLE_MAX_LINE:
                self.think_line += chunk
            return
        lines = (self.think_line + chunk).split("\n")
        self.think_line = lines.pop()
        for raw in lines:
            self.think_pos += len(raw) + 1
            line = raw.strip()
            if "```" in line:                           # a fence opens or closes here, at the start or mid-line
                self.in_fence ^= line.count("```") % 2 == 1
                continue
            if self.in_fence or not _plan_line(line):
                continue
            last = self.line_at.get(line)
            if last is not None and self.think_pos - last < CYCLE_GAP:
                continue                                # too soon to be the next cycle
            self.line_at[line] = self.think_pos
            self.line_counts[line] += 1
            if self.line_counts[line] == CYCLE_REPEATS:
                self.cycling_lines += 1
                if self.cycling_lines == CYCLE_LINES:
                    self.close_reason = (f"thinking repeats {CYCLE_LINES} lines {CYCLE_REPEATS} times each "
                                         f"(a cycle, {self.n_chars} characters in)")
                    return

    def feed(self, chunk, in_think = False):
        """Return True once, when the generated text has become a loop. `in_think`
        enables the long-unit rule, which content must not get: a page repeats card
        markup legitimately, thinking does not repeat a sentence a thousand times."""
        if self.reason:
            return True
        if not chunk:
            return False
        self.n_chars += len(chunk)
        self.tail = (self.tail + chunk)[-self.window:]
        for ch in chunk:   # issue #86: Thai micro-loop fast path
            if THAI.match(ch):
                self.thai_tail = (self.thai_tail + ch)[-self.thai_micro_window:]
                self.non_thai_run = 0
            elif ch.isspace():
                pass   # dividers and spacing neither trip the Thai rule nor reset it
            else:
                self.non_thai_run += 1
                if self.non_thai_run >= NON_THAI_RESET:
                    self.thai_tail = ""
        if (len(self.thai_tail) >= self.thai_micro_window
                and len(set(self.thai_tail)) <= self.thai_micro_distinct):
            self.reason = (f"{len(set(self.thai_tail))} distinct Thai-script characters "
                           f"in the last {self.thai_micro_window}")
            return True
        if in_think:
            self._think_lines(chunk)
            self.think_tail = (self.think_tail + chunk)[-self.think_window:]
            if len(self.think_tail) >= self.think_window:
                unit = self.think_tail[-self.think_unit:]
                n = self.think_tail.count(unit) if PROSE_UNIT.match(unit) else 0
                if n >= self.think_repeats:
                    self.reason = f"unit of {self.think_unit} chars repeated {n} times in the last {self.think_window} of thinking"
                    return True
        if len(self.tail) < self.window:
            return False
        distinct = len(set(self.tail))
        if distinct <= self.max_distinct:
            self.reason = f"{distinct} distinct characters in the last {self.window}"
            return True
        for p in range(1, self.max_period + 1):
            unit = self.tail[:p]
            if self.tail == (unit * (self.window // p + 1))[:self.window]:
                self.reason = f"period {p} repeated across the last {self.window}"
                return True
        return False
