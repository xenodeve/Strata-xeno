"""What an arm need not run again, in what order to run the rest, and when to stop. A machine that repeats itself (the A100: 49 of 49 texts identical)
writes the same tokens as the base arm until a new rule first fires, so a row on which the rule cannot fire is the base arm's row, token for token. 3781
took 113,756 tokens, 19 minutes, in each of two arms that could not touch it (#288).

The base arm must differ from the new one only by the rules named, and run the same engine: a build changes the tokens too (3460 took 10,995 and 11,556 tokens
in two builds with no rule between them). Each check reads the base row's thinking, or its engine log, and errs toward running."""
import json
import re
from pathlib import Path

FENCE = chr(96) * 3
CHECK_WORDS = [who + verb for verb in ('verify', 'check', 'double-check', 'test') for who in ("let's ", "Let's ", 'let me ', 'Let me ')]
CHECK_STARTS = ('Check:', 'Verify:', 'Test:', 'Double-check', 'Double check', 'Sanity check')


def thinking(text):
    return text.split('</think>')[0]


def runs_a_block(head):
    """Would `Fence` with drafts read open a block that runs? Pairs every fence as the engine does: a python one always runs, a bare one when it holds a draft."""
    at = 0
    while True:
        m = re.compile(FENCE + r'([A-Za-z0-9+#-]{0,16})\n').search(head, at)
        if not m:
            return False
        close = head.find('\n' + FENCE, m.end() - 1)
        body = head[m.end() - 1:close + 1] if close >= 0 else head[m.end() - 1:]
        language = m.group(1)
        if language in ('python', 'py', 'Python'):
            return True
        if not language and ('\ndef ' in body or '\nclass ' in body) and body.count('\n') >= 4 and '...' not in body:
            return True
        if close < 0:
            return False
        at = close + 1 + len(FENCE)


def says_check(head):
    if any(w in head for w in CHECK_WORDS):
        return True
    return any(re.search(r'(?:^|\n|[.?!] )' + re.escape(w), head) for w in CHECK_STARTS)


def events_by_row(log, count):
    """The engine's rule lines, row by row: a request's lines come before its metrics line."""
    rows, n = [[] for _ in range(count)], 0
    for line in log.split('\n'):
        m = re.search(r'strata igc (?:sandbox|loop): (.*)', line)
        if m and n < count:
            rows[n].append(m.group(1))
        if 'request metrics' in line:
            n += 1
    return rows


def silent(record, events, rules, after=512):
    """True when none of `rules` could fire on this base row. Unknown rule names are never silent."""
    head = thinking(record['text'])
    closed_by_force = any(e.startswith('closed the thinking') or 'closed the thinking' in e for e in events)
    natural_close = '</think>' in record['text'] and not closed_by_force
    share = record['tokens'] * len(head) / max(len(record['text']), 1)
    for rule in rules:
        if rule == 'draft' and runs_a_block(head):
            return False
        if rule == 'gate' and natural_close:
            return False
        if rule == 'verify' and says_check(head):
            return False
        if rule in ('pass-close', 'falsify') and 'match: ' in head:
            return False
        if rule in ('brute', 'after') and share >= after:
            return False
        if rule not in ('draft', 'gate', 'verify', 'pass-close', 'falsify', 'brute', 'after'):
            return False
    return True


class ArmPlan:
    def __init__(self, base=None, rules=(), short_first=False, verdicts=None, engine_sha=None):
        self.base, self.rules = {}, list(rules)
        self.events = {}
        self.why_not = None
        if base:
            base = Path(base)
            report = base / 'report.json'
            built = json.loads(report.read_text(encoding='utf-8')).get('executable_sha256') if report.exists() else None
            if engine_sha and built != engine_sha:
                self.why_not = f'the base arm ran engine {str(built)[:16]}, this one {engine_sha[:16]}: nothing is reused'
                self.rules = []
            records = json.loads((base / 'records.json').read_text(encoding='utf-8'))
            log = (base / 'engine.log').read_text(encoding='utf-8', errors='replace') if (base / 'engine.log').exists() else ''
            # a log that does not pair with the records cannot say how a row ended: nothing is reused then
            paired = log.count('request metrics') == len(records)
            for r, ev in zip(records, events_by_row(log, len(records))):
                self.base[r['id']] = r
                self.events[r['id']] = ev if paired else None
        self.short_first = short_first
        self.verdicts = json.loads(Path(verdicts).read_text(encoding='utf-8')) if verdicts else None
        if isinstance(self.verdicts, list):
            self.verdicts = {v['id']: v['verdict'] for v in self.verdicts}

    def order(self, rows):
        if not self.short_first or not self.base:
            return rows
        return sorted(rows, key=lambda r: self.base[r['id']]['tokens'] if r['id'] in self.base else float('inf'))

    def reusable(self, rid):
        """The base row itself, when this arm would write it again token for token."""
        r = self.base.get(rid)
        if r is None or self.events.get(rid) is None or not self.rules:
            return None
        return r if silent(r, self.events[rid], self.rules) else None

    def lost(self, rid, verdict):
        """A row the base arm got right and this one did not: the rule broke something, so the run stops and the cause is read first."""
        return bool(self.verdicts) and self.verdicts.get(rid) == 'ok' and verdict != 'ok'
