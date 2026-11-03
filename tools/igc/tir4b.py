"""Qwen3.5-4B on vLLM with the sandbox rules of the Flash-Next engine, rebuilt here: the 4B fine-tune track (#288, PRD R13 layer 2).

Strata's engine runs only the Flash-Next family, so the 4B gets the same rules in Python, by the engine's own words and limits
(include/strata/program/igc_learned_diagnostic.hpp, include/strata/core/igc_sandbox_bridge.hpp):
- after `force_after` tokens of thinking a block opens at the next line end, once (`--igc-sandbox-after`, :552): forcing every N tokens looped;
- a block the model closes runs in the kernel and its output comes back inside the thinking (:460), whoever opened the block, so a fine-tuned
  model that opens its own blocks is measured by the same path;
- a run that finished and printed closes the thinking (`--igc-sandbox-close`, :512): with no close, sandbox alone cost 1.50x plain (#279);
- a thinking that loops is closed (`repeating`, the bench's copy of the engine's checks), and an end of turn inside the thinking after a run closes it.
Not here: MIP's inline arithmetic, which lives in the CUDA engine. Every text the runner put in is a span of the record, with its kind.

tir4b.py run --model M --rows R.jsonl --out DIR [--force-after 512] [--execute] [--close-fit] [--loop-close] [--samples K] [--seed S] [--ids]
tir4b.py math-train OUT.jsonl --n N [--levels 3,4,5] [--exclude ROWS.jsonl ...]     (on the VM: MATH train, never a test row)
tir4b.py keep DIR [DIR ...] --out TRAIN.jsonl [--max-len 32768]                      (the trajectory each question trains on)"""
import argparse, hashlib, json, random, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

OPEN_LINE = 'Need a short Python program here.\n\n```python\n'   # kOpenBlock, as text
NOTHING = '[it ran and printed nothing: call your function and print the result]'   # kNothingPrinted
SPENT = '[no more runs]'
TOO_LONG = '[too long, not run]'
SHOWN = 1024   # kShownOutput, bytes
MAX_BLOCK = 2048   # kMaxBlockTokens
MAX_RUNS = 128   # kMaxRuns
CLOSE = '</think>\n\n'   # {kThinkClose, 271}
LOOP_CLOSE = '\n</think>\n\n'   # {198, kThinkClose, 271}
CHUNK = 1024   # thinking tokens between two loop checks; the engine checks every 16, so a close can come up to this much later
CONTEXT = 262144   # the model card's native context, all of it (developer, 2026-11-03)
# Qwen's precise set for thinking (model card). Its general set adds presence_penalty 1.5, which vLLM counts per request: every block the runner
# answers starts a new request and would reset it, so an arm with blocks would sample differently from one without.
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)
MATH_TAIL = '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'   # the model card's own math prompt
# --native-tool: Qwen3.5's own tool format (the checkpoint's chat_template.jinja), not a fence the runner watches. The tool goes in the system
# turn, the model calls it after its thinking, the answer comes back as a user turn of <tool_response>, and a new thinking opens.
PYTHON_TOOL = dict(type='function', function=dict(
    name='python', description='Runs Python 3 code in a sandbox and returns what it prints. Use it to compute, check or search instead of '
                               'working numbers out by hand. Print every value you need.',
    parameters=dict(type='object', properties=dict(code=dict(type='string', description='The program to run.')), required=['code'])))
TOOL_TURN = '<|im_end|>\n<|im_start|>user\n<tool_response>\n{}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n'
NO_CODE = '[no code parameter: call python with <parameter=code>]'
_CODE = re.compile(r'<parameter=code>\n?(.*?)\n?</parameter>', re.S)
# --hint-calc: CoRT's hint-engineering (arXiv 2506.09820) at test time, START's Hint-infer (2503.04625). Where the model begins a tedious hand
# calculation, its line is taken back and CoRT's own hint opens a program; after a run that printed, CoRT's second hint answers the doubt that
# made the 4B re-check a sum ten times (r1 5456f419). The thinking goes on: no close, which FT1 learned as an early end.
HINT_OPEN = 'It looks tedious, and we can use python code to simplify the reasoning.\n```python\n'
TRUST = "We don't need to doubt the accuracy of python calculations.\n"
HINT_GAP = 1024   # thinking tokens between two hints
# --whole-hint: on Qwenputer's DeepMind rows the 4B cut each step into the pieces of its own method (a long division's `21 \div 19 =`), so a
# value put in at `=` came for a piece, in a form the method could not use, and looped. This opens the reply asking for the whole step as
# one expression; its value is computed whole, sign and uneven quotient included, as Qwenputer's model writes `48213 x 9917 =`.
WHOLE_HINT = 'Let me compute the whole expression exactly in one step:\n'
# --whole-plain: v1 above took base's 121 to 127 of 160 at 0.41x tokens, but lost 14 mul_div_multiple_longer rows: the model rewrote the
# question's plain nested expression as LaTeX \frac inside \frac in a $$ block with no `=`, so the whole was never computed. The question's
# own form is what the evaluator reads exactly (-303, 78 and 319 for three of those rows), so this asks for that form.
WHOLE_HINT_PLAIN = ('Let me write the whole calculation as one plain expression, with +, -, *, / and parentheses, in the order the question '
                    'gives, and compute it exactly:\n')
# --check-hand: what the model already worked out by hand is checked at the end of its line, and a wrong value is answered with the right
# one (developer, 2026-11-04). On 82 Base answers 462 of 8,924 hand values were wrong, 5.6 an answer; rewrites (`= 189 + 7 = 196`), rounding
# to the places written and a long division's uneven quotient digit are not wrong.
# The model met 'Correction: X = V, not W.' by redoing the work by hand, and got it wrong again as often as not (dm-check: 116 of 160
# against base's 121); the correction now says it was computed and is to be used, CoRT's answer to the same distrust.
CORRECTION = 'Correction (computed exactly): {expr} = {right}, not {wrote}. Use {right} and go on.\n'
# --loop-hint: 14 of the control's 16 misses on the r1 rows were loops, 13 of them mid-work, several of them searches by hand. A loop in the
# thinking is cut back to one round and a program opens to test the idea, LOOP_HINTS times; the next loop ends the answer as before.
LOOP_HINT = 'I keep going around the same idea. Let me test it with python instead.\n```python\n'
LOOP_HINTS = 3
# The words the model should learn to write itself, as START's Hint-RFT keeps its hints in the text it trains on.
TRAIN_KINDS = ('forced open', 'close after a fitting result', 'hint open', 'trust hint', 'loop hint')
MASK_KINDS = ('output', 'inline value', 'tool response', 'hand check')   # what the kernel printed and the values software computed: read, never learned


def output_block(body):
    return '\n```output\n' + body + '\n```\n\n'


def shown_text(ok, shown):
    """What the model reads back, cut as the engine cuts it."""
    body = shown or ''
    if ok and not body.strip():
        body = NOTHING
    data = body.encode('utf-8')
    return data[:SHOWN].decode('utf-8', 'ignore') if len(data) > SHOWN else body


class Ids:
    """The special tokens of the 4B's tokenizer, looked up, never carried over from the Flash-Next pack."""
    def __init__(self, tokenizer, eos):
        self.think_close = tokenizer.convert_tokens_to_ids('</think>')
        self.ends = {tokenizer.convert_tokens_to_ids('<|im_end|>'), *eos}
        lines = [tokenizer.encode(s, add_special_tokens=False) for s in ('\n', '\n\n')]
        self.lines = {t[0] for t in lines if len(t) == 1}
        self.fence = {i for i in range(len(tokenizer)) if '```' in tokenizer.decode([i])}
        if not self.fence or len(self.lines) != 2 or not isinstance(self.think_close, int):
            raise RuntimeError('tokenizer: no fence, line or </think> token')
        # repeating() and line_back() read 198 and 271 as the line ends, the Flash-Next pack's ids: a tokenizer with others would blind them
        self.same_lines = self.lines == {198, 271}
        self.equals = {i for i in range(len(tokenizer)) if tokenizer.decode([i]).strip() == '='}   # the tokens an inline value follows
        self.tool_close = tokenizer.convert_tokens_to_ids('</tool_call>')


class Arm:
    def __init__(self, force_after=0, execute=False, close_fit=False, loop_close=False, close_repeat=0, inline_calc=False, native_tool=False,
                 hint_calc=False, loop_hint=False, inline_plain=False, whole_hint=False, check_hand=False, whole_plain=False):
        self.force_after, self.execute, self.close_fit, self.loop_close, self.close_repeat = force_after, execute, close_fit, loop_close, close_repeat
        self.inline_calc, self.native_tool, self.hint_calc, self.loop_hint = inline_calc, native_tool, hint_calc, loop_hint
        self.inline_plain, self.whole_hint, self.check_hand = inline_plain, whole_hint, check_hand
        self.whole_plain = whole_plain


# --inline-calc: MIP Native's inline value (include/strata/core/igc_inline.hpp parse_inline / after_value) for a model the engine cannot run yet.
# When the reasoning writes an arithmetic expression and `=`, the value is computed exactly and put in, with the engine's ` (exact)`; the
# model never does the arithmetic itself. On 82 base answers the 4B wrote 7,283 such equations, 504 of them wrong, in 12 of its 24 misses.
_OPS = re.compile(r'[+\-*/^×÷·⋅−]|\\times|\\cdot|\\frac|\\div')


def with_call(code):
    """(code, name): a program that defines a function of no arguments and never calls it gets the call, and its value printed, as a notebook
    cell would show it. The 4B ended all 6 hinted programs of its first answer (r1 5456f419) with `def find_n(): ...` and no call: each printed
    nothing in 0.001 s, the model took the note to call it as a failure and went back to hand algebra, though the search would have printed
    1, 11, 37. A program that does not parse is left as written, so the run shows the model its error."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, None
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    top_calls = any(isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) for s in tree.body)
    free = [s.name for s in tree.body if isinstance(s, ast.FunctionDef) and not (s.args.args or s.args.posonlyargs or s.args.kwonlyargs
                                                                                 or s.args.vararg or s.args.kwarg) and s.name not in called]
    if not free or top_calls:
        return code, None
    name = free[-1]
    return code + ('' if code.endswith('\n') else '\n') + f'\n_r = {name}()\nif _r is not None:\n    print(_r)\n', name


def loop_start(tokens, longest=4096):
    """Where the second round of the cycle that ends `tokens` begins, so a cut there keeps one round; None when the tail is no cycle of at
    least four rounds (a counting or line-back loop)."""
    for period in range(1, min(longest, len(tokens) // 2) + 1):
        if tokens[-1] != tokens[-1 - period]:
            continue
        i = len(tokens) - period - 1
        while i >= 0 and tokens[i] == tokens[i + period]:
            i -= 1
        if len(tokens) - (i + 1) >= 4 * period:
            return i + 1 + period
    return None


def tedious(expr, value):
    """Arithmetic worth a program: a number of three or more digits, three operators, or a value past 999. `1 + 1` is not."""
    return any(len(n) >= 3 for n in re.findall(r'\d+', expr)) or len(_OPS.findall(expr)) >= 3 or abs(Fraction(value)) >= 1000


def inline_found(before, whole=False):
    """(expression, spelled exact value) for the arithmetic that ends the text before an `=`, or None. The expression is read by
    tools.igc.mathexpr in every form a model writes (LaTeX at any depth, Unicode, ASCII); it refuses a symbol (algebra), a word, a mixed
    number, a root or power that is not exact. `whole`: the expression the --whole-hint asked for, whose value is given even as an uneven
    quotient, since it is the step itself, not a piece of the model's long division."""
    from tools.igc.mathexpr import expression_before, simple_quotient
    found = expression_before(before)
    if found is None:
        return None
    expr, value = found
    said = spelled(value)
    if whole:
        return expr, said
    if '/' in said and simple_quotient(expr):
        return None   # an integer quotient that does not come out: the model is doing long division and wants a digit, not 21/19 (dm div_big)
    if said == re.sub(r'[\s()]', '', expr):
        return None   # nothing new, and `= 15/13 = 15/13 = ...` looped until stuck
    return expr, said


def inline_value(before, whole=False):
    """The spelled exact value of inline_found, or None."""
    found = inline_found(before, whole)
    return found[1] if found else None


def spelled(value):
    """An integer as itself, a value with a terminating decimal as that decimal (the model's own spelling: 9/2 written for 0.9 x 5 broke its
    decimal sum into a loop, dm 6890), and only the rest as a reduced fraction, as Qwenputer's library prints."""
    if value.denominator == 1:
        return str(value.numerator)
    d = value.denominator
    twos = fives = 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        return f'{value.numerator}/{value.denominator}'
    places = max(twos, fives)
    digits = str(abs(value.numerator) * 10 ** places // value.denominator).rjust(places + 1, '0')
    return ('-' if value < 0 else '') + digits[:-places] + '.' + digits[-places:]


class Rollout:
    """One answer, a state machine the driver asks for its next request and hands each result to. Phases: think, seek (a line end for the
    forced open), peek (whether a bare fence opens python), block, answer."""
    def __init__(self, row, sample, prompt, arm, ids, tokenizer, seed, thinking=True):
        self.row, self.sample, self.prompt, self.arm, self.ids, self.tok, self.seed = row, sample, prompt, arm, ids, tokenizer, seed
        self.out, self.spans, self.events = [], [], []
        self.phase, self.thought, self.forced, self.block, self.runs, self.seen = 'think', 0, False, 0, 0, {}
        # --no-think: the prompt already ends in an empty thinking (enable_thinking=False), so the reply is the answer from its first token
        self.thinking = thinking
        if not thinking:
            self.phase = 'answer'
        self.whole_pending = False   # the next value is the whole step the --whole-hint asked for
        self.moduli = None   # --check-hand: the moduli the question and the answer so far name
        if arm.whole_hint:
            self.inject(WHOLE_HINT_PLAIN if arm.whole_plain else WHOLE_HINT, 'whole hint')
            self.whole_pending = True
        self.code_at = self.fence_at = None
        self.by = None
        self.pending = None
        self.finish = None
        self.kernel = None
        self.started = time.monotonic()
        self.seconds = None
        self.cap = 0   # --max-tokens: training data only; an answer past it is never the shortest right one kept
        self.checked = 0   # len(out) at the last loop check
        self.last_hint = -HINT_GAP   # self.thought at the last hint
        self.loop_hints = 0

    def left(self):
        room = CONTEXT - len(self.prompt) - len(self.out) - 64
        return min(room, self.cap - len(self.out)) if self.cap else room

    def model_tokens(self):
        return len(self.out) - sum(e - s for s, e, _ in self.spans)

    def text(self):
        return self.tok.decode(self.out)

    def inject(self, text, kind):
        t = self.tok.encode(text, add_special_tokens=False)
        self.spans.append([len(self.out), len(self.out) + len(t), kind])
        self.out += t
        self.events.append(dict(kind=kind, at=len(self.out) - len(t), model_tokens=self.model_tokens()))

    def done(self, finish):
        self.finish = finish
        self.seconds = round(time.monotonic() - self.started, 3)
        if self.kernel:
            self.kernel.close()
            self.kernel = None

    def request(self):
        """(max_tokens, stop token ids) of the next generation, or None once it is done."""
        if self.finish:
            return None
        left = self.left()
        if left <= 0:
            self.done('length')
            return None
        stops = set(self.ids.ends)
        if self.arm.inline_calc and self.phase in ('think', 'answer') or self.arm.hint_calc and self.phase == 'think':
            stops |= self.ids.equals
        if self.arm.native_tool and self.phase in ('think', 'answer'):
            stops.add(self.ids.tool_close)
        if self.arm.check_hand and self.phase in ('think', 'answer'):
            stops |= self.ids.lines   # every line is checked as it ends, so a correction follows the line it corrects
        if self.whole_pending:
            stops |= self.ids.lines   # the whole step may come as a line of its own, with no `=`
        if self.phase == 'answer':
            return min(left, CHUNK), stops
        stops.add(self.ids.think_close)
        if self.phase == 'peek':
            return min(2, left), stops
        if self.phase == 'seek':
            return min(64, left), stops | self.ids.lines
        if self.phase == 'block':
            return min(MAX_BLOCK - self.block, left), stops | self.ids.fence
        most = min(left, CHUNK)   # every arm is read for a loop between chunks: the stuck stop of the bench, or the close
        if self.arm.execute:
            stops |= self.ids.fence
        if self.arm.force_after and not self.forced and self.thought < self.arm.force_after:
            most = min(most, self.arm.force_after - self.thought)
        return most, stops

    def take(self, new):
        """The tokens of one generation. True when a block waits to run."""
        self.out += new
        if self.phase != 'answer' and self.phase != 'block':
            self.thought += len(new)
        last = new[-1] if new else None
        if self.arm.check_hand and self.phase in ('think', 'answer') and last in self.ids.lines:
            self.check_line()
        if self.whole_pending and self.arm.inline_calc and last in self.ids.lines:
            self.whole_line()
        if self.arm.hint_calc and last in self.ids.equals and self.phase == 'think' and self.hint_due():
            return False
        if (self.arm.inline_calc or self.arm.hint_calc) and last in self.ids.equals and self.phase in ('think', 'answer'):
            text = self.text()
            whole = self.whole_pending
            found = inline_found(text[:text.rstrip().rfind('=')], whole=whole) if self.arm.inline_calc else None
            if found is not None:   # the value the model would have computed, computed: the reasoning goes on from it
                expr, value = found
                self.whole_pending = False
                self.inject(f' {value}' if self.arm.inline_plain else f' {value} (exact)', 'inline value')
                self.events[-1].update(expr=expr[:200], value=value, whole=whole)
            if self.left() <= 0:
                self.done('length')
            elif len(self.out) - self.checked >= CHUNK:   # a request that ends at `=` skips the checks below: a loop of equations would run on
                self.checked = len(self.out)
                if looping(self.out):
                    self.on_loop()
            return False
        self.checked = len(self.out)
        if self.arm.native_tool and last == self.ids.tool_close and self.phase in ('think', 'answer'):
            text = self.text()
            m = _CODE.search(text[text.rfind('<tool_call>'):])
            self.by = 'tool' if self.phase == 'answer' else 'tool in thinking'
            self.events.append(dict(kind='tool call', by=self.by, at=len(self.out), model_tokens=self.model_tokens()))
            self.pending = m.group(1) if m and m.group(1).strip() else None
            if self.pending is None:
                self.tool_reply(NO_CODE)
                return False
            return True
        if last in self.ids.ends:
            if self.phase == 'answer':
                self.done('stop')
            elif self.runs:   # an end of turn inside thinking would leave no answer after a call (:384)
                self.out.pop()
                self.inject(LOOP_CLOSE, 'end of turn in thinking, closed it')
                self.phase = 'answer'
            else:
                self.done('stop')
            return False
        if self.phase == 'answer':
            if self.left() <= 0:
                self.done('length')
            elif looping(self.out):
                self.done('stuck')
            return False
        if last == self.ids.think_close:
            if self.phase == 'block':
                self.events.append(dict(kind='thinking closed in a block', at=len(self.out), model_tokens=self.model_tokens()))
            self.phase = 'answer'
            return False
        if self.phase == 'seek':
            if last in self.ids.lines:
                self.forced = True
                self.inject(OPEN_LINE, 'forced open')
                self.enter('forced')
            return False
        if self.phase == 'peek':
            after = self.tok.decode(self.out[self.fence_at:])
            if after.startswith('python'):
                self.code_at = len(self.tok.decode(self.out[:self.fence_at])) + len('python')
                self.by, self.block, self.phase = 'model', 0, 'block'
                self.events.append(dict(kind='model opened a block', at=self.fence_at, model_tokens=self.model_tokens()))
            else:
                self.phase = 'think'
            return False
        if self.phase == 'block':
            self.block += len(new)
            if last in self.ids.fence:
                text = self.text()
                code = text[self.code_at:text.rfind('```')]
                code = code[1:] if code.startswith('\n') else code
                if not code.strip():
                    self.phase = 'think'
                    return False
                self.pending = code
                return True
            if self.block >= MAX_BLOCK:
                self.inject('\n```' + output_block(TOO_LONG), 'block too long')
                self.phase = 'think'
            return False
        # thinking
        if last in self.ids.fence and self.arm.execute:
            if '```python' in self.tok.decode([last]):
                self.code_at = len(self.text())
                self.by, self.block, self.phase = 'model', 0, 'block'
                self.events.append(dict(kind='model opened a block', at=len(self.out) - 1, model_tokens=self.model_tokens()))
            else:
                self.fence_at, self.phase = len(self.out), 'peek'
            return False
        if self.arm.force_after and not self.forced and self.thought >= self.arm.force_after:
            self.phase = 'seek'
            return False
        if looping(self.out):
            self.on_loop()
        return False

    def on_loop(self):
        """A loop in the thinking: a program to test the idea (--loop-hint, at most LOOP_HINTS), else the close (--loop-close), else the
        bench's guard in every arm, MIP or not: a loop that cannot leave would run to the end of the context alone (external_bench)."""
        if self.phase == 'think' and self.arm.loop_hint and self.loop_hints < LOOP_HINTS:
            floor = self.spans[-1][1] if self.spans else 0
            cut = max(loop_start(self.out) or len(self.out), floor)
            taken = len(self.out) - cut
            del self.out[cut:]
            self.loop_hints += 1
            self.inject(('' if not self.out or self.out[-1] in self.ids.lines else '\n') + LOOP_HINT, 'loop hint')
            self.events[-1].update(cut_tokens=taken)
            self.enter('loop')
        elif self.phase == 'think' and self.arm.loop_close:
            self.inject(LOOP_CLOSE, 'loop close')
            self.phase = 'answer'
        else:
            self.done('stuck')

    def whole_line(self):
        """The --whole-hint's step written as a line of its own, with no `=` (pc-whole dm mul_div 3350: the whole expression, then pieces):
        its value is put after it, computed whole."""
        from tools.igc.mathexpr import _OPERATOR, evaluate
        line = self.text().rstrip('\n').rsplit('\n', 1)[-1]
        expr = re.sub(r'^\s*(?:[A-Za-z][A-Za-z ]*:\s*)?', '', line).strip().strip('$').replace('\\[', '').replace('\\]', '').strip()
        if not expr or not _OPERATOR.search(expr.lstrip('-−')):
            return
        value = evaluate(expr)
        if value is None:
            return
        self.whole_pending = False
        self.inject(f'= {spelled(value)}\n', 'inline value')
        self.events[-1].update(expr=expr[:200], value=spelled(value), whole=True)

    def check_line(self):
        """The line that just ended, every relation on it checked (tools.igc.mathexpr.wrong_relations), and a correction after it for each
        wrong one. A correction the runner put in is not checked again."""
        from tools.igc.mathexpr import moduli_in, wrong_relations
        if self.moduli is None:   # the question's moduli first: `the remainder when N is divided by 1000`
            self.moduli = moduli_in(prompt_text(self.row) if self.row.get('question') else '')
        line = self.text().rstrip('\n').rsplit('\n', 1)[-1]
        if line.startswith('Correction'):
            return
        self.moduli |= moduli_in(line)   # `12+3=15=2` is right in an answer working mod 13
        for expr, wrote, right in wrong_relations(line, self.moduli):
            self.inject(CORRECTION.format(expr=expr, right=right, wrote=wrote), 'hand check')
            self.events[-1].update(expr=expr[:200], value=right, wrote=wrote)

    def hint_due(self):
        """At an `=` that ends tedious arithmetic, spaced HINT_GAP apart: the line goes back and a hinted program opens. True when it opened."""
        if self.runs >= MAX_RUNS or self.thought - self.last_hint < HINT_GAP:
            return False
        text = self.text()
        before = text[:text.rstrip().rfind('=')]
        found = inline_found(before)
        if found is None:
            return False
        expr, value = found
        if not tedious(expr, value):
            return False
        floor = self.spans[-1][1] if self.spans else 0   # never take back text the runner put in
        cut = len(self.out)
        while cut > floor and self.out[cut - 1] not in self.ids.lines:
            cut -= 1
        taken = len(self.out) - cut
        del self.out[cut:]
        self.last_hint = self.thought
        self.inject(HINT_OPEN, 'hint open')
        self.events[-1].update(expr=expr[:200], value=value, cut_tokens=taken)
        self.enter('hint')
        return True

    def enter(self, by):
        self.code_at, self.by, self.block, self.phase = len(self.text()), by, 0, 'block'

    def tool_reply(self, body):
        """The tool's answer in the template's own shape: a user turn of <tool_response>, then a new thinking; or, for a call the model wrote
        inside its thinking (not the template's place), the response inline where it stands."""
        if self.by == 'tool':
            self.inject(TOOL_TURN.format(body), 'tool response')
        else:
            self.inject(f'\n<tool_response>\n{body}\n</tool_response>\n', 'tool response')
        self.phase = 'think'

    def execute(self, code, kernel_factory):
        """(finished, printed, seconds) of one run in this answer's kernel, which keeps its state across runs. The name of a function the run
        called for the model (with_call) is left in self.auto_call."""
        began = time.monotonic()
        self.auto_call = None
        if self.runs >= MAX_RUNS:
            ok, shown = None, SPENT
        else:
            if self.kernel and not self.kernel.alive():
                self.kernel.close()
                self.kernel = None
            self.kernel = self.kernel or kernel_factory()
            ran, self.auto_call = with_call(code)
            ok, shown = self.kernel.run(ran)
        self.runs += 1
        return ok, shown, round(time.monotonic() - began, 3)

    def run_tool(self, code, kernel_factory):
        ok, shown, took = self.execute(code, kernel_factory)
        at = len(self.out)
        self.tool_reply(shown_text(ok, shown))
        self.events[-1].update(kind='output', by=self.by, code=code[:20000], code_chars=len(code), shown=(shown or '')[:4096],
                               shown_chars=len(shown or ''), ok=ok, seconds=took, fits=bool(ok) and bool((shown or '').strip()), run_at=at, auto_call=self.auto_call)

    def run_pending(self, kernel_factory):
        """Runs the block the model closed, then puts its output and, after a fitting result, the close into the thinking."""
        code, self.pending = self.pending, None
        if self.by in ('tool', 'tool in thinking'):
            self.run_tool(code, kernel_factory)
            return
        # The repeat close is for a model that reruns its own program; a hinted block that repeats one was asked for by the runner (gen-mt5:
        # a second hint at the same sum gave the same program and closed most answers)
        if self.arm.close_repeat and self.by in ('model', 'forced') and self.seen.get(code, 0) + 1 >= self.arm.close_repeat:
            self.inject(LOOP_CLOSE, 'repeated program close')
            self.phase = 'answer'
            return
        self.seen[code] = self.seen.get(code, 0) + 1
        ok, shown, took = self.execute(code, kernel_factory)
        fits = bool(ok) and bool((shown or '').strip())   # printed(): it finished and said something
        at = len(self.out)
        self.inject(output_block(shown_text(ok, shown)), 'output')
        self.events[-1].update(kind='output', by=self.by, code=code[:20000], code_chars=len(code), shown=(shown or '')[:4096],
                               shown_chars=len(shown or ''), ok=ok, seconds=took, fits=fits, run_at=at, auto_call=self.auto_call)
        if self.by == 'hint' and fits:
            self.inject(TRUST, 'trust hint')
        if self.arm.close_fit and fits:
            self.inject(CLOSE, 'close after a fitting result')
            self.phase = 'answer'
        else:
            self.phase = 'think'

    def record(self, right):
        text = self.text()
        runs = [e for e in self.events if e['kind'] == 'output']
        return dict(id=self.row['id'], suite=self.row['suite'], answer=self.row['answer'], sample=self.sample, seed=self.seed, text=text,
                    right=right(self.row, text if self.thinking else CLOSE + text), finish=self.finish, tokens=self.model_tokens(), injected=len(self.out) - self.model_tokens(),
                    prompt_tokens=len(self.prompt), thinking_tokens=self.thought, seconds=self.seconds, events=self.events,
                    runs=len(runs), runs_by_model=sum(e['by'] == 'model' for e in runs), runs_forced=sum(e['by'] == 'forced' for e in runs),
                    runs_by_tool=sum(e['by'] in ('tool', 'tool in thinking') for e in runs),
                    inline_values=sum(e['kind'] == 'inline value' for e in self.events),
                    first_run_at=runs[0]['model_tokens'] if runs else None, tool_seconds=round(sum(e['seconds'] for e in runs), 3))


def kv_tokens(llm):
    """The tokens vLLM's KV cache holds, or None when this vLLM does not say."""
    try:
        c = llm.llm_engine.vllm_config.cache_config
        return int(c.num_gpu_blocks) * int(c.block_size)
    except Exception:  # noqa: BLE001 - an older or newer engine object: the caller keeps the --live it was given
        return None


def looping(tokens):
    from tools.igc.external_score import repeating
    return repeating(tokens)


def eos_ids(model, tok, generation_config):
    """The end tokens wherever the checkpoint states them: Qwen3.5-4B ships no generation_config.json, and its config keeps the id under text_config."""
    ids = {tok.eos_token_id} - {None}
    try:
        e = generation_config.from_pretrained(model).eos_token_id
        ids |= set(e if isinstance(e, list) else [e]) - {None}
    except OSError:
        pass
    config = json.loads((Path(model) / 'config.json').read_text(encoding='utf-8'))
    for c in (config, config.get('text_config', {})):
        e = c.get('eos_token_id')
        ids |= set(e if isinstance(e, list) else [e]) - {None}
    return sorted(ids)


def prompt_text(row):
    from tools.igc.standard_rows import plain_prompt
    text = row.get('question') or plain_prompt(row)
    return text + MATH_TAIL if row['suite'] == 'stdmath' else text


def seed_of(seed, row_id, sample):
    return int.from_bytes(hashlib.sha256(f'{seed}:{row_id}:{sample}'.encode()).digest()[:4], 'little')


def drive(llm, rollouts, kernel_factory, live_most=256, workers=32, on_round=None):
    """Continuous: an answer's next request goes into the engine the moment its last one returns, and a block it closed runs on a worker
    meanwhile. Lock-step rounds (one generate() for every live answer, then the next) held each answer to the longest request of its round:
    the inline arm stops at every `=` and ran 5x slower than its control (r1 on the A100, 2 rows to 11 in 20 minutes). At most `live_most`
    answers are live at once, each holding its kernel until it ends, so a few thousand answers do not hold a few thousand kernels."""
    from concurrent.futures import FIRST_COMPLETED, wait
    from vllm import SamplingParams, TokensPrompt
    engine = llm.llm_engine
    pool = ThreadPoolExecutor(workers)
    waiting_rows = list(rollouts)
    flying, running = {}, {}   # request id -> answer; a block's future -> answer
    serial = iter(range(1 << 62))

    def ask(r):
        """Sends r's next request; False once r has ended."""
        q = r.request()
        if not q:
            return False
        most, stops = q
        params = SamplingParams(**SAMPLING, max_tokens=most, stop_token_ids=sorted(stops), seed=(r.seed + len(r.out)) % 2**31,
                                skip_special_tokens=False, detokenize=False)
        rid = str(next(serial))
        engine.add_request(rid, TokensPrompt(prompt_token_ids=r.prompt + r.out), params)
        flying[rid] = r
        return True

    while waiting_rows or flying or running:
        ended = False
        while waiting_rows and len(flying) + len(running) < live_most:
            r = waiting_rows.pop(0)
            r.started = time.monotonic()
            ended |= not ask(r)
        if flying:
            for o in engine.step():
                if not o.finished:
                    continue
                r = flying.pop(o.request_id)
                c = o.outputs[0]
                new = list(c.token_ids)
                if isinstance(c.stop_reason, int) and (not new or new[-1] != c.stop_reason):
                    new.append(c.stop_reason)
                if r.take(new):
                    running[pool.submit(r.run_pending, kernel_factory)] = r
                else:
                    ended |= not ask(r)
        else:
            wait(list(running), return_when=FIRST_COMPLETED)
        for f in [f for f in running if f.done()]:
            r = running.pop(f)
            f.result()
            ended |= not ask(r)
        if ended and on_round:
            on_round()
    pool.shutdown()


def stop_words(stops, ids, tok):
    """The stop token ids of a request as the strings llama-server stops on: it takes words, not ids. The end of turn is its own eos."""
    words = set()
    if stops & ids.equals:
        words.add('=')
    if stops & ids.lines:
        words.add('\n')
    if stops & ids.fence:
        words.add('```')
    if ids.think_close in stops:
        words.add('</think>')
    if getattr(ids, 'tool_close', None) in stops:
        words.add('</tool_call>')
    return sorted(words)


def drive_http(post, tok, ids, rollouts, kernel_factory, live_most=16, on_round=None):
    """drive() over llama-server's /completion, so the 5060 Ti runs the same rules on Windows, where vLLM does not (developer, 2026-11-04).
    Each answer runs on its own thread and the server batches its slots. The server stops on words and leaves the word out of the text, so
    it is put back and the text encoded again: the same Rollout takes it."""
    import threading
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    lock = threading.Lock()
    sampling = dict(temperature=SAMPLING['temperature'], top_p=SAMPLING['top_p'], top_k=SAMPLING['top_k'], min_p=SAMPLING['min_p'],
                    presence_penalty=SAMPLING['presence_penalty'], repeat_penalty=SAMPLING['repetition_penalty'])

    def one(r):
        r.started = time.monotonic()
        while True:
            ask = r.request()
            if not ask:
                break
            most, stops = ask
            reply = post(dict(prompt=r.prompt + r.out, n_predict=most, stop=stop_words(stops, ids, tok), cache_prompt=True,
                              seed=(r.seed + len(r.out)) % 2**31, return_tokens=True, **sampling))
            if reply.get('tokens') is not None:
                # the ids the model generated, its stop token among them: a stop on '\n' inside the token '\n\n' put back as '\n' fed the
                # model a text it never wrote, every line (pc-check: 21 of 48 answers ran to the cap against base's 7)
                new = list(reply['tokens'])
            else:
                text = reply.get('content', '') + (reply.get('stopping_word', '') if reply.get('stop_type') == 'word' else '')
                new = tok.encode(text, add_special_tokens=False) if text else []
                if reply.get('stop_type') == 'eos':
                    new.append(im_end)
            if not new:
                r.done('length')   # nothing came back at the limit: no answer can come
                break
            if r.take(new):
                r.run_pending(kernel_factory)
        if on_round:
            with lock:
                on_round()
    with ThreadPoolExecutor(live_most) as pool:
        list(pool.map(one, rollouts))


def run(args):
    import os
    os.environ.setdefault('STRATA_SANDBOX_THROWAWAY', '1')   # the kernel runs bare only on a VM that is thrown away (exec_bridge.throwaway)
    from serve.sandbox.exec_bridge import Kernel, throwaway
    from tools.igc.select_rows import right
    from transformers import AutoTokenizer, GenerationConfig

    if (args.execute or args.native_tool or args.hint_calc or args.loop_hint) and not throwaway():
        raise SystemExit('a block runs only on a throwaway VM')
    rows = [json.loads(l) for l in Path(args.rows).read_text(encoding='utf-8').splitlines() if l.strip()]
    rows = rows[args.offset:args.offset + args.limit] if args.limit else rows[args.offset:]
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = Ids(tok, eos_ids(args.model, tok, GenerationConfig))
    arm = Arm(args.force_after, args.execute, args.close_fit, args.loop_close, args.close_repeat, args.inline_calc, args.native_tool, args.hint_calc, args.loop_hint, args.inline_plain,
              args.whole_hint, args.check_hand, args.whole_plain)
    if args.whole_hint and not args.inline_calc:
        raise SystemExit('--whole-hint asks for a value that only --inline-calc puts in')
    if args.greedy:   # drive() and arm.json read SAMPLING: temperature 0 is vLLM's greedy
        SAMPLING.update(temperature=0.0)
    if args.native_tool and tok.convert_tokens_to_ids('</tool_call>') == tok.unk_token_id:
        raise SystemExit('the tokenizer has no </tool_call> token')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # --resume: answers already written are kept and not run again. A restart to fix the live count lost every answer in flight but not these.
    kept = json.loads((out / 'records.json').read_text(encoding='utf-8')) if args.resume and (out / 'records.json').exists() else []
    have = {(r['id'], r['sample']) for r in kept}
    rollouts = []
    for row in rows:
        # As text, then encoded: transformers 5 returns a dict from tokenize=True, and list() of it gave the keys as the prompt.
        system = [{'role': 'system', 'content': args.system}] if args.system else []
        text = tok.apply_chat_template(system + [{'role': 'user', 'content': prompt_text(row)}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=not args.no_think, tools=[PYTHON_TOOL] if args.native_tool else None)
        prompt = tok.encode(text, add_special_tokens=False)
        opened = text.rstrip().endswith('</think>') if args.no_think else text.rstrip().endswith('<think>')
        if not opened or not all(isinstance(t, int) for t in prompt):
            raise SystemExit('the chat template did not end as --no-think asks: ' + repr(text[-80:]))
        for k in range(args.samples):
            if (row['id'], k) in have:
                continue
            rollouts.append(Rollout(row, k, list(prompt), arm, ids, tok, seed_of(args.seed, row['id'], k), thinking=not args.no_think))
            rollouts[-1].cap = args.max_tokens
    if args.force_after and not args.execute:
        raise SystemExit('--force-after opens a block that only --execute runs')
    if not ids.same_lines:   # the stuck stop and the loop close both read them
        raise SystemExit(f'the line ends are {sorted(ids.lines)}, not the 198 and 271 the loop checks read')
    began = time.monotonic()
    kwargs, llm, kv = dict(server=args.server), None, None
    if not args.server:
        from vllm import LLM
        kwargs = dict(model=args.model, dtype='bfloat16', max_model_len=CONTEXT, gpu_memory_utilization=args.memory, enable_prefix_caching=True,
                      language_model_only=True, seed=args.seed)
        try:
            llm = LLM(**kwargs)
        except TypeError:   # a vLLM with no text-only switch loads the vision tower too, which costs memory and changes nothing here
            kwargs.pop('language_model_only')
            llm = LLM(**kwargs)
        # Live answers past what the KV cache holds are run in turns and lose their prefix cache: every chunk reads its whole context again.
        # It stalled three runs (an L4 at 164 and 16 live, an A100 at 32 live on half the memory), so the count follows the cache here.
        kv = kv_tokens(llm)
    expect = args.max_tokens or args.expect_len
    if kv and args.live * expect > kv:
        fit = max(1, int(0.85 * kv / expect))
        print(f'--live {args.live} x {expect} tokens is past the KV cache ({kv} tokens): {fit} live', flush=True)
        args.live = fit
    (out / 'arm.json').write_text(json.dumps(dict(vars(args), kv_tokens=kv, sampling=SAMPLING, context=CONTEXT, load_seconds=round(time.monotonic() - began, 1),
                                                   llm={k: v for k, v in kwargs.items() if k != 'model'}, think_close=ids.think_close,
                                                   ends=sorted(ids.ends), lines=sorted(ids.lines), fences=len(ids.fence))), encoding='utf-8')
    began = time.monotonic()
    written = [0, 0.]

    def save(last=False):
        done = [r for r in rollouts if r.finish]
        if len(done) == written[0] or (not last and time.monotonic() - written[1] < 30):
            return
        written[0], written[1] = len(done), time.monotonic()
        records = kept + [r.record(right) for r in done]
        (out / 'records.json').write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
        if args.ids:
            with open(out / 'ids.jsonl', 'w', encoding='utf-8') as f:
                for r in done:
                    f.write(json.dumps(dict(id=r.row['id'], sample=r.sample, prompt_ids=r.prompt, out_ids=r.out, spans=r.spans)) + '\n')
        print(f'{len(done)}/{len(rollouts)} done, {sum(x["right"] for x in records)} right, {round(time.monotonic() - began)} s', flush=True)
    if args.server:   # llama-server on this PC's 5060 Ti: vLLM does not run on Windows
        import urllib.request

        def post(body):
            req = urllib.request.Request(args.server.rstrip('/') + '/completion', data=json.dumps(body).encode('utf-8'),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=3600) as reply:
                return json.loads(reply.read().decode('utf-8'))
        drive_http(post, tok, ids, rollouts, Kernel, live_most=args.live, on_round=save)
    else:
        drive(llm, rollouts, Kernel, live_most=args.live, on_round=save)
    save(last=True)
    records = kept + [r.record(right) for r in rollouts]
    (out / 'report.json').write_text(json.dumps(dict(rows=len(rows), samples=args.samples, answers=len(records), right=sum(r['right'] for r in records),
                                                      tokens=sum(r['tokens'] for r in records), seconds=round(time.monotonic() - began, 1),
                                                      runs=sum(r['runs'] for r in records), runs_by_model=sum(r['runs_by_model'] for r in records),
                                                      length=sum(r['finish'] == 'length' for r in records))), encoding='utf-8')
    print('DONE', flush=True)


def math_train(args):
    """MATH's train split, levels picked, gold from the last \\boxed of the solution. A row whose id matches an evaluated row is left out:
    the ids are the hash of the problem (question_id), the same as standard_rows builds for the test rows."""
    from datasets import load_dataset
    from tools.igc.external_score import question_id
    from tools.igc.standard_score import last_boxed, normalize_math
    banned = set()
    for path in args.exclude:
        banned |= {json.loads(l)['id'] for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()}
    levels = {f'Level {n}' for n in args.levels.split(',')}
    rows = []
    for subject in ('algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'):
        for r in load_dataset('EleutherAI/hendrycks_math', subject, split='train'):
            gold = last_boxed(r['solution'])
            rid = question_id('stdmath', r['problem'])
            if r['level'] in levels and gold and rid not in banned:
                rows.append(dict(id=rid, suite='stdmath', question=r['problem'], answer=normalize_math(gold), group=subject, level=r['level']))
    random.Random(args.seed).shuffle(rows)
    Path(args.out).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows[:args.n]), encoding='utf-8')
    print('rows', min(args.n, len(rows)), 'of', len(rows), 'left out as evaluated', len(banned), flush=True)


def choose(records):
    """Per question, the trajectory it trains on: right, ended, no text of ours but the open and the close, and the shortest. A run whose result
    fitted is preferred, since that is the behaviour taught; a question answered before any block opened keeps its short tool-free answer, so the
    model also learns that an easy question needs no program (PRD R13)."""
    by = {}
    for r in records:
        kinds = {e['kind'] for e in r['events']} - set(TRAIN_KINDS) - set(MASK_KINDS) - {'model opened a block'}
        if r['right'] and r['finish'] == 'stop' and not kinds:
            by.setdefault(r['id'], []).append(r)
    chosen = {}
    for rid, rs in by.items():
        fitted = [r for r in rs if any(e.get('fits') for e in r['events'])]
        plain = [r for r in rs if not r['runs']]
        pick = fitted or plain
        if pick:
            chosen[rid] = min(pick, key=lambda r: r['tokens'] + r['injected'])
    return chosen


def keep(args):
    records, traj = [], {}
    for d in args.dirs:
        records += json.loads((Path(d) / 'records.json').read_text(encoding='utf-8'))
        for l in (Path(d) / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
            t = json.loads(l)
            traj[(t['id'], t['sample'])] = t
    chosen = choose(records)
    n = long = 0
    with open(args.out, 'w', encoding='utf-8') as f:
        for rid, r in chosen.items():
            t = traj[(rid, r['sample'])]
            if len(t['prompt_ids']) + len(t['out_ids']) > args.max_len:
                long += 1
                continue
            f.write(json.dumps(dict(id=rid, sample=r['sample'], prompt_ids=t['prompt_ids'], out_ids=t['out_ids'],
                                    mask=[s[:2] for s in t['spans'] if s[2] in MASK_KINDS], runs=r['runs'])) + '\n')
            n += 1
    print(f'questions {len({r["id"] for r in records})}, kept {n} ({sum(c["runs"] > 0 for c in chosen.values())} with a fitting run), '
          f'over {args.max_len} tokens {long}', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--model', required=True)
    r.add_argument('--rows', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--samples', type=int, default=1)
    r.add_argument('--seed', type=int, default=1)
    r.add_argument('--limit', type=int, default=0)
    r.add_argument('--offset', type=int, default=0)
    r.add_argument('--max-tokens', type=int, default=0)   # 0: the whole context, as every evaluated arm runs
    r.add_argument('--force-after', type=int, default=0)
    r.add_argument('--execute', action='store_true')
    r.add_argument('--close-fit', action='store_true')
    r.add_argument('--loop-close', action='store_true')
    r.add_argument('--close-repeat', type=int, default=0)
    r.add_argument('--inline-calc', action='store_true')
    r.add_argument('--whole-plain', action='store_true', help="--whole-hint asking for one plain expression in the question's order, not LaTeX")
    r.add_argument('--check-hand', action='store_true', help='check every hand value at the end of its line; a wrong one gets a correction')
    r.add_argument('--whole-hint', action='store_true', help='open the reply asking for the whole step as one expression, computed whole')
    r.add_argument('--inline-plain', action='store_true', help="the value alone, as Qwenputer prints it, without the engine's ' (exact)'")
    r.add_argument('--native-tool', action='store_true', help="offer a python tool in Qwen3.5's own tool format and answer its calls")
    r.add_argument('--hint-calc', action='store_true', help="CoRT's hint where a tedious hand calculation begins, and its trust line after a run")
    r.add_argument('--loop-hint', action='store_true', help='a loop in the thinking is cut to one round and a program opens to test the idea')
    r.add_argument('--server', default='', help='a llama-server URL (127.0.0.1) to run on in place of vLLM: this PC, on Windows')
    r.add_argument('--no-think', action='store_true', help='enable_thinking=False: the reply is the answer, as Qwenputer measures arithmetic')
    r.add_argument('--greedy', action='store_true', help='temperature 0, for a protocol that publishes greedy numbers (Qwenputer)')
    r.add_argument('--system', default='', help='a system turn before the question (with --native-tool it follows the tool list)')
    r.add_argument('--ids', action='store_true')
    r.add_argument('--memory', type=float, default=0.9)
    r.add_argument('--live', type=int, default=256)
    r.add_argument('--resume', action='store_true')
    r.add_argument('--expect-len', type=int, default=24000)   # tokens one live answer may reach, for the KV fit; a capped run uses its cap
    m = sub.add_parser('math-train')
    m.add_argument('out')
    m.add_argument('--n', type=int, required=True)
    m.add_argument('--levels', default='3,4,5')
    m.add_argument('--seed', type=int, default=1)
    m.add_argument('--exclude', nargs='*', default=[])
    k = sub.add_parser('keep')
    k.add_argument('dirs', nargs='+')
    k.add_argument('--out', required=True)
    k.add_argument('--max-len', type=int, default=32768)
    args = p.parse_args(argv)
    {'run': run, 'math-train': math_train, 'keep': keep}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
