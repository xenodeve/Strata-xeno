"""tools/igc/tir4b.py and tir4b_train.py without a GPU: a scripted model drives the forced-block state machine through each path the
engine's rules take, and the training labels mask what the kernel printed. python tools/test_igc_tir4b.py"""
import sys, types, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc import tir4b
from tools.igc.tir4b_train import labels_of

SPECIAL = {'</think>': 5, '<|im_end|>': 6, '```python': 7, '```': 8, '</tool_call>': 9, '<tool_call>': 10, '<|im_start|>': 11, '\n\n': 271, '\n': 198}


class Tok:
    """Longest match over the specials, one id per other character."""
    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for s in sorted(SPECIAL, key=len, reverse=True):
                if text.startswith(s, i):
                    out.append(SPECIAL[s])
                    i += len(s)
                    break
            else:
                out.append(1000 + ord(text[i]))
                i += 1
        return out

    def decode(self, ids):
        back = {v: k for k, v in SPECIAL.items()}
        return ''.join(back[i] if i in back else chr(i - 1000) if i >= 1000 else '' for i in ids)

    def convert_tokens_to_ids(self, s):
        return SPECIAL[s]

    def __len__(self):
        return 1000 + 200


class Kernel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.ran = []

    def alive(self):
        return True

    def run(self, code):
        self.ran.append(code)
        return self.replies.pop(0)

    def close(self):
        pass


def play(rollout, script, kernel):
    """The model writes `script` in order; each request takes what fits its limit and stops after a stop token, as vLLM does."""
    stream = TOK.encode(script)
    while True:
        ask = rollout.request()
        if not ask:
            return
        most, stops = ask
        new = []
        while stream and len(new) < most:
            new.append(stream.pop(0))
            if new[-1] in stops:
                break
        if not new:   # the script ran out: the model ends its turn
            new = [SPECIAL['<|im_end|>']]
        if rollout.take(new):
            rollout.run_pending(lambda: kernel)


TOK = Tok()
IDS = tir4b.Ids(TOK, [6])
ROW = dict(id='q1', suite='gsm', answer='42')


def rollout(**arm):
    return tir4b.Rollout(ROW, 0, [1, 2, 3], tir4b.Arm(**arm), IDS, TOK, seed=1)


def kinds(r):
    return [e['kind'] for e in r.events]


class ForcedBlock(unittest.TestCase):
    def test_ids_are_the_tokenizers(self):
        self.assertEqual(IDS.think_close, 5)
        self.assertEqual(IDS.lines, {198, 271})
        self.assertTrue(IDS.same_lines)
        self.assertEqual(IDS.fence, {7, 8})

    def test_forced_block_runs_and_a_fitting_print_closes(self):
        r = rollout(force_after=10, execute=True, close_fit=True)
        k = Kernel([(True, '42')])
        play(r, 'aaaaaaaaaaaaa rest of line\nprint(6*7)\n```The answer is 42.<|im_end|>', k)
        self.assertEqual(k.ran, ['print(6*7)\n'])
        self.assertEqual(kinds(r), ['forced open', 'output', 'close after a fitting result'])
        text = r.text()
        self.assertEqual(text, 'aaaaaaaaaaaaa rest of line\n' + tir4b.OPEN_LINE + 'print(6*7)\n```\n```output\n42\n```\n\n</think>\n\nThe answer is 42.<|im_end|>')
        self.assertEqual(r.phase, 'answer')
        self.assertEqual(r.finish, 'stop')
        rec = r.record(lambda row, t: '42' in t.rsplit('</think>', 1)[-1])
        self.assertTrue(rec['right'])
        self.assertEqual((rec['runs'], rec['runs_forced'], rec['runs_by_model']), (1, 1, 0))
        self.assertEqual(rec['injected'] + rec['tokens'], len(r.out))
        out = [s for s in r.spans if s[2] == 'output'][0]
        self.assertEqual(TOK.decode(r.out[out[0]:out[1]]), '\n```output\n42\n```\n\n')

    def test_nothing_printed_keeps_thinking(self):
        r = rollout(force_after=4, execute=True, close_fit=True)
        k = Kernel([(True, '')])
        play(r, 'abcdef\nx = 1\n```on by hand</think>\n\n42<|im_end|>', k)
        self.assertIn(tir4b.NOTHING, r.text())
        self.assertIn('on by hand</think>', r.text())
        self.assertEqual(kinds(r), ['forced open', 'output'])
        self.assertFalse(r.events[-1]['fits'])

    def test_an_error_keeps_thinking(self):
        r = rollout(force_after=4, execute=True, close_fit=True)
        play(r, 'abcdef\n1/0\n```fix</think>\n\n42<|im_end|>', Kernel([(False, 'ZeroDivisionError')]))
        self.assertEqual(kinds(r), ['forced open', 'output'])
        self.assertIn('fix</think>', r.text())

    def test_force_opens_once(self):
        r = rollout(force_after=4, execute=True)
        play(r, 'abcdef\nprint(1)\n```' + 'b' * 40 + '\n' + 'c' * 40 + '</think>\n\n1<|im_end|>', Kernel([(True, '1')]))
        self.assertEqual(kinds(r).count('forced open'), 1)

    def test_model_opens_its_own_block(self):
        r = rollout(execute=True, close_fit=True)
        k = Kernel([(True, '7')])
        play(r, 'let me check ```python\nprint(7)\n```</think>\n\n7<|im_end|>', k)
        self.assertEqual(k.ran, ['print(7)\n'])
        self.assertEqual(kinds(r), ['model opened a block', 'output', 'close after a fitting result'])
        self.assertEqual(r.record(lambda *_: True)['runs_by_model'], 1)

    def test_bare_fence_then_python(self):
        r = rollout(execute=True)
        k = Kernel([(True, '3')])
        play(r, 'so ```python\nprint(3)\n```then</think>\n\n3<|im_end|>'.replace('```python', '```' + 'python'), k)
        self.assertEqual(k.ran, ['print(3)\n'])

    def test_bare_fence_not_python_goes_on(self):
        r = rollout(execute=True)
        play(r, 'a ```text``` b</think>\n\nx<|im_end|>', Kernel([]))
        self.assertEqual(r.finish, 'stop')
        self.assertEqual(r.runs, 0)

    def test_plain_arm_runs_nothing(self):
        r = rollout()
        play(r, 'a ```python\nprint(1)\n``` b</think>\n\nx<|im_end|>', Kernel([]))
        self.assertEqual((r.runs, r.events, r.spans), (0, [], []))
        self.assertEqual(r.text(), 'a ```python\nprint(1)\n``` b</think>\n\nx<|im_end|>')

    def test_a_loop_stops_a_plain_arm_and_closes_a_sandbox_arm(self):
        plain = rollout()
        play(plain, 'ab' * 700 + '</think>\n\nx<|im_end|>', Kernel([]))
        self.assertEqual(plain.finish, 'stuck')
        self.assertEqual(len(plain.out), tir4b.CHUNK)   # read between chunks, as the bench reads every 64 tokens
        closed = rollout(execute=True, loop_close=True)
        play(closed, 'ab' * 700, Kernel([]))
        self.assertIn('loop close', kinds(closed))

    def test_end_of_turn_in_thinking_after_a_run_closes(self):
        r = rollout(execute=True)
        play(r, '```python\nprint(2)\n```hm<|im_end|>2<|im_end|>', Kernel([(True, '2')]))
        self.assertIn('end of turn in thinking, closed it', kinds(r))
        self.assertIn('hm\n</think>\n\n2', r.text())

    def test_repeated_program_closes(self):
        r = rollout(execute=True, close_repeat=2)
        k = Kernel([(False, 'e')])
        play(r, '```python\nf()\n```x ```python\nf()\n```y</think>\n\nz<|im_end|>', k)
        self.assertEqual(len(k.ran), 1)
        self.assertIn('repeated program close', kinds(r))

    def test_output_is_cut_like_the_engine(self):
        self.assertEqual(len(tir4b.shown_text(True, 'x' * 5000).encode()), tir4b.SHOWN)
        self.assertEqual(tir4b.shown_text(True, '  '), tir4b.NOTHING)
        self.assertEqual(tir4b.shown_text(False, ''), '')


class InlineCalc(unittest.TestCase):
    def test_value_of_the_expression_before_equals(self):
        v = tir4b.inline_value
        self.assertEqual(v('so 3 \\times 4 \\times 7 '), '84')
        self.assertEqual(v('then 14641 - 2662 - 4235 + 352 + 324 '), '8420')   # a sum the 4B got wrong by hand (4d822b73)
        self.assertEqual(v('$0.1 + 0.2 '), '3/10')                              # exact, not a float
        self.assertEqual(v('\\frac{3}{4} + 1/4 '), '1')
        self.assertEqual(v('(140 + 56) / 28 '), '7')
        self.assertEqual(v('2^{10} '), '1024')

    def test_not_arithmetic_is_left_to_the_model(self):
        v = tir4b.inline_value
        self.assertIsNone(v('2x + 3 '))              # algebra
        self.assertIsNone(v('so x '))                # an assignment
        self.assertIsNone(v('a3 + 4 '))              # the tail of a word
        self.assertIsNone(v('9^{999} '))             # too large to compute in a step
        self.assertIsNone(v('\\sqrt{8} + 1 '))       # not rational
        self.assertIsNone(v('(3 + 4 '))              # unbalanced
        self.assertIsNone(v('17 '))                  # no operator
        self.assertIsNone(v('1.2.3 + 1 '))           # a section number: F('1.2').F('3') killed the whole runner (gen-mt5)
        self.assertIsNone(v('(1)(2) + 3 '))          # a call on a number

    def test_the_rollout_puts_the_value_in(self):
        r = rollout(inline_calc=True)
        play(r, 'so 6 \\times 7 = and on</think>\n\n42<|im_end|>', Kernel([]))
        self.assertIn('6 \\times 7 = 42 (exact) and on', r.text())
        self.assertEqual(kinds(r), ['inline value'])
        self.assertEqual((r.events[0]['expr'], r.events[0]['value']), ('6 \\times 7', '42'))
        self.assertEqual(r.record(lambda *_: True)['inline_values'], 1)
        out = [s for s in r.spans if s[2] == 'inline value'][0]
        self.assertEqual(TOK.decode(r.out[out[0]:out[1]]), ' 42 (exact)')

    def test_off_by_default(self):
        r = rollout()
        play(r, 'so 6 \\times 7 = 41</think>\n\nx<|im_end|>', Kernel([]))
        self.assertEqual(r.events, [])

    def test_a_loop_of_equations_is_stuck(self):
        # every request ends at an `=`, never at a chunk: the loop check must still come round, or it runs to the end of the context
        r = rollout(inline_calc=True)
        play(r, 'so 1 + 1 = ' * 3000, Kernel([]))
        self.assertEqual(r.finish, 'stuck')
        self.assertLess(len(r.out), 3 * tir4b.CHUNK)


CALL = '<tool_call>\n<function=python>\n<parameter=code>\nprint(6*7)\n</parameter>\n</function>\n</tool_call>'


class NativeTool(unittest.TestCase):
    def test_a_call_in_the_reply_gets_a_tool_turn_and_thinks_again(self):
        r = rollout(native_tool=True)
        k = Kernel([(True, '42')])
        play(r, 'plan</think>\n\n' + CALL + 'it printed 42</think>\n\n42<|im_end|>', k)
        self.assertEqual(k.ran, ['print(6*7)'])
        self.assertEqual(r.text(), 'plan</think>\n\n' + CALL + '<|im_end|>\n<|im_start|>user\n<tool_response>\n42\n</tool_response><|im_end|>\n'
                         '<|im_start|>assistant\n<think>\nit printed 42</think>\n\n42<|im_end|>')
        self.assertEqual(r.finish, 'stop')
        rec = r.record(lambda *_: True)
        self.assertEqual((rec['runs'], rec['runs_by_tool']), (1, 1))
        out = [e for e in r.events if e['kind'] == 'output'][0]
        self.assertEqual((out['by'], out['shown']), ('tool', '42'))

    def test_a_call_inside_thinking_is_answered_inside_thinking(self):
        r = rollout(native_tool=True)
        k = Kernel([(True, '42')])
        play(r, 'hm ' + CALL + 'so 42</think>\n\n42<|im_end|>', k)
        self.assertIn(CALL + '\n<tool_response>\n42\n</tool_response>\nso 42</think>', r.text())
        self.assertEqual([e['by'] for e in r.events if e['kind'] == 'output'], ['tool in thinking'])

    def test_a_call_with_no_code_is_told_so(self):
        r = rollout(native_tool=True)
        play(r, 'x</think>\n\n<tool_call>\n<function=python>\n</function>\n</tool_call>ok</think>\n\n1<|im_end|>', Kernel([]))
        self.assertIn(tir4b.NO_CODE, r.text())
        self.assertEqual(r.finish, 'stop')

    def test_the_prompt_offers_the_tool(self):
        self.assertEqual(tir4b.PYTHON_TOOL['function']['name'], 'python')
        self.assertEqual(tir4b.PYTHON_TOOL['function']['parameters']['required'], ['code'])


class HintCalc(unittest.TestCase):
    def test_a_tedious_hand_calculation_becomes_a_program(self):
        r = rollout(hint_calc=True)
        k = Kernel([(True, '56088')])
        play(r, 'think a bit\nso 123 \\times 456 =print(123*456)\n```then done</think>\n\n56088<|im_end|>', k)
        self.assertEqual(k.ran, ['print(123*456)\n'])
        self.assertEqual(r.text(), 'think a bit\n' + tir4b.HINT_OPEN + 'print(123*456)\n```' + tir4b.output_block('56088') + tir4b.TRUST
                         + 'then done</think>\n\n56088<|im_end|>')
        hint = [e for e in r.events if e['kind'] == 'hint open'][0]
        self.assertEqual((hint['expr'], hint['cut_tokens']), ('123 \\times 456', len(TOK.encode('so 123 \\times 456 ='))))
        self.assertEqual([e['by'] for e in r.events if e['kind'] == 'output'], ['hint'])
        self.assertEqual(r.phase, 'answer')

    def test_small_arithmetic_is_left_alone(self):
        r = rollout(hint_calc=True)
        play(r, 'so 1 + 1 = 2</think>\n\n2<|im_end|>', Kernel([]))
        self.assertEqual(r.events, [])

    def test_hints_are_spaced(self):
        r = rollout(hint_calc=True)
        k = Kernel([(True, '56088')])
        play(r, '\n123 \\times 456 =print(1)\n```and 789 \\times 321 = 253269</think>\n\n1<|im_end|>', k)
        self.assertEqual(len(k.ran), 1)
        self.assertIn('789 \\times 321 = 253269', r.text())

    def test_no_trust_line_after_an_error(self):
        r = rollout(hint_calc=True)
        play(r, '\n123 \\times 456 =1/0\n```hm</think>\n\n1<|im_end|>', Kernel([(False, 'ZeroDivisionError')]))
        self.assertNotIn(tir4b.TRUST, r.text())


class AutoCall(unittest.TestCase):
    def test_a_function_defined_and_never_called_is_called(self):
        code = 'def find_n():\n    for n in range(5):\n        print(n)\n'
        self.assertEqual(tir4b.with_call(code), (code + '\n_r = find_n()\nif _r is not None:\n    print(_r)\n', 'find_n'))

    def test_left_alone(self):
        for code in ('def f():\n    return 1\nprint(f())\n',          # called
                     'def f(x):\n    return x\n',                      # needs an argument
                     'x = 1\nprint(x)\n',                              # no function
                     'def f(:\n'):                                     # does not parse: the run shows the error
            self.assertEqual(tir4b.with_call(code), (code, None))

    def test_the_last_uncalled_one(self):
        code = 'def a():\n    return 1\ndef b():\n    return a() + 1\n'
        self.assertEqual(tir4b.with_call(code)[1], 'b')

    def test_the_run_records_it(self):
        r = rollout(hint_calc=True)
        k = Kernel([(True, '37')])
        play(r, '\n123 \\times 456 =def f():\n    return 37\n```so</think>\n\n37<|im_end|>', k)
        self.assertTrue(k.ran[0].endswith('_r = f()\nif _r is not None:\n    print(_r)\n'))
        out = [e for e in r.events if e['kind'] == 'output'][0]
        self.assertEqual((out['auto_call'], out['code']), ('f', 'def f():\n    return 37\n'))


class LoopHint(unittest.TestCase):
    def test_the_loop_is_cut_to_one_round_and_a_program_opens(self):
        r = rollout(loop_hint=True)
        k = Kernel([(True, 'True')])
        # 4 + 68 x 15 = 1024 tokens: the loop is read at the end of the first chunk, and what the model writes next answers the hint
        play(r, 'set\n' + 'Wait, is it 5?\n' * 68 + 'print(5 == 5)\n```so 5</think>\n\n5<|im_end|>', k)
        text = r.text()
        self.assertTrue(text.startswith('set\nWait, is it 5?\n' + tir4b.LOOP_HINT))
        self.assertEqual(text.count('Wait, is it 5?'), 1)
        self.assertIn(tir4b.LOOP_HINT + 'print(5 == 5)\n```' + tir4b.output_block('True'), text)
        self.assertEqual(r.finish, 'stop')
        hint = [e for e in r.events if e['kind'] == 'loop hint'][0]
        self.assertGreater(hint['cut_tokens'], 1000)

    def test_a_loop_after_the_last_hint_is_stuck(self):
        r = rollout(loop_hint=True)
        r.loop_hints = tir4b.LOOP_HINTS
        play(r, 'Wait, is it 5?\n' * 400, Kernel([]))
        self.assertEqual([e['kind'] for e in r.events], [])
        self.assertEqual(r.finish, 'stuck')

    def test_loop_start_finds_the_first_round(self):
        tokens = [1, 2, 3] + [7, 8, 9] * 30
        self.assertEqual(tir4b.loop_start(tokens), 6)   # keeps [7, 8, 9] once


class FakeEngine:
    """vLLM's LLMEngine as drive() uses it: each request finishes on a later step, one at a time, so answers interleave."""
    def __init__(self, scripts):
        self.streams = {tuple(p): TOK.encode(s) for p, s in scripts}
        self.queue = []

    def add_request(self, rid, prompt, params):
        self.queue.append((rid, prompt['prompt_token_ids'], params))

    def step(self):
        if not self.queue:
            return []
        rid, ids, params = self.queue.pop(0)
        stream = next(s for p, s in self.streams.items() if tuple(ids[:len(p)]) == p)
        new = []
        while stream and len(new) < params.max_tokens:
            new.append(stream.pop(0))
            if new[-1] in params.stop_token_ids:
                break
        new = new or [SPECIAL['<|im_end|>']]
        out = types.SimpleNamespace(token_ids=new, stop_reason=None)
        return [types.SimpleNamespace(request_id=rid, finished=True, outputs=[out])]


class Drive(unittest.TestCase):
    def setUp(self):
        fake = types.ModuleType('vllm')
        fake.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
        fake.TokensPrompt = lambda prompt_token_ids: dict(prompt_token_ids=prompt_token_ids)
        self.saved = sys.modules.get('vllm')
        sys.modules['vllm'] = fake

    def tearDown(self):
        if self.saved is None:
            sys.modules.pop('vllm')
        else:
            sys.modules['vllm'] = self.saved

    def test_continuous_drive_writes_what_one_answer_alone_writes(self):
        scripts = ['a 2 + 3 = and 4 \\times 5 = done</think>\n\n25<|im_end|>',
                   'let me check ```python\nprint(7)\n```</think>\n\n7<|im_end|>',
                   'plain thinking</think>\n\nok<|im_end|>']
        arms = [dict(inline_calc=True), dict(execute=True, close_fit=True), dict()]
        alone = []
        for s, a in zip(scripts, arms):
            r = tir4b.Rollout(ROW, 0, [1], tir4b.Arm(**a), IDS, TOK, seed=1)
            play(r, s, Kernel([(True, '7')]))
            alone.append(r.text())
        rs = [tir4b.Rollout(ROW, 0, [1, 100 + i], tir4b.Arm(**a), IDS, TOK, seed=1) for i, a in enumerate(arms)]
        engine = FakeEngine([(r.prompt, s) for r, s in zip(rs, scripts)])
        saves = []
        tir4b.drive(types.SimpleNamespace(llm_engine=engine), rs, lambda: Kernel([(True, '7')]), live_most=2, on_round=lambda: saves.append(1))
        self.assertEqual([r.text() for r in rs], alone)
        self.assertTrue(all(r.finish == 'stop' for r in rs))
        self.assertTrue(saves)


class Training(unittest.TestCase):
    def test_labels_mask_prompt_and_output(self):
        self.assertEqual(labels_of([1, 2], [3, 4, 5, 6], [[1, 3]]), [-100, -100, 3, -100, -100, 6])

    def test_choose_prefers_a_fitting_run_then_no_call(self):
        def rec(i, s, right, tokens, events, runs=0, finish='stop'):
            return dict(id=i, sample=s, right=right, tokens=tokens, injected=0, events=events, runs=runs, finish=finish)
        fit = [dict(kind='forced open'), dict(kind='output', fits=True), dict(kind='close after a fitting result')]
        records = [rec('a', 0, True, 100, []), rec('a', 1, True, 300, fit, 1), rec('a', 2, False, 50, fit, 1),
                   rec('b', 0, True, 80, []), rec('b', 1, True, 60, [dict(kind='loop close')]),
                   rec('c', 0, False, 10, [])]
        hinted = [dict(kind='hint open'), dict(kind='output', fits=True), dict(kind='trust hint'), dict(kind='loop hint'),
                  dict(kind='output', fits=True)]
        records.append(rec('d', 0, True, 500, hinted, 2))
        chosen = tir4b.choose(records)
        self.assertEqual(chosen['d']['sample'], 0)   # Hint-RFT: the hints are taught as the model's own words
        self.assertEqual(chosen['a']['sample'], 1)   # the run that fitted, though longer
        self.assertEqual(chosen['b']['sample'], 0)   # a loop close is never taught
        self.assertNotIn('c', chosen)


if __name__ == '__main__':
    unittest.main()
