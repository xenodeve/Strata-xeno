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
        self.assertEqual(v('$0.1 + 0.2 '), '0.3')                               # exact, not a float, and in the model's own decimals
        self.assertEqual(v('0.9 \\times 5 '), '4.5')                             # dm 6890: 9/2 here broke the model's decimal sum into a loop
        self.assertEqual(v('7.95922042 \\times 5 '), '39.7961021')
        self.assertEqual(v('1/3 + 1 '), '4/3')                                  # no terminating decimal: the reduced fraction
        self.assertEqual(v('0.5 - 0.75 '), '-0.25')                             # a negative decimal keeps its sign
        # long division by hand (dm div_big 5866, 6311, 8989): `21 \div 19 =` asks for a quotient digit, and 21/19 back, then 21/19 again for
        # `21/19 =`, looped until stuck where base finished the division and was right
        self.assertIsNone(v('21 \\div 19 '))                     # an integer quotient that does not come out: the model's own long division
        self.assertIsNone(v('15/13 '))                           # the value spelled as the expression: nothing new, and it loops
        self.assertIsNone(v('\\frac{-2142}{-7037} '))
        self.assertEqual(v('84 \\div 7 '), '12')                 # a division that comes out is still given
        self.assertEqual(v('1 \\div 8 '), '0.125')
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
        self.assertEqual(v('(3 + 4 '), '7')          # inside a bracket the text opened: `(3 + 4 = 7)` (tools.igc.mathexpr)
        self.assertIsNone(v('17 '))                  # no operator
        self.assertIsNone(v('1.2.3 + 1 '))           # a section number: F('1.2').F('3') killed the whole runner (gen-mt5)
        self.assertEqual(v('(1)(2) + 3 '), '5')      # implicit multiplication
        self.assertEqual(v('so 12,510,204 \\times 100 '), '1251020400')   # read whole: the old reader took 204 x 100 (dm 4617)
        self.assertEqual(v('so 511.276030 \\times 10 '), '5112.7603')   # never started inside a number (the old reader took 030 x 10)

    def test_the_rollout_puts_the_value_in(self):
        r = rollout(inline_calc=True)
        play(r, 'so 6 \\times 7 = and on</think>\n\n42<|im_end|>', Kernel([]))
        self.assertIn('6 \\times 7 = 42 (exact) and on', r.text())
        self.assertEqual(kinds(r), ['inline value'])
        self.assertEqual((r.events[0]['expr'], r.events[0]['value']), ('6 \\times 7', '42'))
        self.assertEqual(r.record(lambda *_: True)['inline_values'], 1)
        out = [s for s in r.spans if s[2] == 'inline value'][0]
        self.assertEqual(TOK.decode(r.out[out[0]:out[1]]), ' 42 (exact)')

    def test_plain_puts_in_only_the_number(self):
        # --inline-plain, as Qwenputer prints: ` (exact)` inside `$...$` left the model's math unclosed, and it copied the tag
        r = rollout(inline_calc=True, inline_plain=True)
        play(r, 'so $6 \\times 7 =$ and on</think>\n\n42<|im_end|>', Kernel([]))
        self.assertIn('$6 \\times 7 = 42$ and on', r.text())

    def test_off_by_default(self):
        r = rollout()
        play(r, 'so 6 \\times 7 = 41</think>\n\nx<|im_end|>', Kernel([]))
        self.assertEqual(r.events, [])

    def test_an_answer_with_no_thinking_gets_values_too(self):
        # --no-think, as Qwenputer measures arithmetic (enable_thinking=False): the reply starts as the answer
        from tools.igc.select_rows import right
        r = tir4b.Rollout(dict(id='a', suite='arith', answer='478128321'), 0, [1], tir4b.Arm(inline_calc=True), IDS, TOK, seed=1, thinking=False)
        self.assertEqual(r.phase, 'answer')
        play(r, '48213 \\times 9917 = so \\boxed{478128321}<|im_end|>', Kernel([]))
        self.assertIn('48213 \\times 9917 = 478128321 (exact) so', r.text())
        self.assertEqual(r.finish, 'stop')
        # the prompt closed the thinking, so the scorer, which reads after </think>, must see the whole reply
        self.assertTrue(r.record(right)['right'])

    def test_arith_rows_are_scored_by_their_box(self):
        from tools.igc.select_rows import right
        row = dict(id='a', suite='arith', answer='-3/7')
        self.assertTrue(right(row, '</think>\n\nso \\boxed{-3/7}'))
        self.assertFalse(right(row, '</think>\n\nso \\boxed{3/7}'))
        self.assertEqual(tir4b.prompt_text(dict(suite='arith', question='What is 2*3?')), 'What is 2*3?')

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

    def test_a_hint_that_gets_the_same_program_runs_it_again(self):
        # the repeat close is for a model that reruns its own program; a second hint at the same sum asks for it (gen-mt5: most answers closed)
        r = rollout(hint_calc=True, close_repeat=2)
        k = Kernel([(True, '56088'), (True, '56088')])
        gap = ' '.join(str(i) for i in range(300)) + '\n'   # past HINT_GAP tokens with no cycle, so the loop check stays quiet
        play(r, '\n123 \\times 456 =print(123*456)\n```' + gap + '123 \\times 456 =print(123*456)\n```done</think>\n\n56088<|im_end|>', k)
        self.assertEqual(len(k.ran), 2)
        self.assertNotIn('repeated program close', [e['kind'] for e in r.events])

    def test_the_trust_line_is_the_original_by_default_and_can_be_left_out_or_made_positive(self):
        # lb1: the original line, put after every hinted program, did not stop a 4B from re-checking a printed -530 for 12,000 tokens
        hinted = '\n123 \\times 456 =print(123*456)\n```hm</think>\n\n56088<|im_end|>'
        r = rollout(hint_calc=True)
        play(r, hinted, Kernel([(True, '56088')]))
        self.assertIn(tir4b.TRUST, r.text())
        r = rollout(hint_calc=True, trust='none')
        play(r, hinted, Kernel([(True, '56088')]))
        self.assertNotIn(tir4b.TRUST, r.text())
        self.assertNotIn('trust hint', [e['kind'] for e in r.events])
        self.assertIn('output', [e['kind'] for e in r.events])   # the program still runs and its output is still shown
        r = rollout(hint_calc=True, trust='positive')
        play(r, hinted, Kernel([(True, '56088')]))
        self.assertIn(tir4b.TRUST_LINES['positive'], r.text())
        self.assertNotIn(tir4b.TRUST, r.text())

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


class WholeHint(unittest.TestCase):
    """--whole-hint: the software computes the whole step, not the pieces the model would cut it into (dm div_big 5866: `21 \\div 19 =`
    of a long division got 21/19 and looped; the whole -2119381524 / 19 is one value)."""
    def test_the_whole_value_keeps_its_sign_and_its_fraction(self):
        v = tir4b.inline_value
        self.assertEqual(v('$-2119381524 \\div 19 ', whole=True), '-111546396')
        self.assertEqual(v('$-2142 \\div -7037 ', whole=True), '2142/7037')   # dm 8989's gold: an uneven quotient is the answer here
        self.assertEqual(v('so -3 \\times 0.2 ', whole=True), '-0.6')
        self.assertIsNone(v('$-2142 \\div -7037 '))                           # not asked for: still the model's long division

    def test_the_hint_opens_the_answer_and_its_first_value_is_whole(self):
        r = tir4b.Rollout(dict(id='a', suite='arith', answer='-111546396'), 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True, whole_hint=True),
                          IDS, TOK, seed=1, thinking=False)
        play(r, '$-2119381524 \\div 19 =$ so \\boxed{-111546396}<|im_end|>', Kernel([]))
        self.assertTrue(r.text().startswith(tir4b.WHOLE_HINT + '$-2119381524 \\div 19 = -111546396\n$ so'))
        self.assertEqual([e['kind'] for e in r.events], ['whole hint', 'inline value'])
        self.assertTrue(r.events[1]['whole'])
        self.assertTrue(r.record(__import__('tools.igc.select_rows', fromlist=['right']).right)['right'])

    def test_the_plain_hint_and_a_nested_expression(self):
        # dm mul_div_multiple_longer 1020: the question's own nested form, which v1 rewrote as \frac inside \frac and never computed
        r = tir4b.Rollout(dict(id='a', suite='arith', answer='319'), 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True, whole_hint=True,
                          whole_plain=True), IDS, TOK, seed=1, thinking=False)
        play(r, '(-1*-638*(-7)/(-7))/(((-62)/(-279))/(((-170)/3060)/(1/(-2)))) = so \\boxed{319}<|im_end|>', Kernel([]))
        self.assertTrue(r.text().startswith(tir4b.WHOLE_HINT_PLAIN + '(-1*-638*(-7)/(-7))/(((-62)/(-279))/(((-170)/3060)/(1/(-2)))) = 319\n so'))

    def test_the_whole_value_ends_its_line(self):
        # pc4-whole dm div_big 7815: `= 10097` left open, and the model wrote `.000...` after it until the guard stopped the answer
        r = tir4b.Rollout(dict(id='a', suite='arith', answer='10097'), 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True, whole_hint=True,
                          whole_plain=True), IDS, TOK, seed=1, thinking=False)
        play(r, '793836237 / 78621 =\\boxed{10097}<|im_end|>', Kernel([]))
        self.assertIn('793836237 / 78621 = 10097\n\\boxed{10097}', r.text())

    def whole_arm(self, answer, **more):
        return tir4b.Rollout(dict(id='a', suite='arith', answer=answer), 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True,
                             whole_hint=True, whole_plain=True, hand_off=True, **more), IDS, TOK, seed=1, thinking=False)

    def test_a_line_that_only_restates_the_value_closes_the_answer_with_it(self):
        # pc4-whole dm mul_div 1618: `= -18` written again on every line, never boxed; pc6 led it on and it answered 18 (P('-') 0.07).
        # The model's own restated value closes the answer, as it wrote it (developer, 2026-11-04)
        r = self.whole_arm('-18')
        play(r, '(360/27)/4*(-5)/((-20)/4)*-1*3*(-12)/180*-27 =\n= -18\nSo the answer is 18. \\boxed{18}<|im_end|>', Kernel([]))
        self.assertTrue(r.text().endswith('= -18\n\\boxed{-18}<|im_end|>'), r.text()[-60:])
        self.assertEqual(r.finish, 'stop')
        self.assertEqual([e['kind'] for e in r.events].count('hand-off close'), 1)
        self.assertTrue(r.record(__import__('tools.igc.select_rows', fromlist=['right']).right)['right'])

    def test_a_decimal_expansion_of_the_value_is_cut(self):
        # pc4-whole dm div_big 2289: `= -37/18329827` then the model wrote `= -0.0000020186299999...` to the length cap
        r = self.whole_arm('-37/18329827')
        play(r, '333 / -164968443 =\n= -0.00000201862999999999999999999999999999999999\\boxed{-37/18329827}<|im_end|>', Kernel([]))
        self.assertTrue(r.text().endswith('= -37/18329827\n\n\\boxed{-37/18329827}<|im_end|>'), r.text()[-80:])   # closed with the fraction
        self.assertNotIn('0.0000020186299999999', r.text())

    def test_a_rounded_decimal_of_a_given_fraction_closes_with_the_fraction(self):
        # pc8 4104: 1539439/2572 given, then `= 598.5377138413301`, boxed and scored wrong against the fraction (developer, 2026-11-04)
        r = self.whole_arm('1539439/2572')
        play(r, '-1539439 / -2572 =\n= 598.5377138413301\n\\boxed{598.5377138413301}<|im_end|>', Kernel([]))
        self.assertTrue(r.text().endswith('\\boxed{1539439/2572}<|im_end|>'), r.text()[-80:])
        self.assertNotIn('598.5377', r.text())
        self.assertEqual([e['kind'] for e in r.events].count('hand-off close exact'), 1)

    def test_the_question_s_own_computation_replaces_a_wrong_copy(self):
        # pc8 5180: 'What is 3534538042068 less than 1.712?' copied as 3534538042068 - 1.712; the question reads 1.712 - 3534538042068
        row = dict(id='a', suite='arith', answer='-3534538042066.288', question='What is 3534538042068 less than 1.712?')
        r = tir4b.Rollout(row, 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True, whole_hint=True, whole_plain=True, hand_off=True,
                          question_value=True), IDS, TOK, seed=1, thinking=False)
        play(r, '3534538042068 - 1.712 =\\boxed{-3534538042066.288}<|im_end|>', Kernel([]))
        self.assertIn('3534538042068 - 1.712 = -3534538042066.288\n', r.text())
        ev = [e for e in r.events if e['kind'] == 'question value']
        self.assertEqual(ev[0]['copied'], '3534538042066.288')

    def test_a_new_step_after_the_value_is_left_alone(self):
        r = self.whole_arm('95')
        play(r, '190 / 2 =\nso the total is 95 apples\n\\boxed{95}<|im_end|>', Kernel([]))
        self.assertNotIn(tir4b.HAND_OFF, r.text())

    def test_a_whole_line_with_no_equals_gets_its_value(self):
        # pc-whole dm mul_div 3350: the model wrote the whole expression on its own line, with no `=`, then went on in pieces
        r = tir4b.Rollout(dict(id='a', suite='arith', answer='228'), 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True, whole_hint=True,
                          whole_plain=True), IDS, TOK, seed=1, thinking=False)
        play(r, '((( -1 ) / ( -15 )) / ( ( -1 ) / 2 ))\nso \\boxed{-2/15}<|im_end|>', Kernel([]))
        self.assertIn('((( -1 ) / ( -15 )) / ( ( -1 ) / 2 ))\n= -2/15\nso', r.text())
        self.assertTrue(r.events[1]['whole'])

    def test_later_pieces_follow_the_usual_rules(self):
        r = tir4b.Rollout(ROW, 0, [1], tir4b.Arm(inline_calc=True, inline_plain=True, whole_hint=True), IDS, TOK, seed=1, thinking=False)
        play(r, '$-6 \\div 4 =$ then 21 \\div 19 = x<|im_end|>', Kernel([]))
        self.assertIn('$-6 \\div 4 = -1.5\n$', r.text())
        self.assertIn('21 \\div 19 = x', r.text())   # after the whole value, an uneven piece is left alone again


class CheckHand(unittest.TestCase):
    """--check-hand: what the model already worked out by hand is checked, and a wrong value is answered with the right one."""
    def test_a_wrong_hand_value_gets_a_correction(self):
        r = rollout(check_hand=True)
        play(r, 'so 144 - 144 - 572 - 168 + 96 = -744.\nthen 2 + 2 = 4 and</think>\n\nx<|im_end|>', Kernel([]))
        self.assertIn('= -744.\n' + tir4b.CORRECTION.format(expr='144 - 144 - 572 - 168 + 96', right='-644', wrote='-744'), r.text())
        ev = [e for e in r.events if e['kind'] == 'hand check']
        self.assertEqual([(e['wrote'], e['value']) for e in ev], [('-744', '-644')])

    def test_right_values_rounding_and_rewrites_pass(self):
        r = rollout(check_hand=True)
        # e: a long division's quotient digit with its remainder (dm mul_big 4267 would have been "corrected" to 6.5 six times)
        play(r, 'a 9 \\cdot 21 + 7 = 189 + 7 = 196.\nb 151.45 - 51.47 = 99.98\nc 2 / 3 = 0.67\nd 1 + 1 = 2\ne 13 \\div 2 = 6 remainder 1\n'
                '</think>\n\nx<|im_end|>', Kernel([]))
        self.assertEqual([e for e in r.events if e['kind'] == 'hand check'], [])

    def test_a_minus_before_the_expression_is_left_alone(self):
        r = rollout(check_hand=True)   # dm 3578: `-(25 * 130) = -3250` read as (25 * 130) and "corrected" to 3250
        play(r, 'so -(25 * 130) = -3250\nok</think>\n\nx<|im_end|>', Kernel([]))
        self.assertEqual([e for e in r.events if e['kind'] == 'hand check'], [])

    def test_an_uneven_quotient_is_corrected_in_the_places_written(self):
        r = rollout(check_hand=True)   # dm div 7815-like: a decimal written for 5 / 6.744908131, wrong in its 4th place
        play(r, 'so 5 / 6.744908131 = 0.7419\nok</think>\n\nx<|im_end|>', Kernel([]))
        self.assertIn(tir4b.CORRECTION.format(expr='5 / 6.744908131', right='0.7413', wrote='0.7419'), r.text())

    def test_checked_at_the_end_of_its_line_only(self):
        # the value is only known once the model has moved on: a check at the `=` itself would see no value yet
        r = rollout(check_hand=True)
        play(r, '123 \\times 2 = 247 so\nok</think>\n\nx<|im_end|>', Kernel([]))
        self.assertIn('247 so\n' + tir4b.CORRECTION.format(expr='123 \\times 2', right='246', wrote='247'), r.text())


class HttpDrive(unittest.TestCase):
    """drive_http: llama-server's /completion in place of vLLM, so the 5060 Ti on Windows runs the same rules."""
    def test_string_stops_and_the_stop_word_put_back(self):
        scripts = {(1, 200): 'so 6 \\times 7 = and on</think>\n\n42<|im_end|>', (1, 201): 'plain</think>\n\nok<|im_end|>'}
        cursor = dict.fromkeys(scripts, 0)

        def post(body):   # llama-server's /completion: the text up to a stop word, the word given apart, eos at <|im_end|>
            key = tuple(body['prompt'][:2])
            rest = scripts[key][cursor[key]:]
            cut, word = len(rest), ''
            for s in body['stop'] + ['<|im_end|>']:
                j = rest.find(s)
                if 0 <= j < cut:
                    cut, word = j, s
            cursor[key] += cut + len(word)
            if word == '<|im_end|>':
                return dict(content=rest[:cut], stop_type='eos', stopping_word='')
            return dict(content=rest[:cut], stop_type='word' if word else 'limit', stopping_word=word)
        rs = [tir4b.Rollout(ROW, 0, [1, 200], tir4b.Arm(inline_calc=True), IDS, TOK, seed=1),
              tir4b.Rollout(ROW, 0, [1, 201], tir4b.Arm(), IDS, TOK, seed=1)]
        tir4b.drive_http(post, TOK, IDS, rs, lambda: Kernel([]), live_most=2)
        self.assertEqual(rs[0].text(), 'so 6 \\times 7 = 42 (exact) and on</think>\n\n42<|im_end|>')
        self.assertEqual(rs[1].text(), 'plain</think>\n\nok<|im_end|>')
        self.assertTrue(all(r.finish == 'stop' for r in rs))

    def test_the_ids_the_server_returns_are_taken_as_they_are(self):
        # llama-server stopped on '\n' inside the token '\n\n' and returns that token: the text is not encoded again
        def post(body):
            self.assertTrue(body['return_tokens'])
            return dict(content='ab', stop_type='word', stopping_word='\n', tokens=[1000 + ord('a'), 1000 + ord('b'), 271]) \
                if not post.done else dict(content='', stop_type='eos', stopping_word='', tokens=[6])
        post.done = False
        r = tir4b.Rollout(ROW, 0, [1], tir4b.Arm(check_hand=True), IDS, TOK, seed=1, thinking=False)

        def once(body):
            reply = post(body)
            post.done = True
            return reply
        tir4b.drive_http(once, TOK, IDS, [r], lambda: Kernel([]), live_most=1)
        self.assertEqual(r.out, [1000 + ord('a'), 1000 + ord('b'), 271, 6])   # 271 kept, and no second end of turn

    def test_the_stop_words_of_a_request(self):
        r = tir4b.Rollout(ROW, 0, [1], tir4b.Arm(inline_calc=True, execute=True), IDS, TOK, seed=1)
        most, stops = r.request()
        self.assertEqual(sorted(tir4b.stop_words(stops, IDS, TOK)), sorted(['=', '```', '</think>']))


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
