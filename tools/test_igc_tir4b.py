"""tools/igc/tir4b.py and tir4b_train.py without a GPU: a scripted model drives the forced-block state machine through each path the
engine's rules take, and the training labels mask what the kernel printed. python tools/test_igc_tir4b.py"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.igc import tir4b
from tools.igc.tir4b_train import labels_of

SPECIAL = {'</think>': 5, '<|im_end|>': 6, '```python': 7, '```': 8, '\n\n': 271, '\n': 198}


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
        chosen = tir4b.choose(records)
        self.assertEqual(chosen['a']['sample'], 1)   # the run that fitted, though longer
        self.assertEqual(chosen['b']['sample'], 0)   # a loop close is never taught
        self.assertNotIn('c', chosen)


if __name__ == '__main__':
    unittest.main()
