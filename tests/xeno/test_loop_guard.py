"""Stop degenerate output while preserving normal text and tool payloads."""
import threading
import unittest

from serve.server import ByteTokenizer, Service, openai_chunks, openai_collect
from serve.loop_guard import LoopGuard


class RepeatingEngine:
    max_context = 4096
    def __init__(self, text): self.text = text
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for code in self.text.encode('utf-8')[:max_new]:
            if cancel.is_set(): return
            yield code


class LoopGuardTest(unittest.TestCase):
    def test_thai_tone_mark_micro_loop_stops_within_64_chars(self):
        guard = LoopGuard()
        self.assertTrue(guard.feed('สรุป ' + '่' * 80))
        self.assertIn('Thai-script', guard.reason)

    def test_repeated_output_stops_before_token_budget(self):
        engine = RepeatingEngine('A' * 800)
        service = Service(engine, ByteTokenizer(), None)
        events = list(service.run([], False, None, 1000, {}, threading.Event()))
        done = next(data for kind, data in events if kind == 'done')
        self.assertLess(done['completion_tokens'], 800)
        self.assertEqual(done.get('stop_detail'), 'loop')
        self.assertEqual(service.status.get('loops_stopped'), 1)
        self.assertEqual(service.status.get('last_stop_reason'), 'loop')
        service.engine = RepeatingEngine('A normal response.')
        list(service.run([], False, None, 1000, {}, threading.Event()))
        self.assertEqual(service.status.get('last_stop_reason'), 'length')

    def test_openai_response_marks_loop_in_timings(self):
        service = Service(RepeatingEngine('A' * 800), ByteTokenizer(), None)
        chunks = list(openai_chunks(service, {}, [], False, None, 1000, threading.Event()))
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'length')
        self.assertEqual(chunks[-1]['timings']['stop_reason'], 'loop')
        self.assertEqual(openai_collect(iter(chunks))['timings']['stop_reason'], 'loop')

    def test_normal_prose_is_not_stopped(self):
        text = ('The report explains the source, the test, and the measured result. '
                'It keeps each claim tied to its evidence. ') * 8
        service = Service(RepeatingEngine(text), ByteTokenizer(), None)
        events = list(service.run([], False, None, 1000, {}, threading.Event()))
        done = next(data for kind, data in events if kind == 'done')
        self.assertEqual(done['completion_tokens'], len(text.encode('utf-8')))
        self.assertEqual(service.status.get('loops_stopped', 0), 0)

    # #199: a thinking that re-plans the same file again and again - 12 cycles of ~8,400 characters, the same prose
    # lines recurring once per cycle - never trips the 4,096-character unit rule; the guard asks to close the thinking
    CYCLE = ("Let me write the code carefully. It'll be long, aim for a well-structured single file.\n"
             "Sky: use a large sphere with a gradient shader, or set the background colour by time of day.\n"
             "Camera presets: animate the camera to the target with a lerp each frame for smoothness.\n"
             "```js\n" + "function set(x, y, z, c) { V.set(key(x, y, z), c); }   // a voxel at x,y,z\n" * 3 + "```\n"
             "Walls: fill w x h x w with the wood colour, leaving openings for the windows and the doors.\n")

    def test_a_thinking_that_cycles_asks_to_close_in_its_third_cycle(self):
        guard = LoopGuard()
        for i in range(2):
            self.assertFalse(guard.feed(self.CYCLE, in_think=True))
            self.assertIsNone(guard.close_reason, i)
        self.assertFalse(guard.feed(self.CYCLE, in_think=True))   # never a stop: the server closes the thinking
        self.assertIn('thinking repeats', guard.close_reason)
        self.assertIsNone(guard.reason)

    def test_healthy_thinking_with_repeated_code_lines_is_not_closed(self):
        guard = LoopGuard()
        code = '```js\n' + 'mesh.castShadow = true; mesh.receiveShadow = true; scene.add(mesh);\n' * 6 + '```\n'
        for r in range(3):   # three drafts of different steps, each with the same long code lines
            prose = ''.join(f'Step {r}.{i}: place the {i}th lantern beside the path, then check its glow at night.\n'
                            for i in range(40))
            guard.feed(prose + code, in_think=True)
        self.assertIsNone(guard.close_reason)
        self.assertIsNone(guard.reason)

    def test_answer_text_never_asks_to_close(self):
        guard = LoopGuard()
        for _ in range(4):
            guard.feed(self.CYCLE, in_think=False)
        self.assertIsNone(guard.close_reason)


if __name__ == '__main__':
    unittest.main()
