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
    PLAN = ("Let me write the code carefully. It'll be long, aim for a well-structured single file.\n"
            "Sky: use a large sphere with a gradient shader, or set the background colour by time of day.\n"
            "Camera presets: animate the camera to the target with a lerp each frame for smoothness.\n"
            "```js\n" + "function set(x, y, z, c) { V.set(key(x, y, z), c); }   // a voxel at x,y,z\n" * 3 + "```\n"
            "Walls: fill w x h x w with the wood colour, leaving openings for the windows and the doors.\n")

    @staticmethod
    def filler(r):   # a cycle's own drafting, different every time: ~2,300 characters between the plan's repeats
        parts = ['railing', 'lantern', 'bridge', 'koi pond', 'stone path', 'maple', 'bamboo', 'torii', 'moss',
                 'gravel', 'bell', 'eave', 'finial', 'shrine', 'fence', 'willow', 'plum', 'well', 'stair', 'gate']
        moods = ['darker', 'warmer', 'taller', 'wider', 'softer', 'brighter', 'rougher', 'thinner']
        return ''.join(f'Draft {r}.{i}: the {parts[(i * 7 + r) % 20]} near tier {i % 5} looks {moods[(i + r) % 8]}, '
                       f'so shift it by {(i * 13 + r * 5) % 9} voxels toward the {parts[(i * 3 + r) % 20]}.\n'
                       for i in range(30))

    def cycles(self, plan, n):
        return [plan + self.filler(r) for r in range(n)]

    def test_a_thinking_that_cycles_asks_to_close_in_its_third_cycle(self):
        guard = LoopGuard()
        c = self.cycles(self.PLAN, 3)
        for i in range(2):
            self.assertFalse(guard.feed(c[i], in_think=True))
            self.assertIsNone(guard.close_reason, i)
        self.assertFalse(guard.feed(c[2], in_think=True))   # never a stop: the server closes the thinking
        self.assertIn('thinking repeats', guard.close_reason)
        self.assertIsNone(guard.reason)

    def test_three_cycling_lines_are_not_enough(self):
        guard = LoopGuard()
        plan = ''.join(self.PLAN.splitlines(keepends=True)[:3])   # three prose lines, no fourth
        for c in self.cycles(plan, 4):
            guard.feed(c, in_think=True)
        self.assertIsNone(guard.close_reason)

    def test_a_checklist_rechecked_in_a_row_is_not_closed(self):
        # code review of #199: three close passes over one checklist are not a cycle of the whole plan
        guard = LoopGuard()
        prose = ''.join(l for l in self.PLAN.splitlines(keepends=True) if '`' not in l and '{' not in l)
        guard.feed(prose * 3, in_think=True)
        self.assertIsNone(guard.close_reason)

    def test_unfenced_code_does_not_count(self):
        guard = LoopGuard()
        code = ('mesh.castShadow = true; mesh.receiveShadow = true; scene.add(mesh);\n'
                'const roof = buildRoof(tier, width + 2, height, colors.roofTile);\n'
                'for (let i = 0; i < tiers; i++) placeTier(i, baseWidth - i * 2);\n'
                'renderer.shadowMap.enabled = true; renderer.toneMapping = 4;\n')
        for c in self.cycles(code, 4):
            guard.feed(c, in_think=True)
        self.assertIsNone(guard.close_reason)

    def test_a_fence_opened_mid_line_still_closes_at_its_end(self):
        guard = LoopGuard()
        plan = self.PLAN.replace("```js\n", "Here is the helper: ```js\n")
        for c in self.cycles(plan, 3):
            guard.feed(c, in_think=True)
        self.assertIsNotNone(guard.close_reason)

    def test_a_thai_plan_that_cycles_is_closed(self):
        guard = LoopGuard()
        plan = ('ต้องเขียนโค้ดอย่างระมัดระวัง ทำเป็นไฟล์เดียวที่มีโครงสร้างดีและอ่านง่ายสำหรับผู้ใช้\n'
                'ท้องฟ้าใช้ทรงกลมขนาดใหญ่ไล่เฉดสีตามเวลาของวันและมีหมอกบาง ๆ ตอนเช้า\n'
                'กล้องมีมุมมองที่ตั้งไว้ล่วงหน้าและเคลื่อนที่อย่างนุ่มนวลไปยังเป้าหมายทุกเฟรม\n'
                'ผนังเติมด้วยสีไม้ เว้นช่องสำหรับหน้าต่างและประตูของแต่ละชั้นของเจดีย์\n')
        for c in self.cycles(plan, 3):
            guard.feed(c, in_think=True)
        self.assertIsNotNone(guard.close_reason)

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
            guard.feed(self.PLAN + self.filler(0), in_think=False)
        self.assertIsNone(guard.close_reason)


if __name__ == '__main__':
    unittest.main()
