"""Stop degenerate output while preserving normal text and tool payloads."""
import threading
import unittest

from serve.server import ByteTokenizer, Service
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
        self.assertEqual(service.status.get('loops_stopped'), 1)

    def test_normal_prose_is_not_stopped(self):
        text = ('The report explains the source, the test, and the measured result. '
                'It keeps each claim tied to its evidence. ') * 8
        service = Service(RepeatingEngine(text), ByteTokenizer(), None)
        events = list(service.run([], False, None, 1000, {}, threading.Event()))
        done = next(data for kind, data in events if kind == 'done')
        self.assertEqual(done['completion_tokens'], len(text.encode('utf-8')))
        self.assertEqual(service.status.get('loops_stopped', 0), 0)


if __name__ == '__main__':
    unittest.main()
