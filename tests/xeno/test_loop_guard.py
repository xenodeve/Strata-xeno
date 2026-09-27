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


if __name__ == '__main__':
    unittest.main()
