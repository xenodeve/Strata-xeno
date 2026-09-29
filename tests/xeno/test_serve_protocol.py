"""Engine protocol events visible to the HTTP serving layer."""
import io
import queue
import threading
import unittest
from types import SimpleNamespace

from serve.server import StrataEngine


class ProtocolTest(unittest.TestCase):
    def test_prompt_progress_is_a_heartbeat(self):
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = SimpleNamespace(stdin=io.StringIO())
        engine.lines = queue.Queue()
        engine.can_stop = True
        engine.last = {}
        engine.lines.put('PP 10 5 150.0 33.3')
        engine.lines.put('T 123')
        engine.lines.put('DONE 1 10 150.0 20.0 length')
        events = list(engine.generate([7], 4, {}, threading.Event()))
        self.assertEqual(events, [None, 123])
        self.assertEqual(engine.last['prompt_tokens'], 10)


if __name__ == '__main__':
    unittest.main()
