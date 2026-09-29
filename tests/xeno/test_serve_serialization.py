"""A stopped request must drain its engine generator before the next request starts."""
import threading
import unittest

from serve.server import ByteTokenizer, Service


class ClosingEngine:
    max_context = 4096

    def __init__(self, stop_token):
        self.stop_token = stop_token
        self.calls = 0
        self.cleaned = threading.Event()
        self.second_started = threading.Event()

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        if self.calls == 1:
            try:
                yield self.stop_token
            finally:
                self.cleaned.set()
        else:
            self.second_started.set()
            yield self.stop_token


class SerializationTest(unittest.TestCase):
    def test_stopped_generator_is_closed_while_holding_request_lock(self):
        tokenizer = ByteTokenizer()
        stop = tokenizer.encode('<|im_end|>', parse_special=True)[0]
        engine = ClosingEngine(stop)
        service = Service(engine, tokenizer, None)
        first = service.run([], False, None, 4, {}, threading.Event())
        while next(first)[0] != 'done':
            pass
        second = threading.Thread(target=lambda: list(service.run([], False, None, 4, {}, threading.Event())))
        second.start()
        try:
            self.assertTrue(engine.second_started.wait(2), 'second request never started')
            self.assertTrue(engine.cleaned.is_set(), 'prior engine generator still owns cleanup after lock release')
        finally:
            first.close()
            second.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
