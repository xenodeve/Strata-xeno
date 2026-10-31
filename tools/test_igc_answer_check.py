from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from tools.igc.answer_check import load_env, prompt, read_all, read_one, triage


def record(id, answer, text):
    return dict(id=id, answer=answer, finish='stop', text=text, tokens=10, suite='gsm8k')


class Triage(unittest.TestCase):
    def rows(self, verdict, text='x</think>**5**', answer=5):
        return triage([record('a', answer, text)], {'a': dict(verdict=verdict, value=None, reason='r')})[0]

    def test_both_right_needs_no_reading(self):
        self.assertEqual(self.rows('same')['route'], 'agreed-right')

    def test_scorer_right_judge_wrong_is_read(self):
        self.assertEqual(self.rows('different')['route'], 'disagree')

    def test_scorer_wrong_judge_right_is_read(self):
        self.assertEqual(self.rows('same', answer=6)['route'], 'disagree')

    def test_both_wrong_asks_for_the_cause(self):
        self.assertEqual(self.rows('different', answer=6)['route'], 'agreed-wrong')

    def test_hedged_counts_as_wrong_for_the_judge(self):
        self.assertEqual(self.rows('hedged')['route'], 'disagree')
        self.assertEqual(self.rows('hedged', answer=6)['route'], 'agreed-wrong')

    def test_no_answer_never_goes_to_the_judge_and_needs_a_cause(self):
        row = triage([record('a', 5, 'thinking never closed')], {})[0]
        self.assertEqual(row['route'], 'agreed-wrong')

    def test_a_missing_verdict_is_read(self):
        self.assertEqual(triage([record('a', 5, 'x</think>**5**')], {})[0]['route'], 'disagree')


class Reader(unittest.TestCase):
    """A stand-in provider on this machine, speaking the OpenAI-compatible route."""

    def serve(self, replies):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                seen.append((self.path, self.headers['Authorization'], body))
                status, content = replies.pop(0) if replies else (200, '{"verdict": "same"}')
                data = json.dumps(dict(choices=[dict(message=dict(content=content))])).encode()
                self.send_response(status)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        env = dict(ANSWER_CHECK_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1/', ANSWER_CHECK_API_KEY='k',
                   ANSWER_CHECK_MODEL='m', ANSWER_CHECK_WORKERS='1')
        return env, seen

    def case(self):
        return dict(id='a', question='q', gold=5, reply='x</think>**5**')

    def test_the_route_the_key_and_the_settings(self):
        env, seen = self.serve([(200, 'Here: {"verdict": "hedged", "value": "5", "reason": "r"}')])
        self.assertEqual(read_one(env, self.case())['verdict'], 'hedged')
        path, key, body = seen[0]
        self.assertEqual((path, key, body['model'], body['temperature']), ('/v1/chat/completions', 'Bearer k', 'm', 0))

    def test_a_refusal_and_bad_text_are_tried_again(self):
        env, seen = self.serve([(429, ''), (200, 'no json')])
        self.assertEqual(read_one(env, self.case(), wait=0)['verdict'], 'same')
        self.assertEqual(len(seen), 3)

    def test_giving_up_leaves_no_verdict(self):
        env, _ = self.serve([(500, '')] * 2)
        self.assertIsNone(read_one(env, self.case(), tries=2, wait=0))

    def test_a_stopped_run_goes_on(self):
        env, seen = self.serve([])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'verdicts.json'
            path.write_text('{"a": {"verdict": "same"}}', encoding='utf-8')
            verdicts = read_all(env, [record('a', 5, 'x</think>5'), record('b', 5, 'x</think>5'), record('c', 5, 'open')],
                                dict(a='q', b='q', c='q'), path)
            self.assertEqual(sorted(verdicts), ['a', 'b'])
            self.assertEqual(sorted(json.loads(path.read_text(encoding='utf-8'))), ['a', 'b'])
        self.assertEqual(len(seen), 1)

    def test_the_env_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '.env'
            path.write_text('# note\nA=1\nB = x=y\nC=\n', encoding='utf-8')
            self.assertEqual(load_env(path), dict(A='1', B='x=y', C=''))


class Prompt(unittest.TestCase):
    def test_holds_no_thinking_and_no_arm(self):
        text = prompt(dict(question='How many?', gold=5, reply='SECRET THINKING</think>**5**'))
        self.assertNotIn('SECRET', text)
        self.assertIn('**5**', text)
        self.assertIn('How many?', text)


if __name__ == '__main__':
    unittest.main()
