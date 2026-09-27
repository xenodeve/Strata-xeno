"""Anthropic count_tokens must report the actual rendered prompt size over HTTP."""
import json
from pathlib import Path
import tempfile
import urllib.request
import unittest

from serve.frontend import ChatTemplate, anthropic_to_messages
from serve.server import IMAGE_PAD, ByteTokenizer, MockEngine, Service, serve
from tools.strata_tokenizer import Tokenizer


class CountTokensTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.tokenizer = ByteTokenizer()
        self.template = ChatTemplate(root / 'serve' / 'chat_template.jinja')
        self.service = Service(MockEngine(self.tokenizer, 'Hi.', max_context=8), self.tokenizer,
                               self.template)
        self.http = serve(self.service, port=0)

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()

    def test_count_exceeds_context_without_rejecting_request(self):
        body = {'model': 'qwen3.8-flash-next', 'messages': [
            {'role': 'user', 'content': 'Please count this long prompt before deciding to compact it.'}]}
        data = json.dumps(body).encode('utf-8')
        request = urllib.request.Request(
            f'http://127.0.0.1:{self.http.server_address[1]}/v1/messages/count_tokens',
            data, {'content-type': 'application/json'})
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            payload = json.load(response)
        self.assertGreater(payload['input_tokens'], self.service.engine.max_context)
        self.assertEqual(payload['input_tokens'], len(self.tokenizer.encode(
            self.template.render(messages=body['messages']), parse_special=True)))

    def test_image_count_uses_expanded_tokens_and_removes_request_staging(self):
        class FakeVision:
            def __init__(self, directory):
                self.dir = directory
                self.source = directory / 'encoded.sve'
                self.source.write_bytes(b'encoded image')
            def encode(self, source):
                return self.source, 3

        token_dir = Path(r'D:\Github\Strata\packs\q2_0\tokenizer')
        vocab = json.loads((token_dir / 'vocab.json').read_text(encoding='utf-8'))
        tokens = [None] * len(vocab)
        for token, index in vocab.items(): tokens[index] = token
        tokenizer = Tokenizer(tokens, (token_dir / 'merges.txt').read_text(encoding='utf-8').split('\n'),
                              json.loads((token_dir / 'token_type.json').read_text()))
        template = ChatTemplate(token_dir / 'chat_template.jinja')
        body = {'model': 'qwen3.8-flash-next', 'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'What is this?'},
            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'AA=='}}
        ]}]}
        messages, tools, kwargs = anthropic_to_messages(body)
        rendered = tokenizer.encode(template.render(messages, tools=tools, **kwargs), parse_special=True)
        self.assertEqual(rendered.count(tokenizer.encode(IMAGE_PAD, parse_special=True)[0]), 1)
        with tempfile.TemporaryDirectory() as temp:
            service = Service(MockEngine(tokenizer, 'Hi.', max_context=8), tokenizer, template,
                              vision=FakeVision(Path(temp)))
            http = serve(service, port=0)
            try:
                request = urllib.request.Request(
                    f'http://127.0.0.1:{http.server_address[1]}/v1/messages/count_tokens',
                    json.dumps(body).encode(), {'content-type': 'application/json'})
                with urllib.request.urlopen(request, timeout=5) as response:
                    payload = json.load(response)
                self.assertEqual(payload['input_tokens'], len(rendered) + 2)
                self.assertEqual(list(Path(temp).glob('req-*.sve')), [])
            finally:
                http.shutdown(); http.server_close()


if __name__ == '__main__':
    unittest.main()
