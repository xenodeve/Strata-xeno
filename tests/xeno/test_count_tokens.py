"""Anthropic count_tokens must report the actual rendered prompt size over HTTP."""
import base64
import json
from pathlib import Path
import tempfile
import urllib.error
import urllib.request
import unittest

from serve.frontend import ChatTemplate, anthropic_to_messages
from serve.server import IMAGE_PAD, ByteTokenizer, MockEngine, Service, serve


class ImageTokenizer(ByteTokenizer):
    image_token = 1_000_000

    def encode(self, text, parse_special=False):
        if not parse_special or IMAGE_PAD not in text:
            return super().encode(text, parse_special=parse_special)
        parts = text.split(IMAGE_PAD)
        ids = []
        for i, part in enumerate(parts):
            ids.extend(super().encode(part, parse_special=parse_special))
            if i + 1 < len(parts): ids.append(self.image_token)
        return ids


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

    def test_health_reports_dead_engine(self):
        class DeadProcess:
            def poll(self): return 1
        self.service.engine.proc = DeadProcess()
        with self.assertRaises(urllib.error.HTTPError) as result:
            urllib.request.urlopen(
                f'http://127.0.0.1:{self.http.server_address[1]}/health', timeout=5)
        self.assertEqual(result.exception.code, 503)

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

    def test_pdf_document_count_includes_extracted_text(self):
        pdf = (Path(__file__).with_name('fixtures') / 'sample.pdf').read_bytes()
        body = {'model': 'qwen3.8-flash-next', 'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Read this: '},
            {'type': 'document', 'title': 'sample.pdf', 'source': {
                'type': 'base64', 'media_type': 'application/pdf',
                'data': base64.b64encode(pdf).decode()}},
        ]}]}
        request = urllib.request.Request(
            f'http://127.0.0.1:{self.http.server_address[1]}/v1/messages/count_tokens',
            json.dumps(body).encode('utf-8'), {'content-type': 'application/json'})
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
            count = json.load(response)['input_tokens']
        messages, tools, kwargs = anthropic_to_messages(body)
        expected, _, _ = self.service.prepare(messages, tools, kwargs, None)
        self.assertEqual(count, len(expected))
        self.assertIn('Revenue is 42.', messages[0]['content'])

    def test_image_count_uses_expanded_tokens_and_removes_request_staging(self):
        class FakeVision:
            def __init__(self, directory):
                self.dir = directory
                self.source = directory / 'encoded.sve'
                self.source.write_bytes(b'encoded image')
            def encode(self, source):
                return self.source, 3

        tokenizer = ImageTokenizer()
        template = self.template
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

    def test_failed_image_staging_removes_partial_request_file(self):
        class BrokenVision:
            def __init__(self, directory): self.dir = directory
            def encode(self, source): return self.dir / 'missing.sve', 3

        body = {'messages': [{'role': 'user', 'content': [
            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'AA=='}}
        ]}]}
        messages, tools, kwargs = anthropic_to_messages(body)
        with tempfile.TemporaryDirectory() as temp:
            tokenizer = ImageTokenizer()
            service = Service(MockEngine(tokenizer, 'Hi.', max_context=8), tokenizer, self.template,
                              vision=BrokenVision(Path(temp)))
            with self.assertRaises(FileNotFoundError):
                service.prepare(messages, tools, kwargs, None)
            self.assertEqual(list(Path(temp).glob('req-*.sve')), [])


if __name__ == '__main__':
    unittest.main()
