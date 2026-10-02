"""Anthropic event order and streamed tool arguments for Claude Code."""
import threading
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, Event, ToolCall
from serve.server import ByteTokenizer, MockEngine, Service, anthropic_collect, anthropic_events


class FakeService:
    model = 'qwen3.8-flash-next'

    def model_for(self, req):   # upstream #297 aliases (merge 0.1.34, #100)
        return self.model
    def __init__(self, events): self.events = events
    def run(self, ids, thinking, tools, max_new, sampling, cancel):
        yield from self.events


class StreamTest(unittest.TestCase):
    def test_stop_sequence_spanning_token_deltas_is_excluded_and_reported(self):
        tokenizer = ByteTokenizer()
        template = ChatTemplate(Path(__file__).resolve().parents[2] / 'serve' / 'chat_template.jinja')
        service = Service(MockEngine(tokenizer, 'Hello<STOP>tail'), tokenizer, template)
        req = {'stop_sequences': ['<STOP>']}
        result = anthropic_collect(anthropic_events(
            service, req, [], False, None, 100, threading.Event()))
        self.assertEqual(result['content'], [{'type': 'text', 'text': 'Hello'}])
        self.assertEqual(result['stop_reason'], 'stop_sequence')
        self.assertEqual(result['stop_sequence'], '<STOP>')
        self.assertEqual(service.status['last_stop_reason'], 'stop_sequence')

    def test_unmatched_stop_prefix_is_flushed(self):
        tokenizer = ByteTokenizer()
        template = ChatTemplate(Path(__file__).resolve().parents[2] / 'serve' / 'chat_template.jinja')
        service = Service(MockEngine(tokenizer, 'Hello<STXtail'), tokenizer, template)
        req = {'stop_sequences': ['<STOP>']}
        events = list(anthropic_events(service, req, [], False, None, 100, threading.Event()))
        streamed_text = ''.join(data['delta']['text'] for name, data in events
                                if name == 'content_block_delta' and data['delta']['type'] == 'text_delta')
        self.assertEqual(streamed_text, 'Hello<STXtail')
        self.assertEqual(anthropic_collect(iter(events))['stop_reason'], 'end_turn')

    def test_thinking_closes_with_signature_before_text(self):
        service = FakeService([
            ('event', Event('reasoning', text='Let me think.')),
            ('event', Event('content', text='Answer.')),
            ('done', {'finish': 'stop', 'completion_tokens': 3}),
        ])
        events = [item for item in anthropic_events(service, {}, [1, 2, 3], True, None, 8,
                                                     threading.Event()) if item is not None]
        self.assertEqual(events[0][0], 'message_start')
        self.assertEqual(events[0][1]['message']['usage']['input_tokens'], 3)
        signatures = [(i, data) for i, (name, data) in enumerate(events)
                      if name == 'content_block_delta' and data['delta']['type'] == 'signature_delta']
        self.assertEqual(len(signatures), 1)
        thinking_stop = next(i for i, (name, data) in enumerate(events)
                             if name == 'content_block_stop' and data['index'] == 0)
        self.assertLess(signatures[0][0], thinking_stop)
        self.assertEqual(events[-1][0], 'message_stop')
        collected = anthropic_collect(iter(events))
        self.assertEqual(collected['content'][0]['thinking'], 'Let me think.')
        self.assertEqual(collected['content'][1]['text'], 'Answer.')

    def test_tool_arguments_stream_before_tool_use_stop(self):
        call = ToolCall('Write', {}, id='call_1')
        service = FakeService([
            ('event', Event('tool_start', call=call)),
            ('event', Event('tool_args', text='{"file":"a.txt"}', call=call)),
            ('event', Event('tool_call', call=call)),
            ('done', {'finish': 'stop', 'completion_tokens': 4}),
        ])
        events = [item for item in anthropic_events(service, {}, [1], False, None, 8,
                                                     threading.Event()) if item is not None]
        blocks = [data for name, data in events if name == 'content_block_start']
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]['content_block']['type'], 'tool_use')
        parts = [data['delta']['partial_json'] for name, data in events
                 if name == 'content_block_delta' and data['delta']['type'] == 'input_json_delta']
        self.assertEqual(parts, ['{"file":"a.txt"}'])
        stop = next(data for name, data in events if name == 'message_delta')
        self.assertEqual(stop['delta']['stop_reason'], 'tool_use')


if __name__ == '__main__':
    unittest.main()
