"""Anthropic event order and streamed tool arguments for Claude Code."""
import threading
import unittest

from serve.frontend import Event, ToolCall
from serve.server import anthropic_collect, anthropic_events


class FakeService:
    model = 'qwen3.8-flash-next'
    def __init__(self, events): self.events = events
    def run(self, ids, thinking, tools, max_new, sampling, cancel):
        yield from self.events


class StreamTest(unittest.TestCase):
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
