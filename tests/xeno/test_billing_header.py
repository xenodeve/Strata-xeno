"""A changing Claude Code billing stamp must not change cached prompt tokens."""
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, anthropic_to_messages
from serve.server import ByteTokenizer


class BillingHeaderTest(unittest.TestCase):
    def test_system_billing_header_does_not_change_rendered_prompt(self):
        template = ChatTemplate(Path(__file__).resolve().parents[2] / 'serve' / 'chat_template.jinja')
        tokenizer = ByteTokenizer()
        for as_blocks in (False, True):
            prompts = []
            for stamp in ('a5145', 'e76b2'):
                system = ('x-anthropic-billing-header: cc_version=2.1.101.e51; '
                          f'cc_entrypoint=cli; cch={stamp};You are Claude Code.')
                request = {'system': [{'type': 'text', 'text': system}] if as_blocks else system,
                           'messages': [{'role': 'user', 'content': 'Hello'}]}
                messages, tools, kwargs = anthropic_to_messages(request)
                self.assertEqual(messages[0]['content'], 'You are Claude Code.')
                prompts.append(tokenizer.encode(template.render(messages, tools=tools, **kwargs),
                                                parse_special=True))
            self.assertEqual(prompts[0], prompts[1])

    def test_other_system_text_is_preserved(self):
        messages, _, _ = anthropic_to_messages({'system': 'You are Claude Code.', 'messages': []})
        self.assertEqual(messages[0]['content'], 'You are Claude Code.')

    def test_user_text_is_not_stripped(self):
        text = 'x-anthropic-billing-header: A literal example in a user message.'
        messages, _, _ = anthropic_to_messages({'messages': [{'role': 'user', 'content': text}]})
        self.assertEqual(messages[0]['content'], text)


if __name__ == '__main__':
    unittest.main()
