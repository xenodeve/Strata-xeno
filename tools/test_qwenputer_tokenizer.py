"""Native byte binding and capture-safety checks; no upstream code or models."""
import json
import os
from pathlib import Path
import tempfile
import unittest

from tools.strata_tokenizer import bytes_to_unicode, QWEN35_PATTERN
from tools.qwenputer.tokenizer import NativeText, NativeInputError


SPECIALS = ['<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>',
            '<think>', '</think>', '<|endoftext|>', '<|im_start|>', '<|im_end|>']


def fixture(root):
    tokens = list(bytes_to_unicode()[b] for b in range(256)) + SPECIALS
    # A valid merged token whose raw row exceeds the circuit's 16-byte width.
    tokens += ['x' * 17]
    vocab = {s: i for i, s in enumerate(tokens)}
    types = [1] * 256 + [4] * 6 + [3] * 3 + [1]
    descriptor = {'model': 'gpt2', 'pre': 'qwen35', 'vocab_size': len(tokens),
                  'n_merges': 0, 'special_ids': {}, 'add_bos_token': False,
                  'pre_pattern': QWEN35_PATTERN}
    for name, data in [('vocab.json', vocab), ('token_type.json', types),
                       ('tokenizer.json', descriptor)]:
        (root / name).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    (root / 'merges.txt').write_text('', encoding='utf-8')
    return vocab


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.vocab = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        return NativeText.from_pack(self.root)

    def error(self, code, callback):
        with self.assertRaises(NativeInputError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)

    def test_complete_byte_vocabulary_and_special_payload_order(self):
        text = self.load()
        self.assertEqual(text.n, 266)
        self.assertEqual(text.printable[:256], list(range(256)))
        self.assertEqual(text.printable[256:], [258, 259, 260, 261])
        self.assertEqual((text.TCO, text.TCC, text.NL), (256, 257, 10))
        self.assertEqual(text.printable_index(120), 120)
        self.assertIsNone(text.printable_index(265))

    def test_every_byte_is_preserved_without_utf8_decoding(self):
        text = self.load()
        self.assertEqual(text.decode_bytes(range(256)), bytes(range(256)))
        self.assertEqual(text.bytes[233], b'\xe9')
        self.assertEqual(text.row_bytes(233), b'\xe9' + bytes(15))

    def test_thai_combining_marks_and_emoji_roundtrip(self):
        text = self.load()
        for s in ('ภาษาไทย', 'e\u0301 ไทย\n😀', '  17 * 23\n'):
            self.assertEqual(text.decode_bytes(text.encode(s)), s.encode('utf-8'))
        self.assertEqual(text.encode('<tool_call>'), [256])

    def test_feature_projection_does_not_erase_raw_length(self):
        text = self.load()
        self.assertEqual(text.bytes[265], b'x' * 17)
        self.assertEqual(text.row_bytes(265), b'x' * 16)
        self.error('capture_token_width', lambda: text.capture_payload([265]))

    def test_capture_limits_and_nul_are_explicit(self):
        text = self.load()
        self.assertEqual(text.capture_payload([49, 55, 43, 50]), b'17+2')
        self.error('capture_nul', lambda: text.capture_payload([49, 0, 50]))
        self.error('capture_token_count', lambda: text.capture_payload([49] * 81))
        self.error('capture_byte_count', lambda: text.capture_payload([49] * 4, max_bytes=3))
        self.error('capture_limits', lambda: text.capture_payload([49], max_tokens=81))
        self.error('capture_reserved_token', lambda: text.capture_payload([256]))

    def test_full_capture_rows_and_expression_payload_have_separate_budgets(self):
        text = self.load()
        # Capture includes XML scaffolding. Its total can exceed the firmware's
        # 255-byte expression payload, while every individual row stays bounded.
        text.bytes = text.bytes[:265] + (b'0123456789abcdef',)
        ids = [265] * 20
        try:
            actual = text.capture_rows(ids)
        except NativeInputError as error:
            self.fail('valid full capture refused by expression budget: ' + error.code)
        self.assertEqual(actual, b'0123456789abcdef' * 20)
        self.error('capture_byte_count', lambda: text.capture_payload(ids))
        self.error('capture_token_count', lambda: text.capture_rows([265] * 81))

    def test_invalid_ids_never_become_padding(self):
        text = self.load()
        for qid in (-1, 266, True, 1.0):
            with self.subTest(qid=qid):
                self.error('token_id', lambda: text.row_bytes(qid))
                self.error('token_id', lambda: text.decode_bytes([qid]))

    def test_capture_utf8_is_checked_after_complete_byte_concatenation(self):
        text = self.load()
        self.assertEqual(text.capture_payload([0xe0, 0xb8, 0xa0]), 'ภ'.encode('utf-8'))
        for invalid in ([255], [0xe0, 0xb8], [0xc0, 0xaf], [0xed, 0xa0, 0x80]):
            with self.subTest(invalid=invalid):
                self.error('capture_utf8', lambda: text.capture_payload(invalid))

    def test_descriptor_and_types_participate_in_binding(self):
        initial = self.load().binding_sha256
        self.assertEqual(initial, self.load().binding_sha256)
        (self.root / 'chat_template.jinja').write_text('example', encoding='utf-8')
        changed = self.load().binding_sha256
        self.assertNotEqual(initial, changed)
        types = json.loads((self.root / 'token_type.json').read_text())
        types[120] = 5
        (self.root / 'token_type.json').write_text(json.dumps(types))
        self.assertNotEqual(changed, self.load().binding_sha256)

    def test_malformed_binding_refuses_named_field(self):
        path = self.root / 'vocab.json'
        vocab = dict(self.vocab)
        vocab['x' * 17] = 999
        path.write_text(json.dumps(vocab))
        self.error('vocab_ids', self.load)

    def test_descriptor_count_and_pattern_are_checked(self):
        path = self.root / 'tokenizer.json'
        d = json.loads(path.read_text())
        d['vocab_size'] += 1
        path.write_text(json.dumps(d))
        self.error('vocab_size', self.load)
        d['vocab_size'] -= 1
        d['pre_pattern'] = '.*'
        path.write_text(json.dumps(d))
        self.error('tokenizer_semantics', self.load)


@unittest.skipUnless(os.environ.get('QWENPUTER_TOKENIZER'), 'real tokenizer path not supplied')
class GyroBindingTests(unittest.TestCase):
    def test_all_gyro_ids_and_known_specials(self):
        text = NativeText.from_pack(Path(os.environ['QWENPUTER_TOKENIZER']))
        self.assertEqual(text.n, 248320)
        self.assertEqual((text.TCO, text.TCC, text.NL), (248058, 248059, 198))
        self.assertEqual(text.printable[256:], [248066, 248067, 248068, 248069])
        self.assertEqual(text.bytes[2927], b'\xe0\xb8')
        self.assertEqual(text.bytes[55036], b' ' * 128)
        self.assertEqual(len(text.bytes), text.n)
        self.assertEqual(len(text.byte_token), 256)
        for qid in range(text.n):
            self.assertEqual(text.decode_bytes([qid]), text.bytes[qid])
        self.error_for_long_token(text)

    def error_for_long_token(self, text):
        with self.assertRaises(NativeInputError) as caught:
            text.capture_payload([55036])
        self.assertEqual(caught.exception.code, 'capture_token_width')


if __name__ == '__main__':
    unittest.main()
