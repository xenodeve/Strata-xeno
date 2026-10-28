"""Corrupt containers are rejected before any circuit allocation or execution."""
import hashlib
import struct
import unittest

from tools.qwenputer.format import encode_sections, decode_sections
from tools.qwenputer.tokenizer import NativeInputError


class FormatTests(unittest.TestCase):
    def error(self, blob, code, **kwargs):
        with self.assertRaises(NativeInputError) as caught:
            decode_sections(blob, **kwargs)
        self.assertEqual(caught.exception.code, code)

    def test_reproducible_binary_roundtrip_and_exact_float_bits(self):
        sections = {'coef': (4, struct.pack('<III', 0x80000000, 0x3eaaaaab, 0x00000001)),
                    'identity': (1, b'abc')}
        blob = encode_sections(sections)
        self.assertEqual(blob[:8], b'STQPUT01')
        self.assertEqual(blob, encode_sections(dict(reversed(list(sections.items())))))
        self.assertEqual(decode_sections(blob), sections)
        self.assertEqual(len(blob) % 64, 0)

    def test_bounds_versions_truncation_and_unknown_endianness(self):
        blob = encode_sections({'data': (4, b'12345678')})
        self.error(blob, 'artifact_size', max_bytes=32)
        self.error(blob[:-1], 'artifact_size')
        for offset, value, code in ((0, b'BADMAGIC', 'artifact_magic'),
                                    (8, struct.pack('<H', 2), 'artifact_version'),
                                    (12, struct.pack('<I', 0), 'artifact_endian')):
            altered = bytearray(blob)
            altered[offset:offset + len(value)] = value
            self.error(altered, code)

    def test_checksums_detect_payload_and_section_table_changes(self):
        blob = encode_sections({'data': (4, b'12345678')})
        altered = bytearray(blob)
        altered[-64] ^= 1
        self.error(altered, 'artifact_digest')
        # Re-sign the outer digest to test the independent section digest.
        altered[32:64] = hashlib.sha256(altered[64:]).digest()
        self.error(altered, 'section_digest')

    def test_overlap_stride_name_and_reserved_fields_refused(self):
        blob = encode_sections({'one': (4, b'abcd'), 'two': (4, b'efgh')})
        for offset, value, code in ((64 + 16, struct.pack('<Q', 0), 'section_bounds'),
                                    (64 + 32, struct.pack('<I', 3), 'section_stride'),
                                    (64 + 36, struct.pack('<I', 1), 'section_flags'),
                                    (64 + 72, b'\1', 'section_reserved')):
            altered = bytearray(blob)
            altered[offset:offset + len(value)] = value
            altered[32:64] = hashlib.sha256(altered[64:]).digest()
            self.error(altered, code)
        for sections in ({'x' * 17: (1, b'a')}, {'x': (0, b'a')}, {'x': (4, b'a')}):
            with self.subTest(sections=sections), self.assertRaises(NativeInputError):
                encode_sections(sections)


if __name__ == '__main__':
    unittest.main()
