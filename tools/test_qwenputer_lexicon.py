import json
import struct
from types import SimpleNamespace
import unittest
from tools.qwenputer.lexicon import attach_lexicon
from tools.qwenputer.tokenizer import NativeInputError
from tools.qwenputer.format import HEADER, encode_sections, decode_sections


def fixture():
    rows=(b'a',b'x' * 16 + b'`',b'~' * 32,b' ' * 20 + b'\n')
    text=SimpleNamespace(n=len(rows),bytes=rows,binding_sha256='ab' * 32)
    sections={'identity':(32,bytes(160)+bytes.fromhex(text.binding_sha256)),
              'meta':(1,json.dumps({'schema':'STQPUT01-1.0','identities':{'tokenizer_sha256':text.binding_sha256}}).encode()),
              'text_rows':(16,b''.join(b[:16].ljust(16,b'\0') for b in rows)),
              'text_lens':(4,struct.pack('<4I',*map(len,rows)))}
    return sections,text


class FullTokenLexicon(unittest.TestCase):
    def test_complete_bytes_and_offsets_preserve_fence_and_line_boundaries(self):
        sections,text=fixture(); original=dict(sections)
        full=attach_lexicon(sections,text)
        self.assertIn('text_bytes',full)
        self.assertEqual(full['text_bytes'],(1,b''.join(text.bytes)))
        offsets=struct.unpack('<5I',full['text_offsets'][1])
        self.assertEqual(offsets,(0,1,18,50,71))
        self.assertEqual(json.loads(full['meta'][1])['schema'],'STQPUT01-1.1')
        self.assertEqual(sections,original)
        for key in sections.keys()-{'meta'}:
            self.assertEqual(full[key],sections[key])

    def test_binding_and_projected_rows_are_verified_before_upgrade(self):
        for mutate in ('binding','row','length'):
            with self.subTest(mutate=mutate):
                sections,text=fixture()
                if mutate=='binding':text.binding_sha256='cd' * 32
                elif mutate=='row':sections['text_rows']=(16,b'?' + sections['text_rows'][1][1:])
                else:sections['text_lens']=(4,struct.pack('<4I',2,17,32,21))
                with self.assertRaises(NativeInputError):attach_lexicon(sections,text)

    def test_cpu_data_budget_is_checked_without_large_allocations(self):
        sections,text=fixture()
        with self.assertRaisesRegex(NativeInputError,'lexicon_budget'):
            attach_lexicon(sections,text,max_bytes=8)

    def test_minor_one_roundtrip_keeps_legacy_default_version(self):
        sections,text=fixture();full=attach_lexicon(sections,text)
        upgraded=encode_sections(full,minor=1)
        self.assertEqual(HEADER.unpack_from(upgraded)[2],1)
        self.assertEqual(decode_sections(upgraded),full)
        legacy=encode_sections(sections)
        self.assertEqual(HEADER.unpack_from(legacy)[2],0)
        self.assertEqual(decode_sections(legacy),sections)
        for minor in (-1, 2, True):
            with self.subTest(minor=minor), self.assertRaises(NativeInputError):
                encode_sections(sections, minor=minor)


if __name__=='__main__':unittest.main()
