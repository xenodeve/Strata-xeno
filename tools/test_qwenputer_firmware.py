"""ELF bounds/provenance validation only; never executes RISC-V instructions."""
import os
from pathlib import Path
import struct
import unittest
import tempfile

from tools.qwenputer.firmware import parse_elf, read_firmware
from tools.qwenputer.tokenizer import NativeInputError


def image(extra=None):
    segments = [(0x100, 0x40000, 4, 4, 5, 4)]
    if extra is not None:
        segments.append(extra)
    data = bytearray(512)
    struct.pack_into('<16sHHIIIIIHHHHHH', data, 0,
                     b'\x7fELF\x01\x01\x01' + bytes(9), 2, 243, 1,
                     0x40000, 52, 0, 0, 52, 32, len(segments), 0, 0, 0)
    for i, (offset, address, filesz, memsz, flags, align) in enumerate(segments):
        struct.pack_into('<8I', data, 52 + 32 * i, 1, offset, address,
                         address, filesz, memsz, flags, align)
    data[0x100:0x104] = b'\x13\0\0\0'
    return data


class FirmwareTests(unittest.TestCase):
    def error(self, code, data):
        with self.assertRaises(NativeInputError) as caught:
            parse_elf(bytes(data))
        self.assertEqual(caught.exception.code, code)

    def test_valid_image_retains_zero_fill_size_without_expanding(self):
        parsed = parse_elf(bytes(image((0x104, 0x100, 4, 8, 6, 4))))
        self.assertEqual(parsed.entry, 0x40000)
        self.assertEqual([s.region for s in parsed.segments], ['rom', 'ram'])
        self.assertEqual(parsed.segments[1].memory_bytes, 8)
        self.assertEqual(len(parsed.segments[1].data), 4)
        self.assertEqual(len(parsed.sha256), 64)

    def test_ident_machine_and_byte_order_are_checked(self):
        for offset, value, code in [(0, 0, 'elf_ident'), (4, 2, 'elf_ident'),
                                    (5, 2, 'elf_ident'), (18, 62, 'elf_machine')]:
            with self.subTest(offset=offset):
                data = image(); data[offset] = value
                self.error(code, data)

    def test_short_and_unbounded_headers_refuse(self):
        self.error('elf_header', b'\x7fELF')
        for field, value, code in [(28, 500, 'elf_program_headers'),
                                    (42, 31, 'elf_program_headers'),
                                    (44, 65535, 'elf_program_headers')]:
            data = image()
            struct.pack_into('<I' if field == 28 else '<H', data, field, value)
            self.error(code, data)

    def test_filesz_memsz_and_offset_cannot_wrap_or_truncate(self):
        for field, value, code in [(52 + 20, 3, 'elf_segment_size'),
                                    (52 + 4, 0xfffffff0, 'elf_segment_file'),
                                    (52 + 20, 0xffffffff, 'elf_segment_region')]:
            data = image(); struct.pack_into('<I', data, field, value)
            self.error(code, data)

    def test_entry_is_aligned_and_in_initialized_executable_rom(self):
        for entry in (0x40001, 0x40004, 0x100):
            data = image(); struct.pack_into('<I', data, 24, entry)
            self.error('elf_entry', data)

    def test_overlaps_permissions_and_alignment_refuse(self):
        self.error('elf_segment_overlap', image((0x104, 0x40000, 4, 4, 5, 4)))
        data = image(); struct.pack_into('<I', data, 52 + 24, 7)
        self.error('elf_segment_permissions', data)
        data = image(); struct.pack_into('<I', data, 52 + 28, 3)
        self.error('elf_segment_alignment', data)

    def test_firmware_replacement_reread_is_still_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'firmware.elf'
            path.write_bytes(image())
            self.assertEqual(read_firmware(path), bytes(image()))
            path.write_bytes(bytes(4 * 1024 * 1024 + 1))
            with self.assertRaises(NativeInputError) as caught:
                read_firmware(path)
            self.assertEqual(caught.exception.code, 'elf_header')


@unittest.skipUnless(os.environ.get('QWENPUTER_FIRMWARE'), 'firmware data path not supplied')
class PinnedFirmwareTests(unittest.TestCase):
    def test_pinned_elf_can_be_validated_without_execution(self):
        data = Path(os.environ['QWENPUTER_FIRMWARE']).read_bytes()
        fw = parse_elf(data)
        self.assertTrue(fw.segments)
        self.assertEqual(fw.entry & 3, 0)
        self.assertTrue(any(s.region == 'rom' for s in fw.segments))


if __name__ == '__main__':
    unittest.main()
