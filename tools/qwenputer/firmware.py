"""Strict data-only validation of the RV32 firmware memory map; no execution."""
from dataclasses import dataclass
import hashlib
import struct
from pathlib import Path

from tools.qwenputer.tokenizer import NativeInputError


def read_firmware(path: Path) -> bytes:
    try:
        with Path(path).open('rb') as file:
            data = file.read(4 * 1024 * 1024 + 1)
    except OSError as error:
        raise NativeInputError('elf_file') from error
    if not 52 <= len(data) <= 4 * 1024 * 1024:
        raise NativeInputError('elf_header')
    return data


@dataclass(frozen=True)
class Segment:
    address: int
    data: bytes
    memory_bytes: int
    flags: int
    region: str


@dataclass(frozen=True)
class Firmware:
    entry: int
    segments: tuple[Segment, ...]
    sha256: str


def parse_elf(data: bytes) -> Firmware:
    if not isinstance(data, bytes) or not 52 <= len(data) <= 4 * 1024 * 1024:
        raise NativeInputError('elf_header')
    fields = struct.unpack_from('<16sHHIIIIIHHHHHH', data)
    ident, kind, machine, version, entry, phoff, _, flags, ehsize, phsize, phnum, *_ = fields
    if ident[:7] != b'\x7fELF\x01\x01\x01' or ident[7] != 0:
        raise NativeInputError('elf_ident')
    if kind != 2:
        raise NativeInputError('elf_type')
    if machine != 243:
        raise NativeInputError('elf_machine')
    if version != 1 or ehsize != 52:
        raise NativeInputError('elf_header')
    if flags != 0:
        # The reviewed ROM decoder consumes fixed 32-bit RV32IM/Zba words.
        # RVC/RVE and floating-point ABI flags are not part of this contract.
        raise NativeInputError('elf_flags')
    if (phsize != 32 or not 1 <= phnum <= 32 or phoff < 52
            or phoff % 4 or phoff + phnum * phsize > len(data)):
        raise NativeInputError('elf_program_headers')
    regions = [('ram', 0, 0x8000), ('drom', 0x20000, 0x20000),
               ('rom', 0x40000, 0x40000)]
    segments = []
    for i in range(phnum):
        kind, offset, address, physical, filesz, memsz, flags, align = struct.unpack_from(
            '<8I', data, phoff + i * phsize)
        if kind != 1:
            continue
        if filesz > memsz:
            raise NativeInputError('elf_segment_size')
        if offset > len(data) or filesz > len(data) - offset:
            raise NativeInputError('elf_segment_file')
        if memsz == 0:
            continue
        region = next((name for name, base, size in regions
                       if base <= address < base + size and memsz <= base + size - address), None)
        if region is None:
            raise NativeInputError('elf_segment_region')
        if physical != address:
            raise NativeInputError('elf_physical_address')
        if (flags & ~7 or not flags & 4 or (region == 'rom' and flags != 5)
                or (region == 'drom' and flags != 4) or (region == 'ram' and flags & 1)):
            raise NativeInputError('elf_segment_permissions')
        if align not in (0, 1) and (align & (align - 1) or address % align != offset % align):
            raise NativeInputError('elf_segment_alignment')
        if any(address < old.address + old.memory_bytes and old.address < address + memsz
               for old in segments):
            raise NativeInputError('elf_segment_overlap')
        segments.append(Segment(address, data[offset:offset + filesz], memsz, flags, region))
    if (entry % 4 or not any(s.region == 'rom' and s.flags & 1
                            and s.address <= entry and entry + 4 <= s.address + len(s.data)
                            for s in segments)):
        raise NativeInputError('elf_entry')
    return Firmware(entry, tuple(segments), hashlib.sha256(data).hexdigest())
