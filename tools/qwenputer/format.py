"""Canonical bounded STQPUT01 section container; never executes serialized code."""
import hashlib
import re
import struct
from tools.qwenputer.tokenizer import NativeInputError

HEADER = struct.Struct('<8sHHIIIQ32s')
SECTION = struct.Struct('<16sQQII32s8s')
MAX_BYTES = 256 * 1024 * 1024
MAX_SECTIONS = 32


def align64(n):
    return (n + 63) & ~63


def encode_sections(sections, minor=0):
    if type(minor) is not int or minor not in (0, 1):
        raise NativeInputError('artifact_version')
    if not 1 <= len(sections) <= MAX_SECTIONS:
        raise NativeInputError('section_count')
    header_bytes = align64(HEADER.size + len(sections) * SECTION.size)
    body = bytearray(header_bytes - HEADER.size)
    offset = header_bytes
    for index, (name, (stride, data)) in enumerate(sorted(sections.items())):
        if (not re.fullmatch(r'[a-z][a-z0-9_]{0,14}', name)
                or type(stride) is not int or not 1 <= stride <= 4096
                or not isinstance(data, bytes) or len(data) % stride):
            raise NativeInputError('section_input')
        if offset + align64(len(data)) > MAX_BYTES:
            raise NativeInputError('artifact_size')
        SECTION.pack_into(body, index * SECTION.size, name.encode(), offset, len(data), stride, 0,
                          hashlib.sha256(data).digest(), bytes(8))
        body.extend(data)
        body.extend(bytes(align64(len(data)) - len(data)))
        offset = HEADER.size + len(body)
    return HEADER.pack(b'STQPUT01', 1, minor, 0x01020304, len(sections), header_bytes,
                       HEADER.size + len(body), hashlib.sha256(body).digest()) + body


def decode_sections(blob, max_bytes=256 * 1024 * 1024):
    if not HEADER.size <= len(blob) <= min(max_bytes, MAX_BYTES):
        raise NativeInputError('artifact_size')
    magic, major, minor, endian, count, hbytes, total, digest = HEADER.unpack_from(blob)
    if magic != b'STQPUT01':
        raise NativeInputError('artifact_magic')
    if major != 1 or minor not in (0, 1):
        raise NativeInputError('artifact_version')
    if endian != 0x01020304:
        raise NativeInputError('artifact_endian')
    if total != len(blob) or total % 64:
        raise NativeInputError('artifact_size')
    if not 1 <= count <= MAX_SECTIONS or hbytes != align64(HEADER.size + count * SECTION.size):
        raise NativeInputError('section_count')
    if hbytes > total:
        raise NativeInputError('section_bounds')
    if hashlib.sha256(blob[HEADER.size:]).digest() != digest:
        raise NativeInputError('artifact_digest')
    if any(blob[HEADER.size + count * SECTION.size:hbytes]):
        raise NativeInputError('section_padding')
    result, end, previous = {}, hbytes, ''
    for i in range(count):
        raw_name, start, size, stride, flags, sha, reserved = SECTION.unpack_from(blob, HEADER.size + i * SECTION.size)
        try:
            name = raw_name.rstrip(b'\0').decode('ascii')
        except UnicodeError as error:
            raise NativeInputError('section_name') from error
        if (not re.fullmatch(r'[a-z][a-z0-9_]{0,14}', name) or name <= previous
                or raw_name != name.encode().ljust(16, b'\0')):
            raise NativeInputError('section_name')
        if start != end or start % 64 or size > total - min(start, total) or start > total:
            raise NativeInputError('section_bounds')
        if not 1 <= stride <= 4096 or size % stride:
            raise NativeInputError('section_stride')
        if flags:
            raise NativeInputError('section_flags')
        if any(reserved):
            raise NativeInputError('section_reserved')
        data = blob[start:start + size]
        if hashlib.sha256(data).digest() != sha:
            raise NativeInputError('section_digest')
        end = align64(start + size)
        if end > total or any(blob[start + size:end]):
            raise NativeInputError('section_padding')
        result[name] = (stride, bytes(data))
        previous = name
    if end != total:
        raise NativeInputError('section_bounds')
    return result
