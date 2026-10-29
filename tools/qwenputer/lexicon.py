"""CPU-only full token bytes for trusted ownership parsing; no code execution."""
import json
import hashlib
from pathlib import Path
import re
import struct

from tools.qwenputer.tokenizer import NativeInputError


def attach_lexicon(sections, text, max_bytes=64 * 1024 * 1024):
    """Upgrade a verified v1.0 projection without changing any circuit records.

    This does not run the compiler, firmware or reference. The complete table is
    CPU ownership-parser data, deliberately excluded from the GPU weight upload.
    """
    if (type(max_bytes) is not int or not 0 <= max_bytes <= 64 * 1024 * 1024
            or type(text.n) is not int or not 0 < text.n <= 1_048_576
            or len(text.bytes) != text.n):
        raise NativeInputError('lexicon_budget')
    if not re.fullmatch('[a-f0-9]{64}', text.binding_sha256):
        raise NativeInputError('lexicon_binding')
    try:
        identity_stride, identity = sections['identity']
        row_stride, rows = sections['text_rows']
        length_stride, lengths = sections['text_lens']
        meta_stride, meta_bytes = sections['meta']
        meta = json.loads(meta_bytes)
    except (KeyError, ValueError, TypeError) as error:
        raise NativeInputError('lexicon_sections') from error
    if (not isinstance(meta, dict) or not isinstance(meta.get('identities'), dict)):
        raise NativeInputError('lexicon_sections')
    if (identity_stride != 32 or len(identity) != 192
            or identity[-32:] != bytes.fromhex(text.binding_sha256)
            or meta.get('identities', {}).get('tokenizer_sha256') != text.binding_sha256):
        raise NativeInputError('lexicon_binding')
    if (meta_stride != 1 or meta.get('schema') != 'STQPUT01-1.0'
            or 'text_offsets' in sections or 'text_bytes' in sections
            or row_stride != 16 or len(rows) != 16 * text.n
            or length_stride != 4 or len(lengths) != 4 * text.n):
        raise NativeInputError('lexicon_sections')
    # Bound all output allocations before joining rows or packing offsets.
    if 4 * (text.n + 1) > max_bytes:
        raise NativeInputError('lexicon_budget')
    total = 0
    offsets = [0]
    for i, raw in enumerate(text.bytes):
        if not isinstance(raw, bytes):
            raise NativeInputError('lexicon_row')
        total += len(raw)
        if total + 4 * (text.n + 1) > max_bytes:
            raise NativeInputError('lexicon_budget')
        if (struct.unpack_from('<I', lengths, 4 * i)[0] != len(raw)
                or rows[16 * i:16 * (i + 1)] != raw[:16].ljust(16, b'\0')):
            raise NativeInputError('lexicon_projection')
        offsets.append(total)
    result = dict(sections)
    meta['schema'] = 'STQPUT01-1.1'
    result['meta'] = (1, json.dumps(meta, sort_keys=True, separators=(',', ':')).encode())
    result['text_offsets'] = (4, struct.pack(f'<{len(offsets)}I', *offsets))
    result['text_bytes'] = (1, b''.join(text.bytes))
    return result


def upgrade(artifact, tokenizer_dir, output):
    """Create a new local-only data artifact; retain the original unchanged."""
    from tools.qwenputer.format import MAX_BYTES, decode_sections, encode_sections
    from tools.qwenputer.tokenizer import NativeText
    artifact, output = Path(artifact), Path(output)
    if output.exists():
        raise NativeInputError('lexicon_output_exists')
    with artifact.open('rb') as file:
        original = file.read(MAX_BYTES + 1)
    sections = decode_sections(original)
    full = attach_lexicon(sections, NativeText.from_pack(tokenizer_dir))
    blob = encode_sections(full, minor=1)
    if decode_sections(blob) != full:
        raise NativeInputError('lexicon_roundtrip')
    meta = json.loads(full['meta'][1])
    result = {'version': 1, 'kind': 'cpu-token-lexicon-data-upgrade',
              'format': 'STQPUT01-1.1', 'identities': meta['identities'],
              'original_artifact_sha256': hashlib.sha256(original).hexdigest(),
              'artifact_sha256': hashlib.sha256(blob).hexdigest(), 'artifact_bytes': len(blob),
              'cpu_lexicon_bytes': len(full['text_bytes'][1]) + len(full['text_offsets'][1]),
              'numeric_sections_byte_identical': all(full[k] == v for k, v in sections.items() if k != 'meta'),
              'compiler_or_firmware_executed': False,
              'redistribution': 'blocked-pending-firmware-linked-runtime-notices'}
    notices = {name: (artifact.parent / name).read_bytes()
               for name in ('LICENSE.compiler', 'UPSTREAM.json', 'NOTICE.txt')}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'native.qput').write_bytes(blob)
    (output / 'manifest.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    for name, data in notices.items():
        (output / name).write_bytes(data)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(upgrade(args.artifact, args.tokenizer_dir, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
