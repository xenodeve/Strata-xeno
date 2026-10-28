"""Byte-exact Strata tokenizer binding for offline Native circuit compilation.

row_bytes is the compiler's 16-byte *projection*, not a capture validator.
The artifact must retain full byte lengths and the runtime must enforce the
same capture_payload checks before a live token can enter the capture buffer.
Long tokens outside a live capture remain valid language tokens.
"""
import hashlib
import json
from pathlib import Path

from tools.strata_tokenizer import Tokenizer, QWEN35_PATTERN, bytes_to_unicode


SPECIALS = ('<tool_call>', '</tool_call>', '<tool_response>', '</tool_response>',
            '<think>', '</think>')
MAX_FILE_BYTES = 64 * 1024 * 1024


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise NativeInputError('duplicate_json_key')
        result[key] = value
    return result


def _read(path):
    try:
        with path.open('rb') as file:
            data = file.read(MAX_FILE_BYTES + 1)
    except OSError as error:
        raise NativeInputError('tokenizer_file') from error
    if len(data) > MAX_FILE_BYTES:
        raise NativeInputError('tokenizer_file_size')
    return data


def _json(data):
    try:
        return json.loads(data.decode('utf-8'), object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError) as error:
        if isinstance(error, NativeInputError):
            raise
        raise NativeInputError('tokenizer_json') from error


class NativeInputError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class NativeText:
    @classmethod
    def from_pack(cls, directory: Path):
        directory = Path(directory)
        files = {name: _read(directory / name) for name in
                 ('tokenizer.json', 'vocab.json', 'merges.txt', 'token_type.json')}
        if (directory / 'chat_template.jinja').exists():
            files['chat_template.jinja'] = _read(directory / 'chat_template.jinja')
        descriptor = _json(files['tokenizer.json'])
        vocab = _json(files['vocab.json'])
        types = _json(files['token_type.json'])
        if not isinstance(descriptor, dict) or not isinstance(vocab, dict):
            raise NativeInputError('tokenizer_shape')
        if (descriptor.get('model') != 'gpt2' or descriptor.get('pre') != 'qwen35'
                or descriptor.get('pre_pattern') != QWEN35_PATTERN
                or descriptor.get('add_bos_token') is not False):
            raise NativeInputError('tokenizer_semantics')
        n = len(vocab)
        if (not 0 < n <= 1_048_576 or any(type(i) is not int for i in vocab.values())
                or set(vocab.values()) != set(range(n))):
            raise NativeInputError('vocab_ids')
        if type(descriptor.get('vocab_size')) is not int or descriptor['vocab_size'] != n:
            raise NativeInputError('vocab_size')
        if (not isinstance(types, list) or len(types) != n
                or any(type(i) is not int or not 1 <= i <= 6 for i in types)):
            raise NativeInputError('token_types')
        try:
            merges = files['merges.txt'].decode('utf-8').splitlines()
        except UnicodeError as error:
            raise NativeInputError('merges_encoding') from error
        if (type(descriptor.get('n_merges')) is not int or descriptor['n_merges'] != len(merges)
                or len(merges) != len(set(merges))):
            raise NativeInputError('merges_count')
        tokens = [''] * n
        for token, qid in vocab.items():
            if not token:
                raise NativeInputError('empty_token')
            tokens[qid] = token
        specials = descriptor.get('special_ids', {})
        if (not isinstance(specials, dict)
                or any(type(i) is not int or not 0 <= i < n for i in specials.values())):
            raise NativeInputError('special_ids')
        try:
            tok = Tokenizer(tokens, merges, types, pre=descriptor['pre'], special_ids=specials)
            raw = tuple(tok.token_bytes(i) for i in range(n))
        except (ValueError, KeyError, IndexError) as error:
            raise NativeInputError('tokenizer_bytes_or_merges') from error
        if any(name not in vocab or types[vocab[name]] != 4 for name in SPECIALS):
            raise NativeInputError('native_special_ids')
        alphabet = bytes_to_unicode()
        if any(alphabet[b] not in vocab for b in range(256)):
            raise NativeInputError('byte_vocabulary')
        text = cls()
        text.n, text.tok, text.bytes = n, tok, raw
        text.token_types = tuple(types)
        text.TCO, text.TCC = (vocab[s] for s in SPECIALS[:2])
        text.byte_token = {b: vocab[alphabet[b]] for b in range(256)}
        text.printable = list(text.byte_token.values()) + [vocab[s] for s in SPECIALS[2:]]
        text.pindex = {qid: i for i, qid in enumerate(text.printable)}
        text.NL = text.byte_token[10]
        text.binding_files = {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                              for name, data in sorted(files.items())}
        canonical = json.dumps(text.binding_files, sort_keys=True, separators=(',', ':')).encode()
        text.binding_sha256 = hashlib.sha256(canonical).hexdigest()
        return text

    def _id(self, qid):
        if type(qid) is not int or not 0 <= qid < self.n:
            raise NativeInputError('token_id')
        return qid

    def printable_index(self, qid):
        return self.pindex.get(self._id(qid))

    def row_bytes(self, qid):
        return self.bytes[self._id(qid)][:16].ljust(16, b'\0')

    def encode(self, text):
        return self.tok.encode(text, parse_special=True)

    def decode_bytes(self, qids):
        return b''.join(self.bytes[self._id(i)] for i in qids)

    def capture_payload(self, qids, max_tokens=80, max_bytes=255):
        if (type(max_tokens) is not int or not 1 <= max_tokens <= 80
                or type(max_bytes) is not int or not 1 <= max_bytes <= 255):
            raise NativeInputError('capture_limits')
        result = bytearray()
        for count, qid in enumerate(qids, 1):
            if count > max_tokens:
                raise NativeInputError('capture_token_count')
            qid = self._id(qid)
            if self.token_types[qid] != 1:
                raise NativeInputError('capture_reserved_token')
            row = self.bytes[qid]
            if len(row) > 16:
                raise NativeInputError('capture_token_width')
            if b'\0' in row:
                raise NativeInputError('capture_nul')
            if len(result) + len(row) > max_bytes:
                raise NativeInputError('capture_byte_count')
            result.extend(row)
        try:
            result.decode('utf-8', errors='strict')
        except UnicodeDecodeError as error:
            raise NativeInputError('capture_utf8') from error
        return bytes(result)
