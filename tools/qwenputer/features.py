"""Data-only lowering of the reviewed circuit's token embedding features."""
import re
from tools.qwenputer.tokenizer import NativeInputError

FAMILIES = (('V', 32768), ('F0', 32768), ('F1', 32768), ('F2', 4),
            ('J', 65536), ('R', 65536), ('BYTE', 512), ('FLT', 8),
            ('HEX', 16), ('QS', 16), ('QD', 128), ('BD', 256),
            ('NT', 1), ('ST0', 1), ('ST1', 1), ('SS0', 1), ('SS1', 1),
            ('HALT', 1), ('GO', 1), ('SC', 1))
TRACE_VOCAB = sum(size for _, size in FAMILIES)
TEXT_TOKEN = 1 << 24
ZERO, ONE, FAMILY, BIT, DIGIT, CODE = range(6)
IS_TEXT, MU, TAU, TCO, TCC, TEXT_BIT = range(6, 12)


def lower_feature(name):
    simple = {'one': ONE, 'is_text': IS_TEXT, 'mu': MU, 'tau': TAU,
              'isTCO': TCO, 'isTCC': TCC}
    if name in simple:
        return (simple[name], 0, 0)
    for family, (family_name, size) in enumerate(FAMILIES):
        if name == 'is' + family_name:
            return (FAMILY, family, 0)
        match = re.fullmatch(re.escape(family_name) + r'\.b(\d+)', name)
        if (match and family_name in ('V', 'F0', 'F1', 'F2', 'J', 'R', 'BYTE', 'QD', 'BD')
                and int(match[1]) < (size - 1).bit_length()):
            return (BIT, family, int(match[1]))
    match = re.fullmatch(r'digit==(\d)', name)
    if match and int(match[1]) < 8:
        return (DIGIT, int(match[1]), 0)
    match = re.fullmatch(r'code(\d+)', name)
    if match and int(match[1]) < 21:
        return (CODE, int(match[1]), 0)
    match = re.fullmatch(r'tb(\d+)\.b(\d)', name)
    if match and int(match[1]) < 16 and int(match[2]) < 8:
        return (TEXT_BIT, int(match[1]), int(match[2]))
    match = re.fullmatch(r'(rom\.b|drom\.b|x\.rs1|x\.rs2|x\.rd|quo|mem|tmem)\[(\d+)\]', name)
    if match and int(match[2]) < (4 if match[1] in ('rom.b', 'drom.b') else 32):
        return (ZERO, 0, 0)
    if re.fullmatch(r'bn[12]\.r[0-7]\[[0-3]\]', name):
        return (ZERO, 0, 0)
    raise NativeInputError('unknown_feature:' + name)


def feature_values(descriptor, kind, ids, text):
    import numpy as np
    ids = np.asarray(ids)
    if (type(kind) is not int or kind not in (0, 1) or ids.ndim != 1
            or ids.dtype.kind not in 'iu' or (ids < 0).any()
            or (ids >= (TRACE_VOCAB if kind == 0 else text.n)).any()):
        raise NativeInputError('feature_token_domain')
    ids = ids.astype(np.int64)
    opcode, p0, p1 = descriptor
    result = np.zeros(len(ids), dtype=np.float32)
    if opcode == ONE:
        result.fill(1)
    elif opcode == IS_TEXT:
        result.fill(kind)
    elif opcode == MU:
        result.fill(1 - kind)
    elif opcode == TAU:
        result.fill(2 ** -40 if kind else -1)
    elif opcode in (TCO, TCC) and kind:
        result[:] = ids == (text.TCO if opcode == TCO else text.TCC)
    elif opcode == TEXT_BIT and kind:
        raw = np.frombuffer(b''.join(text.row_bytes(int(q)) for q in ids), dtype=np.uint8)
        result[:] = (raw.reshape(-1, 16)[:, p0] >> p1) & 1
    elif opcode in (FAMILY, BIT) and kind == 0:
        base = sum(size for _, size in FAMILIES[:p0])
        hit = (ids >= base) & (ids < base + FAMILIES[p0][1])
        result[:] = hit if opcode == FAMILY else np.where(hit, ((ids - base) >> p1) & 1, 0)
    elif opcode == DIGIT and kind == 0:
        qs = sum(size for _, size in FAMILIES[:9])
        qd = qs + FAMILIES[9][1]
        result[:] = ((ids == qs + p0) | ((ids >= qd) & (ids < qd + 128)
                                                     & ((ids - qd) >> 4 == p0 - 1)))
    elif opcode == CODE:
        if kind:
            payload = np.array([text.pindex.get(int(q), -1) for q in ids], dtype=np.int64)
            code = np.full(len(ids), len(FAMILIES), dtype=np.int64) | (payload << 5)
            result[:] = np.where(payload >= 0, 128 * (2 * ((code >> p0) & 1) - 1), 0)
        else:
            base = 0
            for family, (_, size) in enumerate(FAMILIES):
                hit = (ids >= base) & (ids < base + size)
                code = family | ((ids[hit] - base) << 5)
                result[hit] = 128 * (2 * ((code >> p0) & 1) - 1)
                base += size
    elif opcode != ZERO and opcode not in (FAMILY, BIT, DIGIT, TEXT_BIT, TCO, TCC):
        raise NativeInputError('feature_opcode')
    return result


def check_features(circuit, text, progress=None):
    """Enumerate both complete supported domains against the original callbacks."""
    import hashlib
    import numpy as np
    features = [(0, lower_feature('one'))] + [
        (dim, lower_feature(circuit.dims[dim].name)) for dim in circuit.token_fns]
    digest = hashlib.sha256()
    domains = ((0, np.arange(TRACE_VOCAB, dtype=np.int64)),
               (1, np.arange(text.n, dtype=np.int64)))
    for index, (dim, descriptor) in enumerate(features):
        for kind, ids in domains:
            actual = feature_values(descriptor, kind, ids, text)
            if dim == 0:
                expected = np.ones(len(ids), dtype=np.float64)
            else:
                offset = TEXT_TOKEN if kind else 0
                expected = np.fromiter((circuit.feature_value(dim, int(t) + offset) for t in ids),
                                       dtype=np.float64, count=len(ids))
            bad = np.flatnonzero(actual.astype(np.float64) != expected)
            if len(bad) or not np.isfinite(expected).all():
                token = int(bad[0]) if len(bad) else -1
                raise NativeInputError(f'feature_mismatch:{dim}:{kind}:{token}')
            digest.update(actual.astype('<f4', copy=False).tobytes())
        if progress is not None and (index % 32 == 0 or index == len(features) - 1):
            progress(index + 1, len(features))
    return features, {'features': len(features), 'trace_ids': TRACE_VOCAB,
                      'text_ids': text.n, 'sha256': digest.hexdigest(),
                      'comparisons': len(features) * (TRACE_VOCAB + text.n)}
