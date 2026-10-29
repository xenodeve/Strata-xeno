"""Identity-bound offline input corpus adapter; no acceptance regeneration."""
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
from pathlib import Path
import stat
from tools.igc.matched_corpus import digest, PATTERNS

MATCHED = 'matched-pair-current-readout-development'
EXPANDED = 'expanded-native-embedding-input-development'
MANIFESTS = {
    MATCHED: ('88d4b39a3c2542c644f5eac4647e90f90780eb7bb1dcb3a9d6f5c324edbe25ff', 1536),
    EXPANDED: ('4fcff51db4712d97c92845529e8617e2240badf3da8a48280e4798e3a2a63cb3', 10624),
}
ROW_KEYS = {'id', 'group', 'pair_group', 'split', 'family', 'language', 'intent',
            'lhs', 'rhs', 'operation', 'ready_op', 'expression', 'prompt', 'prompt_sha256'}


def validate_structure(value):
    keys = {'version', 'kind', 'seed', 'rows', 'rows_sha256', 'scope', 'sampling', 'heldout', 'source_pool'}
    if (not isinstance(value, dict) or set(value) != keys or type(value['version']) is not int
            or value['version'] != 1 or value['kind'] != MATCHED or value['seed'] != 20261108
            or not isinstance(value['rows'], list) or len(value['rows']) != 1536):
        raise ValueError('matched manifest schema/count')
    groups, pairs, coverage = defaultdict(set), defaultdict(set), Counter()
    identifiers = set()
    for row in value['rows']:
        if not isinstance(row, dict) or set(row) != ROW_KEYS:
            raise ValueError('matched row keys: supervision cannot become native inputs')
        a, b, op = row['lhs'], row['rhs'], row['operation']
        split, family, language, intent = (row[k] for k in ('split', 'family', 'language', 'intent'))
        if (any(type(x) is not int or abs(x) > 99 for x in (a, b)) or type(op) is not int or op not in (1, 2, 3)
                or split not in ('train', 'development') or family not in ('seen', 'new')
                or language not in ('en', 'th') or intent not in ('compute', 'quote-copy')
                or (split == 'train' and family != 'seen')):
            raise ValueError('matched row domain/split')
        expression = f'({a}) '+{1: '+', 2: '-', 3: '*'}[op]+f' ({b})'
        group, pair_group = digest([a, b, op])[:20], digest([a, b])[:20]
        prompt = PATTERNS[family, language, intent].format(expr=expression)
        identifier = f'{group}-{family}-{language}-{intent}'
        if (row['group'] != group or row['pair_group'] != pair_group or row['id'] != identifier
                or identifier in identifiers or row['expression'] != expression or row['prompt'] != prompt
                or type(row['ready_op']) is not int or row['ready_op'] != (op if intent == 'compute' else 0)
                or row['prompt_sha256'] != hashlib.sha256(prompt.encode('utf-8')).hexdigest()):
            raise ValueError('matched group/intent/prompt identity')
        identifiers.add(identifier)
        groups[split, group].add((family, language, intent))
        pairs[split].add((a, b))
        if (family, language, intent) == ('seen', 'en', 'compute'):
            coverage[split, len(str(abs(a))), len(str(abs(b))), a < 0, b < 0, op] += 1
    if pairs['train'] & pairs['development']:
        raise ValueError('matched train/development pair overlap')
    for (split, _), conditions in groups.items():
        families = ('seen',) if split == 'train' else ('seen', 'new')
        if conditions != set(product(families, ('en', 'th'), ('compute', 'quote-copy'))):
            raise ValueError('matched incomplete paired conditions')
    for split, expected in [('train', 4), ('development', 2)]:
        for key in product((1, 2), (1, 2), (False, True), (False, True), (1, 2, 3)):
            if coverage[(split, *key)] != expected:
                raise ValueError('matched length/sign/op coverage')
        for side in ('lhs', 'rhs'):
            observed = {r[side] for r in value['rows'] if r['split'] == split}
            if not {-8, -3, 3, 8} <= observed:
                raise ValueError('matched critical signed-slot coverage')
    if digest(value['rows']) != value['rows_sha256']:
        raise ValueError('matched row digest')
    return 1536


def validate_manifest(value):
    if not isinstance(value, dict) or value.get('kind') not in MANIFESTS:
        raise ValueError('unrecognized input manifest')
    if value['kind'] == MATCHED:
        validate_structure(value)
    expected, count = MANIFESTS[value['kind']]
    if digest(value) != expected:
        raise ValueError('input manifest identity')
    return count


def native_prompt_rows(rows):
    # Projection only, after the caller binds the complete corpus identity.
    # No labels, operation/domain fields or group metadata enter native inputs.
    return [dict(id=row['id'], prompt=row['prompt'], expression=row['expression']) for row in rows]


def cache_shapes(count):
    if type(count) is not int or count not in (1536, 10624):
        raise ValueError('input cache count')
    return (count, 80, 4, 2560), (count, 80)


def load_manifest(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400 or info.st_size > 8 * 1024**2:
        raise ValueError('bounded regular input manifest')
    value = json.loads(path.read_bytes())
    validate_manifest(value)
    return value


def binding(value):
    validate_manifest(value)
    return (dict(input_manifest_version=1, input_manifest_sha256=digest(value))
            if value['kind'] == MATCHED else {})


def check_cache_header(report, *, count, rows_sha, protocol_sha, sources_sha, matched_sha=None):
    # Expected identities come from the caller's already validated manifest.
    cache_shapes(count)
    if (not isinstance(report, dict) or report.get('success') is not True
            or report.get('rows') != count or report.get('rows_sha256') != rows_sha
            or report.get('protocol_sha256') != protocol_sha
            or report.get('source_manifest_sha256') != sources_sha):
        raise ValueError('input cache header identity/incomplete cache')
    if matched_sha is not None and (type(report.get('input_manifest_version')) is not int
            or report['input_manifest_version'] != 1 or report.get('input_manifest_sha256') != matched_sha):
        raise ValueError('matched cache manifest binding')
    return True


def storage_budget(count, manifest_bytes):
    state, mask = cache_shapes(count)
    if type(manifest_bytes) is not int or not 1 <= manifest_bytes <= 8*1024**2:
        raise ValueError('bounded manifest storage')
    state_bytes = state[0]*80*4*2560*4
    mask_bytes = mask[0]*80
    # Preserve the collector's existing 4GiB reserve for rolling archives/logs.
    return dict(state_bytes=state_bytes, mask_bytes=mask_bytes, header_reserve_bytes=2048,
                manifest_bytes=manifest_bytes, transient_reserve_bytes=4*1024**3,
                fresh_free_bytes=state_bytes+mask_bytes+2048+manifest_bytes+4*1024**3)
