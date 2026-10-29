"""Verify captured input availability without confusing a reused build path with identity.

Only offline provenance resolution. A frozen historical build is never executed
here and never establishes compatibility with the currently served executable.
"""
import hashlib
import json
from pathlib import Path
import re
import stat


def _digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_sources(manifest):
    manifest=Path(manifest)
    data=json.loads(manifest.read_text(encoding='utf-8'))
    if data.get('verified') is not True or not 1<=len(data.get('files',[]))<=128:
        raise ValueError('unverified source manifest')
    output=[]
    for entry in data['files']:
        digest=entry.get('sha256','')
        if not isinstance(digest,str) or not re.fullmatch('[0-9a-f]{64}',digest) or type(entry.get('bytes')) is not int:
            raise ValueError('source content identity')
        path=Path(entry['path'])
        try:
            observed=path.stat()
        except OSError:
            observed=None
        if observed and (observed.st_size,observed.st_mtime_ns)==(entry['bytes'],entry['mtime_ns']):
            method='captured-content-unchanged-stat';resolved=path
        elif observed and observed.st_size==entry['bytes'] and _digest(path)==digest:
            method='current-content-sha256';resolved=path
        else:
            frozen=manifest.parent/'frozen-inputs'/(digest+'.bin')
            try:
                info=frozen.lstat()
                valid=(stat.S_ISREG(info.st_mode) and not getattr(info,'st_file_attributes',0)&0x400
                       and info.st_size==entry['bytes'] and _digest(frozen)==digest)
            except OSError:
                valid=False
            if not valid:
                raise ValueError('captured source changed/unavailable and no matching frozen content')
            method='frozen-content-sha256';resolved=frozen
        output.append(dict(original=str(path),resolved=str(resolved),sha256=digest,verification=method))
    return output
