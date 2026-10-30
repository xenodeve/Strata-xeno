"""Offline whole-archive audit. Stream one observation at a time; never infer/train."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import stat
from tools.igc.coherent_observation import decode_plan, load_compact, first_four_fidelity, _digest, _json_bytes, _exclusive
from tools.igc.observation import _read


def stream_digest(path, size):
    """Hash a bounded regular file with 256-KiB scratch and detect concurrent edits."""
    info=path.lstat()
    if (not stat.S_ISREG(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400
            or info.st_size!=size or not 0<=size<=1024*4*2560*4):
        raise ValueError('audit bounded regular table')
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(256*1024):digest.update(block)
    after=path.lstat()
    if (info.st_size,info.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('audit table changed during reading')
    return digest.hexdigest()


def audit_archive(root, require_native=True):
    root=Path(root)
    info=root.lstat()
    if not stat.S_ISDIR(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
        raise ValueError('audit regular archive directory')
    complete_raw=_read(root/'COMPLETE.json',limit=256*1024)
    complete=json.loads(complete_raw)
    identity_raw=_read(root/'identity.json',limit=16384)
    identity=json.loads(identity_raw)
    plan_raw=_read(root/'plan.bin',limit=3*1024**2);plan=decode_plan(plan_raw)
    if (not isinstance(identity.get('sources'),dict) or not {'engine','profile'}<=identity['sources'].keys()
            or not isinstance(complete.get('rows'),list) or len(complete['rows'])!=len(plan['records'])):
        raise ValueError('audit archive source/count')
    for value in identity['sources'].values():_digest(value)
    for value in complete['rows']:_digest(value)
    dictionary_raw=_read(root/'dictionary.json',limit=256*1024)
    dictionary=json.loads(dictionary_raw)
    n=complete.get('embedding_rows')
    if type(n) is not int or not 0<=n<=1024 or not isinstance(dictionary,list) or len(dictionary)!=n:
        raise ValueError('audit dictionary geometry')
    tokens=set()
    for i,entry in enumerate(dictionary):
        if (not isinstance(entry,dict) or set(entry)!={'index','token','sha256'} or type(entry['index']) is not int
                or entry['index']!=i or type(entry['token']) is not int or not 0<=entry['token']<248320
                or entry['token'] in tokens):raise ValueError('audit dictionary identity')
        _digest(entry['sha256']);tokens.add(entry['token'])
    table_sha=stream_digest(root/'embeddings.bin',n*4*2560*4)
    if table_sha!=complete.get('embedding_sha256'):raise ValueError('audit complete embedding digest')
    if hashlib.sha256(dictionary_raw).hexdigest()!=complete.get('dictionary_sha256'):
        raise ValueError('audit dictionary digest')
    import numpy as np
    with (root/'embeddings.bin').open('rb') as stream:
        for entry in dictionary:
            raw=stream.read(4*2560*4)
            if hashlib.sha256(raw).hexdigest()!=entry['sha256'] or not np.isfinite(np.frombuffer(raw,dtype='<f4')).all():
                raise ValueError('audit dictionary row digest/finite')
    allowed={'COMPLETE.json','identity.json','plan.bin','dictionary.json','embeddings.bin'}
    receipts=[];counts=Counter();references=[];sources=Counter()
    for i,expected in enumerate(plan['records']):
        row=load_compact(root,i)
        if type(row.get('version')) is not int or row['version']!=3:raise ValueError('audit row version')
        _digest(row.get('source_capture_sha256'))
        allowed.add(f'{i}.json');allowed.update(row['files'])
        head=row['head'];required={'argmax','logsumexp','logits_sha256'}
        if (not isinstance(head,dict) or set(head) not in (required,required|{'target','target_logit'})
                or type(head['argmax']) is not int or not 0<=head['argmax']<248320
                or type(head['logsumexp']) not in (int,float) or not math.isfinite(head['logsumexp'])):
            raise ValueError('audit head statistics')
        _digest(head['logits_sha256'])
        if 'target' in head and (type(head['target']) is not int or not 0<=head['target']<248320
                or type(head['target_logit']) not in (int,float) or not math.isfinite(head['target_logit'])
                or head['target_logit']>head['logsumexp']+1e-9):
            raise ValueError('audit target statistics')
        if i<4:
            logits=row['logits'].astype(np.float64);maximum=float(logits.max())
            lse=maximum+math.log(float(np.exp(logits-maximum).sum()))
            if (head['argmax']!=int(logits.argmax()) or head['logsumexp']!=lse
                    or head['logits_sha256']!=row['files'][f'{i}.logits.bin']['sha256']
                    or ('target' in head and head['target_logit']!=float(logits[head['target']]))):
                raise ValueError('audit reference head statistics')
            references.append({k:row[k] for k in ('ordinal','role','owner','provenance','files','head')})
        source='native_verifier' if row['provenance']['source'] else 'synthetic'
        counts[row['role']]+=1;sources[source]+=1
        receipts.append(dict(ordinal=i,role=row['role'],base=row['base'],step=row['step'],
            owner=row['owner'],base_owner=row['base_owner'],prefix_tokens=len(expected['tokens']),
            receipt_sha256=complete['rows'][i],head=head))
        del row
    unexpected={p.name for p in root.iterdir()}-allowed
    if unexpected:raise ValueError('audit incomplete or unexpected archive files')
    native_qualified=sources['native_verifier']==len(plan['records']) and len(references)==4
    if native_qualified:first_four_fidelity(references,root)
    if require_native and not native_qualified:raise ValueError('audit requires actual native Verifier receipts')
    if (_read(root/'COMPLETE.json',limit=256*1024)!=complete_raw
            or _read(root/'identity.json',limit=16384)!=identity_raw
            or _read(root/'plan.bin',limit=3*1024**2)!=plan_raw):
        raise ValueError('audit archive changed during validation')
    return dict(version=1,kind='coherent_archive_audit',rows=len(receipts),anchors=counts['anchor'],outputs=counts['output'],
        plan_sha256=hashlib.sha256(plan_raw).hexdigest(),complete_sha256=hashlib.sha256(complete_raw).hexdigest(),
        model_sha256=plan['model'],tokenizer_sha256=plan['tokenizer'],sources=identity['sources'],
        embedding_sha256=table_sha,embedding_rows=n,source_counts={k:sources[k] for k in ('synthetic','native_verifier')},
        native_qualified=native_qualified,row_receipts=receipts,
        scope='archive/receipt integrity only; not fresh inference, learned accuracy or Hub parity')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-synthetic',action='store_true')
    args=parser.parse_args()
    result=audit_archive(args.archive,require_native=not args.allow_synthetic)
    _exclusive(args.output,_json_bytes(result))
    print(json.dumps({k:result[k] for k in ('rows','anchors','outputs','native_qualified')}))


if __name__=='__main__':main()
