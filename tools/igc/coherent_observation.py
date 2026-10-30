"""Bounded v2 input plans/v3 archives. Never constructs acceptance corpora."""
import hashlib
import re
import struct
import json
import math
import os
from pathlib import Path
from tools.igc.observation import _read

PLAN_MAGIC=b'IGCCPL02'
MAX_RECORDS=2048
MAX_PREFIX=256


def _digest(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError('coherent digest')
    try:return bytes.fromhex(value)
    except ValueError as error:raise ValueError('coherent digest') from error


def _records(records):
    if not isinstance(records,list) or not 1<=len(records)<=MAX_RECORDS:
        raise ValueError('coherent plan count')
    outputs=False;steps={}
    for i,r in enumerate(records):
        if not isinstance(r,dict) or set(r)!={'role','base','step','tokens'}:
            raise ValueError('coherent plan input fields; targets forbidden')
        role,base,step,tokens=(r[k] for k in ('role','base','step','tokens'))
        if (role not in ('anchor','output') or type(base) is not int or type(step) is not int
                or not isinstance(tokens,list) or not 1<=len(tokens)<=MAX_PREFIX
                or any(type(t) is not int or not 0<=t<248320 for t in tokens)):
            raise ValueError('coherent plan record')
        if role=='anchor':
            if outputs or base!=i or step!=0:raise ValueError('coherent anchor order')
        else:
            outputs=True
            if not 0<=base<i or records[base]['role']!='anchor' or not 1<=step<=31:
                raise ValueError('coherent output parent')
            parent=records[base]['tokens']
            if len(tokens)!=len(parent)+step or tokens[:len(parent)]!=parent or step!=steps.get(base,0)+1:
                raise ValueError('coherent output consumed prefix')
            if step>1:
                previous=next(q for q in reversed(records[:i]) if q['role']=='output' and q['base']==base)
                if tokens[:-1]!=previous['tokens']:raise ValueError('coherent output sequence')
            steps[base]=step
    return records


def encode_plan(plan):
    if not isinstance(plan,dict) or set(plan)!={'model','tokenizer','records'}:
        raise ValueError('coherent plan fields')
    records=_records(plan['records'])
    body=PLAN_MAGIC+struct.pack('<II',2,len(records))+_digest(plan['model'])+_digest(plan['tokenizer'])
    for r in records:
        body+=struct.pack('<IIII',int(r['role']=='output'),r['base'],r['step'],len(r['tokens']))
        body+=struct.pack('<'+'I'*len(r['tokens']),*r['tokens'])
    return body+hashlib.sha256(body).digest()


def decode_plan(raw):
    if type(raw) is not bytes or not 112<=len(raw)<=3*1024**2 or raw[:8]!=PLAN_MAGIC:
        raise ValueError('coherent plan size/magic')
    if hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:raise ValueError('coherent plan digest')
    version,count=struct.unpack_from('<II',raw,8)
    if version!=2 or not 1<=count<=MAX_RECORDS:raise ValueError('coherent plan version/count')
    offset=80;records=[]
    for _ in range(count):
        if offset+16>len(raw)-32:raise ValueError('coherent plan truncated')
        role,base,step,n=struct.unpack_from('<IIII',raw,offset);offset+=16
        if role>1 or not 1<=n<=MAX_PREFIX or offset+n*4>len(raw)-32:
            raise ValueError('coherent plan record bytes')
        tokens=list(struct.unpack_from('<'+'I'*n,raw,offset));offset+=n*4
        records.append(dict(role='output' if role else 'anchor',base=base,step=step,tokens=tokens))
    if offset!=len(raw)-32:raise ValueError('coherent plan trailing bytes')
    result=dict(model=raw[16:48].hex(),tokenizer=raw[48:80].hex(),records=_records(records))
    if encode_plan(result)!=raw:raise ValueError('noncanonical coherent plan')
    return result


def decode_capture(raw,plan_raw,index):
    """Read one completed, self-digested producer staging row (at most8MiB)."""
    import numpy as np
    plan=decode_plan(plan_raw)
    if type(index) is not int or not 0<=index<len(plan['records']):raise ValueError('capture index')
    if type(raw) is not bytes or not 172<=len(raw)<=8*1024**2:
        raise ValueError('capture byte bound')
    if hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:raise ValueError('capture digest')
    magic,version,role,ordinal,base,step,n,nrows,owner=struct.unpack_from('<8s7IQ',raw)
    r=plan['records'][index];expected_rows=0 if r['role']=='output' else min(len(r['tokens'])-1,80)
    if (magic!=b'IGCCOH03' or version!=3 or ordinal!=index or not owner
            or (role,base,step,n,nrows)!=(int(r['role']=='output'),r['base'],r['step'],len(r['tokens']),expected_rows)
            or raw[44:76].hex()!=plan['model'] or raw[76:108].hex()!=plan['tokenizer'] or raw[108:140]!=plan_raw[-32:]):
        raise ValueError('capture role/owner/plan binding')
    if len(raw)<196+32:raise ValueError('capture provenance bytes')
    values=struct.unpack_from('<4Q6I',raw,140)
    provenance=dict(zip(('reused','accepted','previous_owner','fence','may_reuse','cache','streams','flags','source','head_fidelity'),values,strict=True))
    p=provenance
    if (p['reused']>=n or p['accepted']!=n or p['previous_owner']>=owner or p['fence']!=owner
            or p['may_reuse']>1 or p['cache']!=(0 if not p['reused'] else 1 if p['may_reuse'] else 2)
            or not p['streams']&1 or p['streams']>7 or p['flags']!=31 or p['source']>1 or p['head_fidelity']>1
            or (index==0 and p['previous_owner']!=0)):
        raise ValueError('capture native provenance')
    if p['may_reuse'] and p['reused']:
        if not index or r['tokens'][:p['reused']]!=plan['records'][index-1]['tokens'][:p['reused']]:
            raise ValueError('capture reused prefix lineage')
    at=196;tokens=list(struct.unpack_from('<'+'I'*n,raw,at));at+=n*4
    if tokens!=r['tokens']:raise ValueError('capture consumed prefix')
    mask=np.frombuffer(raw,dtype='u1',count=80,offset=at);at+=80
    expected=np.zeros(80,dtype='u1')
    if nrows:expected[-nrows:]=1
    if not np.array_equal(mask,expected):raise ValueError('capture missing or foreign feature rows')
    arrays={}
    for name,size in [('embedding',0 if role else 80*4*2560),('attention',0 if role else 80*4*2560),
                      ('mixed',2560),('logits',248320)]:
        if at+size*4>len(raw)-32:raise ValueError('capture truncated tensor')
        a=np.frombuffer(raw,dtype='<f4',count=size,offset=at);at+=size*4
        if not np.isfinite(a).all():raise ValueError('capture nonfinite tensor')
        if name in ('embedding','attention'):
            a=a.reshape((-1,4,2560))
            if not role and np.any(a[:80-nrows]):raise ValueError('capture stale padding')
        arrays[name]=a
    if at!=len(raw)-32:raise ValueError('capture trailing tensor bytes')
    return dict(**r,owner=owner,ordinal=index,mask=mask,provenance=provenance,raw_sha256=hashlib.sha256(raw).hexdigest(),**arrays)


def first_four_fidelity(receipts,root):
    """Continuation gate, not a claim inferred from the collector timing."""
    if len(receipts)!=4:raise ValueError('four producer fidelity receipts required')
    for i,r in enumerate(receipts):
        p=r.get('provenance',{})
        if (r.get('ordinal')!=i or r.get('role')!='anchor' or p.get('source')!=1
                or p.get('head_fidelity')!=1 or p.get('flags')!=31 or p.get('fence')!=r.get('owner')):
            raise ValueError('native producer/head fidelity not established before anchor five')
        for name,size in ((f'{i}.mixed.bin',2560*4),(f'{i}.logits.bin',248320*4),(f'{i}.attention.bin',80*4*2560*4)):
            desc=r.get('files',{}).get(name)
            if not isinstance(desc,dict) or desc.get('bytes')!=size:raise ValueError('fidelity reference file missing/shape')
            raw=_read(Path(root)/name,size=desc['bytes'],limit=4*1024**2)
            if hashlib.sha256(raw).hexdigest()!=desc['sha256']:raise ValueError('fidelity reference digest')
        if r['files'][f'{i}.logits.bin']['sha256']!=r.get('head',{}).get('logits_sha256'):
            raise ValueError('fidelity head reference hash')
    return True


def _json_bytes(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()


def _exclusive(path,raw):
    with path.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())


class CompactArchive:
    """One bounded row at a time; no resume/promotion of partial archives."""
    def __init__(self,root,plan_raw,sources):
        if not isinstance(sources,dict) or not {'engine','profile'}<=sources.keys():raise ValueError('compact sources')
        for v in sources.values():_digest(v)
        self.plan=decode_plan(plan_raw);self.plan_raw=plan_raw;self.root=Path(root)
        if self.root.exists():raise ValueError('compact archive exists or partial; resume refused')
        self.root.mkdir();_exclusive(self.root/'plan.bin',plan_raw)
        self.identity=dict(version=3,kind='coherent_compact_observation',sources=sources,
                           plan_sha256=hashlib.sha256(plan_raw).hexdigest())
        _exclusive(self.root/'identity.json',_json_bytes(self.identity))
        _exclusive(self.root/'embeddings.bin',b'')
        self.dictionary={};self.receipts=[];self.owners=[];self.closed=False;self.failed=False

    def ingest(self,raw,index,target=None):
        import numpy as np
        if self.closed or self.failed or index!=len(self.receipts):raise ValueError('compact row order/lifetime')
        try:
            r=decode_capture(raw,self.plan_raw,index)
            if self.owners and r['owner']<=self.owners[-1]:raise ValueError('compact stale capture owner')
            if r['provenance']['previous_owner']!=(self.owners[-1] if self.owners else 0):raise ValueError('compact previous native owner')
            if target is not None and (type(target) is not int or not 0<=target<248320):raise ValueError('compact offline target')
            entries=[None]*80;files={}
            if r['role']=='anchor':
                first=max(0,len(r['tokens'])-1-80);actual=r['tokens'][first:-1]
                with (self.root/'embeddings.bin').open('ab') as table:
                    for pos,token in enumerate(actual,80-len(actual)):
                        row=r['embedding'][pos].tobytes();digest=hashlib.sha256(row).hexdigest()
                        if token in self.dictionary:
                            entry=self.dictionary[token]
                            if entry['sha256']!=digest:raise ValueError('nonconstant native embedding; dictionary invalid')
                        else:
                            if len(self.dictionary)>=1024:raise ValueError('compact embedding dictionary bound')
                            entry=dict(token=token,index=len(self.dictionary),sha256=digest)
                            self.dictionary[token]=entry;table.write(row)
                        entries[pos]=entry
                name=f'{index}.attention.bin';data=r['attention'].tobytes();_exclusive(self.root/name,data)
                files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            for name,data in [(f'{index}.mixed.bin',r['mixed'].tobytes())]+(
                    [(f'{index}.logits.bin',r['logits'].tobytes())] if index<4 else []):
                _exclusive(self.root/name,data);files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            logits=r['logits'].astype(np.float64);maximum=float(logits.max())
            head=dict(argmax=int(logits.argmax()),logsumexp=maximum+math.log(float(np.exp(logits-maximum).sum())),
                      logits_sha256=hashlib.sha256(r['logits'].tobytes()).hexdigest())
            if target is not None:head.update(target=target,target_logit=float(logits[target]))
            receipt=dict(version=3,ordinal=index,role=r['role'],base=r['base'],step=r['step'],tokens=r['tokens'],provenance=r['provenance'],
                owner=r['owner'],base_owner=r['owner'] if r['role']=='anchor' else self.owners[r['base']],
                plan_sha256=self.identity['plan_sha256'],source_capture_sha256=r['raw_sha256'],
                embedding=entries,mask=r['mask'].tolist(),files=files,head=head)
            encoded=_json_bytes(receipt);_exclusive(self.root/f'{index}.json',encoded)
            self.receipts.append(hashlib.sha256(encoded).hexdigest());self.owners.append(r['owner'])
            return receipt
        except Exception:
            self.failed=True;raise

    def finalize(self):
        if self.closed or self.failed or len(self.receipts)!=len(self.plan['records']):raise ValueError('compact incomplete archive')
        dictionary=_json_bytes(list(self.dictionary.values()));_exclusive(self.root/'dictionary.json',dictionary)
        with (self.root/'embeddings.bin').open('rb') as f:table_hash=hashlib.file_digest(f,'sha256').hexdigest()
        complete=dict(version=3,identity_sha256=hashlib.sha256(_json_bytes(self.identity)).hexdigest(),rows=self.receipts,
            dictionary_sha256=hashlib.sha256(dictionary).hexdigest(),embedding_sha256=table_hash,
            embedding_rows=len(self.dictionary))
        _exclusive(self.root/'COMPLETE.json',_json_bytes(complete));self.closed=True
        return complete


def load_compact(root,index):
    """Only complete archives; checks all referenced bytes without loading all rows."""
    import numpy as np
    root=Path(root)
    complete=json.loads(_read(root/'COMPLETE.json',limit=256*1024))
    identity_raw=_read(root/'identity.json',limit=16384);identity=json.loads(identity_raw)
    plan_raw=_read(root/'plan.bin',limit=3*1024**2);plan=decode_plan(plan_raw)
    if (complete.get('version')!=3 or identity.get('version')!=3
            or identity.get('kind')!='coherent_compact_observation'
            or hashlib.sha256(identity_raw).hexdigest()!=complete.get('identity_sha256')
            or hashlib.sha256(plan_raw).hexdigest()!=identity.get('plan_sha256')
            or not isinstance(complete.get('rows'),list) or len(complete['rows'])!=len(plan['records'])
            or type(index) is not int or not 0<=index<len(plan['records'])):raise ValueError('compact completion identity')
    data=_read(root/f'{index}.json',limit=32768);r=json.loads(data)
    if hashlib.sha256(data).hexdigest()!=complete['rows'][index]:raise ValueError('compact row receipt digest')
    expected=plan['records'][index]
    if any(r.get(k)!=expected[k] for k in ('role','base','step','tokens')) or r.get('ordinal')!=index or r.get('plan_sha256')!=identity['plan_sha256']:
        raise ValueError('compact row identity')
    if type(r.get('owner')) is not int or not 0<r['owner']<2**64:raise ValueError('compact owner')
    p=r.get('provenance',{})
    if (set(p)!={'reused','accepted','previous_owner','fence','may_reuse','cache','streams','flags','source','head_fidelity'}
            or any(type(v) is not int or v<0 for v in p.values())
            or p['accepted']!=len(r['tokens']) or p['reused']>=len(r['tokens']) or p['fence']!=r['owner'] or p['previous_owner']>=r['owner']
            or p['may_reuse']>1 or p['cache']!=(0 if not p['reused'] else 1 if p['may_reuse'] else 2)
            or not p['streams']&1 or p['streams']>7 or p['flags']!=31 or p['source']>1 or p['head_fidelity']>1):
        raise ValueError('compact native provenance')
    if index:
        prev=_read(root/f'{index-1}.json',limit=32768);previous=json.loads(prev)
        if hashlib.sha256(prev).hexdigest()!=complete['rows'][index-1] or p['previous_owner']!=previous.get('owner'):
            raise ValueError('compact previous owner lineage')
        if p['may_reuse'] and p['reused'] and r['tokens'][:p['reused']]!=previous['tokens'][:p['reused']]:
            raise ValueError('compact reused prefix lineage')
    elif p['previous_owner']:raise ValueError('compact initial owner lineage')
    parent=json.loads(_read(root/f"{r['base']}.json",limit=32768))
    if (hashlib.sha256(_read(root/f"{r['base']}.json",limit=32768)).hexdigest()!=complete['rows'][r['base']]
            or r.get('base_owner')!=parent.get('owner')):raise ValueError('compact base owner')
    expected_mask=[0]*80
    n=min(len(r['tokens'])-1,80) if r['role']=='anchor' else 0
    if n:expected_mask[-n:]=[1]*n
    if r.get('mask')!=expected_mask:raise ValueError('compact mask')
    arrays={}
    expected_files={f'{index}.mixed.bin':2560*4}
    if r['role']=='anchor':expected_files[f'{index}.attention.bin']=80*4*2560*4
    if index<4:expected_files[f'{index}.logits.bin']=248320*4
    if not isinstance(r.get('files'),dict) or set(r['files'])!=set(expected_files):raise ValueError('compact file roles')
    for name,size in expected_files.items():
        value=_read(root/name,size=size,limit=4*1024**2);description=r['files'][name]
        if description!=dict(bytes=size,sha256=hashlib.sha256(value).hexdigest()):raise ValueError('compact array digest')
        a=np.frombuffer(value,dtype='<f4')
        if not np.isfinite(a).all():raise ValueError('compact finite')
        arrays[name.split('.')[1]]=a.reshape(80,4,2560) if '.attention.' in name else a
    embedding=np.zeros((80,4,2560),dtype='<f4')
    if not isinstance(r.get('embedding'),list) or len(r['embedding'])!=80:raise ValueError('compact embedding indices')
    table=root/'embeddings.bin'
    dictionary_raw=_read(root/'dictionary.json',limit=256*1024)
    if hashlib.sha256(dictionary_raw).hexdigest()!=complete.get('dictionary_sha256'):raise ValueError('compact dictionary digest')
    dictionary=json.loads(dictionary_raw)
    if type(complete.get('embedding_rows')) is not int or not 0<=complete['embedding_rows']<=1024 or len(dictionary)!=complete['embedding_rows']:
        raise ValueError('compact dictionary count')
    # Per-row hashes are verified while gathering; no40MiB array materialization.
    if table.is_symlink() or not table.is_file() or table.stat().st_size!=len(dictionary)*4*2560*4:raise ValueError('compact table bytes')
    with table.open('rb') as f:
        for j,entry in enumerate(r['embedding']):
            if not expected_mask[j]:
                if entry is not None:raise ValueError('compact foreign padded row')
                continue
            if not isinstance(entry,dict) or type(entry.get('index')) is not int or not 0<=entry['index']<len(dictionary) or dictionary[entry['index']]!=entry:
                raise ValueError('compact embedding reference')
            p=max(0,len(r['tokens'])-1-80)+j-(80-sum(expected_mask))
            if entry['token']!=r['tokens'][p]:raise ValueError('compact embedding token')
            f.seek(entry['index']*4*2560*4);raw=f.read(4*2560*4)
            if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('compact embedding row digest')
            embedding[j]=np.frombuffer(raw,dtype='<f4').reshape(4,2560)
    if not np.isfinite(embedding).all():raise ValueError('compact nonfinite embedding')
    return {**r,**arrays,'embedding':embedding}


def load_input_observation(root,index):
    """Original anchors only; offline answer/target receipts are not features."""
    row=load_compact(root,index)
    if row['role']!='anchor' or row['step']!=0:raise ValueError('output role is not an input/readiness example')
    return {k:row[k] for k in ('embedding','attention','mixed','mask','tokens','owner','ordinal')}
