"""Manifest-only preparation from seven frozen inspected archives. No generators."""
import argparse
from collections import defaultdict
import hashlib
import json
import unicodedata
from pathlib import Path
from tools.igc.coherent_observation import encode_plan,_json_bytes,_exclusive
from tools.igc.observation import _read
from tools.igc.replay_protocol import exclusion_keys

ARCHIVES={
 '1e898e951835e3dd08d07ec5801a33bc8030c4710129d73a2ef6030a3d74c724':64,
 '7e7e8ce4a5355d22d9497cb1b0ed1d9c8729d660592dbfe2dea41405c567c6f0':4480,
 '0aa4fa0c34e45b47f213279a638d7d11d1a04f0256877ddcf5e8be6c803452d9':10624,
 'ce1231adffa9bb4cd1a68074295811ec09d0dbc4a52f7cf59447620256027db3':1536,
 'ac6888b4cea70b8a63d6ab5aa54225ba60e475131e228be0506fbd0a93b4cf0b':320,
 '73a8107416eaed58fe423dab98a86907211f24fdc3bf7d2e2a3e8fbb20638cde':416,
 'a288a7d4e2d43a55fa4f75710da6af8493c061f21274aff1347d8ad815b1ce7f':30,
}
FRAMES={
 ('en','imperative'):'Work out {expr}; return its exact integer value.',
 ('en','question'):'What exact integer value does {expr} have?',
 ('en','quote-copy'):'Copy "{expr}" character for character without evaluation.',
 ('en','label'):'The item label is "{expr}". Acknowledge the label; do not solve it.',
 ('th','imperative'):'คำนวณ {expr} แล้วตอบเป็นจำนวนเต็มที่ถูกต้อง',
 ('th','question'):'ค่าจำนวนเต็มที่แน่นอนของ {expr} คือเท่าไร',
 ('th','quote-copy'):'คัดลอก "{expr}" ให้ตรงทุกตัวอักษรโดยไม่คำนวณ',
 ('th','label'):'ชื่อป้ายนี้คือ "{expr}" ให้รับทราบชื่อป้ายโดยไม่คำนวณ',
}
ORDINARY={
 'en':['Reply with the word amber.','Greet the reader briefly.','Copy the word violet exactly.','Name one cool color.'],
 'th':['ตอบด้วยคำว่าอำพัน','ทักทายผู้อ่านสั้น ๆ','คัดลอกคำว่าสีม่วงให้ตรงทุกตัว','บอกชื่อสีโทนเย็นหนึ่งสี'],
}
UNSUPPORTED={
 'en':['Calculate 123 + 456.','Give the exact fraction for 1 / 3.','Compute the square root of 2.','Execute a loop that counts to 100.'],
 'th':['คำนวณ 123 + 456','ตอบเศษส่วนที่แน่นอนของ 1 / 3','คำนวณรากที่สองของ 2','รันลูปที่นับถึง 100'],
}


def inspected_keys(row):
    if not isinstance(row.get('prompt'),str):raise ValueError('inspected prompt')
    if 'lhs' not in row or 'rhs' not in row:
        if 'lhs' in row or 'rhs' in row or row.get('ready',row.get('ready_op'))!=0:
            raise ValueError('inspected partial arithmetic label')
        return dict(prompt=unicodedata.normalize('NFKC',' '.join(row['prompt'].split())).casefold())
    r=dict(row)
    if 'operation' not in r:r['operation']={'+':1,'-':2,'*':3}.get(r.get('symbol'),r.get('ready_op'))
    r.setdefault('ready_op',r['operation'])
    return exclusion_keys(r)


def compile_plan(rows,output_ids,tokenizer,prefix,suffix,model,tokenizer_digest):
    if (not isinstance(rows,list) or not 1<=len(rows)<=400 or len({r['id'] for r in rows})!=len(rows)
            or len(set(output_ids))!=len(output_ids) or not set(output_ids)<={r['id'] for r in rows}):
        raise ValueError('coherent rows/output selection')
    records=[];requests=[];bases={}
    for i,row in enumerate(rows):
        tokens=tokenizer.encode(prefix+row['prompt']+suffix)
        if not 2<=len(tokens)<=256:raise ValueError('coherent prompt token bound')
        records.append(dict(role='anchor',base=i,step=0,tokens=tokens));bases[row['id']]=i
        requests.append(dict(ordinal=i,sample=row['id'],role='anchor',base=i,step=0,target=None))
    byid={r['id']:r for r in rows}
    for identity in output_ids:
        row=byid[identity];base=bases[identity];completion=tokenizer.encode(row['completion'])
        if not 1<=len(completion)<=32:raise ValueError('coherent completion bound')
        requests[base]['target']=completion[0]
        for j,target in enumerate(completion[1:],1):
            consumed=records[base]['tokens']+completion[:j]
            if len(consumed)>256:raise ValueError('coherent total prefix bound')
            records.append(dict(role='output',base=base,step=j,tokens=consumed))
            requests.append(dict(ordinal=len(requests),sample=identity,role='output',base=base,step=j,target=target))
    raw=encode_plan(dict(model=model,tokenizer=tokenizer_digest,records=records))
    return dict(version=2,kind='coherent-development-manifest',rows=rows,requests=requests,
                plan_sha256=hashlib.sha256(raw).hexdigest(),acceptance64_access=False,
                sealed_overlap='unknown; not inspected or used for sampling'),raw


def prepare(witness,tokenizer,prefix,suffix,model,tokenizer_digest):
    source=witness.get('source_sha256')
    if (witness.get('seed')!=20261101 or witness.get('success') is not True
            or not isinstance(source,dict) or len(source)!=7 or set(source.values())!=set(ARCHIVES)):
        raise ValueError('coherent witness/frozen source set')
    forbidden=defaultdict(set);proof=[]
    for path,digest in source.items():
        raw=_read(Path(path),limit=8*1024**2)
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('coherent inspected archive digest')
        old=json.loads(raw)['rows']
        if len(old)!=ARCHIVES[digest]:raise ValueError('coherent inspected archive count')
        for row in old:
            for k,v in inspected_keys(row).items():forbidden[k].add(v)
        proof.append(dict(sha256=digest,rows=len(old)))
    selected=witness.get('selected',[])
    if len(selected)!=48:raise ValueError('coherent48groups')
    rows=[];output=[];cells=set();keys=[];prompts=set()
    for g,entry in enumerate(selected):
        a,b,op=(entry[k] for k in ('lhs','rhs','operation'))
        if any(type(v) is not int for v in (a,b,op)) or not a or not b or abs(a)>99 or abs(b)>99 or op not in (1,2,3):raise ValueError('coherent arithmetic domain')
        cell=(op,len(str(abs(a))),len(str(abs(b))),a<0,b<0)
        if cell in cells:raise ValueError('coherent repeated grid cell')
        cells.add(cell);symbol={1:'+ ',2:'- ',3:'* '}[op].strip();expr=f'({a}) {symbol} ({b})'
        groupkeys=None
        chosen=(('en','imperative'),('en','question'),('th','imperative'),('th','question'))[g%4]
        for language in ('en','th'):
            for intent in ('imperative','question','quote-copy','label'):
                prompt=FRAMES[language,intent].format(expr=expr);positive=intent in ('imperative','question')
                row=dict(id=f'coherent-{g:02}-{language}-{intent}',group=g,split='development',language=language,intent=intent,
                    lhs=a,rhs=b,operation=op,ready_op=op if positive else 0,symbol=symbol,expression=expr,prompt=prompt)
                kk=exclusion_keys(row)
                if any(v in forbidden[k] for k,v in kk.items()) or kk['prompt'] in prompts:raise ValueError('coherent freshness collision')
                prompts.add(kk['prompt']);groupkeys=kk
                if (language,intent)==chosen:
                    answer={1:a+b,2:a-b,3:a*b}[op]
                    row['completion']=(f'The result is {answer}.\n</think>\n\n{answer}<|im_end|>' if language=='en'
                                       else f'ผลลัพธ์คือ {answer}\n</think>\n\n{answer}<|im_end|>')
                    output.append(row['id'])
                rows.append(row)
        keys.append(groupkeys)
    for language in ('en','th'):
        for kind,values in (('ordinary',ORDINARY[language]),('unsupported-current-mapper',UNSUPPORTED[language])):
            for j,prompt in enumerate(values):
                key=unicodedata.normalize('NFKC',' '.join(prompt.split())).casefold()
                if key in forbidden['prompt'] or key in prompts:raise ValueError('coherent negative prompt collision')
                prompts.add(key);rows.append(dict(id=f'coherent-{language}-{kind}-{j}',split='development',group=None,
                    language=language,intent=kind,ready_op=0,prompt=prompt))
    clusters=list(range(48))
    def root(i):
        while clusters[i]!=i:i=clusters[i]
        return i
    for i in range(48):
        for j in range(i):
            if keys[i]['symmetric_pair']==keys[j]['symmetric_pair'] or keys[i]['fact']==keys[j]['fact']:
                clusters[root(i)]=root(j)
    ids=[root(i) for i in range(48)]
    manifest,raw=compile_plan(rows,output,tokenizer,prefix,suffix,model,tokenizer_digest)
    manifest.update(seed=20261101,archives=sorted(proof,key=lambda x:x['sha256']),clusters=ids,
                    structural_clusters=len(set(ids)),independence='structural only; shared values remain',
                    anchors=400,output_samples=48,positive=192,negative=208,
                    corpus_sha256=hashlib.sha256(_json_bytes(rows)).hexdigest())
    if len(rows)!=400 or len(set(ids))!=44:raise ValueError('coherent frozen witness cluster geometry')
    return manifest,raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('witness','tokenizer','wrapper','model-digest','tokenizer-digest','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    from tools.qwenputer.tokenizer import NativeText
    wrapper=json.loads(_read(Path(a.wrapper),limit=16384))
    tokenizer=NativeText.from_pack(Path(a.tokenizer))
    if tokenizer.binding_sha256!=a.tokenizer_digest:raise ValueError('coherent actual tokenizer binding')
    manifest,raw=prepare(json.loads(_read(Path(a.witness),limit=65536)),tokenizer,
        wrapper['prefix'],wrapper['suffix'],a.model_digest,a.tokenizer_digest)
    root=Path(a.output);root.mkdir(exist_ok=False)
    _exclusive(root/'manifest.json',_json_bytes(manifest));_exclusive(root/'plan.bin',raw)
    print(json.dumps(dict(anchors=400,requests=len(manifest['requests']),plan_sha256=manifest['plan_sha256'],
                         structural_clusters=manifest['structural_clusters'],activation_collection=False)))


if __name__=='__main__':main()
