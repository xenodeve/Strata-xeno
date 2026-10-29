"""Bounded native contextual observations for offline input-head training.

These files contain frozen F32 activations, not executable artifacts. Token IDs
bind their identity and never enter the neural readout as ordinal features.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import time
import numpy as np
import torch
from tools.igc.balanced_corpus import expanded_dataset
from tools.igc.observation import load_observation,_read
from tools.igc.provenance import verify_sources
from tools.qwenputer.memory import system_snapshot,process_snapshot
from tools.qwenputer.tokenizer import NativeText


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def regular(path):
    info=Path(path).lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:
        raise ValueError('context cache requires regular non-reparse files')
    return info


def decode_archive(raw,tokens):
    header=struct.Struct('<8sIIIIIQQ');tokens=np.asarray(tokens)
    if tokens.ndim!=1 or not 2<=len(tokens)<=256 or len(raw)<header.size:
        raise ValueError('context archive input bounds')
    magic,version,hc,dim,capacity,count,anchor,prompt=header.unpack_from(raw)
    if ((magic,version,hc,dim,capacity,anchor,prompt)!=(b'IGCATN01',1,4,2560,80,len(tokens)-1,len(tokens))
            or count!=min(anchor,80) or len(raw)!=header.size+count*(16+40960)):
        raise ValueError('context archive geometry or byte count')
    values=np.zeros((80,4,2560),dtype=np.float32);valid=np.zeros(80,dtype=np.bool_)
    for i,p in enumerate(range(max(0,anchor-80),anchor)):
        offset=header.size+i*(16+40960)
        actual,token,source=struct.unpack_from('<qII',raw,offset)
        if (actual,token,source)!=(p,int(tokens[p]),1):raise ValueError('context archive owner/position/token')
        row=np.frombuffer(raw,dtype='<f4',count=10240,offset=offset+16).reshape(4,2560)
        if not np.isfinite(row).all():raise ValueError('context archive nonfinite')
        index=p-anchor+80;values[index]=row;valid[index]=True
    return values,valid


class ContextInputs:
    def __init__(self,states,valid,device):
        if (states.dtype!=np.float32 or states.ndim!=4 or states.shape[1:]!=(80,4,2560)
                or valid.dtype!=np.bool_ or valid.shape!=states.shape[:2]
                or not 1<=len(states)<=11264 or not valid.any(axis=1).all()):
            raise ValueError('context cache tensor geometry')
        self.states=states;self.valid=valid;self.device=device
    def batch(self,selection):
        if (selection.ndim!=1 or not 1<=len(selection)<=64 or selection.dtype not in (np.int32,np.int64)
                or np.any(selection<0) or np.any(selection>=len(self.states))):
            raise ValueError('bounded contextual minibatch')
        # Only a minibatch crosses to CUDA. The full cache stays file-backed.
        states=np.array(self.states[selection],dtype=np.float32,copy=True,order='C')
        mask=np.array(self.valid[selection],copy=True)
        positions=np.broadcast_to(np.arange(80,dtype=np.int64),mask.shape).copy()
        return (torch.from_numpy(states).to(self.device),torch.from_numpy(mask).to(self.device),
                torch.from_numpy(positions).to(self.device),
                torch.zeros((len(selection),2560),dtype=torch.float32,device=self.device))


def load_cache(root,sources,protocol):
    report=json.loads(_read(root/'report.json',limit=256*1024))
    if (report.get('success') is not True or report.get('rows')!=10624
            or report.get('protocol_sha256')!=digest(protocol)
            or report.get('source_manifest_sha256')!=digest(sources)):
        raise ValueError('context preparation identity/incomplete cache')
    verify_sources(sources)
    corpus=json.loads(_read(root/'corpus.json',limit=16*1024**2))
    if corpus!=expanded_dataset() or corpus['rows_sha256']!=report['rows_sha256']:
        raise ValueError('context frozen corpus')
    arrays=[]
    for name,expected_dtype,shape in [('states',np.float32,(10624,80,4,2560)),('valid',np.bool_,(10624,80))]:
        path=root/f'{name}.npy';info=regular(path);expected=int(np.prod(shape))*np.dtype(expected_dtype).itemsize
        if not expected<info.st_size<=expected+1024 or digest(path)!=report[name+'_sha256']:
            raise ValueError('context cache bytes/content identity')
        array=np.load(path,mmap_mode='r',allow_pickle=False,max_header_size=1024)
        if (array.dtype!=expected_dtype or array.shape!=shape or not array.flags.c_contiguous
                or info.st_size!=array.offset+array.nbytes):raise ValueError('context NPY contract')
        arrays.append(array)
    states,valid=arrays
    ContextInputs(states,valid,'cpu')
    return states,valid,corpus['rows'],report


def prepare(executable,model,pack,reference,sources,protocol,output):
    started=time.monotonic();verified=verify_sources(sources)
    protocol_hash=digest(protocol);source_hash=digest(sources)
    manifest=json.loads(_read(sources,limit=256*1024))
    pinned={Path(record['path']).resolve() for record in manifest['files']}
    if model.resolve() not in pinned or (pack/'index.txt').resolve() not in pinned:
        raise ValueError('context model/pack not bound to verified source manifest')
    regular(executable);exe_hash=digest(executable)
    original=json.loads(_read(reference/'corpus.json',limit=1024**2))['rows']
    tokenizer=NativeText.from_pack(pack/'tokenizer');prefix=suffix=None
    for i,row in enumerate(original):
        observed=load_observation(reference,i);tokens=observed['prefix_tokens'].tolist()
        raw=tokenizer.decode_bytes(tokens);prompt=row['prompt'].encode('utf-8')
        if raw.count(prompt)!=1:raise ValueError('context wrapper occurrence')
        at=raw.index(prompt)
        if prefix is None:prefix,suffix=raw[:at],raw[at+len(prompt):]
        if raw!=prefix+prompt+suffix or tokenizer.encode(raw.decode('utf-8'))!=tokens:
            raise ValueError('context actual native wrapper fidelity')
    corpus=expanded_dataset();rows=corpus['rows'];encoded=[]
    for row in rows:
        prompt=row['prompt'].encode('utf-8');tokens=tokenizer.encode((prefix+prompt+suffix).decode('utf-8'))
        if not 2<=len(tokens)<=256:raise ValueError('context prefix bounds')
        offsets=np.cumsum([0]+[len(tokenizer.bytes[token]) for token in tokens])
        expr=row['expression'].encode('utf-8');left=len(prefix)+prompt.index(expr);right=left+len(expr)
        positions=[p for p in range(len(tokens)) if offsets[p]<right and offsets[p+1]>left]
        if not positions or min(positions)<max(0,len(tokens)-1-80) or max(positions)>=len(tokens)-1:
            raise ValueError('context expression outside observed window')
        encoded.append(tokens)
    # Price before creating the large output or launching the owned child.
    system=system_snapshot();payload=len(rows)*80*4*2560*4
    if (shutil.disk_usage(output.parent).free<payload+4*1024**3
            or system['system_commit_available_bytes']<12*1024**3
            or system['physical_available_bytes']<8*1024**3):
        raise RuntimeError('offline contextual collection preflight; no model launched')
    output.mkdir(exist_ok=False)
    (output/'corpus.json').write_text(json.dumps(corpus,ensure_ascii=False,indent=2),encoding='utf-8')
    states=np.lib.format.open_memmap(output/'states.npy',mode='w+',dtype='<f4',shape=(len(rows),80,4,2560))
    valid=np.lib.format.open_memmap(output/'valid.npy',mode='w+',dtype=np.bool_,shape=(len(rows),80))
    records=[];chunks=[];samples=[]
    for start in range(0,len(rows),64):
        if time.monotonic()-started>1200:raise TimeoutError('context collection deadline; incomplete cache')
        subset=encoded[start:start+64];chunk=output/f'chunk-{start:05}';chunk.mkdir()
        archive=chunk/'captures';archive.mkdir()
        raw=b'IGCPFX01'+struct.pack('<I',len(subset))+b''.join(struct.pack('<I',len(t))+np.array(t,dtype='<i4').tobytes() for t in subset)
        inputs=chunk/'prefixes.bin';inputs.write_bytes(raw)
        command=[str(executable),str(model),str(pack),str(inputs),str(archive),'8']
        tick=time.monotonic()
        with (chunk/'stdout.log').open('w',encoding='utf-8') as stdout,(chunk/'stderr.log').open('w',encoding='utf-8') as stderr:
            child=subprocess.Popen(command,creationflags=0x08000000,stdout=stdout,stderr=stderr)
            try:
                while child.poll() is None:
                    current=system_snapshot()
                    try:process=process_snapshot(child.pid)
                    except OSError:
                        if child.poll() is None:raise
                        break # The owned child exited between poll and OpenProcess.
                    samples.append(dict(row=start,seconds=time.monotonic()-started,process=process,system=current))
                    if time.monotonic()-tick>90:raise TimeoutError('context child deadline')
                    if current['system_commit_available_bytes']<2*1024**3 or current['physical_available_bytes']<1024**3:
                        raise RuntimeError('offline system emergency floor, not a private-commit cap')
                    time.sleep(.1)
                if child.returncode:raise RuntimeError(f'context child exit {child.returncode}: {chunk.name}/stderr.log')
            finally:
                if child.poll() is None:child.kill();child.wait(timeout=10)
        owned=[]
        for j,tokens in enumerate(subset):
            path=archive/f'{j}.attention.bin';data=_read(path,limit=4*1024**2)
            values,mask=decode_archive(data,tokens);states[start+j]=values;valid[start+j]=mask
            records.append(dict(id=rows[start+j]['id'],prefix_sha256=hashlib.sha256(np.array(tokens,dtype='<i4').tobytes()).hexdigest(),
                                archive_sha256=hashlib.sha256(data).hexdigest(),tokens=len(tokens),observed_rows=int(mask.sum())))
            owned.append(path)
        states.flush();valid.flush()
        # Exact F32 consolidated cache is retained. Only its redundant known
        # temporary archives are removed, never a computed recursive target.
        for path in owned:
            if path.resolve().parent!=archive.resolve():raise RuntimeError('context owned cleanup boundary')
            path.unlink()
        archive.rmdir()
        chunks.append(dict(start=start,count=len(subset),seconds=time.monotonic()-tick,input_sha256=hashlib.sha256(raw).hexdigest()))
        (output/'memory.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
        print('context captured',start+len(subset),'/',len(rows),'seconds',round(time.monotonic()-started,2),flush=True)
    if (digest(executable)!=exe_hash or digest(protocol)!=protocol_hash or digest(sources)!=source_hash):
        raise RuntimeError('context binary/protocol/source changed during capture')
    verify_sources(sources)
    del states,valid
    (output/'records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    state_hash=digest(output/'states.npy')
    report=dict(success=True,kind='native first-attention F32 contextual cache, offline training only',rows=len(rows),
                rows_sha256=corpus['rows_sha256'],states_sha256=state_hash,valid_sha256=digest(output/'valid.npy'),
                protocol_sha256=protocol_hash,source_manifest_sha256=source_hash,exe_sha256=exe_hash,
                tokenizer_binding_sha256=tokenizer.binding_sha256,preflight=system,elapsed_seconds=time.monotonic()-started,
                chunks=chunks,source_resolution=verified,collection_source_sha256=digest(Path(__file__)),
                records_sha256=digest(output/'records.json'),cache_sha256=state_hash)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('context cache complete',report['states_sha256'],'seconds',report['elapsed_seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('executable','model','pack','reference','sources','protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();prepare(**vars(args))
