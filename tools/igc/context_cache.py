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
from tools.igc.input_manifest import load_manifest, validate_manifest, native_prompt_rows, cache_shapes, binding, check_cache_header, storage_budget
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


class NpyRows:
    """Exact F32 NPY rows with bounded I/O, never a whole-file working-set map."""
    def __init__(self,path,writable=False):
        self.path=Path(path);regular(self.path)
        self.file=self.path.open('r+b' if writable else 'rb',buffering=0)
        try:
            if np.lib.format.read_magic(self.file)!=(1,0):raise ValueError('context NPY version')
            shape,fortran,dtype=np.lib.format.read_array_header_1_0(self.file,max_header_size=1024)
            self.offset=self.file.tell();self.dtype=dtype;self.shape=shape;self.ndim=len(shape)
            if (dtype!=np.float32 or len(shape)!=4 or shape[1:]!=(80,4,2560)
                    or not 1<=shape[0]<=11264 or fortran or self.offset>1024
                    or self.path.stat().st_size!=self.offset+int(np.prod(shape))*4):
                raise ValueError('context NPY geometry or bytes')
            self.row_bytes=80*4*2560*4
        except BaseException:self.file.close();raise
    @classmethod
    def create(cls,path,count):
        if type(count) is not int or not 1<=count<=11264:raise ValueError('context row count')
        with Path(path).open('xb',buffering=0) as output:
            np.lib.format.write_array_header_1_0(output,dict(descr='<f4',fortran_order=False,shape=(count,80,4,2560)))
            output.truncate(output.tell()+count*80*4*2560*4)
        return cls(path,True)
    def __len__(self):return self.shape[0]
    def _selection(self,selection):
        if (not isinstance(selection,np.ndarray) or selection.ndim!=1 or not 1<=len(selection)<=64
                or selection.dtype not in (np.int32,np.int64) or np.any(selection<0) or np.any(selection>=len(self))):
            raise ValueError('bounded file-row selection')

    def __getitem__(self,selection):
        self._selection(selection)
        result=np.empty((len(selection),80,4,2560),dtype=np.float32)
        return self.read_into(selection,result)

    def read_into(self,selection,result):
        self._selection(selection)
        if (not isinstance(result,np.ndarray) or result.shape!=(len(selection),80,4,2560)
                or result.dtype!=np.float32 or not result.flags.c_contiguous or not result.flags.writeable):
            raise ValueError('bounded writable F32 row destination')
        for i,row in enumerate(selection):
            self.file.seek(self.offset+int(row)*self.row_bytes)
            view=memoryview(result[i]).cast('B');done=0
            while done<len(view):
                count=self.file.readinto(view[done:])
                if not count:raise ValueError('context cache truncated while reading')
                done+=count
        return result
    def __setitem__(self,index,value):
        if (type(index) is not int or not 0<=index<len(self) or value.shape!=(80,4,2560)
                or value.dtype!=np.float32 or not value.flags.c_contiguous):
            raise ValueError('context row write contract')
        self.file.seek(self.offset+index*self.row_bytes)
        view=memoryview(value).cast('B');done=0
        while done<len(view):
            count=self.file.write(view[done:])
            if not count:raise OSError('short context cache write')
            done+=count
    def flush(self):self.file.flush();os.fsync(self.file.fileno())
    def close(self):self.file.close()
    def __del__(self):
        if getattr(self,'file',None):self.file.close()


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
        states=np.ascontiguousarray(self.states[selection],dtype=np.float32)
        mask=np.array(self.valid[selection],copy=True)
        positions=np.broadcast_to(np.arange(80,dtype=np.int64),mask.shape).copy()
        return (torch.from_numpy(states).to(self.device),torch.from_numpy(mask).to(self.device),
                torch.from_numpy(positions).to(self.device),
                torch.zeros((len(selection),2560),dtype=torch.float32,device=self.device))


def load_cache(root,sources,protocol):
    report=json.loads(_read(root/'report.json',limit=256*1024))
    corpus=json.loads(_read(root/'corpus.json',limit=16*1024**2))
    count=validate_manifest(corpus);manifest_binding=binding(corpus)
    check_cache_header(report,count=count,rows_sha=corpus['rows_sha256'],protocol_sha=digest(protocol),
                       sources_sha=digest(sources),matched_sha=manifest_binding.get('input_manifest_sha256'))
    verify_sources(sources)
    arrays=[]
    state_shape,mask_shape=cache_shapes(count)
    for name,expected_dtype,shape in [('states',np.float32,state_shape),('valid',np.bool_,mask_shape)]:
        path=root/f'{name}.npy';info=regular(path);expected=int(np.prod(shape))*np.dtype(expected_dtype).itemsize
        if not expected<info.st_size<=expected+1024 or digest(path)!=report[name+'_sha256']:
            raise ValueError('context cache bytes/content identity')
        array=NpyRows(path) if name=='states' else np.load(path,allow_pickle=False,max_header_size=1024)
        if array.dtype!=expected_dtype or array.shape!=shape:raise ValueError('context NPY contract')
        if name=='valid':
            with path.open('rb') as source:
                if np.lib.format.read_magic(source)!=(1,0):raise ValueError('context mask NPY version')
                _,fortran,_=np.lib.format.read_array_header_1_0(source,max_header_size=1024)
                if fortran or not array.flags.c_contiguous or info.st_size!=source.tell()+array.nbytes:
                    raise ValueError('context mask NPY byte layout')
        arrays.append(array)
    states,valid=arrays
    ContextInputs(states,valid,'cpu')
    return states,valid,corpus['rows'],report


def prepare(executable,model,pack,reference,sources,protocol,output,resume_from=0,prior_log=None,corpus_manifest=None):
    # Refuse an unrecognized explicit corpus before source hashing or model I/O.
    corpus=load_manifest(corpus_manifest) if corpus_manifest is not None else expanded_dataset()
    validate_manifest(corpus);manifest_binding=binding(corpus)
    manifest_file_hash=digest(corpus_manifest) if corpus_manifest is not None else None
    adapter_paths=[Path(__file__).with_name(name) for name in ('input_manifest.py','matched_corpus.py')]
    adapter_hashes={p.name:digest(p) for p in adapter_paths}
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
    rows=corpus['rows'];encoded=[]
    for row in native_prompt_rows(rows):
        prompt=row['prompt'].encode('utf-8');tokens=tokenizer.encode((prefix+prompt+suffix).decode('utf-8'))
        if not 2<=len(tokens)<=256:raise ValueError('context prefix bounds')
        offsets=np.cumsum([0]+[len(tokenizer.bytes[token]) for token in tokens])
        expr=row['expression'].encode('utf-8');left=len(prefix)+prompt.index(expr);right=left+len(expr)
        positions=[p for p in range(len(tokens)) if offsets[p]<right and offsets[p+1]>left]
        if not positions or min(positions)<max(0,len(tokens)-1-80) or max(positions)>=len(tokens)-1:
            raise ValueError('context expression outside observed window')
        encoded.append(tokens)
    # Price before creating the large output or launching the owned child.
    system=system_snapshot()
    budget=storage_budget(len(rows),len(json.dumps(corpus,ensure_ascii=False,indent=2).encode('utf-8'))+1)
    if (shutil.disk_usage(output.parent).free<(budget['fresh_free_bytes'] if not resume_from else 4*1024**3)
            or system['system_commit_available_bytes']<12*1024**3
            or system['physical_available_bytes']<8*1024**3):
        raise RuntimeError('offline contextual collection preflight; no model launched')
    records=[];chunks=[];samples=[]
    recovery={}
    if resume_from:
        if (type(resume_from) is not int or resume_from%64 or not 0<resume_from<len(rows) or prior_log is None
                or (output/'report.json').exists() or json.loads(_read(output/'corpus.json',limit=16*1024**2))!=corpus):
            raise ValueError('context resume boundary')
        log=_read(prior_log,limit=1024**2).decode('utf-8')
        if f'context captured {resume_from} / {len(rows)} ' not in log or 'offline system emergency floor' not in log:
            raise ValueError('context prior failure/committed boundary')
        states=NpyRows(output/'states.npy',True)
        valid=np.load(output/'valid.npy',allow_pickle=False,max_header_size=1024)
        if valid.shape!=(len(rows),80) or valid.dtype!=np.bool_:raise ValueError('context resume mask')
        for start in range(0,resume_from,64):
            prior=output/f'chunk-{start:05}';subset=encoded[start:start+64]
            expected=b'IGCPFX01'+struct.pack('<I',len(subset))+b''.join(struct.pack('<I',len(t))+np.array(t,dtype='<i4').tobytes() for t in subset)
            if (_read(prior/'prefixes.bin',limit=66*1024)!=expected or (prior/'captures').exists()
                    or 'PASS native prefix requests=64 window=8 geometry_layers=48 stage=[0,1)' not in _read(prior/'stdout.log',limit=16384).decode()):
                raise ValueError('context prior completed chunk identity')
            values=states[np.arange(start,start+64,dtype=np.int64)]
            for j,tokens in enumerate(subset):
                anchor=len(tokens)-1;mask=np.arange(80)>=80-min(anchor,80)
                if not np.array_equal(valid[start+j],mask) or not np.isfinite(values[j]).all() or np.any(values[j,~mask]):
                    raise ValueError('context persisted row validation')
                value_hash=hashlib.sha256(values[j].tobytes()).hexdigest()
                records.append(dict(id=rows[start+j]['id'],prefix_sha256=hashlib.sha256(np.array(tokens,dtype='<i4').tobytes()).hexdigest(),
                                    persisted_f32_sha256=value_hash,tokens=len(tokens),observed_rows=int(mask.sum()),
                                    provenance='validated persisted F32 from interrupted capture; original archive digest was not journaled'))
            chunks.append(dict(start=start,count=64,recovered=True))
        # Independently recapture both ends before accepting recovered rows.
        for start in (0,resume_from-64):
            probe=output/f'recovery-check-{start:05}';probe.mkdir(exist_ok=False)
            command=[str(executable),str(model),str(pack),str(output/f'chunk-{start:05}'/'prefixes.bin'),str(probe),'8']
            with (probe/'stdout.log').open('w') as so,(probe/'stderr.log').open('w') as se:
                result=subprocess.run(command,creationflags=0x08000000,stdout=so,stderr=se,timeout=90)
            if result.returncode:raise RuntimeError('context recovery native recapture failed')
            expected=states[np.arange(start,start+64,dtype=np.int64)]
            for j,tokens in enumerate(encoded[start:start+64]):
                actual,mask=decode_archive(_read(probe/f'{j}.attention.bin',limit=4*1024**2),tokens)
                if actual.tobytes()!=expected[j].tobytes() or not np.array_equal(mask,valid[start+j]):
                    raise ValueError('context recovered native feature mismatch')
        recovery=dict(rows=resume_from,prior_failed_log_sha256=digest(prior_log),recaptured_bitwise=128,
                      distinction='persisted rows content-hashed during recovery, not claimed to retain original archive digests')
        print('context recovered',resume_from,'rows;128 independently recaptured bitwise',flush=True)
    else:
        output.mkdir(exist_ok=False)
        (output/'corpus.json').write_text(json.dumps(corpus,ensure_ascii=False,indent=2),encoding='utf-8')
        states=NpyRows.create(output/'states.npy',len(rows))
        valid=np.zeros((len(rows),80),dtype=np.bool_)
    (output/'identity.json').write_text(json.dumps(dict(protocol_sha256=protocol_hash,source_manifest_sha256=source_hash,
        rows_sha256=corpus['rows_sha256'],exe_sha256=exe_hash,recovery=recovery,**manifest_binding),indent=2),encoding='utf-8')
    for start in range(resume_from,len(rows),64):
        if time.monotonic()-started>1200:raise TimeoutError('context collection deadline; incomplete cache')
        subset=encoded[start:start+64];chunk=output/(f'chunk-{start:05}' if not resume_from else f'chunk-{start:05}-resume');chunk.mkdir()
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
                    samples.append(dict(row=start,seconds=time.monotonic()-started,process=process,
                                        collector=process_snapshot(os.getpid()),system=current))
                    if time.monotonic()-tick>90:
                        (output/'memory.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
                        raise TimeoutError('context child deadline')
                    if current['system_commit_available_bytes']<2*1024**3 or current['physical_available_bytes']<1024**3:
                        (output/'memory.json').write_text(json.dumps(samples,indent=2),encoding='utf-8')
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
        states.flush();np.save(output/'valid.npy',valid)
        (output/'records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
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
    if ({p.name:digest(p) for p in adapter_paths}!=adapter_hashes
            or (corpus_manifest is not None and digest(corpus_manifest)!=manifest_file_hash)):
        raise RuntimeError('context input manifest/adapter changed during capture')
    verify_sources(sources)
    states.close();del states,valid
    (output/'records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    state_hash=digest(output/'states.npy')
    report=dict(success=True,kind='native first-attention F32 contextual cache, offline training only',rows=len(rows),
                rows_sha256=corpus['rows_sha256'],states_sha256=state_hash,valid_sha256=digest(output/'valid.npy'),
                protocol_sha256=protocol_hash,source_manifest_sha256=source_hash,exe_sha256=exe_hash,
                tokenizer_binding_sha256=tokenizer.binding_sha256,preflight=system,elapsed_seconds=time.monotonic()-started,
                chunks=chunks,recovery=recovery,source_resolution=verified,collection_source_sha256=digest(Path(__file__)),
                records_sha256=digest(output/'records.json'),cache_sha256=state_hash)
    report.update(manifest_binding,input_manifest_source_sha256=adapter_hashes,storage_budget=budget)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('context cache complete',report['states_sha256'],'seconds',report['elapsed_seconds'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('executable','model','pack','reference','sources','protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume-from',type=int,default=0)
    parser.add_argument('--prior-log',type=Path)
    parser.add_argument('--corpus-manifest',type=Path)
    args=parser.parse_args();prepare(**vars(args))
