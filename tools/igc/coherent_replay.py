"""Native replay ledger for a complete coherent archive; existing CUDA probes only, no model, fitting or routing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
from tools.igc.coherent_audit import audit_archive
from tools.igc.coherent_observation import load_input_observation,_json_bytes,_exclusive,_digest
from tools.igc.native_qualification import decisions,RECORD_BYTES
from tools.igc.observation import _read

# The pipeline probe needs one selected row per run; the last 16 anchors are ordinary negatives, so 60 keeps them with others.
CHUNK=60


def file_digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def input_chunk(path,rows,plane):
    with path.open('xb') as f:
        f.write(b'IGCIIN01'+struct.pack('<I',len(rows)))
        for r in rows:f.write(r[plane].tobytes())
        for r in rows:f.write(bytes(r['mask']))


def readiness_chunk(path,rows):
    with path.open('xb') as f:
        f.write(b'IGCRIN01'+struct.pack('<I',len(rows)))
        for r in rows:f.write(r['mixed'].tobytes())


def pipeline_chunk(path,rows,model,tokenizer):
    with path.open('xb') as f:
        f.write(b'IGCLPF01'+struct.pack('<I',len(rows))+bytes.fromhex(model)+bytes.fromhex(tokenizer))
        for r in rows:
            f.write(struct.pack('<QQ',r['owner'],len(r['tokens'])-1)+r['embedding'].tobytes()+r['attention'].tobytes()
                    +bytes(r['mask'])+r['mixed'].tobytes())


def run(command,log):
    with log.open('ab') as stream:
        code=subprocess.run(command,stdout=stream,stderr=stream,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).returncode
    if code:raise RuntimeError(f'native replay probe failed: {Path(command[0]).name} exit {code}')


def read_logits(prefix,count,width):
    logits=np.frombuffer(_read(Path(str(prefix)+'.logits.bin'),size=count*width*4,limit=64*1024**2),dtype='<f4').reshape(count,width)
    status=np.frombuffer(_read(Path(str(prefix)+'.status.bin'),size=count*4,limit=4096),dtype='<u4')
    if status.any() or not np.isfinite(logits).all():raise ValueError('native replay head status/finite')
    return logits


def read_records(path,count):
    raw=_read(path,size=12+count*RECORD_BYTES,limit=4*1024**2)
    if raw[:8]!=b'IGCLPO01' or struct.unpack_from('<I',raw,8)[0]!=count:raise ValueError('native replay output header')
    return [raw[12+i*RECORD_BYTES:12+(i+1)*RECORD_BYTES] for i in range(count)]


def compose(rows,primary,secondary,readiness,records,receipts):
    """One ledger row per anchor; every list is ordered by anchor ordinal."""
    if not len(rows)==len(primary)==len(secondary)==len(readiness)==len(records)==len(receipts):raise ValueError('native replay row counts')
    return [dict(ordinal=r['ordinal'],owner=r['owner'],receipt_sha256=receipt,primary=[int(v) for v in a],secondary=[int(v) for v in b],
                 readiness_logits=[float(v) for v in logits],record_hex=record.hex())
            for r,a,b,logits,record,receipt in zip(rows,decisions(np.asarray(primary)),decisions(np.asarray(secondary)),readiness,records,receipts,strict=True)]


def replay(archive,manifest,bundle,probes,output):
    if output.exists():raise FileExistsError('native replay output exists')
    output.mkdir(parents=True)
    audit=audit_archive(archive,require_native=True)
    bundle_file=bundle/'controller.bin'
    controller=file_digest(bundle_file)
    single_readout=struct.unpack_from('<I',_read(bundle_file,size=204),8)[0]==2
    manifest=json.loads(_read(manifest,limit=2*1024**2))
    binding=dict(version=1,kind='coherent-native-replay',plan_sha256=audit['plan_sha256'],complete_sha256=audit['complete_sha256'],
                 model_sha256=audit['model_sha256'],tokenizer_sha256=audit['tokenizer_sha256'],controller_sha256=controller,
                 manifest_sha256=hashlib.sha256(_json_bytes(manifest)).hexdigest())
    for key in binding:
        if key.endswith('_sha256'):_digest(binding[key])
    # The manifest's receipt hashes pair each row with the archive it came from.
    receipts=[r['receipt_sha256'] for r in audit['row_receipts'][:audit['anchors']]]
    artifacts={name:bundle/f'{name}.bin' for name in ('input-embedding','input-attention','readiness')}
    identity=lambda name:[str(artifacts[name]),binding['model_sha256'],binding['tokenizer_sha256'],file_digest(artifacts[name])]
    log=output/'probes.log';lines=[_json_bytes(binding)]
    for start in range(0,audit['anchors'],CHUNK):
        rows=[load_input_observation(archive,i) for i in range(start,min(start+CHUNK,audit['anchors']))]
        n=len(rows);base=output/f'chunk-{start}'
        files=[Path(f'{base}.{name}.in') for name in ('embedding','attention','readiness','pipeline')]
        input_chunk(files[0],rows,'embedding');input_chunk(files[1],rows,'attention')
        readiness_chunk(files[2],rows);pipeline_chunk(files[3],rows,binding['model_sha256'],binding['tokenizer_sha256'])
        run([str(probes/'igc_input_probe.exe'),*identity('input-embedding'),str(files[0]),f'{base}.embedding','normal'],log)
        run([str(probes/'igc_input_probe.exe'),*identity('input-attention'),str(files[1]),f'{base}.attention','normal'],log)
        run([str(probes/'igc_readiness_probe.exe'),*identity('readiness'),str(files[2]),f'{base}.readiness','normal'],log)
        run([str(probes/'igc_pipeline_probe.exe'),str(bundle_file),str(files[3]),f'{base}.records'],log)
        for row in compose(rows,read_logits(f'{base}.attention' if single_readout else f'{base}.embedding',n,52),read_logits(f'{base}.attention',n,52),
                           read_logits(f'{base}.readiness',n,2),read_records(Path(f'{base}.records'),n),receipts[start:start+n]):
            lines.append(_json_bytes(row))
        for path in files:path.unlink()
    _exclusive(output/'replay.jsonl',b''.join(lines))
    return output/'replay.jsonl'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('archive','manifest','bundle','probes','output'):parser.add_argument('--'+key,type=Path,required=True)
    print(replay(**vars(parser.parse_args())))


if __name__=='__main__':main()
