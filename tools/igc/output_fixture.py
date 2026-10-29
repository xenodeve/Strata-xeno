"""Serialize previously GPU-checked results for controlled native I3 tests only."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from tools.igc.output_data import load_dataset


def pack_fixture(records,model,tokenizer,artifact):
    if (not 1<=len(records)<=64 or any(type(d) is not bytes or len(d)!=32 for d in (model,tokenizer,artifact))):
        raise ValueError('fixture header bounds')
    output=bytearray(b'IGCOFX01'+struct.pack('<I',len(records))+model+tokenizer+artifact);seen=set()
    for r in records:
        ids=r['ids'];raw=bytes.fromhex(r['checked_result_hex'])
        if (not 2<=len(ids)<=262144 or any(type(t) is not int or not 0<=t<248320 for t in ids)
                or r['position']!=len(ids)-1 or r['invocation_position']!=len(ids)-1 or r['output_offset']!=0
                or len(raw)!=88):raise ValueError('fixture initial prefix/result boundary')
        version,status,invocation,position,value=struct.unpack('<IIQQ64s',raw)
        length,negative=value[:2];digits=value[2:]
        if ((version,status,invocation,position)!=(1,0,r['invocation'],len(ids)-1) or invocation==0
                or not 1<=length<=62 or negative not in (0,1) or any(d>9 for d in digits)
                or any(digits[length:]) or (length>1 and not digits[length-1])
                or (length==1 and digits[0]==0 and negative)):raise ValueError('fixture checked canonical result')
        prefix=hashlib.sha256(np.asarray(ids,dtype='<i4').tobytes()).digest()
        if prefix in seen:raise ValueError('duplicate fixture prefix')
        seen.add(prefix);output+=prefix+raw
    return bytes(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','reference','pack','sources','capture-protocol','artifact','model','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    data=load_dataset(args.dataset,args.reference,args.pack,args.sources,args.capture_protocol)
    model=next(r['sha256'] for r in data['current_source_resolution'] if Path(r['resolved']).resolve()==args.model.resolve())
    selected=[r for r in data['records'] if r['output_offset']==0]
    raw=pack_fixture(selected,bytes.fromhex(model),bytes.fromhex(data['tokenizer'].binding_sha256),hashlib.sha256(args.artifact.read_bytes()).digest())
    with args.output.open('xb') as target:target.write(raw)
    print(json.dumps(dict(entries=len(selected),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),scope='controlled known inputs only; no host arithmetic or I2 eligibility')))
