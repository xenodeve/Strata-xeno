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


def pack_request_fixture(records,rows,model,tokenizer,artifact):
    if not 1<=len(records)<=64 or any(type(d) is not bytes or len(d)!=32 for d in (model,tokenizer,artifact)):
        raise ValueError('request fixture header bounds')
    output=bytearray(b'IGCOFX02'+struct.pack('<I',len(records))+model+tokenizer+artifact);seen=set()
    def integer(value):
        if type(value) is not int:raise ValueError('literal request integer type')
        digits=bytes(map(int,reversed(str(abs(value)))))
        if not 1<=len(digits)<=62:raise ValueError('literal request integer range')
        return bytes([len(digits),int(value<0)])+digits+bytes(62-len(digits))
    for record in records:
        ids=record['ids'];row=rows[record['sample']]
        if (not 2<=len(ids)<=262144 or any(type(t) is not int or not 0<=t<248320 for t in ids)
                or record['position']!=len(ids)-1 or record['invocation_position']!=len(ids)-1 or record['output_offset']!=0
                or type(record['invocation']) is not int or not 1<=record['invocation']<2**64
                or type(row['operation']) is not int or row['operation'] not in (1,2,3)):
            raise ValueError('typed request fixture boundary/domain')
        prefix=hashlib.sha256(np.asarray(ids,dtype='<i4').tobytes()).digest()
        if prefix in seen:raise ValueError('duplicate request fixture prefix')
        seen.add(prefix)
        output+=prefix+struct.pack('<IIIIQQ64s64s',1,1,row['operation'],0,record['invocation'],len(ids)-1,
                                  integer(row['lhs']),integer(row['rhs']))
    return bytes(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','reference','pack','sources','capture-protocol','artifact','model','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--runtime-requests',action='store_true')
    args=parser.parse_args()
    data=load_dataset(args.dataset,args.reference,args.pack,args.sources,args.capture_protocol)
    model=next(r['sha256'] for r in data['current_source_resolution'] if Path(r['resolved']).resolve()==args.model.resolve())
    selected=[r for r in data['records'] if r['output_offset']==0]
    binding=(bytes.fromhex(model),bytes.fromhex(data['tokenizer'].binding_sha256),hashlib.sha256(args.artifact.read_bytes()).digest())
    raw=(pack_request_fixture(selected,{r['id']:r for r in data['rows']},*binding) if args.runtime_requests else pack_fixture(selected,*binding))
    with args.output.open('xb') as target:target.write(raw)
    print(json.dumps(dict(entries=len(selected),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),runtime_requests=args.runtime_requests,
                         scope='controlled known operands; no host answer computation or I2 eligibility')))
