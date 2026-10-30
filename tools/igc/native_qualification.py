"""Offline bounded selection/result oracle; never called by native inference.

Python arithmetic here checks results *after* native execution. It is not an
executor, fallback, source of mapper inputs, or production policy implementation.
"""
import struct
import numpy as np

SIZES=(4,2,2,2,2,10,10,10,10)
RECORD_BYTES=288+4+2560*4


def decisions(logits):
    if (not isinstance(logits,np.ndarray) or logits.ndim!=2 or logits.shape[1]!=52
            or not 1<=len(logits)<=64 or not np.isfinite(logits).all()):
        raise ValueError('finite bounded qualification logits')
    return np.stack([x.argmax(1) for x in np.split(logits,np.cumsum(SIZES)[:-1],axis=1)],axis=1)


def selection(primary,secondary,ready):
    for fields in (primary,secondary):
        if (not isinstance(fields,list) or len(fields)!=9
                or any(type(v) is not int or not 0<=v<n for v,n in zip(fields,SIZES))):
            raise ValueError('bounded categorical fields')
    if type(ready) is not int or ready not in (0,1):raise ValueError('binary readiness')
    if not ready:return 1,None
    if primary!=secondary:return 4,None
    if not primary[0]:return 1,None
    values=[]
    for sign,length,ones,tens in ((primary[1],primary[3],primary[5],primary[6]),
                                  (primary[2],primary[4],primary[7],primary[8])):
        if (not length and tens) or (length and not tens) or (sign and not length and not ones):
            return 5,None
        values.append((ones+10*tens)*(-1 if sign else 1))
    return 0,[primary[0],*values]


def _integer(value):
    digits=bytes(int(c) for c in reversed(str(abs(value))))
    if len(digits)>62:raise ValueError('bounded oracle integer')
    return bytes([len(digits),int(value<0)])+digits+bytes(62-len(digits))


def expected_records(owner,anchor,choice,request):
    if (type(owner) is not int or not 1<=owner<2**64 or type(anchor) is not int
            or not 0<=anchor<262144 or type(choice) is not int or choice not in (0,1,4,5)):
        raise ValueError('qualification owner/selection bounds')
    if choice:
        if request is not None:raise ValueError('inactive selection has no operands')
        return (struct.pack('<IIIIQQ',1,0,0,0,owner,anchor)+bytes(128),
                struct.pack('<IIQQ',1,4,owner,anchor)+bytes(64))
    if (not isinstance(request,list) or len(request)!=3 or any(type(v) is not int for v in request)
            or request[0] not in (1,2,3) or any(abs(v)>99 for v in request[1:])):
        raise ValueError('initial two-digit mapper qualification scope')
    op,a,b=request
    value=a+b if op==1 else a-b if op==2 else a*b
    return (struct.pack('<IIIIQQ',1,1,op,0,owner,anchor)+_integer(a)+_integer(b),
            struct.pack('<IIQQ',1,0,owner,anchor)+_integer(value))


def check_record(raw,owner,anchor,choice,request):
    if type(raw) is not bytes or len(raw)!=RECORD_BYTES:raise ValueError('native record bytes')
    expected_request,expected_result=expected_records(owner,anchor,choice,request)
    if (struct.unpack_from('<QQ6I',raw)!=(owner,anchor,0,1,int(choice==0),
            1 if choice==0 else 2 if choice==1 else 3,choice,0)
            or raw[40:200]!=expected_request or raw[200:288]!=expected_result
            or struct.unpack_from('<I',raw,288)[0]!=(0 if choice==0 else 1)):
        raise ValueError('native provenance/typed request/result/status mismatch')
    output=np.frombuffer(raw,dtype='<f4',count=2560,offset=292)
    if not np.isfinite(output).all():raise ValueError('nonfinite native mapped output')
    return output
