"""Bounded diagnostic-only neural weights; no code, pickle or serving activation."""
import argparse
import hashlib
import json
import inspect
from pathlib import Path
import re
import struct
import numpy as np
import torch
from tools.igc.observation import _read

HEADER=struct.Struct('<8s12IQ32s32s32s32s32s32s')
DESCRIPTOR=struct.Struct('<II4IQQ')
MAX_BYTES=8*1024**2
assert HEADER.size==256 and DESCRIPTOR.size==40


def _digest(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError('adapter binding digest')
    return bytes.fromhex(value)


def schema(kind):
    if type(kind) is not int:raise ValueError('unknown neural adapter kind')
    # Derivation must not perturb a caller's training RNG.
    with torch.random.fork_rng(devices=[]):
        if kind==1:
            from tools.igc.attention_readout import InputReadout
            model=InputReadout(use_final_query=False,contextual=True);encoder=model.encoder
            if (model.norm.eps!=1e-5 or model.position.num_embeddings!=80 or model.stream.out_features!=16
                    or encoder is None or not encoder.norm_first or encoder.self_attn.embed_dim!=64
                    or encoder.self_attn.num_heads!=4 or not encoder.self_attn.batch_first
                    or encoder.self_attn.dropout!=0 or encoder.dropout.p!=0
                    or any(module.p!=0 for module in encoder.modules() if isinstance(module,torch.nn.Dropout))
                    or encoder.norm1.eps!=1e-5 or encoder.norm2.eps!=1e-5 or encoder.activation_relu_or_gelu!=2):
                raise ValueError('input architecture semantics changed; new ABI required')
            semantics=dict(name='query-free-contextual-hc-input-v1',width=2560,hc=4,window=80,
                           per_stream=16,encoder_width=64,encoder_heads=4,feedforward=128,
                           activation='gelu-exact',norm_first=True,layernorm_eps=1e-5,dropout=0,
                           attention_scale=8,output_sizes=[4,2,2,2,2,10,10,10,10],max_digits=2)
        elif kind==2:
            from tools.igc.output_mapping import ResultMapper
            model=ResultMapper()
            if model.norm.eps!=1e-5 or model.query.out_features!=128 or model.position.num_embeddings!=62:
                raise ValueError('output architecture semantics changed; new ABI required')
            semantics=dict(name='late-gated-integer-result-v1',width=2560,result_digits=62,
                           features=12,attention_width=128,attention_scale='sqrt(128)',
                           layernorm_eps=1e-5,gate='sigmoid',non_success='hard-bypass',max_digits=62)
        elif kind==3:
            from tools.igc.readiness_mapping import ReadinessMapper
            model=ReadinessMapper()
            semantics=dict(name='final-state-binary-readiness-v1',width=2560,hidden=128,classes=2,
                           activation='gelu-exact',normalization='fit-only-mean-scale',min_scale=0.0001,
                           decision='argmax-tie-to-zero',max_digits=2)
        else:raise ValueError('unknown neural adapter kind')
        tensors=[dict(name=name,shape=list(value.shape)) for name,value in model.state_dict().items()]
    record=dict(kind=kind,semantics=semantics,tensors=tensors,
                first_party_class_sha256=hashlib.sha256(inspect.getsource(type(model)).encode('utf-8')).hexdigest())
    canonical=json.dumps(record,sort_keys=True,separators=(',',':')).encode('utf-8')
    return {**record,'sha256':hashlib.sha256(canonical).hexdigest()}


def pack_state(kind,weights,model_sha256,tokenizer_sha256,state_sha256,protocol_sha256):
    definition=schema(kind);expected=definition['tensors']
    if not isinstance(weights,dict) or set(weights)!={r['name'] for r in expected}:
        raise ValueError('adapter named tensor set')
    descriptors=bytearray();payload=bytearray()
    for index,record in enumerate(expected):
        tensor=weights[record['name']];shape=record['shape']
        if (not isinstance(tensor,torch.Tensor) or tensor.dtype!=torch.float32 or tensor.device.type!='cpu'
                or list(tensor.shape)!=shape or not bool(torch.isfinite(tensor).all())):
            raise ValueError('adapter tensor dtype/shape/finite values')
        if kind==3 and record['name']=='scale' and not bool((tensor>=0.0001).all()):
            raise ValueError('readiness normalization scale')
        payload+=bytes((-len(payload))%16);offset=len(payload)
        data=tensor.detach().contiguous().numpy().astype('<f4',copy=False).tobytes()
        dimensions=shape+[0]*(4-len(shape))
        descriptors+=DESCRIPTOR.pack(index,len(shape),*dimensions,offset,tensor.numel())
        payload+=data
    header=HEADER.pack(b'STIGCM01',1,kind,1,len(expected),2560,4,80 if kind==1 else 0,
                       62 if kind==2 else 2,1,1,7,0,len(payload),
                       _digest(model_sha256),_digest(tokenizer_sha256),_digest(state_sha256),
                       _digest(protocol_sha256),_digest(definition['sha256']),bytes(32))
    body=header+descriptors;body+=bytes((-len(body))%16);body+=payload
    raw=body+hashlib.sha256(body).digest()
    if len(raw)>MAX_BYTES:raise ValueError('adapter file bound')
    return raw


def read_artifact(raw,model_sha256,tokenizer_sha256,artifact_sha256,production=False):
    expected_model,expected_tokenizer,expected_file=map(_digest,(model_sha256,tokenizer_sha256,artifact_sha256))
    if (type(raw) is not bytes or not HEADER.size+32<=len(raw)<=MAX_BYTES
            or hashlib.sha256(raw).digest()!=expected_file or hashlib.sha256(raw[:-32]).digest()!=raw[-32:]):
        raise ValueError('adapter file bytes/digest')
    fields=HEADER.unpack_from(raw)
    (magic,version,kind,flags,count,width,hc,window,max_digits,abi,domain,operations,reserved,
     payload_bytes,model,tokenizer,state,protocol,schema_hash,tail_reserved)=fields
    if (magic!=b'STIGCM01' or version!=1 or kind not in (1,2,3) or flags!=1 or not 1<=count<=128
            or (width,hc,window,max_digits,abi,domain,operations,reserved)!=(2560,4,80 if kind==1 else 0,62 if kind==2 else 2,1,1,7,0)
            or tail_reserved!=bytes(32) or model!=expected_model or tokenizer!=expected_tokenizer):
        raise ValueError('adapter header/model/numerical compatibility')
    if production:raise ValueError('diagnostic-only adapter cannot be enabled in production')
    definition=schema(kind)
    if schema_hash!=bytes.fromhex(definition['sha256']) or count!=len(definition['tensors']):
        raise ValueError('adapter schema identity')
    descriptor_end=HEADER.size+count*DESCRIPTOR.size;start=(descriptor_end+15)&~15
    if (len(raw)!=start+payload_bytes+32 or any(raw[descriptor_end:start])):
        raise ValueError('adapter payload size/alignment')
    cursor=0;result={}
    for index,record in enumerate(definition['tensors']):
        slot,rank,*rest=DESCRIPTOR.unpack_from(raw,HEADER.size+index*DESCRIPTOR.size)
        dimensions=rest[:4];offset,elements=rest[4:]
        shape=record['shape'];aligned=(cursor+15)&~15
        if (slot!=index or rank!=len(shape) or dimensions!=shape+[0]*(4-len(shape))
                or elements!=int(np.prod(shape)) or offset!=aligned or offset+elements*4>payload_bytes
                or any(raw[start+cursor:start+offset])):
            raise ValueError('adapter tensor schema/offset')
        value=np.frombuffer(raw,dtype='<f4',count=elements,offset=start+offset).reshape(shape)
        if not np.isfinite(value).all():raise ValueError('adapter tensor nonfinite')
        result[record['name']]=value;cursor=offset+elements*4
    if cursor!=payload_bytes:raise ValueError('adapter unused payload bytes')
    if kind==3 and not np.all(result['scale']>=np.float32(.0001)):raise ValueError('readiness normalization scale')
    return dict(kind=kind,diagnostic_only=True,max_digits=max_digits,model_sha256=model.hex(),
                tokenizer_sha256=tokenizer.hex(),state_sha256=state.hex(),protocol_sha256=protocol.hex(),
                schema_sha256=schema_hash.hex(),tensors=result)


def export(state,output,kind,model_sha256,tokenizer_sha256,protocol_sha256):
    from safetensors import safe_open
    raw=_read(state,limit=MAX_BYTES);definition=schema(kind);expected={r['name']:r['shape'] for r in definition['tensors']}
    with safe_open(str(state),framework='pt',device='cpu') as source:
        if set(source.keys())!=set(expected):raise ValueError('adapter saved tensor set')
        for name,shape in expected.items():
            if source.get_slice(name).get_shape()!=shape:raise ValueError('adapter saved tensor shape')
        tensors={name:source.get_tensor(name) for name in expected}
    if _read(state,limit=MAX_BYTES)!=raw:raise ValueError('adapter state changed during export')
    artifact=pack_state(kind,tensors,model_sha256,tokenizer_sha256,hashlib.sha256(raw).hexdigest(),protocol_sha256)
    read_artifact(artifact,model_sha256,tokenizer_sha256,hashlib.sha256(artifact).hexdigest())
    with output.open('xb') as target:target.write(artifact)
    return dict(bytes=len(artifact),sha256=hashlib.sha256(artifact).hexdigest(),diagnostic_only=True,
                schema_sha256=definition['sha256'])


def read_file(path,model_sha256,tokenizer_sha256,artifact_sha256,production=False):
    return read_artifact(_read(Path(path),limit=MAX_BYTES),model_sha256,tokenizer_sha256,
                         artifact_sha256,production=production)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',type=int,choices=(1,2,3),required=True)
    for name in ('state','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('model-sha256','tokenizer-sha256','protocol-sha256'):parser.add_argument('--'+name,required=True)
    print(json.dumps(export(**vars(parser.parse_args())),indent=2))
