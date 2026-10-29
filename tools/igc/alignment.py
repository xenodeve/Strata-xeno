"""Offline annotation/attention diagnosis; token IDs are labels, never NN features."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import torch
from tools.igc.attention_readout import InputReadout
from tools.igc.observation import _read
from tools.igc.output_examples import wrapped_prefix
from tools.igc.embedding_readout import load_cache,TensorInputs
from tools.igc.context_cache import load_cache as load_context,ContextInputs
from tools.igc.readout import SIZES
from tools.qwenputer.tokenizer import NativeText


def digit_positions(row,tokens,tokenizer):
    raw=tokenizer.decode_bytes(tokens);expression=row['expression'].encode('utf-8')
    if raw.count(expression)!=1:raise ValueError('ambiguous expression annotation')
    start=raw.index(expression)
    patterns=(rb'\((-?[0-9]+)\) ([+*-]) \((-?[0-9]+)\)',rb'(-?[0-9]+) ([+*-]) (-?[0-9]+)',rb'(-?[0-9]+)([+*-])(-?[0-9]+)')
    match=next((m for pattern in patterns if (m:=re.fullmatch(pattern,expression)) is not None),None)
    if match is None or [int(match.group(i)) for i in (1,3)]!=[row['lhs'],row['rhs']]:
        raise ValueError('operand annotation/label mismatch')
    if row.get('symbol',match.group(2).decode())!=match.group(2).decode():raise ValueError('operator annotation mismatch')
    offsets=[];cursor=0
    for token in tokens:
        piece=tokenizer.decode_bytes([token]);offsets.append((cursor,cursor+len(piece),piece));cursor+=len(piece)
    if cursor!=len(raw):raise ValueError('native token byte composition')
    result=[];anchor=len(tokens)-1
    for group in (1,3):
        first=match.start(group)+(expression[match.start(group):match.start(group)+1]==b'-');last=match.end(group)-1
        for byte in (last,first if last>first else None):
            if byte is None:result.append(-1);continue
            position=start+byte
            selected=[i for i,(a,b,piece) in enumerate(offsets) if a==position and b==position+1 and piece in tuple(bytes([d]) for d in range(48,58))]
            if len(selected)!=1:raise ValueError('numeric token is not one isolated digit')
            index=selected[0]-anchor+80
            if not 0<=index<80:raise ValueError('operand outside observed pre-anchor window')
            result.append(index)
    return result


def diagnose(embedding,context,embedding_state,context_state,sources,embedding_protocol,context_protocol,reference,pack,output):
    from safetensors.torch import load_file
    started=time.monotonic();output.mkdir(exist_ok=False);source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tokenizer=NativeText.from_pack(pack/'tokenizer');prefix,suffix=wrapped_prefix(reference,tokenizer)
    table,indices,mask,rows,report=load_cache(embedding,sources,embedding_protocol)
    encoded=[tokenizer.encode((prefix+r['prompt'].encode('utf-8')+suffix).decode('utf-8')) for r in rows]
    expected_tokens=np.array(sorted({t for ids in encoded for t in ids[-81:-1]}),dtype='<i4')
    actual_tokens=np.frombuffer(_read(embedding/'native-token-ids.bin',limit=4096),dtype='<i4')
    if not np.array_equal(expected_tokens,actual_tokens):raise ValueError('native token table metadata differs from independent prefix reconstruction')
    mapping={int(t):i for i,t in enumerate(expected_tokens)}
    for i,ids in enumerate(encoded):
        ids=ids[-81:-1];expected=np.zeros(80,dtype=np.int32);valid=np.zeros(80,dtype=np.bool_)
        expected[-len(ids):]=[mapping[t] for t in ids];valid[-len(ids):]=True
        if not np.array_equal(expected,indices[i]) or not np.array_equal(valid,mask[i]):raise ValueError('cache index/prefix annotation mismatch')
    labels=np.array([digit_positions(r,ids,tokenizer) for r,ids in zip(rows,encoded,strict=True)],dtype=np.int64)
    print('alignment labels independently reconstructed',len(labels),'rows',flush=True)
    states,context_mask,context_rows,context_report=load_context(context,sources,context_protocol)
    if context_rows!=rows or not np.array_equal(context_mask,mask):raise ValueError('feature arm corpus/window mismatch')
    torch.set_num_threads(4);torch.manual_seed(20261031)
    selections=np.array([i for i,r in enumerate(rows) if r['split']!='train' and r['ready_op']>0],dtype=np.int64)
    reports={};examples={}
    for name,state,inputs in [('embedding',embedding_state,TensorInputs(table,indices,mask,'cpu')),
                              ('first_attention',context_state,ContextInputs(states,context_mask,'cpu'))]:
        model=InputReadout(False,True);model.load_state_dict(load_file(str(state)));model.eval()
        captured=[];hook=model.key.register_forward_hook(lambda _m,_i,o:captured.append(o))
        all_predictions=[];details=[]
        with torch.no_grad():
            for offset in range(0,len(selections),32):
                selected=selections[offset:offset+32];captured.clear();batch=inputs.batch(selected)
                heads=model(*batch).split(SIZES,dim=1)
                predicted=np.stack([h.argmax(1).numpy() for h in heads],axis=1);all_predictions.extend(predicted)
                scores=torch.bmm(model.queries[None,:,:].expand(len(selected),-1,-1),captured[0].transpose(1,2))/8
                weights=torch.softmax(scores.masked_fill(~batch[1][:,None,:],float('-inf')),dim=-1).numpy()
                for j,index in enumerate(selected):
                    fields=[]
                    for slot,head in enumerate((5,6,7,8)):
                        truth=int(labels[index,slot]);guessed=int(weights[j,head].argmax())
                        fields.append(dict(head=head,target_position=truth,argmax_position=guessed,
                            target_mass=float(weights[j,head,truth]) if truth>=0 else None,
                            top_positions=np.argsort(weights[j,head])[-3:][::-1].tolist()))
                    details.append(dict(id=rows[index]['id'],split=rows[index]['split'],prediction=predicted[j].tolist(),fields=fields))
        hook.remove();summaries={}
        for split in dict.fromkeys(r['split'] for r in details):
            selected=[r for r in details if r['split']==split];fields=[f for r in selected for f in r['fields'] if f['target_position']>=0]
            summaries[split]=dict(examples=len(selected),active_digit_queries=len(fields),
                argmax_on_annotated_digit=sum(f['argmax_position']==f['target_position'] for f in fields),
                mean_annotated_mass=sum(f['target_mass'] for f in fields)/len(fields))
        saved={r['id']:r['predicted'] for r in json.loads((state.parent/'predictions.json').read_text())}
        differences=[r['id'] for r in details if r['prediction']!=saved[r['id']]]
        examples[name]=details
        reports[name]=dict(state_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),summaries=summaries,
            cpu_vs_saved_cuda_prediction_differences=differences,
            caveat='attention localization alone is not causal proof; contextual values can redistribute information')
    states.close()
    result=dict(kind='offline known-development digit-position attention diagnosis; no training',reports=reports,
        cache_sha256=report['cache_sha256'],context_sha256=context_report['cache_sha256'],rows_sha256=report['rows_sha256'],
        tokenizer_binding_sha256=tokenizer.binding_sha256,labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
        seconds=time.monotonic()-started,fresh_acceptance_consumed=False,source_sha256=source_hash,
        torch_version=torch.__version__,device='cpu',threads=4,
        scope='CPU diagnostic replay of saved CUDA weights; token IDs only bind/annotate frozen activation data')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=source_hash:raise RuntimeError('alignment diagnostic source changed')
    (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (output/'details.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('embedding','context','embedding-state','context-state','sources','embedding-protocol','context-protocol','reference','pack','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    diagnose(**vars(parser.parse_args()))
