"""Row-aligned cached evidence only. Never changes a controller or its inputs."""
from collections import Counter,defaultdict
import math
import struct
import numpy as np
from tools.igc.native_qualification import selection,check_record,RECORD_BYTES
from tools.igc.learned_integration import parse_audit


def _truth(row):
    a,b=row['lhs'],row['rhs'];op=row['ready_op']
    if any(type(v) is not int for v in (a,b,op)) or abs(a)>99 or abs(b)>99 or op not in range(4):
        raise ValueError('diagnostic label domain')
    return [op,int(a<0),int(b<0),int(abs(a)>9),int(abs(b)>9),abs(a)%10,abs(a)//10,abs(b)%10,abs(b)//10]


def _summary(rows,name):
    counts=Counter();partition=dict.fromkeys(('selected_correct','readiness_only','embedding_only','both','attention_error','other'),0)
    oracle=dict.fromkeys(('readiness_only','embedding_only','both'),0)
    by_template=defaultdict(Counter);by_language=defaultdict(Counter)
    for row in rows:
        r=row[name];truth=row['truth'];positive=truth[0]!=0
        group=by_template[row['template']];language=by_language[row['language']]
        for c in (counts,group,language):
            c['positive' if positive else 'negative']+=1
            c['selected']+=r['selection']==0
        if not positive:
            counts['negative_calls']+=r['selection']==0
            continue
        correct=r['selection']==0 and r['embedding_tuple']==truth
        for c in (counts,group,language):
            c['selected_correct']+=correct;c['readiness_bypass']+=not row['readiness']['decision']
            c['disagreement_after_readiness']+=r['selection']==4
        counts['embedding_exact']+=r['embedding_tuple']==truth
        counts['attention_exact']+=r['attention_tuple']==truth
        if r['attention_tuple']!=truth:key='attention_error'
        elif correct:key='selected_correct'
        elif not row['readiness']['decision']:key='readiness_only' if r['embedding_tuple']==truth else 'both'
        elif r['embedding_tuple']!=truth:key='embedding_only'
        else:key='other'
        partition[key]+=1
        # Positive-only oracle substitutions are explanatory bounds, not policy.
        for label,primary,ready in [('readiness_only',r['embedding_tuple'],1),
                                    ('embedding_only',truth,row['readiness']['decision']),('both',truth,1)]:
            code,request=selection(primary,r['attention_tuple'],ready)
            oracle[label]+=code==0 and request==[truth[0],row['lhs'],row['rhs']]
    return dict(**counts,positive_partition=partition,positive_oracle_counterfactual=oracle,
                by_template=dict(by_template),by_language=dict(by_language))


def diagnose(rows,references,original_native,candidate_native,readiness_logits):
    count=len(rows)
    if (not 1<=count<=64 or len({r['id'] for r in rows})!=count
            or not isinstance(readiness_logits,np.ndarray) or readiness_logits.dtype!=np.float32
            or readiness_logits.shape!=(count,2) or not np.isfinite(readiness_logits).all()):
        raise ValueError('bounded row identity/readiness logits')
    if any(len(references.get(name,[]))!=count for name in ('retained','candidate')):
        raise ValueError('reference count')
    for raw in (original_native,candidate_native):
        if type(raw) is not bytes or len(raw)!=12+count*RECORD_BYTES or raw[:12]!=b'IGCLPO01'+struct.pack('<I',count):
            raise ValueError('native audit file header/length')
    joined=[];changed=dict(attention_tuple=[],selection=[],native_result=[])
    for i,row in enumerate(rows):
        truth=_truth(row);logits=[float(x) for x in readiness_logits[i]];ready=int(logits[1]>logits[0]);margin=logits[1]-logits[0]
        e=math.exp(-abs(margin));probability=1/(1+e) if margin>=0 else e/(1+e)
        item=dict(index=i,id=row['id'],split=row['split'],template=row.get('template','unlabelled'),
                  language=row.get('language') or ('th' if any('\u0e00'<=c<='\u0e7f' for c in row['prompt']) else 'en'),
                  prompt=row['prompt'],lhs=row['lhs'],rhs=row['rhs'],truth=truth,
                  readiness=dict(logits=logits,margin=margin,decision=ready,softmax_compute=probability,
                                 score_scope='shared unchanged native readiness replay; uncalibrated, not a new threshold'))
        for name,refname,raw in [('original','retained',original_native),('candidate','candidate',candidate_native)]:
            r=references[refname][i]
            if r['id']!=row['id'] or r['index']!=i or r['readiness']!=ready:
                raise ValueError('row/score identity mismatch')
            code,request=selection(r['primary'],r['secondary'],ready)
            if (code,request)!=(r['selection'],r['request']):raise ValueError('recorded selection mismatch')
            record=raw[12+i*RECORD_BYTES:12+(i+1)*RECORD_BYTES]
            check_record(record,r['owner'],r['anchor'],code,request)
            audit=parse_audit(record[:288])
            item[name]=dict(embedding_tuple=r['primary'],attention_tuple=r['secondary'],
                consensus=r['primary']==r['secondary'],selection=code,request=request,
                native_result=audit['result'],native_audit=audit,
                readiness_logits=logits,readiness_decision=ready)
        a,b=item['original'],item['candidate']
        if a['attention_tuple']!=b['attention_tuple']:changed['attention_tuple'].append(row['id'])
        if a['selection']!=b['selection']:changed['selection'].append(row['id'])
        if a['native_result']!=b['native_result']:changed['native_result'].append(row['id'])
        joined.append(item)
    return dict(rows=joined,summaries={name:_summary(joined,name) for name in ('original','candidate')},
                changed_rows=changed,scope='reused inspected coherent capture; oracle bounds are offline diagnosis only')
