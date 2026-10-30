"""Offline row-bound coherent scorecards. No model, fitting, routing or GPU work."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import stat
from tools.igc.coherent_audit import audit_archive
from tools.igc.coherent_observation import _digest, _json_bytes, _exclusive
from tools.igc.observation import _read
from tools.igc.native_qualification import selection, check_record, RECORD_BYTES
from tools.igc.learned_integration import parse_audit

FIELDS=('operator','left_sign','right_sign','left_length','right_length','left_ones','left_tens','right_ones','right_tens')


def _expression(row):
    op=row.get('operation')
    a,b=row.get('lhs'),row.get('rhs')
    if type(op) is not int or op not in (1,2,3) or any(type(v) is not int or abs(v)>99 for v in (a,b)):
        raise ValueError('scorecard supported operand labels')
    return [op,int(a<0),int(b<0),int(abs(a)>9),int(abs(b)>9),abs(a)%10,abs(a)//10,abs(b)%10,abs(b)//10]


def _truth(row):
    op=row.get('ready_op')
    if type(op) is not int or op not in range(4):raise ValueError('scorecard readiness label')
    if not op:return None
    result=_expression(row)
    if result[0]!=op:raise ValueError('scorecard operation label')
    return result


def _head_counts(counts,confusions,fields,truth):
    if truth is None:
        counts['negative']+=1;counts['negative_operator_no_call']+=fields[0]==0
        return
    counts['positive']+=1;counts['positive_exact']+=fields==truth
    active_error=False;padding_error=False
    for i,name in enumerate(FIELDS):
        counts[name+'_correct']+=fields[i]==truth[i]
        padding=(i==6 and not truth[3]) or (i==8 and not truth[4])
        if i>=5:
            width=truth[3 if i in (5,6) else 4]+1
            confusions[(name,truth[i],fields[i],width,padding)]+=1
        if padding:
            counts['padding_slots']+=1;counts['padding_correct']+=fields[i]==0
            padding_error|=fields[i]!=0
        else:
            active_error|=fields[i]!=truth[i]
            if i>=5:
                counts['active_digit_slots']+=1;counts['active_digits_correct']+=fields[i]==truth[i]
    for label,index in [('left',(1,3,5,6)),('right',(2,4,7,8))]:
        counts[label+'_operand_exact']+=all(fields[i]==truth[i] for i in index)
    if padding_error and not active_error:counts['padding_only_error']+=1
    if active_error and not padding_error:counts['active_only_error']+=1
    if active_error and padding_error:counts['active_and_padding_error']+=1


def _bootstrap(groups,clusters):
    units=defaultdict(list)
    for group,ok in groups.items():units[clusters[group]].append(int(ok))
    if not units:return dict(units=0,complete_group_rate_interval95=None)
    rows=list(units.values());rng=random.Random(20261101);samples=[]
    for _ in range(512):
        selected=[rows[rng.randrange(len(rows))] for _ in rows]
        samples.append(sum(sum(v) for v in selected)/sum(len(v) for v in selected))
    samples.sort()
    return dict(units=len(rows),groups=len(groups),seed=20261101,resamples=512,
        complete_groups=sum(groups.values()),complete_group_rate_interval95=[samples[12],samples[499]],
        scope='structural-cluster development bootstrap; shared values can still correlate errors')


def evaluate(manifest,audit,binding,records,expected_controller=None):
    """Consume original-anchor replay rows exactly once; labels stay offline."""
    if (not isinstance(manifest,dict) or manifest.get('version')!=2
            or manifest.get('kind')!='coherent-development-manifest' or manifest.get('acceptance64_access') is not False
            or audit.get('kind')!='coherent_archive_audit' or audit.get('version')!=1
            or manifest.get('plan_sha256')!=audit.get('plan_sha256')):
        raise ValueError('scorecard manifest/archive binding')
    required={'version','kind','plan_sha256','complete_sha256','model_sha256','tokenizer_sha256','controller_sha256','manifest_sha256'}
    if (not isinstance(binding,dict) or set(binding)!=required or type(binding['version']) is not int or binding['version']!=1
            or binding['kind']!='coherent-native-replay'):
        raise ValueError('scorecard replay header')
    for k in required-{'version','kind'}:
        _digest(binding[k])
        if k=='manifest_sha256':
            if binding[k]!=hashlib.sha256(_json_bytes(manifest)).hexdigest():raise ValueError('scorecard frozen manifest identity')
        elif k!='controller_sha256' and binding[k]!=audit[k]:raise ValueError('scorecard replay identity')
    if binding['controller_sha256']!=(expected_controller or audit.get('sources',{}).get('bundle')):
        raise ValueError('scorecard frozen controller identity')
    rows=manifest.get('rows');requests=manifest.get('requests');clusters=manifest.get('clusters')
    if (not isinstance(rows,list) or not 1<=len(rows)<=2048 or manifest.get('anchors')!=len(rows)
            or len(rows)!=audit['anchors'] or not isinstance(requests,list) or len(requests)!=audit['rows']
            or not isinstance(clusters,list) or not 1<=len(clusters)<=48
            or any(type(v) is not int or not 0<=v<len(clusters) for v in clusters)
            or manifest.get('structural_clusters')!=len(set(clusters))
            or len({r.get('id') for r in rows})!=len(rows)):
        raise ValueError('scorecard cohort geometry')
    for row in rows:
        if (not isinstance(row.get('id'),str) or not 1<=len(row['id'])<=128
                or row.get('language') not in ('en','th') or not isinstance(row.get('intent'),str)
                or not 1<=len(row['intent'])<=80 or (row.get('group') is not None and
                    (type(row['group']) is not int or not 0<=row['group']<len(clusters)))):
            raise ValueError('scorecard group/condition identity')
    positives=sum(_truth(r) is not None for r in rows)
    if (manifest.get('positive'),manifest.get('negative'))!=(positives,len(rows)-positives):
        raise ValueError('scorecard cohort labels')
    output_nll=0.;output_top1=0;output_count=0;sample_ok={}
    for i,(q,receipt) in enumerate(zip(requests,audit['row_receipts'],strict=True)):
        if (q.get('ordinal')!=i or any(q.get(k)!=receipt[k] for k in ('role','base','step'))
                or q.get('sample')!=rows[receipt['base']]['id']):raise ValueError('scorecard request lineage')
        target=q.get('target');head=receipt['head']
        if target!=head.get('target') or (receipt['role']=='output' and target is None):
            raise ValueError('scorecard target alignment')
        if target is not None:
            if type(target) is not int or not 0<=target<248320:raise ValueError('scorecard target type')
            good=head['argmax']==target
            output_nll+=head['logsumexp']-head['target_logit'];output_top1+=good;output_count+=1
            sample_ok[receipt['base']]=sample_ok.get(receipt['base'],True) and good
    counts=Counter();readouts={'primary':Counter(),'secondary':Counter()};by=defaultdict(Counter)
    confusions={'primary':Counter(),'secondary':Counter()}
    groups={};joined=[];padding=Counter();record_iter=iter(records)
    partition=dict.fromkeys(('selected_correct','readiness_only','primary_only','secondary_only','both_readouts','readiness_and_readout','other'),0)
    for i,row in enumerate(rows):
        try:prediction=next(record_iter)
        except StopIteration as error:raise ValueError('scorecard missing replay row') from error
        expected=audit['row_receipts'][i]
        if (not isinstance(prediction,dict) or set(prediction)!={'ordinal','owner','receipt_sha256','primary','secondary','readiness_logits','record_hex'}
                or type(prediction['ordinal']) is not int or prediction['ordinal']!=i
                or type(prediction['owner']) is not int or prediction['owner']!=expected['owner']
                or prediction['receipt_sha256']!=expected['receipt_sha256'] or expected['role']!='anchor'):
            raise ValueError('scorecard replay row identity/order')
        logits=prediction['readiness_logits']
        if (not isinstance(logits,list) or len(logits)!=2
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in logits)):
            raise ValueError('scorecard finite readiness logits')
        ready=int(logits[1]>logits[0]);primary=prediction['primary'];secondary=prediction['secondary']
        choice,request=selection(primary,secondary,ready)
        primary_invalid=selection(primary,primary,1)[0]==5
        secondary_invalid=selection(secondary,secondary,1)[0]==5
        raw=prediction['record_hex']
        if not isinstance(raw,str) or not re.fullmatch('[0-9a-f]{'+str(2*RECORD_BYTES)+'}',raw):
            raise ValueError('scorecard native record encoding')
        raw=bytes.fromhex(raw)
        check_record(raw,prediction['owner'],expected['prefix_tokens']-1,choice,request)
        native=parse_audit(raw[:288]);truth=_truth(row);positive=truth is not None
        selected=choice==0;correct=selected and positive and primary==truth
        metrics=dict(rows=1,positive=int(positive),negative=int(not positive),
            readiness_correct=int(ready==int(positive)),selected_correct=int(correct),
            positive_no_call=int(positive and not selected),negative_calls=int(not positive and selected),
            negative_readiness_false_positive=int(not positive and ready),
            selected_wrong=int(selected and not correct),raw_disagreement=int(primary!=secondary),
            raw_primary_invalid_encoding=int(primary_invalid),raw_secondary_invalid_encoding=int(secondary_invalid),
            routed_disagreement=int(choice==4),routed_invalid_encoding=int(choice==5))
        counts.update(metrics)
        for name,fields in [('primary',primary),('secondary',secondary)]:_head_counts(readouts[name],confusions[name],fields,truth)
        conditions=[('language',row['language']),('intent',row['intent'])]
        if positive or row.get('group') is not None:
            expression=_expression(row)
            conditions.extend([('width',f'{expression[3]+1}/{expression[4]+1}'),
                ('sign',f'{expression[1]}/{expression[2]}'),('operation',str(expression[0]))])
        else:conditions.append(('expression','unclassified ordinary/unsupported'))
        for axis,value in conditions:by[axis+':'+value].update(metrics)
        ok=(ready==int(positive) and (correct and secondary==truth if positive else not selected and primary[0]==secondary[0]==0))
        if positive:
            if correct:key='selected_correct'
            elif not ready:key='readiness_only' if primary==secondary==truth else 'readiness_and_readout'
            elif primary!=truth and secondary==truth:key='primary_only'
            elif primary==truth and secondary!=truth:key='secondary_only'
            elif primary!=truth and secondary!=truth:key='both_readouts'
            else:key='other'
            partition[key]+=1
        if row.get('group') is not None:groups[row['group']]=groups.get(row['group'],True) and ok
        composed=[]
        for fields in (primary,secondary):
            fields=fields.copy()
            if fields[3]==0:fields[6]=0
            if fields[4]==0:fields[8]=0
            composed.append(fields)
        pc,pr=selection(*composed,ready)
        padding['selected_correct']+=pc==0 and positive and composed[0]==truth
        padding['negative_calls']+=pc==0 and not positive
        joined.append(dict(id=row['id'],ordinal=i,owner=expected['owner'],group=row.get('group'),
            truth=truth,primary=primary,secondary=secondary,readiness_logits=logits,selection=choice,
            native_result=native['result'],selected_correct=correct,all_input_conditions_pass=ok))
    try:next(record_iter)
    except StopIteration:pass
    else:raise ValueError('scorecard extra replay row')
    passed=(counts['readiness_correct']==len(rows) and counts['selected_correct']==positives and counts['negative_calls']==0
            and all(r['positive_exact']==positives and r['negative_operator_no_call']==len(rows)-positives for r in readouts.values()))
    return dict(version=1,kind='coherent_development_scorecard',binding=binding,counts=dict(counts),
        readouts={k:{**dict(v),'digit_confusions':[dict(slot=key[0],expected=key[1],predicted=key[2],width=key[3],padding=key[4],count=count)
                    for key,count in sorted(confusions[k].items())]} for k,v in readouts.items()},
        by_condition={k:dict(v) for k,v in sorted(by.items())},
        input_gates_pass=passed,positive_partition=partition,
        full_protocol_geometry=(len(rows)==400 and positives==192 and len(groups)==48 and len(set(clusters))==44),
        live_or_acceptance_qualified=False,archive_native_qualified=audit['native_qualified'],rows=joined,
        cluster_bootstrap=_bootstrap(groups,clusters),padding_composition_counterfactual=dict(padding),
        counterfactual_scope='predicted-length zeroing only; unchanged readiness; offline representation variant, not raw scores or deployed policy',
        hard_off_output=dict(positions=output_count,target_top1=output_top1,mean_target_nll=output_nll/output_count if output_count else None,
            samples=len(sample_ok),all_positions_top1=sum(sample_ok.values()),scope='teacher-forced hard-off token statistics; not free generation'),
        output_controls={'learned':'not supplied','oracle_owned':'not supplied'},
        scope='inspected development replay; no fresh inference, acceptance, training, promotion or latency claim')


def _ledger_info(path):
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400 or info.st_size>80*1024**2:
        raise ValueError('scorecard bounded replay ledger')
    return info


def ledger_rows(path):
    _ledger_info(path)
    with path.open('rb') as stream:
        header=stream.readline(4097)
        if len(header)>4096 or not header.endswith(b'\n'):raise ValueError('scorecard ledger header bound')
        yield json.loads(header)
        for _ in range(2049):
            raw=stream.readline(32769)
            if not raw:return
            if len(raw)>32768 or not raw.endswith(b'\n'):raise ValueError('scorecard ledger row bound')
            yield json.loads(raw)
        raise ValueError('scorecard ledger count bound')


def evaluate_files(archive,manifest_path,replay,output,allow_synthetic=False,expected_controller=None):
    if output.exists():raise FileExistsError('scorecard output already exists')
    if output.resolve().is_relative_to(archive.resolve()):raise ValueError('scorecard output must be outside the input archive')
    audit=audit_archive(archive,require_native=not allow_synthetic)
    manifest_raw=_read(manifest_path,limit=2*1024**2);manifest=json.loads(manifest_raw)
    before=_ledger_info(replay)
    with replay.open('rb') as stream:replay_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    records=ledger_rows(replay)
    try:binding=next(records)
    except StopIteration as error:raise ValueError('scorecard missing ledger header') from error
    report=evaluate(manifest,audit,binding,records,expected_controller)
    after=_ledger_info(replay)
    with replay.open('rb') as stream:final_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if ((before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or final_sha!=replay_sha
            or _read(manifest_path,limit=2*1024**2)!=manifest_raw):
        raise ValueError('scorecard input changed during evaluation')
    report['inputs']=dict(manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),replay_sha256=replay_sha)
    _exclusive(output,_json_bytes(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('archive','manifest','replay','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--allow-synthetic',action='store_true')
    parser.add_argument('--expected-controller',help='score a later controller on an archive collected under another bundle')
    args=parser.parse_args()
    report=evaluate_files(args.archive,args.manifest,args.replay,args.output,args.allow_synthetic,args.expected_controller)
    print(json.dumps(report['counts']))


if __name__=='__main__':main()
