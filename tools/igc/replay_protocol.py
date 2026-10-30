"""Prepare identity-bound replay references/schedule only; never load or train a model."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
from tools.igc.input_manifest import validate_manifest
from tools.igc.matched_corpus import digest
from tools.igc.observation import _read

OLD_SHA='0aa4fa0c34e45b47f213279a638d7d11d1a04f0256877ddcf5e8be6c803452d9'
MATCHED_SHA='ce1231adffa9bb4cd1a68074295811ec09d0dbc4a52f7cf59447620256027db3'
SELECTION_SHA='87189b7633a05d1503748fba8aec2baede2980882224db50c5cfe79f053ae8b1'
IDS_SHA='5e63f56e919c677f4dd88a56ceec1df80df90772f1e0e35ae386bfafaa8a064b'
INITIAL_SHA='519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2'
OPERAND=r'(?:\(([+-]?[0-9]{1,2})\)|([+-]?[0-9]{1,2}))'
EXPRESSION=re.compile(r'\s*'+OPERAND+r'\s*([+*-])\s*'+OPERAND+r'\s*')


def exclusion_keys(row):
    a,b=row['lhs'],row['rhs'];symbol=row.get('symbol') or {1:'+',2:'-',3:'*'}.get(row.get('operation'))
    if (any(type(x) is not int or abs(x)>99 for x in (a,b)) or symbol not in ('+','-','*')
            or type(row['ready_op']) is not int or row['ready_op'] not in (0,{'+':1,'-':2,'*':3}[symbol])):
        raise ValueError('replay bounded expression domain')
    match=EXPRESSION.fullmatch(row['expression'])
    if not match or (int(match[1] or match[2]),match[3],int(match[4] or match[5]))!=(a,symbol,b):
        raise ValueError('expression parse/metadata mismatch')
    fact=(('add',*sorted((a,b if symbol=='+' else -b))) if symbol in ('+','-') else
          ('mul',*sorted((abs(a),abs(b))),int(a!=0 and b!=0 and ((a<0)!=(b<0)))))
    return dict(pair=(a,b),symmetric_pair=tuple(sorted((a,b))),expression=(a,symbol,b),fact=fact,
        prompt=unicodedata.normalize('NFKC',' '.join(row['prompt'].split())).casefold())


def filter_training(rows,evaluation):
    forbidden=defaultdict(set)
    for row in evaluation:
        for key,value in exclusion_keys(row).items():forbidden[key].add(value)
    chosen=[];excluded={}
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate source row identity')
    for row in rows:
        reasons=(['not_train'] if row['split']!='train' else [])
        reasons += [key for key,value in exclusion_keys(row).items() if value in forbidden[key]]
        if reasons:excluded[row['id']]=sorted(reasons)
        else:chosen.append(row)
    return chosen,excluded


class Cycle:
    def __init__(self,indices,seed):
        if not indices or len(set(indices))!=len(indices):raise ValueError('nonempty distinct stream')
        self.indices=list(indices);self.rng=random.Random(seed);self.order=[];self.cursor=0
    def take(self,count):
        result=[]
        while len(result)<count:
            if self.cursor==len(self.order):
                self.order=self.indices.copy();self.rng.shuffle(self.order);self.cursor=0
            n=min(count-len(result),len(self.order)-self.cursor)
            result.extend(self.order[self.cursor:self.cursor+n]);self.cursor+=n
        return result


def schedule(matched,old,updates=840):
    if type(updates) is not int or not 1<=updates<=840 or len(matched)%32 or set(old)!={0,1,2,3}:
        raise ValueError('bounded fixed replay schedule')
    m=Cycle(matched,20261114);streams={op:Cycle(old[op],20261115+op) for op in range(4)}
    return [m.take(32)+sum((streams[op].take(8) for op in range(4)),[]) for _ in range(updates)]


def validate_schedule(batches,members,matched,old,updates=840):
    if batches!=schedule(matched,old,updates):raise ValueError('frozen mixing/shuffle schedule mismatch')
    if (len({(r['source'],r['id']) for r in members})!=len(members)
            or any(set(r)!={'source','index','id'} or r['source'] not in ('old','matched')
                   or type(r['index']) is not int or r['index']<0 for r in members)):
        raise ValueError('bounded cache references, never neural features')
    matched_set=set(matched)
    for batch in batches:
        if (len(batch)!=64 or any(type(i) is not int or not 0<=i<len(members) for i in batch)
                or any(i not in matched_set or members[i]['source']!='matched' for i in batch[:32])):
            raise ValueError('replay batch identity/source')
        for op in range(4):
            if any(i not in old[op] or members[i]['source']!='old' for i in batch[32+8*op:40+8*op]):
                raise ValueError('old replay class/source')
    return True


def presentation_histogram(counts):
    # JSON object keys must round-trip without changing types.
    return {str(k):v for k,v in sorted(Counter(counts).items())}


def build(old,matched,legacy_ids):
    if len(legacy_ids)!=343 or len(set(legacy_ids))!=343:raise ValueError('343 unique legacy rows required')
    byid={r['id']:r for r in old};legacy=[byid[i] for i in legacy_ids]
    development=[r for r in matched if r['split']=='development']
    if len(development)!=768 or any(r['split']=='train' for r in legacy):raise ValueError('evaluation split')
    evaluation=legacy+development
    oldtrain=[r for r in old if r['split']=='train'];matchtrain=[r for r in matched if r['split']=='train']
    eligible_old,oldexcluded=filter_training(oldtrain,evaluation)
    eligible_mat,matchexcluded=filter_training(matchtrain,evaluation)
    groups=defaultdict(list)
    for row in eligible_mat:groups[(row['lhs'],row['rhs'],row['operation'])].append(row)
    selected=[];selection_rng=random.Random(20261113);group_counts={}
    for op in (1,2,3):
        choices=sorted(key for key in groups if key[2]==op)
        if len(choices)<56:raise ValueError('insufficient nonoverlapping paired matched groups')
        selection_rng.shuffle(choices);keep=set(choices[:56]);group_counts[str(op)]=len(choices)
        for key in choices:
            conditions={(r['language'],r['intent']) for r in groups[key]}
            if len(groups[key])!=4 or conditions!={('en','compute'),('th','compute'),('en','quote-copy'),('th','quote-copy')}:
                raise ValueError('complete paired matched groups required')
            if key in keep:selected.extend(groups[key])
    selected_ids={r['id'] for r in selected}
    selected=[r for r in matchtrain if r['id'] in selected_ids]
    for side in ('lhs','rhs'):
        if not {-8,-3,3,8}<={r[side] for r in selected}:raise ValueError('critical signed-slot coverage')
    for row in eligible_mat:
        if row['id'] not in selected_ids:matchexcluded[row['id']]=['balanced_group_subsample']
    old_index={r['id']:i for i,r in enumerate(old)};mat_index={r['id']:i for i,r in enumerate(matched)}
    members=[dict(source='old',index=old_index[r['id']],id=r['id']) for r in eligible_old]
    members += [dict(source='matched',index=mat_index[r['id']],id=r['id']) for r in selected]
    mi=list(range(len(eligible_old),len(members)));oi={op:[i for i,r in enumerate(eligible_old) if r['ready_op']==op] for op in range(4)}
    batches=schedule(mi,oi);validate_schedule(batches,members,mi,oi)
    if len(selected)!=672:raise ValueError('fixed matched subset')
    used=Counter(i for batch in batches for i in batch)
    if {used[i] for i in mi}!={40}:raise ValueError('fixed40 matched presentations')
    return dict(version=1,kind='igc-retention-replay-preparation-only',initial_artifact_sha256=INITIAL_SHA,
        members=members,batches=batches,schedule_sha256=digest(batches),member_sha256=digest(members),
        evaluation_ids_sha256=digest(dict(legacy=legacy_ids,matched=[r['id'] for r in development])),
        exclusions=dict(old=oldexcluded,matched=matchexcluded),eligible_old_count=len(eligible_old),
        eligible_matched_before_balance=len(eligible_mat),matched_selected=len(selected),
        old_classes={str(k):len(v) for k,v in oi.items()},matched_groups_available=group_counts,
        presentation_counts=dict(old=26880,matched=26880,matched_each=40,old_each_histogram=presentation_histogram(used[i] for ids in oi.values() for i in ids)),
        recipe=dict(seed=20261031,selection_seed=20261113,matched_shuffle_seed=20261114,old_shuffle_seeds={str(k):20261115+k for k in range(4)},
            updates=840,batch=64,matched_per_batch=32,old_per_operator_per_batch=8,lr=.001,weight_decay=.01,schedule='cosine',final_lr=.00001,final_checkpoint_only=True),
        scope='preparation only; cached row locators are not model features; evaluation labels never enter batches')


def prepared_payload(old_path,matched_path,selection_path):
    paths=[Path(old_path),Path(matched_path),Path(selection_path)];raw=[_read(p,limit=8*1024**2) for p in paths]
    if [hashlib.sha256(b).hexdigest() for b in raw]!=[OLD_SHA,MATCHED_SHA,SELECTION_SHA]:raise ValueError('unrecognized frozen replay inputs')
    old,matched,selection=map(json.loads,raw);validate_manifest(old);validate_manifest(matched)
    ids=selection['regression_ids']
    if hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()!=IDS_SHA:raise ValueError('legacy ordered identity')
    result=build(old['rows'],matched['rows'],ids)
    if (result['eligible_old_count'],result['eligible_matched_before_balance'])!=(9844,680):raise ValueError('frozen semantic exclusion counts')
    result['input_sha256']={p.name+':'+str(i):hashlib.sha256(b).hexdigest() for i,(p,b) in enumerate(zip(paths,raw,strict=True))}
    result['preparer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['expected_cache_identity']={
        'old':dict(states='3d1e27038d0c7dd7cc440dd876266237869595fbd20a701750670601db735806',
            mask='d8c8f95e1891c204d4d67006eee86b8ee0cbcdb93123aceb5784c27dde51f2cc',
            report='b62b3eaca9c3f172d37859a0167ae9377ae071820a34474df68e4c31a3bb3482',
            protocol='0a078202ccba84943916ec2965a32b68e76ab86c3060621c6e2335f36876a4b8'),
        'matched':dict(states='6757cde83254fff23010d26d720bc579880e2854750a4ee163b5224b8e12d82a',
            mask='d96bed0c391bb9bdfd544c23a9615823229920ecc714903087d32add325f74a0',
            report='7a3e5a5b9178df601a14a3bc7e4af4592ca45ccd3f3a5cfb82bdcc2f3b85fecd',
            protocol='2de4d752b3df50a3623624f9acbc95e874b80d0acd02fdc3ee458a679cf75e0c')}
    result['cache_validation_scope']='expected identities only; future consumer must verify actual state/mask/header bytes'
    result['manifest_sha256']=digest(result)
    return result


def verify_prepared(path,old_path,matched_path,selection_path,expected_sha256):
    raw=_read(Path(path),limit=8*1024**2)
    if not isinstance(expected_sha256,str) or re.fullmatch('[0-9a-f]{64}',expected_sha256) is None or hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('reviewed replay file identity')
    value=json.loads(raw)
    if value!=prepared_payload(old_path,matched_path,selection_path):
        raise ValueError('replay source/exclusion/schedule/provenance mismatch')
    return value


def prepare(old_path,matched_path,selection_path,output):
    result=prepared_payload(old_path,matched_path,selection_path)
    data=(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode()
    if len(data)>8*1024**2:raise ValueError('bounded replay metadata output')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'replay.json').write_bytes(data)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old','matched','selection','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();value=prepare(a.old,a.matched,a.selection,a.output)
    print(json.dumps({k:value[k] for k in ['eligible_old_count','eligible_matched_before_balance','matched_selected','old_classes','schedule_sha256','manifest_sha256']}))


if __name__=='__main__':main()
