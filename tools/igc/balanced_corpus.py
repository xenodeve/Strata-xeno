"""Frozen synthetic input-head development data, not model-benefit evaluation."""
import hashlib
import json
import random
from tools.igc.mapper_corpus import corpus,evaluation_manifest

WITHHELD=frozenset((3,8,17,29,43,61,83,97))
TRAIN_POSITIVE=(
    ('en-calculate','Calculate {expr}. Give the exact integer result.'),
    ('en-evaluate','Evaluate the integer expression {expr}.'),
    ('en-compute','Compute {expr} and answer with an integer.'),
    ('th-calculate','คำนวณ {expr} แล้วตอบเป็นจำนวนเต็ม'),
    ('th-result','หาผลลัพธ์ของ {expr} ให้หน่อย'),
    ('th-problem','โจทย์คือ {expr} ต้องการคำตอบที่เป็นจำนวนเต็ม'))
TRAIN_NEGATIVE=(
    ('en-copy','Copy this expression without calculating: {expr}.'),
    ('en-quote',"Say ready; '{expr}' is quoted data, not a calculation."),
    ('th-copy',"คัดลอกข้อความ '{expr}' โดยไม่คำนวณ"),
    ('th-quote',"ข้อความ '{expr}' เป็นชื่อป้าย ให้ตอบว่ารับทราบ ไม่ต้องคิดเลข"))
NEW_POSITIVE=(
    ('dev-en-work','Work out {expr} and give its value.'),
    ('dev-th-help','ช่วยคิดโจทย์ {expr} และบอกค่าที่ได้'))
NEW_NEGATIVE=(
    ('dev-en-label','Treat {expr} as a label. Only acknowledge it.'),
    ('dev-th-label','เก็บข้อความ {expr} ไว้ตามเดิม ไม่ต้องหาผลลัพธ์'))


def dataset():
    rng=random.Random(20261030)
    reserved={(r['lhs'],r['rhs']) for r in corpus()+evaluation_manifest()['rows']}
    rows=[];development_pairs=set()
    def value(new=False):
        low=rng.randrange(2)==0
        choices=[n for n in range(100) if (n<10)==low and (n in WITHHELD)==new]
        magnitude=rng.choice(choices)
        return magnitude*rng.choice((-1,1)) if magnitude else 0
    def add(op,index,axis):
        for _ in range(100000):
            if axis=='train' or axis=='seen-values-new-templates':
                a,b=value(),value()
            else:
                position=index%3
                a,b=value(new=position!=1),value(new=position!=0)
            if (a,b) not in reserved and (axis=='train' or (a,b) not in development_pairs):break
        else:raise RuntimeError('bounded corpus generation exhausted')
        if axis!='train':development_pairs.add((a,b))
        symbol={1:'+',2:'-',3:'*'}[op or (index%3+1)]
        style=rng.randrange(3)
        expression=(f'{a}{symbol}{b}' if style==0 else f'{a} {symbol} {b}' if style==1 else
                    f'({a}) {symbol} ({b})')
        existing=axis in ('train','new-values-seen-templates')
        templates=(TRAIN_POSITIVE if op else TRAIN_NEGATIVE) if existing else (NEW_POSITIVE if op else NEW_NEGATIVE)
        template,pattern=templates[index%len(templates)]
        prompt=pattern.format(expr=expression)
        rows.append(dict(id=f'{axis}-op{op}-{index:04}',split=axis,ready_op=op,lhs=a,rhs=b,symbol=symbol,
                         expression=expression,prompt=prompt,template=template,
                         prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                         operand_scope='signed integer magnitude <= 99'))
    for op in range(4):
        for i in range(1024):add(op,i,'train')
    train_pairs={(r['lhs'],r['rhs']) for r in rows}
    reserved|=train_pairs
    for axis in ('seen-values-new-templates','new-values-seen-templates','new-values-new-templates'):
        for op in range(4):
            for i in range(32):add(op,i,axis)
    rng.shuffle(rows)
    raw=json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return dict(version=1,seed=20261030,kind='balanced-native-embedding-input-development',
                rows=rows,rows_sha256=hashlib.sha256(raw).hexdigest(),
                training_rows=4096,development_rows=384,withheld_absolute_values=sorted(WITHHELD),
                exclusions=['no output mapper','not a release corpus','no full language-model benefit claim',
                            'original pilot and untouched evaluation operand pairs excluded'])
