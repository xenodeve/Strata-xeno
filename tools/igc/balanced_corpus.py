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
EXPANDED_POSITIVE=(
    ('aug-en-solve','Solve this numeric problem: {expr}.'),
    ('aug-en-find','Find the value of {expr}.'),
    ('aug-en-answer','What do you get for {expr}? Answer with the integer.'),
    ('aug-en-arithmetic','Do the arithmetic in {expr}.'),
    ('aug-th-value','อยากรู้ว่า {expr} ได้ค่าเท่าไร'),
    ('aug-th-solve','ช่วยหาคำตอบของ {expr}'),
    ('aug-th-number','คิดเลข {expr} แล้วบอกคำตอบมา'),
    ('aug-th-evaluate','ขอผลคำนวณ {expr} เป็นตัวเลข'))
EXPANDED_NEGATIVE=(
    ('aug-en-label',"This is a label, '{expr}'. Reply OK without solving it."),
    ('aug-en-file',"Use '{expr}' as a filename. Do not evaluate it."),
    ('aug-en-verbatim',"Write the string '{expr}' verbatim."),
    ('aug-en-preserve',"Do not execute '{expr}'; preserve its characters."),
    ('aug-en-characters',"I am asking about characters in '{expr}', not arithmetic."),
    ('aug-en-data',"Read '{expr}' as data. No calculation is wanted."),
    ('aug-en-receipt',"A numeric label is '{expr}'. Acknowledge receipt."),
    ('aug-en-description',"The text '{expr}' appears in a quote. Describe the text, not its result."),
    ('aug-th-string',"อ่าน '{expr}' เป็นข้อความ แล้วคัดลอกตามนั้น"),
    ('aug-th-file',"ตั้งชื่อไฟล์ว่า '{expr}' ไม่ต้องคำนวณ"),
    ('aug-th-preserve',"เก็บตัวอักษรใน '{expr}' ให้เหมือนเดิม"),
    ('aug-th-data',"นี่คือข้อมูล '{expr}' ไม่ได้ขอให้คิดเลข"))


def _dataset(per_class=1024,seed=20261030,positive=TRAIN_POSITIVE,negative=TRAIN_NEGATIVE,extra_reserved=()):
    rng=random.Random(seed)
    reserved={(r['lhs'],r['rhs']) for r in corpus()+evaluation_manifest()['rows']}
    reserved.update(extra_reserved)
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
        templates=(positive if op else negative) if existing else (NEW_POSITIVE if op else NEW_NEGATIVE)
        template,pattern=templates[index%len(templates)]
        prompt=pattern.format(expr=expression)
        rows.append(dict(id=f'{axis}-op{op}-{index:04}',split=axis,ready_op=op,lhs=a,rhs=b,symbol=symbol,
                         expression=expression,prompt=prompt,template=template,
                         prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                         operand_scope='signed integer magnitude <= 99'))
    for op in range(4):
        for i in range(per_class):add(op,i,'train')
    train_pairs={(r['lhs'],r['rhs']) for r in rows}
    reserved|=train_pairs
    for axis in ('seen-values-new-templates','new-values-seen-templates','new-values-new-templates'):
        for op in range(4):
            for i in range(32):add(op,i,axis)
    rng.shuffle(rows)
    raw=json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return dict(version=1,seed=seed,kind='balanced-native-embedding-input-development',
                rows=rows,rows_sha256=hashlib.sha256(raw).hexdigest(),
                training_rows=per_class*4,development_rows=384,withheld_absolute_values=sorted(WITHHELD),
                exclusions=['no output mapper','not a release corpus','no full language-model benefit claim',
                            'original pilot and untouched evaluation operand pairs excluded'])


def dataset():
    return _dataset()


def expanded_dataset():
    old_development={(r['lhs'],r['rhs']) for r in dataset()['rows'] if r['split']!='train'}
    result=_dataset(2560,20261031,TRAIN_POSITIVE+EXPANDED_POSITIVE,TRAIN_NEGATIVE+EXPANDED_NEGATIVE,old_development)
    result.update(version=2,kind='expanded-native-embedding-input-development',
                  development_status='template-withheld from fitting; templates known to developer from v1 diagnostics')
    return result
