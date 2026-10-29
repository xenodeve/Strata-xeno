"""Frozen, small signed-integer readout experiment, not a release evaluation set.

The deployed model remains frozen. Template groups are split before activation
collection; no held-out fit or general latent-capability claim is permitted.
"""
import hashlib
import json
import random


def corpus():
    rng=random.Random(20261029)
    positives=[('en-calculate','Calculate {expr}. Give the exact integer result.'),
               ('th-calculate','คำนวณ {expr} แล้วตอบเป็นจำนวนเต็ม'),
               ('en-evaluate','Evaluate the integer expression {expr}.')]
    negatives=[('en-copy','Copy this expression without calculating: {expr}.'),
               ('th-quote',"ข้อความที่ยกมาคือ '{expr}' ไม่ต้องคำนวณ ให้ตอบว่าอ่านแล้ว"),
               ('en-string',"Is '{expr}' a string containing numbers? Do not solve it.")]
    rows=[]
    for op in range(4):
        for i in range(16):
            heldout=i>=12
            symbol={1:'+',2:'-',3:'*'}[op or (i%3+1)]
            lhs,rhs=rng.randrange(-99,100),rng.randrange(-99,100)
            expression=f'{lhs}{symbol}{rhs}'
            if heldout:
                template,pattern=(('en-exact-result','What is the exact result of {expr}?') if op else
                                  ('en-mention',"Mention the characters in '{expr}'; no arithmetic is requested."))
            else:
                template,pattern=(positives if op else negatives)[i%3]
            prompt=pattern.format(expr=expression)
            row=dict(id=f'op{op}-{i:02}',split='heldout' if heldout else 'train',
                     ready_op=op,lhs=lhs,rhs=rhs,symbol=symbol,expression=expression,
                     prompt=prompt,template=template,operand_scope='signed integer magnitude <= 99')
            row['prompt_sha256']=hashlib.sha256(prompt.encode('utf-8')).hexdigest()
            rows.append(row)
    rng.shuffle(rows)
    return rows


def manifest():
    rows=corpus()
    canonical=json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return dict(version=1,seed=20261029,kind='first-window-readout-feasibility',
                rows_sha256=hashlib.sha256(canonical).hexdigest(),rows=rows,
                split_contract='48 train / 16 template-heldout, balanced readiness/operator classes',
                seam='head_mixed_after_final_hc; one fixed answer-start observation per request',
                exclusions=['no trigger during reasoning','no attention over pre-anchor states',
                            'no output mapper','no frozen-model backward','not a release-quality dataset'])


def evaluation_manifest():
    """New untouched evaluation; the earlier sixteen cases are now diagnostic."""
    rng=random.Random(20261102)
    used={(r['lhs'],r['rhs']) for r in corpus()}
    rows=[]
    for op in range(4):
        for i in range(16):
            while True:
                lhs,rhs=rng.randrange(-99,100),rng.randrange(-99,100)
                if (lhs,rhs) not in used:
                    used.add((lhs,rhs));break
            symbol={1:'+',2:'-',3:'*'}[op or (i%3+1)]
            expression=f'({lhs}) {symbol} ({rhs})'
            thai=i%2==1
            if op:
                prompt=(f'หาค่าจำนวนเต็มของ {expression} แล้วตอบผลที่แน่นอน' if thai else
                        f'Return the integer value for {expression}.')
                template='th-integer-value' if thai else 'en-integer-value'
            else:
                prompt=(f'อ่านข้อความนี้แล้วตอบว่ารับทราบ ไม่ต้องหาค่า: "{expression}"' if thai else
                        f'Acknowledge this quoted expression without evaluating it: "{expression}".')
                template='th-acknowledge' if thai else 'en-acknowledge'
            rows.append(dict(id=f'e-op{op}-{i:02}',split='evaluation',ready_op=op,lhs=lhs,rhs=rhs,
                             symbol=symbol,expression=expression,prompt=prompt,template=template,
                             operand_scope='signed integer magnitude <= 99',
                             prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest()))
    rng.shuffle(rows)
    canonical=json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return dict(version=1,seed=20261102,kind='new-unseen-preanchor-evaluation',rows=rows,
                rows_sha256=hashlib.sha256(canonical).hexdigest(),
                split_contract='64 evaluation-only; never optimizer data; disjoint operand pairs and templates',
                exclusions=['not 62-digit range coverage','not dynamic reasoning triggers','not model answer accuracy'])
