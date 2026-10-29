"""Frozen matched-intent readiness data; labels are never inference features."""
import hashlib
import json
import random
from tools.igc.balanced_corpus import dataset as small_input, expanded_dataset
from tools.igc.mapper_corpus import corpus as old_input, evaluation_manifest
from tools.igc.output_corpus import dataset as old_output


def dataset():
    rng=random.Random(20261107)
    reserved={(r['lhs'],r['rhs']) for r in (small_input()['rows']+expanded_dataset()['rows']
              +old_input()+evaluation_manifest()['rows']+old_output()['rows'])}
    templates={
        'train':[
            ('en-calculate','Calculate {expr}. Give the exact integer result.',
             'en-copy','Copy the characters in {expr} exactly. Do not calculate them.'),
            ('th-calculate','คำนวณ {expr} แล้วตอบผลลัพธ์เป็นจำนวนเต็ม',
             'th-quote','คัดลอกข้อความนี้เท่านั้น ห้ามคำนวณ: "{expr}"'),
            ('en-evaluate','Evaluate the integer expression {expr}.',
             'en-quoted','The following is quoted text, not a calculation request: "{expr}". Acknowledge it.'),
            ('th-solve','ช่วยหาคำตอบของ {expr} ให้หน่อย',
             'th-literal','ให้พิมพ์ {expr} ตามเดิม ไม่ต้องหาคำตอบ')],
        'development':[
            ('en-value','What integer does {expr} evaluate to?',
             'en-no-eval','Please reproduce "{expr}" verbatim, without evaluating the expression.'),
            ('th-value','ผลของ {expr} เท่ากับเท่าไร ตอบเป็นเลขจำนวนเต็ม',
             'th-no-eval','อ่านข้อความ "{expr}" แล้วบอกว่ารับทราบ ไม่ต้องแก้โจทย์')],
    }
    ordinary={
        'train':[
            'Reply with READY.','Write one sentence about a blue sky.',
            'Quote the number 42 without doing arithmetic.','Say hello in Thai.',
            'Name a fruit.','Is the character + punctuation?',
            'Tell me whether multiplication uses the symbol *.','Copy the string "123".',
            'ตอบว่าพร้อม','บอกชื่อผลไม้หนึ่งชนิด','คัดลอกเลข 17 ตามเดิม',
            'ทักทายเป็นภาษาไทย','What is 17 / 3?','Calculate sqrt(81).',
            'หาค่า sin(30 องศา)','Solve x + 3 = 10.'],
        'development':[
            'Answer only OK.','Give a short greeting.',
            'Repeat the literal text 98.','Describe a green leaf.',
            'Name one animal.','Mention the minus sign without using it.',
            'Is "a*b" a string?','Copy 567 without evaluating anything.',
            'ตอบเพียงว่ารับทราบ','บอกชื่อสัตว์หนึ่งชนิด','พิมพ์เลข 29 ตามตัวอย่าง',
            'อธิบายสีของใบไม้สั้นๆ','Find the quotient 29 / 7.',
            'Evaluate log(100).','คำนวณ cos(60 องศา)','Find x when 2*x = 18.'],
    }
    rows=[]
    for split,count in [('train',32),('development',16)]:
        for op,symbol in [(1,'+'),(2,'-'),(3,'*')]:
            for i in range(count):
                while True:
                    a,b=rng.randrange(-99,100),rng.randrange(-99,100)
                    if (a,b) not in reserved:reserved.add((a,b));break
                expression=f'({a}) {symbol} ({b})';pos,yes,neg,no=templates[split][i%len(templates[split])]
                pair=f'{split}-op{op}-{i:02}'
                for ready,template,pattern in [(1,pos,yes),(0,neg,no)]:
                    prompt=pattern.format(expr=expression)
                    rows.append(dict(id=f'{pair}-ready{ready}',pair=pair,split=split,ready=ready,
                        lhs=a,rhs=b,operation=op,expression=expression,prompt=prompt,template=template,
                        language='th' if template.startswith('th-') else 'en',
                        prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest()))
        for i,prompt in enumerate(ordinary[split]):
            rows.append(dict(id=f'{split}-ordinary-{i:02}',pair=None,split=split,ready=0,
                prompt=prompt,template=f'{split}-ordinary-{i:02}',
                language='th' if any('\u0e00'<=c<='\u0e7f' for c in prompt) else 'en',
                prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest()))
    rng.shuffle(rows)
    raw=json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return dict(version=1,seed=20261107,kind='matched-intent-final-activation-readiness-diagnostic',
                rows=rows,rows_sha256=hashlib.sha256(raw).hexdigest(),
                scope='initial signed two-digit integer +,-,* module; fixed answer-start anchor',
                exclusions=['not fresh acceptance','no operands or answer training',
                            'not dynamic reasoning-time readiness','no serving eligibility'])


def anchor_examples(rows,tokenizer,prefix,suffix):
    result=[];seen=set()
    for invocation,row in enumerate(rows,1):
        if row['id'] in seen:raise ValueError('duplicate readiness sample')
        seen.add(row['id']);tokens=tokenizer.encode((prefix+row['prompt'].encode('utf-8')+suffix).decode('utf-8'))
        if not 2<=len(tokens)<=256 or len(result)>=1024:raise ValueError('readiness observation bound')
        result.append(dict(sample=row['id'],split=row['split'],invocation=invocation,
                           invocation_position=len(tokens)-1,position=len(tokens)-1,output_offset=0,
                           target=None,numeric_target=False,ids=tokens))
    return result
