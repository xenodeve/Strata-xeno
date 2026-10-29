"""Small frozen teacher-forced output pilot; oracle labels are offline only."""
import hashlib
import json
import random
from tools.igc.balanced_corpus import dataset as input_v1,expanded_dataset
from tools.igc.mapper_corpus import corpus,evaluation_manifest


def dataset():
    randomizer=random.Random(20261104)
    reserved={(r['lhs'],r['rhs']) for r in (input_v1()['rows']+expanded_dataset()['rows']
              +corpus()+evaluation_manifest()['rows'])}
    used_values=set();rows=[]
    for split,count in [('train',8),('development',2)]:
        for operation in (1,2,3):
            for index in range(count):
                while True:
                    lhs,rhs=randomizer.randrange(-99,100),randomizer.randrange(-99,100)
                    answer={1:lambda:lhs+rhs,2:lambda:lhs-rhs,3:lambda:lhs*rhs}[operation]()
                    if (lhs,rhs) not in reserved and answer not in used_values:break
                reserved.add((lhs,rhs));used_values.add(answer)
                expression=f'({lhs}) '+{1:'+',2:'-',3:'*'}[operation]+f' ({rhs})'
                thai=index%2==1
                prompt=(f'คำนวณ {expression} แล้วตอบผลลัพธ์จำนวนเต็ม' if thai else
                        f'Calculate {expression} and give the exact integer result.')
                completion=(f'ผลลัพธ์คือ {answer}\n</think>\n\n{answer}<|im_end|>' if thai else
                            f'The result is {answer}.\n</think>\n\n{answer}<|im_end|>')
                rows.append(dict(id=f'output-{split}-op{operation}-{index:02}',split=split,
                                 operation=operation,lhs=lhs,rhs=rhs,answer=answer,expression=expression,
                                 language='th' if thai else 'en',prompt=prompt,completion=completion,
                                 prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest()))
    randomizer.shuffle(rows)
    raw=json.dumps(rows,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return dict(version=1,seed=20261104,kind='native-teacher-forced-output-position-pilot',
                rows=rows,rows_sha256=hashlib.sha256(raw).hexdigest(),
                label_scope='offline independent integer oracle; actual checked native results are mapper features',
                exclusions=['not a release corpus','no I2 acceptance cases','no trained result-use claim',
                            'not full62-digit coverage','teacher-forced metrics do not establish free generation'])
