"""Shared offline Hub diagnostic contracts; no tensor runtime import."""
import re

def final_integer(text):
    # Frozen pilot rubric: reasoning must close, and the final section contains
    # exactly one integer. A truncated thinking prefix never counts as an answer.
    if '</think>' not in text:return None
    final=text.rsplit('</think>',1)[1].replace('<|im_end|>','').strip()
    numbers=re.findall(r'(?<![\w.])-?\d+(?!\w|\.\d)',final)
    return int(numbers[0]) if len(numbers)==1 else None


def interpreted_final(text,expression):
    """Offline result extraction/consistency checks, never arithmetic evaluation."""
    result=dict(value=None,equation_consistent=True,contradictory=False)
    if '</think>' not in text:return result
    final=text.rsplit('</think>',1)[1].replace('<|im_end|>','').replace('**','').replace('`','').strip()
    values=[]
    def normalized(value):return re.sub(r'[\s()]','',value).replace('×','*').replace('−','-')
    for line in final.splitlines():
        line=line.strip().replace('−','-')
        bare=re.fullmatch(r'([-+]?[0-9]+)\s*[.!。]?',line)
        statement=re.search(r'(?:the\s+)?(?:result|answer)\s*(?:is|equals|[:=])\s*([-+]?[0-9]+)\s*[.!。]?$',line,re.I)
        thai=re.search(r'(?:ผลลัพธ์|คำตอบ)(?:ที่ได้)?\s*(?:คือ|เป็น|เท่ากับ|[:=])\s*([-+]?[0-9]+)\s*[.!。]?$',line)
        equation=re.fullmatch(r'([0-9\s()+*×\-]+)=\s*([-+]?[0-9]+)\s*[.!。]?',line)
        if equation:
            result['equation_consistent'] &= expression is not None and normalized(equation.group(1))==normalized(expression)
            values.append(int(equation.group(2)))
        elif bare or statement or thai:values.append(int((bare or statement or thai).group(1)))
    unique=set(values);result['contradictory']=len(unique)>1
    if len(unique)==1:result['value']=values[0]
    return result
