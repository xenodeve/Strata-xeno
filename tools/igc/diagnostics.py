"""Offline two-digit mapper diagnosis; never repairs predictions or computes answers."""
import numpy as np


def canonical_value(fields,side):
    if side not in (0,1) or len(fields)!=9:
        raise ValueError('diagnostic operand shape')
    sign,length,ones,tens=(int(fields[i]) for i in ((1,3,5,6) if side==0 else (2,4,7,8)))
    if sign not in (0,1) or length not in (0,1) or not (0<=ones<10 and 0<=tens<10):
        return None
    if (length==0 and tens!=0) or (length==1 and tens==0):return None
    magnitude=ones+10*tens
    if magnitude==0 and sign:return None
    return -magnitude if sign else magnitude


def analyze(rows,predicted):
    predicted=np.asarray(predicted)
    if predicted.shape!=(len(rows),9):raise ValueError('diagnostic prediction shape')
    selected=[(r,p) for r,p in zip(rows,predicted,strict=True) if r['ready_op']>0]
    n=len(selected)
    report=dict(n=n,operator_correct=0,lhs_exact=0,rhs_exact=0,operand_swaps=0,
                lhs_digit_reversal=0,rhs_digit_reversal=0,invalid_operand_predictions=0)
    for side,name in enumerate(('lhs','rhs')):
        sign_i,len_i,ones_i,tens_i=(1,3,5,6) if side==0 else (2,4,7,8)
        report[name+'_sign_correct']=sum(int(p[sign_i])==int(r[name]<0) for r,p in selected)
        report[name+'_length_correct']=sum(int(p[len_i])==int(abs(r[name])>=10) for r,p in selected)
        report[name+'_ones_nonpadding']=dict(correct=sum(int(p[ones_i])==abs(r[name])%10 for r,p in selected),n=n)
        twos=[(r,p) for r,p in selected if abs(r[name])>=10]
        report[name+'_tens_nonpadding']=dict(correct=sum(int(p[tens_i])==abs(r[name])//10 for r,p in twos),n=len(twos))
    for row,fields in selected:
        a,b=canonical_value(fields,0),canonical_value(fields,1)
        report['operator_correct']+=int(int(fields[0])==row['ready_op'])
        report['lhs_exact']+=int(a==row['lhs'])
        report['rhs_exact']+=int(b==row['rhs'])
        report['invalid_operand_predictions']+=int(a is None)+int(b is None)
        report['operand_swaps']+=int(row['lhs']!=row['rhs'] and a==row['rhs'] and b==row['lhs'])
        for name,value in [('lhs',a),('rhs',b)]:
            expected=row[name];magnitude=abs(expected)
            reverse=(magnitude%10)*10+magnitude//10
            reverse=reverse*(-1 if expected<0 else 1)
            if magnitude>=10 and reverse!=expected and value==reverse:
                report[name+'_digit_reversal']+=1
    return report
