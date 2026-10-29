"""New left-aligned neural digit view. Exact executor ABI remains unchanged."""
import numpy as np
import torch
from tools.igc.attention_readout import InputReadout
from tools.igc.readout import targets as abi_targets

SIZES=[4,2,2,2,2,10,11,10,11]


def targets(rows):
    result=abi_targets(rows)
    for i,row in enumerate(rows):
        for key,first,second in [('lhs',5,6),('rhs',7,8)]:
            digits=str(abs(row[key]))
            result[i,first]=int(digits[0]);result[i,second]=int(digits[1]) if len(digits)==2 else 10
    return result


def canonical_fields(predictions):
    values=np.asarray(predictions)
    if (values.ndim!=2 or values.shape[1]!=9
            or any(np.any((values[:,i]<0)|(values[:,i]>=size)) for i,size in enumerate(SIZES))):
        raise ValueError('ordered neural prediction shape/classes')
    result=np.empty(len(values),dtype=object)
    for i,row in enumerate(values):
        converted=row.tolist();valid=True
        for sign,length,first,second in [(1,3,5,6),(2,4,7,8)]:
            negative,two,most,least=map(int,row[[sign,length,first,second]])
            if ((not two and least!=10) or (two and (most==0 or least==10))
                    or (not two and most==0 and negative)):
                valid=False;break
            converted[first]=least if two else most
            converted[second]=most if two else 0
        result[i]=converted if valid else None
    return result


class OrderedInputReadout(InputReadout):
    def __init__(self):
        super().__init__(use_final_query=False,contextual=True)
        # Preserve shared and first ten output-class initialization exactly;
        # only the two new explicit PAD logits require additional parameters.
        for index in (6,8):
            original=self.heads[index];replacement=torch.nn.Linear(64,11)
            with torch.no_grad():
                replacement.weight[:10].copy_(original.weight)
                replacement.bias[:10].copy_(original.bias)
            self.heads[index]=replacement
