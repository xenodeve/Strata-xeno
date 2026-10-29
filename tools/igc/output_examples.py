"""Pure output-position planning, also usable by the CUDA training environment."""
import json
from tools.igc.observation import _read,load_observation


def position_examples(rows,tokenizer,prefix,suffix):
    examples=[];seen=set()
    for invocation,row in enumerate(rows,1):
        if row['id'] in seen:raise ValueError('duplicate output sample')
        seen.add(row['id'])
        base=tokenizer.encode((prefix+row['prompt'].encode('utf-8')+suffix).decode('utf-8'))
        completion=tokenizer.encode(row['completion'])
        if not 1<=len(completion)<=64 or not 2<=len(base)<=256:raise ValueError('output sequence bounds')
        anchor=len(base)-1
        for j,target in enumerate(completion):
            tokens=base+completion[:j]
            if len(tokens)>256 or len(examples)>=1024:raise ValueError('output observation bound')
            piece=tokenizer.decode_bytes([target])
            numeric=bool(piece) and all(c in b'-0123456789' for c in piece)
            examples.append(dict(sample=row['id'],split=row['split'],invocation=invocation,
                                 invocation_position=anchor,position=len(tokens)-1,output_offset=j,
                                 target=target,numeric_target=numeric,ids=tokens))
    return examples


def wrapped_prefix(reference,tokenizer):
    original=json.loads(_read(reference/'corpus.json',limit=1024**2))['rows']
    prefix=suffix=None
    for i,row in enumerate(original):
        observed=load_observation(reference,i);tokens=observed['prefix_tokens'].tolist()
        raw=tokenizer.decode_bytes(tokens);prompt=row['prompt'].encode('utf-8')
        if raw.count(prompt)!=1:raise ValueError('output wrapper occurrence')
        at=raw.index(prompt)
        if prefix is None:prefix,suffix=raw[:at],raw[at+len(prompt):]
        if raw!=prefix+prompt+suffix or tokenizer.encode(raw.decode('utf-8'))!=tokens:
            raise ValueError('output native wrapper fidelity')
    return prefix,suffix
