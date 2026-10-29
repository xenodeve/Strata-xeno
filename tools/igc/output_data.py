"""Bounded, provenance-checked output training data; no model or arithmetic execution."""
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.igc.observation import _read,load_observation
from tools.igc.output_corpus import dataset
from tools.igc.output_examples import position_examples,wrapped_prefix
from tools.igc.provenance import verify_sources
from tools.qwenputer.tokenizer import NativeText


def validate_record(record,example,observed,index):
    if record.get('index')!=index or any(record.get(k)!=v for k,v in example.items()):
        raise ValueError('output label/request identity')
    if (observed['prefix_tokens'].tolist()!=example['ids'] or observed['position']!=example['position']
            or observed['prompt_tokens']!=len(example['ids'])
            or record.get('engine_done',{}).get('generated')!=1
            or record['engine_done'].get('prompt_tokens')!=len(example['ids'])
            or len(record.get('unadapted_next_token',[]))!=1):
        raise ValueError('output completed native request boundary')
    for key in ('position','prefix_sha256','mixed_sha256','logits_sha256','metadata_sha256'):
        if observed[key]!=record.get('observation',{}).get(key):
            raise ValueError('output captured tensor identity')


def load_dataset(root,reference,pack,sources,capture_protocol):
    root=Path(root)
    raw_report=_read(root/'report.json',limit=256*1024);report=json.loads(raw_report)
    corpus=dataset()
    if (report.get('success') is not True or report.get('positions')!=421 or report.get('samples')!=30
            or report.get('rows_sha256')!=corpus['rows_sha256']
            or report.get('source_manifest_sha256')!=hashlib.sha256(_read(sources,limit=256*1024)).hexdigest()
            or report.get('protocol_sha256')!=hashlib.sha256(_read(capture_protocol,limit=256*1024)).hexdigest()
            or report.get('hard_private_commit_cap') is not None
            or report.get('actual_hub_vision_initialized') is not True
            or report.get('launcher_gpu_environment')!={'CUDA_VISIBLE_DEVICES':'1,0','CUDA_DEVICE_ORDER':''}):
        raise ValueError('output collection/protocol identity')
    resolution=verify_sources(sources)
    if json.loads(_read(root/'corpus.json',limit=1024**2))!=corpus:
        raise ValueError('output frozen corpus identity')
    raw_records=_read(root/'records.json',limit=8*1024**2)
    if hashlib.sha256(raw_records).hexdigest()!=report['records_sha256']:
        raise ValueError('output record digest')
    records=json.loads(raw_records)
    tokenizer=NativeText.from_pack(pack/'tokenizer')
    if tokenizer.binding_sha256!=report['tokenizer_binding_sha256']:
        raise ValueError('output tokenizer binding')
    prefix,suffix=wrapped_prefix(reference,tokenizer)
    examples=position_examples(corpus['rows'],tokenizer,prefix,suffix)
    if len(records)!=len(examples):raise ValueError('output collection incomplete')
    vectors=[]
    for index,(record,example) in enumerate(zip(records,examples,strict=True)):
        observed=load_observation(root,index)
        validate_record(record,example,observed,index)
        vectors.append(observed['mixed'])
    return dict(mixed=np.stack(vectors),records=records,rows=corpus['rows'],tokenizer=tokenizer,
                collection=report,collection_report_sha256=hashlib.sha256(raw_report).hexdigest(),
                current_source_resolution=resolution)
