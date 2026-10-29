"""Prepare inputs and check observations offline; never runs an arithmetic oracle in a timed call."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct

ORDER = ['qwenputer', 'exact', 'exact', 'qwenputer']


def distribution(values):
    values = sorted(values)
    return dict(n=len(values), min_ms=values[0], median_ms=statistics.median(values),
                p90_ms=values[math.ceil(0.9*len(values))-1], max_ms=values[-1])


def validate_results(records, cases, blocks):
    samples = [r for r in records if r.get('kind') == 'sample']
    required = [(i,b,s,arm) for i in range(len(cases)) for b in range(blocks)
                for s,arm in enumerate(ORDER)]
    found = [(r['case'],r['block'],r['slot'],r['arm']) for r in samples]
    if found != required:
        raise ValueError('incomplete_or_reordered_ABBA')
    for r in samples:
        if r['status'] != 0 or r['actual'] != cases[r['case']]['expected']:
            raise ValueError('execution_or_oracle_failure')
        for key in ['wall_ms','prepare_ms','issue_ms','wait_ms']:
            if not math.isfinite(r[key]) or r[key] < 0:
                raise ValueError('invalid_timing')
        if sum(r[k] for k in ['prepare_ms','issue_ms','wait_ms']) > r['wall_ms'] + 0.01:
            raise ValueError('inconsistent_stage_budget')
    output = []
    for i in range(len(cases)):
        item = dict(case=i)
        for arm in ['qwenputer','exact']:
            rows = [r for r in samples if r['case']==i and r['arm']==arm]
            item[arm] = distribution([r['wall_ms'] for r in rows])
            item[arm]['stage_median_ms'] = {key:statistics.median(r[key] for r in rows)
                                           for key in ['prepare_ms','issue_ms','wait_ms']}
            item[arm]['steps'] = [r['steps'] for r in rows]
        output.append(item)
    return dict(kind='component-only', cases=output, blocks=blocks,
                oracle_timing='outside native measured intervals')


def normalize(raw, arm):
    if arm == 'exact':
        return raw
    # The reviewed calc firmware closes with an empty thinking block. Keep the
    # raw wire output, require the full envelope, and refuse extra text/answers.
    match = re.fullmatch(r'\s*<tool_response>\s*(-?[0-9]+)\s*</tool_response>'
                         r'\s*<think>\s*</think>\s*',raw)
    if match is None:
        raise ValueError('unexpected_response_envelope:'+repr(raw))
    return match[1]


def prepare(output, tokenizer):
    from tools.qwenputer.tokenizer import NativeText
    from tools.qwenputer.call import CallContext, eligible_call
    from tools.test_igc_exact import request
    text = NativeText.from_pack(tokenizer)
    # Frozen small overlap of firmware arithmetic and the typed integer ABI.
    # This is a latency corpus, not a replacement for held-out correctness tests.
    triples = [(17,23,3),(-123,456,1),(1000,1,2)]
    cases, binary = [], bytearray(struct.pack('<II',0x31424349,len(triples)))
    for i,(a,b,op) in enumerate(triples):
        symbol = {1:'+',2:'-',3:'*'}[op]
        expression = f'{a}{symbol}{b}'
        call = '\n<function=calc><parameter=expression>'+expression+'</parameter></function>\n'
        qids = text.encode(call)
        eligible_call(text,qids,CallContext(live_assistant=True,enrolled_calc=True))
        prefix = [text.TCO,*qids,text.TCC]
        binary += struct.pack('<I',len(prefix)) + request(a,b,op,i+1)
        binary += b''.join(struct.pack('<II',1,qid) for qid in prefix)
        value = {1:lambda:a+b,2:lambda:a-b,3:lambda:a*b}[op]()
        cases.append(dict(expression=expression, expected=str(value), a=str(a),b=str(b),
                          operation=op, prefix_tokens=len(prefix)))
    output.mkdir(parents=True,exist_ok=False)
    (output/'corpus.bin').write_bytes(binary)
    manifest = dict(version=1,cases=cases,corpus_sha256=hashlib.sha256(binary).hexdigest(),
                    tokenizer_sha256=text.binding_sha256,
                    input_scope='prebuilt native inputs; tokenization/activation mapping excluded')
    (output/'corpus.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


def summarize(source, corpus, output, blocks=2):
    records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()
               if line.startswith('{')]
    cases = json.loads(corpus.read_text(encoding='utf-8'))['cases']
    for r in records:
        if r.get('kind') in ['sample','warmup','cold']:
            r['actual'] = normalize(r['actual_raw'],r['arm'])
            if r['status'] != 0 or r['actual'] != cases[r['case']]['expected']:
                raise ValueError('observed_result_mismatch')
    report = validate_results(records,cases,blocks)
    report['raw_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    report['corpus_manifest_sha256'] = hashlib.sha256(corpus.read_bytes()).hexdigest()
    report['setup'] = [r for r in records if r.get('kind')=='setup']
    report['memory'] = [r for r in records if r.get('kind')=='memory']
    report['cases_input'] = cases
    output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    make = sub.add_parser('prepare')
    make.add_argument('--output',type=Path,required=True)
    make.add_argument('--tokenizer',type=Path,required=True)
    check = sub.add_parser('summarize')
    check.add_argument('--source',type=Path,required=True)
    check.add_argument('--corpus',type=Path,required=True)
    check.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.command=='prepare':
        prepare(args.output,args.tokenizer)
    else:
        summarize(args.source,args.corpus,args.output)


if __name__=='__main__':
    main()
