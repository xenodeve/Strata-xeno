"""Frozen retained-readout evaluation on an identity-bound development cache."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from tools.igc.readout import SIZES, targets
from tools.igc.development_audit import predicted_padding

STATE_SHA = '519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2'
STATE_REPORT_SHA = '6b47259d3b8ce9fb88319cebced1fbbb6ce1d9dd188bfc6d069c707668133870'
PROTOCOL_SHA = '8e732a66675f68d7ef7fee0b270a783f4be6a28032fd3067203930f0377cd366'
CACHE_PROTOCOL_SHA = '2de4d752b3df50a3623624f9acbc95e874b80d0acd02fdc3ee458a679cf75e0c'
CACHE_SHA = '6757cde83254fff23010d26d720bc579880e2854750a4ee163b5224b8e12d82a'
CACHE_REPORT_SHA = '7a3e5a5b9178df601a14a3bc7e4af4592ca45ccd3f3a5cfb82bdcc2f3b85fecd'


def checked(rows, predicted):
    predicted = np.asarray(predicted)
    if (not rows or len({r['id'] for r in rows}) != len(rows)
            or predicted.shape != (len(rows), 9) or not np.issubdtype(predicted.dtype, np.integer)
            or any(np.any((predicted[:, i] < 0) | (predicted[:, i] >= size)) for i, size in enumerate(SIZES))):
        raise ValueError('categorical prediction shape/class or duplicate identity')
    return targets(rows), predicted


def score(rows, predicted):
    truth, predicted = checked(rows, predicted)
    positive = truth[:, 0] > 0
    exact = np.all(truth == predicted, axis=1)
    operands = np.all(truth[:, 1:] == predicted[:, 1:], axis=1)
    counts = dict(rows=len(rows), positives=int(positive.sum()), negatives=int((~positive).sum()),
        positive_exact_tuple=int((positive & exact).sum()),
        positive_operands_exact=int((positive & operands).sum()),
        positive_no_call=int((positive & (predicted[:, 0] == 0)).sum()),
        positive_wrong_nonzero_operator=int((positive & (predicted[:, 0] != 0) & (truth[:, 0] != predicted[:, 0])).sum()),
        negative_false_call=int((~positive & (predicted[:, 0] != 0)).sum()),
        negative_no_call=int((~positive & (predicted[:, 0] == 0)).sum()),
        positive_operator_correct=int((positive & (truth[:, 0] == predicted[:, 0])).sum()),
        signs_correct=0, signs_total=0, lengths_correct=0, lengths_total=0,
        active_digits_correct=0, active_digits_total=0, padding_correct=0, padding_total=0,
        padding_only_operands=0, active_only_operands=0, both_digit_error_operands=0, other_error_operands=0)
    fields, digits = defaultdict(Counter), defaultdict(Counter)
    for index, (t, p) in enumerate(zip(truth, predicted, strict=True)):
        fields['ready_operator'][f'{t[0]}->{p[0]}'] += 1
        if not positive[index]:
            continue
        for i in range(1, 9):
            fields[str(i)][f'{t[i]}->{p[i]}'] += 1
        for side, (sign, length, one, ten) in [('lhs', (1, 3, 5, 6)), ('rhs', (2, 4, 7, 8))]:
            counts['signs_correct'] += int(t[sign] == p[sign]); counts['signs_total'] += 1
            counts['lengths_correct'] += int(t[length] == p[length]); counts['lengths_total'] += 1
            active = [one, ten] if t[length] else [one]
            active_bad = any(t[i] != p[i] for i in active)
            pad_bad = not t[length] and p[ten] != 0
            other_bad = t[sign] != p[sign] or t[length] != p[length]
            counts['active_digits_correct'] += sum(int(t[i] == p[i]) for i in active)
            counts['active_digits_total'] += len(active)
            if not t[length]:
                counts['padding_total'] += 1; counts['padding_correct'] += int(not pad_bad)
            if other_bad:
                counts['other_error_operands'] += 1
            elif active_bad and pad_bad:
                counts['both_digit_error_operands'] += 1
            elif active_bad:
                counts['active_only_operands'] += 1
            elif pad_bad:
                counts['padding_only_operands'] += 1
            for slot, i in [('ones', one), ('tens', ten)]:
                digits[f'{side}/{int(t[length])+1}/{slot}'][f'{t[i]}->{p[i]}'] += 1
    return dict(counts=counts, field_confusion={k: dict(v) for k, v in fields.items()},
                digit_confusion={k: dict(v) for k, v in digits.items()})


def cluster_interval(rows, values):
    if len(rows) != len(values):
        raise ValueError('cluster outcome count')
    grouped = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        if not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('bounded cluster outcome')
        grouped[row['pair_group']].append(float(value))
    means = np.array([np.mean(grouped[k]) for k in sorted(grouped)], dtype=np.float64)
    result = dict(clusters=len(means), estimate=float(means.mean()) if len(means) else None,
                  interval95=None, unit='signed pair; equal-weight mean of within-pair outcome means',
                  resamples=4096, seed=20261109)
    if len(means) >= 2:
        indices = np.random.default_rng(20261109).integers(len(means), size=(4096, len(means)))
        result['interval95'] = np.quantile(means[indices].mean(axis=1), [.025, .975]).tolist()
    return result


def matched_contrasts(rows, predicted):
    truth, predicted = checked(rows, predicted)
    family, intent = defaultdict(dict), defaultdict(dict)
    for row, t, p in zip(rows, truth, predicted, strict=True):
        success = bool(np.array_equal(t, p)) if row['ready_op'] else p[0] == 0
        for dest, key, name in [(family, (row['split'], row['group'], row['language'], row['intent']), row['family']),
                                (intent, (row['split'], row['group'], row['language'], row['family']), row['intent'])]:
            if name in dest[key]:
                raise ValueError('duplicate matched condition')
            dest[key][name] = (row, bool(success))
    def transitions(table, first, second, positive=None):
        counts = dict(both_correct=0, **{first+'_only_correct': 0, second+'_only_correct': 0}, both_wrong=0, pairs=0)
        refs, deltas, joints = [], [], []
        for conditions in table.values():
            if set(conditions) != {first, second}:
                continue
            a, good_a = conditions[first]; b, good_b = conditions[second]
            if (a['lhs'], a['rhs'], a['operation'], a['pair_group']) != (b['lhs'], b['rhs'], b['operation'], b['pair_group']):
                raise ValueError('matched expression identity')
            if positive is not None and bool(a['ready_op']) != positive:
                continue
            key = 'both_correct' if good_a and good_b else first+'_only_correct' if good_a else second+'_only_correct' if good_b else 'both_wrong'
            counts[key] += 1; counts['pairs'] += 1
            refs.append(a); deltas.append(int(good_b)-int(good_a)); joints.append(int(good_a and good_b))
        return counts, cluster_interval(refs, deltas), cluster_interval(refs, joints)
    pos, delta, _ = transitions(family, 'seen', 'new', True)
    neg, negdelta, _ = transitions(family, 'seen', 'new', False)
    joint, _, jointci = transitions(intent, 'compute', 'quote-copy')
    joint['quote_only_correct'] = joint.pop('quote-copy_only_correct')
    return dict(family_positive_transitions=pos, family_positive_new_minus_seen=delta,
                family_negative_no_call_transitions=neg, family_negative_no_call_new_minus_seen=negdelta,
                intent_joint=joint, intent_joint_success=jointci)


def infer(model, cache, count, batch_size=64, deadline_seconds=180, observer=None):
    if type(count) is not int or not 1 <= count <= 1536 or type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError('bounded evaluation batch/count')
    if not 0 < deadline_seconds <= 180:
        raise ValueError('bounded evaluation deadline')
    model.eval(); model.requires_grad_(False)
    started = time.monotonic(); predictions = []
    with torch.no_grad():
        for offset in range(0, count, batch_size):
            if time.monotonic()-started > deadline_seconds:
                raise TimeoutError('retained evaluation deadline')
            selection = np.arange(offset, min(offset+batch_size, count), dtype=np.int64)
            logits = model(*cache.batch(selection))
            if logits.shape != (len(selection), sum(SIZES)) or not bool(torch.isfinite(logits).all()):
                raise ValueError('nonfinite or malformed retained logits')
            predictions.append(np.stack([h.argmax(1).cpu().numpy() for h in logits.split(SIZES, dim=1)], axis=1))
            if observer is not None:
                observer(offset+len(selection))
            if time.monotonic()-started > deadline_seconds:
                raise TimeoutError('retained evaluation deadline after batch')
    return np.concatenate(predictions)


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def validate_cuda_environment():
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1,0' or 'CUDA_DEVICE_ORDER' in os.environ
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise ValueError('frozen CUDA visibility/order/workspace required')


def frozen_cache_report(path):
    from tools.igc.context_cache import regular
    if regular(path).st_size > 256*1024 or digest(path) != CACHE_REPORT_SHA:
        raise ValueError('frozen cache report identity')
    return json.loads(Path(path).read_bytes())


def partition_scores(rows, predicted):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        length = f"{len(str(abs(r['lhs'])))}/{len(str(abs(r['rhs'])))}"
        for name in [r['split'], r['split']+'/family/'+r['family'], r['split']+'/language/'+r['language'],
                     r['split']+'/length/'+length,
                     '/'.join([r['split'], r['family'], r['language'], length])]:
            groups[name].append(i)
    result = {}
    for key, indices in groups.items():
        selected = [rows[i] for i in indices]; p = predicted[indices]
        t, _ = checked(selected, p); positive = t[:, 0] > 0
        posrows = [r for r, keep in zip(selected, positive) if keep]
        negrows = [r for r, keep in zip(selected, positive) if not keep]
        value = score(selected, p)
        value['positive_exact_cluster'] = cluster_interval(posrows, np.all(t[positive] == p[positive], axis=1))
        value['negative_false_call_cluster'] = cluster_interval(negrows, p[~positive, 0] != 0)
        result[key] = value
    return result


def evaluate(cache, state, sources, protocol, output, cache_protocol=None):
    from tools.igc.context_cache import ContextInputs, load_cache, regular
    from tools.igc.attention_readout import InputReadout
    from tools.qwenputer.memory import system_snapshot, process_snapshot
    from safetensors.torch import load_file
    state = Path(state); cache = Path(cache); output = Path(output); protocol = Path(protocol)
    sources = Path(sources)
    if regular(state).st_size > 2*1024**2 or digest(state) != STATE_SHA:
        raise ValueError('retained state identity')
    if cache_protocol is None:
        raise ValueError('explicit frozen cache protocol required')
    cache_protocol = Path(cache_protocol)
    if (regular(protocol).st_size > 64*1024 or digest(protocol) != PROTOCOL_SHA
            or regular(cache_protocol).st_size > 64*1024 or digest(cache_protocol) != CACHE_PROTOCOL_SHA):
        raise ValueError('frozen evaluation/cache protocol identity')
    receipt = state.parent/'report.json'
    if regular(receipt).st_size > 256*1024 or digest(receipt) != STATE_REPORT_SHA:
        raise ValueError('retained fit receipt identity')
    fitted = json.loads(receipt.read_bytes())
    if (fitted.get('parameters'), fitted.get('activation_stage'), fitted.get('query_mode'),
            fitted.get('contextual_block'), fitted.get('trained_state_sha256')) != (
            96964, 'native_first_attention_output', 'learned_queries_only', True, STATE_SHA):
        raise ValueError('retained architecture/stage mismatch')
    if output.exists():
        raise ValueError('evaluation output must be new')
    validate_cuda_environment()
    frozen_cache_report(cache/'report.json')
    start = time.monotonic(); system = system_snapshot()
    if (system['physical_available_bytes'] < 8*1024**3 or system['system_commit_available_bytes'] < 12*1024**3
            or shutil.disk_usage(output.parent).free < 128*1024**2):
        raise RuntimeError('offline retained evaluation admission; no GPU work launched')
    source_files = [Path(__file__), Path(__file__).with_name('attention_readout.py'),
                    Path(__file__).with_name('readout.py'), Path(__file__).with_name('context_cache.py'),
                    Path(__file__).with_name('input_manifest.py'), Path(__file__).with_name('development_audit.py'),
                    sources, state, receipt, protocol, cache_protocol, cache/'corpus.json',cache/'report.json']
    identities = {str(p): digest(p) for p in source_files}
    states, valid, rows, cached = load_cache(cache, sources, cache_protocol)
    try:
        if len(rows) != 1536 or cached['states_sha256'] != CACHE_SHA:
            raise ValueError('frozen matched cache identity')
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 2*1024**3:
            raise RuntimeError('offline retained mapper needs existing2GiB GPU admission')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True); torch.backends.mha.set_fastpath_enabled(False)
        tensors = load_file(str(state), device='cpu')
        if any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in tensors.values()):
            raise ValueError('retained tensors must be finite F32')
        model = InputReadout(use_final_query=False, contextual=True)
        model.load_state_dict(tensors, strict=True)
        if sum(p.numel() for p in model.parameters()) != 96964:
            raise ValueError('retained parameter count')
        torch.cuda.reset_peak_memory_stats(); model.to('cuda')
        resources = []
        def observe(count):
            resources.append(dict(rows=count,seconds=time.monotonic()-start,
                process=process_snapshot(os.getpid()),system=system_snapshot()))
            print('retained evaluated',count,'/1536',flush=True)
        inferred = time.monotonic()
        raw = infer(model, ContextInputs(states, valid, 'cuda'), len(rows), observer=observe)
        inference_seconds = time.monotonic()-inferred
        if any(not torch.equal(tensors[k], v.detach().cpu()) for k, v in model.state_dict().items()):
            raise RuntimeError('retained parameters changed')
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('unexpected evaluation gradient')
        variant = np.array([predicted_padding([int(v) for v in p]) for p in raw], dtype=np.int64)
        by_expression = defaultdict(list)
        for i, row in enumerate(rows):
            by_expression[row['split']+'/'+row['group']].append(i)
        expression_scores = {key: dict(pair_group=rows[ix[0]]['pair_group'],
            raw=score([rows[i] for i in ix],raw[ix]),
            paired=matched_contrasts([rows[i] for i in ix],raw[ix])) for key,ix in by_expression.items()}
        truth = targets(rows); positive = truth[:, 0] > 0
        a = np.all(raw == truth, axis=1); b = np.all(variant == truth, axis=1)
        split_indices = {split: [i for i,r in enumerate(rows) if r['split']==split]
                         for split in dict.fromkeys(r['split'] for r in rows)}
        matched_by_split = {split: matched_contrasts([rows[i] for i in ix],raw[ix])
                            for split,ix in split_indices.items()}
        composition = dict(scope='separate predicted-length composition, not learned padding or deployed policy',
            changed_rows=int(np.any(raw != variant,axis=1).sum()),
            positive_recovered=int((positive & ~a & b).sum()),positive_regressed=int((positive & a & ~b).sum()),
            partitions=partition_scores(rows,variant),matched=matched_contrasts(rows,variant),
            matched_by_split={split: matched_contrasts([rows[i] for i in ix],variant[ix])
                              for split,ix in split_indices.items()})
        value = dict(success=True,scope='inspected matched development; no training, acceptance or live policy',
            state_sha256=STATE_SHA,cache_sha256=CACHE_SHA,protocol_sha256=PROTOCOL_SHA,
            cache_protocol_sha256=CACHE_PROTOCOL_SHA,source_sha256=identities,rows=len(rows),
            activation_stage='native_first_attention_output',query_mode='learned_queries_only',parameters=96964,
            raw=partition_scores(rows,raw),matched=matched_contrasts(rows,raw),matched_by_split=matched_by_split,
            expression_groups=expression_scores,padding_counterfactual=composition,
            inference_seconds=inference_seconds,elapsed_seconds=time.monotonic()-start,
            preflight=system,resources=resources,process_after=process_snapshot(os.getpid()),
            cuda=dict(device=torch.cuda.get_device_name(0),torch_version=torch.__version__,build=torch.version.cuda,
                tf32=False,sdp='math',peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved()),
            weights_unchanged=True,gradients=False,optimizer=False,fresh_acceptance_access=False)
        predictions = [dict(id=r['id'],target=t.tolist(),raw=p.tolist(),padding_variant=v.tolist())
                       for r,t,p,v in zip(rows,truth,raw,variant,strict=True)]
        content = json.dumps(value,indent=2); prediction_bytes=json.dumps(predictions,indent=2)
        if len(content.encode())+len(prediction_bytes.encode()) > 16*1024**2:
            raise ValueError('bounded evaluation report size')
        if identities != {str(p): digest(p) for p in source_files}:
            raise RuntimeError('evaluation input or source identity changed')
        if digest(cache/'states.npy') != CACHE_SHA or digest(cache/'valid.npy') != cached['valid_sha256']:
            raise RuntimeError('evaluation cache content changed')
        value['cache_content_unchanged'] = True
        value['elapsed_seconds'] = time.monotonic()-start
        output.mkdir(exist_ok=False)
        (output/'predictions.json').write_text(prediction_bytes,encoding='utf-8')
        value['predictions_sha256']=digest(output/'predictions.json')
        (output/'report.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
        print(json.dumps(dict(success=True,raw={k:v['counts'] for k,v in value['raw'].items() if '/' not in k},
                             padding_recovered=composition['positive_recovered'],elapsed_seconds=value['elapsed_seconds'])),flush=True)
        return value
    finally:
        states.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache','state','sources','protocol','cache-protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    evaluate(**vars(parser.parse_args()))
