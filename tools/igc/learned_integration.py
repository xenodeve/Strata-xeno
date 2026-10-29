"""First-anchor live learned diagnostic; not a benchmark or acceptance corpus."""
import struct


def parse_audit(raw):
    """Decode a request-end receipt offline; never fed back into inference."""
    if len(raw) != 288:
        raise ValueError("learned audit size")
    owner, anchor, armed, attempts, evaluations, terminal, selection, reserved = struct.unpack_from("<QQ6I", raw)
    version, domain, op, request_reserved, request_owner, request_anchor, left, right = struct.unpack_from("<IIIIQQ64s64s", raw, 40)
    result_version, status, result_owner, result_anchor, value = struct.unpack_from("<IIQQ64s", raw, 200)
    if (armed or reserved or attempts not in (0, 1) or evaluations > attempts or terminal not in (1, 2, 3, 4)
            or anchor >= 262144 or result_version != 1 or (result_owner, result_anchor) != (owner, anchor)
            or status > 4 or version != 1 or request_reserved):
        raise ValueError("learned audit header/ownership")

    def number(data):
        length, negative = data[:2]
        digits = data[2:]
        if (not 1 <= length <= 62 or negative > 1 or any(d > 9 for d in digits)
                or any(digits[length:]) or (length > 1 and not digits[length-1])
                or (length == 1 and digits[0] == 0 and negative)):
            raise ValueError("learned audit integer")
        value = int("".join(str(d) for d in reversed(digits[:length])))
        return -value if negative else value

    request = [op, number(left), number(right)] if attempts and domain == 1 and op in (1, 2, 3) else None
    if status == 0:
        if (not owner or attempts != 1 or evaluations != 1 or terminal != 1 or selection != 0
                or request is None or (request_owner, request_anchor) != (owner, anchor)):
            raise ValueError("learned successful audit contract")
        result = number(value)
    else:
        if any(value):
            raise ValueError("learned failed audit contains value")
        result = None
    return dict(invocation=owner, anchor=anchor, attempts=attempts, evaluations=evaluations,
                terminal=terminal, selection=selection, status=status, request=request, result=result)


def run(profile, server, executable, bundle, capture, sources, pack, prior_memory, output):
    import hashlib
    import importlib.util
    import json
    import os
    from pathlib import Path
    import re
    import shutil
    import subprocess
    import sys
    import threading
    import time
    import psutil
    from tools.igc.observation import load_prefix
    from tools.igc.result_rubric import interpreted_final
    from tools.igc.hub_contract import current_job, validate_launcher_env
    from tools.igc.provenance import verify_sources
    from tools.qwenputer.memory import system_snapshot, process_snapshot
    from tools.qwenputer.tokenizer import NativeText

    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()

    validate_launcher_env(os.environ)
    if any((p.info['name'] or '').lower() in ('strata.exe', 'strata-vision.exe')
           for p in psutil.process_iter(['name'])):
        raise RuntimeError('existing engine; no duplicate launch')
    output.mkdir(exist_ok=False)
    verify_sources(sources)
    bound_paths = {Path(row['path']).resolve() for row in json.loads(sources.read_text())['files']}
    if any(path.resolve() not in bound_paths for path in (profile, server, pack/'index.txt')):
        raise ValueError('live diagnostic inputs differ from the selected captured Hub source paths')
    cfg = json.loads(profile.read_text(encoding='utf-8-sig'))
    if cfg.get('replicas') or cfg.get('cjk_guard'):
        raise ValueError('not the qualified selected Hub profile')
    network_paths = [bundle.parent / name for name in ('input-embedding.bin', 'input-attention.bin', 'readiness.bin', 'output.bin')]
    pinned = [profile, server, executable, bundle, sources, prior_memory, Path(__file__), *network_paths,
              Path(load_prefix.__code__.co_filename), Path(interpreted_final.__code__.co_filename),
              Path(current_job.__code__.co_filename)]
    identity = {str(path): digest(path) for path in pinned}
    old = json.loads(prior_memory.read_text(encoding='utf-8'))
    prior_private = max(sum(p['process_private_commit_bytes'] for p in r['processes']) for r in old)
    prior_working = max(sum(p['working_set_bytes'] for p in r['processes']) for r in old)
    # Task safety admission derived from a recorded matched-profile run, not a
    # Hub policy, per-process cap or cache/workload change. CUDA driver peaks
    # remain unknown until this live run; retain explicit emergency reserves.
    # Conservative two-wave-lane snapshot reservation; inactive lanes may not
    # allocate, but admission does not assume that optimization.
    component_bytes = 12_801_500 + 13_107_280 + 2*6_553_600 + sum(p.stat().st_size for p in network_paths)
    corpus = json.loads((capture/'corpus.json').read_text(encoding='utf-8'))['rows']
    replay = json.loads((capture.parent/'igc-pipeline-replay-01/reference.json').read_text())
    positives = [r['index'] for r in replay['records'] if r['selection'] == 0][:3]
    negatives = [i for i, row in enumerate(corpus) if row['ready_op'] == 0][:2]
    indices = positives + negatives + positives[:1]
    cases = []
    for i in indices:
        row = corpus[i]
        cases.append(dict(id=row['id'], original_index=i, ids=load_prefix(capture, i)['prefix_tokens'],
                          operation=row['ready_op'], lhs=row.get('lhs'), rhs=row.get('rhs'),
                          expression=row.get('expression'), prompt=row['prompt']))
    tokenizer = NativeText.from_pack(pack/'tokenizer')
    local = json.loads(json.dumps(cfg));local['exe'] = str(executable);local['log'] = str(output/'engine.log')
    args = list(local['args']);original = Path(args[args.index('--expert-profile')+1]);original_hash = digest(original)
    shutil.copyfile(original, output/'initial-profile.bin')
    args[args.index('--expert-profile')+1] = str(output/'initial-profile.bin')
    args[args.index('--expert-profile-save')+1] = str(output/'learned-profile.bin')
    args += ['--igc-learned-diagnostic-bundle', str(bundle)]
    local['args'] = args
    (output/'config.json').write_text(json.dumps(local, indent=2), encoding='utf-8')
    (output/'cases.json').write_text(json.dumps(dict(kind='old inspected diagnostic, not fresh acceptance64', cases=cases),
                                                ensure_ascii=False, indent=2), encoding='utf-8')
    sys.path[:0] = [str(server.parent), str(server.parent.parent)]
    spec = importlib.util.spec_from_file_location('igc_learned_pinned_hub', server)
    helper = importlib.util.module_from_spec(spec);sys.modules[spec.name] = helper;spec.loader.exec_module(helper)
    if helper.layer_split_of(local):
        raise ValueError('layer split is not the selected D2x expert offload')
    env = helper.child_env(local)
    env.pop('STRATA_TIMELINE', None);env.pop('STRATA_DUMP_FIRST_LOGITS', None)
    validate_launcher_env(env)
    vision_cfg = ({key: (os.path.abspath(os.path.join(cfg.get('cwd') or '.', value))
        if key in ('exe', 'model', 'mmproj') and isinstance(value, str) and not os.path.isabs(value) else value)
        for key, value in cfg['vision'].items()} if cfg.get('vision') else None)
    owned = {executable.resolve()}
    if vision_cfg:
        for key in ('exe', 'model', 'mmproj'):
            path = Path(vision_cfg[key]);identity[str(path)] = digest(path)
        owned.add(Path(vision_cfg['exe']).resolve())
    arguments = helper.engine_args(local) + (helper.effort_end_args(local, str(executable), tokenizer) or [])
    # Include prepared tokenizer/helper/case allocations in the measured parent
    # footprint, rather than admitting first and adding an unreported harness.
    system = system_snapshot()
    admission = dict(system=system, harness=process_snapshot(os.getpid()), prior_sum_private_peak_bytes=prior_private,
                     prior_sum_working_peak_bytes=prior_working, component_allocation_bytes=component_bytes,
                     commitment_reserve_bytes=2*1024**3, physical_reserve_bytes=1024**3,
                     hub_physical_admission_bytes=24*1024**3,
                     hub_admission_source='strata-hub.bat:96,121-126 (selected Gyro262K NEEDGB=24)',
                     origin='task safety estimate from prior actual Hub-profile evidence; not a Hub limit',
                     hard_private_commit_cap=None, job=current_job())
    admission['admitted'] = (system['system_commit_available_bytes'] >= prior_private + component_bytes + 2*1024**3
                            and system['physical_available_bytes'] >= max(24*1024**3, prior_working + 1024**3)
                            and shutil.disk_usage(output).free >= 1024**3)
    (output/'preflight.json').write_text(json.dumps(admission, indent=2))
    if not admission['admitted']:
        (output/'report.json').write_text(json.dumps(dict(status='not_run_resource_admission', preflight=admission,
            source_sha256=identity, engine_started=False), indent=2))
        print('NOT RUN: task safety admission; no engine started; see preflight.json', flush=True)
        return
    started = time.monotonic();root = psutil.Process(os.getpid())
    stop = threading.Event();cancel = threading.Event();active = [None];memory = [];failure = []
    original_popen = subprocess.Popen

    def headless(*a, **kw):
        kw['creationflags'] = kw.get('creationflags', 0) | 0x08000000
        return original_popen(*a, **kw)

    def monitor():
        while not stop.wait(.2):
            state = system_snapshot();children = []
            for child in root.children(recursive=True):
                try:
                    if Path(child.exe()).resolve() in owned:
                        children.append(process_snapshot(child.pid))
                except (OSError, psutil.Error):
                    pass
            memory.append(dict(seconds=time.monotonic()-started, system=state, processes=children))
            if (time.monotonic()-started > 600 or (active[0] is not None and time.monotonic()-active[0] > 120)
                    or state['system_commit_available_bytes'] < 2*1024**3
                    or state['physical_available_bytes'] < 1024**3):
                failure.append(dict(reason='task deadline or system emergency floor; not an engine verdict', system=state))
                cancel.set()
                for child in root.children(recursive=True):
                    try:
                        if Path(child.exe()).resolve() in owned:
                            child.terminate()
                    except (OSError, psutil.Error):
                        pass
                return

    engine = vision = vision_log = None;records = []
    subprocess.Popen = headless
    thread = threading.Thread(target=monitor, daemon=True);thread.start()
    try:
        if vision_cfg:
            vision_log = (output/'vision.log').open('w', encoding='utf-8')
            vision = helper.Vision(vision_cfg, log=vision_log, env=helper.vision_env(local, env), lazy=False)
        engine = helper.StrataEngine(str(executable), arguments, cwd=local.get('cwd'), log=local['log'], env=env)
        engine.silence_s = helper.engine_silence_s(local)
        for case in cases:
            if cancel.is_set():
                raise RuntimeError('task safety stop before request')
            active[0] = time.monotonic()
            try:
                tokens = [int(t) for t in engine.generate(case['ids'], 96, {'temperature': 0}, cancel) if t is not None]
            finally:
                active[0] = None
            if failure or not engine.alive():
                raise RuntimeError('live diagnostic incomplete; inspect evidence')
            text = tokenizer.decode_bytes(tokens).decode('utf-8', errors='replace')
            # Offline oracle only after the entire request. The engine receives
            # the original token prefix, never these labels or computed answers.
            op, left, right = case['operation'], case['lhs'], case['rhs']
            expected = None if not op else left+right if op == 1 else left-right if op == 2 else left*right
            final = interpreted_final(text, case['expression'])
            records.append(dict(id=case['id'], original_index=case['original_index'], expected=expected,
                final=final, text=text, tokens=tokens, done=engine.last))
            (output/'records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
            print('LIVE', case['id'], 'expected', expected, 'final', final['value'], 'finish', engine.last.get('finish'), flush=True)
    finally:
        if engine is not None:
            engine.close()
        if vision is not None:
            vision.close()
        if vision_log is not None:
            vision_log.close()
        stop.set();thread.join(timeout=2);subprocess.Popen = original_popen
        (output/'memory.json').write_text(json.dumps(memory, indent=2))
        (output/'failure.json').write_text(json.dumps(failure, indent=2))
    if identity != {path: digest(Path(path)) for path in identity} or digest(original) != original_hash:
        raise RuntimeError('live source/profile changed')
    lines = (output/'engine.log').read_text(encoding='utf-8', errors='replace')
    audits = [parse_audit(bytes.fromhex(h)) for h in re.findall(r'strata IGC learned audit: head_status=\d+ record=([0-9a-f]{576})', lines)]
    if len(audits) != len(cases) or len(records) != len(cases):
        raise ValueError('incomplete live audit/output pairing')
    for case, record, audit in zip(cases, records, audits, strict=True):
        record['execution'] = audit
        record['selected_tuple_correct'] = audit['request'] == [case['operation'], case['lhs'], case['rhs']] if audit['evaluations'] else None
        record['final_matches_execution'] = record['final']['value'] == audit['result'] if audit['status'] == 0 else None
    report = dict(status='completed_live_diagnostic', kind='live first-anchor learned route; old diagnostic; not benchmark/acceptance',
                  records=records, source_sha256=identity, preflight=admission, failure=failure, acceptance64_included=False,
                  differences=['opt-in learned bundle', 'request-end audit sync/hex logging',
                               'task-only profile/output paths and bounded cleanup', 'direct Hub engine protocol, not HTTP'])
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PASS completed live requests and paired execution receipts; quality reported separately', flush=True)


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('profile', 'server', 'executable', 'bundle', 'capture', 'sources', 'pack', 'prior-memory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    run(**vars(parser.parse_args()))
