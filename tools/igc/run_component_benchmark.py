"""One bounded profiler-off component session; launch through qwenputer.headless."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from tools.igc.component_benchmark import normalize, summarize
from tools.qwenputer.memory import system_snapshot


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    args=parser.parse_args()
    out=args.evidence
    if (out/'identities.json').exists():
        raise RuntimeError('session_already_started')
    if os.environ.get('STRATA_TIMELINE'):
        raise RuntimeError('profiler_must_be_off')
    before=system_snapshot()
    gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.used,memory.free,driver_version',
                        '--format=csv,noheader'],capture_output=True,text=True,check=True,
                       timeout=15,creationflags=0x08000000)
    (out/'preflight.json').write_text(json.dumps(dict(system=before,gpu=gpu.stdout,
                           cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),indent=2),encoding='utf-8')
    # This is a small standalone component, never a changed hub admission policy.
    if before['physical_available_bytes']<2*1024**3 or before['system_commit_available_bytes']<2*1024**3:
        raise RuntimeError('insufficient_component_test_headroom')
    paths=[args.exe,args.artifact,args.binding,out/'corpus.bin',out/'corpus.json',
           Path('tests/igc/component_benchmark.cpp'),Path('src/kernels/qwenputer.cu'),
           Path('src/kernels/qwenputer_runner.cu'),Path('src/kernels/igc_exact.cu')]
    identities={str(p):digest(p) for p in paths}
    (out/'identities.json').write_text(json.dumps(identities,indent=2),encoding='utf-8')
    manifest=json.loads((out/'corpus.json').read_text(encoding='utf-8'))
    runs=[]
    for i,arm in enumerate(['qwenputer','exact','exact','qwenputer','paired']):
        command=[str(args.exe),str(args.artifact),str(args.binding),str(out/'corpus.bin'),arm,'2']
        started=time.time()
        begin=time.perf_counter()
        run=subprocess.run(command,capture_output=True,text=True,timeout=300,creationflags=0x08000000)
        elapsed_ms=(time.perf_counter()-begin)*1000
        name=f'{i}-{arm}'
        (out/f'{name}.jsonl').write_text(run.stdout,encoding='utf-8')
        (out/f'{name}.stderr').write_text(run.stderr,encoding='utf-8')
        if run.returncode:
            raise RuntimeError(f'{name}: exit={run.returncode}: {run.stderr}')
        rows=[json.loads(line) for line in run.stdout.splitlines()]
        for r in rows:
            if r['kind'] in ['cold','warmup','sample']:
                actual=normalize(r['actual_raw'],r['arm'])
                if r['status']!=0 or actual!=manifest['cases'][r['case']]['expected']:
                    raise RuntimeError(f'{name}: offline exact check failed: {r!r}')
        runs.append(dict(name=name,command=command,started_unix=started,process_wall_ms=elapsed_ms,
                         raw_sha256=digest(out/f'{name}.jsonl'),system_after=system_snapshot()))
        (out/'runs.json').write_text(json.dumps(runs,indent=2),encoding='utf-8')
        print(f'completed {name} process_wall_ms={elapsed_ms:.3f} exact_check=pass',flush=True)
    summarize(out/'4-paired.jsonl',out/'corpus.json',out/'summary.json')
    if any(digest(path)!=identities[str(path)] for path in paths):
        raise RuntimeError('input_or_executable_changed_during_session')
    (out/'complete.json').write_text(json.dumps(dict(all_frozen_inputs_unchanged=True,
        output_checks='all cold, warmup and measured results exact',system_after=system_snapshot()),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
