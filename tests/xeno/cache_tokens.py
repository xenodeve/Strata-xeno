"""Compare raw greedy token IDs across CPU-only and GPU-cache placement.

Explicitly runs a model. Use only during authorized GPU time. Every child is owned
by this script; timeout cleanup never targets another process or server.
"""
import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate
from tools.strata_tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--exe')
    p.add_argument('--max-new', type=int, default=96)
    p.add_argument('--slots', type=int, default=2048)
    p.add_argument('--timeout', type=int, default=240)
    p.add_argument('--arm', choices=['off', 'on', 'both'], default='both')
    p.add_argument('--case', choices=['sky', 'thai', 'code', 'long', 'prefill'], default='sky')
    p.add_argument('--devices', default='1', help='CUDA_VISIBLE_DEVICES order (Phase 3: 1,0)')
    p.add_argument('--secondary-expert-mib', type=int, default=0)
    p.add_argument('--secondary-free-floor-mib', type=int)
    p.add_argument('--primary-reserve-mib', type=int)
    p.add_argument('--prefill-chunk', type=int)
    p.add_argument('--max-context', type=int, default=4096)
    p.add_argument('--spec', type=int)
    p.add_argument('--sample-vram', action='store_true')
    p.add_argument('--profile', type=Path)
    p.add_argument('--secondary-stage-only', action='store_true')
    p.add_argument('--exclusive-primary', action='store_true')
    args = p.parse_args()
    if args.secondary_expert_mib < 0 or (args.secondary_expert_mib and args.devices != '1,0'):
        p.error('secondary experts require --devices 1,0 and non-negative MiB')
    if args.secondary_stage_only and not args.secondary_expert_mib:
        p.error('--secondary-stage-only requires --secondary-expert-mib')
    if args.exclusive_primary and args.arm != 'on':
        p.error('--exclusive-primary cannot use the forced CPU-miss arm')
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = Path(cfg['tokenizer'])
    vocab = json.loads((tokenizer_dir / 'vocab.json').read_text(encoding='utf-8'))
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tokenizer = Tokenizer(tokens, (tokenizer_dir / 'merges.txt').read_text(encoding='utf-8').split('\n'), json.loads((tokenizer_dir / 'token_type.json').read_text(encoding='utf-8')))
    template = ChatTemplate(tokenizer_dir / 'chat_template.jinja')
    prompts = {
        'sky': 'Explain why the sky is blue in three short sentences.',
        'thai': 'เขียนย่อหน้าสั้น ๆ อธิบายว่าทำไมการนอนหลับจึงสำคัญต่อสุขภาพ',
        'code': 'Write a Python function that returns the n-th Fibonacci number iteratively.',
        'long': 'Write a detailed technical explanation of how a B-tree works, including insertion, deletion, splitting, and examples.',
        # ~3.3K prompt tokens: exercises more than one 2048-token prefill chunk.
        'prefill': '\n\n'.join(f'Section {i}: A B-tree node {i} holds sorted keys; insertion splits a full node at its median, deletion merges or borrows from a sibling, and every leaf stays at the same depth.' for i in range(1, 91)) + '\n\nSummarise the sections above in five bullet points.',
    }
    prompt = template.render([{'role': 'user', 'content': prompts[args.case]}], enable_thinking=False)
    ids = tokenizer.encode(prompt, parse_special=True)
    prompt_path = args.out / 'prompt.ids'
    prompt_path.write_text(','.join(map(str, ids)), encoding='utf-8')
    values = {'--expert-cache', '--expert-profile', '--max-context', '--adapt-swaps', '--pcie-frac', '--pool-workers'}
    base = []
    i = 0
    while i < len(cfg['args']):
        key = cfg['args'][i]
        if key in values:
            i += 2
        else:
            base.append(key)
            i += 1
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = args.devices
    env['PATH'] = os.pathsep.join(cfg.get('lib_dirs', []) + [env.get('PATH', '')])
    results = {}
    for arm in (['off', 'on'] if args.arm == 'both' else [args.arm]):
        command = [args.exe or cfg['exe'], *base, '--tokens-file', str(prompt_path), '--max-new', str(args.max_new), '--greedy', '--max-context', str(args.max_context), '--pool-workers', '6', '--adapt-swaps', '0', '--pcie-frac', '0', '--expert-cache', str(args.slots), '--expert-profile', str(args.profile or Path(cfg['cwd']) / 'data' / 'expert-profile.bin'), '--no-prefill-borrow']
        if args.primary_reserve_mib is not None:
            command += ['--vram-reserve-mib', str(args.primary_reserve_mib)]
        if args.prefill_chunk is not None:
            command += ['--prefill', str(args.prefill_chunk)]
        if args.spec is not None:
            command += ['--spec', str(args.spec)]
        if args.secondary_free_floor_mib is not None:
            command += ['--secondary-free-floor-mib', str(args.secondary_free_floor_mib)]
        if args.secondary_expert_mib:
            command += ['--secondary-expert-mib', str(args.secondary_expert_mib)]
        if args.secondary_stage_only:
            command.append('--secondary-stage-only')
        if args.exclusive_primary:
            command.append('--exclusive-primary-experts')
        if arm == 'off':
            command.append('--cache-cpu-only')
        (args.out / f'{arm}.command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
        t0 = time.monotonic()
        print(f'RUN {arm} pid owner=cache_tokens timeout={args.timeout}s', flush=True)
        samples = []
        stop_sampling = threading.Event()
        last_cpu = [None]
        def cpu_busy_pct():
            """System-wide CPU busy % since the previous sample (GetSystemTimes; kernel time includes idle)."""
            idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
            if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return -1
            now, prev = (idle.value, kern.value + user.value), last_cpu[0]
            last_cpu[0] = now
            if prev is None or now[1] == prev[1]:
                return -1
            return round(100.0 * (1.0 - (now[0] - prev[0]) / (now[1] - prev[1])), 1)
        def sample_vram():
            while not stop_sampling.is_set():
                try:
                    probe = subprocess.run(
                        ['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu',
                         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
                    cpu = cpu_busy_pct()
                    if probe.returncode == 0:
                        for line in probe.stdout.splitlines():
                            gpu, used, free, util = (int(part.strip()) for part in line.split(','))
                            samples.append((round(time.monotonic() - t0, 2), gpu, used, free, util, cpu))
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
                stop_sampling.wait(0.5)
        sampler = threading.Thread(target=sample_vram, daemon=True) if args.sample_vram else None
        if sampler is not None:
            sampler.start()
        try:
            with (args.out / f'{arm}.stdout').open('w', encoding='utf-8') as stdout, (args.out / f'{arm}.stderr').open('w', encoding='utf-8') as stderr:
                run_code = subprocess.run(command, cwd=cfg['cwd'], env=env, stdout=stdout, stderr=stderr, timeout=args.timeout).returncode
        finally:
            stop_sampling.set()
            if sampler is not None:
                sampler.join(timeout=4)
                with (args.out / f'{arm}.vram.csv').open('w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(('elapsed_s', 'gpu', 'used_mib', 'free_mib', 'gpu_util_pct', 'cpu_busy_pct'))
                    writer.writerows(samples)
        text = (args.out / f'{arm}.stdout').read_text(encoding='utf-8')
        token_line = next((line for line in text.splitlines() if line.startswith('output') and ':' in line), None)
        if run_code or token_line is None:
            raise RuntimeError(f'{arm}: exit={run_code}; inspect {args.out / (arm + ".stderr")}')
        tokens = [int(t) for t in token_line.split(':', 1)[1].split()]
        results[arm] = {'tokens': tokens, 'seconds': time.monotonic() - t0, 'exe_sha256': hashlib.sha256(Path(command[0]).read_bytes()).hexdigest()}
        (args.out / f'{arm}.result.json').write_text(json.dumps(results[arm], indent=2), encoding='utf-8')
        print(f'{arm}: {len(tokens)} tokens in {results[arm]["seconds"]:.1f}s including load', flush=True)
    if args.arm == 'both':
        a, b = results['off']['tokens'], results['on']['tokens']
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)) if len(a) != len(b) else None)
        result = {'equal': first is None, 'first_difference': first, 'off_count': len(a), 'on_count': len(b)}
        (args.out / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
        return 0 if first is None else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
