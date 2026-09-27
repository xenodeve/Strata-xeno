"""Compare raw greedy token IDs across CPU-only and GPU-cache placement.

Explicitly runs a model. Use only during authorized GPU time. Every child is owned
by this script; timeout cleanup never targets another process or server.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
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
    p.add_argument('--case', choices=['sky', 'thai', 'code', 'long'], default='sky')
    args = p.parse_args()
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
    env['CUDA_VISIBLE_DEVICES'] = '1'
    env['PATH'] = os.pathsep.join(cfg.get('lib_dirs', []) + [env.get('PATH', '')])
    results = {}
    for arm in (['off', 'on'] if args.arm == 'both' else [args.arm]):
        command = [args.exe or cfg['exe'], *base, '--tokens-file', str(prompt_path), '--max-new', str(args.max_new), '--greedy', '--max-context', '4096', '--pool-workers', '6', '--adapt-swaps', '0', '--pcie-frac', '0', '--expert-cache', str(args.slots), '--expert-profile', str(Path(cfg['cwd']) / 'data' / 'expert-profile.bin'), '--no-prefill-borrow']
        if arm == 'off':
            command.append('--cache-cpu-only')
        (args.out / f'{arm}.command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
        t0 = time.monotonic()
        print(f'RUN {arm} pid owner=cache_tokens timeout={args.timeout}s', flush=True)
        with (args.out / f'{arm}.stdout').open('w', encoding='utf-8') as stdout, (args.out / f'{arm}.stderr').open('w', encoding='utf-8') as stderr:
            run_code = subprocess.run(command, cwd=cfg['cwd'], env=env, stdout=stdout, stderr=stderr, timeout=args.timeout).returncode
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
