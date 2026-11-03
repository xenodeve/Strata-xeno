"""Qwen3.5-4B on vLLM with the sandbox rules of the Flash-Next engine, rebuilt here: the 4B fine-tune track (#288, PRD R13 layer 2).

Strata's engine runs only the Flash-Next family, so the 4B gets the same rules in Python, by the engine's own words and limits
(include/strata/program/igc_learned_diagnostic.hpp, include/strata/core/igc_sandbox_bridge.hpp):
- after `force_after` tokens of thinking a block opens at the next line end, once (`--igc-sandbox-after`, :552): forcing every N tokens looped;
- a block the model closes runs in the kernel and its output comes back inside the thinking (:460), whoever opened the block, so a fine-tuned
  model that opens its own blocks is measured by the same path;
- a run that finished and printed closes the thinking (`--igc-sandbox-close`, :512): with no close, sandbox alone cost 1.50x plain (#279);
- a thinking that loops is closed (`repeating`, the bench's copy of the engine's checks), and an end of turn inside the thinking after a run closes it.
Not here: MIP's inline arithmetic, which lives in the CUDA engine. Every text the runner put in is a span of the record, with its kind.

tir4b.py run --model M --rows R.jsonl --out DIR [--force-after 512] [--execute] [--close-fit] [--loop-close] [--samples K] [--seed S] [--ids]
tir4b.py math-train OUT.jsonl --n N [--levels 3,4,5] [--exclude ROWS.jsonl ...]     (on the VM: MATH train, never a test row)
tir4b.py keep DIR [DIR ...] --out TRAIN.jsonl [--max-len 32768]                      (the trajectory each question trains on)"""
import argparse, hashlib, json, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OPEN_LINE = 'Need a short Python program here.\n\n```python\n'   # kOpenBlock, as text
NOTHING = '[it ran and printed nothing: call your function and print the result]'   # kNothingPrinted
SPENT = '[no more runs]'
TOO_LONG = '[too long, not run]'
SHOWN = 1024   # kShownOutput, bytes
MAX_BLOCK = 2048   # kMaxBlockTokens
MAX_RUNS = 128   # kMaxRuns
CLOSE = '</think>\n\n'   # {kThinkClose, 271}
LOOP_CLOSE = '\n</think>\n\n'   # {198, kThinkClose, 271}
CHUNK = 1024   # thinking tokens between two loop checks; the engine checks every 16, so a close can come up to this much later
CONTEXT = 262144   # the model card's native context, all of it (developer, 2026-11-03)
# Qwen's precise set for thinking (model card). Its general set adds presence_penalty 1.5, which vLLM counts per request: every block the runner
# answers starts a new request and would reset it, so an arm with blocks would sample differently from one without.
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)
MATH_TAIL = '\n\nPlease reason step by step, and put your final answer within \\boxed{}.'   # the model card's own math prompt
TRAIN_KINDS = ('forced open', 'close after a fitting result')   # the words the model should learn to write itself, as START's Hint-RFT
MASK_KINDS = ('output',)   # what the kernel printed: read, never learned


def output_block(body):
    return '\n```output\n' + body + '\n```\n\n'


def shown_text(ok, shown):
    """What the model reads back, cut as the engine cuts it."""
    body = shown or ''
    if ok and not body.strip():
        body = NOTHING
    data = body.encode('utf-8')
    return data[:SHOWN].decode('utf-8', 'ignore') if len(data) > SHOWN else body


class Ids:
    """The special tokens of the 4B's tokenizer, looked up, never carried over from the Flash-Next pack."""
    def __init__(self, tokenizer, eos):
        self.think_close = tokenizer.convert_tokens_to_ids('</think>')
        self.ends = {tokenizer.convert_tokens_to_ids('<|im_end|>'), *eos}
        lines = [tokenizer.encode(s, add_special_tokens=False) for s in ('\n', '\n\n')]
        self.lines = {t[0] for t in lines if len(t) == 1}
        self.fence = {i for i in range(len(tokenizer)) if '```' in tokenizer.decode([i])}
        if not self.fence or len(self.lines) != 2 or not isinstance(self.think_close, int):
            raise RuntimeError('tokenizer: no fence, line or </think> token')
        # repeating() and line_back() read 198 and 271 as the line ends, the Flash-Next pack's ids: a tokenizer with others would blind them
        self.same_lines = self.lines == {198, 271}


class Arm:
    def __init__(self, force_after=0, execute=False, close_fit=False, loop_close=False, close_repeat=0):
        self.force_after, self.execute, self.close_fit, self.loop_close, self.close_repeat = force_after, execute, close_fit, loop_close, close_repeat


class Rollout:
    """One answer, a state machine the driver asks for its next request and hands each result to. Phases: think, seek (a line end for the
    forced open), peek (whether a bare fence opens python), block, answer."""
    def __init__(self, row, sample, prompt, arm, ids, tokenizer, seed):
        self.row, self.sample, self.prompt, self.arm, self.ids, self.tok, self.seed = row, sample, prompt, arm, ids, tokenizer, seed
        self.out, self.spans, self.events = [], [], []
        self.phase, self.thought, self.forced, self.block, self.runs, self.seen = 'think', 0, False, 0, 0, {}
        self.code_at = self.fence_at = None
        self.by = None
        self.pending = None
        self.finish = None
        self.kernel = None
        self.started = time.monotonic()
        self.seconds = None

    def left(self):
        return CONTEXT - len(self.prompt) - len(self.out) - 64

    def model_tokens(self):
        return len(self.out) - sum(e - s for s, e, _ in self.spans)

    def text(self):
        return self.tok.decode(self.out)

    def inject(self, text, kind):
        t = self.tok.encode(text, add_special_tokens=False)
        self.spans.append([len(self.out), len(self.out) + len(t), kind])
        self.out += t
        self.events.append(dict(kind=kind, at=len(self.out) - len(t), model_tokens=self.model_tokens()))

    def done(self, finish):
        self.finish = finish
        self.seconds = round(time.monotonic() - self.started, 3)
        if self.kernel:
            self.kernel.close()
            self.kernel = None

    def request(self):
        """(max_tokens, stop token ids) of the next generation, or None once it is done."""
        if self.finish:
            return None
        left = self.left()
        if left <= 0:
            self.done('length')
            return None
        stops = set(self.ids.ends)
        if self.phase == 'answer':
            return left, stops
        stops.add(self.ids.think_close)
        if self.phase == 'peek':
            return min(2, left), stops
        if self.phase == 'seek':
            return min(64, left), stops | self.ids.lines
        if self.phase == 'block':
            return min(MAX_BLOCK - self.block, left), stops | self.ids.fence
        most = left
        if self.arm.execute:
            stops |= self.ids.fence
        if self.arm.loop_close:
            most = min(most, CHUNK)
        if self.arm.force_after and not self.forced and self.thought < self.arm.force_after:
            most = min(most, self.arm.force_after - self.thought)
        return most, stops

    def take(self, new):
        """The tokens of one generation. True when a block waits to run."""
        self.out += new
        if self.phase != 'answer' and self.phase != 'block':
            self.thought += len(new)
        last = new[-1] if new else None
        if last in self.ids.ends:
            if self.phase == 'answer':
                self.done('stop')
            elif self.runs:   # an end of turn inside thinking would leave no answer after a call (:384)
                self.out.pop()
                self.inject(LOOP_CLOSE, 'end of turn in thinking, closed it')
                self.phase = 'answer'
            else:
                self.done('stop')
            return False
        if self.phase == 'answer':
            if self.left() <= 0:
                self.done('length')
            return False
        if last == self.ids.think_close:
            if self.phase == 'block':
                self.events.append(dict(kind='thinking closed in a block', at=len(self.out), model_tokens=self.model_tokens()))
            self.phase = 'answer'
            return False
        if self.phase == 'seek':
            if last in self.ids.lines:
                self.forced = True
                self.inject(OPEN_LINE, 'forced open')
                self.enter('forced')
            return False
        if self.phase == 'peek':
            after = self.tok.decode(self.out[self.fence_at:])
            if after.startswith('python'):
                self.code_at = len(self.tok.decode(self.out[:self.fence_at])) + len('python')
                self.by, self.block, self.phase = 'model', 0, 'block'
                self.events.append(dict(kind='model opened a block', at=self.fence_at, model_tokens=self.model_tokens()))
            else:
                self.phase = 'think'
            return False
        if self.phase == 'block':
            self.block += len(new)
            if last in self.ids.fence:
                text = self.text()
                code = text[self.code_at:text.rfind('```')]
                code = code[1:] if code.startswith('\n') else code
                if not code.strip():
                    self.phase = 'think'
                    return False
                self.pending = code
                return True
            if self.block >= MAX_BLOCK:
                self.inject('\n```' + output_block(TOO_LONG), 'block too long')
                self.phase = 'think'
            return False
        # thinking
        if last in self.ids.fence and self.arm.execute:
            if '```python' in self.tok.decode([last]):
                self.code_at = len(self.text())
                self.by, self.block, self.phase = 'model', 0, 'block'
                self.events.append(dict(kind='model opened a block', at=len(self.out) - 1, model_tokens=self.model_tokens()))
            else:
                self.fence_at, self.phase = len(self.out), 'peek'
            return False
        if self.arm.force_after and not self.forced and self.thought >= self.arm.force_after:
            self.phase = 'seek'
            return False
        if self.arm.loop_close and looping(self.out):
            self.inject(LOOP_CLOSE, 'loop close')
            self.phase = 'answer'
        return False

    def enter(self, by):
        self.code_at, self.by, self.block, self.phase = len(self.text()), by, 0, 'block'

    def run_pending(self, kernel_factory):
        """Runs the block the model closed, then puts its output and, after a fitting result, the close into the thinking."""
        code, self.pending = self.pending, None
        if self.arm.close_repeat and self.seen.get(code, 0) + 1 >= self.arm.close_repeat:
            self.inject(LOOP_CLOSE, 'repeated program close')
            self.phase = 'answer'
            return
        self.seen[code] = self.seen.get(code, 0) + 1
        began = time.monotonic()
        if self.runs >= MAX_RUNS:
            ok, shown = None, SPENT
        else:
            if self.kernel and not self.kernel.alive():
                self.kernel.close()
                self.kernel = None
            self.kernel = self.kernel or kernel_factory()
            ok, shown = self.kernel.run(code)
        self.runs += 1
        took = round(time.monotonic() - began, 3)
        fits = bool(ok) and bool((shown or '').strip())   # printed(): it finished and said something
        at = len(self.out)
        self.inject(output_block(shown_text(ok, shown)), 'output')
        self.events[-1].update(kind='output', by=self.by, code=code[:20000], code_chars=len(code), shown=(shown or '')[:4096],
                               shown_chars=len(shown or ''), ok=ok, seconds=took, fits=fits, run_at=at)
        if self.arm.close_fit and fits:
            self.inject(CLOSE, 'close after a fitting result')
            self.phase = 'answer'
        else:
            self.phase = 'think'

    def record(self, right):
        text = self.text()
        runs = [e for e in self.events if e['kind'] == 'output']
        return dict(id=self.row['id'], suite=self.row['suite'], answer=self.row['answer'], sample=self.sample, seed=self.seed, text=text,
                    right=right(self.row, text), finish=self.finish, tokens=self.model_tokens(), injected=len(self.out) - self.model_tokens(),
                    prompt_tokens=len(self.prompt), thinking_tokens=self.thought, seconds=self.seconds, events=self.events,
                    runs=len(runs), runs_by_model=sum(e['by'] == 'model' for e in runs), runs_forced=sum(e['by'] == 'forced' for e in runs),
                    first_run_at=runs[0]['model_tokens'] if runs else None, tool_seconds=round(sum(e['seconds'] for e in runs), 3))


def looping(tokens):
    from tools.igc.external_score import repeating
    return repeating(tokens)


def eos_ids(model, tok, generation_config):
    """The end tokens wherever the checkpoint states them: Qwen3.5-4B ships no generation_config.json, and its config keeps the id under text_config."""
    ids = {tok.eos_token_id} - {None}
    try:
        e = generation_config.from_pretrained(model).eos_token_id
        ids |= set(e if isinstance(e, list) else [e]) - {None}
    except OSError:
        pass
    config = json.loads((Path(model) / 'config.json').read_text(encoding='utf-8'))
    for c in (config, config.get('text_config', {})):
        e = c.get('eos_token_id')
        ids |= set(e if isinstance(e, list) else [e]) - {None}
    return sorted(ids)


def prompt_text(row):
    from tools.igc.standard_rows import plain_prompt
    text = row.get('question') or plain_prompt(row)
    return text + MATH_TAIL if row['suite'] == 'stdmath' else text


def seed_of(seed, row_id, sample):
    return int.from_bytes(hashlib.sha256(f'{seed}:{row_id}:{sample}'.encode()).digest()[:4], 'little')


def drive(llm, rollouts, kernel_factory, live_most=256, workers=32, on_round=None):
    """Every live answer's next request goes out in one batch; the blocks closed in that round run side by side; then the next round. At most
    `live_most` answers are live at once, each holding its kernel until it ends, so a few thousand answers do not hold a few thousand kernels."""
    from vllm import SamplingParams, TokensPrompt
    pool = ThreadPoolExecutor(workers)
    waiting_rows = list(rollouts)
    live = []
    while live or waiting_rows:
        while waiting_rows and len(live) < live_most:
            r = waiting_rows.pop(0)
            r.started = time.monotonic()
            live.append(r)
        asks = [(r, r.request()) for r in live]
        asks = [(r, q) for r, q in asks if q]
        if asks:
            prompts = [TokensPrompt(prompt_token_ids=r.prompt + r.out) for r, _ in asks]
            params = [SamplingParams(**SAMPLING, max_tokens=most, stop_token_ids=sorted(stops), seed=(r.seed + len(r.out)) % 2**31,
                                     skip_special_tokens=False, detokenize=False) for r, (most, stops) in asks]
            outs = llm.generate(prompts, params, use_tqdm=False)
            waiting = []
            for (r, _), o in zip(asks, outs):
                c = o.outputs[0]
                new = list(c.token_ids)
                if isinstance(c.stop_reason, int) and (not new or new[-1] != c.stop_reason):
                    new.append(c.stop_reason)
                if r.take(new):
                    waiting.append(r)
            list(pool.map(lambda r: r.run_pending(kernel_factory), waiting))
        live = [r for r in live if not r.finish]
        if on_round:
            on_round()
    pool.shutdown()


def run(args):
    import os
    os.environ.setdefault('STRATA_SANDBOX_THROWAWAY', '1')   # the kernel runs bare only on a VM that is thrown away (exec_bridge.throwaway)
    from serve.sandbox.exec_bridge import Kernel, throwaway
    from tools.igc.select_rows import right
    from transformers import AutoTokenizer, GenerationConfig
    from vllm import LLM
    if args.execute and not throwaway():
        raise SystemExit('a block runs only on a throwaway VM')
    rows = [json.loads(l) for l in Path(args.rows).read_text(encoding='utf-8').splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = Ids(tok, eos_ids(args.model, tok, GenerationConfig))
    arm = Arm(args.force_after, args.execute, args.close_fit, args.loop_close, args.close_repeat)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rollouts = []
    for row in rows:
        prompt = tok.apply_chat_template([{'role': 'user', 'content': prompt_text(row)}], tokenize=True, add_generation_prompt=True, enable_thinking=True)
        for k in range(args.samples):
            rollouts.append(Rollout(row, k, list(prompt), arm, ids, tok, seed_of(args.seed, row['id'], k)))
    if args.force_after and not args.execute:
        raise SystemExit('--force-after opens a block that only --execute runs')
    if args.loop_close and not ids.same_lines:
        raise SystemExit(f'--loop-close: the line ends are {sorted(ids.lines)}, not the 198 and 271 the loop checks read')
    began = time.monotonic()
    kwargs = dict(model=args.model, dtype='bfloat16', max_model_len=CONTEXT, gpu_memory_utilization=args.memory, enable_prefix_caching=True,
                  language_model_only=True, seed=args.seed)
    try:
        llm = LLM(**kwargs)
    except TypeError:   # a vLLM with no text-only switch loads the vision tower too, which costs memory and changes nothing here
        kwargs.pop('language_model_only')
        llm = LLM(**kwargs)
    (out / 'arm.json').write_text(json.dumps(dict(vars(args), sampling=SAMPLING, context=CONTEXT, load_seconds=round(time.monotonic() - began, 1),
                                                   llm={k: v for k, v in kwargs.items() if k != 'model'}, think_close=ids.think_close,
                                                   ends=sorted(ids.ends), lines=sorted(ids.lines), fences=len(ids.fence))), encoding='utf-8')
    began = time.monotonic()
    written = [0, 0.]

    def save(last=False):
        done = [r for r in rollouts if r.finish]
        if len(done) == written[0] or (not last and time.monotonic() - written[1] < 30):
            return
        written[0], written[1] = len(done), time.monotonic()
        records = [r.record(right) for r in done]
        (out / 'records.json').write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')
        if args.ids:
            with open(out / 'ids.jsonl', 'w', encoding='utf-8') as f:
                for r in done:
                    f.write(json.dumps(dict(id=r.row['id'], sample=r.sample, prompt_ids=r.prompt, out_ids=r.out, spans=r.spans)) + '\n')
        print(f'{len(done)}/{len(rollouts)} done, {sum(x["right"] for x in records)} right, {round(time.monotonic() - began)} s', flush=True)
    drive(llm, rollouts, Kernel, live_most=args.live, on_round=save)
    save(last=True)
    records = [r.record(right) for r in rollouts]
    (out / 'report.json').write_text(json.dumps(dict(rows=len(rows), samples=args.samples, answers=len(records), right=sum(r['right'] for r in records),
                                                      tokens=sum(r['tokens'] for r in records), seconds=round(time.monotonic() - began, 1),
                                                      runs=sum(r['runs'] for r in records), runs_by_model=sum(r['runs_by_model'] for r in records),
                                                      length=sum(r['finish'] == 'length' for r in records))), encoding='utf-8')
    print('DONE', flush=True)


def math_train(args):
    """MATH's train split, levels picked, gold from the last \\boxed of the solution. A row whose id matches an evaluated row is left out:
    the ids are the hash of the problem (question_id), the same as standard_rows builds for the test rows."""
    from datasets import load_dataset
    from tools.igc.external_score import question_id
    from tools.igc.standard_score import last_boxed, normalize_math
    banned = set()
    for path in args.exclude:
        banned |= {json.loads(l)['id'] for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()}
    levels = {f'Level {n}' for n in args.levels.split(',')}
    rows = []
    for subject in ('algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'):
        for r in load_dataset('EleutherAI/hendrycks_math', subject, split='train'):
            gold = last_boxed(r['solution'])
            rid = question_id('stdmath', r['problem'])
            if r['level'] in levels and gold and rid not in banned:
                rows.append(dict(id=rid, suite='stdmath', question=r['problem'], answer=normalize_math(gold), group=subject, level=r['level']))
    random.Random(args.seed).shuffle(rows)
    Path(args.out).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows[:args.n]), encoding='utf-8')
    print('rows', min(args.n, len(rows)), 'of', len(rows), 'left out as evaluated', len(banned), flush=True)


def choose(records):
    """Per question, the trajectory it trains on: right, ended, no text of ours but the open and the close, and the shortest. A run whose result
    fitted is preferred, since that is the behaviour taught; a question answered before any block opened keeps its short tool-free answer, so the
    model also learns that an easy question needs no program (PRD R13)."""
    by = {}
    for r in records:
        kinds = {e['kind'] for e in r['events']} - set(TRAIN_KINDS) - set(MASK_KINDS) - {'model opened a block'}
        if r['right'] and r['finish'] == 'stop' and not kinds:
            by.setdefault(r['id'], []).append(r)
    chosen = {}
    for rid, rs in by.items():
        fitted = [r for r in rs if any(e.get('fits') for e in r['events'])]
        plain = [r for r in rs if not r['runs']]
        pick = fitted or plain
        if pick:
            chosen[rid] = min(pick, key=lambda r: r['tokens'] + r['injected'])
    return chosen


def keep(args):
    records, traj = [], {}
    for d in args.dirs:
        records += json.loads((Path(d) / 'records.json').read_text(encoding='utf-8'))
        for l in (Path(d) / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
            t = json.loads(l)
            traj[(t['id'], t['sample'])] = t
    chosen = choose(records)
    n = long = 0
    with open(args.out, 'w', encoding='utf-8') as f:
        for rid, r in chosen.items():
            t = traj[(rid, r['sample'])]
            if len(t['prompt_ids']) + len(t['out_ids']) > args.max_len:
                long += 1
                continue
            f.write(json.dumps(dict(id=rid, sample=r['sample'], prompt_ids=t['prompt_ids'], out_ids=t['out_ids'],
                                    mask=[s[:2] for s in t['spans'] if s[2] in MASK_KINDS], runs=r['runs'])) + '\n')
            n += 1
    print(f'questions {len({r["id"] for r in records})}, kept {n} ({sum(c["runs"] > 0 for c in chosen.values())} with a fitting run), '
          f'over {args.max_len} tokens {long}', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--model', required=True)
    r.add_argument('--rows', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--samples', type=int, default=1)
    r.add_argument('--seed', type=int, default=1)
    r.add_argument('--limit', type=int, default=0)
    r.add_argument('--force-after', type=int, default=0)
    r.add_argument('--execute', action='store_true')
    r.add_argument('--close-fit', action='store_true')
    r.add_argument('--loop-close', action='store_true')
    r.add_argument('--close-repeat', type=int, default=0)
    r.add_argument('--ids', action='store_true')
    r.add_argument('--memory', type=float, default=0.9)
    r.add_argument('--live', type=int, default=256)
    m = sub.add_parser('math-train')
    m.add_argument('out')
    m.add_argument('--n', type=int, required=True)
    m.add_argument('--levels', default='3,4,5')
    m.add_argument('--seed', type=int, default=1)
    m.add_argument('--exclude', nargs='*', default=[])
    k = sub.add_parser('keep')
    k.add_argument('dirs', nargs='+')
    k.add_argument('--out', required=True)
    k.add_argument('--max-len', type=int, default=32768)
    args = p.parse_args(argv)
    {'run': run, 'math-train': math_train, 'keep': keep}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
