"""The 4B's supervised round on its own forced-block trajectories (tir4b.py keep), after START's Hint-RFT: the open line and the close
are in the loss, so the model learns to write them itself; what the kernel printed is masked, as ReTool masks the interpreter's output.

bf16 weights and a bf16 LoRA, no quantisation, as START, ReTool and CoRT train (developer, 2026-11-03: the precision matches the research);
the adapter is merged and written in the checkpoint's own format, so vLLM serves the tuned model the way it serves the base.
tir4b_train.py --model M --data TRAIN.jsonl --out DIR [--rank 32] [--alpha 64] [--lr 1e-4] [--epochs 2] [--accum 16]"""
import argparse, json, math, random, shutil, time
from pathlib import Path


def labels_of(prompt_ids, out_ids, mask):
    """-100 on the prompt and on each masked span of the answer, the answer's own token everywhere else."""
    labels = [-100] * len(prompt_ids) + list(out_ids)
    for start, end in mask:
        for i in range(len(prompt_ids) + start, len(prompt_ids) + end):
            labels[i] = -100
    return labels


def decoder_of(model):
    """The text decoder under the LoRA wrapper: the module that holds the embeddings, the layers and the final norm."""
    for _, m in model.named_modules():
        if all(hasattr(m, a) for a in ('embed_tokens', 'layers', 'norm')) and 'visual' not in type(m).__name__.lower():
            return m
    raise RuntimeError('no text decoder found')


def chunked_loss(hidden, labels, head, chunk=4096):
    """Cross-entropy at the labelled positions only, through the output head a chunk at a time: the whole logits of a 32k-token answer
    over a 248k vocabulary would be 32 GB in fp32."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    h = hidden[0, :-1]
    y = labels[0, 1:]
    keep = y != -100
    h, y = h[keep], y[keep]
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, len(y), chunk):
        total = total + checkpoint(lambda a, b: F.cross_entropy(head(a).float(), b, reduction='sum'), h[i:i + chunk], y[i:i + chunk], use_reentrant=False)
    return total / max(1, len(y)), len(y)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--rank', type=int, default=32)
    p.add_argument('--alpha', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--accum', type=int, default=16)
    p.add_argument('--warmup', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args(argv)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = [json.loads(l) for l in Path(args.data).read_text(encoding='utf-8').splitlines() if l.strip()]
    # The checkpoint's own class (Qwen3_5ForConditionalGeneration): its state dict keeps the file's names, so the merged weights load as the base does.
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='cuda')
    model.config.use_cache = False
    decoder = decoder_of(model)
    names = sorted({n.rsplit('.', 1)[-1] for n, m in decoder.named_modules() if isinstance(m, torch.nn.Linear)})
    # Every linear projection of the text decoder, the Gated DeltaNet ones too; the vision tower and the head stay as they are.
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, target_modules=names, exclude_modules=r'.*visual.*'))
    model.print_trainable_parameters()
    (out / 'lora.json').write_text(json.dumps(dict(vars(args), target_modules=names)), encoding='utf-8')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
    model.enable_input_require_grads()
    decoder = decoder_of(model)
    head = model.get_output_embeddings()
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    steps = math.ceil(len(data) * args.epochs / args.accum)
    warm = max(1, int(steps * args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    log = open(out / 'train.log', 'a', encoding='utf-8')
    order = random.Random(args.seed)
    step = seen = 0
    began = time.monotonic()
    model.train()
    for epoch in range(args.epochs):
        batch = list(range(len(data)))
        order.shuffle(batch)
        run_loss = run_tokens = 0
        for n, i in enumerate(batch, 1):
            d = data[i]
            ids = torch.tensor([d['prompt_ids'] + d['out_ids']], device='cuda')
            labels = torch.tensor([labels_of(d['prompt_ids'], d['out_ids'], d['mask'])], device='cuda')
            hidden = decoder(input_ids=ids).last_hidden_state
            loss, count = chunked_loss(hidden, labels, head)
            (loss / args.accum).backward()
            run_loss += loss.item() * count
            run_tokens += count
            seen += 1
            if n % args.accum == 0 or n == len(batch):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                line = dict(step=step, of=steps, epoch=epoch, loss=round(run_loss / max(1, run_tokens), 4), tokens=run_tokens,
                            lr=sched.get_last_lr()[0], seconds=round(time.monotonic() - began, 1))
                log.write(json.dumps(line) + '\n')
                log.flush()
                print(line, flush=True)
                run_loss = run_tokens = 0
    model.save_pretrained(out / 'adapter')
    merged = model.merge_and_unload()
    merged.save_pretrained(out / 'merged', safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(out / 'merged')
    for f in Path(args.model).iterdir():   # the chat template, the processor and generation configs the base serves with
        # not the base's weight index: it names the base's shards, and vLLM then looked for files the merge never wrote
        if f.suffix in ('.json', '.jinja', '.txt') and not f.name.endswith('.index.json') and not (out / 'merged' / f.name).exists():
            shutil.copy(f, out / 'merged' / f.name)
    print('DONE', round(time.monotonic() - began, 1), flush=True)


if __name__ == '__main__':
    main()
