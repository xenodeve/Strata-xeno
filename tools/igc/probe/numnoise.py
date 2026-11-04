"""numnoise.py: the control for htab.py part 1. Forcing the value's tokens by the sampler gave a greedy continuation identical to inserting
them in 25 of 30 points. Is the rest numerical noise (bf16: a one-pass prefill against token-by-token steps, or a different batch size) or a
difference in what the model reads? Here the SAME insertion (value in the prefix, greedy 48 tokens) is run twice, as one row and as three
identical rows; if the continuations part at a similar rate, bf16 noise is the whole story."""
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, IDS, POINTS = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda:0').eval()
rows = {json.loads(l)['id']: json.loads(l) for l in open(IDS, encoding='utf-8')}
pts = json.load(open(POINTS))[:30]
EOS = [t for t in (tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>')) if t is not None]


def run(prefix, k):
    x = torch.tensor([prefix], device='cuda:0').repeat(k, 1)
    with torch.no_grad():
        o = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=48, do_sample=False, pad_token_id=EOS[0], eos_token_id=EOS)
    return o[:, x.shape[1]:].cpu().tolist()


same, div = 0, []
for p in pts:
    r = rows[p['id']]
    prefix = r['prompt_ids'] + r['out_ids'][:p['start'] + p['n']]
    a = run(prefix, 1)[0]
    b = run(prefix, 3)[0]
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        same += 1
    else:
        div.append(next(i for i in range(n) if a[i] != b[i]))
print(f'NUMNOISE the same insertion at batch 1 and batch 3: greedy 48 tokens identical in {same} of {len(pts)} points; first divergence at {div}', flush=True)
