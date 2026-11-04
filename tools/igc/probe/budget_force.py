"""budget_force.py MODEL ROWS OUT ARMDIR [ARMDIR ...]: what the model answers when the token budget is spent, asked for its best answer then.

The thinking budget of Qwen3 and of s1: when the budget ends, the reply is closed with a short sentence, `</think>`, and the model writes its answer from what it
has thought so far. It cuts nothing before the budget, so it cannot take an answer from a reply that would have finished; it only acts on the replies that were cut
at the cap or stopped in a loop, which have no answer at all. Compared in #298 with the regex fallback (the last answer the model had written).

For every reply cut at the cap (finish `length` or `stuck`) of each ARMDIR (a tir4b --ids output with records.json): prefix = prompt + the reply's tokens, then the
budget sentence and `</think>`, then up to 400 tokens sampled at T 0.6 / top-p 0.95 / top-k 20 (seed 1). Not scored here (the scorer needs sympy and lark): the answers are written to OUT, `budget_score.py` scores them.
Replies that finished are not touched. Needs a GPU for the model alone; no vLLM beside it."""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BUDGET = '\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n'


def main():
    model_dir, rows_path, out_path, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map='cuda:0').eval()
    result, t0 = {}, time.monotonic()
    for arm in arms:
        d = Path(arm)
        recs = {r['id']: r for r in json.loads((d / 'records.json').read_text(encoding='utf-8'))}
        traces = {}
        for line in (d / 'ids.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                t = json.loads(line)
                traces[t['id']] = t
        cut = [i for i, r in recs.items() if r.get('finish') in ('length', 'stuck') and i in traces]
        out = {}
        for k, i in enumerate(cut):
            t = traces[i]
            ids = t['prompt_ids'] + t['out_ids'] + tok.encode(BUDGET, add_special_tokens=False)
            torch.manual_seed(1)
            with torch.no_grad():
                g = model.generate(torch.tensor([ids], device='cuda:0'), max_new_tokens=400, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
            answer = tok.decode(g[0, len(ids):], skip_special_tokens=True)
            out[i] = dict(finish=recs[i].get('finish'), answer=answer, thinking_tail=tok.decode(t['out_ids'][-300:]))
            print(f'  {d.name} {k + 1}/{len(cut)} {i}: ...{answer[-100:]!r}', flush=True)
        n = len(cut)
        result[d.name] = dict(cut=n, rows=out)
        print(f'{d.name}: {n} cut replies answered ({time.monotonic() - t0:.0f} s)', flush=True)
    Path(out_path).write_text(json.dumps(result, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
