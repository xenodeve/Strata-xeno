# MIP Native: natural-language firing spec

Tracks #276, PRD R12 and `Strata-xeno_MIP_Native_Handoff_v2.md`. Decisions are in the issue comment
`igc:sidecar-decisions-20261031`. This file holds the contracts a build follows; nothing here is built yet.

Status: draft for the developer's review.

## Goal

Catch calculation the model writes in natural language, on the same safety terms as the parser. This is a
requirement of the work. Measurements tune it and do not decide whether it exists.

## Levels

| Level | Example | Where the operands are |
|---|---|---|
| 0 | `68824 * 42716 =` | An expression before `=`; the parser reads it |
| 1 | `half of 180 is`, `5 pies * 8 pieces each` | In the same sentence |
| 2 | `the remaining apples cost` | Earlier in the context, or an earlier result |
| 3 | The model writes a result with no working | Nowhere readable; out of scope |

Off traces, labelled by value (`tools/igc/auto_label.py`, 293 GSM8K and 184 MATH-500 points after filtering):
29 / 30 / 41 % and 28 / 30 / 42 % at levels 0 / 1 / 2. Approximate; ambiguous points are dropped.

## Decode flow

1. Keep a quantity table: each number in the prompt and in generated text, with position and surrounding tokens.
   Inserted values join it. Values always come from tokens.
2. A slot is a sampled token that starts a number: a digit, or a leading space, `$`, ` $`, `-`, ` -` before one.
   It is intercepted before it enters KV or a draft is accepted.
3. Each source proposes `{operation, left, right, confidence}`, where left and right are table indices.
   Sources are the parser, the sidecar, and on supported models the activation probe with retrieval heads.
4. One fusion decision. Any disagreement on operation or operand does not fire.
5. Shared checks: operands exist in the table, operation fits operand kinds, executor answers inside its time limit.
6. Verify with the main model, then deliver as tokens or let the model continue.

No source fires alone. The main model's agreement is never a reason to fire, only its disagreement a reason not to.

## Request schema

```text
InlineRequest
  kind        evaluate | simplify | factor | solve | differentiate | integrate | limit | root
  program     postfix steps over table indices or literals
  backend     gpu_integer | gpu_rational | host
  source      parser | sidecar | activation | fused
  level       0 | 1 | 2
InlineResult
  status      value | refused | unsupported | timed_out | fault
  tokens      the delivered token ids, or none
  note        which source note was used
```

Only `evaluate` exists now. Each new kind states its backend. GPU first; `host` only where the GPU cannot yet do the
work, and it is named in the result. Nothing is rounded: a value with no exact form the backend can state is refused.

## Fusion rules, first version

| Situation | Decision |
|---|---|
| Parser matched | Continue to checks |
| Sidecar and activation agree on operation and operands | Continue at the normal threshold |
| One source only | Continue above the strict threshold |
| Sources disagree | No fire |
| Model without activation | Sidecar must pass the strict threshold |

## Veto

The value is treated as a draft and the verifier reads per-digit `p(value digit)`, `p(top)` and entropy.

| Model's state | Reading | Decision |
|---|---|---|
| Flat distribution | Unsure | May fire |
| Peaked elsewhere at digit count or leading digits | Wrong operand or operation is likely | Veto |
| Peaked elsewhere at trailing digits only | The model's arithmetic is the likely fault | May fire if the rest passes |

The sampled digit at the slot is a free first signal. Every veto logs the value, what the model favoured, the
operands and the source; these become hard negatives.

## Sidecar

- Predicts structure only: operation and table pointers. It never predicts a value.
- Runs at a slot, batch-prefilling tokens new since the last slot.
- Sizes swept at about 5M, 30M, 100M, counting non-embedding parameters. The smallest that meets precision with
  usable level 2 recall wins.
- Embedding frozen from `Qwen3.5-0.8B`, reduced to 128-256 dimensions. Ids from 248070 map to a zero vector.
- Template markers are removed or neutralised in training text. `<think>` and `</think>` stay.

## Measures

- Answers containing a wrong firing at or below 0.5 %. Per-firing target `1 - 0.005 / n`, with n measured after the
  sidecar is on. Claiming it needs at least 600 clean answers; report intervals.
- Recall at levels 1 and 2, the primary measure of this path.
- The rule pointer is the floor: GSM8K precision 46/411, recall 9/209.
- Offline on labelled traces during development. Full GSM8K test decides. Test splits never train.
- ABBA of parser, parser plus sidecar, and the full system, for tokens, time and accuracy.

## Open

Delivery wording at levels 1 and 2, where ` (exact).` may not fit a sentence. Whether a closed-thinking guard is safe
on multi-step problems. Whether the indexer gives a cheaper pointer than retrieval heads.
