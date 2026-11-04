# MIP catch points: every place in the pipeline, ordered by the chance they help (2026-11-04, #298)

The developer asked for every point in an LLM's pipeline where MIP can notice a computation. They then said to write them all into the plan and to test them from the most likely to the least. Terms are defined in `CONTEXT.md`.

**How each point is judged before it is used.** Three numbers, all from the same answers:

- lead time: tokens before the model writes the computation, or before the text goes out;
- caught / should have been caught, with the denominator given;
- false alarms.

A point that acts on the model (injects, cuts or corrects) must also show that quality does not drop. That means right answers lost on a sample of the base's right rows, beside the misses it fixed. A point that loses quality or does not improve it is dropped at once (the standing rule).

**Where they run.**

- The target is strata-xeno on Qwen3.6-35B-A3B NVFP4.
- Speed work comes first. It owns the GPU slots it needs.
- A point that can be tested without the engine is tested while the speed work runs, on stored answers or on the reference runner (`tools/igc/tir4b.py` on llama.cpp, a quick check of an idea, not a result for strata-xeno).

## The order

| rank | point | why this rank (evidence) | first experiment | needs |
|---|---|---|---|---|
| 1 | **The question itself** (software reads its expressions) | 4 of the 22 DeepMind rows lost by `whole` were the model copying the question wrong (pc4-whole: 3 signs, 1 term). Works for any model and costs nothing | Offline, on `dm-rows`: the share of questions whose expression `mathexpr` reads, and whether that value equals gold. Then a runner arm that gives the question's value before the model starts | nothing (offline) |
| 2 | **The hand-off** (did the model use the value) | 28 of 160 DeepMind rows had the software's right value in the text and a wrong answer. Kinds: `.000…` after the value (fixed in 6b7d61da), `= v` repeated, a fraction rewritten as a long decimal, recomputing | Rules in MIP first (developer): cut a repeat of a value already given; stop a decimal expansion of a value already given exactly. Re-run DeepMind (5 minutes) against pc4-whole | runner, then engine |
| 3 | **The confirmed window** (fix before the text goes out) | dm-check showed the model distrusting a correction that came after its own wrong value. A wrong value the model never sees should not be distrusted | In the runner: instead of appending a Correction line, cut back to the `=` and put the right value in its place, then compare with check-hand. In the engine: the verify window, with GDN state rollback at the cut | runner, then q36 |
| 4 | **The shape of the thinking** (loops, repeated 'Wait', hypotheses → a program) | 13 of 15 r1 misses end in a loop, 10 with no arithmetic error. pc5-full gave 3 of 5 loops a program, but the box bug spoiled half the runs | Re-run the loop-hint arm with the box fixed (3aed1e9b) on the r1 misses, at a cap long enough that answers are not cut (pc5's 30K cut 6 of 7) | runner (box), then engine |
| 5 | **Next-token confidence on digits** (entropy) | No measurement yet. It costs nothing and works for any model, and would tell which hand values to check first | Offline, teacher-forced on stored answers (HF 4B): AUROC of digit entropy for the hand values the parser calls wrong (pc5-ctrl: 28 wrong of 579) | nothing (offline, GPU minutes) |
| 6 | **The MoE router** (experts that light up on arithmetic or code) | No measurement. Qwen3.6 has 256 experts; Strata can dump routing; the signal costs nothing | Routing traces of Qwen3.6 on answers with marked calculation spans; per-expert lift; AUROC of a router-only detector | q36 routing dump |
| 7 | **The raw draft trend** | No measurement; MTP proposes 1-2 tokens ahead | On q36: proposals that are digits or operators, against the computations that follow | q36 + MTP |
| 8 | **Hidden state during decode** (layer 1) | AUROC 0.922 on the 4B (text baseline 0.75). Must be calibrated per model | Retrain on Qwen3.6 hidden states; lead time and coverage beside 6 and 7 | q36 hidden-state dump |
| 9 | **Hidden state at the end of the prompt** | No measurement. One decision per question | The same probe, at the last prompt token: does this question need a program or a whole computation | q36 |
| 10 | **The final answer** (substitute back into the question) | Only some question kinds allow it | Offline: the share of LiveBench math and logic rows where an answer can be checked by software | nothing (offline) |
| 11 | **The text layers 2-5** (operators, `=`, fences, hand values) | Built; coverage measured (pc5-full: 77 % of computable `=` got the software's value, 4/14 wrong hand values corrected) | Keep; improve the 10 missed wrong values first (read why) | runner, then engine |
| 12 | **Attention** (copying a number against making one) | Research-heavy; no clear action of its own | Last | q36 |

## Running now (2026-11-04)

- Speed (first, decided): strata-q36 on the NVIDIA NVFP4 GGUF needed a loader fix (`[2048, 1]` shared-expert gate, `family_layout.hpp`). Next: baseline against llama.cpp on both cards in one session, then nsys on the 2K prompt.
- Point 1 offline, on stored DeepMind rows.
- Point 5 offline, once the GPU has a free slot.

## Hypothesis H-trust (developer, 2026-11-04): open, to be tested

> A model re-checks a value software gave it, again and again until the context is full, because the value arrives in a form the model can tell is not its own. A logit override, where the model writes the right value itself, should reduce that more than injected text does.

- **For it:** the developer has seen the behaviour many times, across many models. In this session it showed up three times: dm-check redid the work by hand after a Correction line, pc4-whole looped on `= -18`, and pc4-whole expanded a fraction into an endless decimal.
- **Against it, or at least to be explained:** the sign case. The model's own `= -18` line was kept, and P('-') was still only 0.014-0.225 (`probe/sign_drop.py`). That looks like belief, not form.
- **What could make an injection detectable:** a note or correction text the model never writes, and a tokenization that differs from the model's own. Token for token, identical text cannot be told apart.
- **Tests:**
  1. Teacher-forced, the model's probability of each injected value, and whether a low probability predicts a recheck or a loop afterwards.
  2. A/B on DeepMind: the current injection against an injection in the model's own tokens, which is the logit-override equivalent.

## The goal restated (developer, 2026-11-04): software's result acts inside the generation itself

"Not only the LLM calling software and reading the answer: the software's result should influence the LLM's generation mechanism directly."

| level | what MIP does | changes what the model believes | every model |
|---|---|---|---|
| logits / sampler | force the value's tokens; suppress reflection words after a verified value | no | yes |
| **residual stream** | at the position where the model writes a result, read the value it holds, and write software's value in its place, so the model writes the right value itself | **yes** | needs automatic calibration per model, like the probe |
| KV cache | edit the keys and values at the number's positions | partly | calibration |

**First experiment (4B, HF, offline): read and write a number in the residual stream.**
1. **Read:** a linear map from the hidden state at `=` (layer L) to the value the model is about to write. How exact is it, at which layer?
2. **Write:** add Δ = W(software's value) − W(the model's value) at that position. How often, out of how many should, does greedy decoding then write software's value by itself? Compare with no Δ.
3. **Re-checking:** P('Wait' / 'check' ...) after the value, against forcing the tokens.

The map is fitted from the model's own `a op b = c` runs, so it is calibrated per model automatically.
