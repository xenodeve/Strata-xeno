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

## The plan as of 2026-11-04, afternoon (supersedes the order above where they differ)

### A. Speed on the target, first (developer's decision)
| step | state |
|---|---|
| Merge v0.1.45 | done on `xeno/merge-0145`; GPU gates not run because the D2x shards are missing |
| Qwen3.6 NVFP4 on q36 | exists on `xeno/q36-nvfp4`; the NVIDIA GGUF loads after the `[2048,1]` shape fix; the logits gate against llama.cpp on this file is not run yet |
| Speed gap measured | strata-q36 prompt is 3-6.5x slower and decode 2-2.6x slower than llama.cpp on both cards. Root cause: the 5060 Ti on PCIe 4.0 x4 (7.0 GB/s) carries every host expert of every prompt chunk, because q36 has no expert split; the 4070 does 20.8-25 GB/s |
| Expert split ported to q36 | in progress (agent), `STRATA_Q36_PREFILL_SPLIT=1` opt-in |
| 4070 window graph (#227) for decode | after the split |

### B. MIP: software's result inside the generation (developer's restated goal)
Results so far are from the 4B reference runner unless marked. None is on strata-xeno or Qwen3.6 yet.

| # | mechanism | result | next |
|---|---|---|---|
| 1 | the question's own computation (`question_read`) | 160/160 offline; in pc9 it fired on 4 of 4 copy errors | on LiveBench: the share of questions with one stated computation |
| 2 | hand-off rules | pc6 146, pc7 143 (reverted), pc8 151 (close with the restated value), **pc9 156/160** (+ close with software's fraction, + the question value), 0.05x base tokens | read 5409; move into the engine |
| 5 | digit confidence | AUROC 0.90 outside one trace (24 wrong values) | on Qwen3.6, more traces |
| 11 | text layers (inline / whole / check-hand) | whole 118→138 on DeepMind; check-hand 4/14 on reasoning | move into the engine |
| — | **logit override** (force the value's tokens in the sampler) | design | engine step 4, with the confirmed window |
| — | **reflection-word suppression after a verified value** (NoWait / CGRS style, only where software verified) | design | 4B llama-server `logit_bias`, on reasoning rows |
| — | **residual-stream read/write of a number** (the model holds software's value itself) | running (`probe/num_rw.py`, 4B, 3-digit products) | if it reads and writes, port to Qwen3.6 in the engine as MIP's software-linked layer |
| — | H-trust test 1 (does a low own-probability of a given value predict re-checking) | queued | |

### C. Then
1. LiveBench: greedy vs T 0.6 pilot, and the MIP-off baseline on Qwen3.6 + strata-xeno.
2. MIP as one engine module for every model, connected to the q36 path (logit override, confirmed window, residual write).
3. Catch points 1b/6-10/12 on Qwen3.6 (raw draft trend, router, probes, final check, attention).
4. SWE-bench Pro or Live later.

## Detection coverage study (developer, 2026-11-04 evening): the guideline needs detection first

The developer's sharpest statement of the goal: software's results should not only change the output, they should shape the model's behaviour afterwards, like a guideline present all the time, so the model does not walk in the dark. How far it gets then depends on how smart the model is at finding the way with the guideline in hand. That needs detection that covers as much as possible, so detection is measured first, every signal on the same traces and the same set of computations (`scratchpad/probe/detect_cov.py`, run on the A100 on the stored control traces `pc5-ctrl`: 10 rows, up to 30,000 tokens each, no MIP in them).

**Ground truth (text only).** A computation is a `=` whose left side ends in an arithmetic expression `mathexpr.expression_before` reads and that has an operator. Its onset is the first token of that expression. Right or wrong: the value written after the `=` against the exact value (a rounded value is right). On the 10 rows: 579 computations, 19 wrong (the first reader called 132 wrong: fractions read as their numerator, rounded values, chained equalities; read before use).

**What is measured, per signal and per window k (2, 8, 32 tokens before the onset).**
- AUROC for "an onset is within the next k tokens";
- coverage = onsets with an alarm in the k tokens before them, out of all onsets, with the threshold set so that 1 % of all positions alarm; `random` is 1 - 0.99^k;
- the median lead in tokens and the precision of the alarms;
- separately, for a wrong value: AUROC of the digit's confidence and of a probe at the position before its first digit, with the counts.

**Signals.** `next` (mass of digit and operator tokens in the next-token distribution: the stand-in for the MTP draft proposals; the real MTP head is read in the engine), `probe_L` (a linear probe on the residual stream after layer L, fitted on the first rows and tested on the last), `router` (which experts were picked, Qwen3.6 only). Hidden state at the end of the prompt (point 9), the final-answer substitution (point 10), the shape of the thinking (point 4) and attention (point 12) are next. Text layers 2-5 are the baseline: they see a computation when it is written (lead 0).

**Reading the table.** A signal is useful for the guideline if it covers many onsets above `random` at a low alarm rate with a lead of at least a few tokens. Covering few onsets with a long lead is a different tool from covering most with lead 0. Held-out rows are 3 of 10: the numbers are a first read, not a verdict; the models are the 4B and Qwen3.6-35B-A3B bf16 reading the same stored tokens (the traces were written by the 4B).
