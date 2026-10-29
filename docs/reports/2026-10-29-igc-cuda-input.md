# I2: CUDA input training and final-epoch instability

The declared CUDA experiment completed but failed. Its final epoch lost the
previous low training loss, so the saved final weights cannot qualify for fresh
evaluation or integration. No earlier checkpoint is substituted as a passed
result. Evaluation64 remains untouched.

## Runtime and input qualification

The already installed official Torch 2.11.0+cu130 runtime supports SM120.
An eight-sample same-state CPU/CUDA forward/backward comparison passed:
predictions identical, maximum output difference 2.7656555e-5, losses
0.0005793123273 versus 0.0005793272285. All parameter gradients pass per-component
`atol=1e-6, rtol=1e-3` plus maximum absolute difference below 1e-4. A deliberately
wrong gradient fails. Peak allocated/reserved memory was 178,252,288/205,520,896
bytes. This checks task-owned input heads, not quantized-Gyro backward.

Earlier diagnostic checks used relative error near zero or a dimension-independent
absolute vector-norm floor. They rejected the mathematically zero attention-key
bias gradient and tiny saturated-head gradients despite tiny absolute differences.
The corrected comparison is per component; its negative control is retained.
A separate test verifies attention key-bias translation invariance/zero gradient.
Evidence: `igc-cuda-readout-batch{,-green,-mixed}.log` and
`igc-cuda-readout-batch.json`. Failed checks remain preserved.

`TensorInputs` stores only frozen activations, bounded gather indices, masks and
positions on the selected device. Cache ordinals never enter the neural network.
The expanded corpus has 10,240 fit/384 development rows; original pilot,
fresh-evaluation and previous development pairs are excluded. Canonical row hash:
`fe0e9d86210b4c0c7ef544a7b4a5982dc9c8f53c3a3eebada7a2e34b2356fa7c`.
All 217 required native embedding rows match independent Q8_0 decoding across
four HC streams. Nineteen input/cache/corpus/diagnostic/readout tests pass.
No model body, installation, download or served configuration change.

## Frozen run outcome

The [protocol](../plans/2026-10-29-igc-cuda-input-experiment.md) fixes the
96,964-parameter contextual/query-free head, seed 20261031, 70 epochs, batch 64,
AdamW 0.001/0.01 without foreach/fused updates, F32/TF32 off/math SDP, and the
600/660-second fit/outer bounds. Only final weights were eligible for evaluation.

Run `igc-cuda-expanded-fit-01` completed in 238.512 seconds. Epoch mean loss:

| epoch | loss |
|---:|---:|
| 20 | 0.003962334 |
| 40 | 0.000241551 |
| 60 | 0.000044880 |
| 69 | 0.000021242 |
| 70 | 1.025377404 |

The final snapshot gives only 130/7,680 exact routed fit tuples, with 14/2,560
false calls. The three development axes give zero exact tuples; the two new-
template axes each have 32/32 false calls. This is a failed final snapshot,
not evidence about the quality of an unsaved earlier checkpoint.

Final state SHA256:
`82f0b26473a4ca9ca377e2bbb6d413f3b3de21c1d914a5a2138504ed30b73009`.
Protocol SHA256:
`720549305e4606224bff306e8814bd9afa3161610b9286fca52463c08ccb022e`.
GPU peak allocated/reserved: 745,498,112/941,621,248 bytes. Recorded lifetime
process private peak: 3,829,866,496 bytes. No private-commit cap or Hub baseline
change. These are head-training observations, not inference speed claims.
Evidence: `igc-cuda-expanded-fit.log` and
`igc-cuda-expanded-fit-01/{report,predictions}.json`.

## Debug-mantra reproduction and immediate failure mechanism

The target is the exact final-epoch loss jump under the frozen seed/data/runtime,
not a generic GPU or memory claim.
The diagnostic reruns the same recipe, preserves a model/optimizer/RNG checkpoint
after epoch 69, and records epoch-70 batch losses, gradients and actual updates.
The investigation compares loss history and final-state SHA256 to the failed
run, then distinguishes optimizer, numerical and data/lifetime hypotheses.

The first diagnostic wrapper failed before its first optimizer update because
Torch added an optimizer profiling wrapper around the intercepted method. Its
assertion/log is preserved; no reproduction was claimed. A bounded caller-frame
walk fixes only that harness. The second reproducer's complete loss history and
final-state SHA256 match the failed run exactly. It traced all 160 final-epoch
batches. Evidence is
`igc-cuda-instability-debug-02.log` and `igc-cuda-instability-debug-02/`.

The immediate mechanism is a constant-rate Adam update overshooting a fitted
solution. At epoch 70 batch 155, gradient L2 is 1.0274 and the actual update L2 is
0.07611. Loss rises from 0.00085203 to 0.36760777 on the same batch; the next
updates cascade. Replaying the epoch-69 model/optimizer/RNG checkpoint matches
all 156 pre-failure batch losses bit for bit. CPU loss is 0.0008519723, CUDA loss
0.0008520325, and all gradients agree under the declared mixed FP32 rule.

The actual direction is initially downhill: gradient dot update = -0.03951294.
Predetermined fractions of that same update give:

| fraction | same-batch loss |
|---:|---:|
| 0 | 0.000852033 |
| 0.01 | 0.000541836 |
| 0.1 | 0.000058891 |
| 0.5 | 0.004338040 |
| 1 | 0.367607772 |

This falsifies an incorrect local gradient as the explanation and supports
reducing step size near convergence. Evidence:
`igc-cuda-instability-debug-02/{summary,falsification}.json`, `epoch70.jsonl` and
`igc-adam-falsification.log`. It does not establish generalization quality or
every possible future instability.

The [new frozen validation](../plans/2026-10-29-igc-cosine-input-experiment.md)
starts fresh with identical data/seed/architecture/70 epochs and changes only
the global learning-rate schedule to cosine decay from 0.001 to 0.00001.
There is no conditional bad-batch workaround or earlier-checkpoint selection.
Twenty input/cache/corpus/diagnostic/readout tests pass, including schedule
endpoints, monotonic bounds and absence of epoch restarts.

## Fresh cosine validation outcome

The fresh run completed in 230.275 seconds. It remained stable to final epoch
loss 0.000204544, with final learning rate 0.00001. All 10,240 fitting readiness/
operator classes and 7,680 positive operand tuples match, with no fitting false
calls. Thus the observed convergence failure is removed in this validation.
Generalization still fails:

| development axis | exact routed tuples | false calls |
|---|---:|---:|
| seen values/new templates | 50/96 | 28/32 |
| new values/seen templates | 77/96 | 0/32 |
| new values/new templates | 33/96 | 32/32 |

I2 remains open and fresh64 remains untouched. Stability and input quality are
separate gates; a perfectly fitted training set is not release readiness.
Final weights SHA256:
`ec7273bed946ea3fa03ca375f453a602de82b34a96cdca88da880945505c427a`.
GPU peak allocated/reserved remains 745,498,112/941,621,248 bytes. The report
records training-source hashes and verifies that source did not change during
the run. Evidence: `igc-cuda-cosine-fit.log` and
`igc-cuda-cosine-fit-01/{report,predictions}.json`.

Next input work must address contextual extraction/false triggers rather than
repeating overfit checks. Frozen-model contextual features and stronger coverage
remain research options; none is silently claimed as implemented. No serving
or default change follows from this result.

The [native-head bridge plan](../plans/2026-10-29-igc-native-head-training-bridge.md)
is independent I3 preparation, not implemented or validated output mapping.
