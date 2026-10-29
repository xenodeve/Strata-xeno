# I2: validate convergence decay after the reproduced Adam overshoot

The constant-rate run failed and remains failed. A same-seed rerun reproduced
its entire loss history and final weight SHA256 exactly. Epoch 70 batch 155 is
the boundary: loss 0.00085203 before the update becomes 0.36760777 afterward;
later batches cascade. Replaying from the epoch-69 model/optimizer/RNG state
matches all 156 pre-failure batch losses exactly. CPU and CUDA gradients agree
under the declared mixed FP32 tolerance at that state.

The measured update is initially a descent direction (`gradient dot update`
is -0.03951294), but its full length overshoots. Fixed fractions give same-batch
losses: zero 0.00085203, 0.01 step 0.00054184, 0.1 step 0.00005889, 0.5 step
0.00433804, full step 0.36760777. This supports shrinking updates near convergence,
not corrupt labels or a guessed hardware/memory fix. No checkpoint is promoted
from the failed run.

## Single declared change

Start from fresh seed 20261031 with the same expanded corpus, native cache,
architecture, runtime, optimizer, batch order, 70 epochs and resource bounds as
the CUDA constant-rate protocol. Do not warm-start or choose a best epoch.
Change only the learning-rate policy to global-step cosine decay:

`lr(step) = 0.001 * (0.01 + 0.99 * (1 + cos(pi * step / (total_steps - 1))) / 2)`

There are 11,200 updates. First lr is 0.001, final lr is 0.00001, without epoch
restarts. This is a general convergence schedule, not a conditional adjustment
at the observed bad batch. No gradient clipping, output repair, changed data or
relaxed quality gate. Preserve constant rate as an explicit reproduction control.

Verify endpoints, global bounds and monotonicity before fitting. Retain all losses,
final weights, code/runtime/source hashes, LR policy and peaks. The existing
native-cache preparation protocol remains separately verified; do not rewrite
its historical metadata. Model source files must remain unchanged during fitting.

## Unchanged acceptance

Use final epoch only. Require all 7,680 training positive tuples, all 96 positive
tuples on each development axis, and no false calls. If this fails, keep fresh64
untouched and report the result, including whether the final-epoch instability
was removed independently of generalization. Stability is not model quality.
Only an eligible frozen head proceeds to one-time fresh64 evaluation: all 48
positive tuples and no false calls on16 negatives. No claim of output mapping,
62-digit extraction, repeated triggers or usable integration follows from this
experiment alone. Usable integration is followed by actual Hub benchmarking.
