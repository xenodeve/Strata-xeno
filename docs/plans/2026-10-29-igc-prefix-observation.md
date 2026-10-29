# I2 diagnostic: native first-stage observations

The frozen embedding-only input experiment fits its training set but fails new
wording and negative routing. The next discriminator is contextual information
from the deployed model, rather than another overfit run. This diagnostic does
not change the Hub baseline or claim to reproduce the paper's complete first
transformer layer.

## Change inventory and contract

- `tests/igc/prefix_probe.cpp`: optional offline executable using the existing
  `Verifier` layer range `[0,1)` with the original 48-layer model geometry. Load
  the actual native dense matrices, embedding and every layer-zero expert, with
  an all-resident tier. Use real GDN recurrence/commit, reset between requests,
  and capture the already implemented first-attention tap before MoE combine.
  No zero experts, fabricated activations, sampling or model answer calculation.
- `CMakeLists.txt`: build under the existing optional IGC probe switch, CUDA
  `89;120`. No serving flag, core operator or default-path change.
- `tools/test_igc_prefix_probe.py`: malformed input before allocation, actual
  full-model captured-row fidelity, window-size equivalence, repeat and reset
  isolation. A narrow binary input bounds requests and tokens before GPU work.
- `docs/BLUEPRINT.md` and report: document diagnostic ownership, memory and the
  exact seam. Report backend/sanitizer gaps, not inferred coverage.

Input is `IGCPFX01`, a little-endian uint32 request count (1..64), followed by
uint32 token count (2..256) and int32 token IDs for each prefix. Refuse trailing,
short, oversized or out-of-vocabulary data and existing output names. Bound
input to 66 KiB. Output uses the existing `IGCATN01` archive contract, retaining
only the last 80 consumed pre-anchor rows. The final anchor is not consumed.

Use the existing session range allocation, a fixed verifier maximum of eight
rows and explicit streams. Fence commits before reset or teardown. Every
request is replayed from zero state twice and must match bitwise. The optional
window argument is 1..8; it is an experiment condition, not a speed claim.

## Verification before training

First establish a real missing-target RED. Build the optional executable. Check
refusal cases, then compare all recorded rows with immutable full-model captures
from the same native executable/source/configuration. Compare windows one and
eight and A/B/A request reset. Any mismatch blocks use as a training source and
is investigated under `/debug-mantra`. The model geometry remains unchanged;
PLE is first applied at layer one and is absent from this stage by source.

Start with at most three short prefixes and a bounded headless run. Price actual
resident/transient allocations before any large corpus. A successful probe
qualifies this seam only; it does not close I2 generalization, I3 result use or
integrated Hub correctness/performance gates. Keep the untouched acceptance64
separate from development data.
