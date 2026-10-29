# I2 fixed-fit numerical and provenance guards

Issue #276. This checkpoint implements the approved guards and pricing route.
Actual backward pricing and the single fixed fit have not run at this commit.
The frozen matched protocol and existing retained weights/cache remain unchanged.

`training_guards.py` rejects nonfinite loss, gradients, parameters and optimizer
state. It does not clip or repair values. Finite updates preserve the existing
categorical cross-entropy sum and positive-only operand mask. Numerical errors
stop before model export; failure notes include epoch, batch, completed steps,
IDs, elapsed time and system/process observations.

`embedding_readout --price-only` runs exactly one shuffled TRAIN batch backward,
without constructing an optimizer or saving weights. `--pricing-receipt` checks
fresh initialization, first-batch order, recipe, code, protocol and complete cache
identity before a separate fit. The consumed receipt SHA is recorded and rechecked.
Matched context uses the frozen cache-report/mask and GPU environment checks.
Cache files are streamed again before success; no whole-cache memory mapping.

Final-only evaluation switches to eval/frozen/no-grad and retains raw categorical
predictions. Paired and signed-pair cluster metrics reuse the declared retained
baseline definitions; predicted-length padding is a separately reported variant.
The final model is saved only after finite and provenance checks pass. A failed
quality gate still remains an experimental result, never a deployment decision.

Test-first evidence: injected NaN loss, finite-loss/NaN-gradient, parameter overflow,
optimizer moment overflow, legacy loss/update equality, unchanged backward-only
weights, admission/deadline, initialization fingerprint, full cache/reset binding,
raw/composed separation and CLI flag forwarding. Relevant suite passed 40/40:
9.094 s on default CPU Python and 6.999 s on the original CUDA Python runtime.
Logs: `igc-training-guards-{red,green}-*.log`, `igc-training-reset-red-01.log`,
`igc-training-price-cli-red-01.log`, `igc-training-runtime-tests-01.log`.

Independent review identified and closed three provenance gaps before execution:
full mask/report/manifest cache binding; consumed pricing receipt digest; matched
GPU visibility/order enforcement. Manual simplify/security and scrutinize traced
CLI to input identity, train-only selection, guarded update, final evaluation and
export. Existing fitter always fit a new model, so the no-optimizer price route
was required. Named simplify/security-review skills were unavailable; these were
manual reviews. No backend source changed; no new HIP/SYCL/Linux build is claimed.

The approved experiment remains one fresh seed 20261031 fit, 70 epochs, 840 updates,
batch 64, context head 96,964 parameters, AdamW 0.001/0.01 cosine to 0.00001, final
checkpoint only, 1,800 s fit / 1,920 s outer bounds. Before any backward batch, recheck
8 GiB physical, 12 GiB commitment, 2 GiB GPU and 16 MiB output plus 4 GiB disk reserve.
These are disclosed offline admission checks, not a Hub private-commit cap.
No acceptance64, second tap, threshold/model default, pagefile, push or backup change.
