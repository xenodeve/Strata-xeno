# I3: gated result-to-activation mapper component

The reusable native-head forward/STE seam is qualified. Implement the independent
neural result mapper next while contextual input data is collected. This is a
late head-input variant; it is not the paper's early-layer injection and cannot
be reported as backward through Gyro's frozen body.

The mapper receives a validated typed integer result, the current frozen mixed
activation, and a learned per-position gate. Encode the exact ABI's decimal digits
as categorical result features with explicit positional order and sign. Preserve
the executor's little-endian ABI while presenting a left-aligned view to the
network. Do not convert results into text, tokenizer IDs, direct output logits or
deterministically selected answer tokens. Attention over result digits conditions
a learned residual; a learned gate scales that residual. No base weights train.

Bind result version/status/invocation/consumed position before exposing numeric
features. Failure statuses carry no numeric value and hard-disable the residual;
they are not encoded as successful zero. A hard feature-off returns the original
activation without running modules. A near-zero learned gate is not that bypass.
Malformed records are refused, never normalized or repaired. Bound batches1..64,
digits62 and tensors2560/F32. Model records never contain pointers or code.

Change inventory: `tools/igc/output_mapping.py` for the offline typed feature
boundary and attention/gated residual; `tools/test_igc_output_mapping.py` for
canonical/ownership/status tests, hard-off/non-success identity, numerical and
gradient/mask behavior. BLUEPRINT documents the component. No production injection
or claim of a trained result mapper until an actual native-forward LM-loss fit
and held-out faithful-result controls are completed.

Training must use identity-bound real Gyro activations and correct result features
initially, as in the paper's independent output training. Validate full result use,
ordinary/non-arithmetic preservation and corrupted/absent-result controls. A
single arbitrary token-class loss or first-digit-only diagnostic does not close
I3. Full sequence training needs native teacher-forced output-position captures;
the existing first-anchor archive alone cannot establish that. Keep this dependency
explicit rather than substituting a forced token emitter or claiming completion.

Reference: [IGC v1, methods](https://arxiv.org/html/2501.00684v1), independently
implemented from the published method; no official released code is assumed.
