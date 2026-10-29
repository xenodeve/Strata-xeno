# First live learned-controller diagnostic

Issue #276, I2/I3 follow-up. Retain the controlled-input output fixture as an
explicitly separate comparator. Connect the qualified learned pipeline to the
actual native Gyro serve path without prompt lookup, host operands or arithmetic.
This first milestone supports one committed answer-start anchor, not repeated
reasoning calls or a complete session/snapshot ledger.

Change inventory:

- `program/igc_learned_diagnostic.hpp`: bind actual model/tokenizer content;
  prepare/release context and pipeline on the verifier stream; begin a request
  with valid prefix lineage, capture only accepted rows, stage once before the
  first answer window, and audit once after the request fence.
- `src/program/generate.cpp`: opt-in diagnostic bundle flag, unsupported-mode
  guards, early observation, request-entry reset, accepted prompt/decode capture,
  first-window staging and request-end audit. A missing feature window bypasses
  IGC without forcing cold model prefill. Preserve D2x expert offload and MTP.
- Root `CMakeLists.txt`: link the optional diagnostic components only in the
  existing CUDA diagnostic build. Feature off installs no hook or cache.
- `tools/test_igc_learned_cli.py`: behavior-first admission and incompatibility
  tests against an actual diagnostic executable, before allocating the model.
- `tests/igc/*`: retain component/lifecycle controls; add controller lifecycle
  checks where the connection introduces behavior not covered by them.
- `tools/igc/learned_integration.py`, `tools/test_igc_learned_integration.py`:
  bounded actual-Hub live harness and offline request/result evidence validation.
- `tools/igc/observation.py`, `hub_contract.py`, `result_rubric.py`, and the two
  older output collectors: share pure prefix/admission/rubric contracts without
  loading the array runtime into a live diagnostic that needs no array math.
  Preserve tensor-reader and old-import behavior; test both interfaces.
- `docs/BLUEPRINT.md`, the evidence report and Issue #276 comment: actual source,
  build/input identities, outcomes and incomplete gates; no default change.

Surveyed entry points include verifier initialization, the serve request's
restore/resume selection, short prompt windows, batched/wave prompt callbacks,
first decode window, accepted-input commit and final commit fence. The first
implementation must label missing batched features as unavailable; it must not
claim serial-forward equivalence or disable the actual Hub prefill path. A later
producer qualification must preserve stream ownership and wave ordering. Never
reuse features merely because a restored session has matching geometry or KV.

Run CLI admission tests red on the pre-change diagnostic executable, implement
the hook, build both D2x architectures 89;120, then repeat admission checks. Use
the unchanged selected Gyro 262K / 5-worker Hub workload for bounded live checks,
with telemetry and headless cleanup but no private-commit cap or workload
restriction absent from Hub. Price/observe additional staging rather than invent
a memory threshold. Compare native audit tuples/results with an offline oracle,
and inspect generated result use separately. Preserve the sealed acceptance64.

Same-session feature-off raw-token parity, general input extraction quality,
repeated readiness/rollback, snapshots, batched/wave eligibility and real Hub
performance are separate open gates. A live diagnostic success does not close
them or justify default activation.
