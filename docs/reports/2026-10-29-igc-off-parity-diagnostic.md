# Disabled-path raw-token variation: reproduced before the new output hook

Do not mark raw parity passed. The original smoke showed3/4 prehook/candidate-off
agreement, while both candidate-on/off unmatched cases agree. Follow-up controls
preserve actual selected Gyro-M262K/five-worker/Vision/D2x/MTP-suffix settings,
greedy96-token bound and the same initial expert profile and four prompt order.
No adapter, profiler, timeline, state-hash or workload-changing flags are enabled.
No private-commit cap or served changes. Each arm is a fresh owned process.

| session/arm | prehook A1 | candidate B1 | candidate B2 | prehook A2 |
|---|---:|---:|---:|---:|
| A-B-B-A Thai case output length |37|37|39|37|
| same prehook executable in **all four arms** |37|39|37|39|

The same-prehook control's report explicitly has `same_executable_control:true`;
A/B are only arm labels there. All use SHA256
`ae6c485fb6b765ae4005183bac1712508b801dbfb1da35ee46bae50966597d7d`.
The mixed session's candidate is
`0f68c5f648b8324c1a168f0229eef64349754815a71ac32f70e0c32ab63fbbdd`.
The other three prompts' raw IDs match in every shown comparison. Both Thai
sequences complete normally; no limit, crash or memory rejection occurred.

This is a **main-based pre-injection diagnostic executable**, not a fresh proof
about an untouched release/main binary. It predates the new native output hook
and contains earlier opt-in observation changes. Thus the new output hook cannot
be the sole mechanism of this variability. It does not establish a root cause,
waive the repository's parity gate or prove that candidate code has no effect.

Per-run DONE/MTP/tier counters are retained. Counts already differ before visible
text divergence, but differing aggregates are not causal proof. Do not propose a
fix from that association. Existing tracker search found related historical
[determinism #18](https://github.com/xenodeve/Strata-xeno/issues/18) (closed; different
model/build and cross-boot context) and an explicit older D2x nondeterminism note
in [#260](https://github.com/xenodeve/Strata-xeno/issues/260#issuecomment-6246341550).
Those reports do not prove this short Gyro serve case has the same cause. In
particular, the periodic16K checkpoint hypothesis is not demonstrated by these
sub100-token prompts. No source base or Hub setting was changed as a remedy.

The diagnostic runner now uses `run_completed`, a separate repeat-comparison
matrix and an explicit narrow single-integer-final **format** rubric. It does not
silently reinterpret the earlier smoke's `success:true` as acceptance or its
correct equation-plus-answer as bad mathematics. Original reports/source remain.

Plans are immutable companions:
[A-B-B-A](../plans/2026-10-29-igc-off-parity-diagnostic.md) and
[same-prehook control](../plans/2026-10-29-igc-prehook-repeat-control.md).
Private evidence: `igc-output-off-parity-01/` and `igc-prehook-repeat-01/`, including
inputs/report/memory/arm records and logs; wrapper logs with the same names.
Both DONE0/owned trees drained.101.422s/103.062s are whole diagnostic session times,
not speed measurements. Fresh acceptance64 and all learned checkpoints untouched.

Continue independent I2/executor/output implementation. Before release/default
activation, localize the baseline variation or obtain valid matched parity
evidence with its limitations explicit. A different diagnostic placement setting
must never be labeled the actual Hub baseline or silently become serving default.
