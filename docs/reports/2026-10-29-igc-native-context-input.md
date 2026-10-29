# I2 native contextual cache: collection and memory correction

Tracker [#276](https://github.com/xenodeve/Strata-xeno/issues/276).
[Native-feature protocol](../plans/2026-10-29-igc-native-context-input-experiment.md).

All10,624 expanded-v2 rows are collected:10,240 training and384 known
development cases. No fresh acceptance64 is consumed and no new input-quality
result is claimed yet. The native extractor's full-model fidelity gate is in
the [qualification report](2026-10-29-igc-prefix-observation.md).

## Failed capture and causal check

The original collector stops after9,408 committed rows at its physical emergency
floor, not a private-commit cap. `igc-context-collection.log` retains the failure.
Last completed-chunk telemetry shows system commit26,990,555,136 of87,963,570,176
bytes, headroom60,973,015,040, physical available2,209,153,024. Minimum recorded
physical available is1,178,488,832; the initial code did not persist the final
below-floor sample, so the exact boundary value is unavailable. After owned
process cleanup physical available is34,227,286,016. No model failure is inferred.

The fail path holds a whole-file F32 mmap while filling the35GB cache. A bounded
one-GiB read/control reproduces its mechanism (`igc-context-memory-mechanism.json`):

| Condition | Process working set bytes | Private commit bytes |
|---|---:|---:|
| Mapping created, before touching data |28,459,008|727,252,992|
|1,048,576,000 data bytes touched|1,077,198,848|727,252,992|
|Mapping closed|28,626,944|659,120,128|
|Same data read through a reused64-row buffer|238,346,240|869,253,120|
|Buffer released|28,626,944|659,120,128|

This identifies retained mapped-page working set as the collector's avoidable
cost. `NpyRows` now performs bounded ordinary I/O without retaining a large map,
for both collection and training. F32 values, shape, masks, corpus and neural
recipe remain unchanged. A new test covers exact file rows, selection order and
truncation. Final telemetry includes the collector itself and persists a failing
sample before any future emergency exit.

## Recovery and completed result

Original source, failed log, telemetry and mask are preserved. Completed chunks'
exact token inputs/output markers and every persisted F32/mask/padding value are
validated. Both ends are independently recaptured (128 rows, byte-identical)
before adding the remaining1,216 rows. The failed last-chunk artifacts remain.
Recovered records are content-hashed during recovery; original per-archive hashes
were not journaled before failure and are not claimed to have survived.

`igc-context-collection-resume.log` DONE0. Recovery plus remainder validation and
final hashing takes159.109s, not a production-latency measurement. Collector peak
working set1,012,563,968 and peak private commit1,638,666,240 bytes; minimum
physical available30,963,875,840 and system headroom59,050,577,920 bytes. No
Windows settings, thresholds, source base, model or Hub profile are changed.

- Corpus SHA256 `fe0e9d86210b4c0c7ef544a7b4a5982dc9c8f53c3a3eebada7a2e34b2356fa7c`.
- F32 NPY SHA256 `3d1e27038d0c7dd7cc440dd876266237869595fbd20a701750670601db735806`.
- Mask SHA256 `d8c8f95e1891c204d4d67006eee86b8ee0cbcdb93123aceb5784c27dde51f2cc`.
- Native-feature protocol SHA256 `0a078202ccba84943916ec2965a32b68e76ab86c3060621c6e2335f36876a4b8`.

The new protocol initially collided with the older contextual-block filename.
The old protocol is restored byte-identically, SHA256
`3b9f10bfacf9964961b670330ee638e3560dcbe0bb2a83a9d60e6c919f3a3c3a`,
and the native experiment has its own filename. Historical weights/reports/data
are untouched. The frozen comparison uses the new path and explicit hash.

Next is the predeclared70-epoch native-feature fit, same model/data/seed/cosine
recipe as its embedding control. Final weights only, no development checkpoint
selection. Passing collection is separate from I2 generalization, I3 faithful
result use and actual integrated Hub performance.

## Completed70-epoch fit: I2 remains failed

After the separately documented time-budget correction, `igc-native-context-fit-02`
finishes DONE0 in1,407.866s, final training loss0.00001152515. Final state SHA256
`519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2`.
All7,680 fit tuples are exact, all10,240 operators/readiness labels match, and
none of2,560 fit negatives calls the calculator. Fresh acceptance64 is untouched.

| Known development axis | Embedding control exact/96 | Native first-attention exact/96 | Native false calls/32 |
|---|---:|---:|---:|
|Seen values / new templates|50|96|30|
|New values / seen templates|77|50|0|
|New values / new templates|33|67|32|

These are the same frozen expanded-v2 cases/seed/architecture/optimizer/schedule,
with only activation features changed. The mixed outcome is not a blanket
improvement: one new-value axis regresses and unseen negative wording still
triggers incorrectly. Peak CUDA allocated728,959,488/reserved920,649,728 bytes is
for this offline readout fit, not the full language model or serving overhead.

All positive operator, sign and length fields match. Nonpadding tens digits also
match on every axis. New-value failures involve ones/padding-digit predictions:
41 noncanonical operand predictions on new-values/seen-templates and25 on both-new.
There are no operand swaps or digit reversals in these cases. Examples:

- `Work out 93 * -3 and give its value.` correctly selects multiply and length1
  for the right operand, but predicts ones4/tens3. This is an **invalid encoding**,
  not a valid -34 operand or an executed arithmetic answer.
- `Treat 0+-84 as a label. Only acknowledge it.` incorrectly selects addition
  instead of no call; operand extraction does not establish readiness correctness.

Next falsifiable hypotheses: a left-aligned neural digit representation with an
explicit padding class may match the tokenizer better while preserving the exact
executor's little-endian ABI; readiness may need a separate learned intent readout
from the normal full-model anchor state rather than early token features alone.
Neither is established by this outcome. Preserve this failed run and freeze a
distinct experiment before testing either; do not repair old predictions after the
fact or call the old head eligible. Full-sequence output training proceeds as an
independent necessary component, using oracle/checked-native results initially.
