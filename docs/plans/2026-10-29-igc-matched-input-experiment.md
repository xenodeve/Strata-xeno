# I2 matched current-readout development protocol v1

Issue #276. Preparation only: no collection, training, model replacement, gating
change or fresh acceptance access is authorized by this preparation result.
The fixed-anchor live smoke remains independently eligible when resources admit
it; this protocol is not a new prerequisite for that wiring check.

## Question and frozen inputs

Investigate the observed single-value/digit-slot failure without assuming an
attention cause. Current readout has nine categorical heads for ready/operator,
signs, lengths and two decimal slots per operand, supporting signed integers
within +/-99. Keep that schema, canonical padding zero, strict exact scoring,
architecture, loss and decoder unchanged. Do not repair predicted padding.

The only source pool is the already inspected expanded-v2 corpus file with
SHA256 `0aa4fa0c34e45b47f213279a638d7d11d1a04f0256877ddcf5e8be6c803452d9`;
canonical rows SHA256 `fe0e9d86210b4c0c7ef544a7b4a5982dc9c8f53c3a3eebada7a2e34b2356fa7c`.
Its 6,254 signed pairs were excluded from the sealed evaluation pairs during the
original construction. Select a subset; do not import or call the acceptance
manifest to regenerate it. CLI refuses any other source-file digest. This is
historically exposed DEVELOPMENT, not new unseen/release acceptance evidence.

## Corpus design

`tools/igc/matched_corpus.py`, seed20261108, produces 1,536 rows. It selects four
training and two development signed pairs in each of 16 length/sign strata:
lhs/rhs length1/2 x lhs/rhs negative/nonnegative. Splits share no signed pair,
even across operations. Each pair is used for all three operations +,-,*.

Train:64 pairs,192 pair-operation groups,768 rows. Each group has seen-family
English/Thai x compute/quote-copy intent. Development:32 pairs,96 groups,768
rows; each group has BOTH seen/new-to-fit template families x BOTH languages x
BOTH intents with identical expression bytes. Template wording is historically
known; "new" means excluded from this fit, not never inspected by developers.
`pair_group` identifies the statistical cluster; `group` adds operation, and
row IDs add family/language/intent. Expressions include explicit parentheses
for signed operands and are retained exactly inside each prompt.

In strata containing a one-digit operand, half the selected pairs contain
absolute3/8, half do not. Selection is deterministic and coverage-led, without
using predictions. Both splits must contain each signed value -8,-3,3,8 in each
operand slot. Two-digit/two-digit strata use ordinary pairs. This is a focused,
balanced diagnostic distribution, not a claim about real workload prevalence.
No arithmetic answer enters features. Negative rows have ready_op0 and the
same operands; their operand fields are not silently made eligible for execution.

Reproduction, using a NEW destination only:

```
python -B -m tools.igc.matched_corpus --source-corpus <audited-old-corpus.json> --output <new-directory>
```

Frozen rows SHA256 `bd250c5bb91ea0035caa580e9f11ca4753c8ccc38f5df8e5dcab63913d6c8681`.
Serialized corpus SHA256 `2a07d8b45ba2e4b02ddd571cdbe3fbb88b6493cb2a3aadb84e89c947ed0d7e19`.
The current generated file occupies857,013 bytes. No existing dataset is changed.

## Later execution, not run in this preparation

The separately versioned `input_manifest` adapter now binds this manifest and
the old expanded-v2 corpus without regenerating acceptance data. Collector uses
`--corpus-manifest`; cache/fitter loading verifies count, geometry, protocol,
sources and the matched manifest fingerprint. Do not retag data as expanded-v2
or relax identity checks. This metadata qualification is not a completed capture
or full-array load of the new dataset; see the adapter report.

First capture the new prompts' actual native first-attention states,80 x4 x2560
F32 tail and mask, preserving actual Hub profile and identity/accepted-prefix
semantics. Existing numeric pairs do not make old PROMPT activations reusable.
Verify every completed archive, prefix, mask, source identities and final receipts.
Neither expected labels nor token IDs may replace inference activation features.
Evaluate the retained input mapper as a baseline before any future fit; preserve
its weights and report that this corpus has historical exposure.

A future separately started fit uses the existing96964-parameter contextual
InputReadout (learned queries only), categorical field schema and positive-only
operand CE including zero padding. Keep initialization/optimizer recipe fixed:
model seed20261031, deterministic F32/TF32 off,70 epochs,batch64, AdamW lr0.001,
weight_decay0.01, cosine to0.00001, four CPU threads. Train only train rows;
normalization/fit statistics use those rows alone. Final checkpoint only, no
development checkpoint/hyperparameter selection. Initial fit bound1800s is a
stop, not permission to change the schedule or silently restart after failure.
Record the actual command, executable/feature/state/code/protocol hashes and
loss/finite-gradient checks. No fit is run or approved by this preparation report.

## Predeclared metrics and gates

Before fitting freeze all inputs above. Report ready/operator confusion,
positive exact operands and routed tuples, negative false calls, signs, lengths,
active digit accuracy and padding separately. Produce true/predicted digit
confusion by lhs/rhs,length and slot, and explicit +/-3/8 summaries. Count
noncanonical encodings as failures; do not zero padding or reconstruct a valid
number from only the convenient digits. Analyze template and intent contrasts
within the SAME pair/operation/language. Report both family scores with their
paired transitions, not a raw cross-cohort percentage as a template effect.

For this diagnostic development gate require all384 positive tuples exact and
zero false calls among384 negatives, with no relaxation of field validation.
Failure keeps I2 open and the sealed acceptance64 unopened. Passing is necessary
development evidence, not deployment approval, a statistical release guarantee
or proof of attention causality. Quantify uncertainty at the32 signed-pair
clusters, not as768 independent development samples. Later repeated inspection
requires a new version/protocol; never relabel known cases as fresh acceptance.

Readiness111/112 and output-mapper30/32 are different old tasks. New input data
does not establish result use. Basic live smoke checks wiring/ownership; usable
integration and matched real-Hub quality/latency benchmarks remain separate gates.
No gating-policy adoption follows from the old first-attention-only counterfactual.

## Resource needs and limits

Preparation is stdlib metadata plus the existing NumPy label tests, no GPU/model.
Raw fixed-size one-tap cache for1,536 rows is5,033,164,800 bytes (4.6875GiB),
plus122,880 mask bytes. The fresh collector budget is9,329,114,037 bytes including
2,048 header bytes,857,013 manifest bytes and the existing4GiB transient reserve.
This is an offline collector budget, not a Hub process cap or a measured peak.
Keeping both embedding and attention doubles the raw-state figure. These are
calculated storage, not measured peak RAM/VRAM. Start with one tap; do not collect
until the output drive has room for data, transient archives and retained evidence.

Full-model capture/live work still needs actual Hub admission plus disclosed task
headroom checks. Do not change pagefile/profile/workload or stop unrelated apps.
Offline fit has not been priced for this corpus; use the prior bounded mapper
recipe only as a starting budget, not an ETA or a proven peak. No cloud job,
download, system change, served default or model-weight update is included.

## Separate composition and centroid investigations

The raw categorical contract/gate above remains primary. A separately reported
OFFLINE composition variant uses only predicted length: class0 means one digit,
so set that operand's unused tens to0; class1 means two digits, so change nothing.
Keep operator/sign/length/active digits unchanged and preserve the original array.
Reject invalid field shape/class/range; do not use oracle length, fix negative
zero, infer a missing length, or rescue an active wrong digit. This variant is
a changed representation contract, not evidence that the head learned padding.
It cannot replace historical raw scores or retained artifacts/gating. Report
raw/composed tuple transitions, padding-only/active-only/both operand errors,
and downstream agreement with readiness held fixed. Adoption needs its own
quality/canonical/ownership controls, not the old development uplift alone.

For a centroid hypothesis, existing F32 cache plus native `prefixes.bin` archives
could support bounded reads, but metadata has only ONE exact full-prompt matched
single3/control4 pair in the known seen-template development cohort. Replacing
only the selected signed operand with train controls1,2,4,5,6,7,9 and preserving
operation/other operand/style/full wording yields one match each for1,2,4,9;
none for the new-template cohort. No centroid result is claimed from this thin
matched support, and the34.8GB cache was not scanned for a post-hoc explanation.

Before such an exploratory run, freeze probe IDs and native token-position
alignment from saved canonical prefix IDs, not a character offset or averaging
the80-row window. Split centroids/probes by operand role, length, sign, operation,
language and template; record every class's training sample/unique-pair counts.
Use training examples ONLY for centroids and any fitted centering/scaling. Keep
single-digit classes separate from the same digit inside multidigit numbers.
Raw-tap comparison: per-HC2560-vector L2 normalization, then concatenate streams,
cosine distance to training-only class means normalized the same way. No fitted
development normalization or PCA. Learned-readout comparison: frozen recorded
weights, extract each field's64-dimensional pre-classifier `result[:,field]`,
L2-normalize and use training-only cosine centroids for that SAME field/stratum.
Report zero-norm vectors/refused or unavailable classes explicitly. Do not
substitute logits for this representation or pool unrelated field queries.
Bound reads through `NpyRows`, never map/touch the entire cache; no gradients,
capture or model-body execution. A nearest centroid4 would describe geometry
under that representation, not establish attention/centroid causality.

## Research holdouts versus later deployment training

This paired diagnostic intentionally includes3/8 in both splits and is NOT a
leave-one-value-out experiment or a finished all-digits deployment dataset.
Keep the original exclusion split/weights/results intact. A future LOVO0..9
research protocol needs ten separately initialized fits, excluding the selected
single absolute value from BOTH operand roles in each fold while distinguishing
digit occurrences inside multidigit values. Canonical zero has no negative-zero
stratum. Match template/intent/position controls; never use another fold's fit
as an untouched holdout or silently merge fold scores into release acceptance.

Later deployment training should explicitly cover all0..9 single values plus
declared multidigit combinations/slots. Its validation uses separate held-out
pairs/templates/tasks; old3/8 exclusions are no longer fresh evidence after that
training. These are bounded planning distinctions only: no LOVO/deployment fit,
N-digit architecture, retained-model replacement or policy change occurs here.
