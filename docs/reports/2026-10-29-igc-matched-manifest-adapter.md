# I2 matched-manifest collector/fitter adapter

Issue #276; continuation after `cf72dbf63851629a41813bdb9a1c683bad50fb5b`.
This prepares the existing offline collector/fitter. No activation collection,
training, model replacement, deterministic-padding runtime change or sealed
acceptance access occurs in this checkpoint. Generators and datasets are unchanged.

## Bound contract and change inventory

`tools/igc/input_manifest.py` distinguishes structural validation from full
semantic identity. Matched v1 requires1536 rows, exact row fields, bounded signed
integers/ops, prompt/expression/intent agreement, unique row/group identifiers,
complete paired conditions, pair-disjoint splits, balanced length/sign/op strata,
signed3/8 coverage and the canonical row digest. Unknown target/answer fields
are refused. A structurally valid alternative corpus is still NOT accepted:
full manifest fingerprint must match the declared experiment.

- Matched semantic manifest SHA256:
  `88d4b39a3c2542c644f5eac4647e90f90780eb7bb1dcb3a9d6f5c324edbe25ff`.
- Expanded-v2 semantic manifest SHA256:
  `4fcff51db4712d97c92845529e8617e2240badf3da8a48280e4798e3a2a63cb3`.

Semantic fingerprints ignore JSON whitespace/key order, preserving parsed
manifest equality for the recorded expanded-v2 dataset while refusing changed
metadata/rows. Expanded-v2 cache validation no longer regenerates its constructor
or reads acceptance data. The legacy balanced-v1 embedding path is left intact;
its constructor is not invoked in these tests. Default legacy COLLECTION still
uses its existing expanded-v2 constructor; the explicit matched path never does.

`context_cache.prepare` adds optional `--corpus-manifest`, validated before
source/model I/O. Native encoding projects only id/prompt/expression, then encodes
the existing canonical wrapper plus prompt TEXT. IDs are receipt metadata;
the native prefix payload contains only token IDs, not labels, operation/domain,
targets or groups. Operand numbers naturally remain in the user's expression.
The expression-window/tokenization fidelity checks remain unchanged.

Matched completed caches record version/fingerprint binding, adapter/generator
source hashes and storage budget. Source/manifest hashes are checked again at
capture end. `context_cache.load_cache` and `embedding_readout` derive shape/count
from the pinned corpus; matched caches require the new binding. Existing source,
protocol, array-content hashes, F32/mask shape/layout, regular-file, nonfinite and
bounded64-row checks remain. Training still selects only split=train labels,
while network features remain cached activations/masks/window positions; receipt
source hashes now include the adapter dependencies. No training math or recipe
changes. The offline padding-composition variant remains separate and unused here.

## Test-first verification and limits

Behavioral stubs fail schema/paired projection and unpinned-corpus assertions.
The old CLI fails the new malformed-manifest test at its unsupported argument;
the implemented CLI instead refuses the manifest BEFORE missing model/source
paths and creates no output. Header stub fails identity binding, and storage
stub fails the literal byte budget. The final combined suite passes28 tests:
six new manifest/CLI/header/storage checks plus22 existing matched/audit,
context-cache, tensor-input and diagnostic/readout accounting controls.

Actual old expanded-v2 and new matched JSON files both pass pinned metadata,
header and native-projection checks:10624/1536 rows with corresponding cache
geometry. Valid matched header is fingerprint-bound; altered rows/protocol/source/
fingerprint are refused. Existing2-row NPY tests cover bounded order and truncation
with real tiny arrays. No5GB new array or35GB old cache was scanned or loaded.
Thus metadata/adapter and small I/O qualification are complete; full matched
capture/array load/training and native-prefix replay remain unrun.

A read-only peer source review found no actionable correctness issue. It did not
run code, training or inference and did not open acceptance64. Exact named
simplify/security-review skills remain unavailable; manual review is the stated
scope. No native backend file changed; CUDA/HIP/SYCL builds and model parity are
not claimed for this Python/docs checkpoint. Blueprint is updated with the code.

## Storage and measured resource inventory

The fresh one-tap budget is a calculated9,329,114,037 bytes:

| Component | Bytes |
|---|---:|
|1536 x80 x4 x2560 F32 states|5,033,164,800|
|Boolean masks|122,880|
|Two NPY header reserves|2,048|
|Manifest including newline reserve|857,013|
|Existing rolling-archive/log reserve|4,294,967,296|

The collector retains its existing12GiB system-commit and8GiB available-physical
preflight and emergency2GiB commit/1GiB physical floors. These are safeguards for
the partial native-prefix DATA COLLECTOR, not actual Hub admission, a private
commit cap or a production baseline restriction. No threshold is lowered to fit.
An old completed collector run sampled paired process+collector commitment up to
3,895,414,784 bytes and working set2,729,193,472 bytes. Child private lifetime peak
was2,470,428,672; collector1,638,666,240. These are historical observations, not
guaranteed bounds for a future run and not full Gyro serving memory.

At22:12:26UTC, GetPerformanceInfo reports physical available24,798,400,512 bytes,
commit available43,996,471,296; C free11,875,082,240 and D free10,868,359,168 bytes.
Both drives exceed the single-tap budget, while neither fits two retained taps
plus a shared4GiB reserve (about14.363GB). Sequential taps reduce concurrent
working/transient space; they do not eliminate cumulative retained data. Do not
delete existing evidence/user files to fit. Canonical F32/prefix/record/receipt
retention remains; the existing verified redundant-chunk archive cleanup is not
expanded to any old dataset. A second tap requires newly verified storage or an
approved separate destination. No deletion or second capture occurs here.

Primary5060Ti had16050MiB free; secondary/display4070SUPER8921MiB free; no heavy
engine/build/probe process was active. These are inventory samples, not a run
admission guarantee. Full Hub fixed-anchor testing still fails its independent
24GiB physical / disclosed51.061GB task commitment prerequisites at this sample.

## Next bounded run proposal (not executed)

The current build-directory prefix probe is SHA256
`7435ba0f4c26f4a01aa5aba76c7d5b81d4bb816ed8c89110f055e37c3fd0229d`.
The old context cache records `0b625245a92ea4c6f962513b632ac132ff790798504d319c91c37864f60a6da6`;
its expected frozen content-addressed copy was not found. Preserve this
distinction: do not relabel the current exe as the old one or reuse a build name
as identity. The current probe has earlier independent native-weight validation,
but that alone is not old-cache bitwise comparison.

First propose sequential64-row prefix-component fidelity replays at both ends of
the old corpus (128 rows total), using saved native prefix IDs and the current
hash-pinned probe. Compare produced F32 states/masks with bounded `NpyRows`
reads of the corresponding old cache; never map the whole file. New output
directories, explicit actual primary GPU mapping,90s per child and a bounded
headless outer run; preserve all receipts/outputs. Recheck resources/exe/source
identities immediately before dispatch. A mismatch stops this feature-comparison
path for diagnosis; it is not silently treated as an interchangeable capture.

Only after fidelity and current budget gates pass, propose fresh matched1536-row
first-attention collection with the explicit corpus manifest and NEW output path:

```
python -B -m tools.igc.context_cache --corpus-manifest <matched-corpus.json> \
  --executable <qualified-prefix-probe> --model <same-Gyro-M-file> --pack <same-pack> \
  --reference <old-early-reference> --sources <verified-source-manifest> \
  --protocol docs/plans/2026-10-29-igc-matched-input-experiment.md --output <new-cache>
```

Keep the inherited actual launcher CUDA_VISIBLE_DEVICES=1,0 and default device
order,256-token native-prefix bound,64-row chunks,90s child and1200s collection
deadline with1320s headless outer bound. This is offline feature collection,
not live Hub inference, a throughput comparison or new mapper training. Verify
full final corpus/records/F32/mask receipts and cleanup before any later fit.
Reprice offline fit memory/storage and retain old models before starting that
separate step. No collection, training, pagefile/profile/default change, push or
private backup upload was performed here. The centroid and LOVO planning notes
remain in the matched protocol; no centroid or N-digit architecture is introduced.
