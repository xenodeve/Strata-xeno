# I2 matched first-attention collection

Issue #276. Collection-only checkpoint on the existing dedicated IGC branch.
Implementation: `5beea972dc4c01451a52cd37df919b43802a3b2b`; no runtime,
model, retained mapper, gate threshold or served configuration change.
This report is separate from the frozen experiment protocol, whose content
identity remains bound to the captured data.

## Scope and fixed reference

The native prefix probe uses deployed Gyro-M weights and the genuine verifier
stage `[0,1)`, with full 48-layer geometry. Its tap is first-attention output
before layer-zero MoE. This is a partial-prefix feature collector, not a full
Hub model run, answer-quality evaluation or latency benchmark.

The old reference executable is `0b625245a92ea4c6f962513b632ac132ff790798504d319c91c37864f60a6da6`.
The current executable is `7435ba0f4c26f4a01aa5aba76c7d5b81d4bb816ed8c89110f055e37c3fd0229d`.
They retain separate identities; passing selected replay rows does not make
these binaries identical. No tolerance change or cache relabeling is allowed.

Replay selected the first and last 64 OLD expanded-v2 prefixes, indices 0–63
and 10560–10623, using the existing frozen metadata and bounded file-row reads.
The 34.8 GB reference was not rehashed in full; the receipt identifies the
metadata hashes, pre/post file stamps and actual selected-row content hashes.

## Results

- Prefix fidelity: 128/128 rows matched F32 BYTES and masks exactly. Both 64-row
  children also passed reset/repeat, device-versus-host tap and retained-input
  boundary controls. No oracle feature input or numerical tolerance was used.
- New collection: 1,536/1,536 rows in 24 chunks, 768 train and 768 development.
  Each request passed its two-reset bitwise repeat. All 24 children exited 0.
  The outer owned-job launcher reached `DONE exit=0` and drained.
- Readback: `context_cache.load_cache` verified the complete new state/mask
  digests and manifest/protocol/source binding. A separate bounded row pass
  verified every ID, saved prefix hash, canonical mask, finite values and zero
  invalid-window padding. Per-row F32 receipts were saved independently.
- Collection elapsed 162.109 s, including preparation, child runs, writing and
  final hashing. Replay elapsed 15.031 s across both chunks including validation.
  These are instrumented collection costs, not a profiler-off performance claim.

| Identity | SHA256 |
| --- | --- |
| Matched semantic manifest | `88d4b39a3c2542c644f5eac4647e90f90780eb7bb1dcb3a9d6f5c324edbe25ff` |
| Saved states | `6757cde83254fff23010d26d720bc579880e2854750a4ee163b5224b8e12d82a` |
| Saved mask | `d96bed0c391bb9bdfd544c23a9615823229920ecc714903087d32add325f74a0` |
| Frozen matched protocol | `2de4d752b3df50a3623624f9acbc95e874b80d0acd02fdc3ee458a679cf75e0c` |
| Row-qualification receipt | `51d845506789d1a6d7b9e544c7cc6e51f2bba95c3e55df98c65eed4b68b3dea7` |

The model remains Gyro-M TQ2_0, SHA256
`7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f`;
tokenizer binding `889964f986b93c35102f25f5e396efeed65fcb180429c35d97536fb7f6b72106`.
Source validation retained captured content hashes plus unchanged file statistics,
with content-checked frozen copies where required. It was not a fresh 92 GB model
hash. Actual GPU visibility remained `1,0`, with device-order override unset.
The per-row feature geometry remains unchanged.

## Resource observations

Before collection: physical available 19,076,947,968 B; system commitment
60,635,881,472 / 87,963,570,176 B, headroom 27,327,688,704 B. Selected volume free
11,423,174,656 B after preserving the 338,224,598 B fidelity outputs. Fresh one-tap
budget 9,329,114,037 B included the existing 4 GiB reserve; admission passed.
After preparation, the collector's own preflight recorded
physical available 19,267,637,248 B and commitment headroom 26,183,196,672 B.
These are separate snapshots, not a single synthesized measurement.

During the 1,222 memory samples: largest observed child lifetime private peak
2,508,738,560 B; collector 1,032,040,448 B. Largest paired CURRENT private commit
3,516,252,160 B is distinct from adding nonconcurrent lifetime peaks. Minimum
system commit headroom 21,367,848,960 B and physical availability 15,412,563,968 B.
Maximum sample gap 1.922 s, including inter-chunk work; the samples do not guarantee
an unsampled instantaneous maximum. No private-commit cap was introduced.

Each probe reported 712,522,496 B in its tracked device allocations,
66,498,560 B native dense weights, 675,430,400 B embedding mapping and 13,107,280 B
context storage. These categories are not summed as a measured total VRAM peak;
whole-device peak VRAM was not sampled. Other applications were left running.

New retained cache footprint 5,036,449,559 B plus 338,224,598 B fidelity output.
Selected volume free after qualification 6,370,004,992 B; no second tap collected.
Old caches, raw predictions, corpus generators and user files were preserved.
Only the collector's known new redundant per-chunk archives were consolidated
into verified F32 rows under its existing cleanup rule.

After qualification: physical available 18,047,754,240 B and system commit headroom
22,176,935,936 B. The independent full-Hub 24 GiB physical requirement and disclosed
51,061,155,112 B commitment admission still fail. No full-Hub launch was attempted.

## Evidence and reproduction

Local evidence basenames (private artifact root retained locally):
`igc-matched-prefix-fidelity-01/report.json`,
`igc-matched-context-cache-01/{identity,report,records,memory}.json`,
`igc-matched-context-cache-01-command.json`,
`igc-matched-context-row-qualification-01.json`,
`igc-matched-context-qualification-01.json`, and their owned-launcher logs.
The command receipt contains exact paths, arguments and source hashes.

Native replay command form:

```
igc_prefix_probe.exe <same-model> <same-pack> <saved-64-prefixes.bin> <new-output-directory> 8
```

Collection used the existing CLI with `--corpus-manifest` pointing to the pinned
matched corpus and the frozen protocol above, 64-row chunks, 90 s per child,
1,200 s collector bound and 1,320 s owned outer timeout. It stored only the native
first-attention tap. No collector source change was needed.

Regression verification: `python -B -X utf8 -m unittest
 tools.test_igc_context_cache tools.test_igc_input_manifest` passed 10/10 in
3.391 s, with `DONE exit=0` in `igc-matched-context-regression-01.log`.
No CUDA/HIP/SYCL code changed; no new backend build, model parity or sanitizer
claim is made for this documentation-only checkpoint.

## What this does not close

The matched corpus is inspected development data, with paired compute and
quote-copy conditions and train/development pair isolation. It does not open
or consume sealed acceptance64. No new training, operand inference evaluation,
model substitution, deterministic-padding composition, policy/default change
or full-Hub smoke took place. I2 and the full project remain open.

Next evidence is evaluation of the retained readout on this cache before a
separately started fit under the frozen protocol and a priced training budget.
Neither evaluation nor training was run at this collection-only checkpoint.
The full-Hub live run still requires its own admission;
this smaller collector cannot substitute for it.
