# P1 offline compiler/artifact/oracle checkpoint

P1's offline gates passed. P2 host-loader work follows. Nothing in this checkpoint
establishes Native CUDA execution, model utilization, ownership/parity or practical
performance; those remain P2–P8 and capability-roadmap gates.

## Deliverables

- Complete Strata tokenizer binding, strict ELF/source checks and approved offline bootstrap.
- Known feature lowering with exhaustive callback comparison over both full token domains.
- Data-only STQPUT01 binary, ordered exact-FP32 coefficients, int32 sparse indices,
  channel/head/state geometry, retained output/owner dimensions and provenance.
- Independent AST/Fraction mathematical oracle, checked original IR traces/state,
  first-party synthetic instruction/fault vectors and a separate held-out specification.
- Bounded bundle assembler, compiler MIT notice and explicit firmware-distribution hold.
- [Format contract](../QWENPUTER_FORMAT.md), [input evidence](2026-10-28-qwenputer-p1-inputs.md)
  and [capability roadmap](../plans/2026-10-28-qwenputer-capability-roadmap.md).

## Evidence

| Gate | Actual result |
|---|---|
| Source pin | 13 unchanged upstream blobs at ca13310e2f9f788d1dd3feeea8213c1b0c23913c; canonical manifest a5519351051213218a2236f1a4737af0e26e310fe3fb74d3f9269ef5bfd8c7c5 |
| Firmware | Approved b643ad61291a947fa6c7498de9170b1d4da062dc1f34810feba9c4eebdec862d only; no rebuild |
| Tokenizer | Gyro binding 889964f986b93c35102f25f5e396efeed65fcb180429c35d97536fb7f6b72106 |
| Boot | 62 trace tokens, 50 instructions, 99 initialized tables |
| Features | 550 × (230,324 Trace + 248,320 Text) = 263,254,200 exact comparisons; proof 190676489d4adacf6ee6a78d80f9fcbd1a2bc98eced7210c5881f567762478f6 |
| Reproducibility | Seed0 exports 03/04 match all five output files byte for byte |
| Artifact | 18,298,944 bytes; edcd063736a3b3575cf81b5bac524943e0694e23c12af0cd261b44b150e704b0 |
| Mathematical development corpus | 29 cases; 21 solved math cases, eight explicit errors including three documented conservative-capacity refusals |
| Checked original IR | All 29 cases, 74,406 tokens × 2,918 residual dimensions, owned predictions and saved final state/lag data |
| Synthetic datapaths | 50 original-IR instruction/fault fixtures, including all eight faults; no new external firmware executed |
| Independent register check | 42 saved non-fault outcomes, every register bit exactly 0/1 |
| Held-out specification | 64 cases, seed 20261103, SHA256 d1ddd54c5c86d93eab55e3bee98873be16fa777b175583e76b5ce77e34da32f3; no firmware/Native execution |
| Final component regression | 88 passed, three skipped, 64 subtests; 8.46 s, DONE0. Added held-out-spec unit test also passed |
| Broad tools unit suite | 748 passed, 14 skipped, two failures, 294 subtests; both failures reproduced on unchanged main |

The two pre-existing failures are
`test_setup_draft_vocab.py::SmallCardNote::test_sizes_follow_the_shipped_subsets`
(missing `cjk` table entry) and
`test_strata_mcp.py::Helpers::test_builtin_table_matches_setup`
(fallback model list omits Gyro M/S). No unrelated setup/MCP code was changed.
Bare tools pytest collection also imports five manual `*_test.py` CLI smoke
scripts with pytest arguments; those collection errors are retained. The proper
unit-module sweep used `-o python_files=test_*.py` and the existing root checkout's
`STRATA_GGUF_PY`; no setup/install/download was run.

## Diagnoses and limits

Random Python hash seeds changed set-derived sparse term order. Two seed0 controls
matched, seed1 changed terms. Export/oracle processes now require seed0 at process
start and verify actual hash randomization is off; upstream bytes are unchanged.

The pinned firmware conservatively rejects `10**61`, a 62-digit literal plus zero,
and that literal times one. The source/PC/register diagnosis identifies pre-trim
operand-capacity checks, while the literal itself and subtraction work. The corpus
keeps independent exact answers and `mathematical_success=false` for refusals.
It does not relabel them as solved tasks. Firmware remains unchanged.

Native-call tests currently specify trusted live/enrollment/quote/fence/history
context. Integrating that context/controller with actual model output is P5/P7.
In particular, upstream core predicts GO at any TCC: the runtime must gate real
takeover and handle unrelated/invalid captures without executing them. No model
eligibility or controller completion is claimed by offline tests.

All tensor/runtime sizes quoted here are static file/state accounting. Actual GPU
allocation, scratch/graph peaks, resource fit and latency remain to be measured.
Compiler source licensing is recorded; linked firmware runtime notices/build
provenance remain a distribution hold. Artifacts stay local.

## Workflow and publication

Manual simplicity, input-boundary and owner-trace reviews ran with the regression
checks. Named `/simplify`, `/verify` and `/security-review` were unavailable in the
checked skill roots; this report does not claim those named skills ran. Original
Standards/Spec findings from the input/bootstrap checkpoint were fixed and tested.
Fresh whole-branch review remains an end-of-implementation gate.

PRD R9 preserves the 24 foundation stories, ten gates, nine phases and all three
previously validated Mermaid sources. It adds the complete goal/capability matrix,
internal sandbox, practical optimization/residency, fair cached-prefix comparison,
token/context accounting and current conditional rollout policy. The roadmap is
a local companion draft; future public synchronization must make it accessible.

Fresh issue #276 readback still contains R4, updated 20:02:21Z. No completion
result, request ID or form URL was received for the earlier Submit. No replacement
write was attempted. Local R9 is not claimed synchronized. No push/PR/merge,
served-profile change, deployment, install, new firmware or model download occurred.
