# Native first-stage observation qualification

Tracker: [#276](https://github.com/xenodeve/Strata-xeno/issues/276).
Protocol: [bounded prefix extraction](../plans/2026-10-29-igc-prefix-observation.md).

## Result and scope

The optional `igc_prefix_probe` reproduces every first-attention row in two
existing full-model Gyro captures bitwise, using both one-token and eight-token
windows. A/B/A request order and internal repeated zero-state execution match.
This qualifies an offline source for contextual input-head experiments. It does
not establish input generalization, output result use or integrated-model speed.

The executable retains the real 48-layer geometry and uses the existing
`Verifier::set_stage(0,1,...)` path, its range-owned session, native Q8 embedding,
actual layer-zero dense matrices and all 512 layer-zero expert blobs in a static
GPU tier. It stops before the language head and samples no answer. The observed
seam is exactly the earlier `FirstAttention` tap before MoE combine; it is not
the paper's full first transformer layer. Layer-one PLE does not participate.

No source operator, production flag or served profile changes. This is an
offline diagnostic configuration, not the 262K/5-worker Hub benchmark baseline.

## Evidence

- Missing-executable RED: the opt-in refusal test fails with WinError2 before
  implementation. CUDA `89;120` optional-target build passes.
- `igc-prefix-fidelity.log`: malformed input passes, first real request fails
  because `FileExpertSource` inherits the base unsupported `read_into` operation.
  Source tracing identifies the wrong API. The correction uses its real `blob`
  interface with a completed synchronous upload before requesting another blob;
  no synthesized expert or fallback arithmetic.
- `igc-prefix-fidelity-02.log`: two tests pass in 3.619s. This includes malformed,
  short, oversized, out-of-vocabulary and trailing-data refusals before CUDA/model
  access; then full-capture comparison for requests 0,1,0 at window sizes 1,8.
  Test wall time is not a latency benchmark.
- The final repeat comparison is an explicit byte comparison, including signed
  zero. Its rebuild and verification are recorded separately below.

Sources are the existing `igc-i2-early-fit-01` capture and its frozen source
manifest. Input tokens, positions, archive metadata and all recorded F32 values
are compared, not just a checksum or a few logits. Model/tokenizer/source hashes
are unchanged from that collection; no new claim of rehashing the entire model.

Preflight via GetPerformanceInfo: system committed 24,966,488,064 bytes of
87,963,570,176; available commitment 62,997,082,112; physical available
33,436,352,512 bytes. Primary GPU was empty, display GPU used 1,706 MiB. These
are preflight samples, not measured peak residency. The probe reports explicit
owned allocation and native-dense/mapped-embedding bytes; total process/device
peaks will be recorded for collection sizing before a large dataset is started.

Runs use the existing bounded headless owned-job wrapper with no private-commit
cap. No HIP/SYCL build, dedicated prefix sanitizer pass, large-corpus collection,
trained input-quality improvement or full Hub disabled parity is claimed here.

Final byte-comparison build: `igc-prefix-build-final.log` DONE0;
`igc-prefix-fidelity-final.log` two tests pass, DONE0. Final executable SHA256
`0b625245a92ea4c6f962513b632ac132ff790798504d319c91c37864f60a6da6`.
