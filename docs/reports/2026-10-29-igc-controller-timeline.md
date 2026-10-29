# Learned-controller host scopes in the existing timeline

Issue #276, G8 measurement preparation. The red analyzer tests exposed the
missing controller-specific report and JSON key. The new report names the host
work already performed by `IgcLearnedDiagnostic`; it is not an optimization or
a claim that the live learned-model path has passed.

## Changes

`include/strata/program/igc_learned_diagnostic.hpp` records eight literal
`igc.controller.*` scopes: `load_host`, `bind`, `prepare`, `begin`, `stage`,
`cancel`, `audit` and `release`. Counts/prefix length/eligibility metadata use
the existing integer arguments, never dynamic span names, input text or paths.
`tests/xeno/perf/timeline.py` adds `igc_controller_report` to the CLI and JSON.
It returns attempt counts, inclusive host milliseconds and maxima by name.
`tests/igc/learned_lifecycle_test.cpp` flushes the existing recorder after all
of its controller/stream objects have been destroyed.

These are host regions, including checks, copies/enqueues or waits where the
underlying operation has them. A failed/refused call is an attempt; the report
does not infer its status or count GPU invocations. There is no GPU-runtime or
combined-total field. Do not add these values to overlapping verifier, parent
or GPU durations. No host timestamp is inserted into the captured head. This
does not yet split the actual input/readiness/arithmetic/output kernels inside
a live verifier window; that remains necessary when analyzing that mechanism.

The optional wrapper is absent on the hard-off feature path. With timeline off,
its existing lifecycle methods take the standard timeline enabled branch; the
cost is unmeasured and is not called free. No tensor math, graph address,
allocation policy, invocation ABI, engine protocol, Hub profile or default is
changed. No new environment variable is introduced.

## Verification

The new tests first failed with the missing report/CLI key. All 27 analyzer tests
then passed, including explicit hand-computed host times and an unrelated GPU
span that must not appear as controller GPU time. An unchanged trace returns an
empty controller report. The full CUDA 89;120 candidate and actual-wrapper test
build pass. Actual wrapper timeline OFF/ON runs both pass native literal result, nonzero
output mapping, cancellation, audit, cached-prefix/lineage, graph and reprepare
controls. They use old genuine saved activations and source-content binding,
without loading the language model or consuming the sealed acceptance64. The
IGC component is enabled in both controls; these are not feature-off model
parity runs. The timeline-OFF process has no timeline variable/file. The ON trace parses and exports to
Perfetto using the normal analyzer, with all eight scopes present:

| Host region | Attempts | Inclusive host ms | Maximum host ms |
|---|---:|---:|---:|
| load_host | 1 | 39.569 | 39.569 |
| bind | 1 | 52,343.727 | 52,343.727 |
| prepare | 3 | 32.763 | 16.457 |
| begin | 10 | 0.485 | 0.136 |
| stage | 12 | 3.486 | 2.195 |
| cancel | 4 | 0.118 | 0.046 |
| audit | 12 | 13.293 | 4.087 |
| release | 4 | 4.175 | 2.128 |

This is one actual component trace with successful and intentionally refused
lifecycle cases. It is not a per-turn distribution, per-success average, whole
startup timing or production invocation count. The bind region covers source
model/tokenizer content validation; its internal read/hash split is unmeasured.
Do not attribute that entire region to disk, SHA computation or model inference.
The existing `-744` and within-run bitwise mapping/reset assertions pass in both
processes; this does not establish cross-process raw-token model parity.
No GPU operations were changed, and no new sanitizer qualification is claimed
for this host-scope-only follow-up. HIP/SYCL/Linux are unrun.

Private receipts use `igc-controller-timeline-*` and `igc_controller_timeline_*`
under the existing evidence root. This is a functional instrumentation check,
not profiler-off ABBA, hard-off model parity or an end-to-end performance result.
No fix or speed claim follows from combining these runs with other sessions.

Frozen candidate SHA256:
`4cb668aac953b0712cb82095f8acdc41ed727e56294d457e6ab6542df87c5871`;
wrapper test `031a817643dd450789d30b83d24882c5334dc5f134a6d43e8cfb616ac3b3aeb8`;
raw trace `1bc212a6bc6e6e70ef2d16f43a38fc588cd8567c5998f47dea69d1e51bad3240`.
`igc-controller-timeline-qualification-01.json` records commands, source hashes,
binary hashes, telemetry and the parsed scopes. The sampled processes' highest
observed lifetime private peaks were 312,442,880 bytes (timeline OFF) and
316,465,152 bytes (ON), with 264/263 samples and a maximum gap of 0.204 seconds.
These are individual component observations, not a measured memory difference
or a full-model capacity estimate. Both bounded headless jobs completed and
drained their owned process trees.
