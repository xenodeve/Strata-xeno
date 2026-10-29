# I3 disabled-path parity diagnosis

The first three-arm functional smoke has an unresolved raw-token mismatch on one
Thai ordinary prompt, while candidate-on/off unmatched prompts agree. Do not
attribute it to the hook or dismiss it as noise without reproducing it.

First discriminator: unchanged pre-hook A, candidate with feature absent B, B, A
in one session. Each process starts from the same original expert profile and
receives exactly the same four smoke requests in the same order: known12*64,
known92-18, English ordinary, Thai ordinary. Max96/greedy; actual selected
Gyro-M262K/five-worker/Vision/D2x/MTP-suffix profile and launcher environment.
No adapter flags, state-hash/timeline/profiler, memory cap, cache/offload/priority
change or other heavy job. This diagnoses output reproducibility; no speed claim.

Freeze raw IDs, first differing index, DONE/MTP/tier counters and all stdout/stderr,
build/source/config/profile hashes. Compare A-self, B-self and each cross pair per
case, rather than only aggregate final-answer accuracy. Baseline self-difference
would disprove a uniquely candidate-induced mismatch; stable self arms but repeat
cross differences require tracing candidate changes. Equal outputs on one repeat
do not establish broad release parity. Do not overwrite the original smoke.

Only after this controlled reproduction, use existing state diagnostics/timeline
on both arms as needed to localize a mechanism. Any instrumentation changes are
diagnostic and must not be mislabeled as the actual Hub baseline. Do not disable
MTP/adaptation or change workload to manufacture an acceptance pass.

Run via `output_integration --smoke --parity-only`, cap-free owned headless tree,
diagnostic system preflight/emergency floors and900s total/outer960s. Report those
admission checks as harness safety, not an existing Hub private-memory restriction.
All source files stay fixed until DONE. I2/fresh acceptance64 and output-model
training are unaffected. No production default activation or full I3 completion.
