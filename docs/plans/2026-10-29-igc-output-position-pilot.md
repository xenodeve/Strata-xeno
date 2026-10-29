# I3 pilot: real native teacher-forced output positions

The gated residual/native-head STE composition passes, but the first-anchor
archive alone cannot train or establish full result use. Collect actual frozen
Gyro head-input states at every teacher-forced completion position next.

## Frozen dataset and controls

Use a deterministic30-question pilot:24 training and6 development, balanced
add/subtract/multiply and English/Thai. Seed20261104. Signed input magnitude at
most99. Exclude every operand pair in the earlier input pilot, expanded input
corpus/development and untouched acceptance64. Hold development result values out
of fitting as well as operand pairs. This pilot does not establish62-digit range,
new languages or spontaneous use; it prepares full-sequence output training.

Frozen row SHA256 `913fbdf842ce9cc55862200167147dee26f116f81e99ea5792e49ccb817725e3`.
Using the verified native wrapper/tokenizer produces421 teacher-forced positions,
181 numeric-symbol targets and a maximum86-token input prefix. This includes
the real end-of-turn token. These counts are preparation, not model execution.

`tools/igc/output_corpus.py` produces independent Python oracle labels **only for
offline training/evaluation**. Before fitting, execute all typed requests through
the actual I1a CUDA executor and compare its canonical results to the independent
oracle. Use those checked typed results as mapper inputs, not formatted answers
or token IDs. Oracle operands/results isolate I3 while I2 remains unpassed.

Completions are short correct reasoning text, closing the existing thinking
block and giving the integer answer. Teacher forcing consumes previous correct
completion tokens; expected next-token labels enter only the LM loss. They never
enter the result mapper as features. The model's own one-token prediction is
retained as a baseline observation, not substituted for the teacher-forced token.

## Actual model path and ownership

Use the selected Hub profile/model/context262K/workers5/cache/offload/RoPE and
its real `StrataEngine`, `engine_args` and `child_env` helpers. Only substitute
the isolated candidate executable, task-owned logs/profile output and existing
opt-in observation flags. Direct native token requests allow exact canonical
prefixes plus a teacher-forced assistant continuation; they are data collection,
not an HTTP/Hub latency measurement. The source wrapper must reproduce all64
existing full native prefixes before use.

The selected profile also enables its Vision encoder. Start it with the same
actual `Vision` and `vision_env` helpers/order before the language engine, retaining
its GPU setting and source-pin its executable/mmproj/model files. A first collector
attempt refused this profile before model startup because its conservative guard
excluded Vision; do not bypass that by disabling the user's subsystem. Monitor
and close only the owned encoder/engine. No image encoding is requested by the
text-only corpus.

Preserve the launcher environment as well as JSON: selected `strata-hub.bat`
lines131-132 sets `CUDA_VISIBLE_DEVICES=1,0` and clears `CUDA_DEVICE_ORDER`.
The JSON has no GPU-list/environment override. A second diagnostic attempt
omitted those inherited settings and the engine correctly refused the D2x
secondary arena before capture. Both owned processes drained. Require the exact
resolved launcher mapping before any model allocation, with a negative test;
do not reinterpret that harness failure as a model/branch compatibility defect.

The actual engine consumes each prefix and completes its owned one-token request.
Capture its mixed activation/logits and verify exact token-prefix/position metadata
before associating a label. Prefix cache reuse remains the engine's normal valid
reuse. A sampled token that differs from the next teacher-forced token is discarded
through normal prefix reconciliation, never silently treated as accepted state.

Bind invocation ID/result position to the initial request anchor. Reuse the checked
result for later output positions without recomputing arithmetic or advancing
extra language positions. This is offline dataset ownership; production commit,
rejected suffix and exactly-once injection remain I4 requirements.

Bound the total to1,024 observation records, each prefix256 tokens and each request
30s. Total900s/outer960s including boot/cleanup. Use actual Hub resource admission,
the disclosed cap-free diagnostic system headroom checks and passive process/job
telemetry. Headless owned process tree, no private-commit cap, no served changes,
no duplicate engine while another heavy job runs. If resources are unavailable,
do not launch. Keep raw records and source/build/config/tokenizer/protocol hashes.

## Later fit and acceptance

Freeze the fit recipe after inspecting only dataset completeness/shape, before
evaluating development predictions. Use the qualified native-head forward and
declared late-head STE; only the gated mapper trains. Score teacher-forced numeric
tokens and complete-number sequence consistency separately from ordinary words.
Controls include absent/corrupted/result-swapped features, zero residual and the
unchanged base. Full free-running response correctness requires the native mapper
inside the engine and cannot be inferred from teacher-forced accuracy. No usable
integration, I3 completion or Hub benefit claim follows from this capture alone.

Change inventory: frozen output corpus and tests; task-owned data collector using
existing Hub helpers and diagnostic flags; bounded reader/source validation;
BLUEPRINT and evidence report. No new core inference behavior in this pilot.
