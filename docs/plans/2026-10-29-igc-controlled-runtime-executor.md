# Controlled I1a-to-I3 runtime connection

The completed fixturev1 experiment isolates output mapping with earlier checked
GPU results. It must not be described as runtime arithmetic. Next replace that
diagnostic input with an explicitly versioned **typed request** fixturev2.

V2 header: `IGCOFX02`, count1..64, model/tokenizer/artifact digests. Each192-byte
record holds canonical prefix SHA256 and the existing160-byte ExactRequest ABI1.
No expected result, formatted answer or token target is stored in the engine's
fixture. Validate domain/op/reserved/owner/position/canonical operands before GPU
allocation; complete-prefix matching and actual content binding remain. Keepv1
reader for historical output-only replay, label request/result mode explicitly.

At the matched first decode window, copy **one** typed request to the bound primary
stream and enqueue I1a `execute_exact(...,count=1)`. Broadcast its device result to
the verifier's bounded row slots with one copy kernel, then native result mapping
and native head proceed on the same stream. No host answer calculation, device
result readback, text serialization, new language positions or repeated arithmetic
per decoded token. Unmatched requests still stage an empty failure. Diagnostic
per-window status checking remains disclosed and is not a production speed path.

Broadcast must leave source row0 untouched while copying to1..count-1; validate
count1..64/explicit stream and test under regular graph/repeats/sanitizers. Exact
errors remain valueless; no numeric zero fallback. The current controlled corpus
has two-digit operands; artifact interface capacity62 is not learned range evidence.

Serialize v2 from the frozen corpus operands and canonical request metadata only.
Independent expected values stay **outside** the engine for offline evaluation.
Test request parsing, wrong domain/op/reserved/owner/anchor/digit, duplicated
prefix, truncation/trailing bytes and unmatched prompts. Runtime result bytes must
match the prior I1a-checkedv1 result for all30 known requests before free generation.
Then repeat the bounded actual-Hub-profile smoke, preserving MTP/D2x and no cap.

Before this new run, separate answer correctness from the original single-integer
format rubric. Offline extraction accepts a bare integer or an explicit English/
Thai result statement, and a literal equation only when its normalized left side
matches the known requested expression. Multiple conflicting declared values or
a wrong equation are flagged; a truncated thinking section is not a final answer.
No arithmetic/executable text is evaluated by this checker. Preserve the original
format score separately; never rewrite earlier reports. Tests freeze the observed
equation-plus-correct-answer case and ambiguous/contradictory/wrong-equation cases.

This is controlled executor/output integration, not learned I2 or a release gate.
Retain the existing same-prehook-binary instability finding and do not manufacture
raw parity by changing actual Hub settings. Broader output, dynamic trigger,
transaction/snapshot/UI and practical end-to-end benchmarks remain necessary.
