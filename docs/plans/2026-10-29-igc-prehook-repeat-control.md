# Same-prehook-binary repeat discriminator

The A-B-B-A session shows candidate-self3/4, prehook-self4/4 and one failing cross
pair. The earlier prehook run also emitted the alternative39-token Thai sequence,
but cross-session raw tokens are not an acceptance gate. Test a narrow additional
disproof of a candidate-only mechanism: four fresh processes of the **same frozen
prehook binary**, same selected Hub profile and same four requests/order.

Invoke the existing `--smoke --parity-only` harness with both `--candidate` and
`--baseline` pointing to the exact same prehook executable. A/B names remain arm
labels; report source hashes and `same_executable_control:true` explicitly. No
candidate code, adapter flags, state-hash/profiler, model/profile/settings change,
memory cap or concurrent heavy job. Start each from the same expert profile.

Keep all raw IDs, per-arm counters, first differing position and outcome. A
same-binary difference demonstrates pre-existing workload instability without
proving its cause. If all four match, that does not prove the baseline universally
stable; preserve all earlier evidence and choose the next discriminator from it.
This is correctness diagnosis, not speed measurement or a parity acceptance pass.

Use the existing cap-free bounded task tree/diagnostic system safety floors,
900s/outer960s. Do not alter those checks or any binary/source while it runs.
