# Serving upstream integration audit

Engine issue: xenodeve/Strata-xeno#6. Tracker: xenodeve/Qwen3.8-Flash-Next-Tuning#15. This branch is stacked on `xeno/afk-phase2` and keeps the Phase 1 cache candidate parked.

| Upstream PR | Local result | Evidence |
|---|---|---|
| #7 engine queue race | Accepted as cherry-pick `8f2187d` | `test_serve_serialization.py` was red before the patch because request two entered while request one's generator cleanup was pending; green after. |
| #8 conversation cache | Accepted as cherry-pick `c996648` | `test_serve_protocol.py` was red before the patch because prompt-progress `PP` was discarded; green after. A live two-request protocol run later returned `REUSED 55` for a continuation prompt. |
| #9 CPU pool idle sleep | Accepted as cherry-pick `dd05359` | `pool_idle_sleep.exe` used 312 ms process CPU during 150 ms idle with two workers before the patch, 31 ms after. |
| #10 small prompt reads through verify windows | Parked; cherry-pick `9aedbd2` was reverted by `eab0066` | A live same-input run with `--short-read 0` versus `64` changed output token 5 on the first request; second output matched. The result repeated with a one-slot profile containing a cold expert at layer 47, expert 511, so a hot GPU cache hit is not sufficient to explain it. `REUSED 55`/`REUSED 17` confirm the continuation route. Trace logs show batched versus windows. Evidence under `%TEMP%/strata-afk-runs/short-read*`. The diagnostic script is parked in a local stash named `park short-read parity probe`. |

The native Q2_0 engine cannot run spec 4 with zero cache allocation. The cold one-slot profile makes routed GPU hits unlikely but is not a proof of zero hits; #10 requires deeper state comparison before it can be used. PR #7–#9 are independent of that unresolved numerical path.
