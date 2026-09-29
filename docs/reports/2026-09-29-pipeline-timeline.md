# The pipeline timeline, and what its first runs show (#33, #31)

Date: 2026-09-29. Branch `xeno/claude-merge-0.1.20`. Exe `tl.exe` (sha256 `e5b2b9854f8…`). Runs:
`%TEMP%/strata-claude-stage/TL*.{stdout,stderr}`, traces `tl2.json` (default prompt path), `tl3.json` (copy issuer
thread, ring 384) and `tl4.json` (dual-GPU decode). Reports: `tl*.report.txt`. All runs used HIGH class and
`--pool-priority 2`.

## The instrument

`STRATA_TIMELINE=<file>` records every host thread and GPU lane on one clock. `tests/xeno/perf/timeline.py` turns the
file into a budget. Usage, lanes and cost are in `tests/xeno/perf/README.md`.

- **Tests.** `xeno_timeline_core` (threads, append flushes, lane reuse), `xeno_timeline_gpu` (device spans in order,
  on the host clock) and `tests/xeno/test_timeline.py` (15 hand-worked traces). The server side is covered by
  `serve/test_timeline.py`.
- **Clock.** The C++ `steady_clock` and Python's `perf_counter` share the QPC timebase. Checked with
  `scratchpad/clockcheck.cpp`: the C++ reading fell between two Python readings 3 of 3 times.
- **Cross-check against the counters.** On `tl4`, decode ran 76 rounds at 39.28 ms per round. Each timeline span
  (ms/round) matches the engine's own counter from the same run:

  | timeline span | ms/round | counter | ms/round |
  |---|---|---|---|
  | `wait gpu` | 19.0 | `wait for rings` | 18.99 |
  | `cpu experts` | 12.5 | `pool` | 12.54 |
  | `verify stage` | 1.3 | `host` | 1.26 |
  | `verify commit` | 1.0 | `commit` | 0.96 |
  | `4070 wait` | 0.7 | secondary `wait` | 0.71 |

- **Cost.** Prefill takes about 6 % longer with the timeline on, in a same-session ABBA on the 8,023-token prompt
  (on 12.78/12.16 s, off 11.95/11.55 s). The output was identical (`8c6c97e1`).

## What the first runs show (measured; the causes marked as hypotheses are not)

### Default prompt path (`tl2`, 8,023 tokens, single GPU)

The prompt path took 11,317 ms.

**The copy engine was busy 4,197 ms (37.1 %).** Its idle time between copies splits three ways:

| idle between copies | ms | share of run |
|---|---|---|
| not issued yet (host) | 4,942 | 43.7 % |
| issued, not started (device) | 901 | 8.0 % |
| in the issue call | 2 | 0.0 % |

While a copy was not issued yet, the issuing thread (main) was in `router sync` for 4,540 ms of the 4,942.

- **The largest gaps are about 200 ms each, at every fourth layer (3, 7, 11, … 47).** Those are the QSA layers,
  where `qsa attn` alone takes about 126 ms of GPU time. Each gap is a `router sync`. The host blocks on the
  router's result, so it issues nothing while attention runs.
  - This settles #31's question for the default path: the copy engine idles because the only thread that issues
    copies is waiting in the router synchronize.
- **The chunk start costs the GPU 1,268 ms (11.2 % of its timeline) in `embed+steps`.** Over the same time the host
  is in `ple rows (host)` for 672 ms (the first chunk's PLE read, which nothing reads ahead) and in the
  per-token embedding loop (577 ms outside any other span).
  - Hypothesis: the GPU waits for those two host tasks. A batched embedding lookup, and a PLE read-ahead for the
    first chunk, would remove most of the 1.27 s.
- **The host thread's own budget:** `router sync` 4,615 ms (40.8 %), `stager wait` 2,716 ms (24.0 %) and
  `copy issue` 1,166 ms.
- **All 20,395 copies were `copy staged`.** None of the experts was pinned in this configuration.
- **The four stager threads spent 95.3 % of their time in `stager wait buffer`,** and only 4.6 % in memcpy (0.091 ms
  per expert). The measured numbers:
  - one copy takes 0.206 ms;
  - `issue -> copy start` has a p50 of 3.19 ms.

  Sixteen copies at 0.206 ms is about 3.3 ms, close to that p50. Hypothesis: the stager's 16 pinned buffers cap how
  many copies can be in flight, so the main thread waits in `stager wait`.

### Copy issuer thread, ring 384 (`tl6`; supersedes `tl3`)

> **Correction (same day).** The first version of this section, measured on `tl3`, said that "a copy that is
> already in flight does not progress while a `gdn` or `qsa attn` kernel runs". It gave 40 copies averaging 35.7 ms
> during `gdn`, 161 averaging 10.8 ms during `qsa attn`, and the copy engine busy 8,642 ms. That was an instrument
> fault.
>
> - **The fault:** in `issue_one` the copy's start event was recorded **before** the host blocked in
>   `stager->wait()`, so a span covered the host's wait as copy-engine time. The code review of #33 found it.
> - **The fix:** the start is now recorded right before the copy.
> - **The check:** `xeno_copy_overlap_probe` shows that a pinned H2D copy progresses at full speed (6.3-6.9 GB/s)
>   during a 100 ms single-block kernel, a VRAM-streaming kernel, and a kernel that fills every SM for 1.5 s.
>
> The default-path numbers above are unaffected: copy busy was 4,197 ms before the fix and 4,194 ms after it (`tl7`).

The prompt path took 11,533 ms, the same as `tl2` within noise. The copy engine was busy 5,326 ms (46.2 %), and a
copy takes 0.204 ms (median, during every phase). Its idle time between copies splits three ways:

| idle between copies | ms | share of run |
|---|---|---|
| issued, not started (device) | 4,494 | 39.0 % |
| not issued yet (host) | 378 | 3.3 % |
| in the issue call | 12 | 0.1 % |

The 378 ms not issued yet is all `issuer wait ring slot`.

- **The gaps are about 144 ms each, at the QSA layers.** Moving the issuer off the main thread moved the idle time
  from the host to the device: the copies are issued, but they wait for their ring slot. The ring holds 384 entries,
  which is about 79 ms of copies at 0.206 ms each; a QSA layer's attention runs about 126 ms.
  - Hypothesis: to hide a layer's attention, the prefetch depth has to be about 700 or more entries. The ring slots
    are VRAM (1.38 MB each) and the stager buffers are pinned host memory.
- **The issuer thread spent 83.6 % of its time in `stager wait`,** and the main thread spent 3,617 ms in
  `wait issuer`.

## Off-mode cost

Timeline exe `tl.exe` (sha256 `e5b2b9854f84b…`) against the pre-timeline exe built from `4a5c9ce` (`pretl.exe`,
`70f6f9d7bf31a…`). Both had the timeline off. The runs were in the same session and alternated; every run used HIGH
class and `--pool-priority 2`. The outputs were identical (`8c6c97e1` for 8K, `d2edde68` for code256).

| workload | runs per arm | pretl | tl (timeline off) |
|---|---|---|---|
| 8K prefill, mean | 4 | 12.07 s (11.89-12.48) | 12.27 s (11.87-12.69) |
| code256 decode on two GPUs, mean | 4 | 84.42 tok/s | 84.26 tok/s |

The difference is inside the run-to-run spread. The logs are `offP*`, `offT*`, `code*` in
`%TEMP%/strata-claude-stage`.

## Next steps (for #31 / #32)

1. **Stop the router synchronize from blocking the only issuing thread.** Issue ahead of it, or plan routes on the
   device. Target: about 4.5 s of copy idle on the 8K prompt.
2. **Deepen the prefetch within the memory budget.** The ring and stager depth set how much of a layer's attention
   the copies can hide (`tl6`). That is a VRAM and pinned-RAM trade, so the developer decides it.
3. **Measure a batched embedding lookup and a first-chunk PLE read-ahead** against the 1.27 s `embed+steps` cost.
