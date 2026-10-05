# Speed by prompt length: Strata-xeno against upstream v0.1.38 with its own settings

Intel Core i5-13500, 48 GB DDR5-7000, RTX 5060 Ti 16 GB (PCIe 4.0 x4 through the chipset) + RTX 4070 SUPER 12 GB
(PCIe 4.0 x16, the display card), NVIDIA driver 616.92, Windows 11. Swift 1.5 Qwen3.8 Flash-Next IQ2_XS with the same pack and MTP draft layer in every arm
(#165). Every run: [`matrix.json`](matrix.json), with each arm's exe, sha256, flags and environment.

**Method.** Upstream's speed tables (`2026-09-29-speed-0126`) use one-shot `strata generate`. Upstream's layer split
only runs with `--serve`, so these runs use serve mode instead:
- one code-agent prompt per length (C++ and CUDA source with a task), sent as system + user messages;
- the server applies the chat template, so "1K" is 1,064 tokens here against upstream's 1,017;
- a fresh server boot for every run, then one request: greedy, 256 output tokens, the speed figures from the server log.

The two methods are comparable in shape, not identical.

## The arms

- **Strata-xeno D2x:** the served config (`strata-flash-next-d2x.json`, exe `run-main-13d06a7`, sha256 `161435e9…`):
  - the 4070 as an exclusive expert tier (`--secondary-expert-mib 6400 --exclusive-primary-experts`);
  - the prompt path split across both cards in waves;
  - `--short-read 256`, `--kv-resident 65536`;
  - paired adaptive swaps (`--adapt-swaps 8 --adapt-every 1`), 13 pool workers.
- **Upstream:** a pristine v0.1.38 build (`99f3dbd`, exe `run-up138`, sha256 `289c0c33…`) with the flags its own
  `setup.py` writes for this model:
  - `--expert-profile data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --kv int8`;
  - `--max-context 262144 --kv-resident 32768`;
  - the adaptive tier, pool workers and `--short-read` at upstream's defaults.
  - Two forms:
    - **layer split:** `--layer-split auto --vram-reserve-mib 2560`. The reserve keeps the display card's 2.5 GB of
      headroom; upstream's default is 700 and applies to both cards.
    - **+ peer:** `--peer-device 1 --peer-reserve-mib 2560`, plus `STRATA_ARENA_PIN_GIB=8`. Without it the peer's
      stream fails with "out of memory" under WDDM. The cards have no P2P, so upstream keeps the prompt path on the
      primary.
- **Strata-xeno + upstream's flags (fx):** the D2x exe with upstream's flags above, plus `--exclusive-primary-experts`
  (dynamic experts: a GPU that owns an expert keeps no host copy). The adaptive tier stays at upstream's defaults.
  - Three forms: one GPU; layer split (which also needs `--no-prefill-borrow` in the fork); + peer.

## A. Strata-xeno D2x against upstream (one session, 20:10-20:52)

Two runs per arm, in the order D2x, layer split, peer, peer, layer split, D2x at each length.

### Prompt (tokens/s)

| arm | 1K | 4K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Strata-xeno D2x** | **416** | **1,021** | 1,469 | 1,536 | 1,508 |
| upstream, layer split | 284 | 768 | **1,566** | **1,734** | **1,690** |
| upstream + peer | 212 | 605 | 1,055 | 1,073 | 1,048 |

### Output (tokens/s)

| arm | 1K | 4K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Strata-xeno D2x** | **69.2** | **72.7** | **63.2** | **58.0** | **62.8** |
| upstream, layer split | 40.4 | 40.0 | 39.3 | 34.8 | 37.8 |
| upstream + peer | 53.2 | 50.0 | 46.1 | 44.2 | 46.9 |

### Draft acceptance (share of drafted tokens accepted)

| arm | 1K | 4K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strata-xeno D2x | 0.80 | 0.79 | 0.75 | 0.65 | 0.75 |
| upstream, layer split | 0.95 | 0.78 | 0.69 | 0.67 | 0.66 |
| upstream + peer | 0.87 | 0.80 | 0.72 | 0.75 | 0.62 |

**Strata-xeno leads output at every length**, by 30-45 % over upstream's faster form there (+ peer) and 61-82 % over
its layer split. **It also leads prompt reading at 1K-4K**, by 33-47 % over the layer split.

**Upstream's layer split reads long prompts faster:** by 6.6 / 12.9 / 12.1 % at 32K / 64K / 128K.

A likely reason, not measured: the D2x prompt path splits the experts across the two cards and the 5060 Ti reads its
share over the x4 link, while the layer split gives each card whole layers.

The two runs of a cell differ by a median of 2.7 % for prompts and 5.0 % for output, and by up to 18 % (upstream +
peer, 4K prompt). Output moves with the share of accepted drafts (above). The output lead is far outside that spread;
the 6.6 % prompt gap at 32K is not.

## B. Strata-xeno's exe with upstream's flags and dynamic experts (another session, 22:02-22:41)

Two runs per arm, in the order one GPU, layer split, peer, peer, layer split, one GPU at each length.

The first 1K layer-split run did not boot: the fork's layer split refuses prompt-path borrowing ("add
--no-prefill-borrow"). The flag was added and the second run booted, so the cells marked ¹ are one run.

### Prompt (tokens/s)

| arm | 1K | 4K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| one GPU | 191 | 584 | 1,010 | 1,078 | 1,051 |
| layer split | 262¹ | 725 | 983 | 1,048 | 1,029 |
| + peer | 188 | 628 | 1,053 | 1,069 | 1,044 |

### Output (tokens/s)

| arm | 1K | 4K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| one GPU | 46.2 | 36.9 | 43.3 | 48.2 | 54.1 |
| layer split | 41.2¹ | 44.1 | 54.7 | 61.1 | 50.5 |
| + peer | 48.3 | 44.6 | 67.3 | 58.3 | 65.0 |

### Draft acceptance

| arm | 1K | 4K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| one GPU | 0.93 | 0.75 | 0.76 | 0.73 | 0.77 |
| layer split | 0.98¹ | 0.95 | 0.93 | 0.95 | 0.78 |
| + peer | 0.87 | 0.74 | 0.79 | 0.67 | 0.76 |

### Where the decode experts ran: CPU share of decode expert entries (%)

From the fork's `request metrics` line in each run's log; upstream's exe does not print it.

| arm | 1K | 4K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| fx, one GPU | 40 | 27 | 27 | 30 | 30 |
| fx, layer split | 29 | 28 | 27 | 27 | 23 |
| fx, + peer | 17 | 11 | 8 | 10 | 10 |
| D2x (session A) | 18 | 19 | 21 | 21 | 24 |

- **No layer or expert is read from the SSD.** Every fork run with a result logs `nvme loads 0`: all 39 of them
  (10 D2x, 29 fx). With dynamic experts the GPUs hold part of the experts and RAM holds the rest.
- **Upstream's flags on the fork read prompts far slower than D2x.** At every length they read 188-1,078 tokens/s,
  where D2x reads 416-1,536; at 32K and above, 983-1,078 against 1,469-1,536. D2x's own settings are what make the
  prompt path fast; dynamic experts alone are not.
- **Output is mixed.** At 1K-4K the fx arms write 37-48 tokens/s, where D2x writes 69-73. At 32K-128K the peer form
  writes 58-67, level with or above D2x's 58-63. The two runs of a cell differ by a median of 5.6 % and by up to 45 %
  (layer split at 32K: 64.7 against 44.6), so these cells are loose.
- **The D2x row is not in this session.** It is session A's figure, so a comparison between it and these rows crosses
  sessions. This project measures prefill drifting between sessions on the same exe.

## C. Upstream on one GPU: the 4070 SUPER (x16) against the 5060 Ti (x4) (a third session, 22:44-22:49)

The order was 4070, 5060 Ti, 5060 Ti, 4070. Every run is shown, because the 4070's two 1K runs differ by half.

| | 1K prompt | 4K prompt | 1K output | 4K output |
| --- | --- | --- | --- | --- |
| upstream on the 4070 SUPER (x16) | 432 / 278 | 772 / 909 | 33.4 / 38.0 | 32.2 / 28.3 |
| upstream on the 5060 Ti (x4) | 200 / 195 | 601 / 639 | 37.7 / 37.6 | 35.9 / 33.0 |

**Not a clean link comparison.** The 4070 arm carries `--vram-reserve-mib 2560` (the display card's headroom), and the
5060 Ti arm runs upstream's default of 700. The cards hold 12 GB and 16 GB.

The 32K runs and the fork's one-GPU run on the 4070 did not finish. Claude Code's low-memory reaper stopped the queue
during the first 32K boot (22:50), and the queue has not been run again.

## Why this PC is slower than upstream's RTX 5070

Upstream's own table for IQ2_XS (`2026-09-29-speed-0126`): an RTX 5070 12 GB on PCIe 5.0 x16, Ryzen 5 7600 (AVX-512),
64 GB DDR5-5200. It is another machine and a one-shot method, so it gives context, not a pairing:

| tokens/s | 1K | 4K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX 5070 prompt (upstream's figure) | 534 | 1,256 | 2,092 | 1,754 | 1,752 |
| RTX 5070 output (upstream's figure) | 79.6 | 78.6 | 76.3 | 63.7 | 62.7 |
| Strata-xeno D2x prompt (here) | 416 | 1,021 | 1,469 | 1,536 | 1,508 |
| Strata-xeno D2x output (here) | 69.2 | 72.7 | 63.2 | 58.0 | 62.8 |

Upstream's own settings on this PC reach, at best, 284-1,734 tokens/s for prompts and 44-53 for output (table A):
below the 5070's figures at every length.

The likely causes fall into three groups by how well they are supported:

- **Measured: upstream's settings run this PC out of RAM.** It keeps every expert in RAM, and its docs list IQ2_XS at
  48 GB, which is exactly this machine. In the Claude Code session of the same day (#163, `memwatch2.csv`):
  - free RAM fell to 0.01-0.6 GB in every upstream form;
  - the system paged from disk at a median of 2,245-3,179 hard page-ins a second;
  - D2x, which holds part of the experts on the GPUs, kept 15.3 GB free and paged at a median of 116 a second.
- **Supported, not isolated: the 5060 Ti's x4 link slows prompt reading.** Moving upstream's one-GPU run to the 4070's
  x16 link reads prompts faster, by 80 % at 1K and 36 % at 4K on the means (table C). The arms also differ in VRAM and
  reserve. Output barely moves (28-38 tokens/s on either card), so the link is not what limits upstream's output here.
- **Unmeasured hypotheses:**
  - the i5-13500 has no AVX-512, and upstream computes the RAM-held experts on the CPU;
  - the 5060 Ti and the 4070 SUPER are slower GPUs than the 5070.

  Nothing here measures either one on its own.

## The Claude Code request shape, same day (#163)

The prompt and output speed of real agent turns, from one session (base, layer split, peer, one GPU, then back). The
arms are D2x and upstream's three forms above; the upstream one-GPU arm is on the 5060 Ti. The scripts are `smkv.py`,
`upab-*` and `memab-*`. Two runs per arm.

| | prefix 9,932 tokens | six follow-up parts (408-2,816 tokens) | output over six 200-token turns | 400-token answer |
| --- | --- | --- | --- | --- |
| **Strata-xeno D2x** | **8.01 / 8.00 s** | **11.14 / 10.97 s** | **71.7 / 71.7 tok/s** | **75.6 / 74.7 tok/s** |
| upstream, layer split | 7.96 / 7.97 s | 13.26 / 13.10 s | 61.3 / 64.9 | 57.9 / 63.2 |
| upstream + peer | 14.34 / 13.40 s | 23.80 / 23.85 s | 73.8 / 70.7 | 69.7 / 64.2 |
| upstream, one GPU | 13.08 / 13.12 s | 23.67 / 23.67 s | 59.2 / 56.3 | 54.0 / 50.8 |

The follow-up parts are 1K-3K-token reads, which is where table A shows D2x ahead. The 9,932-token prefix is a tie
with the layer split.

## Where the records are

`C:\Strata-exp\merge-138-record\` on the measuring PC:
- the scripts: `speedtab_serve.py`, `mkprompts.py`, `mkup.py`, `mkfx.py`, and `mkspeedmd.py`, which wrote
  `matrix.json` and these tables;
- the raw rows: `speedtab-serve.json`, `speedtab-fx.json`, `speedtab-4070.json`;
- each run's config, engine log and server output: `speedtab-serve/<length>-<arm>-<run>.*`.
