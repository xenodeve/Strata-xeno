<h1 align="center">Strata-xeno</h1>

<p align="center"><b>A fork of <a href="https://github.com/Niko1221/Strata">Strata</a> tuned for two graphics cards and
for coding agents like Claude Code</b><br>
Qwen3.8-Flash-Next (125B parameters) on a gaming PC · NVIDIA · Windows · free and open source</p>

<p align="center"><a href="https://strata-xeno-website.vercel.app">Website</a> ·
<a href="bench/results/2026-10-04-speed-xeno-vs-upstream-0138/README.md">Benchmarks</a> ·
<a href="docs/BLUEPRINT.md">Blueprint</a> · <a href="https://github.com/Niko1221/Strata">Upstream Strata</a></p>

**Strata is [Niko1221](https://github.com/Niko1221/Strata)'s work.** It runs
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), a large mixture-of-experts model that normally
needs a server, on one graphics card plus system RAM.

**Strata-xeno** follows upstream (last merged: v0.1.38) and changes it for one goal: **the lowest latency in a real
coding-agent turn** on a PC with two cards. Those turns are a long cached conversation, then short new parts and
answers. The fork changes:
- how the experts are spread over the GPUs and RAM;
- how prompts are read across two cards;
- what the server does for Claude Code;
- the web app.

Every number below is measured on the fork's own PC, with the conditions next to it.

## How fast is it?

Measured against **upstream v0.1.38 run with its own settings**: the flags its `setup.py` writes, in its two two-GPU
forms. The setup:
- one PC, one session, two runs per arm in alternating order;
- Swift 1.5 IQ2_XS;
- one code-agent prompt per length, 256 output tokens, greedy, with the MTP draft layer.

The PC is an Intel Core i5-13500 with 48 GB DDR5, an RTX 5060 Ti 16 GB (PCIe 4.0 x4) and an RTX 4070 SUPER 12 GB (x16,
the display card), on Windows 11.

**Reads your prompt** (tokens/s)

| | 1K | 4K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Strata-xeno** | **416** | **1,021** | 1,469 | 1,536 | 1,508 |
| upstream v0.1.38, layer split | 284 | 768 | **1,566** | **1,734** | **1,690** |
| upstream v0.1.38, peer tier | 212 | 605 | 1,055 | 1,073 | 1,048 |

**Writes answers** (tokens/s)

| | 1K | 4K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Strata-xeno** | **69.2** | **72.7** | **63.2** | **58.0** | **62.8** |
| upstream v0.1.38, layer split | 40.4 | 40.0 | 39.3 | 34.8 | 37.8 |
| upstream v0.1.38, peer tier | 53.2 | 50.0 | 46.1 | 44.2 | 46.9 |

- **Answers are 30-45 % faster at every length** than upstream's fastest form here.
- **Short prompts are read 33-47 % faster.** Short prompts are what an agent sends after its first turn.
- **Upstream's layer split reads long prompts (32K and up) 6.6-12.9 % faster.** That is where the fork is still
  behind.
- **In Claude Code's own request shape** (#163), the fork reads six follow-up turns in 11.1 s against upstream's best
  13.2 s, and answers at 71.7 tokens/s against 72.3 (a tie with the peer tier).
- **It needs less RAM.** It uses 23.7 GiB and leaves 15.3 GB free. Upstream's settings, which keep every expert in
  RAM, used 36.9-39.8 GiB and ran this 48 GB PC out of free RAM.

Every run, the flags of each arm, draft acceptance and what was not measured:
[bench/results/2026-10-04-speed-xeno-vs-upstream-0138](bench/results/2026-10-04-speed-xeno-vs-upstream-0138/README.md).
Earlier comparisons and every lever tried: [docs/reports/](docs/reports/).

## What it adds over upstream

**Engine** (C++/CUDA):
- **Dynamic experts.** An expert a GPU owns has no copy in RAM, and the model starts by placing experts on the cards
  first. That is where the RAM saving comes from.
- **A second-GPU expert tier.** The second card holds its own experts and runs them during decode, one CUDA graph per
  layer. It swaps experts in pairs and keeps a free-VRAM floor so the display card stays usable.
- **Prompt reading on two cards:** an expert split and a two-lane wave. The second card computes the experts it owns
  while the first runs the rest.
- **The engine's own speed work:**
  - KV streaming tuned for depth;
  - short prompt parts read through the decode path;
  - checkpoints written off the critical path;
  - a CPU pool that rests between rounds;
  - AVX-VNNI CPU kernels for Q2_0 experts.
- **Capacity mode** for models larger than RAM: a bounded RAM cache with the NVMe behind it.
- **Thai in the draft vocabulary,** so Thai answers are drafted ahead too.

**Server and app** (Python, React):
- **For Claude Code:**
  - Anthropic Messages API additions;
  - request priority, and a separate cache slot for side requests, so they do not evict the main conversation's
    cache;
  - a loop guard and a thinking budget.
- **A new web app at `/`:** Chat, Dashboard, Live, Requests, Hardware and Settings, with a full Thai interface. The
  classic app stays at `/classic/`.
- **Agent tooling:**
  - coding tools named as in Claude Code, with its permission rules, hooks, sub-agents and rewind;
  - projects with several folders;
  - MCP and skills imported from your other coding apps.

**Measurement:**
- `STRATA_TIMELINE` records the whole pipeline in one run;
- an ABBA runner pairs arms in one session;
- a blueprint of the system is checked at commit time.

The full list, each item with its code and tests: [the website's register](https://strata-xeno-website.vercel.app/details).

## What you need

The fork is developed and measured on **NVIDIA, Windows 11, two cards**. Upstream's requirements are the floor:
- an RTX 20-50 series card with 12 GB of VRAM or more;
- 48 GB of RAM for IQ2_XS;
- about 80 GB of free disk on an SSD.

Full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

One card works. The second-GPU tier and the two-card prompt path need a second NVIDIA card.

AMD and Linux support comes from upstream's code. The fork has not built, measured or tested it.

## Install

```text
git clone https://github.com/xenodeve/Strata-xeno
cd Strata-xeno
START-HERE.bat --build        (Linux: ./setup.sh --build)
```

**`--build` matters.** Without it, setup downloads **upstream's** ready-made engine
(`PREBUILT_URL` in `setup.py`), which has none of the fork's engine changes. You would get the fork's server and app
on upstream's engine. With `--build`:
- setup installs the build tools if they are missing (Visual Studio Build Tools + CUDA Toolkit on Windows; it asks
  first);
- it compiles the engine from this checkout, which takes 20-40 minutes once.

The fork itself is built with VS2022, CUDA 13.3 and Ninja ([docs/BLUEPRINT.md](docs/BLUEPRINT.md)). The `--build`
path through setup has not been run end to end on a fresh PC for the fork.

Setup asks which model to use and writes `strata-<model>.json`. **The two-card profile is not written by setup.** The
fork's served configuration ("D2x") adds flags to that file's `args`, among them:
- `--secondary-expert-mib 6400`, `--exclusive-primary-experts`;
- `--adapt-swaps 8 --adapt-every 1`, `--pcie-frac 0`;
- `--short-read 256`, `--kv-resident 65536`.

It also sets these in the environment:
- `STRATA_PREFILL_EXPERT_SPLIT=1`, `STRATA_PREFILL_WAVE=1`, `STRATA_PREFILL_SPLIT_MIN=256`;
- `CUDA_VISIBLE_DEVICES`, with the primary card as device 0.

Do not list both cards in the config's `gpu` field: that makes the server add `--layer-split`. Every flag of the
measured profile is in the [benchmark's matrix](bench/results/2026-10-04-speed-xeno-vs-upstream-0138/matrix.json)
(arm `d2x`).

Models, sizes and what fits in how much RAM are unchanged from upstream: [docs/MODELS.md](docs/MODELS.md).

## Using it

- **In the browser:** `http://127.0.0.1:8080` opens the web app.
- **Claude Code:** `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`. Other apps that use Anthropic's API:
  `http://127.0.0.1:8080/v1/messages`.
- **OpenAI-compatible apps:** base URL `http://127.0.0.1:8080/v1`, any API key and any model name.
- **Thinking:** off, low, medium or high, in the chat menu or as your app's "reasoning effort".
- **From another device:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Always use a key.
- **It answers one request at a time.** A conversation's first prompt is read in full; follow-ups reuse the cached
  part.

More: [the API and every setting](docs/DETAILS.md#using-it), [where chats are stored](docs/INSTALL.md#where-things-are-stored).

## Something went wrong?

- **The PC freezes while the model loads.** That is normal for 1-3 minutes, longest the first time.
- **It is very slow and the disk light keeps blinking.** Not enough free RAM. Close other programs, or pick a smaller
  size.
- **Port 8080 is already in use.** Strata is already running.

More: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

Where to report a problem:
- **The fork's code** (the server, the app, the two-card engine paths): [this repository's
  issues](https://github.com/xenodeve/Strata-xeno/issues).
- **Something that also happens on upstream Strata:** [upstream's issues](https://github.com/Niko1221/Strata/issues).

## For developers and coding agents

- **[AGENTS.md](AGENTS.md):** the rules. Lowest latency first, root causes, a measured win before anything becomes a
  default, and PRD → issue → PR.
- **[docs/BLUEPRINT.md](docs/BLUEPRINT.md):** how the engine and the server fit together, and every configuration
  surface.
- **[docs/reports/](docs/reports/):** every finding, and a report per upstream merge.
- **[bench/results/](bench/results/):** measurements with their raw rows.
- **How the engine works** (upstream's): [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md),
  [docs/DETAILS.md](docs/DETAILS.md) and [the paper](docs/paper/Strata-Paper.pdf).

## Credits and license

**Strata** is the work of [Niko1221](https://github.com/Niko1221/Strata) and its contributors, under the
[MIT License](LICENSE); Strata-xeno is a fork of it under the same license. If Strata runs well for you, you can
support its author: [buymeacoffee.com/strataengine](https://buymeacoffee.com/strataengine).

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It is compressed by
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5) and Unsloth.
Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). A few parts and every model carry
their own licenses: [which ones](docs/HOW_IT_WORKS.md#license).
