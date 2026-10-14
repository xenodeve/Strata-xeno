# Gyro (agentionai rotor quants) on Strata-xeno

Gyro is agentionai's quantization of Qwen3.8-Flash-Next: the experts in trellis ("rotor") types, the activations
rotated by a block-128 Hadamard transform. The fork's `main` runs both published files (merged from agentionai's `rc2`
on 2026-10-13, #217), and its setup installs them.

The targets are the published files of `agentionai/Qwen3.8-Flash-Next-Gyro-GGUF`, used unchanged:

| File | Experts | GPU memory for the experts | Status |
|---|---|---:|---|
| `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf` (58.5 GB) | TQK6 gate/up, TQK7 down, block-128 Hadamard | 24.0 GiB | supported by agentionai's rc2 code and validated by agentionai (Linux, RTX 5090, RTX 3090); on the fork it runs on the same code without Gyro-M's TQ2_T tuning, and has not been measured on the fork's PC yet |
| `Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` (92.0 GB) | TQ2_T, block-128 Hadamard | 29.9 GiB | validated by agentionai (Linux, RTX 5090, RTX 3090); **served daily on the fork's PC** (Windows, two cards) since 2026-10-13 |

Both files carry their own experts in the trellis types TQ2_T/TQK6/TQK7 and record the activation rotation in
`prism.hadamard.*` metadata. Strata reads both natively. About half of each file is the
n-gram (PLE) table, which stays on disk and is read row by row; it is not loaded into RAM or VRAM.

The draft head is Strata's own MTP runtime, prepared from the original model's MTP weights as for the other
Qwen3.8-Flash-Next models. It is not the GGUF draft file from the Gyro repository.

## Install with setup (Windows and Linux, NVIDIA)

```text
START-HERE.bat --family gyro --model M        (Linux: ./setup.sh --family gyro --model M)   recommended
START-HERE.bat --family gyro --model S        Gyro-S: smaller; recommended below 48 GB of RAM
```

Or run `START-HERE.bat` and choose "Qwen3.8-Flash-Next (Gyro)" from the model list.

**Gyro-M is labelled recommended in setup.** By the developer's own tests on the fork's PC (2026-10-14), its quality is
close to ISTA-DASLab's IQ3_S, which is larger where it counts: IQ3_S keeps 50.3 GB of experts in memory and asks for
62 GB of RAM; Gyro-M keeps 29.9 GB and asks for 48. On disk Gyro-M's GGUF is the bigger file (92 against 83.6 GB),
because about half of it is the n-gram table, which is read from the SSD. Its speed is about IQ3_S's. The
comparison's runs are not recorded in this repository; no benchmark here backs the quality claim yet.

**Gyro-S's quality is on par with ISTA-DASLab's IQ2_XS in less memory.** The
[model card](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) at the pinned revision (`8fa6f45c`, with
the updated Gyro-S of 2026-10-10 and Gyro-M of 2026-10-11) gives these numbers. They are agentionai's measurements:
KL divergence and top-1 agreement against unsloth's Q8_0 on a held-out corpus of 2026 technical writing and code
(`-c 2048`, 60 chunks), with GPU memory as GPU-resident weights (the n-gram table stays on disk):

| file | GPU memory | KLD (lower is better) | top-1 |
|---|---:|---:|---:|
| **Gyro-S** | **27.64 GiB** | **0.401** | **75.29 %** |
| ISTA-DASLab GSQ-RCO IQ2_XS | 36.52 GiB | 0.420 | 74.22 % |
| **Gyro-M** | **34.98 GiB** | **0.291** | **78.75 %** |
| ISTA-DASLab GSQ-RCO IQ3_XXS | 43.80 GiB | 0.311 | 77.68 % |

- Gyro-S is slightly ahead of IQ2_XS with 24 % less memory.
- Gyro-M is ahead of IQ3_XXS with 20 % less memory. The card has no IQ3_S row; the IQ3_S comparison above is the
  developer's own.
- These numbers were not reproduced on the fork.

What setup does differently for Gyro:
- **It compiles the engine on this PC** (10-20 minutes, once). No ready-made engine has the trellis types. The build
  uses agentionai's llama.cpp fork as its GGML, pinned at `dc255f2a` (`AGENTION_LLAMA_COMMIT` in `setup.py`). Its
  ggml has the trellis types the prompt path reads through MMQ, and its gguf-py is what the pack and MTP tools use.
  - The Gyro engine goes to its own folder, `engine-gyro/` (build folder `build-gyro/`). The other models keep
    `engine/`, so an update of one never rebuilds the other with the wrong GGML.
  - Setup installs the build tools if they are missing (Visual Studio Build Tools and the CUDA Toolkit on Windows; it
    asks first).
- **It downloads one GGUF** from the repository at a pinned revision (`8fa6f45c`) and checks its size and SHA-256.
  `--gguf-dir <folder>` uses a copy you already have.
- **It writes the pack with `--compat-bf16`** to `packs/gyro-m` (or `gyro-s`), the config `strata-gyro-m.json` and
  the start script `run-gyro-m`.
- **RAM:** Gyro-M asks for 48 GB. That is measured: the fork's PC runs it with a private commit of 43.5-46 GiB at
  131-262K context. Gyro-S asks for 32 GB, an estimate from its experts (24 GiB, scaled the way the Coder's 23.4 GiB
  are), and setup recommends it below 48 GB.
- **Speed work:** Gyro-S's TQK6 / TQK7 experts get the fork's CPU-kernel unroll and AVX-VNNI (#220). The TQ2_T-only
  decode tuning (one 8-byte window per four states, big-endian windows) is Gyro-M's. The tier and swap work (#229)
  applies to both.

Not offered by setup for Gyro yet:
- **AMD.** Use `tools/gyro_setup.sh --backend hip` on Linux; it is experimental (below).
- **The CUDA 12 engine** (Pascal / Volta).
- **The low-RAM mode.** `--experts-bin` / `--mmap-experts` have not been tried with the trellis types.
- **Images.** The vision encoder's text side cannot read the Gyro GGUF.

The fork's measured two-card profile is not written by setup; see the README.

## Install by hand (Linux): `tools/gyro_setup.sh`

`tools/gyro_setup.sh` does every step below on Linux:

```sh
tools/gyro_setup.sh --model S --backend cuda --cuda-arch 120       # RTX 50 series
tools/gyro_setup.sh --model S --backend cuda --cuda-arch 89        # RTX 40 series
tools/gyro_setup.sh --model S --backend cuda --cuda-arch 86        # RTX 30 series / RTX A6000
tools/gyro_setup.sh --model S --backend hip  --hip-arch gfx1201    # Radeon RX 9070 / AI PRO R9700 (experimental)
```

It prints the command that starts the server. The steps, by hand:

## 1. Build

From this repository. Strata's GGML dependency must be the agentionai llama.cpp fork, which has the trellis types:

```sh
git clone --depth 1 https://github.com/agentionai/llama.cpp ../agention-llama.cpp
# NVIDIA. RTX 50 series: use CUDA 13 (CUDA 12.8 crashes the prompt path on sm_120: issues #220, #224)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=$PWD/../agention-llama.cpp
# AMD (experimental: kernel parity and an end-to-end run validated on gfx1151; in a non-root container add
# --group-add $(getent group render | cut -d: -f3), otherwise HIP finds no device)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSTRATA_GGML_DIR=$PWD/../agention-llama.cpp
cmake --build build -j
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## 2. Download and pack

```sh
.venv/bin/pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 .venv/bin/hf download agentionai/Qwen3.8-Flash-Next-Gyro-GGUF \
  Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --local-dir models
STRATA_GGUF_PY=../agention-llama.cpp/gguf-py .venv/bin/python tools/iq_pack.py \
  --gguf models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --out packs/gyro-s --compat-bf16
build/native_expert_parity --load-check packs/gyro-s models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf
```

The pack is 1.5 GB (dense weights, tokenizer, the Hadamard and expert tables); the experts are read from the
GGUF. Keep the GGUF on a fast local SSD: the n-gram table is read from it for every token.
`--load-check` must end with `0 failures`.

## 3. Draft head

```sh
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

## 4. Serve

Save as `strata-gyro-s.json` at the repository root:

```json
{
  "exe": "build/strata",
  "args": [
    "--pack", "packs/gyro-s",
    "--native", "models/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf",
    "--expert-profile", "data/expert-profile.bin", "--expert-cache", "auto",
    "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
    "--mtp", "mtp/rt", "--max-context", "131072", "--kv", "int8"
  ],
  "cwd": ".",
  "tokenizer": "packs/gyro-s/tokenizer",
  "model_name": "gyro-s-strata",
  "log": "strata-gyro-s.log",
  "host": "127.0.0.1",
  "port": 8080
}
```

```sh
.venv/bin/python -m serve.server --engine strata --config strata-gyro-s.json --port 8080
```

The web interface is at `http://127.0.0.1:8080`; API clients use `http://127.0.0.1:8080/v1` (OpenAI) or
`/v1/messages` (Anthropic). `--expert-cache auto` sizes the expert cache to the free VRAM: on a 32 GB card at
128k context it holds 23,464 of 24,576 experts (22.9 GiB). On smaller cards it holds fewer and streams the rest
from RAM; nothing else changes.

## Measured on the fork's PC (Gyro-M, two cards, Windows)

The fork serves Gyro-M daily (hub key 3, `strata-gyro-m-d2x.json`). The PC is an i5-13500 with 48 GB DDR5, an RTX 5060
Ti 16 GB (PCIe 4.0 x4, primary) and an RTX 4070 SUPER 12 GB (x16, display card, the second expert tier), on Windows 11.
The config is not setup's:
- the agentionai-ggml engine;
- 262K context with KV streaming;
- both cards' expert tiers, with the adaptive swaps and a profile learned from use (#229).

Coding-agent turns (`turns217.py`: five sessions, 23 turns from 1.3K to 57K tokens, 256 new tokens), same-session ABBA,
2026-10-14:

| measure | value | source |
|---|---|---|
| decode, median per arm | 60.1 -> 65.1 tok/s with the #229 swaps (later arms 64.7-71.4) | #229, #220 |
| prompt read | 1,400-2,600 tok/s for 4K-57K tokens; shorter is slower (1.6K: 1,000, 0.6K: 430) | timeline `tl-s229e` |
| where the experts ran | 5060 Ti 69-70 %, 4070 SUPER 21-22 %, CPU 8-9 % | ABBA STATS |
| draft acceptance | 65-69 % | ABBA |
| private commit | 43.5 GiB at 131K, 45.4-46.2 GiB at 262K | #217, the #229 soak |

The fork's own CPU kernels for TQ2_T (MSVC unroll + AVX-VNNI, one 8-byte window per four states, big-endian windows) are
bit-exact against the reference decode (`tq_cpu_parity`). They took a one-token expert from 700 to 410-428 µs on a
P-core (#220). Where the rest of a decode round goes: #220 and its critical-path page.

**Not measured on the fork:** Gyro-S; setup's single-card Gyro config; a fresh `START-HERE.bat --family gyro` run end to
end (the setup path was added on 2026-10-14 and checked with its unit tests only); AMD; the stock-GGML build (it runs
decode, but the prompt path then has no MMQ for the trellis types); token parity against llama.cpp.

## Measured by agentionai

All runs: greedy or the stated sampling, same prompt tokens as llama.cpp, MTP draft `--spec 4 --spec-min-p 0.5`,
int8 KV. "llama.cpp" is the agentionai fork with the CUDA trellis fixes (in its `main` since 2026-10-04).

### RTX 5090 (32 GB), Gyro-S, all experts on the GPU (2026-10-03)

Decode tokens/s, single prompts of 34-177 tokens, 384 new tokens, `--max-context 16384` (all 24,576 experts cached):

| | prose | JSON | code |
|---|---:|---:|---:|
| Strata | 139.9 | 221.3 | 209.0 |

Server, `tools/gyro_setup.sh` config (128k context, 23,463 experts cached), thinking off, temperature 0.7,
top-k 20, top-p 0.95, min-p 0.05, 512 new tokens, 3 repetitions:

| | prose | JSON | code | copy |
|---|---:|---:|---:|---:|
| Strata | 131.7 | 200.1 | 195.2 | 201.8 |
| llama.cpp, MTP draft | 116.8 | 213.0 | 179.0 | 220.7 |
| llama.cpp, no draft | 121.8 | 123.2 | 122.5 | 121.2 |

Long real-text prompts, prefill tokens/s: 4k tokens 3,456 (llama.cpp 2,520), 16k tokens 4,883 (llama.cpp 2,471).

### RTX 5090, Gyro-M (2026-10-03)

The experts (29.9 GiB) do not all fit next to the KV cache; `auto` caches 18,007 of 24,576. Same server protocol:
prose 113.8, JSON 170.8, code 153.0, copy 175.7 tokens/s; prefill 4k 3,369, 16k 4,854 tokens/s. llama.cpp needs
`-ncmoe 8` for this file on a 32 GB card and decodes at 70.8 tokens/s without a draft.

### RTX 3090 (24 GB), 2026-10-03

Same server protocol, 128k context, `auto` cache:

| | experts cached | prose | JSON | code | copy | prefill 4k | prefill 16k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gyro-S | 15,756 | 31.8 | 45.2 | 42.2 | 45.1 | 1,402 | 1,863 |
| Gyro-M | 11,811 | 27.8 | 39.4 | 34.4 | 37.7 | 1,363 | 1,910 |
| llama.cpp, Gyro-S, `-ncmoe 10`, 32k context, no draft | (10 layers on the CPU) | 16.0 | 16.2 | 16.2 | 16.5 | 468 (`-ncmoe 11`) | 479 |

On a 24 GB card, cache misses run on the CPU, so the CPU matters: this box had a Zen 2 EPYC (AVX2, no AVX-512).
The llama.cpp row is the build before the CUDA trellis fixes; those added about 3 % at 24 GB, where the CPU layers
dominate.

The Strata rows predate Strata's own CPU kernels for the trellis types (`src/kernels/cpu/tq_*.cpp`, AVX2 and
AVX-512, picked at run time). Before them a cache miss ran ggml-cpu's `vec_dot`, one token at a time; the new kernels
take about 45 % of its time per expert on AVX2 and decode a block once for all the tokens of a verify window
(`build/tq_cpu_parity --bench` measures them on your CPU). `STRATA_NO_TQ_KERNELS=1` goes back to ggml-cpu;
`STRATA_TQ_ISA=avx2` or `scalar` caps the instruction set (for comparisons).

### Smaller caches (RTX 5090, 2026-10-02, before the int8 decode kernels)

| Expert cache | prose | JSON | code |
|---|---:|---:|---:|
| 24,576 slots (all experts) | 116.4 | 194.3 | 173.5 |
| 22,000 slots | 111.4 | 175.7 | 166.1 |
| 9,000 slots (about a 16 GB card's room) | 81.4 | 74.0 | 96.8 |
| 3,000 slots (about an 8 GB card's room) | 48.0 | 52.4 | 57.9 |

The smaller caches were emulated on the 5090 (fast GPU, PCIe 5.0 x16 at ~51 GB/s). A real 8 or 16 GB card is a
slower GPU, often on a slower link: read those rows as "it stays usable", not as a prediction for a specific card.
The decoded output was read and is fluent and correct; token-level parity against llama.cpp has not been
checked. A real chat at 128k context worked through the web interface.

### Two GPUs (2026-10-04, 2× RTX 4090, no NVLink/P2P)

- **Layer split** (Strata splits the layers across both cards automatically) works with Gyro, including the
  idle-card prompt helper (`STRATA_PREFILL_HELP=1`). Repeated runs give identical output.
- **Decode with a second card** (`--peer-device 1` or `--expert-cache-device1 auto`) works and matches single-GPU output.
- **Prompt processing on the peer card** needs GPU-to-GPU access (NVLink or data-centre GPUs); without it Strata keeps
  the prompt on the first GPU and logs `no P2P`.

Single RTX 4090 smoke run (Gyro-S, MTP draft, greedy): code 107.5, prose 84.3, JSON 76.3 tokens/s.

## Numerics

- **Decode** (Strata's own expert kernels) reads the trellis weights through an int8 copy of the codebook with one
  global scale (`tq_lut_i8`, relative RMS error 0.85 % of the codebook's standard deviation, far below the
  quantization error itself), against q8_1 activations.
- **Prompt processing** decodes them otherwise: the FP16 path through the fp16 codebook (`tq_lut_f16`, what GGML's
  own dequantizer reads); the MMQ path through the llama.cpp fork's kernels, which read the int8 codebook as well
  on the fork's current `main`.
- So decode and prompt results agree within tolerance, not bit for bit. `native_expert_parity` checks one expert
  on the decode kernels against GGML's float dequantizer and a float SwiGLU: relative L1 error below 3e-2, the
  bound every expert type gets there. Its "bitwise equal" lines compare Strata's decode-once and per-entry decode
  kernels with each other, not with a float reference.

## Known limits

- **Not measured yet:** token parity against llama.cpp.
- **Benchmark mode only:** `strata --tokens-file ... --max-new N` decodes N tokens past the end of the answer;
  compare speeds only up to the answer's end.
- **AMD:** the trellis kernels pass parity on gfx1151 (ROCm 7.2.1) and a short end-to-end run there generated
  correctly (~28-31 tokens/s); no speed tuning or RDNA4 run yet.
- **Gyro-M:** 29.9 GiB of experts do not all fit a 32 GB card next to the KV cache; `auto` streams the rest.
- **Not validated end to end yet** (Strata runs them and logs a warning): an AMD GPU other than gfx1151, and
  prompt processing on a peer GPU over P2P.
