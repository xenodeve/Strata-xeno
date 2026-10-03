# #84 N0: which host-cache policy cuts NVMe misses? (Swift IQ3_XXS, capacity mode)

Date: 2026-10-02. PRD: Strata-xeno #84 (slices #85-#91). Branch `xeno/84-n0-trace`. Full table:
[`2026-10-02-n0-cache-policy-sweep-table.md`](2026-10-02-n0-cache-policy-sweep-table.md).

## Answer

**No new policy pays.**
- The runtime's decayed LFU is the best of the policies tested, or within about 1 %, at every RAM cache from 2 to 16 GiB and on every workload.
- Speculation-aware admission is worse everywhere, by 3-38 %.
- The recent-tokens window (Apple's "LLM in a Flash" idea) changes nothing from 6 GiB up. Below that it moves results by at most about 2 %.
- W-TinyLFU and window-k LRU are mostly worse: -2.2 % to +6.3 %, better only on Thai at 2, 8 and 12 GiB.
- The prompt-tail seed cuts decode misses by 0.1-1.5 %.

**The lever is elsewhere.** It is the cache size, and the cost of each miss (#64, #81, #82). One measured fact points at prefetch: the experts that rejected drafts route are used again soon (see "Pollution" below). That makes the draft positions a ready-made predictor for N4 / #44.

## Method

- **Traces:** `--dump-routing` format 2 (#85/#86): commit-aware, with the GPU-owned set and the boot host set.
- **Exe:** `C:\Strata-exp\run-84v2\strata.exe` (`7b04d0b7`). Both GPUs, daily args with `--adapt-swaps 0` (the trace's ownership is a boot snapshot), `STRATA_POOL_SPIN_US=0` (#64), RAM cache 6 GiB, greedy.
- **Workloads:**
  - **code:** `bash-files`, `cpp-bug`, `json-review`, `py-async` (`tests/xeno/perf/trace-prompts`) and the 8K `long8k` prompt, 512 new tokens each.
  - **thai:** `thai-net`, `thai-plan` and the `thai256` bench prompt, 512 each.
  - **agent:** one serve session of 8 requests (`tests/xeno/perf/n0_agent_session.py`). It is synthetic, not a recorded Claude Code session: a short turn, a ~11K-token document with an 800-token answer, and short follow-ups (one in Thai) that reuse the prefix, run twice.
- **Replay:** `tests/xeno/perf/n0sim.py`, sweep `tests/xeno/perf/n0_sweep.py`.
  - Decode windows only. Each trace starts from its own boot state, filled in profile order.
  - At 6 GiB, the profile-order boot gives the same loads as the trace's exact boot set (0.0 % difference) on all three workloads.
- **Parity (#87):** 5060 only, swaps off, 6 GiB. Simulated 117.728 NVMe loads per round, the engine 116.609 (+0.96 %, tolerance ±3 %).
- **Unit:** loads per emitted token (committed positions). "vs baseline" compares the same traces at the same cap.

## Results

### Temporal locality (#88): moderate

Mean overlap of a committed token's experts with the last k committed tokens' experts, per layer:

| k | code | thai | agent |
|---:|---:|---:|---:|
| 1 | 0.348 | 0.464 | 0.280 |
| 4 | 0.527 | 0.645 | 0.500 |
| 16 | 0.719 | 0.789 | 0.724 |

### Pollution (#88): real by count, but not pollution

**15-25 % of loads** (code 0.155, Thai 0.248, agent 0.169 at 6 GiB) are routed only by draft positions that were later rejected. Those experts are not wasted, though.

Speculation-aware admission gives them no score and evicts them first. It **increases** loads per token:

| | 2 GiB | 4 | 6 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| code | +3.8 % | +7.1 % | +8.0 % | +7.9 % | +9.4 % | +7.8 % |
| thai | +25.1 % | +35.9 % | +37.1 % | +32.5 % | +38.0 % | +36.4 % |
| agent | +2.7 % | +5.6 % | +8.6 % | +8.9 % | +10.2 % | +10.6 % |

The median reload distance also falls under it (Thai at 6 GiB: 26 → 8 windows). The experts evicted on probation come back within a few windows. **A rejected draft still routes much of what the next committed tokens need.**

### Window policies (#89)

Loads per token against the baseline (decayed LFU):

| policy | code | thai | agent |
|---|---|---|---|
| window + LFU, k = 1-16 | -0.2 % to +0.3 % at 2-4 GiB; **0.0 % from 6 GiB** | -0.8 % to +0.7 % at 2-4 GiB; 0.0 % from 6 GiB | -0.9 % to 0.0 % at 2-4 GiB; 0.0 % from 6 GiB |
| window-k LRU (k = 4) | +0.0 % to +4.0 % | -2.2 % to +4.2 % | +1.3 % to +5.9 % |
| W-TinyLFU (1 % window) | +0.0 % to +3.6 % | -2.1 % to +6.3 % | +1.6 % to +6.3 % |

From 6 GiB the window protects 2-34 % of the resident bytes and still changes no eviction. The decayed LFU already scores recently used experts above the rest, so its victim is never inside the window.

### Prompt-tail seed (#90, agent trace, baseline + seed)

| cap GiB | best seed_k | decode loads/token | seed reads | start hit rate (first 8 decode windows) |
|---:|---:|---:|---:|---:|
| 2 | 4 | -0.1 % | 3.4 GB | 0.385 → 0.391 |
| 6 | 16 | -0.4 % | 3.5 GB | 0.691 → 0.704 |
| 8 | 16 | -1.2 % | 2.8 GB | 0.775 → 0.801 |
| 16 | 16 | -1.5 % | 0.8 GB | 0.957 → 0.964 |

The seed is small. In serve, only prompt parts of at most `--short-read` (64) tokens run as windows and leave routes (#86), so a request's seed comes from its short new-turn part only. Generate traces have no prompt routes at all.

### Cache size is what moves the number

Baseline loads per emitted token:

| | 2 GiB | 4 | 6 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| code | 39.89 | 25.31 | 17.83 | 12.88 | 6.87 | 3.41 |
| thai | 45.28 | 26.01 | 18.27 | 12.74 | 7.47 | 4.00 |
| agent | 63.49 | 37.34 | 24.71 | 17.02 | 8.08 | 3.34 |

## What this changes

- **N3 (admission and bounded caches, #11):** keep the decayed LFU. Do not build window, W-TinyLFU or speculation-aware admission. Each one measured as equal or worse here.
- **N4 (prefetch, #44):** the evidence is now specific. Experts routed at draft positions are reused, so reading the drafts' host misses as one batch is a prefetch candidate. That batch is all layers' experts for the draft positions, read when the drafts are verified or earlier. **Unmeasured:** it needs the router outputs before the layer runs, which is the open question #44 already states.
- **Per-miss cost** stays the main lever at a given cache: #64 (the scheduler stall), #81 (IoRing and aligned direct reads), #82 (the mirror's copy choice).

## Limits

- **Traces:** 8 generate traces of 512 tokens, plus one synthetic 8-request serve session. Not a recorded Claude Code session.
- **Ownership:** adaptive swaps were off, so GPU ownership was static. The daily config swaps experts between the GPU tiers and the host, which this replay does not model.
- **Draft model:** the MTP head's acceptance on Swift is 0.28-0.81 per window (Thai lowest). A different draft model changes the pollution numbers.
- **Precision:** the simulator is double-precision and the runtime scores in float32. Parity is +0.96 %.

---

# สรุปภาษาไทย

## คำตอบ

**ไม่มีนโยบายใหม่ตัวไหนคุ้ม**
- decayed LFU ที่ runtime ใช้อยู่ดีที่สุดในบรรดาที่ทดสอบ หรือห่างไม่เกินราว 1 % ทุกขนาด RAM cache ตั้งแต่ 2 ถึง 16 GiB และทุก workload
- การรับเข้าแบบรู้ speculative แย่ลงทุกกรณี 3-38 %
- window ของ token ล่าสุด (ไอเดีย "LLM in a Flash" ของ Apple) ไม่เปลี่ยนอะไรเลยตั้งแต่ 6 GiB ขึ้นไป ต่ำกว่านั้นขยับผลได้ไม่เกินราว 2 %
- W-TinyLFU และ window-k LRU ส่วนใหญ่แย่ลง: -2.2 % ถึง +6.3 % ดีขึ้นเฉพาะ Thai ที่ 2, 8 และ 12 GiB
- การ seed จากท้าย prompt ลด miss ตอน decode ได้ 0.1-1.5 %

**คันโยกอยู่ที่อื่น** คือขนาด cache และต้นทุนต่อ miss แต่ละครั้ง (#64, #81, #82) มีข้อเท็จจริงที่วัดได้หนึ่งข้อที่ชี้ไปทาง prefetch: expert ที่ draft ที่ถูก reject เรียก ถูกใช้อีกในเวลาไม่นาน (ดู "การปนเปื้อน" ข้างล่าง) ทำให้ตำแหน่ง draft เป็นตัวทำนายสำเร็จรูปสำหรับ N4 / #44

## วิธีการ

- **trace:** `--dump-routing` format 2 (#85/#86) รู้ผล commit มี set ที่ GPU ถือและ set บน host ตอน boot
- **exe:** `C:\Strata-exp\run-84v2\strata.exe` (`7b04d0b7`) สองการ์ด args แบบ daily แต่ใส่ `--adapt-swaps 0` (ความเป็นเจ้าของใน trace เป็น snapshot ตอน boot), `STRATA_POOL_SPIN_US=0` (#64), RAM cache 6 GiB, greedy
- **workload:**
  - **code:** `bash-files`, `cpp-bug`, `json-review`, `py-async` (`tests/xeno/perf/trace-prompts`) และ prompt 8K `long8k` ตัวละ 512 token ใหม่
  - **thai:** `thai-net`, `thai-plan` และ prompt bench `thai256` ตัวละ 512
  - **agent:** session ของ serve หนึ่งชุด 8 request (`tests/xeno/perf/n0_agent_session.py`) เป็นแบบสังเคราะห์ ไม่ใช่ session ที่บันทึกจาก Claude Code จริง: turn สั้น, เอกสาร ~11K token กับคำตอบ 800 token และ follow-up สั้น ๆ (หนึ่งตัวเป็นภาษาไทย) ที่ใช้ prefix ซ้ำ รันสองรอบ
- **การเล่นซ้ำ:** `tests/xeno/perf/n0sim.py` sweep ด้วย `tests/xeno/perf/n0_sweep.py`
  - เฉพาะ window ของ decode แต่ละ trace เริ่มจากสถานะ boot ของตัวเอง เติมตามลำดับของ profile
  - ที่ 6 GiB การ boot ตามลำดับ profile ให้จำนวนโหลดเท่ากับ set ตอน boot ของ trace ที่แน่นอน (ต่าง 0.0 %) ทั้งสาม workload
- **Parity (#87):** 5060 ใบเดียว ปิด swap 6 GiB จำลองได้ 117.728 การโหลด NVMe ต่อรอบ engine ได้ 116.609 (+0.96 %, tolerance ±3 %)
- **หน่วย:** การโหลดต่อ token ที่ปล่อยออก (ตำแหน่งที่ commit) "vs baseline" เทียบ trace เดียวกันที่ cap เดียวกัน

## ผล

### Locality ตามเวลา (#88): ปานกลาง

ค่าเฉลี่ยการซ้อนทับของ expert ของ token ที่ commit กับ expert ของ k token ที่ commit ล่าสุด ต่อ layer:

| k | code | thai | agent |
|---:|---:|---:|---:|
| 1 | 0.348 | 0.464 | 0.280 |
| 4 | 0.527 | 0.645 | 0.500 |
| 16 | 0.719 | 0.789 | 0.724 |

### การปนเปื้อน (#88): มีจริงตามจำนวน แต่ไม่ใช่การปนเปื้อน

**15-25 % ของการโหลด** (code 0.155, Thai 0.248, agent 0.169 ที่ 6 GiB) ถูกเรียกเฉพาะโดยตำแหน่ง draft ที่ภายหลังถูก reject แต่ expert เหล่านั้นไม่ได้เสียเปล่า

การรับเข้าแบบรู้ speculative ไม่ให้คะแนนพวกมันและไล่ออกก่อน ผลคือ**เพิ่ม**การโหลดต่อ token:

| | 2 GiB | 4 | 6 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| code | +3.8 % | +7.1 % | +8.0 % | +7.9 % | +9.4 % | +7.8 % |
| thai | +25.1 % | +35.9 % | +37.1 % | +32.5 % | +38.0 % | +36.4 % |
| agent | +2.7 % | +5.6 % | +8.6 % | +8.9 % | +10.2 % | +10.6 % |

ระยะการโหลดซ้ำ (median) ภายใต้นโยบายนี้ก็ลดลงด้วย (Thai ที่ 6 GiB: 26 → 8 window) expert ที่ถูกไล่ออกเพราะอยู่ในสถานะทดลองกลับมาภายในไม่กี่ window **draft ที่ถูก reject ยังเรียก expert จำนวนมากที่ token ที่ commit ถัดไปต้องใช้**

### นโยบายแบบ window (#89)

การโหลดต่อ token เทียบกับ baseline (decayed LFU):

| นโยบาย | code | thai | agent |
|---|---|---|---|
| window + LFU, k = 1-16 | -0.2 % ถึง +0.3 % ที่ 2-4 GiB; **0.0 % ตั้งแต่ 6 GiB** | -0.8 % ถึง +0.7 % ที่ 2-4 GiB; 0.0 % ตั้งแต่ 6 GiB | -0.9 % ถึง 0.0 % ที่ 2-4 GiB; 0.0 % ตั้งแต่ 6 GiB |
| window-k LRU (k = 4) | +0.0 % ถึง +4.0 % | -2.2 % ถึง +4.2 % | +1.3 % ถึง +5.9 % |
| W-TinyLFU (window 1 %) | +0.0 % ถึง +3.6 % | -2.1 % ถึง +6.3 % | +1.6 % ถึง +6.3 % |

ตั้งแต่ 6 GiB window ปกป้องไบต์ที่อยู่ใน cache 2-34 % แต่ไม่เปลี่ยนการไล่ออกเลยสักครั้ง decayed LFU ให้คะแนน expert ที่เพิ่งใช้สูงกว่าตัวอื่นอยู่แล้ว ตัวที่ถูกไล่จึงไม่เคยอยู่ใน window

### Seed จากท้าย prompt (#90, trace agent, baseline + seed)

| cap GiB | seed_k ที่ดีที่สุด | การโหลดตอน decode ต่อ token | การอ่านเพื่อ seed | start hit rate (8 window แรกของ decode) |
|---:|---:|---:|---:|---:|
| 2 | 4 | -0.1 % | 3.4 GB | 0.385 → 0.391 |
| 6 | 16 | -0.4 % | 3.5 GB | 0.691 → 0.704 |
| 8 | 16 | -1.2 % | 2.8 GB | 0.775 → 0.801 |
| 16 | 16 | -1.5 % | 0.8 GB | 0.957 → 0.964 |

seed ได้ผลน้อย ใน serve มีแค่ส่วนของ prompt ที่ไม่เกิน `--short-read` (64) token เท่านั้นที่วิ่งเป็น window และทิ้ง route ไว้ (#86) seed ของแต่ละ request จึงมาจากส่วนสั้น ๆ ของ turn ใหม่เท่านั้น trace ของ generate ไม่มี route ของ prompt เลย

### ขนาด cache คือสิ่งที่ขยับตัวเลข

การโหลดต่อ token ที่ปล่อยออกของ baseline:

| | 2 GiB | 4 | 6 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| code | 39.89 | 25.31 | 17.83 | 12.88 | 6.87 | 3.41 |
| thai | 45.28 | 26.01 | 18.27 | 12.74 | 7.47 | 4.00 |
| agent | 63.49 | 37.34 | 24.71 | 17.02 | 8.08 | 3.34 |

## สิ่งที่ผลนี้เปลี่ยน

- **N3 (การรับเข้าและ cache ที่มีขอบเขต, #11):** คง decayed LFU ไว้ ไม่ต้องสร้าง window, W-TinyLFU หรือการรับเข้าแบบรู้ speculative ทุกตัววัดได้ว่าเท่าเดิมหรือแย่กว่าที่นี่
- **N4 (prefetch, #44):** ตอนนี้หลักฐานเจาะจงขึ้นแล้ว expert ที่ตำแหน่ง draft เรียกถูกใช้ซ้ำ การอ่าน miss บน host ของ draft เป็น batch เดียวจึงเป็นผู้สมัครสำหรับ prefetch batch นั้นคือ expert ของทุก layer สำหรับตำแหน่ง draft อ่านตอนที่ verify draft หรือก่อนหน้านั้น **ยังไม่ได้วัด:** ต้องมีผลลัพธ์ของ router ก่อนที่ layer จะทำงาน ซึ่งเป็นคำถามเปิดที่ #44 ระบุไว้อยู่แล้ว
- **ต้นทุนต่อ miss** ยังเป็นคันโยกหลักที่ขนาด cache เดิม: #64 (scheduler ค้าง), #81 (IoRing และการอ่านตรงแบบ align), #82 (การเลือกสำเนาของ mirror)

## ข้อจำกัด

- **trace:** trace ของ generate 8 ชุด ชุดละ 512 token บวก session serve แบบสังเคราะห์ 8 request ไม่ใช่ session ที่บันทึกจาก Claude Code จริง
- **ความเป็นเจ้าของ:** ปิด adaptive swap ไว้ ความเป็นเจ้าของบน GPU จึงคงที่ config แบบ daily สลับ expert ระหว่าง tier บน GPU กับ host ซึ่งการเล่นซ้ำนี้ไม่ได้จำลอง
- **draft model:** อัตราการรับของ MTP head บน Swift อยู่ที่ 0.28-0.81 ต่อ window (Thai ต่ำสุด) draft model อื่นจะเปลี่ยนตัวเลขการปนเปื้อน
- **ความแม่นยำ:** simulator คำนวณแบบ double ส่วน runtime ให้คะแนนแบบ float32 parity อยู่ที่ +0.96 %
