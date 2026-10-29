# Actual learned-wrapper lifecycle on captured activations

Issue #276, G3-G5 component follow-up after `ada2e825`. This tests the real
`IgcLearnedDiagnostic` object and its verifier head hook, with actual model and
tokenizer content binding. It does not load language-model weights or generate
new language tokens. The input is one old inspected synthetic prompt, its
canonical 66-token prefix, and genuine saved early/final activations at anchor
65. Fresh acceptance64 remains sealed.

## Changes and root-cause evidence

Implementation index: `program/igc_learned_diagnostic.hpp:116` (`cancel`),
`:124` (`release`), and `src/core/igc_pipeline.cpp:73` (pending head-status
invalidation). The public lifecycle tests live in
`tests/igc/learned_lifecycle_test.cpp`; the faster status-reset repro is in
`tests/igc/pipeline_trial_test.cpp`.

The unused context-only invalidation method is replaced with an explicit
`cancel(error)` lifecycle API. A context-only cancellation stub failed
`cancel_must_clear_retained_result_before_next_head`: the hook reads retained
pipeline state independently of the host `active_` flag. Cancellation now closes
staging, invalidates feature history and stages an unavailable device invocation
on the owning stream. Another `begin()` is required before staging again.

The native pipeline test separately reproduced a stale successful head status
after staging a new invocation. The request/result had been reset, while
`head_status` retained the previous value. `stage_inputs()` now sets all 64 head
statuses to an invalid sentinel before any new head runs. This adds a 256-byte
asynchronous memset per enabled stage; its latency is not measured or claimed
free. Numeric execution and invocation schemas are unchanged.

The expanded wrapper test passed cancellation, audit, graph and cache controls
but failed `stage_requires_new_begin_after_reprepare`. `release()` had reset the
staging flag while retaining the monotonic invocation counter. It now closes
staging until the next `begin()`, preserving owner-ID monotonicity. The complete
sequence then passed. These are component/API lifecycle findings, not evidence
that the served Hub's user-cancellation path previously exhibited these faults;
the new API is not wired to that path yet.

## Qualified behaviors

The test content-binds the existing Gyro GGUF/tokenizer before allocating the
small controller. Through the actual head hook, native readouts/execution return
the literal expected result `31 * -24 = -744` and apply a nonzero learned output
mapping. It checks:

- Cancellation clears the retained value and successful audit before another
  ordinary or captured head can use it; cancellation is idempotent.
- Cancellation before the first stage closes that request without execution.
- Cancellation during graph capture is refused. The test ends capture and
  successfully retries revocation before any further head, without treating a
  failed cancellation as permission to continue.
- Valid same-prefix feature reuse gives the same mapped bytes with a new owner.
  Denied lineage, changed prefix and ineligible scope bypass without a stale
  answer or forced cold language prefill.
- Captured graphs are destroyed before release. Double release is safe; after
  reprepare, staging requires a new begin, and a fresh owned request works.

All lifecycle calls are serialized by the owner. Prefill producers must be
drained and prior borrowed-input work fenced before cancellation or release.
An arbitrary CUDA failure is not the tested capture-refusal case: failed
revocation/deferred completion must stop execution. Fault injection for that
case remains unrun. Module cancellation also does not undo language KV/GDN,
mixed rows, RNG or accepted-token state.

## Build, test and resource receipts

The full CUDA `89;120` build is pinned by
`igc-learned-lifecycle-final-build-01.identity.json`:

- Candidate executable: `74494c52e952eff3ba36eb7495112ba92136db65128aa65df874a65fe1d1a7e3`.
- Wrapper test: `ac8d973026fc3002f301495d8c3726fb5e1c98ae9cce60389ce9b80a72d89e7c`.
- Fixture input: `565967c0257d6098fd2cd9cf308079e4fbc737f8ae9a37df47323b3d6c6a7660`.
- Canonical prefix fixture: `e2e6218792db343e4e39c81ac47e5cd39825ba2302b0b339c28eec252e8330f2`.

Final qualification is complete in
`igc-learned-lifecycle-final-validation-01/report.json`. Regular, memcheck,
initcheck, synccheck and racecheck pass with zero reported errors/hazards.
Pipeline trial, ordinary pipeline, context and prefill-snapshot regressions
pass; ordinary pipeline output is byte-identical to the earlier three-record
reference. The rebuilt engine's reader/prefix/rubric/audit/CLI suite passes
24/24 without skips (`igc-live-harness-regressions-lifecycle-01.log`). All final
sources and binaries are checked again and preserved before reuse.

The final regular run took 51.594 seconds for the **whole content
proof plus component test**. It read the 92 GB model to verify source identity;
this is not invocation latency or a Hub benchmark. Engine content binding is at
startup, not per generated token. Sampled process private commit peaked at
260,890,624 bytes with a maximum observed sampling gap of 0.266 seconds; the
highest observed lifetime peak is 260,923,392 bytes. The earlier green run observed 301,629,440 bytes with a maximum
sampling gap of 0.204 seconds; these are separate runs, not a memory saving. This
is the controller test, not the full engine plus Vision workload. Process,
physical and system-commit fields are separate in the retained telemetry.

At 19:31 UTC, after the GPU job, physical availability is 26,182,852,608 bytes
and system commitment is 40,780,709,888 / 87,963,570,176 bytes. Available
commitment is 47,182,860,288 bytes: 1,686,847,488 below the prior matched full
engine+Vision sampled private peak. The disclosed task admission estimate,
including known storage and reserves, is 51,061,155,112 bytes, a shortfall of
3,878,294,824. The old peak is not an upper bound. No live model launch was
attempted; this is a resource-admission stop, not an engine failure. Read-only
ownership inspection finds no leftover task model, build or sanitizer process.

No Hub profile, defaults, process cap, Windows settings or unrelated process
changed. HIP/SYCL/non-Windows and deferred GPU fault injection are unrun.
Actual live learned generation, batched feature fidelity, language/API cancel,
snapshots, generalization, disabled raw-token parity and practical performance
remain open. This does not complete I2/I3/I4 or the sandbox roadmap.

## ภาษาไทย

ทดสอบ lifecycle ของ `IgcLearnedDiagnostic` และ head hook จริง โดยผูกกับเนื้อหา
Gyro/tokenizer ที่มีอยู่ แต่ไม่โหลด weights ของ language model และไม่สร้าง token
ใหม่ ใช้ prompt สังเคราะห์เดิมพร้อม prefix 66 tokens และ activations จริงที่เคย
บันทึกไว้ ณ anchor 65 ชุด fresh acceptance64 ยังไม่เปิดอ่าน

Context-only cancel ทำให้ test พบว่า hook ยังใช้ผลเก่าได้ เพราะอ่าน retained
pipeline โดยไม่ขึ้นกับ `active_` จึงเปลี่ยนเป็น `cancel(error)` ที่ปิด staging,
ล้าง feature history และสั่ง pipeline ให้เป็น unavailable ก่อน head ถัดไป
ต้องมี `begin()` ใหม่จึง stage ได้ อีก test พบว่า invocation ถูก reset แล้วแต่
head status ยังค้างว่าสำเร็จ จึงตั้งทั้ง 64 status เป็น invalid เมื่อ stage ใหม่
เพิ่ม asynchronous memset 256 ไบต์ต่อ stage ที่เปิดใช้ ยังไม่ได้วัดต้นทุน latency

หลัง release/reprepare พบว่า stage ได้ก่อน begin เพราะ flag ถูกเปิดใหม่ทั้งที่
ยังเก็บ invocation counter ไว้ จึงแก้ให้ release ปิด staging จนกว่าจะ begin ใหม่
โดยไม่ย้อน owner ID ทั้งหมดเป็นข้อค้นพบระดับ component ไม่ใช่ข้อสรุปว่าผู้ใช้ Hub
เคยเจอบัคนี้ New cancel API ยังไม่ได้เชื่อมกับการยกเลิกคำขอของ Hub

ผ่าน hook จริงได้ `31 * -24 = -744` และมี learned output mapping ที่ไม่เป็นศูนย์
ตรวจ cancel และ audit ก่อน head ถัดไป, cancel ซ้ำ, cancel ก่อน stage, การปฏิเสธ
cancel ระหว่าง capture แล้ว retry หลังจบ capture, cached prefix ที่ถูกต้อง,
history/prefix ที่ไม่ตรง, ineligible scope และ release/reprepare Graph ถูกทำลาย
ก่อนปล่อย storage ไม่สมมุติว่า module cancel คืน language KV/GDN หรือ RNG ให้
ถ้า revocation หรือ deferred CUDA completion ล้มต้องหยุด ไม่ใช่ resume ต่อ

Build CUDA 89;120 ผ่าน Hash กับ receipt อยู่ข้างต้น รอบสุดท้ายผ่าน regular,
memcheck, initcheck, synccheck และ racecheck โดยรายงานศูนย์ error/hazard พร้อม
pipeline trial/context/prefill-snapshot regressions, ordinary pipeline output
ที่ตรง reference ทุกไบต์ และ host/CLI 24/24 ไม่มี skip ตรวจและเก็บ source/binary
ก่อนใช้ build directory ซ้ำแล้ว รอบ regular ใช้ 51.594 วินาทีรวมการอ่านโมเดล
92 GB เพื่อตรวจ identity และทดสอบ component
จึงไม่ใช่เวลาคำนวณ invocation หรือ Hub benchmark โดย engine ตรวจเนื้อหาตอน startup
ไม่ได้ตรวจใหม่ทุก token วัด sampled private peak รอบสุดท้ายได้ 260,890,624 ไบต์
โดย sampling gap สูงสุด 0.266 วินาที และ lifetime peak ที่สังเกตได้คือ 260,923,392 ไบต์
ส่วน green รอบก่อนวัดได้ 301,629,440 ไบต์และ sampling gap สูงสุด 0.204 วินาที
เป็นคนละรอบ ไม่ได้อ้างว่าประหยัด memory และไม่ใช่ peak ของ engine+Vision

เวลา 19:31 UTC หลัง GPU job จบ มี physical memory ว่าง 26,182,852,608 ไบต์
System commit อยู่ที่ 40,780,709,888 จาก 87,963,570,176 ไบต์ เหลือ 47,182,860,288
ต่ำกว่า sampled private peak เดิมของ engine+Vision อยู่ 1,686,847,488 ไบต์ก่อน
เผื่อของใหม่ ส่วน task admission estimate รวม storage/reserve คือ 51,061,155,112
จึงขาด 3,878,294,824 ไบต์ Peak เดิมยังไม่ใช่เพดาน ไม่ได้ลองเปิดโมเดลเต็มและไม่ใช่
engine error ไม่พบ task model, build หรือ sanitizer ค้างจากการตรวจแบบ read-only

ไม่ได้เปลี่ยน Hub profile/default, ใส่ process cap, แก้ Windows settings หรือหยุดแอปอื่น
ยังไม่ได้ตรวจ HIP/SYCL, ระบบอื่น หรือ deferred GPU fault injection ส่วน live learned
generation, batched fidelity, language/API cancel, snapshots, generalization,
disabled parity และ performance ยังต้องทำ จึงยังไม่ครบ I2/I3/I4 หรือ roadmap sandbox
