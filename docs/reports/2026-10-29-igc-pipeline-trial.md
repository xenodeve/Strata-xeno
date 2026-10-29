# Native pipeline trial checkpoint

Issue #276, G3/I4 prerequisite after `cb7bbd66`. This is a module-state component,
not completion of repeated model-selected calls or the live-engine transaction.
The fresh acceptance set remains sealed.

## Behavior

`LatentPipeline::begin_trial()` retains its bounded mutable buffers on the owning
created CUDA stream. They include invocation provenance, typed request/result,
readouts, selection, row ownership and head statuses. Neural weights are fixed;
mapper scratch is recomputed. `finish_trial(true)` keeps the candidate module
state; `finish_trial(false)` restores the base. Both run outside graph capture.
Nested trials, a foreign stream and staging a new request during a trial are
refused before state mutation. Hard-off calls do nothing.

The checkpoint requests **7,816 device bytes**, allocated lazily on the first
trial and retained for reuse. The first-anchor engine route does not start a
trial and allocates none of this storage. Allocation and D2D checkpoint/restore
costs are unmeasured trade-offs, not free operations or speed claims.

Completion is asynchronous. The owner must fence and handle deferred CUDA
errors before publishing results; it must not resume a failed restore. Immediate
copy failures quarantine the component. Deferred GPU fault injection is not
tested. Independent source review confirms the covered state and guards, while
retaining that failure/fence obligation.

The caller must also discard rejected language output rows. Module rollback
does not restore mixed activations, KV/GDN, source context or language RNG.
Partial acceptance requires abort and replay of only the accepted INPUT prefix;
this component does not implement that orchestration or persistent snapshots.

## Evidence

The initial test fails with the deliberate unimplemented checkpoint stub, then
passes after implementation. A second test exposes request staging during an
active trial and passes after the guard. A negative control that omits the
rollback copy fails `abort_restores_pending_state_and_head_status`; the source is
restored byte-for-byte before the final build.

Three previously inspected real-activation fixtures cover disagreement, exact
execution and no-call selection. The exact case is `31 * -24 = -744` (literal
test expectation). Tests check pending-to-applied abort, accepted replay, commit
and repeated graph launches, unavailable-context reset, a new owner, nesting,
foreign streams, capture refusal and stable checkpoint storage. Production
execution uses the native neural pipeline/exact device executor, not host math.
These are saved activations, not a new live Gyro-M run or quality evaluation.

Full CUDA `89;120` build passes. Candidate executable SHA256:
`8e5a75ce7bdf3551cd0d03072eb8fb732433c8ac77cdffaec615df8f0d3434b9`.
The final test binary and source identities are in
`igc-pipeline-trial-final-build-02.identity.json`.

Regular, memcheck, initcheck and synccheck pass the 64-row graph/lifecycle cases.
The first full trial racecheck exceeds 60 seconds; that attempt is not a pass.
Copy-only, ordinary nontrial graph and one-row trial controls pass; the one-row
instrumented test takes 43.781 seconds. A final full 64-row trial run then completes
within a separately disclosed 150-second bound with zero hazards/errors/warnings.
This completes the final binary's full-row instrumentation; the earlier timeout
does not by itself establish a code race or its cause. Sanitizer wall times are
not uninstrumented performance measurements.

`igc-pipeline-trial-final-validation.json` consolidates the separate receipts.
The ordinary pipeline output remains byte-identical on the three records;
context and prefill-snapshot regressions pass. The rebuilt engine also passes
the 24-test host/CLI suite without skips. Final trial-test SHA256:
`c1c99bc079d972e784fb8db8b68d2291a0f512743f57b3171de3eb6e7be23b47`.

The prior failed setup of build02 never executed a build; build02b is the actual
successful command. HIP/SYCL/non-Windows and live-model parity/performance are
not run. Resource admission for live testing remains independent of this work;
there is no private-commit cap, Hub change or serving default activation.

## ภาษาไทย

เพิ่ม trial checkpoint ให้ native pipeline เพื่อเก็บสถานะก่อนทดลอง แล้วเลือกเก็บ
หรือทิ้ง candidate state ได้ เก็บทั้ง invocation, request/result, readouts,
selection, row ownership และ head status ส่วน weights คงเดิมและคำนวณ scratch ใหม่
ปฏิเสธ trial ซ้อน, stream ที่ไม่ใช่เจ้าของ, การเปิดคำขอใหม่กลาง trial และการ finish
ใน graph capture โดยไม่แก้สถานะเดิม Hard-off ไม่ทำงานหรือจัดสรรเพิ่ม

Checkpoint ขอ device memory 7,816 ไบต์เมื่อใช้ trial ครั้งแรกแล้วใช้ซ้ำ เส้นทาง
first-anchor ใน engine ยังไม่เรียก trial จึงไม่จัดสรรส่วนนี้ ยังไม่ได้วัดต้นทุน
allocation หรือ D2D copy และไม่ได้อ้างว่าทำให้เร็วขึ้น

การ finish เป็น asynchronous เจ้าของต้องรอ fence และจัดการ deferred CUDA error
ก่อนเผยแพร่ผล ถ้า restore ล้มต้องไม่ resume โดยทดสอบ fault injection ส่วนนี้ยัง
ไม่ได้ทำ ส่วน module rollback ไม่คืน language activations, KV/GDN, source context
หรือ RNG ต้องทิ้ง rejected language rows แยก ถ้ารับ INPUT เพียงบางส่วนต้อง abort
แล้ว replay เฉพาะ prefix ที่รับไว้ ยังไม่ใช่ persistent snapshot หรือ I4 ทั้งระบบ

ทดสอบ red/green ทั้ง checkpoint ที่ยังไม่ implement และการเปิดคำขอใหม่กลาง trial
Negative control ที่ตัดการคืนสถานะออกล้มตามที่คาด แล้วคืน source ก่อน build สุดท้าย
ใช้ real-activation fixtures เดิมสามกรณี ไม่ได้เปิด fresh acceptance ชุดใหม่
กรณี exact executor ได้ `31 * -24 = -744` ตรวจ abort/replay/commit, graph ซ้ำ,
reset, owner ใหม่, nesting, stream, capture และขนาด storage คงที่ ผ่านใน native
pipeline แต่ยังไม่ใช่ live Gyro-M inference หรือผลประเมินคุณภาพ

Full CUDA build 89;120 ผ่าน Regular, memcheck, initcheck และ synccheck ผ่านกรณี
64 แถว Full-trial racecheck รอบแรกเกิน 60 วินาทีจึงไม่นับว่าผ่าน จากนั้น copy-only,
ordinary nontrial graph และ trial หนึ่งแถวผ่าน โดยแบบหนึ่งแถวใช้ 43.781 วินาที
รอบสุดท้ายแบบเต็ม 64 แถวจบภายในขอบเขต 150 วินาทีและรายงานศูนย์ hazard/error/warning
จึงปิดการตรวจขอบเขตเต็มของ binary สุดท้ายแล้ว Timeout เดิมอย่างเดียวไม่ได้พิสูจน์
ว่ามี code race หรือบอกสาเหตุ และเวลาใน sanitizer ไม่ใช่ตัวเลข performance ปกติ

Receipt สรุปคือ igc-pipeline-trial-final-validation.json Output ของ ordinary
pipeline ยังตรงกันทุกไบต์ในสาม records ส่วน context/prefill-snapshot regressions
และ host/CLI 24 tests ผ่าน ไม่มี skip Hash ของ trial test สุดท้ายอยู่ข้างต้น
ยังไม่ได้ตรวจ HIP/SYCL, ระบบอื่น, live parity หรือ performance ไม่ได้เปลี่ยน Hub
หรือใส่ private-commit cap
