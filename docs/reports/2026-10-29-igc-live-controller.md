# First-anchor learned controller: implementation and qualification

Issue #276, I2/I3 follow-up. This opt-in diagnostic has component and admission
qualification. Live inference, quality, parity and performance gates remain open.
The sealed fresh acceptance set of 64 prompts remains unread. No Hub profile,
served binary or default changed.

## Engine connection

`--igc-learned-diagnostic-bundle PATH` is available only in the optional CUDA
diagnostic build. It cannot combine with the older controlled-input fixture.
The engine validates four networks and binds actual model/tokenizer contents
before model allocation. Effective embedding, dense, head and PLE sources must
belong to those native shards; a foreign override fails closed. Layer splits,
pipelined windows, multiple batch sessions and control vectors are refused.
D2x expert offload remains allowed.

`IgcLearnedDiagnostic` clears the old invocation before reading each new request.
It retains genuine early activations for accepted INPUT tokens, including only
the accepted speculative prefix. Both input readouts stage once before the first
answer window. Readiness, strict typed selection, one-shot exact execution and
late output mapping run on the verifier's created CUDA stream, including graph
launches. No host operands, call serialization or textual result prefill enter
this path. After the commit fence, audit reads one retained 288-byte invocation;
that diagnostic receipt is never model context.

Feature reuse requires a verified continuation of the same live prefix. Missing
or foreign history bypasses IGC without forcing cold language prefill. This
milestone allows one answer-start invocation per request; repeated reasoning
calls, transactional rollback, snapshots and UI/API remain separate gates.

## Batched producer and ownership

The automatic batched/wave prompt path stays enabled. Each initialized primary
wave lane owns two 80-row F32 snapshot buffers. Embedding broadcast and first
layer attention HC rows are copied before their source is overwritten and
emitted at the existing ordered chunk tail. Wave tail ordering and explicit
stream/event dependencies order the two lanes' cache updates. Final R and KV
are never substituted for early activations.

Publishing rows does not accept input ownership. Only a successfully completed
prompt part advances ownership. Review found that `run_wave()` bypassed the
single-lane acceptance wrapper. The wave branch now accepts its full completed
span exactly once after success and returns before acceptance on failure.
Independent source review confirms this wiring; the full-model wave branch is
not yet run. A component test refuses staged rows before ownership acceptance
and makes them available afterward.

| Requested device storage | Bytes |
|---|---:|
| Pipeline | 12,801,500 |
| Context window | 13,107,280 |
| Snapshots per initialized lane | 6,553,600 |
| Total with two lanes | 39,015,980 |

The four-network bundle, including manifest, is 4,947,912 serialized bytes.
Snapshots do not add a host expert stager. Requested device bytes are not
measured process private commit. Actual host transients, driver/graph costs,
peaks, expert-cache effects and serial/batched activation fidelity require a
full-engine measurement.

## Review and verification

The original rejection-only CLI tests missed a valid-route defect: admission
checked `native_shards` before preset expansion. A positive test reproduced it.
Review also found unbound model-source overrides. Admission now checks the
parsed preset first and resolved sources later.

The final reader, prefix, rubric, audit and CLI suite passes **24/24, no skips**
(`igc-live-harness-regressions-final-04.log`). Four admission tests include a
positive native route and four override subcases; they read actual model metadata
without loading weights. An earlier combined attempt omitted the required CUDA
DLL paths and returned empty native output. The final command uses the headless
launcher's CUDA environment; that setup failure is not an engine regression.

Snapshot tests cover counts 1, 80, 81 and 137, two-stream consumption, source
reuse, bounds, unaccepted ownership and failed/duplicate publication. Regular,
memcheck, initcheck and synccheck pass on the final build. The first final
racecheck ended without an outcome or remaining process and was never counted
as passed. A separate bounded repeat passes with zero errors/warnings/hazards.
`igc-prefill-snapshot-final-validation.json` consolidates the receipts. This is a
copy-only component, not qualification of full model kernels or all global races.

The full CUDA `89;120` build passes (`igc-live-engine-build-04.log`).
Candidate SHA256:
`1aae86e391a88f5a827c5b95ed728bb230860baab0c056b46f9c0fdf6b86dfdb`.
Snapshot-test SHA256:
`a035eab3b9611bf411c70c143e78ee9cdbef81bb59058b6a9f102977540e153d`.
Both binaries and source hashes are frozen before build-directory reuse.
HIP, SYCL and non-Windows builds are not run.

## Resource diagnosis and harness correction

The selected Hub checks 24 GiB physical availability for the 262K profile
(`strata-hub.bat:96,121-126`), sets `CUDA_VISIBLE_DEVICES=1,0`, clears
`CUDA_DEVICE_ORDER`, and has no private-commit cap. The diagnostic retains the
model, context, five workers, cache/offload and eager Vision settings. Only
task-owned profile/output copies, telemetry and bounded cleanup are added.

The first attempted live diagnostic was **not started**: task admission measured
50,191,040,512 commitment bytes available. The prior matched engine plus Vision
run had sampled summed private commit peaking at 48,869,707,776 bytes and working
set at 23,837,585,408 bytes. These are observations, not upper bounds for the new
workload. The harness retains disclosed emergency reserves of 2 GiB commitment
and 1 GiB physical memory, conservatively accounts for known component storage,
and applies time bounds. These are task admission/cleanup conditions, not Hub
policy or a process memory cap. Actual job flags are 8192 (kill on close), with
no process/job memory limit. Admission failure is not an engine failure.

At 18:32 UTC, `GetPerformanceInfo` reports 26,502,156,288 physical bytes available
and commitment of 38,789,734,400 / 87,963,570,176 bytes, leaving 49,173,835,776
bytes. Physical availability passes the Hub check. Commitment headroom exceeds
the old observed private peak by only 304,128,000 bytes before new storage and
reserve. Windows reports 35,052 MiB allocated pagefile, 941 MiB in use. Pagefile
usage differs from system commitment; free disk does not itself extend the
current limit. No task model/build/sanitizer leftovers were found. No unrelated
process, pagefile, security or Hub setting was changed. Protected processes make
the process-private inventory incomplete; its sum is not total system commit.

An import-only experiment found an avoidable harness cost: tensor-reader/NumPy
import increased private commit from 13,107,200 to 661,786,624 bytes but working
set only from 23,576,576 to 34,189,312 bytes. This is primarily commitment, not
650 MB of physical RAM. A standard-library prefix reader avoids that import;
shared `hub_contract.py` and `result_rubric.py` preserve older collectors'
interfaces. Fresh imports use 11,497,472 private bytes without NumPy. The live
harness measures prepared tokenizer/helper memory before admission. Inference
settings, numerics and safety reserves are unchanged.

## Remaining gates

Next is a bounded live learned-controller run when resources admit it, then
actual wave fidelity, cached continuation, cancellation and reset. Input
generalization, fresh acceptance, final-answer/result consistency, same-session
disabled raw-token parity, latency, peaks and practical Hub readiness remain
open. This is not completion of I2/I3 or the broader sandbox roadmap.

## ภาษาไทย

เชื่อม learned controller เข้ากับ engine แล้ว โดยยังเป็น diagnostic แบบ opt-in
ตรวจทั้งสี่ network และผูกกับเนื้อหา model/tokenizer จริงก่อนจัดสรรโมเดล ถ้า
override embedding, dense, head หรือ PLE ไปใช้ไฟล์นอก native shards จะปฏิเสธ
ส่วน D2x expert offload ยังใช้ได้ ไม่ได้เปลี่ยน profile, served binary หรือค่าเริ่มต้น

เมื่อเริ่มคำขอจะล้าง invocation เก่าก่อนอ่าน prompt เก็บ early activations เฉพาะ
INPUT ที่ยอมรับแล้ว รวม prefix ที่ commit จาก speculative window จากนั้น input
readout, readiness, การเลือก operands, exact executor และ output mapper ทำงาน
บน stream ของ verifier รวมถึง graph launches ไม่มี host operands หรือข้อความ
ผลคำนวณแทรกใน context Audit ขนาด 288 ไบต์อ่านหลัง commit fence และไม่ส่งเข้าโมเดล
ตอนนี้รองรับหนึ่ง invocation ที่ต้นคำตอบ ส่วนการเรียกซ้ำระหว่างคิด, rollback,
snapshot และ UI/API ยังไม่ครบ ถ้า history หายหรือเป็นคนละ prefix จะ bypass IGC
โดยไม่บังคับ cold language prefill

Batched/wave ยังทำงานตามเดิม แต่ละ lane เก็บสอง buffer ขนาด 80 แถวจาก embedding
และ attention ชั้นแรกก่อน source ถูกเขียนทับ ส่งต่อที่ chunk tail ตามลำดับ wave
ไม่ใช้ final R หรือ KV แทน early activations การส่งแถวกับการยอมรับ INPUT แยกกัน
Review พบว่า wave ข้าม wrapper ที่ยอมรับ INPUT จึงเพิ่มการยอมรับทั้งช่วงหลัง
run_wave สำเร็จเพียงครั้งเดียว ถ้าล้มจะไม่ยอมรับ ตรวจ source แล้ว แต่ยังไม่ได้รัน
branch นี้กับโมเดลจริง Component test ยืนยันว่าใช้แถวไม่ได้ก่อนยอมรับ ownership

Pipeline ขอ device memory 12,801,500 ไบต์, context 13,107,280 ไบต์ และ snapshot
6,553,600 ไบต์ต่อ lane ถ้าใช้สอง lane รวมเป็น 39,015,980 ไบต์ Bundle รวม manifest
มีขนาด 4,947,912 ไบต์ ไม่มี host expert stager เพิ่ม ตัวเลขนี้ไม่ใช่ private commit
ที่วัดจริง ยังต้องวัด host transients, driver/graph, peak และผลต่อ expert cache
รวมถึงตรวจ activation fidelity ของ batched path

Regression รอบสุดท้ายผ่าน 24/24 ไม่มี skip รวม positive admission และสี่กรณี
override นอก receipt แก้ guard ที่ตรวจ shards ก่อนขยาย preset แล้ว CLI อ่านเพียง
metadata ไม่โหลด weights รอบก่อนขาด CUDA DLL path จึงไม่มี native output
รอบสุดท้ายใช้ environment จาก headless launcher ปัญหา setup รอบนั้นไม่ใช่ engine
regression ส่วน snapshot ผ่าน 1, 80, 81 และ 137 แถว พร้อมสอง stream, source reuse,
bounds, ownership และ failed/duplicate publication Memcheck, initcheck, synccheck
และ racecheck รายงานศูนย์ error หรือ hazard ในขอบเขต component ที่ทำ copy เท่านั้น
Racecheck รอบแรกไม่มีผลจึงไม่นับว่าผ่าน ใช้ bounded repeat ที่จบสมบูรณ์แทน
Full CUDA build สำหรับ 89;120 ผ่านแล้ว Hash และ receipt อยู่ข้างต้น ยังไม่ได้
build HIP, SYCL หรือ Linux

ยังไม่ได้เริ่ม live learned inference เพราะ resource admission ของงานนี้ไม่ผ่าน
Hub ตรวจ RAM ว่าง 24 GiB และไม่มี private-commit cap ครั้งแรกมี commitment เหลือ
50,191,040,512 ไบต์ รัน engine + Vision เดิมเคยมี private commit รวมสูงสุด
48,869,707,776 ไบต์ และ working set 23,837,585,408 ไบต์ แต่ใช้เป็นเพดานรันใหม่ไม่ได้
Harness มี emergency reserve สำหรับ commitment 2 GiB และ physical memory 1 GiB
พร้อม timeout เป็นเงื่อนไขการทดสอบ ไม่ใช่นโยบาย Hub หรือหลักฐานว่า Gyro เปิดไม่ได้
Job มีเพียง kill-on-close ไม่มี process/job memory limit

เวลา 18:32 UTC มี RAM ว่าง 26,502,156,288 ไบต์ ผ่านเกณฑ์ Hub ส่วน system commit
อยู่ที่ 38,789,734,400 จาก 87,963,570,176 ไบต์ เหลือ 49,173,835,776 ไบต์ มากกว่า
peak เดิมเพียง 304,128,000 ไบต์ก่อนเผื่อของใหม่และ reserve Pagefile จัดสรรไว้
35,052 MiB ใช้อยู่ 941 MiB ค่า pagefile usage, system commit และดิสก์ว่างเป็นคนละค่า
ไม่พบโปรเซสโมเดล, build หรือ sanitizer ของงานนี้ค้าง ไม่ได้เปลี่ยน settings หรือ
หยุดแอปอื่น โปรเซสที่มีการป้องกันอ่านไม่ได้ทั้งหมด จึงใช้ผลรวม private commit
ของโปรเซสที่อ่านได้แทน system commit ไม่ได้

แก้ค่าใช้จ่ายของ harness ด้วย การ import tensor reader/NumPy ทำให้ private commit
เพิ่มจาก 13,107,200 เป็น 661,786,624 ไบต์ แต่ working set เพิ่มจาก 23,576,576 เป็น
34,189,312 ไบต์ จึงไม่ใช่ RAM เพิ่มราว 650 MB Prefix reader ใหม่ใช้ standard library
และรักษา interface ของ collector เดิม Fresh import ใช้ private commit 11,497,472
ไบต์โดยไม่โหลด NumPy Harness วัด memory หลังเตรียม tokenizer/helper ก่อน admission
ไม่ได้เปลี่ยน inference settings, numerics หรือ reserve

งานถัดไปคือ live run เมื่อทรัพยากรพร้อม แล้วตรวจ wave fidelity, cached continuation,
cancel และ reset ยังต้องทดสอบ generalization, fresh acceptance, ความตรงกันของคำตอบ
กับผล execute, disabled raw-token parity, latency และ peak บน Hub ชุด acceptance
ใหม่ 64 prompts ยังไม่เปิดอ่าน I2/I3 และ roadmap sandbox จึงยังไม่ถือว่าเสร็จ
