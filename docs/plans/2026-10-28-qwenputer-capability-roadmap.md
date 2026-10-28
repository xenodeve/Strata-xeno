# Qwenputer: จาก Native foundation สู่ความสามารถที่ใช้จริง

เอกสารนี้ต่อจาก [PRD #276](https://github.com/xenodeve/Strata-xeno/issues/276)
และแผน P0–P8 เดิม การทำวงจร calc ให้รันได้เป็น foundation ของงาน ยังไม่ใช่
หลักฐานว่าโมเดลได้ประโยชน์ครบทุกด้าน เป้าหมายปลายทางคือปรับความสามารถที่
upstream มีให้ใช้กับ Strata-xeno ได้ แล้ววัดและ optimize จนใช้ในงานประจำได้
โดยรักษาความถูกต้อง เริ่ม Native Flash-Next ก่อน แล้วจึงขยายไป Universal

ทุกเป้าหมายในเอกสารนี้มีผลร่วมกัน รวมทั้ง external-tool round trips และข้อมูลที่ไม่จำเป็นต้องอ่านใหม่
ให้เปลี่ยนการประเมินผลจากการทำนายเป็นการ execute แล้วคืนผลที่พอใช้ตัดสินใจ
รอบถัดไป วัดจาก task ที่แก้สำเร็จและ held-out tests ไม่ใช่จำนวน calls ที่ถูกซ่อน

## จะรับงานจากประโยชน์ใด

| Goal | Milestone | หลักฐานที่ต้องวัด |
|---|---|---|
| A sandbox ทดสอบสมมุติฐาน/logic/บัคภายใน | C3 | เลือก/เขียน experiments เอง, hypothesis discrimination และ bug fixes ผ่าน held-out tests |
| B execute ถูกต้องและคุณภาพ math/code ดีขึ้น | C0/C1/C3/C4 | independent exactness, formulation/execution/interpretation แยกกัน, task correctness |
| C reasoning tokens ลดลงและงานต่อ token เพิ่ม | C1/C3 | reasoning/total tokens และ correct completed tasks ต่อ token |
| D feedback prefill/context ลดลง | C1/C3 | fresh-feedback tokens, retained-context growth, replay bytes/tokens และ valid cache reuse |
| E external tools/ข้อมูลไม่จำเป็นลดลง | C1/C3 | calls ต่อ successful task และ minimal sufficient feedback ที่ยังมีหลักฐานพอ |
| F เวลาจนตอบถูกลดลง | C0–C5 | initial/incremental prefill, decode, circuit, bridge, retries และ total wall-clock รวมหลาย loop |
| G ปรับความสามารถที่ upstream มี | C0–C5/R | capability ledger พร้อม maturity/source และ acceptance ของแต่ละรายการ |
| H optimize/residency/offload จนใช้จริง | C0–C5 | cold/warm/tail, memory/cache, startup, cancellation, stability และ paired quality-preserving runs |
| I Native ก่อน Universal | C0–C5 | Native gates ผ่านก่อน model-adapter/compatibility gates ของ Universal |
| J default เมื่อดีขึ้นโดยไม่มี trade-off | G9 ทุก milestone | intended coverage ครบ, measured no-trade-off win, reversible default; ไม่ครบหรือมีต้นทุนเป็น opt-in |

- **ใช้การคำนวณภายในได้เอง:** วัดตั้งแต่เลือกใช้ ส่ง arguments รันสำเร็จ
  ไปจนถึงนำผลไปตอบ รวมการเรียกเกินจำเป็นและการรับมือโจทย์ที่ไม่รองรับ
- **เรียก tools ภายนอกลดลง:** นับต่อ task ที่แก้สำเร็จด้วยเกณฑ์เดียวกัน
  การซ่อนหรือระงับ tool call แล้วตอบผิดไม่ถือว่าดีขึ้น
- **คณิตศาสตร์หลายขั้นถูกต้องขึ้น:** แยกการตั้งสมการ การคำนวณ และการตีความผล
  ใช้ held-out tasks และคำตอบอิสระ ชุดบังคับเรียกวงจรพิสูจน์ได้เฉพาะกลไก
- **เขียนและ debug โปรแกรมผ่านวงจร:** เริ่มจากโปรแกรมที่ตรวจและอนุญาตไว้
  ให้สร้าง test รัน ตรวจ output/state แก้ แล้วรันใหม่ ผ่าน independent held-out
  tests ก่อนอ้างว่าช่วยแก้โค้ดได้ ไม่ได้หมายถึงแก้ learned weights เอง
- **จำลอง logic และทดสอบภายใน:** เริ่มจาก branches, loops, state transitions
  และ invariants ขนาดเล็ก ให้โมเดลเลือกโจทย์ เขียน logic/test cases ส่งให้
  compiled circuit รัน แล้วใช้ผลแก้และทดสอบซ้ำ วัดด้วย held-out tests แยก
  ไม่ถือว่าการจินตนาการ trace หรือบังคับรันโปรแกรมสำเร็จพิสูจน์ความสามารถนี้
- **เข้าถึงหน่วยความจำได้:** model-addressable store/filesystem เป็นอีก
  interface หนึ่ง ต่างจาก session snapshots ใน P6
- **งาน geometry และ 3D:** ตั้ง numeric contracts ของ vector/matrix,
  transform และ intersection ก่อน แล้วตรวจ task/invariant tests งานกลุ่มนี้
  ยังเป็นเป้าหมายทดลอง ไม่ใช่ความสามารถ upstream ที่พิสูจน์แล้ว

## Capability ledger

สถานะต่อไปนี้มาจากการตรวจ source ตาม revision ไม่ใช่ผลทดสอบบน Strata
upstream main ที่ตรึงไว้คือ `ca13310e2f9f788d1dd3feeea8213c1b0c23913c`

| ความสามารถ | ต้นแบบและระดับความพร้อม | งานที่ Strata ต้องเพิ่ม | เกณฑ์รับงาน / สถานะ |
|---|---|---|---|
| RV calc | main, integrated arithmetic bridge; [RV docs](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/docs/rv32.md) | sparse Native executor, ownership, replay, tokenizer binding, API/UI | P1–P8; reference/export อยู่ระหว่างทำ ยังไม่มี Native CUDA result |
| Inline arithmetic, LaTeX และ tool aliases | older CHIP8 integrated chat; [running](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/docs/running.md) | RV/native trigger และ role/history gates ที่ไม่ยิงผิด | ทดสอบ false triggers, quoted examples, ownership และ natural model use; ยังไม่ port |
| Store: put/append/get/list/delete | experimental CHIP8 chat บน main; RV implementation แยกอยู่ที่ [rv-store](https://github.com/divagr18/qwenputer/blob/01cf657d353bdc0c795c2d57a28a3e3f483f20a1/experimental/riscv/docs/rv-store.md) | command bridge, separate numeric ABI, exact operations, quotas, isolation และ persistence | property tests เทียบ independent map/filesystem; ใช้ข้าม session และป้องกันการปนกัน; ยังไม่ port |
| Store save/load | experimental store persistence; source เดียวกับแถวก่อน | model-visible store lifecycle และ identity แยกจาก P6 decoder snapshot | roundtrip, interrupted writes, restore validation, cross-session tests; ยังไม่ port |
| RV CPU/C/MicroPython | standalone demos ใน RV docs; merged-chat MicroPython bridge ยังไม่เสร็จ | allowlisted program loading, I/O และ bounded runtime ก่อน แล้วจึง coding/debug loop | output/state เทียบ reference และ held-out program tests; firmware ใหม่ยังไม่ได้อนุมัติ execute |
| Multi-conversation batching | upstream RV มีการ batch | scheduler, per-slot graphs/state, ownership/cancellation isolation | serial/batched equality, fairness, memory และ latency; v1 ยังปฏิเสธ mode ที่ไม่รองรับ |
| Oversight probes และ exact GDN log | [author-reported research prototype](https://divagr.com/blog/building-oversight-tools-into-a-models-weights); ไม่พบ monitor source ครบใน branches ที่ตรวจ | งานวิจัยแยก พร้อม provenance และ independent evaluation | ไม่อ้าง generic deception detection หรือใช้เป็น safety guarantee |
| Planted trackers/head interventions | research code ใต้ `research/hidden-reasoning` บน main | ต้องแยก threat model และ evaluation ออกจาก calc | research track ยังไม่เริ่ม |
| Games/CHIP8 video | upstream demonstrations | runtime/display/input integration ตาม workload | lower-priority parity demos; ยังไม่ port |
| sqrt/log/exp/trig และ geometry | advanced skills บางส่วนเป็นแผนใน upstream docs ไม่ใช่ released implementation | math programs/circuits และ error/precision contracts ใหม่ | independent numeric/task tests ก่อนอ้าง capability |

RV store branch `01cf657d353bdc0c795c2d57a28a3e3f483f20a1` แยกจาก main
(5 commits ahead, 20 behind ตาม source audit) ไม่ควร merge หรือ reuse ABI โดย
ไม่ตรวจ Store เคยต้องปัด gate/up/product เป็น BF16 เพื่อแก้ wrong token ที่เกิด
จาก FP32 จึงห้ามยืม arithmetic ABI ไปใช้โดยไม่พิสูจน์ ขอบเขตที่รายงานไว้คือ
53,248 content tokens, chunks ละ 104 tokens และ 512 ชื่อที่เคยสร้าง การลบ
ไม่ได้คืนโควตาชื่อ ตัวเลขเหล่านี้เป็นข้อจำกัดของต้นแบบ ไม่ใช่ quota ที่ Strata
เลือกแล้ว และยังไม่มีการ execute branch/firmware นี้

## Milestones หลัง foundation

| ช่วง | สิ่งที่ส่งมอบ | ต้องผ่านก่อน |
|---|---|---|
| C0 | Native arithmetic foundation P0–P8 พร้อม practical-use evaluation | exactness, ownership, history, API/UI, parity, resource/performance gates เดิมครบ |
| C1 | natural utilization และ inline/alias support ที่มีประโยชน์จริง | model selection/arguments/incorporation, false-trigger และ external-tools-per-successful-task tests |
| C2 | model-addressable store/filesystem และ persistence | source/numeric ABI review, approved artifacts, independent operation/quota/isolation tests |
| C3 | bounded allowlisted programs และ coding/debug loop | program bridge/controller, state inspection, revised-program loop และ held-out tests |
| C4 | geometry/3D และ advanced math ที่เลือกจากงานจริง | numeric contracts, independent references และ task invariants |
| C5 | batching และ Universal adapters | per-session isolation, scheduling และ model-specific exactness/eligibility gates |
| R | oversight/trackers/demos | research evidence แยก ไม่เป็นเงื่อนไขอ้าง calc สำเร็จ |

รายการนี้เป็นเป้าหมาย ไม่ใช่คำอ้างว่าทำเสร็จแล้ว การอนุมัติ execute compiler
และ calc firmware digest ปัจจุบันไม่ครอบคลุม firmware หรือ branch ใหม่

### C3: sandbox ภายในระหว่างคิด

agent ต้องใช้ sandbox ระหว่าง reasoning เพื่อทดลองสมมุติฐาน ทดสอบ logic
และไล่หาบัคได้ ให้ controller ภายในรับ experiment ที่ระบุ `program_id`,
program/test inputs, assertions, seed และ limits แล้วส่งให้ compiled native
circuit รัน ผลตอบกลับต้องแยก assertion pass/fail, output, terminal reason
และ state trace ที่จำกัดขนาดไว้ ก่อนให้ agent แก้สมมุติฐานและรันใหม่
ข้อความที่โมเดลเล่าว่าได้ทดลองแล้วไม่ใช่หลักฐานการ execute

เริ่มจาก reviewed allowlisted programs แล้วขยายเป็น model-generated logic
ภายใต้ interpreter/program module ที่ตรวจแล้วเมื่อ bridge รองรับ ไม่กำหนดว่า
ต้องใช้ภาษาใดก่อนเห็น instruction/firmware support จริง ห้ามเปลี่ยนไปใช้
host Python/Bash เพื่อให้ดูเหมือน sandbox ทำงานแล้ว

แยก experiment state ออกจาก persistent model memory และ language KV/GDN
กำหนด reset/rollback, deterministic seed เมื่อจำเป็น, step/time/output/memory
bounds, cancellation และการปฏิเสธ malformed program งานที่ล้มต้องไม่ทำ
conversation เสียหรือกลายเป็น verified answer ทดสอบด้วย independent oracles,
held-out bugs/hypotheses, spontaneous use, อัตราแก้บัคสำเร็จและ total latency
ต้องเพิ่ม internal controller/program-loading/inspection bridge ก่อนรับ C3
snapshot ของ P6 หรือ calc ที่ถูกบังคับเรียกใช้แทน milestone นี้ไม่ได้

workflow หลักคืออ่าน prompt/source ครั้งแรก หรือ reuse canonical cached prefix
ที่ยังตรงกับ source แล้ววน hypothesis → execute → compact feedback → revise
คืนเฉพาะค่า, expected/actual, failed assertion, minimal counterexample หรือ state
location ที่เกี่ยวข้อง โดยขอ bounded detail เพิ่มได้ แยก immutable inputs ที่
reuse ได้ออกจาก experiment state ที่ต้อง reset/rollback เมื่อ source edit
เปลี่ยน prefix ต้อง update/invalidate cache ตาม identity ให้ถูก

external-tool baseline ต้องใช้ normal valid prefix caching ด้วย แยก initial
prefill, incremental feedback prefill, reasoning decode, sandbox execution,
bridge และ external orchestration วัดจำนวน loop และเวลาจนแก้บัคผ่าน held-out
tests โดยเฉพาะงานหลายรอบหรือ logic ซับซ้อน พร้อม fresh-feedback-prefill tokens
และ retained-context growth จึงจะเห็นต้นทุนสะสมที่ลดได้จริง

สมมุติฐานเรื่องประหยัด reasoning tokens/context ต้องวัดแยกจากความเร็ววงจร
เทียบ task/model/settings เดียวกันในสามทาง: textual reasoning, Native sandbox
และ external-tool route ที่เกี่ยวข้อง รายงาน correct completed tasks ต่อ
reasoning token และต่อ total model token, จำนวน tokens ที่สร้าง, context ที่
เพิ่มขึ้น และขนาดผลที่ replay กลับ วัด wall-clock จนตอบถูกโดยรวม circuit,
transfer, bridge และ retries ด้วย พร้อมความถูกต้องของ hypothesis tests และ
bug fixes ที่ตรวจอิสระ ไม่สรุปว่า tokens ลดแล้ว latency จะลดตามเสมอ

คืนเฉพาะผล/assertions/state evidence ที่มีขนาดจำกัด ไม่ส่ง trace ทั้งหมดเข้า
language context ให้หยุดสถานะภาษาระหว่าง internal steps แต่ยังต้องรายงาน
จำนวน circuit trace steps และ compute time แยกไว้ การประหยัด context ต้อง
ไม่ซ่อนงานหรือบังคับตัด reasoning ที่จำเป็นต่อการเขียน test และตีความผล

### C4: ตรวจและแก้ฉาก Three.js

ใช้ฉากที่โมเดลสร้างเป็น task จริง: ตรวจ overlap/collision และ clearance
ของ object pairs, bounds หลัง world transforms และตัวอย่างสถานะขณะเคลื่อนที่
แล้วปรับตำแหน่งหรือ geometry และตรวจซ้ำ ต้องแยก broad-phase AABB/OBB
ออกจาก mesh intersection เพื่อไม่ย้าย object ผิดเพราะ bounding boxes ทับกัน
ทั้งที่ mesh ไม่ชน กำหนด coordinate space, tolerances, contact/clearance rules
และ independent geometry oracle ก่อน วัด scene violations ที่ซ่อมได้, false
repairs และ total task time

อ้างอิง semantics ของ [Three.js Box3](https://threejs.org/docs/pages/Box3.html)
และ [OBB](https://threejs.org/docs/pages/OBB.html) ที่ตรวจแล้ว: Box3 ใช้ AABB,
OBB ใช้กรอบที่หมุนได้ และ bounds ต้องสอดคล้องกับ world transform เอกสารนี้
ใช้เป็นต้นแบบของ data/kernel contracts ไม่ได้ระบุว่า Three.js รันอยู่ใน
RV calc firmware ปัจจุบัน

## Optimize จนใช้ได้จริง

ใช้ upstream เป็นต้นแบบ semantics และ correctness แล้วปรับ kernel, layout,
scheduling, bridge หรือ residency เมื่อการวัดชี้ว่าช่วยจริง ไม่ต้องรักษาวิธีที่
ช้าของต้นแบบ และไม่สรุปว่าต้นแบบยังไม่ optimize โดยไม่มีหลักฐาน

ทุก loop มี hypothesis, measurement, exactness/quality tests และผลก่อน–หลัง
ทดสอบกับ hub/Gyro-M 262K/5 workers เดิม ทั้ง end-to-end task latency,
time-to-first-result, trace/entry/replay, ordinary-text overhead, cold/warm/tail,
startup, RAM/VRAM/transient peaks, expert hits, cancellation/UI responsiveness
และ repeated-session stability ใช้ profiler อธิบายกลไก แล้วใช้ profiler-off
paired runs กับ counters จาก runs เดียวกันยืนยัน ห้ามลด workload หรือคุณภาพ
เพื่อให้ตัวเลขเร็วขึ้น Reference/oracle/checksum อยู่นอก timed intervals

**Residency เป็นงานทดลองที่ต้องทำต่อ:** แยก immutable weights, mutable
GDN/lag/bridge state, scratch, committed/shadow state และ cold snapshots
เทียบ lean all-resident, whole-module on-demand และ per-layer paging/prefetch
ที่มีเหตุผล ทุก trace step ของวงจรปัจจุบันผ่านหลาย layer จึงยังอ้างไม่ได้ว่า
บาง layer ไม่จำเป็น Existing expert offload ไม่ได้โหลดวงจรชนิดอื่นให้อัตโนมัติ
ถ้าผลทดลองดี ให้ทำ adapter/cache/graph-lifecycle เพิ่ม พร้อมทดสอบ exactness,
address stability และผลต่อ expert cache โดยไม่ย้าย active state เงียบ ๆ

static reserve ใน v1 เป็น correctness baseline ไม่ใช่ข้อห้ามต่อการ offload
วงจร Math ต้องวัด first-call, warm repeat, tail latency และ memory peaks
ก่อนเลือกรูปแบบใช้งานจริง

## Rollout ตามคำสั่งล่าสุด

สมมุติฐานอีกข้อคือ quantized model ที่ใช้ Native execution อาจแก้งานที่ต้อง
คำนวณหรือตรวจโปรแกรมได้ดีกว่า higher-precision model ที่ไม่มี execution
ให้ ablate โมเดล quantized ชุดเดิมแบบเปิด/ปิดก่อน แล้วเทียบ higher precision
เฉพาะเมื่อมี model lineage เดียวกันและทรัพยากรที่ใช้ได้จริง ตรึง prompts,
corpus และ success criteria แยกผลของ quantization ออกจาก capability ที่เพิ่ม
และรายงานความต่างของ hardware/latency แยกไว้ ไม่มีการอนุมัติดาวน์โหลดโมเดล
ขนาดใหญ่หรือใช้ทรัพยากรเสียเงินจากสมมุติฐานนี้

ถ้าวัดแล้วดีขึ้นและไม่พบ trade-off ใน intended usage ที่ทดสอบครบ ให้เปิดเป็น
ค่าเริ่มต้นใน hub ได้ โดยมีวิธีย้อนกลับและไม่ขัดจังหวะ session ที่ใช้อยู่
ถ้าใช้ memory เพิ่ม ช้าใน first call ลด compatibility หรือมีข้อเสียอื่น ให้
เป็น opt-in พร้อมตัวเลขประโยชน์และต้นทุน ถ้า coverage ยังไม่ครบหรือ downside
ยังไม่แน่ชัด ให้คง non-default ไว้ ยังไม่มีผลที่ผ่านเงื่อนไขเปิด default ในตอนนี้

---

# English mirror

## Goal traceability

| Goal | Milestone | Required evidence |
|---|---|---|
| A internal hypothesis/logic/debug sandbox | C3 | spontaneous experiment selection/encoding, held-out hypothesis discrimination and verified fixes |
| B exact execution and better verified math/code | C0/C1/C3/C4 | independent exactness, separate formulation/execution/interpretation and task correctness |
| C fewer reasoning tokens and more successful tasks per token | C1/C3 | reasoning/total tokens and correctly completed tasks per token |
| D less feedback prefill/context | C1/C3 | fresh-feedback tokens, retained-context growth, replay volume and valid prefix reuse |
| E fewer external tools/irrelevant feedback | C1/C3 | calls per successful task and minimal sufficient evidence |
| F lower latency to correct results | C0–C5 | initial/incremental prefill, decode, circuit, bridge, retries and full multi-loop wall time |
| G adapt upstream capabilities | C0–C5/R | capability/source/maturity ledger with per-capability acceptance |
| H practical optimization/residency/offload | C0–C5 | cold/warm/tail, memory/cache, startup, cancellation, stability and paired quality-preserving comparisons |
| I Native before Universal | C0–C5 | Native gates before Universal model-adapter/compatibility gates |
| J default only for a no-trade-off measured win | G9 at every milestone | adequate intended coverage and reversible default; trade-offs or incomplete evidence remain opt-in |

The existing #276/P0–P8 plan is the Native arithmetic foundation, not completion
of the entire capability goal. Adapt upstream capabilities for practical use in
Strata-xeno, Native Flash-Next first and Universal later. A working calculator
attachment does not prove model benefit.

All documented outcomes are joint goals, including external-tool round trips and unnecessary feedback rereading:
replace predicted experiment evaluation with actual internal execution and return
minimal sufficient feedback for the next hypothesis, assessed per successful task
and independent held-out tests, not hidden calls.

Acceptance must measure spontaneous selection, arguments, successful execution
and incorporation, unnecessary calls and unsupported tasks; external tool calls
per successfully solved task under equal success criteria; and multi-step math
with formulation, arithmetic and interpretation assessed separately. Forced-call
tests prove only the execution mechanism. Coding/debug means allowlisted compiled
programs with test generation, execution, output/state inspection, revision and
reruns, judged by independent held-out tests. It does not mean automatic editing
of learned weights or a proven general debugger. Model-addressable storage is
distinct from P6 snapshots. Geometry/vector/matrix/transform/intersection work
needs numeric contracts and task/invariant tests and remains a research goal.

An explicit programmable scenario is internal simulation/testing of small
deterministic branches, loops, state transitions and invariants: the model
selects and encodes logic/tests, the native compiled circuit executes, and the
model inspects results, revises and reruns. Judge selection, test encoding and
bug fixes by independent held-out tests, not imagined execution or only a forced
program run. Execution remains bounded/module-owned with no arbitrary host,
network or OS filesystem access.

The audited upstream main is ca13310e2f9f788d1dd3feeea8213c1b0c23913c.
The capability/source ledger above distinguishes integrated RV arithmetic;
older integrated CHIP8 inline arithmetic/LaTeX/aliases; experimental CHIP8 and
separate-branch RV filesystem/store operations and persistence; standalone
RV/C/MicroPython demos with unfinished merged-chat MicroPython bridge; upstream
RV batching; author-reported oversight research without complete monitor source;
planted tracker/head-intervention research; lower-priority games/video demos;
and advanced math skills that are plans rather than released implementations.
These source links establish upstream maturity, not Strata test results.

The RV store branch 01cf657d353bdc0c795c2d57a28a3e3f483f20a1 is divergent
(5 ahead/20 behind in the audit). Its BF16 gate/up/product rounding fixed an
observed FP32 wrong token, so arithmetic's numerical ABI cannot be reused without
proof. Reported limits are 53,248 content tokens, 104-token chunks and 512
ever-created names; deletion does not recover name capacity. These are upstream
limits, not chosen Strata quotas. No new branch or firmware has been executed.

Milestones are C0 (P0–P8 plus practical-use evidence), C1 (natural utilization and
safe inline/alias triggers), C2 (model-addressable filesystem/persistence), C3
(bounded programs and coding/debug loop), C4 (geometry/advanced math), C5
(batching/Universal), and separate R research tracks. Each capability requires
source/numeric ABI review, approved artifacts where applicable, independent
correctness and isolation tests, and practical resource/performance validation.
The current compiler/calc firmware execution approval does not cover new firmware.

### C3 internal reasoning sandbox

This is a core capability milestone: while reasoning, the agent constructs an
experiment with `program_id`, supported program/test inputs, assertions, seed and
limits; an internal controller executes it through the compiled native circuit;
the agent observes assertion results, output, terminal reason and a bounded state
trace, then revises and reruns. Narrated experimentation is not execution evidence.
Start with reviewed allowlisted programs, then model-generated bounded logic
under a reviewed interpreter/program module once its bridge exists. Do not promise
arbitrary languages or replace it with host Python/Bash. Internal controller,
program-loading and inspection integration are required; calc or P6 snapshots
alone do not meet C3.

Isolate experiment state from persistent model memory and language KV/GDN. Define
reset/rollback, reproducible seeds where relevant, step/time/output/memory limits,
cancellation, malformed-program rejection and independent oracles. A failed
experiment must not corrupt conversation or masquerade as a verified answer.
Evaluate held-out bug fixes and hypothesis discrimination, spontaneous use and
total task latency. No arbitrary host/network/OS filesystem execution is implied.

Token/context savings are evaluation hypotheses: compare the same tasks, model
and settings using textual reasoning, Native sandbox and relevant external-tool
routes. Report correctly completed tasks per reasoning token and per total model
token, generated/reasoning tokens, retained-context growth, replayed-result size,
and wall-clock latency to a correct answer including circuit, transfers, bridge
and retries. Independently verify hypothesis-test accuracy and bug-fix success.
Fewer language tokens do not establish lower latency. Return bounded useful
results/assertions/state evidence rather than the full machine trace; freeze
language state during internal steps while separately accounting for circuit
trace steps and compute time. Do not hide computation or suppress reasoning
needed to formulate tests/programs and interpret results.

The primary debug workflow ingests prompt/source once or reuses a valid canonical
cached prefix, then repeats hypothesis → internal execution → compact feedback →
revision. Feedback exposes concise values, expected/actual, failed assertions,
minimal counterexamples or relevant state locations, with bounded detail on
demand. Reuse immutable experiment inputs; reset/rollback experiment state
separately. Source edits update/invalidate prefix identity. The external-tool
baseline also uses its normal valid prefix caching. Separate initial prefill,
incremental feedback prefill, reasoning decode, sandbox execution, bridge and
external orchestration. Record loop count, fresh-feedback tokens, context growth
and total latency to an independently held-out-test-verified fix, especially
complex multi-iteration work, to assess cumulative overhead savings.

### C4 scene validation and repair

For model-created Three.js scenes, test pair overlap/collision, separation and
clearance, world-transformed bounds and moving-state samples; repair positions
or geometry and rerun. Separate AABB/OBB broad-phase tests from mesh intersection
so bounding-volume false positives do not cause incorrect repairs. Define world
coordinates, numerical tolerances, contact/clearance policy and an independent
geometry oracle; measure repaired scene violations, false repairs and task time.
The verified [Box3](https://threejs.org/docs/pages/Box3.html) and
[OBB](https://threejs.org/docs/pages/OBB.html) documentation provides bounding and
transform semantics to adapt into native data/kernel contracts, not evidence
that Three.js executes in the current RV calc firmware.

Optimize semantics-preserving kernels/layout/scheduling/bridges/residency from
measured bottlenecks. Upstream correctness is the reference, not a performance
ceiling or a requirement to preserve inefficient implementations. Do not assume
upstream is unoptimized without evidence. Each bounded loop has a hypothesis,
measurements and quality/exactness regression gates. Use the actual unchanged
hub/Gyro-M262K/5-worker workload for end-to-end task latency, first result,
trace/entry/replay, ordinary-text overhead, cold/warm/tail, startup, peak and
transient RAM/VRAM, expert hits, cancellation/UI responsiveness and stability.
Use profiler traces to explain and profiler-off paired counters from the same
runs to establish differences. No reduced workload or quality, and no oracle,
reference calculation or checksum verification inside timed intervals.

Investigate lean all-resident, whole-module on-demand and justified layer paging
or prefetch. Separate immutable weights, active mutable GDN/lag/bridge state,
scratch, committed/shadow state and cold snapshots. Current traces traverse
multiple layers every step; do not assume unused conditional layers. Existing
expert offload does not automatically support circuit tensors. Implement the
necessary adapters/cache/graph lifecycle if experiments justify them, validating
exactness, address stability and expert-cache effects. V1 static reservation is a
correctness baseline, not a prohibition on Math-circuit offload. Measure first
call, warm repeat, tail and memory peaks before choosing production residency.

Latest rollout policy permits a reversible hub default only after demonstrated
benefit with no observed trade-off over adequately covered intended usage:
correctness/quality, compatibility, ordinary and computation latency, cold/warm/
tail, memory/cache, startup, repeated sessions, cancellation and stability. Any
memory cost, slow first call, narrower compatibility or other trade-off stays
opt-in with measured costs/benefits. Incomplete coverage or uncertain downside
stays non-default. Do not interrupt active sessions casually. No current result
qualifies for default activation.

Evaluate whether quantized-plus-Native execution can outperform a higher-precision
same-lineage model without execution on computation/verification tasks. The
primary ablation is the same quantized model with/without Qwenputer; add higher
precision only when legitimately available and feasible. Hold prompts, corpus,
lineage and success criteria fixed, separate quantization from capability effects,
and report hardware/latency differences independently. This hypothesis grants no
large model download or paid provisioning authority.
