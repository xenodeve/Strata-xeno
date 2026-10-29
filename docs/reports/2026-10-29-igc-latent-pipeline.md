# Native learned-controller pipeline on recorded Gyro activations

Issue #276. This is a component connection checkpoint in I2/I3, not completion
of either phase or a live language-model benchmark. The primary path is still
IGC-inspired learned mapping plus exact native execution; compiled Qwenputer is
the secondary comparator. No serving default or profile changed.

## What is connected

`LatentPipeline` loads the bound four-network controller bundle once. At request
entry it reads the two device-resident pre-anchor activation windows, runs both
input readouts, and arms a new invocation. At the head hook it uses the actual
final mixed state for readiness, requires agreement on the typed operands, runs
the exact CUDA executor once, retains the result, and applies the learned output
mapper. There are no operand tokens, prompt-to-answer lookup, host calculator or
per-token host decision in this path. The input-readout window is 80 rows; the
head accepts 1–64 rows on one created CUDA stream.

An invocation now retains its first attempted request and selection status in
addition to ownership, counters and result. Dispatcher scratch changes on later
graph launches, so it is not an execution audit record. The retained record is
288 bytes. Failed selection, missing context and invalid masks cannot publish a
numeric result; staging a new request clears prior eligibility. This record is
not yet a cross-request session ledger or a serialized snapshot.

The combined pipeline requests 12,801,500 device bytes, excluding the activation
history, source tensors and CUDA context. The separately qualified history cache
requests 13,107,280 bytes. These are allocation sizes, not measured full-engine
resident/transient peaks or an estimate of a new Hub memory limit.

## Frozen diagnostic and results

The 64 cases are the old, already inspected I2 pilot/development diagnostic.
They are **not** the sealed fresh acceptance64; that acceptance set remains
unread. Each input contains a coherent pair of real saved Gyro embedding and
first-attention windows, its mask, and the final mixed state from the same
prefix. The native input file contains no labels, operands or answers. Offline
references are read only by the validator after native execution.

| Check | Observed result |
|---|---|
| Calculation requests in the old diagnostic | 48 |
| Selected requests with correct operand tuple and exact result | 33/48 |
| Calculation requests bypassed | 15/48: 11 no-call, 4 disagreement |
| Negative requests incorrectly selected | 0/16 |
| Native versus offline selection, typed ABI and result records | 64/64 match |
| Normal launches versus captured graph; repeated launches | Bitwise equal records and modified mixed states |
| Successful result broadcast to 64 head rows | Bitwise equal rows, no additional execution |
| Missing context after a real result; empty mask mislabeled available | Valueless failure, unchanged mixed-state bytes |
| Offline output-mapper reference | Maximum absolute difference 3.814697265625e-6; rtol 1e-4, atol 1e-3 |

Representative native results are `31 * -24 → -744`, `20 - -15 → 35`, and
`-28 + -49 → -77`. They demonstrate the saved-activation controller/executor
connection, not a newly generated Gyro answer or final-answer faithfulness.

## Verification and identities

Test-first checks failed with `pipeline_unimplemented` and
`retained_attempted_request`, respectively, before the implementation and retained
audit fields were added. Final regular-path checks include hard-off bypass,
normal/graph repetition, restaging and invalid-mask controls. The final offline
validator and retained-invocation regression pass (`igc-pipeline-final-validation-01.log`).

| CUDA Compute Sanitizer 13.3 tool | Coverage and result |
|---|---|
| memcheck | Full old diagnostic64 plus lifecycle controls; zero errors; output byte-identical to the final regular run |
| initcheck | Same full64 coverage; zero errors; byte-identical |
| synccheck | Same full64 coverage; zero errors; byte-identical |
| racecheck | Three representatives covering selection statuses 0, 1 and 4, plus graph1/64 and missing/empty-mask controls; zero errors/warnings; byte-identical to corresponding final-run records |

The first racecheck attempt over all64 timed out at240 seconds. Its partial
output and log are retained; it is not a passed full64 racecheck. The bounded
three-record replay completed within180 seconds using the same executable and
kernel shapes. Racecheck's shared-memory hazard coverage does not establish
absence of every possible global-memory race. No performance claim uses these
instrumented runs.

| Frozen input | SHA256 |
|---|---|
| Controller manifest | `3e853f3190cfb710b3da6f4513f5f981c6af272c533a410fbeb63406390a4fa5` |
| Coherent diagnostic input, 420,091,980 bytes | `669becea5404dd4aff88eac2efeadab872b8fe30ae5ddfb9ee912927534721f7` |
| Final native probe | `b2b57ec6773d79f3f9f8c93b7a4134d3ebb3a90011812f9dc7c60df72cf27124` |
| Gyro model | `7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f` |
| Tokenizer | `889964f986b93c35102f25f5e396efeed65fcb180429c35d97536fb7f6b72106` |

Private evidence basenames: `igc-pipeline-final-native-01.log`,
`igc-pipeline-replay-01/{reference,validation}.json`,
`igc-pipeline-sanitizers-01/`, and `igc-pipeline-race-subset-01/`. Machine-private
absolute paths are deliberately omitted. All code commits in this worktree are
local until explicitly published; a hash is an identity, not a public source link.

## Remaining work

Attach the pipeline to the actual content-bound Gyro engine, with accepted-input
activation ownership and valid cached-prefix lineage. Qualify actual batched and
wave prefill rather than treating serial replay as equivalent. The first-anchor
prototype does not yet implement repeated calls during reasoning, transactional
shadow/commit/rollback, snapshots or all capability milestones. Learned input
generalization, same-session feature-off parity, live result use, real Hub
end-to-end performance and practical resource fit remain open. HIP, SYCL and
non-Windows builds were not run for these CUDA diagnostic sources.

## ภาษาไทย

ขั้นนี้เชื่อม input readout สองตัว, readiness, ตัวเลือก operands, executor บน
CUDA และ output mapper เข้าด้วยกันแล้ว ข้อมูลเข้ามาจาก activations ของ Gyro ที่
บันทึกไว้ ไม่มี operands หรือคำตอบแฝงอยู่ในไฟล์ที่ส่งให้ native probe

ชุดทดสอบเป็น diagnostic เดิม 64 ข้อที่เคยตรวจแล้ว ไม่ใช่ acceptance64 ที่เก็บไว้
ใน 48 ข้อที่ต้องคำนวณ ระบบเลือกได้ 33 ข้อ ทุกข้อที่เลือกมี operands และผลคำนวณ
ถูกต้อง อีก 15 ข้อไม่ได้เรียกคำนวณ ส่วน 16 ข้อที่ไม่ควรเรียก ไม่มี false call
การรันปกติ การรัน graph และการรันซ้ำให้ผลตรงกันทุก byte เมื่อล้าง context หรือ
ส่ง mask ว่าง ก็ไม่มีคำตอบเก่าหลงเหลือมาใช้

ตัวอย่างผลบน GPU คือ `31 * -24 = -744`, `20 - -15 = 35` และ
`-28 + -49 = -77` ผลเหล่านี้ยืนยันการเชื่อม pipeline บน activations ที่บันทึกไว้
ยังไม่ใช่คำตอบที่ Gyro สร้างสดผ่าน engine เต็มระบบ ขั้นถัดไปคือเชื่อมเข้ากับ
engine ตรวจการถือครอง context และทดสอบ result use จริง ก่อน benchmark ตาม
โปรไฟล์ Hub ที่ใช้อยู่

memcheck, initcheck และ synccheck ผ่านครบ 64 ข้อ ส่วน racecheck ชุดเต็มชน
timeout 240 วินาที จึงเก็บเป็นผลที่ยังไม่ผ่าน จากนั้นทดสอบตัวแทนสามสถานะรวม
graph 1/64 แถวและกรณีล้าง context แล้วผ่านโดยไม่พบ hazard ผลตรงกับรันปกติทุก
byte ไม่มีการใช้เวลาจาก sanitizer มาอ้างความเร็วของระบบ
