# Native Qwenputer P0 feasibility — 2026-10-28

P0 is the only authorized implementation phase. The compiled arithmetic circuit, model bridge and decoder integration are not implemented by this diagnostic. PRD: [#276](https://github.com/xenodeve/Strata-xeno/issues/276), approved R2; the English/Thai published body was read back exactly (62,037 characters, SHA256 `066617000239adb3708d2b7fc2d5e0220dbcf0feb6f57fb4c07dfe1ea0e3d59e`). No implementation issue breakdown, PR, push or merge was performed.

## ปิด private cap ตามที่อนุญาต — inference smoke ผ่านแล้ว

สาเหตุของ startup failure รอบก่อนคือ hard process cap ของ harness เมื่อผู้ใช้สั่งให้ปิด cap เฉพาะการทดสอบ workload เดิมบน main ก็โหลดและสร้าง output ได้ โดยไม่ต้องแก้ engine หรือเปลี่ยน Gyro-M, context 262K, workers 5 และ cache settings ไม่มีการเพิ่ม --ram-cache-gib หรือเปลี่ยน pagefile, security และ served defaults

อ่าน /t4-afk จาก C:/Users/xenod/.agents/skills/t4-afk/SKILL.md แล้ว ขอบเขต AFK รอบนี้คือแก้ startup diagnosis และเก็บหลักฐาน ส่วน P1 Native circuit ยังไม่เริ่ม

### กลไกและการแก้

OwnedJob เดิมตั้ง JOB_OBJECT_LIMIT_PROCESS_MEMORY (0x100) ไว้ที่ 41,999,998,976 bytes ทั้งก่อนและหลัง reboot พบ event 9 ของ Strata กับ process peak ชิด cap ขณะที่ system commit ยังมี headroom จึงระบุ guard rejection ได้โดยไม่ต้องเดาจาก free RAM

JobMemoryMonitor.set_process_limit รับ explicit None สำหรับข้อยกเว้นนี้ ล้างเฉพาะ process-memory flag/value ของ task-owned job แล้ว query ยืนยันว่า limit เป็น None โดยยังรักษา kill-on-close ไว้ Controller ไม่มีเงื่อนไขอื่นที่หยุดตาม private commit 42 GB ส่วน soft notification 40 GB มีไว้เก็บข้อมูล ไม่จำกัดหรือ kill process

ยังคง timeout 60 วินาที, system commit floor 2 GiB, physical floor 1 GiB และ process-tree cleanup ก่อน launch ต้องมี system headroom อย่างน้อย 55 GB + 4 GiB และ physical available 26 GiB ตัวเลข preflight นี้เป็น safety allowance ไม่ใช่ process cap

### ผล run05

| รายการ | ค่าที่วัดหรืออ่านกลับ |
|---|---|
| Tag | gyro-no-private-cap-05 |
| Main baseline exe SHA256 | 3948784248ce3193ddd636792ce6f377fdb5e727eea5b241dcb279b8b8e46042 |
| Job flags ก่อนและหลัง | 8192 (0x2000, KILL_ON_JOB_CLOSE) |
| Process / job hard commit limit | null / null ทั้งก่อนและหลัง |
| Exit / controller stop | 0 / null |
| Process lifetime private peak ตลอด run | 46,740,692,992 bytes (46.7407 GB) |
| Sampled system commitment peak | 63,456,198,656 bytes |
| Sampled system commitment headroom ต่ำสุด | 24,507,371,520 bytes |
| Sampled physical available ต่ำสุด | 12,529,254,400 bytes |
| Hard limit events 9/10 | ไม่มี |
| Output | 16 tokens, first ID 15968 |
| Secondary compute | 1773 entries / 1449 groups, stage-only=0 |
| MTP | ยอมรับ 7 จาก 14 drafts |
| Cleanup | owned_job_drain_verified=true; ไม่พบ task-owned model process ค้าง |

หลัง host experts โหลดครบ loader รายงาน private commit 26.08 → 42.29 GiB ที่เวลา arrival 16.187 วินาที Sample ใกล้กันที่ 16.172 วินาทีมี job peak 45,493,002,240 bytes ค่านี้เป็น checkpoint ช่วงโหลด ส่วน 46.7407 GB เป็น peak รวม startup, prefill และ decode ไม่เรียกสองช่วงนี้แทนกัน

ถอด raw IDs ด้วย project reference tools.strata_tokenizer.Tokenizer จาก GGUF metadata ได้ข้อความก่อน EOS ว่า “Testing software ensures reliability and prevents bugs before deployment.” ซึ่งตรงกับ prompt ที่ขอหนึ่งประโยคเกี่ยวกับ testing software Raw IDs กับข้อความอยู่ใน inference-evidence.json

CLI เขียน special tokens ต่อจนถึง max-new 16 จึงไม่ใช้ผลนี้อ้างว่า served response จะส่ง special tokens เหล่านั้น ไม่มี per-token arrival timestamp โดยตรง และไม่ใช้เวลาใน diagnostic run เดียวอ้าง performance win หรือ arithmetic accuracy

ก่อน run มี system headroom 71,357,169,664 bytes และ physical available 36,422,467,584 bytes ไม่มี stale task process Exe, profile และ learned-profile hashes ตรงเดิม Input 21 ไฟล์ได้รับ full rehash หลัง reboot แล้ว และ metadata ยังตรงก่อนเริ่มรอบนี้ Run จบใน 39.078 วินาทีโดยไม่มี allocation error, telemetry error หรือ post-run error

### Side effects และการทวนผล

Served JSON กับ learned profile ต้นฉบับยังมี hash เดิม ไฟล์ model, pack weights, tokenizer และ MTP ไม่เปลี่ยน metadata แต่ engine สร้าง derived tail cache ใหม่ตามพฤติกรรมเดิมของ setup_tail_file: tail-232775360fcd05ab.bin มี 3172 slots ขนาด 4,144,615,424 bytes และลบ stale tail-0f52493d0054a5fe.bin ที่เคยอยู่ใน manifest

Source generate.cpp:1330–1375 ตั้ง cache key และลบ key เก่าอยู่แล้ว รอบนี้ไม่ได้แก้พฤติกรรมดังกล่าวหรือเรียก cleanup เอง จึงไม่อ้างว่า pack directory ทั้งต้นไม่เปลี่ยนหลัง run ข้อมูล weights/config ยังเดิม ส่วน tail cache เป็น artifact ที่ engine สร้างคืนได้

Test ใหม่เห็น red ก่อนแก้ explicit None แล้วผ่านหลังแก้ โดยตรวจ disabled → 64 MiB → disabled และยืนยัน kill-on-close คงอยู่ Memory tests ผ่าน 14 ข้อ ชุดรวมผ่าน 21 จาก 23 tests อีก 2 GPU tests ถูก skip ใช้ 3.380 วินาทีและ DONE exit=0 ผู้ตรวจอิสระตรวจ None path, test และ controller แล้วไม่พบ Critical หรือ Important finding

ตอนถอดข้อความได้ลอง Rust tokenizers.Tokenizer.from_file กับ pack tokenizer.json ก่อน แต่ไฟล์นั้นเป็น Strata descriptor จึงไม่ใช่รูปแบบที่ parser นี้รับ จากนั้นใช้ project Tokenizer.from_gguf ตาม source และถอด output สำเร็จ ไม่ได้แก้ model/tokenizer เพราะ error จากการเลือกเครื่องมือผิด

สำหรับภาพ Task Manager: Library preparation สำเร็จ แต่ Windows helper จบไม่ได้เพราะ os.setxattr ไม่พร้อมใช้งาน และไม่เหลือ PNG ที่อ่านได้ จึงยังไม่ได้ตรวจ pixels หรือยืนยันการเก็บ Library metadata ไม่ลอง transfer ซ้ำอีก ตัวเลข RAM 13.7/47.7 GB, available 34 GB, committed 15.4/81.9 GB อ้างเฉพาะ OCR ที่ parent ส่งมา ค่าระบบหลัง process จบไม่ได้แสดง job limit ของ process ตอนกำลังรัน

### ขอบเขตที่ยืนยัน

Startup และ component inference ของ baseline ผ่านภายใต้ข้อยกเว้นการทดสอบที่ผู้ใช้สั่งให้ปิด private cap เท่านั้น งบ 42 GB ของ repo ยังไม่เปลี่ยน และ peak ที่วัดได้สูงกว่างบ 4.7407 GB ต้องคงข้อค้นพบนี้ไว้ใน resource gate

ยังไม่มี candidate raw-token parity, ABBA, arithmetic circuit หรือ ownership integration ที่ผ่านจากรอบนี้ คำเตือน MMQ/expert split เป็นข้อจำกัดของเส้นทาง performance ที่แยกจาก startup ส่วน P1+ ยังไม่ implement

Raw run05 result.json SHA256: a2ae15b6eb44354a25fb547aef66981b35d08c3fe8490c273144c2c2155e4f3d
Controller SHA256: 08a9a0a8fdfd56e43ddb4379da04db85056ee4f77b400f7977b0de92a4773cfc
Telemetry SHA256: f0a2f092301ebfe78361070ecb4668b94a647e402f5760a34601e065b9fb4305

หลักฐานอยู่ใน gyro-no-private-cap-05/{result.json,on.stdout,on.stderr,stderr-events.json,inference-evidence.json,input-postcheck.json}, no-private-cap-preflight.json และ p0-no-cap-unit-suite.log

Public-safe PRD ในเครื่องเป็น R6 แล้วและเก็บ exact R5 payload เดิมไว้ Issue #276 ยังเป็น R4 และรอ Submit ของ R5 ไม่มีการ retry หรือ bypass และยังไม่อ้างว่า R6 sync สำเร็จ

## หลัง restart — กลไกการปฏิเสธยังเดิม

ผู้ใช้ยืนยัน restart แล้ว ตรวจ boot time ได้ 2026-10-28T20:47:44.891043Z ก่อนเริ่มไม่พบ task-owned model/controller ค้าง exe/profile/learned profile/prompt hashes ตรงเดิม และ rehash model/pack/tokenizer/MTP inputs21 ไฟล์ รวม 98,025,159,488bytes ตรง manifest เดิมทั้งหมด ใช้ 73.109s การอ่านนี้อุ่น filesystem cache จึงไม่ใช้เวลา startup รอบนี้อ้าง speed win/loss

รันเพียงหนึ่งครั้งด้วย main baseline เดิมและ corrected v3 controller: `gyro-memory-after-reboot-04` ก่อน launch system headroom71,977,607,168bytes และ available physical37,120,352,256bytes ผ่าน preflight เดิม ไม่มีการผ่อน guard หรือเงื่อนไขความปลอดภัย

| หลักฐาน | ก่อน restart run03 | หลัง restart run04 |
|---|---:|---:|
| Applied per-process guard bytes | 41,999,998,976 | 41,999,998,976 |
| Job process peak bytes | 41,999,466,496 | 41,999,060,992 |
| เหลือถึง guard bytes | 532,480 | 937,984 |
| Minimum sampled system commitment headroom bytes | 9,409,363,968 | 30,002,733,056 |
| Minimum sampled physical available bytes | 20,670,574,592 | 31,733,825,536 |
| Hard limit notification | 9, PID51164 | 9, PID33900 |
| Terminal outcome | error1455 ก่อน token แรก | error1455 ก่อน token แรก |

run04 รับบรรทัด error ที่ 13.078s และอ่าน event9 ที่ 13.109s ของ controller นี่เป็นเวลา arrival/dequeue ไม่ใช่เวลาของ CPU instruction ที่เรียก VirtualAlloc Sampling gaps สูงสุด 32ms Process private lifetime peak ที่อ่านและ job peak ตรงกัน; system limit จริงของรอบ 04 คือ 87,963,570,176bytes ไม่สมมติว่าเท่ารอบ 03 ซึ่งเป็น 90,968,305,664bytes ไม่มี telemetry/post-run error และ owned_job_drain_verified=true ไม่พบ task-owned model process หลังจบ

การ restart เพิ่ม system headroom แต่ไม่ได้เปลี่ยนกลไก: process guard ยังปฏิเสธ commit ก่อน token แรก ไม่ใช้ผลนี้อ้างว่า physical RAM หมด, main ใช้ Gyro ไม่ได้ หรือเปลี่ยน branch แล้วจะหาย ไม่ถอดหรือเพิ่ม guard และไม่รันซ้ำอีก

หลักฐานใหม่: `after-reboot-preflight.json`, `after-reboot-input-hashes.json`, `after-reboot-comparison.json` และ `gyro-memory-after-reboot-04/` ใน evidence directory เดิม SHA256 ของ run04 result.json คือ `4d6ce2a6a7a7b4ddc00e0d697ff64abf71b8bb88215ec998390609289ac2c41f` Controller SHA256 `4d261fac840933b9ddd64158975d727af605cb5ec3fee60245db256f20fef916`; telemetry SHA256 `a836bdff1fdccd7494f5c2ed9b65b17ef8e83d7f8391b442d92c38952c499e80`

### Host arena และทางเลือกที่ยังต้องตัดสินใจ

Pack มี 24,576 experts ×1,305,600bytes =32,086,425,600bytes ตรวจ profile ตาม source-defined STRP v1 uint16 pair region แล้วครบ 24,576 คู่และไม่ซ้ำ primary5,695 กับ secondary5,607 เป็นคนละคู่ ดังนั้น host-owned13,274 experts มี logical payload17,330,534,400bytes (16.1403GiB) ซึ่ง load_rest5214–5228 จะ commit ทั้งหมด เพราะ profile นี้ไม่ตั้ง --ram-cache-gib

sample ใกล้ก่อนข้อความ secondary staging สำเร็จมี process private28,044,435,456bytes บวก payloadhost ทั้งชุดได้ 45,374,969,856bytes นี่เป็น additive estimate ของ startup ไม่ใช่ peak ที่เปิดสำเร็จ ไม่รวม page rounding และ allocations หลังจากนั้น แต่สอดคล้องกับ guard failure ที่วัดจริง Source ไม่มี automatic42GBcap สำหรับโหมดนี้

--expert-cache/--vram-reserve-mib กับ secondary/displaybudgets กำหนด GPUslots; exclusive primary/secondary และ tail-file ช่วยไม่เก็บ host copy ซ้ำและเปิดอยู่แล้ว ลด GPUcache ไม่ได้รับประกันว่าจะลด totalcommit เพราะ expert ที่ GPU ไม่ถือจะกลับไป host และบน Windowsdeviceallocation ยังมี commitcharge (generate.cpp5997–6018) ไม่เปลี่ยน cache จากการเดาว่าจะประหยัด

--ram-cache-gib เป็นกลไกเดิมที่จำกัด logicalhostexpertcache และให้ส่วนเกินอยู่ NVMe (generate.cpp6576–6588; expert_source.cpp5248–5305) ยังคง model/context/workers และ GPU tiers ได้ แต่เปลี่ยน D2x จากไม่มี NVMe tier เป็น capacity mode มี I/O miss latency และต้องวัดใหม่ อีกทั้ง cache cap ไม่ใช่ hard process cap มี slabslack และ in-flight/heldexperts จึงต้องเหลืองบสำหรับ graph/requestpeaks ไม่เลือกค่าแทนผู้ใช้

ถ้าต้องรักษาทั้ง 42GBhardguard และ D2xresidency เดิม ยังไม่มี existingknob ที่พิสูจน์ว่าทำให้ fit ต้องแยกงานวัด allocationcategories และลด commit ที่ไม่จำเป็นโดยไม่เปลี่ยน semantics หรือให้ผู้ใช้เลือก capacity mode อย่างชัดเจน --mmap-experts ไม่ใช่ drop-in เพราะขัดกับ explicitexclusive-primary path (generate.cpp3569–3577) และ pinningcap ไม่ได้จำกัด pageable arena นี้ (5604–5608) ไม่มีการใช้ทางเลือกใดจริงในรอบนี้

### Final helper verification และ GitHub form

หลังเพิ่มสอง assertions สุดท้าย รันครบ 22tests อีกครั้ง:20 ผ่าน,2GPUskipped,3.398s,DONE0 (`p0-memory-suite-assertions-final.log`) ไม่มี code เปลี่ยนหลังจากนั้น

อ่าน issue#276 หลัง reboot แบบ read-only แล้วยังเป็น R4, updated_at2026-10-28T20:02:21Z,48291characters การเขียน R5 ผ่าน mcp__github__issue_write ยังเป็น pendinginteractiveSubmit ไม่ได้ retry หรือเปลี่ยนช่องทาง Returnedmetadata มีเพียง isError=true และข้อความว่าแสดง interactiveform แล้ว/issue ยังไม่ updated/ให้รอ Submit ไม่มี approval/requestID หรือ formURL ส่งกลับมา Public-safeR5Markdown และ Mermaid sources ไม่ได้แก้เพิ่มในรอบ reboot นี้

## ผลวินิจฉัยล่าสุด — guard ของการทดสอบเป็นข้อจำกัดที่พบจริง

ใช้ `/debug-mantra` จาก `C:/Users/xenod/.agents/skills/debug-mantra/SKILL.md` และคง main e5737a27 ใน worktree Native เดิม ไม่มีการเปลี่ยน served profile, branch, pagefile หรือ security settings และไม่ผ่อน hard process guard 42,000,000,000 bytes

รอบ `gyro-memory-diagnosis-03` ใช้ baseline exe SHA256 `3948784248ce3193ddd636792ce6f377fdb5e727eea5b241dcb279b8b8e46042` พร้อม Gyro-M 262K/5 workers และ frozen expert profile เดิม เครื่องมือใหม่อ่าน system commitment ด้วย GetPerformanceInfo, process private commit ด้วย GetProcessMemoryInfo และ job limits/peaks ด้วย QueryInformationJobObject พร้อม completion-port notifications ก่อนเริ่ม process

ผล: engine ออก 1 ก่อน token แรก error1455 เหมือนรอบ02 แต่คราวนี้จับ JOB_OBJECT_MSG_PROCESS_MEMORY_LIMIT (9) ของ PID51164 ได้ที่ controller11.610s ใกล้กับการรับบรรทัด error11.594s ซึ่งเป็นเวลา arrival ไม่ใช่เวลาเรียก VirtualAlloc แบบเจาะจง PeakProcessMemoryUsed เท่ากับ41,999,466,496 bytes เทียบ guard ที่ Windows ปัดแล้ว41,999,998,976 bytes เหลือ532,480 bytes รอบนี้จึงชน hard process guard จริง ไม่ใช้การคาดเดาจาก free RAM หรือ error code อย่างเดียว

system commitment ที่เก็บด้วย API ถูกต้องมี peak81,558,941,696 bytes จาก limit90,968,305,664 bytes; headroom ต่ำสุด9,409,363,968 bytes ส่วน physical available ต่ำสุด20,670,574,592 bytes ค่าเหล่านี้ยังเป็น samples ไม่ใช่ภาพ ณ instruction ที่ allocation ล้มเหลว แต่ positive process-limit event และ peak ชิด guard เป็นหลักฐานของ guard ที่ผูกกับ run นี้ ตัว controller ไม่ส่ง timeout/physical-floor stop

trace source ของ e573: generate.cpp6598 → ArenaExpertSource::load_rest5226 → host_commit5702 → PinnedArena::commit_interior599 เป็น pageable host expert commit ทีละ expert ก่อนอ่าน payload Layout ทั้ง48 layers กำหนด1,305,600 bytes ต่อ expert สอดคล้องกับ TQ2_T ของสาม matrices คำขอแบบปัดครอบหน้าเป็น1,306,624 หรือ1,310,720 bytes ไม่ใช่29.88GiB ก้อนเดียว ไม่มี debugger แบบ console ที่เข้าถึงได้ใน PATH/Windows Kits Debuggers/WindowsApps/VS Community ที่ตรวจ จึงยังไม่มี layer/expert identity หรือ exact page-rounded choice ของคำขอที่ล้มเหลว ไม่อ้างว่า instrument ได้แล้ว

สมมติฐานและผลตรวจ:

| สมมติฐาน | หลักฐาน/ผล |
|---|---|
| H1 task process guard ปฏิเสธ commit | ยืนยันว่าเกิด guard rejection: event9 ตรง PID, peak เหลือ532,480 bytes ก่อน cap; small control สร้าง1455+event9 ได้โดย system ยังมี headroom |
| H2 system commitment หมดเป็นคำอธิบายที่จำเป็น | ไม่รองรับในรอบนี้: system headroom samples ยังเหลืออย่างน้อย9.409GB และพบ process guard ที่มีผลจริง ไม่อ้างว่าได้วัด OS ณ instruction ที่ผิดพลาด |
| H3 allocation size ผิดหรือ overflow | ไม่พบหลักฐาน: logical blob1,305,600 bytes ตรง format/layout; exact failed index ยังไม่เก็บ |
| H4 MMQ warning ทำให้ startup ออก | ไม่ใช่ terminal path: หลัง warning engine ยังโหลด dense/MTP และ staging ทั้งสอง GPU ก่อนออกจาก host commit |

main ยังเป็น baseline ที่เป็นไปได้ การเปิด TQ MMQ/expert split ขึ้นกับ build dependency/configuration และต้องวัดแยก ไม่ใช้เปลี่ยน branch เป็นวิธีแก้1455 ข้อมูลนี้ยังไม่พิสูจน์ว่าเปิดได้เมื่อไม่มี guard หรือทรัพยากรเพียงพอตลอด inference จึงไม่ผ่อน cap ไม่แก้ production behavior และไม่ข้ามไป P1 ขั้นต่อไปต้องให้ผู้ใช้ตัดสินใจเรื่อง resource-fit gate หรือขอบเขตงานลด commit จากหลักฐาน โดยยังไม่เปลี่ยน context/quantization/profile แทนผู้ใช้

### การแก้เครื่องมือวัดและการทวนผล

field system_commit/commit_limit ของ controller รุ่นแรกมาจาก GlobalMemoryStatusEx และไม่ใช่ system-wide commitment ที่รับรองแล้ว ตัวเลข78,573,899,776/90,968,305,664 ของรอบ02จึงเป็น legacy non-authoritative evidence เก็บ run02 JSON/controller เดิมไว้ ไม่เขียนทับและไม่ใช้ค่านั้นวินิจฉัย system exhaustion

`tools/qwenputer/memory.py` แยก field/source/unit ของ system, process และ job; bootstrap ของ owned job จับ notifications ก่อนเริ่มเครื่องมือ การไม่มี event9/10 และ soft event11 ไม่กลายเป็นคำตัดสินว่า OS limit หมด Small control ใช้ child guard ไม่เกิน64MiB, commit สำเร็จเพิ่ม1MiB แล้วคำขอ8MiBถูกปฏิเสธ จับ9/11ได้จริงโดยไม่ต้องใช้โมเดล

ผู้ตรวจอิสระตรวจ ABI กับ Windows SDK และพบช่องโหว่ใน error paths: dequeued events อาจหายถ้า query ถัดไปผิดพลาด, telemetry failure อาจข้าม cleanup และ post-run probe อาจข้ามการเขียนรายงาน แก้ด้วยการเก็บ event ทันที, nested cleanup และ persist partial report ก่อน optional probes Tests ทำให้ข้อบกพร่องเกิดก่อนแก้และผ่านหลังแก้ ไม่มี model rerun หลังแก้ error paths นี้ สคริปต์v2ที่ใช้รอบ03เก็บเดิม; v3เตรียมและตรวจ syntax ไว้สำหรับครั้งหน้าเท่านั้น

ชุดสุดท้าย:22 tests,20ผ่าน,2 GPU probe tests skipped (ไม่ได้ตั้ง QWENPUTER_PROBE ใน suite นี้), DONE0; log p0-memory-suite-final.log ชุด GPU/control-flow เดิม9 tests และ10 cases ยังคง identity เดิม ไม่อ้างว่ารันใหม่ HIP/SYCL ไม่ได้รัน เพราะไม่มี engine/backend source ที่เปลี่ยนในรอบ telemetry นี้

การตรวจ persistence พบว่า default virtualized sandbox TEMP สร้างไฟล์ได้แต่ atomic replace ถูกปฏิเสธ WinError5 ทดสอบ differential แล้ว workspace และ explicit allowed Temp replace ผ่าน จึงให้ test fixture สร้างใน workspace writable root ไม่เปลี่ยน runtime persistence เพื่อซ่อน error

### Reproducer และ breadcrumbs

Exact executed argv (no production source changes):

```text
python -B -X utf8 tools/qwenputer/headless.py --log C:/Users/xenod/AppData/Local/Temp/strata-qwenputer-p0-20261028/gyro-memory-diagnosis-03.log --timeout 90 -- python -B -X utf8 C:/Users/xenod/AppData/Local/Temp/strata-qwenputer-p0-20261028/run_gyro_smoke_v2.py baseline --tag gyro-memory-diagnosis-03
```

Controller preflight ต้องมี system headroom >=42GB+2GiB และ physical available>=22GiB; คืนผลก่อน launch ถ้าไม่ผ่าน ใช้ hard guard42GB, soft notification40GB (ไม่ block), bound60s, system commit floor2GiB, physical floor1GiB และ sample target20ms (จริง15–47ms) ไม่มีงานหนักอื่นเริ่มระหว่าง repro

Evidence directory เดิมมี `memory-small-control.json`, `debug-mantra-preflight.json`, `gyro-memory-diagnosis-03/{command.json,result.json,stderr-events.json,on.stderr,test-config.json}` และ raw run02เดิม SHA256 ของ result.json รอบ03:9e67faf7c37d104dd502e10b144d47eae8bd5c4aa35e18722a50f19c69983985

Original model smoke ยัง FAIL ภายใต้ guard ไม่ใช่ model inference PASS; diagnostic attribution และ helper fixes ผ่านแล้ว P0 รวมยังไม่จบ ไม่มีงานหนักค้าง และไม่มีการทดสอบ performance/ABBA เพิ่ม

## Earlier component and discovery evidence — historical

## Source, isolation and instructions

- Baseline `e5737a27f485773a7ce22784fef6af7b480dcc9a`; branch `xeno/native-qwenputer-p0`, isolated under `D:/Github/Strata/.worktrees/native-qwenputer-p0`. Main's unrelated files remain outside this worktree.
- Loaded the installed `xeno-skills/skills/t4/using-t4/SKILL.md` and relevant planning, TDD, execution and memory instructions. The local `C:/Users/xenod/.agents/skills/to-prd/SKILL.md` requires tracker publication with `ready-for-agent`; issue #276 corrects the earlier local-only deliverable. `/to-issues` remains separate. The approved `/to-prd`-first order is retained.
- App-managed worktree creation is unavailable for this durable task; manual Git fallback was used after confirming `.worktrees` is ignored and the target absent. No existing checkout was reset.
- Evidence/build directory: `C:/Users/xenod/AppData/Local/Temp/strata-qwenputer-p0-20261028`. No downloaded third-party compiler, reference, firmware or model was executed. Existing vendor sources were reused; no install/download of weights occurred.
- Fast-mode preference is recorded; active task model/service tier cannot be verified or changed through available supported controls. No change is claimed.

## Hardware and build provenance

Primary is physical GPU 1, RTX 5060 Ti, UUID `GPU-059b90e2-2b5c-00b8-f3ba-f6dea8de083e`, 16,311 MiB. Secondary/display is physical GPU 0, RTX 4070 SUPER, UUID `GPU-fba37e4b-ea9e-66e9-c3fd-a16b2e833bc4`, 12,282 MiB. Both report WDDM. Engine mapping is `CUDA_VISIBLE_DEVICES=1,0`; the final probe checks the resulting primary UUID. Driver package is 621.01.

CPU is i5-13500, 20 logical processors; physical RAM 51,208,699,904 bytes. Preflight available physical memory was 25,140,346,880 bytes before and 25,238,638,592 after. These are snapshots, not measured model peaks. The 42 GB Strata private-commit budget and 2.5 GB total display headroom policy remain unchanged; no full model memory-compliance result is available.

Build compiler: CUDA 13.3.73 in the v13.3 toolkit (SDK version.json 13.3.1), MSVC 19.44.35228.0, Ninja, Release. Actual runtime/driver API values and loaded CUDART identity are recorded separately in the probe JSON; they must not be inferred from a directory name. The generic ggml CPU mode warning is recorded, not treated as a speed claim.

Unchanged baseline built 152/152 targets with root CMake hash `5b21527aaabdbd52b30cc3d10d8300cb32c58b342b459d40bfb328076bbb2382` and no tracked diff before build. Executable is 108,903,936 bytes, SHA256 `3948784248ce3193ddd636792ce6f377fdb5e727eea5b241dcb279b8b8e46042`. `cuobjdump --list-elf` found 59 sm_89 and 59 sm_120 entries. `--help` exited 0; executable hash remained unchanged. This is not a secondary-expert inference smoke.

Final candidate build/probe identities and observed outcomes are appended below.

## Historical Swift inventory (superseded by Gyro-M)

Profile `D:/Github/Strata/strata-flash-next-d2x.json` SHA256 is `07557c23cbac89c467c627f5723ba14a5142084ddb33ff7a1f88e7b28714f535`. `inputs.json` pins every configured executable, tokenizer, expert pack, MTP and profile input plus vendor files. Full files are streamed through SHA256; directory digests combine sorted relative names, byte counts and individual file digests. Search scanned 172,818 files without errors in existing model storage, staging, this model's HF cache and existing engine storage. No exact shard filename or matching content-addressed blob filename was found. This does not claim a scan of every drive.

| Role | Exact path / identity | SHA256 or tree digest |
|---|---|---|
| Served executable | `C:/Strata-exp/run-d2x-260/strata.exe` | `da93215c3eda1ae9e8e1d50ec7872696a9d502d51e9690fabdae48f499b005ed` |
| Tokenizer | `D:/Github/Strata/packs/swift15-iq2_xs/tokenizer` | `9016263eb8b85a4f9fc43c456c42426954836339945945f11f58a6ae992e0f99` |
| Expert pack | `D:/Github/Strata/packs/swift15-iq2_xs` | `0cdc7fa04c61a0cf4686815f0cfba498a7c072891367edfe23ae33a8802daa57` |
| MTP | `D:/Github/Strata/mtp/rt` | `6d5680c0a9346d807da0535fb7e6bbd9fdd79d47ad64edc6f84d648394c52aab` |
| Expert profile | `D:/Github/Strata/data/ranked-exl3only-profile.bin` | `a796734ba1489f10a2fe3daec34071da2ebd5add66a963d9a0469f4af925393d` |
| Existing ggml source | `D:/Github/Strata/third_party/llama.cpp/ggml` | `6872d48be8fb540ad4d82a12ff6ea51f2c7762cece2b69ea1752eb5dd3ddeb98` |

The parent Git repository's HEAD reported from a non-independent vendor directory is not a ggml revision proof. The version file pins `3cf03257f219afbe7334045ff7c6a06ac68c627d`; the actual existing source tree is separately hashed above.

Both required files belong at the configured native/PLE directory `C:/AI/models/ukisai-Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS/`:

| Filename | Exact bytes | SHA256 |
|---|---:|---|
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf` | 39,788,473,344 | `ca3b302d3d1eb6eddce3ea4acc79854e6cc5fdc0583e374a6a8b6bb67327743b` |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf` | 28,363,693,824 | `e4996a6de18fd49c58d9a0040fcfd73787db6407c51973f29884bee7b14c9dbb` |

Total 68,152,167,168 bytes (63.47 GiB). Local SHA256SUMS agrees with pinned [Hugging Face metadata](https://huggingface.co/api/models/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d?recursive=true&expand=false), retrieved as metadata only. Neither observed C: nor D: free space can hold this pair. Do not delete user files, change profiles/quantization or download silently. This was a historical Swift limitation. The user subsequently selected Gyro-M; no Swift restoration is needed or proposed.

## Scope and verification limits

The task-owned GPU diagnostic increments a counter and checksum to validate graph control flow. Its host checksum assertions are test oracles, not production arithmetic, and its elapsed times are diagnostic observations, not an arithmetic latency benchmark. It allocates before launch, initializes the condition on every launch, uses one same-device WHILE body and no body host callbacks/allocations. Cancellation is requested from a separate stream; maximum work is 4,096 finite steps. Rebuild and current-device-switch checks do not prove integration with an actual language decoder.

No greedy raw-token parity, model load, served D2x secondary expert smoke, Qwenputer circuit replay, ordinary-turn overhead, native arithmetic success rate, timeline or ABBA was run. No performance/default change is proposed. No prefill/cache/decoder/server code changed. HIP and SYCL toolchains are unavailable; their build validation is explicit debt under AGENTS.md, not a pass. No target machine security or TDR settings changed.

## Reproduction and deviations

Commands use `tools/qwenputer/headless.py`, which invokes existing tools with CREATE_NO_WINDOW, loads vcvars in a child only and prepends CUDA bin plus bin/x64. Full engine configurations use quoted `-DCMAKE_CUDA_ARCHITECTURES=89;120`; the standalone primary-only diagnostic may use 120. Exact argv and DONE exit markers are kept in each log.

- Default sandbox MSVC configure failed C1902; the identical installed toolchain configure passed with scoped elevated access. No PDB/toolchain/security modification was made.
- An initial unquoted PowerShell CMake argument split the ggml path and configured llama.cpp. That configure directory is rejected as baseline evidence. A new correctly quoted Strata directory was configured, and source/project/architectures were verified.
- Do not rebuild the completed baseline directory after candidate CMake edits. Its recorded executable remains immutable.
- The Bash SDD workspace helper cannot start in the sandbox; equivalent plan-scoped ledger files were created directly. No completion was inferred from a failed helper.

P0 is not a release or readiness approval for P1. The Gyro-M profile choice and model validation, plus backend validation debt, remain explicit gates. The only public issue operation is the approved PRD #276; implementation tickets and publication beyond it remain separate.


## Current Gyro-M test baseline (user steering, local-only)

The user selected Gyro-M for daily-use testing. No profile/default was edited and no model was loaded. `gyro-inspection.json` parses only the GGUF header using the existing project reader; `gyro-inputs.json` streams full SHA256 once per unique input (native/PLE aliases share the same file). Both metadata and hashing completed. The whole 91.97 GB GGUF was read for its digest; this was not inferred from a filename or remote checksum.

Verified metadata: GGUF v3, general.architecture=qwen4exp, name Qwen3.8-Flash-Next Gyro-M, 48 blocks, hidden size 2560, 512 experts / 10 used / expert FFN 640, native context 262144. Tensor counts: Q8_0=667, F32=389, TQ2_T=144, BF16=24. Pack format v5 declares Hadamard-folded TQ type144 experts; hadamard.txt exists. This is not Swift IQ2_XS.

Tokenizer: gpt2/qwen35 descriptor, vocabulary248320, separate vocabulary/merges/types. MTP uses the existing rt dense/experts/draft-vocab files. Both profiles reference it with spec4; runtime numerical compatibility still requires execution. Geometry and file existence are not a compatibility pass.

Hub evidence distinguishes key3/native262K from key3y/512K YaRN. No active --config could be identified by the permitted process inspection. Native262K uses5 workers and run-gyro-d145r; 512K uses13 workers, YaRN scale2 and run-gyro-reader. Both preserve secondary expert offload, prefill split/wave, KV int8/resident65536, display reserve3277, MTP/suffix and an expert-profile-save target. Profile selection was sent as a concise pending user question; no model-dependent comparison was launched while waiting.

Any later controlled run must copy/freeze the learned expert profile into task-owned files and redirect its save destination so a baseline arm cannot mutate the candidate's input or the daily-use learned profile. Use a private port if a server is needed; do not stop/reconfigure the daily-use server. Test configuration copies must preserve the selected serving settings and document any justified difference, including vision companion/accounting.

| Current input | SHA256 / tree digest |
|---|---|
| `D:\Github\Strata\strata-gyro-m-d2x.json` | `06ef847e063bb9643c7bf3f486e2ffdadc29737493268238c21f59188d5f6c5b` |
| `D:\Github\Strata\strata-gyro-m-d2x-512k.json` | `8da3719abc96fe0597f5e84ecf326f12c880729ab52c9ce54229e1b5ca4a1d15` |
| `C:\Strata-exp\run-gyro-d145r\strata.exe` (144387584 bytes) | `8fbf47992bb7539eaa6cb049efa6423f09927ecf2cee4c9cecd321479926b300` |
| `D:\Github\Strata\packs\gyro-m\tokenizer` (10093356 bytes) | `cbad9f317a1b84da005fcecbd6b3d4423d7ce3f53cc45b0ccab8cbb5aeb22b88` |
| `D:\Github\Strata\packs\gyro-m` (4962575916 bytes) | `75bfe5952f2affc039067f5f3b96e1495d044c9f990349794bb86da872f0d009` |
| `C:\AI\models\agentionai-Qwen3.8-Flash-Next-Gyro-GGUF\Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf` (91965774272 bytes) | `7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f` |
| `D:\Github\Strata\mtp\rt` (824236860 bytes) | `6d5680c0a9346d807da0535fb7e6bbd9fdd79d47ad64edc6f84d648394c52aab` |
| `D:\Github\Strata\data\gyro-m-learned-profile.bin` (196632 bytes) | `3e7699a41dd3acc063541af2119079bc5174c67e811cf26527de512858e7cdba` |
| `C:\Strata-exp\run-gyro-reader\strata.exe` (127988224 bytes) | `4eca09d932ed1a706c35aa8094e70529d712704efe290ac744fb3551cbb1e209` |

No revised PRD or local model/profile details were published. Parent reports that automatic approval review rejected the proposed public Gyro-M issue edit because authorization to disclose local details was missing. This execution neither retried that publication nor uploaded R3. GitHub #276 remains the previously authorized R2 PRD.

## Final component results

| P0 check | Observed result |
|---|---|
| Unchanged engine build | PASS152/152, sm89+120,59 ELF entries each, help0, immutable hash recorded above |
| Candidate engine and diagnostic build | PASS154/154, sm89+120,59 ELF entries each, help0 |
| Real graph create/instantiate | PASS on RTX5060Ti; primary UUID checked from JSON |
| Graph control flow | PASS0/1/17, repeat5/repeat0, output-bound3, cancel, current-device switch, recreated graph1/0 |
| Negative control | PASS detector: real GPU increment2 produces one.steps2/expected1, ok=false, exit1 as intended |
| Final full helper/GPU suite | PASS9 tests,3.814s,0 failures, DONE exit0 |
| Gyro input inventory/full hashes | PASS, all configured engine/model/tokenizer/pack/MTP/profile inputs exist |
| Profile selection | RESOLVED:Gyro-M262K/5workers selected by user |
| Model/secondary tier inference smoke, greedy parity, ABBA | Baseline startup attempted and failed before token0; actual secondary execution/candidate/parity/ABBA unpassed; see current attempt below |
| Full model resource gate | NOT RUN; inventory snapshots and diagnostic process metrics are not full model peaks |
| HIP/SYCL | NOT RUN: toolchains unavailable; explicit validation debt |

- Candidate probe: `C:\Users\xenod\AppData\Local\Temp\strata-qwenputer-p0-20261028\candidate-build\tests\qwenputer\qwenputer_conditional_probe.exe`, 102912 bytes, SHA256 `af91478941e6c6331597ddf9dc0b870f2ff120cee7eca40956491b8865b405fc`.
- Candidate engine: `C:\Users\xenod\AppData\Local\Temp\strata-qwenputer-p0-20261028\candidate-build\strata.exe`, 108903424 bytes, SHA256 `550499c3ea61f4077e4c5fca61fd202889498a156fc8d6c0b44c53ee2a68040e`.

Probe API values: headers=13030, runtime=13050, driver_api=13050; observed values are retained without assuming they equal the compiler version. CUDART DLL-name lookup was empty, so a loaded DLL identity is not established by it. Primary free VRAM before/after=15908995072/15908995072 bytes; this net observation is not a peak. Process private at end=511991808, peak working set=232329216 bytes. Cancel completed at 369 of4096 steps. Timings in probe.json are diagnostic observations only.

## Independent review and remaining work

A fresh-context Astra reviewer read the P0 code/diff and requirements without edits or execution. No Critical findings. One Important finding was verified: the old launcher wrote DONE after killing only the immediate child, while a grandchild survived. Its real parent/grandchild test failed before the fix. The launcher now assigns a waiting worker to a task-owned Windows kill-on-close job before releasing a stdin handshake. Commands and descendants inherit that job. Timeout or normal completion terminates/drains the job before DONE. Assignment failure cannot start the tool. The regression test and final9-test suite passed.

Deferred minor: the probe source comment says recreated allocations have fresh addresses; CUDA may reuse addresses. The test establishes destruction/recreation/state reset, not address relocation. No relocation claim is made here.

The reviewer set aside P1+ production arithmetic/integration because outside authorized P0; final-source GPU/provenance checks because then pending (now executed above); Gyro model runs because profile selection remains pending; HIP/SYCL because unavailable (still release-validation debt). None is silently counted as passed.

R3 local reading draft applies the actual installed kien-thai skill (technical explainer, topic-first/condition-first structure, clear paragraphs, consistent developer terms, no polite padding or recap). Its full24/10/9 coverage and verbatim preserved R2 contracts are recorded separately. Native-model availability check contacted localhost11434 only and was refused; no model/service was started or downloaded. The skill's direct drafting fallback was used.

Follow-up requested by parent: corrected the probe comment to describe destruction/recreation and reset only; allocator addresses may be reused, so this is not an address-relocation test. The diff changes comments only and no rebuild/runtime result is claimed for that wording edit.


## Selected 262K model smoke attempt (2026-10-28, private evidence)

The user selected the native262K/five-worker profile. Its SHA256 remained06ef847e063bb9643c7bf3f486e2ffdadc29737493268238c21f59188d5f6c5b. Runtime test copies redirect both learned-profile paths to task-owned files; source profile and original learned-profile hashes were unchanged afterward. Exact tokenizer vocabulary, merges and token types matched the GGUF; special IDs were checked, not inferred from count. Details are in gyro-tokenizer-binding.json.

Attempt01 stopped before launching the engine because Windows rounded the requested job commit limit to page size. Attempt02 verified the actual41999998976-byte limit (requested42,000,000,000), then ran the unchanged baseline with the selected profile, identical engine arguments/env and a21-token greedy smoke prompt, max-new16. No server, vision encoder, MCP or API listener was launched; vision accounting settings were retained. This is component smoke, not a full served-turn benchmark.

Attempt02 exited1 after14.641s, before any generated token, with placement-first load: VirtualAlloc(MEM_COMMIT) failed with Windows error1455. The resource controller itself did not signal its physical-floor/timeout stop. Sampled engine private peak33,921,974,272bytes, working-set peak5,388,955,648bytes; system commit peak78,573,899,776 of90,968,305,664bytes; minimum available physical19,596,664,832bytes. The job's committed-memory guard and sampled PrivateUsage are not interchangeable evidence; this attempt does not prove that the repo's private-commit budget is inherently exceeded. Do not silently increase limits or shrink the selected profile.

The startup loaded MTP weights and staged5607 secondary experts/6.82GiB in SECONDARY COMPUTE mode. This proves secondary-tier setup, not executed secondary compute or TQ2_T/MTP numerical correctness. No inference/parity/ABBA pass is claimed. Stock-vendor baseline logs also state no layers onMMQ and expert split cannot run, so this binary cannot be presented as validating the daily-use Gyro prefill split.

The selected profile's deployment comment records run-gyro-d145r/sha8fbf47992bb7 as xeno/defer-m145@9c3f16f8949473fbd3b3531608b4d2c930040308, defaultggml plus trellis overlay. That commit/branch exists locally. This differs from plan baselinee573 and its stock vendor. No exact matching deployed binary was found among top-levelbuild*/strata.exe candidates; profile comments are provenance evidence, not a reproducible binary-source proof by themselves. A user choice between the daily-use branch and the originalmain baseline has been requested; no source checkout switch, merge or further model retry was done while pending.

No task-owned model process remains running. Candidate model inference, actual secondary execution, greedy parity, resource fit and performance gates remain unpassed. Existing component build/probe evidence retains its original identity.

## Authorized R4 public synchronization

The later user authorization superseded the earlier public-PRD restriction. Existingissue#276 now matches the public-safe R4 Markdown exactly; bodySHA256 1dcf885081d171b87e8f07026877a3a81bd9ca713e492bcd8bdf1126515f8dba. Its bilingual normativeR2 technical appendix is https://github.com/xenodeve/Strata-xeno/issues/276#issuecomment-6479163682. No private paths/UUIDs/local resource logs were included. Three Mermaid diagrams passedMermaid11.16.0 parse pluslocalheadlessChrome render and visual inspection; actualGitHub renderedSVGs were not verified. No unrelated issue, repositorypush,PR,merge,deployment orlibraryupload occurred.
