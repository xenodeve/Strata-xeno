# Native Qwenputer — ปิด P0

P0 พร้อมให้ทบทวนก่อนเริ่ม P1 แล้ว การตรวจ CUDA conditional graph และการโหลด Gyro-M ผ่านทั้ง unchanged main และเส้นทาง hub จริงผ่าน ไม่มี production Native circuit ในงานช่วงนี้

Source base ของ worktree ยังเป็น `e5737a27f485773a7ce22784fef6af7b480dcc9a` บน branch `xeno/native-qwenputer-p0` ไม่ได้ย้ายไป branch ของ daily-use engine ไม่ได้แก้ served config, push หรือเปิด PR

## รายการตรวจตามแผน P0

| รายการ | ผล | หลักฐานและขอบเขต |
|---|---|---|
| แยก worktree และ build directory | ผ่าน | worktree `native-qwenputer-p0`; baseline และ candidate อยู่คนละ directory; เก็บงานเดิมไว้ |
| Conditional WHILE บน primary จริง | ผ่าน | `probe.json`: create/instantiate/launch/relaunch, 0/1/17/5 iterations, zero หลัง relaunch, output bound, cancel, device switch และ rebuilt graph รวม 10 cases |
| Negative control | ผ่าน | จงใจเปลี่ยน GPU increments แล้ว semantic assertion จับผลผิดได้ ไม่ใช่ทดสอบแค่การมี binary |
| Toolchain/device provenance | ผ่าน | RTX 5060 Ti เป็น primary, RTX 4070 SUPER เป็น secondary/display, WDDM, driver 621.01, NVCC 13.3.73, MSVC 14.44.35207; runtime/driver API อ่านได้ 13050 และ headers 13030 |
| Full engine build `89;120` | ผ่าน | baseline 152/152, candidate 154/154, help exit 0 ทั้งคู่; แต่ละ executable มี cubin 59 รายการสำหรับ sm89 และ 59 สำหรับ sm120 |
| Gyro-M input/tokenizer binding | ผ่าน | เลือก 262,144 context / 5 workers; ตรวจ GGUF, pack, PLE, MTP, tokenizer content และ special IDs; ไม่ดาวน์โหลด Swift |
| Main component smoke และ secondary tier | ผ่าน | cap-free run05 exit 0, 16 raw output IDs, secondary compute 1773 entries / 1449 groups, MTP accepted 7/14 |
| เส้นทาง hub ที่ใช้จริง | ผ่านเฉพาะ startup/inference smoke | run07 เปิด server/executable ตัวเดียวกับ hub ผ่าน config copy และ no-window shim; API สร้าง reasoning 16 tokens; ไม่มี private cap หรือ soft threshold; ไม่ใช้ผลนี้อ้างคุณภาพคำตอบ |
| No-window/timeout/owned cleanup | ผ่านหลังแก้ launcher | review พบ server เดิมไม่ส่ง no-window flag ต่อใน run06 จึงไม่อ้างว่า run06 ผ่านเงื่อนไขนี้; run07 ใช้ task shim และ child control ยืนยัน GetConsoleWindow=false; ทุก command จบด้วย DONE และ task-owned drain |
| Unit/control tests หลังแก้ telemetry | ผ่าน | 23 tests: 21 pass, 2 GPU tests skipped; GPU suite แยกก่อนหน้านี้ 9 tests ผ่าน และ probe 10 cases ผ่าน ไม่รวมจำนวนสองชุดเข้าด้วยกัน |
| Evidence, blueprint และ review | ผ่าน | เก็บ raw logs และ manifest; Standards/Spec review ไม่มี blocker; มี Low หนึ่งข้อเรื่อง ctypes ABI ซ้ำ ซึ่งไม่จำเป็นต่อการแก้ startup |

Candidate-disabled inference parity, arithmetic correctness, backend matrix, memory fit ของ Native circuit และ ABBA อยู่ใน G6–G9/P8 หรือช่วง implementation ที่เกี่ยวข้อง ยังไม่ผ่านและไม่ใช้เป็นเงื่อนไขเพิ่มให้ P0 การ build/probe สำเร็จไม่ได้พิสูจน์ข้อเหล่านี้

## Hub ไม่ได้ตั้ง hard cap 42 GB ในเส้นทางที่ตรวจ

ใน repository พบ `strata-hub.bat`; ไม่พบไฟล์ชื่อ `serve-hub.bat` ที่ root ที่ตรวจ ไม่อ้างว่าไม่มีไฟล์ชื่อนี้ที่อื่นทั้งเครื่อง

เส้นทางที่ตรวจคือ `strata-hub.bat` → `C:\Strata-exp\src-main\serve\server.py` → `strata-gyro-m-d2x.json` → `C:\Strata-exp\run-gyro-d145r\strata.exe --serve`:

- `strata-hub.bat:93–97` เลือก Gyro-M และตั้ง `NEEDGB=24`; `:119–126` ตรวจ RAM ว่างก่อนเริ่มเพียงครั้งเดียว ไม่ใช่เพดานระหว่างทำงาน
- `strata-hub.bat:24–27,130–137` เลือก Python/server, ตั้ง GPU order และส่ง profile ให้ server ไม่มี memory-limit argument
- `C:\Strata-exp\src-main\serve\server.py:57–59` เลือก import root ของ server ตัวนั้น; `:751–753` เปิด engine แล้วเรียก `contain()`
- `C:\Strata-exp\src-main\serve\winjob.py:49–59` สร้าง Windows Job โดยตั้งเฉพาะ `KILL_ON_JOB_CLOSE=0x2000` แม้ ctypes structure ประกาศช่อง ProcessMemoryLimit/JobMemoryLimit แต่ไม่ได้ตั้ง memory-limit flags
- Profile ที่เลือกไม่มี `--ram-cache-gib` หรือ process/job memory-limit setting ค่า `--expert-cache` ควบคุม expert placement ไม่ใช่ private-commit cap
- `AGENTS.md:295–296` ระบุ 42 GB เป็นงบที่วัดเทียบ ไม่มี config/launcher บังคับ ส่วน `--ram-cache-gib` เป็นขนาด host expert cache สำหรับ capacity mode ไม่ใช่เพดาน memory ของทั้ง process

เพดานมาจาก P0 harness เดิม `run_gyro_smoke.py:31–35` ซึ่งตั้ง `JOB_OBJECT_LIMIT_PROCESS_MEMORY=0x100` และ 42,000,000,000 bytes ให้ task-owned Job ต่อมาปิดตามคำสั่งผู้ใช้โดย query ยืนยันว่าไม่มี hard process/job cap ยังเก็บ kill-on-close และ emergency timeout ไว้ Capped run03/04 จึงเป็นหลักฐานวินิจฉัย harness ไม่ใช่ baseline ของ hub

Comment ใน `strata-hub.bat:14–17` ระบุชื่อ Claude Code profile ที่เปิด hub แต่ไม่พบ command ชื่อนั้นใน PATH ของ session นี้ จึงยังยืนยัน wrapper ของ Claude Code ทั้งเส้นทางไม่ได้ และไม่ได้ตรวจ job restrictions ของทุก process ในระบบ ข้อสรุปเรื่องไม่มี cap จำกัดเฉพาะ launcher/server/profile ที่อ่านและ task run ที่วัดแล้ว

## Run07 — ใช้ hub path จริงหลังแก้ no-window wrapper

ใช้ Python/server/executable, lib_dirs, model, context, workers, cache/offload, profile env และ engine priority settings เดิมของ hub ต่างเฉพาะ port เป็น `127.0.0.1:18091`, log/learned-profile outputs ของ task, no-window shim และ outer job สำหรับ cleanup ไม่มีการเติม CUDA PATH หรือเปลี่ยน backend ผ่าน harness นี้ ให้ server จัดการ dependencies ตาม config เดิม

Standards review พบว่า server เดิมเปิด console children โดยไม่ส่ง `CREATE_NO_WINDOW` แม้ outer wrapper ซ่อน Python แล้ว จึงแก้เฉพาะ task ด้วย `server_no_window_shim.py` ซึ่งใช้ runpy เปิด source server เดิมและ OR flag นี้ให้ทุก subprocess.Popen โดยรักษา flags อื่น AST check ยืนยัน omission เดิม และ small real-child control ยืนยัน GetConsoleWindow=false ผล run06 ยังเก็บไว้ แต่ไม่อ้างว่าตรงกติกา no-window ไม่มีหลักฐานจากการดูหน้าจอว่ามีหรือไม่มีหน้าต่างปรากฏใน run06 ตัว server ที่ใช้งานจริงไม่ได้ถูกแก้

Task wrapper เก็บ passive memory samples มี preflight, 120 วินาที emergency timeout และ system/physical floors ซึ่งไม่ถูกเรียกใช้ใน run ที่สำเร็จ ไม่อ้างความเท่ากันของ timing กับการเปิด hub ปกติ และยังไม่ได้เก็บ runtime DLL map หรือยืนยัน source revision ของ daily executable จาก BUILD.json ซึ่งไม่มีใน directory นั้น

| รายการ | ค่าจริง |
|---|---|
| Daily executable SHA256 | `8fbf47992bb7539eaa6cb049efa6423f09927ecf2cee4c9cecd321479926b300` |
| Server SHA256 | `10be781197be5ffe7e41e9307c53d7444631f697cc1c81d456a2b75ee950ccbc` |
| Original profile SHA256 | `06ef847e063bb9643c7bf3f486e2ffdadc29737493268238c21f59188d5f6c5b` |
| Outer Job flags / hard limits ก่อนและหลัง | 8192 / process null, job null; notification threshold null |
| Engine lifetime private peak | 46,784,667,648 bytes |
| รวม peak job commit ทุก process ของ task | 49,285,103,616 bytes; ไม่เรียกค่านี้ว่า engine private commit |
| Sampled system commit peak / limit | 69,268,910,080 / 87,963,570,176 bytes |
| Sampled system headroom ต่ำสุด | 18,694,660,096 bytes |
| Sampled physical available ต่ำสุด | 8,244,015,104 bytes |
| Samples / gap สูงสุด | 100 / 1.219 วินาที; ไม่ใช่ exact system peak ที่ boundary |
| API | HTTP สำเร็จ, completion 16 tokens ใน reasoning, finish_reason=length, content=null |
| Actual request routing | primary 3023 + secondary 1018 + CPU 4119 entries; NVMe loads 0 |
| MTP | accepted 11/11 |
| Cleanup / originals | task-owned drain verified; served JSON กับ learned profile ต้นฉบับ hash เดิม |

API run นี้หยุดคำตอบที่ 16 tokens ขณะโมเดลยังคิด จึงพิสูจน์ว่าโหลดและ inference ทำงาน แต่ไม่ได้พิสูจน์คำตอบสุดท้าย คุณภาพ reasoning หรือเลขคณิต ไม่มีการใช้ตัวเลขเวลาใน run เดียวอ้าง performance win

Hub engine ใน run07 วาง GPU experts 5695 + 4264 ตัว ต่างจาก main component run05 ที่ 5695 + 5607 แม้ใช้ profile settings เดียวกัน จึงเก็บสอง executable เป็นคนละ control และห้ามนำมาลบ timing กันเพื่ออ้างผล Native circuit ผล main เป็น development control; ผล hub เป็น operational reference การทดสอบ candidate ภายหลังต้องมี unchanged source-matched control และ matched hub workload อย่างชัดเจน

Server log เตือน primary VRAM ว่างประมาณ 488 MiB ตอนเริ่ม request นี่เป็นข้อจำกัดที่ P4 ต้องวัดและแก้เรื่อง static reservation ก่อนอ้าง Native memory fit ไม่ได้พิสูจน์ว่ามี paging หรือว่า Native circuit รันได้ในพื้นที่นี้ งบ 42 GB ของ repo ยังไม่เปลี่ยน การเอา cap ออกเป็นข้อยกเว้นการทดสอบที่ผู้ใช้สั่ง ไม่ใช่นโยบาย budget ใหม่

## Evidence และ identity

Private evidence root: `C:\Users\xenod\AppData\Local\Temp\strata-qwenputer-p0-20261028` เก็บ `probe.json`, build/cubin/help logs, `gyro-no-private-cap-05/`, `hub-matched-smoke-06/`, `hub-matched-smoke-07/`, `shim-child-control.json` และ `p0-final-input-manifest.json` ห้ามนำ private absolute paths หรือ machine inventory ส่วนนี้ไปใส่ public PRD

Main control executable SHA256: `3948784248ce3193ddd636792ce6f377fdb5e727eea5b241dcb279b8b8e46042` Candidate SHA256: `550499c3ea61f4077e4c5fca61fd202889498a156fc8d6c0b44c53ee2a68040e` Hashes ถูกอ่านยืนยันอีกครั้งก่อนปิด P0 ไม่ได้ rebuild ทับ

Manifest แยก permanent input ออกจาก runtime tail cache ซึ่ง engine สร้างและล้างเองตาม cache key ไม่ใช้ hash ของ pack tree ที่รวม cache เก่าอ้างว่า permanent inputs เปลี่ยน และไม่ลบ raw manifest เดิม Full input rehash ทำหลัง reboot; ตอนปิดงานตรวจ size/mtime ของไฟล์ใหญ่เทียบ snapshot และ hash executable/config ใหม่ โดยระบุวิธีตรวจไว้ใน manifest

## งานถัดไปและข้อจำกัด

เตรียม [P1 execution proposal](../plans/2026-10-28-qwenputer-p1-execution-proposal.md) จาก source ที่ pin แล้วเท่านั้น ยังไม่ได้ execute compiler, firmware หรือ reference simulator ต้องผ่าน action-time approval ของคำสั่งนั้นก่อน

GitHub #276 ยังอ่านกลับได้เป็น R4 Local PRD เตรียมเป็น R7 เพื่อระบุ actual-hub baseline และผล P0 ให้ตรงกับคำสั่งล่าสุด มี R5 Submit form ค้างอยู่ Connector ไม่คืน approval ID หรือ URL ของ form จึงระบุตำแหน่งใน UI ให้แน่นอนไม่ได้ ไม่ retry, ใช้ CLI แทน หรืออ้างว่า R7 sync แล้ว ยังไม่มีการเปลี่ยน public PRD ระหว่างที่ form ค้าง
