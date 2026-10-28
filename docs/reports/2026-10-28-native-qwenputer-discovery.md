# Native Qwenputer: ผลสำรวจและ baseline วันที่ 2026-10-28

**อัปเดตล่าสุด:** ผู้ใช้เรียก `/to-prd` ก่อนอย่างชัดเจนแล้ว ลำดับ `/grill-me` ที่บันทึกด้านล่างเป็นประวัติก่อนคำสั่งนี้ ไม่ใช่ blocker ของ planning ปัจจุบัน ดู [PRD](../plans/2026-10-28-native-qwenputer-prd.md) และ [implementation plan](../superpowers/plans/2026-10-28-native-qwenputer.md) ฉบับ R2 สำหรับขอบเขตและสถานะล่าสุด ยังไม่มี production implementation

สถานะล่าสุด: **ผู้ใช้อนุมัติ architecture แล้วด้วยข้อความ “เอาตามนั้น”** หลังได้รับข้อเสนอ native C++/CUDA sparse FFN/GDN บน GPU เดียวกับ decoder พร้อม weights artifact แยกและ GPU trace loop ไม่เปิดการตัดสินใจเรื่องนี้ใหม่ ยังไม่ได้ implement เพราะขั้นตอน T4 แบบ user-invocation-only ยัง pending ได้รับ working brief ที่ parent ส่งข้อความจากผู้ใช้มาให้แล้ว: Native ก่อน Universal, ใช้วงจรที่ compile เป็น weights จริงใน forward/decode ของ Qwen3.8 Flash-Next, คง expert offload และเส้นทางปิด feature เดิม, เริ่มพิสูจน์ด้วย arithmetic การอ่านลิงก์ไม่สำเร็จไม่ใช่ blocker สำหรับขอบเขตนี้อีกต่อไป

## เนื้อหาบทสนทนาที่ยังเข้าถึงไม่ได้

- ลิงก์ที่ผู้ใช้ส่ง: <https://chatgpt.com/share/6ae225a4-01d0-83ec-ada0-50f5564e152f>
- ตัวอ่านเว็บคืนชื่อ `ChatGPT - ประเมิน Qwenputer` แต่ไม่มีเนื้อหาข้อความ (`Total lines: 0`) การเห็นชื่อแชตไม่ใช่การอ่านบทสนทนาสำเร็จ
- พบแท็บต้นฉบับใน browser inventory แต่การขออ่านผ่าน `cua.getTab('721139517', {browser:'3'})` ถูกปฏิเสธด้วย `Host access is restricted by policy` ยังไม่ได้อ่านเนื้อหาแท็บ และไม่ได้ลองข้ามข้อจำกัดด้วยช่องทางอื่น
- ต่อมา parent ส่งข้อความต้นฉบับที่มีอยู่และ working brief มาให้แล้ว ใช้ brief ดังกล่าวเป็นขอบเขต โดยไม่อ้างว่าอ่านคำตอบ ChatGPT ที่ไม่สามารถเข้าถึงได้

## Repository และ workflow

- พื้นที่ทำงาน: `D:\Github\Strata`, branch `main`, commit ย่อ `e5737a27` (`Merge pull request #238 from xenodeve/xeno/telemetry-history`)
- ก่อนเริ่มไม่มี tracked diff แต่มี untracked files จำนวนมาก รวม build/model profiles, reports, launcher backups และ crash dumps เดิม ไม่ได้แก้หรือจัดเก็บไฟล์เหล่านั้นเข้า commit
- อ่าน `AGENTS.md`, `CLAUDE.md`, baseline/revision log ของ `docs/BLUEPRINT.md`, `docs/OPEN-WORK-LEDGER.md`, `docs/adr/0001-q2-exp-baseline.md` และ personal memory index ที่เกี่ยวกับ Strata แล้ว
- Blueprint ระบุว่าส่วนเนื้อหายังอธิบาย baseline เก่ากว่า `main` ต้องตรวจ source ปัจจุบันประกอบก่อนอาศัยตำแหน่งอ้างอิง
- ไม่มี `Obsidian-*/Home.md` หรือ `docs/agents/` ใน checkout นี้ ไม่ได้สร้าง governance files เพิ่ม
- โหลด skill ที่ผู้ใช้ระบุแล้ว: `C:\Users\xenod\.codex\plugins\cache\xeno-skills\xeno-skills\0.1.0\skills\t4\using-t4\SKILL.md`
- อ่าน `t4-agent-memory`, `t4-dev-workflow`, `t4-bro`, `karpathy-guidelines` ตามเส้นทางของ `/using-t4` รวมถึง Superpowers สำหรับสำรวจ ออกแบบ และวิเคราะห์ test failure
- `t4-dev-workflow` ระบุว่า `/grill-me`, `/grill-with-docs`, `/to-prd`, `/to-issues` เป็น user-invocation-only และห้ามทำผลลัพธ์แทนเอง ขั้นตอนเหล่านี้ยังไม่ได้ทำ และยังยืนยันไม่ได้ว่ามีผลลัพธ์พร้อมอยู่ในแชตที่อ่านไม่ได้หรือไม่
- ค้น open issues และ PRs ด้วย `gh ... --repo xenodeve/Strata-xeno --state open --search Qwenputer` ได้ `[]` ทั้งคู่ ขอบเขตการค้นคือคำว่า `Qwenputer` ไม่ใช่การตรวจทุก thread
- ไม่ได้เปิด issue, publish, push, สร้าง PR หรือเปลี่ยน config/launcher ของ server การเขียน tracker ยังอยู่นอกขอบเขตที่ได้รับจาก task นี้

## เครื่องมือและทรัพยากรที่ตรวจได้

ค่าทรัพยากรเป็น snapshot ตอนสำรวจ ไม่ใช่ peak ระหว่างโหลดโมเดล

| รายการ | ผลที่อ่านได้ |
|---|---|
| GPU 0 | RTX 4070 SUPER, 12,282 MiB, ใช้ 3,118 MiB |
| GPU 1 | RTX 5060 Ti, 16,311 MiB, ใช้ 185 MiB |
| Python | 3.11.9 |
| PyTorch | 2.14.0+cpu |
| Transformers | 5.14.1 |
| NumPy / safetensors / pytest | 2.4.6 / 0.8.0 / 9.1.1 |
| chipformer | `importlib.util.find_spec('chipformer')` เป็น `None` |
| โมเดลใน `C:\AI\models` | ไม่พบชื่อ directory ที่ตรง `qwenputer` หรือ `chipformer`; ไม่ได้สแกนทุก drive |
| พื้นที่ว่าง C: / D: | 38,579,011,584 / 7,765,823,488 bytes |

`Get-CimInstance Win32_OperatingSystem` และ `Win32_Processor` ถูกปฏิเสธ (`Access to a CIM resource was not available to the client`) จึงไม่มีผลตรวจ CPU/RAM จาก CIM ใน session นี้ ไม่ได้เปลี่ยนสิทธิ์ระบบเพื่อให้อ่านได้

ไฟล์ exe ที่ตรวจ hash เพื่อระบุ inventory เท่านั้น ยังไม่ได้รัน:

- `engine-xeno/strata.exe`: SHA256 `935E774B4472E88A03DD83F4D14BDED74A57B01263973D3D85351DC41935A276`
- `build/strata.exe`: SHA256 `E1C7770988502BA9FE59E73D2A1263E1AF578D0C8820B3FF72C7B13A13A0B9E8`

## Source ของ Qwenputer ที่อ่าน

ตรึง source ที่ commit `ca13310e2f9f788d1dd3feeea8213c1b0c23913c` ของ <https://github.com/divagr18/qwenputer> ดาวน์โหลดเฉพาะข้อความ source 10 ไฟล์ไปยัง `%TEMP%\strata-qwenputer-discovery-20261028` เพื่ออ่าน ไม่ได้ import, install หรือรัน source เหล่านั้น

- [`experimental/riscv/chipformer/rv/chat.py`](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/chat.py#L100): `RvChat` ยังโหลด merged `Qwen3_5ForCausalLM` ทั้งโมเดลใน BF16 ก่อนย้ายไป device แม้เปิด executor; ใช้ตรง ๆ ไม่ได้แปลว่าโหลดเฉพาะวงจรเล็ก
- `chat.py:133-170`: executor สร้าง residual หลัง chip layers จาก embedding และค่าที่วงจรเก็บไว้ การใช้วงจรเดิมต้องรักษาการเชื่อมต่อและ token mapping นี้
- `chat.py:324-347`: firmware snapshot ถูกฉีดหลัง prompt prefill
- `chat.py:393-426`: chip token ผ่านเฉพาะวงจร; ชั้นภาษาทำงานและตำแหน่ง `pos` เพิ่มเฉพาะ text token ต้องรักษา invariant นี้เมื่อเชื่อมกับการ decode ของ Strata
- [`tritonexec.py:1-16`](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/tritonexec.py#L1): `Executor2` ใช้ sparse rows และ FP32 แทน dense HF modules เป็นการประเมินวงจรเดิม ไม่ใช่ host calculator
- `fastexec.py:88-92`: reference executor ปิด TF32 เพื่อรักษาผลคูณ FP32 ให้ exact
- [`experimental/riscv/pyproject.toml`](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/pyproject.toml): HF extra ต้องการ `torch>=2.14.1` และ `transformers>=5.19.0` สูงกว่าที่ติดตั้งอยู่ ทั้งนี้ไม่ได้พิสูจน์ว่าเส้นทาง C++ ที่จะสร้างต้องใช้ Python dependencies เหล่านี้

## จุดเชื่อมต่อใน Strata ที่อ่านแล้ว

รายการนี้เป็นผลอ่าน source ไม่ใช่ design ที่ผู้ใช้อนุมัติ หรือแผนไฟล์ที่จะเปลี่ยน

- `include/strata/core/layout.hpp:27-64`: geometry มี Flash-Next 48 layers, `n_embd=2560`, GDN/QSA, hyper-connections และ MoE; `check_layer`/`check_all` ตรวจ tensor ตามสัญญาของ kernels จึงไม่มีหลักฐานว่าตัวโหลดเดิมรับ 11 dense Qwen3.5 chip layers ได้โดยตรง
- `include/strata/core/verify.hpp:1-23,66-96`, `src/core/verify.cpp:2012,2407`: `Verifier::run` ประเมิน speculative window และ `commit` ทำ accepted prefix ถาวร Native Plugin ต้องไม่ทำให้ trace tokens ล่วงล้ำเข้า language window หรือ commit ที่ไม่ตรงกัน
- `include/strata/core/conversation_snapshot.hpp:50-110`: ขนาด checkpoint, snapshot validation และ restore อิง state ที่ session เป็นเจ้าของ ต้องกำหนดการเก็บ circuit state และการ restore ให้ครบก่อนขยาย checkpoint
- `src/program/generate.cpp:5919-5935`: `--vram-elastic` ถูกปิดเมื่อมี layer split, helper caches หรือ fork tiers ที่เข้าข่าย ไม่ควรสมมติว่าใช้ลด expert cache บน D2x dual GPU ระหว่างรันได้แล้ว
- `docs/adr/0001-q2-exp-baseline.md`: การคง raw-token correctness และ Q2 CPU/GPU exactness เป็นข้อกำหนดเดิม ไม่มีการผ่อนเกณฑ์ใน session นี้

## Baseline tests ที่รันจริง

รันจาก `D:\Github\Strata` บน source เดิม ไม่มี candidate implementation และไม่มีการโหลด engine/model:

```powershell
python -B -m pytest -q -p no:cacheprovider tests/xeno/test_serve_protocol.py serve/test_session_save_retry.py serve/test_slots.py serve/test_timeline.py tests/xeno/test_timeline.py
```

1. ใน sandbox: `37 failed, 35 passed, 1 skipped, 2 warnings, 12 subtests passed in 10.92s` โดย HTTP tests ที่เชื่อม `127.0.0.1` พบ `PermissionError: [WinError 10013]` จาก `socket.connect` ดู raw log ซึ่งมี subfailures ด้วยเป็นหลัก
2. อ่าน test setup ยืนยันว่าใช้ mock engine และ HTTP บน loopback จากนั้นรันคำสั่งเดิมด้วยสิทธิ์ network ของคำสั่งนั้น: **`65 passed, 1 skipped, 19 subtests passed in 11.69s`**, exit code 0
3. test ที่ skip คือ symlink ซึ่งต้องมี privilege บน Windows (`serve/test_slots.py:216`) ไม่ได้เปลี่ยน privilege เพื่อบังคับให้รัน

Logs:

- `%TEMP%\strata-qwenputer-discovery-20261028\baseline-tests.log` — sandbox run พร้อม traceback
- `%TEMP%\strata-qwenputer-discovery-20261028\baseline-tests-localhost.log` — run ที่ผ่านหลังอนุญาต loopback

ผลนี้ยืนยันเฉพาะชุดทดสอบเดิมที่ระบุ ไม่ยืนยัน Native Qwenputer, full test suite, GPU kernels, token parity หรือความเร็ว

## Design checkpoint สำหรับผู้ใช้พิจารณา

ส่วนนี้บันทึก checkpoint ที่เสนอไปก่อนอนุมัติ ไม่ใช่ PRD ตาม T4 สถานะล่าสุดคือผู้ใช้อนุมัติ runtime ที่สรุปไว้ข้างต้นแล้ว

| ข้อกำหนด | สิ่งที่ข้อเสนอจะรักษา |
|---|---|
| Native ก่อน | เชื่อมกับ Flash-Next ที่มีอยู่ ไม่สร้าง Universal adapters ในระยะแรก |
| วงจรจริง | ประเมิน sparse FFN/GDN ที่ compiler สร้าง พร้อม machine state ไม่เรียก Python calculator หรือใช้ RISC-V emulator เป็นตัวตอบจริง |
| Offload และ default | weights ผู้เชี่ยวชาญกับ CPU/RAM/NVMe path เดิมคงอยู่; feature opt-in และมี same-session feature-off token parity |
| สถานะ | circuit มี RAM/PC/register/bridge state แยกต่อ session; trace ไม่ขยับ language KV/GDN/positions; text ที่วงจรสร้างจึงค่อยกลับไปยัง language path |
| T4 | อ่าน skills จริงแล้ว แต่ขั้นตอน user-invocation-only ยัง pending; ไม่สร้าง PRD/issues แทนเอง |
| การทดสอบจริง | เทียบ trace กับ exact oracle; เข้า/ออก/resume, snapshot, cancel/step bound; รัน CUDA บน 5060 Ti พร้อมวัด peak RAM/VRAM; timeline อธิบายเวลาและ ABBA ปิด profiler วัดผลพร้อม counters |

ทางเลือกที่พิจารณาก่อนผู้ใช้อนุมัติ runtime:

1. **แนะนำ: circuit weights เป็น artifact แยก แต่ประเมินภายใน C++/CUDA forward/decode ของ Strata** — รักษา base weights และ expert pack เดิมได้; ไม่โหลด Qwen3.5 merged model; ต้องเพิ่ม loader และ sparse circuit executor โดย artifact นี้ยังเป็น weights ของวงจรจริง ข้อแตกต่างที่ต้องยอมรับคือไม่ได้รวมอยู่ในไฟล์ weights ของโมเดลหลักไฟล์เดียว
2. **รวม chip layers และขยาย residual เข้า Flash-Next pack โดยตรง** — ใกล้รูปแบบ merged checkpoint ของ Qwenputer มากกว่า แต่ต้องเปลี่ยนสัญญา dimensions, loader, kernels, head และ checkpoints หลายจุด; correctness ของ Flash-Next MoE/HC ยังไม่มีการพิสูจน์จาก source ของ Qwenputer ซึ่งเป็น Qwen3.5 dense

ทั้งสองเป็นงาน Native ได้หากประเมิน compiled circuit จริง ต่อมาผู้ใช้อนุมัติ native runtime พร้อม artifact แยกแล้ว จึงไม่ต้องเลือกซ้ำ ประโยชน์ต่องาน coding/3D ยังเป็นสมมติฐาน ไม่ใช่เกณฑ์ที่พิสูจน์แล้ว

## งานที่ยังค้าง

- **Denied:** อ่านแท็บบทสนทนาล่าสุด เป้าหมายคือแท็บ `721139517` บน browser `3`; ถูก policy ปฏิเสธก่อนคืนเนื้อหา ไม่มีผลลัพธ์ไม่แน่นอนและไม่ได้ retry
- **Resolved for working scope:** ได้รับข้อความ brief จาก parent แล้ว ไม่ต้องรอลิงก์แชร์สำหรับขอบเขตที่ระบุข้างต้น
- **Approved:** native sparse circuit ภายใน engine บน primary GPU พร้อม artifact แยกและ GPU trace loop ผู้ใช้ตอบ “เอาตามนั้น” ต่อข้อเสนอนี้ ยังไม่มี implementation
- **Pending:** ขั้นตอน user-invocation-only ของ T4 ตามที่กล่าวด้านบน ต้องดูผลลัพธ์ที่ผู้ใช้มีแล้วก่อนเริ่มใหม่ ห้ามทำซ้ำหรือสร้างแทนโดยสมมติ
- ยืนยันการติดตั้งจริงด้วยการอ่าน frontmatter ที่ `C:\Users\xenod\.agents\skills\{grill-me,grill-with-docs,to-prd,to-issues}\SKILL.md` ทั้งสี่ระบุ `disable-model-invocation: true`; ไม่ใช่ข้อสันนิษฐานจากการไม่เห็นรายชื่อ skills
- ก่อนรัน exporter/compiler หรือ firmware จาก third-party source ต้องขออนุมัติการรันตามข้อจำกัดของ task; ยังไม่ได้ขอหรือพยายามรัน ไม่มีการปฏิเสธหรือผลลัพธ์ไม่แน่นอนของการรันดังกล่าว
- ยังไม่มี implementation, build candidate, firmware execution, GPU test, greedy raw-token parity, timeline ของ Qwenputer หรือ profiler-off ABBA ตัวเลขความเร็วของ upstream ไม่ใช่ผลวัดบนเครื่องนี้
- ยังไม่ได้ติดตั้ง dependencies, ดาวน์โหลด weights, รัน third-party code, แก้ served profile หรือรบกวนบริการเดิม

## การตั้งค่า model และ Fast ที่ตรวจเพิ่ม

ผู้ใช้กำหนดให้ใช้ Fast เมื่อเลือก Sol/Terra/Luna หรือ model ที่ไม่ใช่ Astra โดย effort ยังเลือกตามความยากของงาน ไม่ได้อนุญาตให้เริ่ม implementation ก่อนการตัดสินใจที่ค้าง

- อ่าน `C:\Users\xenod\.codex\config.toml` ด้วย `tomllib` ได้ค่า default: `model = "gpt-6.1-sol"`, `model_reasoning_effort = "medium"`, `service_tier = "priority"`; ไม่มี `[features].fast_mode` ที่ตั้งไว้อย่างชัดเจน
- ค่าเหล่านี้เป็น local defaults ไม่ใช่หลักฐานของ model/effort/tier ที่ให้บริการ turn นี้จริง
- `read_thread` ของ task `01a18364-ec23-70f4-b71b-151779c8c01b` คืน `hostId = "durable"` แต่ไม่คืน model หรือ service tier และไม่พบ local rollout ของ task นี้ จึงยังยืนยันค่าที่ใช้งานจริงไม่ได้
- tools สำหรับสร้าง/ส่งงานที่มีใน session มี model/effort controls แต่ไม่มี service-tier argument หรือ control สำหรับเปลี่ยน tier ของ turn ที่กำลังรัน ไม่ได้สร้าง task เพิ่มเพื่อเปลี่ยนการตั้งค่า
- [Configuration Reference](https://learn.chatgpt.com/docs/config-file/config-reference) ระบุว่า `fast` map เป็น request value `priority`; [Speed](https://learn.chatgpt.com/docs/agent-configuration/speed) ระบุการเปิดผ่าน `/fast` หรือ `service_tier = "fast"` พร้อม feature flag ทั้งนี้ไม่ใช่หลักฐานว่า turn นี้เปิด Fast แล้ว
- ไม่ได้แก้ global config ซึ่งมีผลกว้างกว่าคำสั่งแบบยกเว้น Astra และไม่ได้เปลี่ยน model, effort หรือ tier ของ task นี้

## เพิ่มเติม: เป้าหมาย latency ต่ำที่สุดและ native จริง

ผู้ใช้ยืนยันเป้าหมายว่าอยากได้ latency ต่ำที่สุดและให้ model ใช้ได้แบบ native จริง จากนั้นอนุมัติข้อเสนอ runtime ด้านล่างด้วยข้อความ “เอาตามนั้น” แล้ว ยังไม่มีผล benchmark ของ implementation

### เปรียบเทียบเส้นทางที่รัน ไม่ใช่จำนวนไฟล์

artifact แยกกับ pack รวมสามารถถูก loader แปลงเป็น sparse arrays, token maps และ GPU graph ชุดเดียวกันได้ หาก bytes, layout, placement และ graph หลังโหลดเหมือนกัน จำนวนไฟล์ต้นทางไม่เพิ่ม kernel, transfer หรือ host callback ใน request path นี่เป็นข้อสรุปแบบมีเงื่อนไขเรื่องเส้นทางรัน ไม่ใช่ผลวัดว่าตัว loader สองแบบที่ยังไม่ได้เขียนเร็วเท่ากัน

การรวมไฟล์อย่างเดียวจึงไม่มีหลักฐานว่าช่วย latency ส่วนการนำ merged dense Qwen3.5 มาโหลดตาม `RvChat` เป็นคนละ runtime: `chat.py:111-112` โหลดทั้งโมเดลและใช้ dimensions ของ merged model การพอร์ตแบบนั้นตรง ๆ ไม่ได้ทำให้เป็น runtime ที่สั้นที่สุดสำหรับ Flash-Next

จุดที่ source ปัจจุบันแสดงว่ามี host mediation:

- `chat.py:54-70,92-95`: graph ครอบเพียง circuit step ของหนึ่ง token; Python จัด embedding และ replay ต่อ token
- `chat.py:172-184`: `bool((code != 0).any())` และ `.tolist()` นำผลจาก GPU มาเลือก token บน host
- `chat.py:393-426`: Python เลือกว่าจะเดิน chip trace หรือ Qwen และขยับตำแหน่ง text
- `tritonexec.py:272-288`: แต่ละ circuit sublayer มี FFN/head kernel และ apply kernel การ capture ลดการเรียก host API แต่ไม่ได้ทำให้ kernels เหล่านี้หายไป

ยังไม่ได้จับ timeline ของ upstream บนเครื่องนี้ จึงยังไม่รู้ milliseconds ที่แต่ละจุดใช้ และไม่อ้างจำนวน ms หรืออัตราเร็วที่จะได้คืน

### ข้อเสนอที่แนะนำให้อนุมัติ

**Native C++/CUDA sparse circuit ที่อยู่บน primary GPU และเป็นส่วนหนึ่งของ decode state machine ของ Strata** โดยเก็บ weights เป็น artifact แยกเพื่อไม่ต้องเปลี่ยน base model/pack แต่หลังโหลดไม่มี Python/tool service/RISC-V emulator อยู่บนเส้นทางคำนวณจริง

1. โหลด compiled FFN/GDN weights, token feature maps และ firmware snapshot ครั้งเดียว จัด state/buffers และ warm graph ก่อน request; active circuit ไม่ย้ายไป NVMe ระหว่าง burst ใช้ memory budget ที่คำนวณจาก artifact จริงก่อนจัด expert cache
2. circuit bridge รับ text ที่ commit จริงรวมถึง prompt; การตรวจ takeover และ circuit state ต้องอยู่ภายใต้ขอบเขต accepted prefix ของ speculative verifier เดียวกัน ห้าม commit machine state ของ rejected drafts หรือ text ที่เลย takeover และไม่แก้ latency ด้วยการปิด MTP ตลอดการใช้งาน plugin
3. ในช่วงที่วงจรเป็นเจ้าของการสร้าง token ให้ GPU เลือก token ถัดไปและเดิน trace เองใน graph ที่มีเงื่อนไขและเพดานการทำงาน หยุดเมื่อคืนสิทธิ์, output buffer เต็ม, มี cancel หรือถึง step limit ข้อเสนอใช้ CUDA conditional WHILE เป็นกลไกที่ต้องทดสอบบน runtime จริงก่อนยืนยันว่ารองรับ
4. trace ขยับเฉพาะ circuit state; language KV/GDN/PLE และ text positions ไม่ขยับ ข้อความที่วงจรสร้างต้องถูกป้อนกลับ language path ตามลำดับ text เดิม แล้วจึงให้ model สร้างต่อ
5. ใช้ same-device buffers/stream dependencies สำหรับเข้า/ออก circuit; อ่านผลสถานะร่วมกับ completion ที่ decoder ต้องอ่านอยู่แล้วเมื่อทำได้ หลีกเลี่ยงการอ่าน token กลับ host ทุก instruction สิ่งที่ host ยังทำคือรับ request, จัด session/cancel/limits, ส่ง status/stream และจัดผลตอบกลับ
6. การ replay text ที่วงจรสร้างเป็นชุดอาจใช้ verifier window เพื่อ amortize การอ่าน dense weights แต่ทำได้ต่อเมื่อพิสูจน์ state/token equivalence กับ single-token path แล้ว การใช้ batched prefill ธรรมดาอาจเปลี่ยน numerics จึงยังไม่เลือกแทนการทดสอบ
7. เริ่มจาก kernels ที่รักษาสมการและ rounding ของ reference แล้วใช้ timeline ตัดสินการ fuse/reorder ภายหลัง ไม่ตั้งสมมติฐานว่าการ fuse ทุกอย่างจะเร็วกว่า

[CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-graph-nodes) รองรับ conditional loops ที่ GPU ประเมินเงื่อนไขเอง; body ต้องอยู่บน device เดียวและมีข้อจำกัดชนิด nodes จึงควรเก็บ conditional trace graph เป็น top-level graph บน primary ไม่สมมติว่าใส่เป็น child ของ verifier graph ได้ทุกกรณี

### ข้อกำหนดที่ต้องพิสูจน์ก่อนถือว่า native stage ผ่าน

- **คณิตศาสตร์:** same-input trace และทุก circuit dimension เทียบ compiler IR/reference; RISC-V emulator ใช้เป็น oracle ใน test เท่านั้น ผลตอบจริงต้องมาจาก sparse FFN/GDN kernels
- **Exactness:** รักษา sparse reduction, decay/write/read, convolution history, beta, normalization และ gate จริง ตัวอย่าง `tritonexec.py:143-148` ใช้ `sqrt_rn`/`div_rn`; คำว่า FP32 อย่างเดียวไม่พิสูจน์ exactness และไม่อนุญาตให้เปิด fast-math/quantize วงจรโดยไม่มี gate
- **Speculation:** takeover กลาง window, rejected drafts, MTP catch-up, text ที่ตามหลัง takeover และ output FIFO ต้องไม่ทำให้ machine/language state หลุดกัน
- **Session:** snapshot ครอบคลุม RAM/register/PC, lag histories, bridge state, pending output และ text position; restore/cancel/step limit ไม่คืน state บางส่วนแล้วรันต่อ
- **API/UI:** `serve/frontend.py:832` และ `serve/server.py:4328` มี parser ของ external tool calls ต้องแยก native-owned calls ออกจาก external dispatch อย่างชัดเจน; `serve/ui/src/lib/status.ts:37` ต้องบอกสถานะคำนวณตั้งแต่เริ่ม burst
- **ทรัพยากร:** memory ของ circuit ไม่ใช่แค่จำนวน bytes ของ firmware `tritonexec.py:240-241` จัด state เป็น `[batch,V,128,128]` กับ history `[batch,3,D]` ต่อ head group ต้องวัด peak และผลของ expert slots ที่ลดลงจริง การ reserve VRAM อาจเพิ่ม latency ของ language decode และต้องรายงานถ้าเกิด
- **ค่าเดิม:** feature-off ไม่โหลด circuit, ไม่ reserve circuit VRAM, ไม่เพิ่ม graph nodes และไม่เปลี่ยน prompt; gate ด้วย greedy raw-token parity ของ same-session baseline ตามกฎ repo
- **Latency:** timeline แยก bridge/entry/trace/return และ host waits ก่อนปรับ performance; profiler-off ABBA ที่ `--pool-priority 2 --process-priority 2` วัดทั้ง ordinary text overhead กับ arithmetic turn รวมเวลาเข้า/ออกและอ่านผล พร้อม same-run counters, exe hashes และ peak memory

ผู้ใช้อนุมัติ runtime นี้แล้ว ข้อดีด้าน latency ข้างต้นยังเป็น design hypothesis จนกว่าจะมี native implementation และผลวัดบน D2x จริง ขั้นตอน T4 แบบ user-invocation-only และ approval สำหรับรัน third-party compiler/firmware ยัง pending ตามที่บันทึกไว้

## ขั้นตอนถัดไปหลังอนุมัติ architecture

ขั้นแรกตาม `t4-dev-workflow` คือผู้ใช้เรียก `/grill-me` ซึ่ง agent เรียกเองไม่ได้เพราะ `C:\Users\xenod\.agents\skills\grill-me\SKILL.md` ระบุ `disable-model-invocation: true` และ T4 ระบุว่า “Do not improvise the output.” การอนุมัติ architecture ไม่ใช่การยกเว้นกฎนี้ ไม่เริ่มคำถามเลือก architecture ใหม่; ตรวจเฉพาะช่องว่างในแบบที่อนุมัติแล้ว จากนั้นจึงไป `/grill-with-docs`, survey, `/to-prd`, `/to-issues` ตามลำดับ การ publish tracker ยังต้องแยกขออนุมัติก่อนคำสั่ง publish จริง

ข้อความที่ผู้ใช้สามารถส่งเพื่อเรียกขั้นแรกใน task เดิม:

```text
ใช้ skill grill-me ตรวจแบบ Native Qwenputer ที่อนุมัติแล้วใน docs/reports/2026-10-28-native-qwenputer-discovery.md โดยยึด native C++/CUDA บน GPU หลัก, artifact แยก, GPU trace loop และเกณฑ์ทดสอบที่ตกลงไว้
```

ใน Codex CLI ยังมีวิธีเลือก `/skills` แล้วเลือก `grill-me` ตาม [Developer commands](https://learn.chatgpt.com/docs/developer-commands#use-skills-with-skills) ไม่ต้องสร้าง task ใหม่ ไม่ได้ตรวจเมนูของ desktop task ด้วย UI

สำรวจ build เพิ่มโดยไม่ build: `build/CMakeCache.txt` ชี้ CUDA 13.3 `nvcc.exe`, MSVC 14.44 และ architecture `120` ที่ `D:/Github/Strata`; local `driver_types.h` มี `cudaGraphNodeTypeConditional` และ `cudaGraphCondTypeWhile` จริง นี่ยืนยันเฉพาะ headers/config เดิม ไม่ยืนยันว่า graph instantiate/run ผ่านบน hardware แล้ว `hipcc`, `icx`, `dpcpp` ไม่พบผ่าน `Get-Command` ใน PATH ปัจจุบัน ยังสรุปว่าไม่ได้ติดตั้งทั้งระบบไม่ได้
