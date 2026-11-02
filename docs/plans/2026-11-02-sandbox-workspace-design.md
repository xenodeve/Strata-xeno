# Sandbox ที่ทำงานกับ project จริงได้: ข้อเสนอการออกแบบ

สถานะ: ข้อเสนอ ยังไม่ได้ build. ติดตามที่ #287. ต่อยอดจาก C1 และ C2 ใน `docs/plans/2026-10-29-igc-primary-prd.md`.

## สรุป

- **ทำได้** โดยเพิ่มชั้นความสามารถ 3 ชั้นต่อจากตัวแปลภาษาเดิม: Python จริงพร้อม library, สำเนา project ที่แก้และรัน test ได้, และ VM สำหรับงานเสี่ยงสูง
- **ของจริงปลอดภัยเพราะระบบปฏิบัติการกั้น** ไม่ใช่เพราะโมเดลเชื่อฟัง โค้ดใน sandbox ไม่มีสิทธิ์เขียน project จริง อ่านความลับไม่ได้ และออก network ไม่ได้
- **การนำขึ้นของจริงใช้ diff ที่ทดสอบแล้ว** ผ่าน permission กับ checkpoint เดิม โมเดลไม่ต้องพิมพ์โค้ดซ้ำ จึงไม่มีช่องให้พิมพ์ต่างจากที่ทดสอบ
- **ความเสี่ยงที่เหลือคือ test ผ่านแต่โค้ดยังผิด** และโมเดลเรียก Edit ตรงๆ โดยไม่ทดสอบ ข้อหลังปิดด้วย gate ได้ ข้อแรกปิดไม่ได้
- **ลำดับ:** วัดก่อนว่า library เปลี่ยนคำตอบไหมบน A100 ที่ทิ้งได้ แล้วค่อยสร้างตัวกั้นบน Windows

## 1. เป้าหมาย และสิ่งที่รับประกันไม่ได้

ปัญหาที่ต้องแก้: โมเดลแก้ project จริงโดยยังไม่รู้ว่าสิ่งที่แก้ใช้ได้ไหม พอพังก็ต้องแก้ตามในของจริง ถ้าลองในสำเนาก่อน พังก็แค่เปลี่ยนวิธี ของจริงไม่กระทบ

**กฎที่ระบบต้องบังคับได้เสมอ**

| รหัส | กฎ | ผู้บังคับ |
|---|---|---|
| I1 | โค้ดใน sandbox เขียน project จริงไม่ได้ | ระบบปฏิบัติการ |
| I2 | โค้ดใน sandbox อ่านความลับไม่ได้ และ environment ถูกล้าง | ไม่ map ไฟล์เข้ามา ร่วมกับ deny list |
| I3 | ไม่มี network เว้นแต่ผู้ใช้เปิดเป็นรายปลายทาง | ระบบปฏิบัติการ |
| I4 | ตัวกั้นเริ่มไม่ได้ ฟีเจอร์ปิด ไม่รันแบบไม่กั้น | โค้ดของเรา |
| I5 | ทุกผลลัพธ์บอกว่าเกิดในสำเนา ไม่ใช่ของจริง | ข้อความผลลัพธ์ |
| I6 | ของจริงเปลี่ยนผ่าน permission และ checkpoint เดิมเท่านั้น | `serve/permissions.py`, `serve/checkpoints.py` |
| I7 | เวลา หน่วยความจำ ดิสก์ process และ output มีเพดานทุกอย่าง | Job Object และเพดานของเรา |
| I8 | ปิดเป็นค่าเริ่มต้น เส้นทางเดิม byte-identical | กฎ repo |
| I9 | ทุกขั้นมีสถานะเป็นคำบนหน้าเว็บ | `serve/ui/src/lib/status.ts` |

**ที่รับประกันไม่ได้**

- **Test ผ่านไม่ได้แปลว่าถูก** ข้อ 3781 ของ LiveCodeBench ผ่านตัวอย่างแต่ช้าเกินเวลาจริง
- **Test ที่โมเดลเขียนเองอาจพาไปผิดทาง** ExecCritic รายงานว่าเพิ่มจาก 81.5 % เป็น 85.7 % บน SWE-bench Verified แต่บน SWE-bench Pro เพิ่มแค่ 0.7 จุด และเตือนว่า test ที่ไม่น่าเชื่อถืออาจชี้ไปทางผิด
- **โมเดลยังเรียก Edit ตรงได้** ถ้าไม่เปิด gate ในข้อ 4.7
- **Sandbox ไม่ทำให้คำสั่งของโมเดลน่าเชื่อถือ** เอกสาร MXC เขียนตรงๆ ว่า containment จำกัดสิ่งที่ workload ทำได้ แต่ไม่ได้ทำให้สิ่งที่มันสร้างน่าเชื่อถือ

คำว่า "ไม่มีทาง" จึงใช้ได้กับ I1-I3 เท่านั้น ส่วนคำตอบที่ผิดลดได้ แต่ไม่หมด

## 2. คนอื่นทำอะไรไปแล้ว

| ระบบ | ทำอะไร | สิ่งที่เรารับมา |
|---|---|---|
| [Cursor shadow workspace](https://cursor.com/blog/shadow-workspace) | ลองแก้ในหน้าต่างซ่อนแล้วส่ง lint กลับ ปี 2024 ยังไม่ได้ทำส่วนรันโค้ด | ปฏิเสธ hardlink เพราะการเขียนอาจทะลุถึงไฟล์จริง และ `clonefile` ใช้ 45 วินาที ปลายทางที่เขาอยากได้คือ folder proxy ระดับ kernel |
| [Fork, Explore, Commit](https://arxiv.org/abs/2602.08199) | งานวิจัย branch context: fork, explore, commit หรือ abort คนแรกที่ commit ชนะ | สร้าง branch ราว 300 µs ไม่ขึ้นกับจำนวนไฟล์ แต่เป็น FUSE บน Linux และกันเฉพาะไฟล์ ไม่ย้อน network |
| [Claude Code](https://code.claude.com/docs/en/sandboxing) | bubblewrap หรือ Seatbelt, network ผ่าน proxy | ค่าเริ่มต้น fail-open และอ่าน `~/.ssh` ได้ บน native Windows ไม่กั้น เราต้องเข้มกว่านี้ (I2, I4) |
| [Codex บน Windows](https://codex.danielvaughan.com/2026/05/14/codex-cli-windows-sandbox-engineering-restricted-tokens-acls-elevated-architecture/) | Restricted token, ACL และ firewall (แหล่งรอง) | การตั้ง ACL ก่อนทุกคำสั่งเพิ่ม latency ชัดเจน และมีปัญหากับ antivirus กับ group policy |
| [Microsoft MXC](https://learn.microsoft.com/en-us/windows/apps/develop/mxc/) | Policy JSON กำหนด read-only, read-write, denied path และ network deny | ตรงกับ I1-I3 มากที่สุด MIT, มี CLI `wxc-exec.exe`, ไม่มี SDK สำหรับ C++ หรือ Python, ไม่มีตัวเลข latency |
| [Dev Drive](https://www.elevenforum.com/t/microsoft-improves-refs-file-system-to-block-cloning-on-dev-drive-in-windows-11-24h2.29068/latest) | ReFS block cloning | คัดลอกแบบ copy-on-write ได้ แต่ต้องมี volume แยก เครื่องนี้ C: กับ D: เป็น NTFS |

**เหตุการณ์จริงที่กำหนดกฎ I2 และ I3:** กรณี [PocketOS](https://www.tomshardware.com/tech-industry/artificial-intelligence/claude-powered-ai-coding-agent-deletes-entire-company-database-in-9-seconds-backups-zapped-after-cursor-tool-powered-by-anthropics-claude-goes-rogue) agent เจอ token ในไฟล์ที่ไม่เกี่ยวกับงาน แล้วใช้ `curl` ลบ volume production ถ้า token อ่านไม่ได้หรือ network ออกไม่ได้ เหตุนี้เกิดไม่ได้ กรณี [Replit](https://oecd.ai/en/incidents/2025-07-19-1eb1) เกิดจากการแตะของจริงโดยตรง ซึ่งเป็นเหตุผลของ I6

**หลักฐานว่าวิธีนี้ช่วย:** Cursor เรียก lint feedback ว่าเป็นหนึ่งในวิธีที่ได้ผลที่สุดเมื่อไม่เปลี่ยนโมเดล ส่วนงานวิจัย SWE-bench ข้างต้นบอกว่า feedback จากการรันช่วยจริงแต่ไม่สม่ำเสมอ เราจึงต้องวัดเอง

## 3. ตัวเลขที่วัดบนเครื่องนี้ (2026-11-02)

| รายการ | ค่า |
|---|---|
| Block ในตัวแปลภาษาเดิม | ราว 1.3 ms |
| เปิด Python เปล่า | 51 ms |
| `import numpy` / `import sympy` | 153 / 494 ms |
| `git status` บน repo นี้ | 35 ms |
| คัดลอก 4,569 ไฟล์ 53 MB (ไฟล์ติดตามที่เล็กกว่า 2 MB) ลง C: | 4.1 วินาที |
| สแกนสำเนาหาการเปลี่ยนแปลง | 246 ms |
| ลบสำเนา | 0.6 วินาที |
| Library ในเครื่อง | sympy, numpy, mpmath, networkx (ไม่มี scipy, pandas, z3) |
| ความต้องการใน 339 block ที่เก็บไว้ | sympy 21, sys 4, random 4, numpy 2 |
| Windows | 10.0.26300.9711 ผ่านเกณฑ์ MXC ที่ระบุ 26300.9550 สำหรับ 26H2 |
| พื้นที่ว่าง | C: 41 GiB, D: 5.8 GiB |
| อ่านไฟล์ติดตามทั้งหมดเข้าหน่วยความจำ | 535 ms, 53 MB |
| ค้นคำตรงตัวในหน่วยความจำ (Python) | 24 ms ทั้ง repo |
| Parse ไฟล์ Python 1,053 ไฟล์ ได้ 7,842 symbol (`ast`) | 2.4 วินาที |
| อ่าน `serve/agent.py` ทั้งไฟล์ เทียบ outline บวกหนึ่งฟังก์ชัน | 11,325 เทียบ 1,021 token |
| อ่าน `serve/permissions.py` เทียบแบบเดียวกัน | 7,071 เทียบ 853 token |
| อ่าน `tools/igc/external_bench.py` เทียบแบบเดียวกัน | 4,571 เทียบ 555 token |
| `igc_sandbox.hpp` ทั้งไฟล์ | 135,477 token |
| เครื่องมือในเครื่อง | serena, uvx, `tree_sitter` ของ Python มี ส่วน rg, ctags, clangd, pyright ไม่มี |

## 4. โครงสร้าง

```text
 โมเดล ──┬─ block ระหว่างคิด ─ engine ─┐
         └─ tool (MCP หรือ agent__) ───┤
                                       ▼
                        serve/sandbox: ตัวจัดการ session
                  ┌──────────┬───────────┴──┬────────────┐
                  ▼          ▼              ▼            ▼
              T0 ตัวแปล   T1 Python จริง  T2 Workspace  T3 VM
              ในengine    ไม่เห็นไฟล์      สำเนา project  งานเสี่ยงสูง
                             └─────── ตัวกั้น OS ───────┘
                                       │
                                  diff + ผล test
                                       ▼
                 serve/permissions → serve/checkpoints → project จริง
```

### 4.1 ชั้นความสามารถ

| ชั้น | รันอะไร | เห็นอะไร | เวลาเริ่ม | ใช้เมื่อ |
|---|---|---|---|---|
| T0 ตัวแปลภาษา | Python subset | ไม่มีไฟล์ | 1 ms | คำนวณล้วน |
| T1 Python จริง | CPython กับ library ใน allowlist | โฟลเดอร์ชั่วคราวของตัวเอง | ต้องอุ่นไว้ | sympy, numpy, โค้ดเต็มภาษา |
| T2 Workspace | T1 บวกคำสั่ง test และ build ใน allowlist | สำเนา project | สร้างสำเนา 4 วินาทีต่อแชต | อ่าน แก้ รัน test |
| T3 VM | อะไรก็ได้ | ที่ map เข้าไป | วินาทีถึงนาที | project ไม่น่าไว้ใจ, service, browser |

### 4.2 การยกระดับ

ใช้การปฏิเสธของตัวแปลภาษาเป็นตัวเลือกชั้น โดยไม่เดาจากข้อความ T0 ตอบ `Unsupported` หรือ `ModuleNotFoundError` ที่ระบุชื่อได้แน่นอนอยู่แล้ว และไม่มีผลข้างเคียง จึงรันซ้ำใน T1 ได้ปลอดภัย ราคาของการลอง T0 คือ 1 ms ถ้าโค้ดเรียกไฟล์หรือคำสั่ง ก็ยกเป็น T2 เมื่อแชตนั้นผูก project แล้วเท่านั้น

สถานะระหว่าง block: T0 เล่นโค้ดเดิมซ้ำเหมือนเดิม ส่วน T1 และ T2 ใช้ namespace ค้างต่อ turn เพราะ `import sympy` ซ้ำทุก block แพงเกินไป Block ที่ล้มอาจทิ้งสถานะครึ่งทาง ผลลัพธ์จึงต้องบอกตรงๆ

### 4.3 ตัวกั้น

ซ่อนไว้หลัง interface เดียว เพื่อเปลี่ยน backend ได้โดยไม่แตะส่วนอื่น

```text
Isolation.start(policy) -> Box
policy: readonly[], readwrite[], denied[], network=deny, memory, time, processes, disk
Box.run(cmd, stdin) / Box.kill() / Box.usage()
```

| Backend | ที่ใช้ | ข้อดี | ข้อเสีย |
|---|---|---|---|
| MXC `processcontainer` | Windows, ตัวแรกที่จะลอง | Policy ตรงกับ I1-I3, MIT, เป็นของ Microsoft | เรียกผ่าน CLI, ยังไม่มีตัวเลข latency, เพิ่ม dependency |
| AppContainer กับ Job Object | Windows, ตัวสำรอง | ไม่เพิ่ม dependency | ต้องดูแล ACL เอง, Python ใน AppData อ่านไม่ได้จึงต้องมี Python ของ sandbox เอง, Codex เลือกไม่ใช้เพราะออกแบบมาสำหรับแอปเดี่ยว |
| bubblewrap กับ overlayfs | Linux | แบบเดียวกับ Claude Code และ overlay มีในตัว | ใช้ได้เฉพาะ Linux |
| Windows Sandbox หรือ WSL2 | T3 | กั้นด้วย VM | ช้า และ Windows Sandbox อาจเปิดหน้าต่างซึ่งขัดกฎ repo ต้องตรวจก่อน |
| `none` | A100 ที่ทิ้งได้เท่านั้น | ใช้วัดผลใน S1 | ต้องสั่งด้วย flag ที่บอกชัดว่าไม่มีตัวกั้น และ refuse บน Windows |

Job Object รับหน้าที่เรื่องหน่วยความจำ จำนวน process และการฆ่าทั้งต้นเมื่อจบ เพราะ Job Object ไม่มีกฎไฟล์หรือ network

### 4.4 มุมมอง project

เป้าหมายคือ base ที่อ่านอย่างเดียว บวก delta ที่เขียนได้ NTFS ไม่มี overlay จึงทำเป็นขั้น

| ขั้น | วิธี | ค่าใช้จ่าย | เมื่อไหร่ |
|---|---|---|---|
| 1 | คัดลอกไฟล์ติดตาม ไฟล์ที่แก้ค้าง และไฟล์ใหม่ที่ไม่ถูก ignore ลง temp พร้อมชี้โฟลเดอร์ใหญ่แบบ read-only | 4.1 วินาทีต่อแชตบน repo นี้ ทำเบื้องหลังตอนเริ่มแชต | เริ่มที่นี่ |
| 2 | Projected File System ให้โฟลเดอร์ดูเหมือนของจริงและดึงไฟล์เมื่อถูกอ่าน | สร้างเกือบทันที แต่ต้องเปิด Windows feature และยังไม่ได้ทดสอบกับตัวกั้น | เมื่อ repo ใหญ่จนคัดลอกช้า |
| 3 | Dev Drive block clone | ต้องสร้าง volume ReFS | ตัวเลือกของผู้ใช้ |

กฎของการคัดลอก:
- **ตัดออกเสมอ:** `.git`, `.env`, กุญแจ และทุกอย่างที่ `is_secret` ใน `serve/permissions.py` จับได้
- **ไม่ map:** ไฟล์ขนาดใหญ่อย่าง model GGUF, `build-*`, `node_modules` ถ้าไม่จำเป็น
- **ห้าม hardlink:** การเขียนลงไฟล์เดิมจะทะลุถึงของจริง
- **อ่านจาก working tree ไม่ใช่ HEAD:** โมเดลต้องเห็นสิ่งที่ผู้ใช้แก้ค้างไว้

**ความสดของข้อมูล:** ทุกครั้งก่อนรัน เช็ค `git status` (35 ms) ของจริง ถ้าไฟล์ที่ delta ไม่ได้แตะเปลี่ยน ก็อัปเดตเงียบๆ ถ้าไฟล์ที่ delta แก้อยู่เปลี่ยนด้วย ให้แจ้ง `[เปลี่ยนในของจริงด้วย: path]` แล้วรอโมเดลตัดสิน

### 4.5 ทางเข้า 2 ทาง

**ทางที่ 1: tool สร้างก่อน** ใช้ได้กับทุกโมเดลและทุก harness ผ่าน MCP

| Tool | หน้าที่ |
|---|---|
| `sandbox_run` | รัน Python หรือคำสั่ง |
| `sandbox_read`, `sandbox_grep` | อ่านและค้นในสำเนา |
| `sandbox_edit` | แก้ในสำเนา |
| `sandbox_test` | รัน test พร้อมเทียบกับ base |
| `sandbox_diff` | ดูสิ่งที่เปลี่ยน |
| `sandbox_apply` | ขอนำ diff ที่ทดสอบแล้วขึ้นของจริง |

**ทางที่ 2: ระหว่างคิด** block ใน reasoning ไม่ต้องจบ turn จุดนี้คือข้อได้เปรียบเฉพาะของ Strata

```text
engine → server : SBX <slot> <id> <bytes>\n<program>
server → engine : SBXR <id> <exit> <bytes>\n<ผลที่ตัดแล้ว>
```

Server เป็นคนรัน เพราะมี `Shell`, `permissions`, `hooks` และ project ของแชตอยู่แล้ว ส่วน engine แค่หยุดรอแล้วส่งผลเป็น forced token เหมือน T0 ถ้าแชตไม่ผูก project T0 ยังทำงานใน engine เหมือนเดิมและไม่มีการส่งออก ค่า GEN key `sandbox=<id>` ระบุ session และต้องลง `docs/BLUEPRINT.md`

**การผูก project:** แชตบนหน้าเว็บรู้ project อยู่แล้ว ส่วน API จาก Claude Code ส่งโฟลเดอร์มาแบบเชื่อถือไม่ได้ จึงต้องลงทะเบียนโฟลเดอร์ที่อนุญาตใน run config ก่อน โมเดลเลือกโฟลเดอร์เองไม่ได้

### 4.6 สิ่งที่โมเดลเห็น

โมเดลรู้เฉพาะสิ่งที่ถูกพิมพ์เข้า context และทุกไบต์ที่พิมพ์เป็น forced token จึงใช้เพดาน 2 ชั้น

- **ต่อ block:** 1,024 ไบต์เหมือนเดิม ส่วนเกินเก็บเป็นไฟล์ในสำเนา แล้วบอกวิธีอ่านต่อ
- **ต่อ turn:** มีงบรวม หมดแล้วตอบ `[หมดงบการอ่าน]`

ผลลัพธ์เป็นรูปแบบคงที่:

```text
[sandbox: สำเนา ไม่ใช่ของจริง] exit 1, 2.3 s
tests: 41 ผ่าน, 1 ตก (ก่อนแก้ 42 ผ่าน) ← พังเพราะการแก้นี้
FAILED tests/test_x.py::test_y  assert 3 == 1
แก้แล้ว 2 ไฟล์ในสำเนา ยังไม่แตะ project จริง
```

บรรทัด "ก่อนแก้" สำคัญที่สุด เพราะบอกว่า test ตกเพราะการแก้นี้หรือตกอยู่แล้ว Test ตกอยู่ก่อนเป็นต้นเหตุหลักของการหลงทาง

**กฎปิด thinking:** ใน T2 ไม่ปิดอัตโนมัติหลัง print เพราะผลเทียบคู่บน A100 แสดงว่าการปิดเร็วทำให้เสียคำตอบ และงานแก้โค้ดต้องวนหลายรอบ ใช้แค่ตัวกัน loop กับงบรวม

### 4.7 การนำขึ้นของจริง

```text
sandbox_test ผ่าน → บันทึก {diff hash, ไฟล์, คำสั่ง test, ผล, base}
       │
sandbox_apply → ตรวจ base ยังตรงไหม → ask ผ่าน permissions (แสดง diff และผล test)
       │
checkpoint → เขียนของจริงทีละไฟล์แบบ atomic → รัน test ซ้ำบนของจริงถ้าผู้ใช้เปิดไว้
```

โมเดลไม่ต้องพิมพ์โค้ดซ้ำ จึงตัดความผิดพลาดตอนคัดลอกและประหยัด token ด้วย ถ้า rewind ก็ใช้ `serve/checkpoints.py` เดิม

**Gate ที่เลือกเปิดได้:** ก่อน Edit หรือ Write จริงทำงาน เทียบผลลัพธ์กับ diff ที่ทดสอบแล้ว

| โหมด | ถ้าไม่ตรง |
|---|---|
| `off` | ทำงานเหมือนเดิม |
| `warn` (เสนอเป็นค่าเริ่มต้นเมื่อเปิดฟีเจอร์) | ป้ายบนหน้าเว็บว่า "ยังไม่ผ่านการทดสอบ" |
| `ask` | ถามผู้ใช้ |
| `deny` | ปฏิเสธและบอกโมเดลให้ทดสอบก่อน |

บน harness อื่นทำผ่าน hook ของเขา เช่น `PreToolUse` ของ Claude Code

Gate จะเตือนเพิ่มเมื่อ diff แก้ไฟล์ test พร้อมกับทำให้ test ผ่าน เพราะนั่นคือการแก้ข้อสอบ

### 4.8 สถานะและบันทึก

เพิ่มคำใน `agentStatus` ตามหลักของเว็บ เช่น *กำลังลองในสำเนา*, *กำลังรัน test ในสำเนา*, *กำลังเทียบกับก่อนแก้*, *รออนุมัติการนำขึ้นของจริง* และเก็บบันทึกทุกคำสั่งกับ diff ต่อแชตให้ผู้ใช้ตรวจย้อนหลังได้

### 4.9 การค้นและดูโครงสร้างแบบ native

แนวทางของ Serena คือให้โมเดลอ่านเป็น symbol ไม่ใช่เป็นไฟล์ แนวนี้เหมาะกับ sandbox มาก เพราะทุกไบต์ที่พิมพ์เป็น forced token ใน 3 ไฟล์ที่วัด outline บวกหนึ่งฟังก์ชันใช้ token เหลือ 9-12 % ของการอ่านทั้งไฟล์ ซึ่งเป็นตัวอย่างเล็กที่เลือกฟังก์ชันไว้เองไม่ใช่งานจริง ไฟล์ขนาด `igc_sandbox.hpp` อ่านทั้งไฟล์ไม่ได้อยู่แล้ว

Serena ใช้ language server ผ่าน LSP และทำงานเป็น MCP แยก process ตัว LSP ให้ความแม่นด้านอ้างอิงข้ามไฟล์ แต่เริ่มช้าและกินหน่วยความจำ จึงเสนอ 3 ชั้น เริ่มจากถูกไปแพง และใช้ชื่อคำสั่งตาม Serena เพื่อให้ความรู้ที่โมเดลมีอยู่ใช้ต่อได้

| ชั้น | เครื่องมือ | ตอบอะไร | ราคา | จำกัด |
|---|---|---|---|---|
| L1 ข้อความ | ตัวค้นของเราบนไฟล์ในหน่วยความจำ เคารพ `.gitignore` | ชื่อไฟล์, โครงโฟลเดอร์, คำ, regex | 24 ms บน repo นี้ | ไม่รู้โครงสร้างภาษา |
| L2 โครงสร้าง | tree-sitter ฝังเป็น library ภาษา C | outline, ตำแหน่ง symbol, เนื้อฟังก์ชัน, แก้ทั้งฟังก์ชัน | ต้องวัด ตัวเทียบคือ `ast` 2.4 วินาทีทั้ง repo | รู้แค่ syntax อ้างอิงข้ามไฟล์เป็นการเดาจากชื่อ |
| L3 ความหมาย | language server ใน box เช่น clangd, pyright | references, definition, rename, diagnostics | เริ่มช้า ต้องวัดต่อภาษา | ต้องติดตั้งต่อภาษา และ project ต้องตั้งค่าได้ |

**คำสั่งในสำเนา**

| คำสั่ง | ชั้น | เทียบกับ Serena |
|---|---|---|
| `tree(path, depth)`, `find_file(glob)` | L1 | `list_dir`, `find_file` |
| `search(pattern, glob)` | L1 | `search_for_pattern` |
| `outline(path)` | L2 | `get_symbols_overview` |
| `find_symbol(name, body=False)` | L2 | `find_symbol` |
| `references(symbol)` | L3 แล้วตกมา L1 พร้อมบอกว่าเป็นการเดา | `find_referencing_symbols` |
| `replace_body(symbol, text)` | L2 | `replace_symbol_body` |
| `insert_after(symbol, text)`, `insert_before(symbol, text)` | L2 | `insert_after_symbol`, `insert_before_symbol` |
| `rename(symbol, new)` | L3 เท่านั้น | `rename_symbol` |
| `diagnostics(path)` | L3 แล้วตกไปใช้ compiler หรือ linter | ไม่มีโดยตรง |

**ความหมายของ native ในงานนี้**

- **ไม่ผ่าน MCP และไม่จบ turn:** ระหว่างคิดเรียกได้ในรูป block และ server ตอบผ่าน `SBX`
- **ไม่แยกตัวไปดูของจริง:** ทุกชั้นอ่านสำเนา จึงเห็นสิ่งที่โมเดลเพิ่งแก้ทันที ซึ่งเป็นเรื่องที่ Cursor ทำ shadow workspace ขึ้นมาเพื่อแก้
- **ทนโค้ดครึ่งทาง:** tree-sitter parse ไฟล์ที่ยังผิด syntax ได้ และ parse ซ้ำเฉพาะส่วนที่เปลี่ยน
- **ตัว index อยู่ใน box:** parser อ่านไฟล์ที่ไม่น่าไว้ใจ จึงไม่รันใน server ซึ่งมีสิทธิ์เต็ม

**ไม่ได้เอา Serena ทั้งตัวเข้า box ในรุ่นแรก** เพราะมันเพิ่ม Python กับ language server หนึ่งชุดต่อ session ส่วนความสามารถหลักของ L1 และ L2 ทำได้โดยไม่ต้องมี process เพิ่ม ถ้า L3 คุ้ม ค่อยพิจารณาใช้ Solid-LSP ของ Serena เป็นตัวคุย language server ใน box

**Repo map:** [Aider](https://aider.chat/docs/repomap.html) จัดอันดับ symbol ด้วย graph แล้วส่งแผนที่ย่อราว 1,000 token ให้โมเดลตั้งแต่ต้น แนวนี้ใช้กับ `outline(".")` ได้ แต่แผนที่ที่ส่งทุก turn เป็นค่าใช้จ่ายประจำ จึงให้เรียกเมื่อต้องใช้ก่อน

**การค้นเชิงความหมาย (embedding):** เป็นทางเลือกภายหลัง [งานศึกษาบน SWE-QA](https://arxiv.org/abs/2608.01507) รายงานว่า vector search ถูก 65.2 % เทียบกับ subagent ที่ใช้ grep 46.2 % ส่วนตัวเลขลด token 94-98 % ที่เห็นทั่วไปเป็นของผู้ขาย ยังไม่มีการยืนยันอิสระ ชั้นนี้ต้องมี embedding model และ index จึงรอจนเห็นว่า L1-L3 ตอบไม่พอ

## 5. กรณีที่ต้องรองรับ

### งานที่ไม่แตะ project

| # | กรณี | การจัดการ |
|---|---|---|
| 1 | คำนวณล้วน | T0 |
| 2 | sympy, numpy | T0 ปฏิเสธ แล้วยกเป็น T1 |
| 3 | Library ไม่มีในเครื่อง | ไม่ `pip install` ตอบรายการที่มี ผู้ใช้เพิ่มเข้า wheelhouse ได้ |
| 4 | โปรแกรมอ่าน stdin | ช่วย `feed("...")` ป้อนตัวอย่าง stdin ว่างตอบ `EOFError` พร้อมคำแนะนำ |
| 5 | `random`, เวลา | T1 ของจริง ตั้ง `PYTHONHASHSEED` คงที่ |
| 6 | กราฟ, GUI | ไม่มีจอ matplotlib ใช้ Agg รูปไม่เข้าการคิด |
| 7 | ภาษาอื่น | T2 ผ่าน toolchain ใน allowlist แบบ read-only |

### การอ่าน project

| # | กรณี | การจัดการ |
|---|---|---|
| 8 | ไฟล์ทั่วไป | `read` และ `grep` ภายใต้งบ |
| 9 | ไฟล์ใหญ่หรือ binary | ตอบขนาดและชนิด ไม่พิมพ์ |
| 10 | ต้องใช้ `node_modules`, `.venv` | ชี้แบบ read-only และย้าย cache ไป delta |
| 11 | Submodule | รวมตามสถานะปัจจุบัน |
| 12 | Symlink หรือ junction ออกนอก project | ตัดออก และตรวจ path จริงทุกครั้ง |
| 13 | Path ยาว, Unicode, ตัวพิมพ์ | ใช้ `\\?\` และทดสอบเฉพาะ |
| 14 | CRLF, encoding | คัดลอกเป็น byte |
| 15 | ไฟล์ที่แก้ค้าง | อ่านจาก working tree |
| 16 | ความลับ | `is_secret` บวก denied path และไม่มี network |
| 17 | Token ในไฟล์ธรรมดา (แบบ PocketOS) | อาจตรวจไม่เจอ I3 ทำให้ใช้ไม่ได้ |

### การแก้และ test

| # | กรณี | การจัดการ |
|---|---|---|
| 18 | แก้แล้วรัน test | เส้นทางหลัก |
| 19 | ลบหรือเปลี่ยนชื่อไฟล์ | เก็บใน delta และแสดงใน diff |
| 20 | แก้เป็นจำนวนมาก | สรุปจำนวนไฟล์และบรรทัด ไม่พิมพ์ทั้งหมด |
| 21 | Test เขียนไฟล์ | เขียนใน delta ไม่นับเป็น diff ถ้าถูก ignore |
| 22 | Test ตกอยู่ก่อนแล้ว | เทียบกับ base เสมอ |
| 23 | Test ไม่แน่นอน | รันซ้ำเมื่อตก แล้วรายงานว่าไม่แน่นอน |
| 24 | Test ช้า | เลือกเฉพาะที่เกี่ยวข้องก่อน เกินเวลาให้รันเบื้องหลังและตอบ `ยังรันอยู่` |
| 25 | ต้องใช้ฐานข้อมูลในเครื่อง | SQLite ได้ service อื่นรันใน box ได้ถ้าอยู่ใน allowlist |
| 26 | ต้องใช้ network | ตกพร้อมเหตุผล เปิดรายปลายทางได้โดยผู้ใช้เท่านั้น |
| 27 | ต้องใช้ GPU | ไม่ให้ ตอบว่าต้องรันนอก sandbox โดยผ่าน permission |
| 28 | Build หนัก เช่น engine นี้ | ไม่รันระหว่างคิด ใช้ทาง tool พร้อมเพดานเวลา |
| 29 | เปิด server ทดสอบ | loopback ใน box ปิดพร้อม job ต้องทดสอบว่า backend ยอมไหม |
| 30 | Process ค้าง | Job Object ฆ่าทั้งต้น |
| 31 | โมเดลแก้ test ให้ผ่าน | Gate เตือน |

### เรื่องที่เกิดพร้อมกัน

| # | กรณี | การจัดการ |
|---|---|---|
| 32 | ผู้ใช้แก้ไฟล์ระหว่างรอ | ตรวจก่อนรัน แจ้งเมื่อชน |
| 33 | โมเดลแก้ของจริงแล้วสำเนาเก่า | Server รู้ทุก Edit จึง sync ทันที |
| 34 | สลับ branch | Base เปลี่ยน แจ้งโมเดลและเก็บ delta เดิมไว้ |
| 35 | สองแชตใน project เดียว | Delta แยกกัน ชนกันตอน apply |
| 36 | Engine ล่ม | Delta อยู่บนดิสก์ กลับมาทำต่อได้ จากนั้นเก็บกวาดตามอายุ |
| 37 | ผู้ใช้กดหยุด | ฆ่า job แล้วจบ request |
| 38 | Regenerate | Kernel ใหม่ ไม่ทิ้ง delta |
| 39 | บีบ context | ฝากสถานะสรุปไว้ใน session |
| 40 | Batch slot | หนึ่ง session ต่อ slot ซึ่งโครง `InlineTurn` รองรับแล้ว |

### พฤติกรรมของโมเดล

| # | กรณี | การจัดการ |
|---|---|---|
| 41 | เข้าใจว่าแก้ของจริงแล้ว | I5 และ reply ที่อ้างว่าเสร็จโดยยังไม่ apply ถูกทำป้าย |
| 42 | Path ที่แต่งขึ้น | ตอบ path ที่ใกล้เคียง |
| 43 | วน block เดิม | ตัวกัน loop และงบ turn |
| 44 | พิมพ์ท่วม | เพดาน output |
| 45 | ไม่ยอมใช้ sandbox | ใช้ gate ไม่บังคับ block ในงานโค้ด ผลวันนี้ชี้ว่าการบังคับเร็วทำให้เขียนโปรแกรมไม่ตรงโจทย์ |
| 46 | ข้อความในไฟล์ชักนำ | ไม่มี network และความลับ ส่วน diff ที่เป็นผลต้องผ่าน permission |
| 47 | ผ่าน test แต่ผิด | ป้ายบอกว่าผ่านอะไรบ้าง ไม่พูดว่าถูก |

### ตัวกั้นล้มเหลว

| # | กรณี | การจัดการ |
|---|---|---|
| 48 | เริ่มไม่ได้ | ปิดฟีเจอร์และบอกเหตุผล (I4) |
| 49 | Antivirus หรือ group policy ขวาง | เช็คตอนเริ่มด้วยชุดทดสอบสั้น |
| 50 | ดิสก์เต็ม | Quota ต่อ session ใช้ไดรฟ์ที่ว่างมากสุด |
| 51 | หน่วยความจำไม่พอ | เพดาน Job Object ค่าเริ่มต้นต้องอยู่ในงบ RAM ของ server |
| 52 | Windows เก่า | ตกไปใช้ AppContainer หรือปิด |
| 53 | Linux | bubblewrap กับ overlayfs |
| 54 | Project อยู่บน network drive หรือ WSL | ปฏิเสธในรุ่นแรก |
| 55 | รันด้วยสิทธิ์ admin | ลดสิทธิ์ก่อนเข้า box |

### การค้นและโครงสร้าง

| # | กรณี | การจัดการ |
|---|---|---|
| 56 | ภาษาที่ไม่มี grammar | ตกไป L1 และบอกว่าไม่มี outline |
| 57 | Macro หนักแบบ C++ | tree-sitter อาจอ่านผิด ตรวจกับข้อความจริงก่อนแก้ และตกไป L1 |
| 58 | ชื่อ symbol ซ้ำ | ตอบทุกตัวพร้อมไฟล์และบรรทัด ให้เลือก ไม่แก้แบบเดา |
| 59 | References จากการเดาชื่อ | ติดป้ายว่าไม่ยืนยัน ไม่ใช้ทำ rename |
| 60 | ไฟล์ขนาดใหญ่มาก | Index แบบแบ่งส่วน และส่ง outline ทีละระดับ |
| 61 | Language server ไม่พร้อม | เรียกคำสั่ง L3 แล้วตกลงชั้นล่างพร้อมบอก |
| 62 | ไฟล์ที่สร้างอัตโนมัติ | ตัดออกจาก index ตาม `.gitignore` และรูปแบบที่รู้จัก |

## 6. ภัยคุกคามและการทดสอบ

ชุด escape test ต้องผ่าน 100 % ก่อนเปิดทุกขั้น และรันทุกครั้งที่เริ่ม server แบบย่อ

| ภัย | ตัวกัน | การทดสอบ |
|---|---|---|
| เขียน project จริง | Read-only ใน policy | เขียนทับด้วย path ตรง, path สั้น, junction |
| อ่านความลับ | ไม่ map และ denied path | อ่าน `.env`, `~/.ssh`, credential manager |
| ออก network | Deny by default | TCP, UDP, DNS, ICMP, proxy ของเครื่อง |
| คุยกับ server หรือ engine | Loopback ปิด | เรียก port ของ Strata, named pipe |
| ค้างหลังจบ | Job kill-on-close | Fork, detach, scheduled task, service |
| กินทรัพยากร | Job Object และ quota | Fork bomb, memory bomb, disk bomb, output bomb |
| DLL จาก project | โหลดจากโฟลเดอร์ระบบก่อน | วาง DLL ปลอมในสำเนา |
| Registry และ COM | ไม่มีสิทธิ์ | เขียน HKCU, เปิด COM server |
| Clipboard และหน้าจอ | UI restriction | อ่าน clipboard, screenshot |
| GPU | ไม่ให้ | เรียก CUDA และ DirectX |
| หนีจาก interpreter | กั้นนอก Python | โค้ด introspection ทำอะไรไม่ได้นอกเหนือ policy |
| ตัว apply ถูกหลอก | ตรวจ path จริงและ base hash | Diff ที่ชี้ออกนอก project หรือผ่าน symlink |
| Parser ถูกโจมตีด้วยไฟล์ | Index ทำงานใน box | ไฟล์ซ้อนลึก ไฟล์ขนาดมหาศาล ไฟล์ encoding พัง |

## 7. ทางที่ไม่เลือก

| ทาง | เหตุผล |
|---|---|
| ขยายตัวแปลภาษาให้รัน sympy | ต้องทำ Python เต็มภาษา และเสียจุดเด่นที่ตรวจเทียบ CPython ได้ |
| จำกัดภายใน Python | ถูกเลี่ยงได้ ต้องกั้นจากภายนอก |
| Hardlink | เขียนทะลุถึงของจริง |
| Docker เป็นหลัก | เริ่มช้าและไม่รัน toolchain Windows เก็บไว้เป็น T3 |
| ให้โมเดลลงมือ `git apply` เอง | ข้าม permission |
| Fail-open แบบ Claude Code | ขัด I4 เมื่อ sandbox คือสิ่งที่ผู้ใช้เชื่อใจ |
| WinFsp | เพิ่ม driver ภายนอก ใช้ ProjFS ก่อน |

## 8. ลำดับการสร้าง

แต่ละขั้นมีเงื่อนไขผ่าน ถ้าไม่ผ่านให้หยุดและแก้ ไม่ทำขั้นถัดไป

| ขั้น | งาน | ผ่านเมื่อ |
|---|---|---|
| S0 | จบการเลือกค่าปิด thinking บน A100 | Token ลดและความแม่นไม่ตกกว่า plain |
| S1 | T1 บน A100 ด้วย backend `none` เทียบ T0 บนข้อพลาดของ HLE และ SuperGPQA | ข้อคู่ได้คำตอบถูกเพิ่ม หรือ token ลดโดยไม่เสียข้อถูก |
| S2 | ทดลอง MXC และ AppContainer บนเครื่องนี้ด้วยโปรแกรมของเราเอง ไม่ใช่โค้ดโมเดล | Escape test 100 %, เริ่ม box ไม่เกิน 100 ms เมื่ออุ่นไว้ |
| S3 | T1 บน Windows พร้อม kernel ค้าง | ความเร็ว block ที่อุ่นแล้วไม่เกิน 20 ms |
| S4 | T2 กับทาง tool | ระบบเฝ้าไฟล์ยืนยันว่าไม่มีการเขียนของจริงเลยตลอดทั้งชุดทดสอบ |
| S4.1 | L1 และ L2 ใน box เทียบกับ `read` ทั้งไฟล์ | Token ต่องานลดโดยแก้ถูกตำแหน่งไม่น้อยลง |
| S4.2 | L3 สำหรับ Python และ C++ | ตัด references ที่ผิดลงจริง คุ้มกับเวลาเริ่มและหน่วยความจำ |
| S5 | `sandbox_apply`, gate, สถานะหน้าเว็บ | ทุก apply มี checkpoint และ rewind ได้ |
| S6 | ทางระหว่างคิด พร้อมงบ | เทียบคู่กับทาง tool บน token และเวลา |
| S7 | วัดสมมติฐานหลักบนชุดแก้บัคจริง เช่น SWE-bench subset | จำนวน edit ที่พังในของจริงลด โดยคำตอบถูกไม่ลด |
| S8 | Fork หลายทางแบบ first-commit-wins และ ProjFS | ทำเมื่อ S7 ชี้ว่าคุ้ม |

ตัวเลขใน S2 และ S3 เป็นเป้าเริ่มต้น ปรับได้เมื่อมีข้อมูล

**การวัดที่ตอบคำถามของคุณตรงๆ (S7):** แบ่งงานเดียวกันเป็น 2 แขน แขนหนึ่งแก้ของจริงตามปกติ อีกแขนลองในสำเนาก่อน นับจำนวนครั้งที่ของจริงอยู่ในสถานะพัง จำนวน edit ที่ต้องแก้ซ้ำ คำตอบถูก token และเวลา

## 9. สิ่งที่ต้องให้คุณตัดสิน

| ข้อ | เรื่อง | ข้อเสนอ |
|---|---|---|
| D1 | อนุญาตโค้ดที่โมเดลเขียนรันใน CPython บนเครื่องนี้ภายใต้ตัวกั้น OS | ตัดสินหลัง S1 และ S2 ตอนนี้ยังห้ามตามเดิม |
| D2 | ใช้ MXC เป็น dependency หรือทำเองด้วย AppContainer | ลองทั้งคู่ใน S2 แล้วเลือกจากตัวเลข |
| D3 | เปิด Windows feature ProjFS หรือสร้าง Dev Drive | ยังไม่ต้อง ทำเมื่อ repo ใหญ่จนการคัดลอกช้า |
| D4 | ค่าเริ่มต้นของ gate | `warn` |
| D5 | Project แรกที่จะทดสอบ | Repo ขนาดเล็กที่มี test เร็วก่อน ยังไม่ใช่ Strata |

## 10. สิ่งที่ยังไม่รู้

- Latency จริงของ MXC และ AppContainer เพราะไม่มีเอกสารตัวเลข
- MXC รองรับ Python ของระบบกับ toolchain ได้แค่ไหน และกัน registry กับ COM ได้แค่ไหน
- ProjFS ทำงานร่วมกับตัวกั้นได้ไหม
- Windows Sandbox เปิดแบบไม่มีหน้าต่างได้ไหม
- ทางระหว่างคิดประหยัดเวลากว่าทาง tool เท่าไหร่ ยังไม่เคยวัด round trip ของ tool บน D2x
- tree-sitter เร็วแค่ไหนใน box และอ่าน header C++ ขนาดนี้ได้แม่นแค่ไหน
- Gyro-M จะใช้คำสั่งระดับ symbol เองไหม หรือยังถนัด `grep` ทั้งไฟล์
- สมมติฐานหลักยังไม่มีข้อมูลจาก Gyro-M

## แหล่งอ้างอิง

- [Cursor: Shadow Workspace](https://cursor.com/blog/shadow-workspace)
- [Fork, Explore, Commit: OS Primitives for Agentic Exploration](https://arxiv.org/abs/2602.08199)
- [Toward Systems Foundations for Agentic Exploration](https://arxiv.org/abs/2510.05556)
- [Claude Code: Sandboxing](https://code.claude.com/docs/en/sandboxing)
- [Codex Windows sandbox engineering (แหล่งรอง)](https://codex.danielvaughan.com/2026/05/14/codex-cli-windows-sandbox-engineering-restricted-tokens-acls-elevated-architecture/)
- [Microsoft Execution Containers](https://learn.microsoft.com/en-us/windows/apps/develop/mxc/)
- [microsoft/mxc](https://github.com/microsoft/mxc)
- [CreateProcessInSandbox](https://learn.microsoft.com/en-us/windows/win32/secauthz/createprocessinsandbox)
- [Serena](https://github.com/oraios/serena)
- [Aider: repository map](https://aider.chat/docs/repomap.html)
- [Deep Agentic Search for Repository-Level Code Question Answering](https://arxiv.org/abs/2608.01507)
- [Morph: Python Sandbox](https://www.morphllm.com/python-sandbox)
- [Fly.io: Firecracker vs gVisor](https://fly.io/learn/firecracker-vs-gvisor/)
- [ReFS block cloning on Dev Drive](https://www.elevenforum.com/t/microsoft-improves-refs-file-system-to-block-cloning-on-dev-drive-in-windows-11-24h2.29068/latest)
- [ExecCritic: Learn to Test, Test to Improve for Coding Agents](https://arxiv.org/abs/2609.09133)
- [Can Old Tests Do New Tricks for Resolving SWE Issues?](https://arxiv.org/abs/2510.18270)
- [PocketOS incident (Tom's Hardware)](https://www.tomshardware.com/tech-industry/artificial-intelligence/claude-powered-ai-coding-agent-deletes-entire-company-database-in-9-seconds-backups-zapped-after-cursor-tool-powered-by-anthropics-claude-goes-rogue)
- [Replit incident (OECD)](https://oecd.ai/en/incidents/2025-07-19-1eb1)
