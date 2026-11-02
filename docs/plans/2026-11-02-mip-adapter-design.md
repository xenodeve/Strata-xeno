# MIP Adapter รุ่นแรก: เรียก MCP จากในการคิด

สถานะ: อยู่บน branch `xeno/mip-adapter` ยังไม่ merge ผ่าน test ที่ไม่ใช้ GPU แล้ว แต่ยังไม่เคยรันกับ engine จริงหรือโมเดลจริง ต่อยอดจาก Roadmap หัวข้อ 5 ถึง 7 และ `docs/plans/2026-11-02-sandbox-workspace-design.md`

## สรุป

- **ทำอะไรได้แล้ว** โปรแกรมใน block ระหว่างคิดค้น tool ด้วย `mcp.tools("คำค้น")` และเรียกด้วย `mcp.<server>.<tool>(arg=value)` ได้ MCP server เดิมไม่ต้องแก้อะไร เพราะ adapter เป็น MCP client ธรรมดาและประกอบ JSON-RPC เอง
- **ทำไมต้องเป็นทางนี้** ผลลัพธ์อยู่ในตัวแปรของโปรแกรม ส่วนที่ไม่ print ไม่เข้า context เลย tool ที่ตอบมา 20,000 ตัวอักษรอาจเหลือเลขตัวเดียวในการคิด
- **รุ่นนี้อ่านอย่างเดียว** tool ที่ server ไม่ประกาศ `readOnlyHint: true` จะถูกปฏิเสธ พร้อมบอกชื่อบน tool path เดิมให้
- **นี่ยังไม่ใช่ MIP ตามเป้าหมาย** ไม่มี latent feedback (M3) ไม่มี headless state (M5) และไม่มี learned router โมเดลยังต้องเขียน Python สั้น ๆ เอง ผลก็ยังกลับเข้าการคิดเป็นข้อความผ่าน forced token แบบ text-feedback
- **ยังไม่มีตัวเลขว่าคุ้มไหม** ทุกอย่างที่พูดถึงการประหยัด token ในเอกสารนี้เป็นสมมติฐาน วิธีวัดอยู่ในข้อ 4

## 1. สิ่งที่มีตอนนี้

### เส้นทาง

```text
โมเดลเขียน block ที่มี mcp.
  │
  ▼
engine: Session::run เห็น mcp. แล้วส่งไป kernel แทนตัวแปลภาษา (เฉพาะเมื่อตั้ง exec_ ไว้)
  │  SBX <fresh> <base64>
  ▼
server: _pump → _run_block → ExecBridge.handle → Kernel.run
  │  โปรแกรมรันใน box ไม่มี network ทางออกเดียวคือ pipe เดิม
  │
  │  kernel → server  {"mcp": {"op": "call", "server": ..., "tool": ..., "arguments": ...}}
  │  server → kernel  {"mcp": <MIPResult เป็น JSON>}
  │  วนจนกว่าจะได้คำตอบสุดท้ายของ block
  ▼
McpAdapter: ตรวจทุกอย่าง → McpServer.call → MCP server ตัวจริง
  │
  ▼
โปรแกรมเลือกสิ่งที่ต้องการแล้ว print
  │  SBXR <ok> <base64>
  ▼
engine: ส่งเป็น forced token ไม่เกิน 1,024 ไบต์
```

### ตัวอย่างที่โมเดลเขียน

```python
for line in mcp.tools("weather"):
    print(line)
```

```output
mcp.fake.weather(city: str, units: 'c' | 'f' = ..., days: int = ..., tags: list = ...)  # The weather of a city, as fields.
```

```python
w = mcp.fake.weather(city="bkk")
print(w["temp"])
```

```output
31
```

ผลลัพธ์แบบมีโครงสร้างได้เป็น `dict` หรือ `list` นอกนั้นได้ `str` ถ้าเรียกไม่สำเร็จ โปรแกรมจะเจอ `ToolError` ที่มีเหตุผลอยู่ในข้อความ จับเองก็ได้ ปล่อยทิ้งโมเดลก็เห็นเป็น error ของ block ตรงบรรทัดที่เขียน

### ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `serve/mip/core.py` | `MIPInvocation` และ `MIPResult` ตาม Roadmap 5.1 พร้อมรูปแบบ JSON |
| `serve/mip/mcp_adapter.py` | การค้น, ตรวจ argument, ด่าน read-only, งบ และการเรียกผ่าน `McpHub` |
| `serve/sandbox/kernel_main.py` | ออบเจกต์ `mcp` ใน kernel ไม่ import อะไรจาก repo และไม่รู้เรื่อง server เลย |
| `serve/sandbox/exec_bridge.py` | `Kernel.run` วนตอบคำขอจนจบ block, `begin` และ `stop` ของแต่ละ request |
| `serve/server.py` | `run_with_mcp` สร้าง adapter ของ request, `StrataEngine.generate` ส่งให้ bridge, `sandbox_bridge` สร้าง bridge |
| `include/strata/core/igc_sandbox_bridge.hpp` | `uses_mcp` ในเงื่อนไขส่งงานไป kernel |

### ด่านก่อน tool จะรัน

ตรวจฝั่ง server ทั้งหมด เพราะโปรแกรมเป็นของโมเดลและเขียนอะไรลง pipe ก็ได้ ลำดับเป็นแบบนี้

| ลำดับ | ตรวจ | ถ้าไม่ผ่าน |
|---|---|---|
| 1 | request ถูกหยุดหรือยัง | `cancelled` |
| 2 | request นี้เสนอ tool นี้ไหม | `unsupported` |
| 3 | กฎ deny ของแชต (`permissions.decide`) | `refused` |
| 4 | `annotations.readOnlyHint` เป็น `true` จริง ๆ ไหม | `refused` พร้อมชื่อบน tool path |
| 5 | argument ตรง `inputSchema` ไหม (required, type, enum) | `refused` |
| 6 | งบเหลือไหม | `refused` |

ถ้าผ่านครบแล้วเรียก server ไม่สำเร็จจะได้ `fault` ซึ่งรวม tool รายงาน error, หมดเวลา และ server หยุด

การ invoke จาก reasoning ไม่ถือเป็นการอนุญาต ตรงกับบรรทัด Authorization ใน Roadmap หัวข้อ 10 Adapter จึงไม่ถามผู้ใช้และไม่รัน tool ที่อาจเปลี่ยนของ

### งบ

| ค่า | ตัวเลข | เหตุผล |
|---|---|---|
| `CALLS_PER_BLOCK` | 8 | เท่า `max_rounds` ของ tool path ทั้งคำตอบ block ที่ต้องการมากกว่านี้คือวนซ้ำ |
| `CALLS_PER_TURN` | 32 | สี่ block เต็ม decode รอทุกการเรียก เกินนี้คือการเฝ้ารอ ไม่ใช่การตรวจ |
| `BLOCK_SECONDS` | 120 | เวลา tool รวมต่อ block แต่ละการเรียกรอไม่เกินเวลาที่ block เหลือ |
| `max_result_chars` | ตามตั้งค่าของ hub | ตัดพร้อมหมายเหตุแบบเดียวกับ `McpHub` |

ค่าเหล่านี้ตั้งจากเหตุผล ยังไม่ได้วัด

ผลแบบมีโครงสร้างที่ยาวเกินเพดานจะกลายเป็น `str` ที่ตัดแล้วพร้อมหมายเหตุ เพราะ dict ครึ่งตัวอ่านไม่ได้ โปรแกรมที่คาดว่าจะได้ dict ก็จะเจอ error ตอนเลือก field ซึ่งบอกได้ตรง ๆ ว่าผลถูกตัด

## 2. เทียบกับ Milestone

Roadmap วาง M1 ไว้หลัง M0 ซึ่งยังไม่ผ่าน รุ่นนี้สร้างก่อนตามคำสั่งของผู้พัฒนา จึงเชื่อกับ native ยังไม่ได้

| Milestone | ทำแล้ว | ยังขาด |
|---|---|---|
| **M1** Protocol Extraction | Envelope ครบทุก field ของ 5.1 มี JSON form และ test | Native ยังไม่ได้ใช้ envelope นี้ field ที่เกี่ยวกับ state เช่น `accepted_prefix_identity` ยังว่าง ไม่มี registry และ state contract |
| **M2** MCP Adapter | ค้น tool ตามคำ, แปลง schema เป็น signature, เรียกแบบ typed, ตรวจ argument ฝั่ง server | ทำเฉพาะ tool อ่านอย่างเดียว ทดสอบกับ fake server เท่านั้น ยังไม่เคยเจอ MCP server จริง |
| **M4** Selective Decode | โมเดลไม่ต้องเขียน JSON-RPC ชื่อ field หรือปีกกา adapter ประกอบให้ | ทำได้แค่บางส่วน โมเดลยังเขียน Python ทั้งบรรทัด ซึ่งก็คือ tool syntax แบบสั้น ส่วนงานแบบ Roadmap 6.5 ที่เขียนรายงานลงเอกสารยังทำไม่ได้ เพราะ tool นั้นไม่ใช่ read-only |

## 3. สิ่งที่ไม่ได้สร้าง

| เรื่อง | ผลตอนนี้ |
|---|---|
| Latent feedback (M3) | ผลกลับเข้าการคิดเป็นข้อความเท่านั้น ทุกไบต์ที่ print เป็น forced token |
| Headless state (M5) | ไม่มี KV หรือ state ใหม่ ไม่มีอะไรข้าม prefill ได้ |
| Learned router | ไม่มี Adapter ไม่อ่าน activation เลย โมเดลเลือก tool ด้วยการเขียนโค้ด |
| ข้อความบอกโมเดล | ยังไม่มี prompt ใดบอกว่า `mcp` มีอยู่ โมเดลจึงจะไม่เรียกเอง ต้องทำก่อนวัด |
| Isolation box | บน branch นี้ยังไม่มี `box_factory` บน Windows ทุก block ที่ถึง kernel จึงได้คำตอบ `no isolation box` |
| สถานะบนหน้าเว็บ | `agentStatus` ยังไม่มีคำสำหรับขั้นนี้ ซึ่งขัดหลักใน `AGENTS.md` ต้องทำก่อนเปิดให้ผู้ใช้ |
| Tool ที่เขียนของ, การขอ consent | ถูกปฏิเสธทั้งหมด ไปทาง tool path แทน |
| JSON Schema ส่วนอื่น | `anyOf`, `$ref`, `minimum` ปล่อยให้ server ตรวจเอง |
| Resources, prompts ของ MCP | ไม่รองรับ มีแต่ tool |
| Batch slot และ replica | ไม่รองรับ หนึ่ง kernel ต่อหนึ่ง request |

**สิ่งที่ `readOnlyHint` รับประกันไม่ได้** ค่านี้เป็นคำของ server เอง สเปก MCP เรียก annotation ว่าเชื่อถือไม่ได้ ถ้า server ใส่ `true` ให้ tool ที่ลบของ ด่านนี้ก็ปล่อยผ่าน ด่านนี้จึงกันได้แค่ tool ที่พูดความจริง ส่วน tool path เดิมเองก็ไม่ได้ผ่าน `permissions.decide` สำหรับ MCP tool คนที่กันจริงคือการเห็น call บนหน้าเว็บ

## 4. วิธีวัดเทียบกับ `standard_mcp`

ทั้งหมดนี้ยังไม่ได้ทำ ต้องมีข้อความบอกโมเดลก่อน และบนเครื่องนี้ต้องรอ box จึงรันบน A100 ที่ทิ้งได้ด้วย `STRATA_SANDBOX_THROWAWAY=1`

**แขน** งานชุดเดียวกัน โมเดลเดียวกัน seed เดียวกัน ต่างกันแค่ทางเรียก

| แขน | ทาง |
|---|---|
| `standard_mcp` | tool path เดิม จบ turn แล้วผลทั้งก้อนเข้า context ผ่าน prefill |
| `mip_text` | block ระหว่างคิดผ่าน adapter นี้ |

**งาน** ต้องมีทั้งแบบที่ควรได้เปรียบและแบบที่ไม่ควรได้เปรียบ ไม่อย่างนั้นผลจะเข้าข้างตัวเอง

| ชนิด | ตัวอย่าง | สมมติฐาน |
|---|---|---|
| ผลใหญ่ เลือกนิดเดียว | อ่าน JSON ยาวแล้วตอบหนึ่ง field | `mip_text` ชนะชัด |
| เรียกต่อกันหลายครั้ง | ใช้ผลรอบแรกเป็น argument รอบสอง | `mip_text` น่าจะชนะเรื่องจำนวนรอบ |
| ต้องอ่านเนื้อหาทั้งก้อน | สรุปเอกสาร | ไม่ควรต่างกัน หรือ `mip_text` แพ้เพราะ forced token ไม่ใช่ prefill |

**ตัวชี้วัด** อ่านจากรอบเดียวกันเสมอ ห้ามเอาตัวเลขต่างรอบมารวมกัน

| ตัวชี้วัด | อ่านจากไหน |
|---|---|
| Tokens | `completion_tokens` รวมทุกรอบของ `run_with_mcp` |
| Feedback size | `standard_mcp`: ขนาดผลที่เข้า context ต่อรอบ, `mip_text`: ไบต์ของ `SBXR` ที่ engine รับ |
| Context growth | `prompt_tokens` รอบสุดท้ายลบรอบแรก และจำนวนไบต์ใน reasoning ที่มาจาก forced token |
| Turns | จำนวนรอบ tool ของ `run_with_mcp` เทียบจำนวนการเรียกใน log `[strata] mip` |
| Task success | คำตอบถูกตามชุดงาน ตรวจด้วย `answer_check.py` เคียงกับกฎ |
| เวลา | span `mip.call` ใน `STRATA_TIMELINE` เทียบเวลา prefill ของผลบน tool path |

**ลำดับการตัดสิน** ใช้เกณฑ์เดียวกับ sandbox ห้ามเสียข้อที่ `standard_mcp` ทำได้ จากนั้นดูคำตอบถูกที่เพิ่ม แล้วค่อยดู token และเวลา

**ความเร็ว** ถ้าจะอ้างว่าเร็วขึ้น ต้องวัดบน D2x แบบ ABBA ที่ `--pool-priority 2` และ `--process-priority 2` ตามกฎของ repo พร้อมบอกสัดส่วน expert hit ส่วนบน A100 เป็นหลักฐานเรื่องคำตอบถูกเท่านั้น

## 5. การตัดสินใจที่ผู้พัฒนาควรดู

| # | เรื่อง | สิ่งที่ทำ | เหตุผล |
|---|---|---|---|
| 1 | Hub เข้า `ExecBridge` ตรงไหน | ไม่ส่งตอนสร้าง แต่ส่ง adapter ต่อ request ผ่าน `begin` | การเปิด MCP เป็นราย request (`strata_mcp`, `strata_mcp_off`) และ `reload_hub` เปลี่ยนตัว hub ได้ ถ้าถือ hub ตัวเดียว request ที่ไม่ได้เปิดก็เข้าถึงได้ และ hub จะเก่าหลังแก้ config |
| 2 | ใครสร้าง `ExecBridge` | เพิ่ม `sandbox_bridge` ใน `main()` | ก่อนหน้านี้ `serve/server.py` ไม่เคยสร้าง มีแต่ `tools/igc/external_bench.py` config ที่ไม่มี `--igc-sandbox-exec` จึงไม่เปลี่ยนอะไร |
| 3 | ข้อความ error จาก transport | ส่งเข้า kernel เป็นประโยคคงที่ ตัวจริงไปอยู่ใน log ของ server | ข้อความของ `McpError` อาจมี URL หรือบรรทัด log ที่มี token |
| 4 | กฎ deny ของแชต | เรียก `permissions.decide` แล้วเคารพเฉพาะ `deny` | เข้มกว่า tool path ซึ่งไม่เช็คกฎนี้กับ MCP tool เพราะทางนี้ผู้ใช้มองไม่เห็นการเรียก ส่วน `ask` ไม่นับ เพราะ `decide` ตอบ `ask` กับทุกชื่อที่ไม่รู้จัก |
| 5 | เรียกไม่สำเร็จ | โยน exception แทนการคืนค่า | ถ้าคืน dict ที่มี error โปรแกรมจะพังตอนเลือก field ด้วยข้อความที่ไม่บอกอะไร |
| 6 | คำตอบที่มาหลัง request จบ | ทิ้ง ไม่เขียนลง stdin ของ engine | Engine อ่าน `SBXR` ที่มาช้าเป็นคำสั่งผิด แล้วตอบ `ERR` ซึ่งทำให้ request ถัดไปล้ม ฝั่ง Python ยังเหลือช่องแคบระหว่างตัดสินใจกับตอนเขียน การแก้ที่ต้นเหตุคือให้ serve loop ข้าม `SBXR` เอง ซึ่งเป็นการแก้ `generate.cpp` จึงอยู่นอกงานนี้ |
| 7 | การหาคำว่า `mcp.` | ดูเป็นไบต์ ไม่ parse | Comment ที่มี `mcp.` จะพาไป kernel โดยไม่จำเป็น ราคาคือเปิด kernel หนึ่งตัว และถ้าไม่มี box block นั้นจะรันไม่ได้ทั้งที่ตัวแปลภาษารันได้ |
| 8 | ค้นด้วย substring | `no` เจอใน `note` | ง่ายและเร็ว แต่คำสั้นจะเจอทุกอย่าง เรียงให้ชื่อมาก่อนคำอธิบาย |
| 9 | Kernel ที่ไม่มี box | ส่ง environment ที่ Python ต้องใช้เท่านั้น (`kernel_env`) | เดิมสืบทอดทั้งหมด ซึ่งอาจมี `STRATA_API_KEY` ค่านี้ใช้เฉพาะบน VM ที่ทิ้งได้ ส่วน box จัดการ environment ของตัวเอง |

## 6. ข้อกังวลที่ยังเปิดอยู่

- **Block แรกที่ย้ายไป kernel เล่นโค้ดเดิมซ้ำ** `run_outside(source)` รัน block ที่จบแล้วทั้งหมดอีกครั้ง สิ่งที่เคย print จึงกลับมาอยู่หน้า output และเพดาน 1,024 ไบต์ตัดจากท้าย ผลของ tool ที่เพิ่ง print อาจหายไป ปัญหานี้มีกับ `import sympy` อยู่แล้ว
- **เวลา block ยังไม่มีเพดานจริง** งบเวลาครอบเฉพาะการเรียก tool โปรแกรมที่ค้างเองต้องรอ box คุม
- **Server ที่หยุดแล้วเริ่มใหม่ระหว่างการเรียก** อาจใช้เวลาถึง `start_timeout_s` เกินงบเวลาของ block แต่ยังอยู่ในเพดาน 600 วินาทีของ engine
- **Tool ที่ตอบเป็น JSON ใน text block** ได้เป็น `str` โปรแกรมต้อง `json.loads` เอง server ส่วนมากตอนนี้ยังตอบแบบนี้ ซึ่งยังไม่รู้ว่า Gyro-M จะทำเป็นไหม
- **Gyro-M จะใช้ไหม** เคยไม่เรียกฟังก์ชันที่เขียนเอง 36 จาก 36 block การค้นแบบ `mcp.tools` อาจต้องช่วยมากกว่าที่คิด
