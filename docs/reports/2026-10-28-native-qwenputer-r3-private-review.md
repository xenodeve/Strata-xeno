# PRD R3: ให้ Strata คำนวณด้วย Native Qwenputer

ฉบับร่างในเครื่องสำหรับตรวจเนื้อหา — 28 ตุลาคม 2026

สถาปัตยกรรมและแผน R2 อนุมัติแล้ว ขณะนี้ทำได้ถึง P0 เท่านั้น ฉบับ R3 จัดเนื้อหาใหม่ตาม `/kien-thai` และเปลี่ยนโมเดลทดสอบเป็น Gyro-M ตามคำสั่งล่าสุด ยังไม่แก้ GitHub #276 หรืออัปโหลดฉบับนี้

## Problem Statement — ปัญหาที่ต้องแก้

เมื่อคำตอบต้องใช้เลขคำนวณ โมเดลภาษายังมีโอกาสสร้างตัวเลขผิด แม้จะอธิบายวิธีคิดได้ถูกต้อง Strata จึงต้องมีวงจรคำนวณที่โมเดลใช้ระหว่างตอบ แล้วนำผลกลับไปคิดต่อได้ในคำตอบเดียวกัน

เป้าหมายคือความถูกต้องของเลขที่รองรับ และ latency ต่ำที่สุดที่พิสูจน์ได้บนเครื่องจริง งานนี้ยังไม่รับประกันว่าการเขียนโค้ดหรืองาน 3D จะดีขึ้น ต้องทดสอบประโยชน์เหล่านั้นแยกภายหลัง

## Solution — เมื่อเปิดใช้แล้วควรทำงานอย่างไร

ตัวอย่างเช่น เมื่อโมเดลต้องหา `17 × 23` ก็ส่งนิพจน์เข้าวงจรภายใน รอผลที่คำนวณเสร็จ แล้วใช้ผลนั้นตอบหรือคิดขั้นถัดไป แอปที่เรียก API เห็นคำตอบต่อเนื่อง ไม่ต้องรับคำขอให้ไปเปิดเครื่องคิดเลขภายนอก

Native ในแผนนี้หมายถึงวงจร FFN/GDN ที่ compile ไว้เป็น tensor แล้วรันด้วย C++/CUDA บน GPU หลัก วงจรเลือก trace token และเดินงานต่อบน GPU เอง ระหว่างนั้นสถานะของโมเดลภาษาต้องหยุดอยู่ที่จุดเดิม พอคำนวณสำเร็จจึงป้อนข้อความผลลัพธ์กลับเข้าโมเดลและตอบต่อ

ไฟล์ weights ของวงจรแยกจากโมเดลหลักได้ เมื่อโหลดเป็น tensor และ graph ชุดเดียวกันแล้ว การแยกไฟล์ไม่ได้เพิ่มขั้นตอนทุกก้าวระหว่างคำนวณ ส่วน Python compiler และตัวอ้างอิง RISC-V ใช้เตรียมข้อมูลหรือทดสอบนอกเส้นทางตอบจริงเท่านั้น

เมื่อปิด feature เส้นทางเดิมต้องให้ผลเหมือน baseline เมื่อเปิดใช้ ระบบยัง offload experts ได้ตามเดิม แต่ต้องวัด RAM/VRAM ที่วงจรใช้เพิ่ม รวมถึงผลต่อ expert cache และ latency ของข้อความทั่วไปด้วย

## ขอบเขตที่ตกลงแล้วและเรื่องที่ยังต้องเลือก

- ทำ Native สำหรับ Strata Flash-Next ก่อน ส่วน Universal เป็นงานระยะถัดไป
- ใช้ Gyro-M เป็นโมเดลทดสอบ ไม่ต้องกู้ Swift กลับมาเพื่อ baseline นี้
- ใช้ GPU หลัก RTX 5060 Ti รันวงจร ส่วน RTX 4070 SUPER ยังทำหน้าที่ offload experts และแสดงผลตาม profile เดิม
- เก็บ weights ของวงจรเป็น artifact แยก ใช้ standalone conditional WHILE graph และจำกัดงานต่อรอบ
- ไม่เปลี่ยน profile ที่เสิร์ฟอยู่ ไม่เปิด feature เป็นค่าเริ่มต้น และไม่ deploy จากแผนนี้

ผู้ใช้เลือก Gyro-M 262K context, 5 workers จาก profile `strata-gyro-m-d2x.json` เป็น baseline แล้ว ให้ตรึงไฟล์และ settings ชุดนี้สำหรับเทียบ baseline กับ candidate ไม่ใช้ profile 512K/YaRN แทน

ถ้า conditional graph ใช้ไม่ได้ หรือวงจรเกินงบหน่วยความจำ ต้องเสนอทางเลือกพร้อมหลักฐานให้ตัดสินใจ ไม่เปลี่ยนสถาปัตยกรรมหรือย้ายไปคำนวณบน CPU เอง ส่วนการเผยแพร่ R3 และรายละเอียด profile ต้องรอคำอนุญาตแยก

## User Stories — สิ่งที่ต้องทำได้

1. **US01** ในฐานะผู้ใช้ Strata ต้องการผลเลขที่คำนวณตรงตามขอบเขตที่รองรับ เพื่อใช้คำตอบนั้นต่อได้
2. **US02** ในฐานะผู้ใช้ ต้องการให้โมเดลใช้วงจรภายใน เพื่อไม่ต้องพึ่งบริการเครื่องคิดเลขภายนอก
3. **US03** ในฐานะผู้ใช้ ต้องการให้โมเดลคิดต่อหลังได้ผล เพื่อให้การคำนวณอยู่ในคำตอบเดียวกัน
4. **US04** ในฐานะผู้ใช้ ต้องการให้พฤติกรรมเดิมคงอยู่เมื่อปิด feature เพื่อเลือกใช้ baseline ได้
5. **US05** ในฐานะผู้ใช้ ต้องการเห็น overhead ของข้อความทั่วไปแยกจากเวลาคำนวณ เพื่อประเมินต้นทุนของการเปิด feature
6. **US06** ในฐานะผู้ใช้ ต้องการเก็บระบบ offload ระหว่าง GPU, RAM และ NVMe ไว้ เพื่อให้โมเดลขนาดใหญ่ยังใช้งานได้
7. **US07** ในฐานะผู้ใช้ ต้องการเห็นสถานะคำนวณทันที เพื่อรู้ว่าระบบกำลังทำงานขั้นไหน
8. **US08** ในฐานะผู้ใช้ ต้องการให้ refresh หรือ reconnect แล้วเห็นงานที่กำลังรันต่อ เพื่อไม่เสียความต่อเนื่อง
9. **US09** ในฐานะผู้เรียก API ต้องการให้ native call อยู่ภายใน เพื่อไม่ต้องรับคำขอ external calc tool
10. **US10** ในฐานะผู้เรียก API ต้องการให้เครื่องมืออื่นทำงานตามเดิม เพื่อไม่ให้ Native เปลี่ยนวิธี dispatch tools
11. **US11** ในฐานะผู้ใช้ ต้องการให้อ่าน tool-call เก่าและข้อความที่ยกมาเป็นประวัติ เพื่อไม่ให้ prefill คำนวณคำสั่งเก่าซ้ำ
12. **US12** ในฐานะผู้ใช้ ต้องการให้ปฏิเสธข้อความที่ยาวเกินหรือไม่รองรับ เพื่อไม่ให้การตัดข้อความกลายเป็นคำตอบผิด
13. **US13** ในฐานะผู้ใช้ ต้องการให้ cancel และ limits หยุดงานได้ เพื่อไม่ให้ผลเลขที่ยังไม่ครบหลุดออกมาเป็นคำตอบสำเร็จ
14. **US14** ในฐานะผู้ใช้ ต้องการแยกข้อผิดพลาดจากผลเลขและเครื่องยนต์ที่เสีย เพื่อเลือกว่าจะลองใหม่หรือเริ่ม session ใหม่
15. **US15** ในฐานะผู้ใช้ ต้องการคำนวณหลายครั้งต่อกันได้ เพื่อให้สถานะวงจรและผู้ถือสิทธิ์การตอบยังถูกต้อง
16. **US16** ในฐานะผู้ใช้ ต้องการสลับบทสนทนาและใช้ prefix cache ได้ เพื่อไม่ให้ประวัติการคำนวณปะปนกัน
17. **US17** ในฐานะผู้ใช้ ต้องการ save/restore ทั้งโมเดลและวงจร หรือปฏิเสธข้อมูลที่ใช้ร่วมกันไม่ได้ก่อนเปลี่ยนสถานะ
18. **US18** ในฐานะผู้ดูแล ต้องการผูก artifact กับ tokenizer, firmware และ semantics ที่ตรงกัน เพื่อให้โหลดผิดชุดแล้วแจ้งชัดเจน
19. **US19** ในฐานะผู้ดูแล ต้องการปฏิเสธ artifact หรือ firmware ที่เสียก่อนจองหน่วยความจำก้อนใหญ่หรือส่งเข้า GPU
20. **US20** ในฐานะผู้ดูแล ต้องการจำกัดงานบน GPU เพื่อให้ Windows ยังตอบสนองได้โดยไม่แก้ TDR หรือค่าความปลอดภัย
21. **US21** ในฐานะผู้ดูแล ต้องการเห็นเวลาเริ่มระบบ ยอดใช้หน่วยความจำ และ expert hits บนเครื่องจริง เพื่อประเมินผลต่อการเสิร์ฟ
22. **US22** ในฐานะผู้พัฒนา ต้องการตรวจ trace กับ oracle อิสระและทดสอบทุกจุดที่เปลี่ยนผู้ถือสิทธิ์ เพื่อจับข้อผิดพลาดที่ผลเลขอย่างเดียวอาจมองไม่เห็น
23. **US23** ในฐานะผู้พัฒนา ต้องการพิสูจน์ parity ตอนปิด feature และวัด latency ใน session เดียวกัน ก่อนตัดสินใจนำไปใช้
24. **US24** ในฐานะผู้พัฒนา ต้องการบันทึกสัญญาของ Native ให้ชัด เพื่อให้ Universal ในอนาคตยังรักษาข้อกำหนดเดิมได้

## Implementation Decisions — สิ่งที่ implementation ต้องรักษา

### วงจรและ artifact ต้องตรวจสอบย้อนกลับได้

แยก compiler/exporter, artifact loader, circuit executor และตัวเชื่อมกับ decoder ออกจากกัน แต่ให้ทดสอบผ่านจุดเชื่อมระดับ conversation เป็นหลัก การทดสอบระดับล่างมีไว้แยกสาเหตุเมื่อจุดเชื่อมนี้ล้ม

artifact ต้องมี schema, compiler revision และ patch, numeric ABI, token map, firmware, geometry และ hashes ที่ผูกกันไว้ ต้องตรวจ license ของ source, firmware, runtime libraries และ weights ตามที่มาจริงด้วย

ก่อน allocate หรือ upload ต้องตรวจชนิด ขนาด รูปร่าง ช่วงค่า และขอบเขตของไฟล์ รวมถึง ELF machine/endian, entry point, segment overlaps และเงื่อนไข `memsz >= filesz` ไฟล์ที่ไม่ตรงต้องหยุดก่อนแก้สถานะ session

### คณิตศาสตร์ของวงจรต้องตรงตาม compiler

ต้องรักษา sparse FFN, normalization, head และ GDN RAM ตามสัญญาที่ตรึงไว้ ห้ามใช้ host calculator หรือ opcode emulator คำนวณคำตอบจริงแทน

FP32 atomic add เพียงอย่างเดียวไม่ได้พิสูจน์ว่าผลซ้ำจะเหมือนเดิม ต้องมีลำดับ reduction ที่แน่นอน หรือหลักฐานเรื่องช่วงค่าและ partial sums ที่เพียงพอ แล้วตรวจซ้ำกับ trace และผลเลขจาก oracle อิสระ

### GPU เดินงานเอง แต่ทุกครั้งต้องจบในขอบเขตที่กำหนด

graph ของวงจรเป็น standalone graph บน GPU หลัก ไม่เอาไปเป็น child ของ graph ที่มีอยู่ ต้องตั้ง condition ใหม่ทุก launch และจัดการ graph/state แยกตามกติกาที่ CUDA รองรับ

ระหว่าง trace ไม่ให้ host เลือก token ทีละก้าว หน่วยความจำที่ต้องใช้ต้องเตรียมก่อนเข้า graph งานต้องมี step/time/output limits, cancellation และ terminal reason ที่แยกสำเร็จ หยุด และผิดพลาดออกจากกัน

### การเข้าและออกจากวงจรต้องรักษาสถานะภาษา

จุด ENTER ต้องอิงจำนวน **input tokens ที่ commit แล้ว** ถ้าอยู่แถว `r` ของ verifier window ต้องคิดขอบเขต `r + 1` ให้ถูก ต้องรอ commit fence และทิ้ง speculative suffix ที่ใช้ไม่ได้ก่อนข้าม stream หรือ snapshot

ให้ลองเดิน candidate window บนสถานะวงจรสำรอง (shadow state) ก่อน แล้วจำกัด input prefix ตามผลยอมรับของ verifier และจุดรับช่วงแรกที่ใช้ได้ จากสถานะฐานที่เก็บไว้ (base state) ให้ replay เฉพาะ prefix ที่ยอมรับแล้ว และทิ้ง shadow state ของส่วนที่ถูกปฏิเสธ

ระหว่าง machine-only steps ห้ามขยับ language KV/GDN/PLE, logical position, RNG counters หรือ penalty history ข้อความผลลัพธ์ที่เสร็จแล้วจึงผ่าน exact language replay ครั้งเดียว แล้วค่อยให้ MTP และการตอบปกติทำงานต่อ

ต้องทดสอบ ENTER ทุกตำแหน่งใน window รวมแถวแรก แถวสุดท้าย rejected drafts และการสลับซ้ำ ส่วน execution modes ที่ยังไม่รองรับต้องปฏิเสธเมื่อเปิด feature โดยคงเส้นทางตอนปิดไว้

### การรับนิพจน์ต้องไม่มีการตัดข้อความเงียบ ๆ

เริ่มจาก native XML tool-call syntax และ eligibility/role gates ของโมเดลเดิม ตัวอย่างในประวัติ ข้อความที่ยกมา และ tool names อื่นต้องไม่เปิดวงจร

รองรับเลขและตัวดำเนินการตาม firmware ที่ตรึงไว้ รวม rational division, floor division/modulo แบบ Python, gcd, powers ที่เป็นจำนวนเต็มไม่ติดลบและมีขอบเขต ตลอดจน decimal operations ที่ firmware รองรับ

ขอบเขตเดิมมี 80 capture tokens, 16 bytes ต่อ row, expression payload 255 bytes และ `MAXD=62` ต้องปฏิเสธ overflow, NUL, UTF-8 ที่ผิด, syntax ที่ไม่รองรับ และข้อความที่ถูกตัด ถ้าจะขยายขอบเขต ต้องเปลี่ยน compiler ABI และชุด oracle ไปด้วย

### ประวัติและ session ต้องคืนสถานะได้ทั้งชุด

แต่ละบทสนทนามี RAM, PC, registers, lag history, trace cursor และข้อมูล bridge ของตัวเอง ต้องผูกข้อความที่แสดงกับ canonical internal token history ด้วย identity และ prefix hashes ที่ตรวจได้ ห้ามเดาประวัติที่ซ่อนอยู่จากข้อความที่ไม่ครบ

v1 save ได้ที่ coherent quiescent boundary เท่านั้น ถ้ายังมี burst ทำงาน ให้ตอบ busy/refused หรือรอผ่านคิวที่จำกัดไว้ การ restore ต้องตรวจ identity ก่อนแก้ state และถ้า transfer ได้เพียงบางส่วนก็ต้อง invalidate ทั้ง context ห้าม resume จากสถานะครึ่งชุด Graph handles ไม่อยู่ใน snapshot; เมื่อ address เปลี่ยนต้องสร้างใหม่

### API และหน้าจอต้องแสดงงานตามจริง

native call/result เป็นเรื่องภายใน เครื่องมืออื่นยัง dispatch แบบเดิม สถานะคำนวณต้องเริ่มแสดงตั้งแต่เริ่มขั้นตอน ไม่รอให้จบก่อน แล้วต้องตามงานเดิมต่อได้หลัง reconnect

ผลระหว่างทางเก็บไว้ก่อน ถ้า limit, cancel หรือ fault ต้องทิ้งผลที่ยังไม่ครบ แล้วกลับ transactional boundary ที่ถูกต้องหรือ invalidate session ห้ามส่งเลขบางส่วนในสถานะสำเร็จ

### Offload และงบหน่วยความจำยังเป็นข้อจำกัดเดิม

เริ่มจาก static reserve สำหรับวงจรบน GPU หลัก ไม่ใช้ elastic VRAM ในเส้นทางที่รองรับ D2x ไม่ได้ แยกไฟล์ cold storage บน NVMe ออกจาก working state ที่ต้อง resident ระหว่าง burst

ยังต้องรักษา D2x expert/prefill offload บนสอง GPU การไม่รองรับ decoder layer split ใน v1 ไม่ได้แปลว่าตัด secondary expert tier ออก ต้องวัดผลของ reserve ต่อ expert slots, hit split และ latency ด้วย

## Testing Decisions — จะพิสูจน์ว่าใช้ได้อย่างไร

จุดทดสอบหลักคือหนึ่ง conversation ที่เข้า native call ได้ รับผล แล้วให้โมเดลตอบต่อ จากจุดนี้ต้องตรวจ API, ประวัติ, session และภาษาได้ครบ ส่วน circuit replay และ kernel tests ใช้พิสูจน์รายละเอียดที่จำเป็น

ทดสอบการบังคับเรียกวงจรแยกจากการดูว่าโมเดลเลือกเรียกเองได้ดีแค่ไหน ต้องตรึง compiler, corpus และ seeds แล้วแยก development fixtures ออกจาก held-out free-running acceptance cases ด้วย

เมื่อเทียบ baseline กับ candidate ให้ใช้ Gyro-M ชุดเดียวกันที่ 262K context, 5 workers และค่า RoPE เดิมของ profile/GGUF ต้องตรึง model inputs, tokenizer, MTP และ runtime settings ให้เหมือนกัน ต่างเฉพาะ executable กับ paths สำหรับหลักฐานของแต่ละ arm ส่วน learned expert profile ให้เริ่มจากสำเนาที่มีเนื้อหาเดียวกัน เพื่อไม่ให้การรัน arm หนึ่งเปลี่ยน input ของอีก arm

ชุดเลขต้องมี signed add/sub/mul, rational division, negative floor/modulo, gcd, precedence, powers, decimals, ขอบเขต 62 หลัก, divide-by-zero, capacity/complexity errors และการคำนวณต่อเนื่อง Oracle เลขต้องเป็นอิสระจาก parser/compiler ของ firmware เพื่อไม่ให้ผิดแบบเดียวกันทั้งคู่

### Gates ที่ต้องผ่านก่อนนำไปใช้

| Gate | หลักฐานที่ต้องมี |
|---|---|
| **G0** | conditional graph จริงบน GPU หลัก: 0/1/N, relaunch/reset, output bound, stop/cancel พร้อม runtime, driver, device และ OS |
| **G1** | export/schema ทำซ้ำได้ hashes ตรง feature map ครบ มีที่มาและ license; ไฟล์ผิดหรือใหญ่เกินต้องหยุดก่อน allocate |
| **G2** | trace และผลเลขไม่ผิดจาก reference กับ oracle อิสระ ครบ instruction/error paths, repeated runs และ held-out cases; มีหลักฐาน deterministic reduction และผล Compute Sanitizer ตามที่รองรับ |
| **G3** | ทุกจุด ENTER/commit, MTP, rejected drafts และ suffix ถูกต้อง; machine trace ไม่ขยับสถานะภาษา และ replay ผลครั้งเดียว |
| **G4** | สลับสองบทสนทนา ใช้ cache อ่านประวัติใหม่ และ save/restore ได้โดยไม่มี state เก่าหรือ hidden prefix ปะปน |
| **G5** | bounds, input ผิด, cancel และ GPU fault ให้ terminal reason ชัด ไม่คืนเลขบางส่วนเป็นสำเร็จ; status และ refresh ผ่านจริง |
| **G6** | unchanged baseline กับ candidate ที่ปิด feature ให้ greedy raw tokens ตรงกันใน session เดียว รวมข้อความไทย โค้ด และ long-prefix turns; prompt, budget, protocol, snapshot และการไม่ allocate วงจรต้องตรง |
| **G7** | full D2x builds มี sm_89 และ sm_120 ตรวจ binary จริงและรัน secondary tier ก่อน ABBA; build ทุก backend ที่แตะและรักษา Q2 parity งาน HIP/SYCL ที่ยังทำไม่ได้ต้องระบุเป็นหนี้ validation |
| **G8** | ใช้ timeline อธิบายกลไก แล้ววัด ABBA แบบปิด profiler พร้อม counters จาก run เดียวกัน, hashes, priorities, resources, hit split, acceptance, entry/exit และ DONE แยกงานที่ถูกรบกวนออกมา |
| **G9** | ผู้ใช้ตรวจความถูกต้อง ต้นทุนทรัพยากร และ latency ก่อนเลือก rollout แบบ opt-in; ไม่แก้ served defaults หรือ deploy จาก PRD นี้ |

G2 ต้องรัน memcheck, initcheck, synccheck และ racecheck ในส่วนที่ใช้ได้ ถ้าเครื่องมือไม่รองรับก็ต้องบอกไว้ ผล racecheck อย่างเดียวไม่พิสูจน์ว่า global atomics ให้ผลแน่นอน

G8 ห้ามรวม oracle, reference หรือ checksum verification ไว้ในช่วงจับเวลา ให้ตรวจผลหลังจับเวลา และยืนยันว่า production path ไม่เรียก host arithmetic/reference ต้องรายงาน distribution, startup และ peak RAM/VRAM ด้วย ทุก arm ใช้ `--pool-priority 2` และ `--process-priority 2`

เกณฑ์ noise 13.6% เป็นกติกาประจำ repo ไม่ใช่ statistical significance ถ้าผลต่ำกว่าเกณฑ์นี้และยังไม่มี paired ABBA ที่สม่ำเสมอ ก็ยังยืนยันผลไม่ได้ งบ private commit 42 GB กับ headroom จอ 2.5 GB รวมการใช้ของโปรแกรมอื่นยังคงเดิม

## ลำดับงานและสิ่งที่จะส่งมอบ

| Phase | สิ่งที่จะได้ | เงื่อนไขก่อนเดินต่อ |
|---|---|---|
| **P0** | inventory/hashes, baseline build, CUDA probe และรายงานข้อจำกัดบนเครื่องจริง | ผู้ใช้ตรวจผลก่อน P1 |
| **P1** | compiler/tokenizer adapter ที่ตรึง revision, artifact และ oracle corpus | อนุมัติการรัน third-party code เมื่อจะใช้จริง |
| **P2** | loader ที่ตรวจไฟล์และถอด token features ได้ตรงสัญญา | artifact ที่ผิดต้องถูกปฏิเสธก่อนแก้ state |
| **P3** | native CUDA circuit step ที่ตรง reference | ผ่าน numerical proof/replay และ sanitizer gates |
| **P4** | GPU trace runner ที่จำกัดงานและ cancel ได้ | ยืนยัน graph, limits และ repeated execution จริง |
| **P5** | ตัวเชื่อม verifier/commit/MTP/suffix และประวัติ prompt | ownership transitions ครบและสถานะภาษาคงเดิมระหว่าง trace |
| **P6** | canonical history และ snapshot ของโมเดลกับวงจร | ทดสอบหลายบทสนทนาและ restore failures |
| **P7** | API, internal tool handling, status และ reconnect | ตรวจผ่าน API/UI จริง |
| **P8** | หลักฐาน correctness, resources, latency และข้อเสนอ rollout | ผ่าน gates และให้ผู้ใช้เลือกก่อนนำไปใช้ |

P2–P4 ใช้ artifact/ABI จาก P1; P5 ต้องได้ circuit runner ที่ผ่านการตรวจแล้ว; P6–P7 ใช้สัญญา ownership จาก P5 ส่วน P8 ตรวจระบบที่รวมครบ ไม่ใช้ผล microbenchmark แทนผลของหนึ่ง turn

## Out of Scope — งานที่ยังไม่รวม

Universal adapters, การรับโมเดลทุกตระกูล, การดาวน์โหลด merged Qwen3.5, host calculator สำหรับคำตอบจริง, การเปลี่ยน quantization หรือขนาด context เงียบ ๆ, การปรับ TDR/ความปลอดภัย, การเปลี่ยน defaults/deploy และข้ออ้างว่าคุณภาพโค้ดหรือ 3D ดีขึ้นโดยยังไม่ทดสอบ

v1 ยังไม่รับทุก execution mode ต้องรองรับหรือปฏิเสธแต่ละชุดอย่างชัดเจน การใช้ storage เก็บไฟล์ไม่ได้ทำให้ working state ที่วงจรต้องใช้บน GPU หายไป

## Further Notes — หลักฐานปัจจุบันและรายละเอียดสำหรับ implementation

พบ Gyro-M GGUF v3 จริง ขนาด 91,965,774,272 bytes ชื่อ metadata คือ `Qwen3.8-Flash-Next Gyro-M`, architecture `qwen4exp`, 48 layers, hidden size 2560 และ 512 experts เลือกใช้ 10 ตัว Tensor experts เป็น `TQ2_T`; ยังมี `Q8_0`, `BF16` และ `F32` ในไฟล์ด้วย จึงไม่ใช้สมมติฐาน IQ2_XS ของ Swift แทน

tokenizer ที่พบมี vocabulary 248,320 ใช้ descriptor แบบ `gpt2`/`qwen35` และมีไฟล์ vocabulary/merges แยก จำนวน vocabulary หรือ geometry ที่เท่ากันอย่างเดียวใช้ยืนยันความเข้ากันไม่ได้ ขณะเตรียม P0 ได้ตรวจเนื้อหา vocabulary, merges และ token types ของ pack กับ GGUF แล้วว่าตรงกันทุกค่า และบันทึก special IDs จริงแยกไว้

ยังต้องผูก circuit artifact กับ tokenizer และ special IDs ชุดนี้ แล้วพิสูจน์การรัน TQ2_T และ MTP จริง การมีไฟล์หรือ metadata ตรงไม่ใช่หลักฐาน runtime parity ส่วน Q2 regression tests เดิมยังต้องรักษาไว้ แต่ใช้แทนหลักฐานความถูกต้องของ Gyro-M ไม่ได้

P0 build baseline และ candidate ครบแล้ว Probe รุ่นสุดท้ายสร้างและ instantiate graph จริง และผ่านทั้ง 10 cases ครอบคลุม zero/one/N, relaunch, output bound, cancel, current-device switch และการสร้าง graph ใหม่ ส่วนชุดทดสอบรวมสุดท้ายผ่าน 9 tests ใน 3.814 วินาที ครอบคลุม preflight, launcher และ probe รวมการตรวจ negative control ด้วย การโหลด Gyro-M, greedy parity, ABBA และยอดใช้หน่วยความจำของโมเดลยังไม่รัน

รายละเอียดระดับ schema, state, token bridge, module inventory, primary sources และข้อจำกัด CUDA อยู่ใน [ข้อกำหนด R2 ที่เก็บไว้ครบ](2026-10-28-native-qwenputer-r2-reference.md) ฉบับอ้างอิงนี้ยังเป็นข้อกำหนดสำหรับ implementation และมี SHA256 `d68fad629d75bc4d3f0f4915a37a0ebdb8478155cd1e298d64e2efb6fc0f7634`

ถ้า R2 กับ R3 ต่างกัน ให้ใช้ R3 แทนเฉพาะสี่รายการนี้:

1. โมเดลทดสอบเปลี่ยนจาก Swift เป็น Gyro-M
2. ใช้ profile 262K context, 5 workers ที่ผู้ใช้เลือก โดยตรึง RoPE และ runtime settings เดิมในการเทียบแต่ละ arm
3. ไฟล์ Swift ที่หายเป็นผลสำรวจในอดีต ไม่ต้อง restore เพื่อ baseline ใหม่
4. ใช้สถานะหลักฐาน P0 ที่ตรวจแล้วในฉบับนี้ แทนสถานะก่อนรันที่บันทึกไว้ใน R2

ข้อกำหนดอื่นของ R2 ยังมีผลครบ ผล build และ conditional probe ยืนยันได้เฉพาะความเป็นไปได้ของส่วนที่ทดสอบ ยังไม่พิสูจน์ model inference, ownership correctness, การลงตัวกับงบทรัพยากร หรือความพร้อมด้าน performance และไม่เปลี่ยน identity ของผลทดสอบเก่า

ใช้ [แผน P0–P8](../superpowers/plans/2026-10-28-native-qwenputer.md) คู่กับ [รายงาน P0](../reports/2026-10-28-qwenputer-feasibility.md) ก่อนเริ่มแต่ละ phase ส่วน [coverage checklist](2026-10-28-native-qwenputer-r3-coverage.md) ใช้ตรวจว่าการเขียนให้อ่านง่ายไม่ได้ตัดข้อกำหนด R2 ออก
