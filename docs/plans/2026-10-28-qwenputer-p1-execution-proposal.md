# P1 — คำสั่ง upstream แรกที่เสนอให้ตรวจอนุมัติ

เอกสารนี้เตรียมจากการอ่าน source เท่านั้น ยังไม่ได้รัน upstream code และยังไม่อนุมัติ P1 implementation การอนุมัติที่ต้องขอเป็นการ execute structural circuit compiler ครั้งแรก ไม่ใช่การดาวน์โหลดโมเดลหรือรัน firmware

## แหล่งที่มาและขอบเขต

Repository: `https://github.com/divagr18/qwenputer`

Revision ที่ต้องยืนยันก่อน execute: `ca13310e2f9f788d1dd3feeea8213c1b0c23913c`

Compiler source มี [MIT license ที่ revision นี้](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/LICENSE) ต้องเก็บ copyright/license notice ไว้ ส่วน firmware และ dependencies ยังต้องตรวจ notices/provenance สำหรับขอบเขตที่จะใช้จริง ไม่ถือว่า source license เป็นการอนุมัติ execute

Entry point คือ [`chipformer.rv.core.build(target=None, text=None)`](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/core.py) ซึ่งสร้าง standalone circuit IR เมื่อไม่ส่ง text ค่านี้ใช้ default geometry ของ upstream Qwen3.5-4B จึงเป็น structural probe ยังไม่ใช่ Gyro-M adapter หรือ artifact ของ Strata

ชุด source ที่ import ตาม static inspection: package initializers ของ `chipformer`, `dsl`, `q35`, `rv`; `dsl/graph.py`; `q35/ir.py`, `q35/prims.py`; `rv/core.py`, `rv/ref.py`, `rv/spec.py` ต้อง materialize source ที่ pin, เก็บ hashes/license และตรวจ import closure ก่อนเริ่ม ไม่ใช้ไฟล์จาก branch main ที่เปลี่ยนได้

NumPy เป็น dependency ที่จำเป็น ส่วน [`q35/__init__.py`](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/q35/__init__.py) จะลอง import Torch แล้วปิด reduced-precision BF16 reduction ใน process ลูกด้วย แม้คำสั่งนี้ไม่ได้ใช้ Torch สร้าง model ตรวจ installed metadata แล้วพบ NumPy 2.4.6 และ Torch 2.14.0+cpu ไม่ได้ติดตั้งเพิ่มหรือ import upstream เพื่อสำรวจ

ไม่ต้องใช้ Transformers, HF loader, merged checkpoint, Triton หรือ text bridge สำหรับคำสั่งนี้ ไม่เรียก RvChat เพราะเส้นทางนั้นโหลด model/tokenizer ด้วย from_pretrained

## คำสั่งที่เสนอ

หลัง materialize reviewed checkout ที่ revision ข้างต้นเท่านั้น ใช้ working directory:

```text
C:\Users\xenod\AppData\Local\Temp\strata-qwenputer-p0-20261028\upstream-qwenputer-ca13310\experimental\riscv
```

คำสั่ง upstream ที่ขออนุมัติให้ทำครั้งเดียว:

```text
C:\Users\xenod\AppData\Local\Programs\Python\Python311\python.exe -B -c "from chipformer.rv import core; c = core.build(); print(c.stats())"
```

Directory นี้ยังไม่ได้ materialize จึงไม่อ้างว่าคำสั่งพร้อมรันอยู่แล้ว ใช้ task-owned headless launcher เดิมกำหนด working directory นี้, log ลง private evidence, เก็บ memory telemetry และ emergency timeout 60 วินาที ไม่มี private-commit cap ที่แอบเปลี่ยน baseline; คำสั่งนี้ไม่ใช่ model baseline หรือ performance test หาก preflight ไม่ปลอดภัยให้ไม่ launch ไม่ใช้วิธีเพิ่ม memory/pagefile หรือหยุดงานอื่น

ผลที่คาดตาม source คือ circuit object ใน RAM แล้วพิมพ์ structural statistics จาก [`Graph.stats()`](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/q35/ir.py) Runtime, RAM และสถิติจริงยังไม่เคยวัด ไม่กำหนดตัวเลขสำเร็จขึ้นเอง ไม่มี persistent artifact จากคำสั่งโดยตรง นอกจาก log/telemetry ของ wrapper และไม่โหลด model/firmware หรือรัน arithmetic trace

## สิ่งที่ต้องอนุมัติแยกภายหลัง

Firmware `tests/data/rv/calc-bridge.elf` ที่ pin มีขนาด 29,932 bytes, Git blob `6698dd454c677ce2759610e9c5fed4eb9e1959eb`; `calc.elf` 29,348 bytes, blob `aeaa4c4de6ff44184b301ec480de972c7f98ef95` ค่าเหล่านี้เป็น Git blob IDs ไม่ใช่ SHA256 ยังไม่ได้ execute

`elf.program()` อ่าน ELF และ `Machine.from_program()` สร้าง state ส่วน `bridge.park()` เริ่ม reference simulation และ execute firmware จึงไม่อยู่ในการอนุมัติ structural command ข้างต้น การ rebuild firmware, เรียก simulator, สร้าง corpus หรือ export artifact ต้องมีคำสั่งและขอบเขตให้ตรวจแยกเมื่อพร้อม ห้ามถือว่าการอนุมัติ structural probe ครอบคลุมทั้งหมด

Host reference จะใช้เป็น oracle เท่านั้น ไม่ใช้แทน production FFN/GDN circuit งาน Native ยังต้องผ่าน artifact validation, exact numerics, GPU trace ownership, tokenizer binding และ disabled parity ตาม PRD เดิม
