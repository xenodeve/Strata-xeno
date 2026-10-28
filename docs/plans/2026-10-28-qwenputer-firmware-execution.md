# P1 — ขอบเขตการ execute compiler และ firmware ขั้นถัดไป

**สถานะ:** ผู้ใช้อนุมัติ pinned compiler/firmware digest นี้สำหรับ local export/oracles และ bounded Native tests ตามขอบเขตแล้ว คำสั่ง bootstrap ด้านล่างรันผ่านจริง: 62 trace tokens, 50 instructions และตั้งต้น 99 tables ดู [ผล P1](../reports/2026-10-28-qwenputer-p1-inputs.md) ไม่มีการอนุมัติ publish/deploy/เปลี่ยน served defaults เพิ่ม

การอนุมัติครั้งก่อนครอบคลุมคำสั่ง standalone `core.build(); print(c.stats())` ซึ่งรันผ่านแล้ว เอกสารนี้ขออนุมัติส่วนที่เพิ่มขึ้นจริง: compile text bridge ด้วย Gyro tokenizer adapter และ execute calc-bridge firmware ที่ pin ไว้เพื่อเตรียม boot state/oracles และทดสอบ Native ตาม PRD ในเครื่อง การอนุมัตินี้ไม่ใช่การเปลี่ยนสถาปัตยกรรมหรือเปิด feature ใน served setup

## Inputs ที่ตรวจแล้ว

- Source: divagr18/qwenputer revision `ca13310e2f9f788d1dd3feeea8213c1b0c23913c`
- Vendored source: 13 ไฟล์ที่ไม่แก้จาก upstream รวม MIT notice; `UPSTREAM.json` SHA256 `69cb09356ea256f3094173779b7ee6391b8f3906f403d394f7107929e43d55c8` ตัว verifier ตรวจ Git blob/SHA256 ทุกไฟล์และปฏิเสธ extra .pyd/.pyc ก่อน import
- Imports: Python standard library, NumPy และ Torch ที่ติดตั้งอยู่แล้ว; `chipformer` package initializers, `dsl.graph`, `q35.ir`, `q35.prims`, `rv.core`, `rv.spec`, `rv.ref`, `rv.asm`, `rv.elf` ไม่มี Transformers/HF loader/Triton หรือ language model
- Firmware: `experimental/riscv/tests/data/rv/calc-bridge.elf`, 29,932 bytes, Git blob `6698dd454c677ce2759610e9c5fed4eb9e1959eb`, SHA256 `b643ad61291a947fa6c7498de9170b1d4da062dc1f34810feba9c4eebdec862d`
- Gyro-M tokenizer: 248,320 IDs, binding SHA256 `889964f986b93c35102f25f5e396efeed65fcb180429c35d97536fb7f6b72106` จาก descriptor, vocab, merges, types และ chat template
- Data-only preflight ผ่านแล้ว: ELF32 little-endian, RV32 machine, entry `0x40000`, segments/permissions/bounds ถูกต้อง ไม่ได้ execute compiler หรือ firmware จาก preflight

## คำสั่งแรกที่เสนอให้ execute

Working directory: `D:\Github\Strata\.worktrees\native-qwenputer-p0`

```powershell
python -B -X utf8 tools/qwenputer/headless.py --log C:/Users/xenod/AppData/Local/Temp/strata-qwenputer-p0-20261028/p1-bootstrap-execute.log --timeout 120 -- python -B -X utf8 -m tools.qwenputer.bootstrap --tokenizer-dir D:/Github/Strata/packs/gyro-m/tokenizer --firmware C:/Users/xenod/AppData/Local/Temp/strata-qwenputer-p0-20261028/upstream-qwenputer-review-ca13310/experimental/riscv/tests/data/rv/calc-bridge.elf --output C:/Users/xenod/AppData/Local/Temp/strata-qwenputer-p0-20261028/p1-bootstrap-01 --max-boot-tokens 200000 --execute-reviewed-reference
```

`bootstrap.py` ตรวจ source/input อีกครั้งแล้วเรียก `core.build(text=NativeText)`, `elf.program`, `ref.Machine.from_program`, `Machine.run` และ `core.snapshot` เมื่อเลือก explicit execution flag เท่านั้น Reference machine จำลอง ISA ใน process ลูก ไม่ใช่การให้ Windows execute ELF โดยตรง และจะใช้เป็น offline oracle เท่านั้น Production ยังต้องคำนวณด้วย FFN/GDN ใน C++/CUDA

Boot จำกัด 200,000 trace tokens และ wrapper มี emergency timeout 120 วินาที Source run อาจสร้าง record สุดท้ายเกิน token budget เล็กน้อย จึงตรวจจำนวนหลังรันด้วยและไม่ publish ว่า success หากเกิน ไม่เพิ่ม private-commit cap, ไม่ติดตั้ง dependency, ไม่โหลด language model และไม่เปลี่ยน pagefile/TDR/security

Outputs อยู่ใน task-owned `p1-bootstrap-01`: validated firmware copy, logical boot state JSON, boot trace JSON และ manifest ที่เขียนหลังสำเร็จ คาด RAM ไม่เกินประมาณ 2 GB และ disk ต่ำกว่า 8 MB จากขนาด inputs/โครงสร้าง; เป็น estimate ยังไม่ใช่ค่าที่วัด branch นี้ Standalone structural run ก่อนหน้ามี process private peak 890,548,224 bytes

## การใช้งานต่อภายใต้ scope เดิม

ขอให้การอนุมัติระบุ pinned compiler subset และ firmware digest นี้สำหรับ local compiler/export/reference/corpus และ bounded Native GPU tests ตาม P1–P8 เพื่อไม่ต้องถามซ้ำเมื่อเปลี่ยนเพียง test case คงแยก oracles ออกจาก timed intervals และ production execution และต้องขอใหม่เมื่อ source/dependency/firmware เปลี่ยนเป็น code ที่ยังไม่อนุมัติ

Explicit execution branch รันผ่าน offline boot แล้ว และ reference arithmetic `17*23` ได้ `391` ตรง expected ทั้งคู่ยังไม่ใช่ native CUDA arithmetic Code ของ preflight, byte adapter, ELF validator และ source trust checks มี tests แยกที่ไม่ execute upstream

Firmware build script อ้าง picolibc และ libgcc; ELF ระบุ GCC 13.2.0 แต่ยังไม่ยืนยัน build/toolchain provenance และ notices ของ linked runtime ครบ จึงยังไม่เผยแพร่ generated binaries การทดสอบในเครื่องนี้ไม่ใช่การอนุมัติ redistribute, push, merge, deploy หรือแก้ served defaults
