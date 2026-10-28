# Native Qwenputer P1 — input binding และ offline bootstrap

P1 ยังไม่เสร็จ งานช่วงนี้ทำ tokenizer adapter, ELF/source guards และ offline boot state ที่ exporter จะใช้ ยังไม่มี native.qput, CUDA arithmetic หรือ decoder integration

## สิ่งที่ implement แล้ว

- `tools/qwenputer/tokenizer.py`: อ่าน Strata descriptor/vocab/merges/types/template, bind ทุกไฟล์ด้วย SHA256, เก็บ raw bytes ครบทุก ID, แยก 16-byte compiler projection ออกจาก full capture validation ตรวจ IDs, reserved tokens, NUL, width/count/byte budgets และ UTF-8 หลังรวม bytes เพื่อรองรับอักษรที่แยกข้าม token
- `firmware.py`: ตรวจ ELF32 little-endian/RV32, header table, file/region bounds, memsz/filesz, permissions, alignment, segment overlap และ initialized executable entry ก่อนขยาย state ทุกการอ่าน firmware รวมรอบสองยังถูกจำกัด 4 MiB
- `source.py`: pin manifest digest แยกจาก manifest เอง ตรวจ 13 source files ทั้ง SHA256/Git blob และห้าม extra import files/directories หรือ Windows reparse points ไม่ให้ dependency ที่ไม่ได้ review บัง source ที่ pin
- `bootstrap.py`: ค่าเริ่มต้นเป็น data-only preflight ส่วน compiler/firmware import และ execution ต้องเลือก explicit flag จึงทำงาน เขียน logical boot state/trace/manifest ใน task output แยกจาก production
- `third_party/qwenputer/`: reviewed MIT compiler subset ที่ revision `ca13310e2f9f788d1dd3feeea8213c1b0c23913c` ไม่มี firmware binary, model, HF loader หรือ Triton executor มี .gitattributes รักษา byte identity เพราะเครื่องใช้ core.autocrlf=true

## ผลที่รันจริง

Standalone structural command exit 0: 11 layers, 2,679 dims, 99 heads, 42,014 FFN units; process private peak 890,548,224 bytes รุ่น wrapper แรกพิมพ์ stats แล้วเกิด NoSuchProcess ระหว่างเก็บ telemetry เพราะ worker ออกจากระบบพอดี แก้ race แล้วรันคำสั่งเดิมผ่าน เก็บ raw failure ไว้ ไม่เรียกข้อผิดพลาดของ wrapper ว่า compiler failure

หลังได้รับ explicit approval สำหรับ compiler/firmware digest แล้ว `p1-bootstrap-execute.log` exit 0: Gyro text bridge 11 layers, 2,918 dims, 103 heads, 42,114 FFN units; boot 62 trace tokens / 50 instructions และตั้งต้น 99 tables

| Input/ผล | Identity |
|---|---|
| Compiler manifest used by bootstrap-01 (historical CRLF metadata) | `69cb09356ea256f3094173779b7ee6391b8f3906f403d394f7107929e43d55c8` |
| Current canonical LF compiler manifest | `a5519351051213218a2236f1a4737af0e26e310fe3fb74d3f9269ef5bfd8c7c5` |
| Firmware | `b643ad61291a947fa6c7498de9170b1d4da062dc1f34810feba9c4eebdec862d` |
| Gyro tokenizer binding | `889964f986b93c35102f25f5e396efeed65fcb180429c35d97536fb7f6b72106` |
| Logical boot state | `c2b9b2715f83a186c9c7b84a8559cad78f40fb8703a0c3591a277af833b72843` |
| Boot trace | `f93d8e46ce0697518d9f025cf0ff4bdcc65d197742ce14b4ee15a6ca9482d779` |

Metadata correction after checkpoint `0217dc22`: `git diff --cached --check` reported CRLF whitespace in the task-owned README and manifest. The commit was made before that nonzero result was handled. Only these two metadata files are normalized to LF, with explicit Git attributes; all 13 pinned upstream source files retain their original bytes and hashes. Raw bootstrap-01 evidence retains its historical manifest digest. The source verifier now pins the canonical LF digest above.

Reference arithmetic smoke: input `17*23`, expected response body `\n391\n`, actual `\n391\n`, PASS บน pinned firmware จาก boot state ได้ 1,505 trace tokens / 912 instructions ใช้ 3.786 ms เฉพาะ Machine.record loop บน i5-13500 ไม่รวม imports/tokenization/boot-state load นี่เป็น offline upstream/reference result ไม่ใช่ native CUDA, integrated model, end-to-end latency หรือ speed comparison

## Tests และ review

Final regression suite `p1-preparation-full-suite-02.log`: 63 tests, 60 pass / 3 skip, exit 0 ใน 6.029 วินาที รวม actual Gyro tokenizer ทั้ง 248,320 IDs, pinned ELF data, source trust tests, existing P0 tools และ tokenizer regressions

Skipped: GPU probe integration 2 tests (มีผลจริงแยกใน P0 แล้ว) และ privileged Windows symbolic-link fixture 1 test เพราะ WinError 1314 ส่วน Windows junction fixture รันและผ่านจริง ไม่เปลี่ยน privileges หรือ security settings เพื่อให้ test ผ่าน

Review พบ 3 จุดที่แก้ก่อน checkpoint: import-directory shadowing, unbounded firmware reread และ malformed UTF-8 capture แต่ละจุดมี named RED→GREEN regression tests ข้อแรกมี Windows junction control เพิ่ม ทุก raw logs อยู่ใน private evidence root เดิม

| Gate ของ subitem นี้ | ผล |
|---|---|
| simplify | named skill not-run: ไม่พบใน local skill roots; manual pass ใช้ existing tokenizer และ bounded readers โดยไม่เพิ่ม framework |
| verify | ran: 63-test suite, actual bootstrap และ reference arithmetic smoke |
| code-review | ran: Standards และ Spec อ่าน diff; แก้ findings พร้อม tests |
| scrutinize | ran: owner trace ตั้งแต่ input binding → source verification → explicit execution branch → bounded boot → output manifest |
| security-review | named skill not-run: ไม่พบ; scoped boundary review และ malformed-input/import-shadowing tests รันแล้ว ไม่ได้ติดตั้ง audit tool เพิ่ม |

## สิ่งที่ยังไม่เสร็จ

Export schema/coefficients/state แบบ data-only, exhaustive token-feature equivalence, independent mathematical corpus, native C++ loader และ CUDA step/runner ยังต้องทำตาม P1–P4 ส่วน ownership/history/API/UI/parity/performance อยู่ P5–P8 ไม่มีการเปลี่ยน hub/served defaults, push, PR หรือ deployment

Compiler source มี MIT notice แต่ linked firmware ใช้ picolibc/libgcc และยังไม่ยืนยัน notices/build provenance ครบ จึงยังไม่เผยแพร่ generated binaries Source code, firmware digest และขอบเขต local testing ได้รับอนุมัติแล้ว ไม่ต้องถาม phase ซ้ำสำหรับ worklist เดิม ต้องขอใหม่หากเปลี่ยนเป็น unrecognized source/firmware หรือ protected action อื่น
