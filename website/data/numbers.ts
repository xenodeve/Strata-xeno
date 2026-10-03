import type { ChartSpec } from "@/components/charts/Chart";
import type { L } from "@/lib/i18n";
import { issue, pr } from "./sources";

export type Measured = {
  id: string;
  label: L;
  /** The headline figures, each with its own unit. */
  figures: { value: string; unit: string; caption: L }[];
  /** The conditions, in full, shown with the number. Never shortened away. */
  conditions: { k: L; v: L }[];
  source: { label: string; href: string }[];
};

export const MEASURED: Measured[] = [
  {
    id: "ram",
    label: { en: "RAM follows VRAM", th: "RAM ลดตาม VRAM" },
    figures: [
      { value: "16.5", unit: "GiB", caption: { en: "Strata-xeno, two GPUs", th: "Strata-xeno สอง GPU" } },
      { value: "35.3", unit: "GiB", caption: { en: "upstream v0.1.26, two-GPU layer split", th: "upstream v0.1.26 แบบ layer split สอง GPU" } },
      { value: "23.2", unit: "GiB", caption: { en: "Strata-xeno, one GPU", th: "Strata-xeno หนึ่ง GPU" } },
      { value: "33.2", unit: "GiB", caption: { en: "upstream v0.1.26, one-GPU default", th: "upstream v0.1.26 ค่าเริ่มต้นหนึ่ง GPU" } },
    ],
    conditions: [
      { k: { en: "What was measured", th: "สิ่งที่วัด" }, v: { en: "Process working set (resident RAM). Not private commit: under Windows WDDM, commit also counts VRAM.", th: "working set ของโปรเซส (RAM ที่ใช้อยู่จริง) ไม่ใช่ private commit ซึ่งบน Windows WDDM นับ VRAM รวมด้วย" } },
      { k: { en: "Machine", th: "เครื่อง" }, v: { en: "Intel Core i5-13500, 48 GB DDR5, RTX 5060 Ti 16 GB + RTX 4070 SUPER 12 GB, Windows 11.", th: "Intel Core i5-13500, DDR5 48 GB, RTX 5060 Ti 16 GB + RTX 4070 SUPER 12 GB, Windows 11" } },
      { k: { en: "Model", th: "โมเดล" }, v: { en: "Q2_0 pack; greedy 256-token decode runs plus an 8K prompt; upstream's setup used a 16,384-token context limit.", th: "pack แบบ Q2_0; decode แบบ greedy 256 โทเคน และ prompt 8K; setup ของ upstream ใช้ context สูงสุด 16,384 โทเคน" } },
      { k: { en: "Compared with", th: "เทียบกับ" }, v: { en: "Upstream Strata v0.1.26, as its own setup chooses it, in the same session (2026-09-30), two runs per arm. Code decode speed was equal (82–89 vs 84–87 tok/s).", th: "Strata upstream v0.1.26 ตามที่ setup ของมันเลือกเอง ในเซสชันเดียวกัน (2026-09-30) สองรอบต่อฝั่ง ความเร็ว decode ของโค้ดเท่ากัน (82–89 เทียบ 84–87 tok/s)" } },
      { k: { en: "What it is", th: "คืออะไร" }, v: { en: "The whole dynamic-experts package together, not one mechanism. RAM saved is the size of the experts the GPUs hold, so it depends on how much VRAM is free.", th: "ผลรวมของชุด dynamic experts ทั้งหมด ไม่ใช่กลไกเดียว RAM ที่ประหยัดเท่ากับขนาด expert ที่ GPU ถือ จึงขึ้นกับว่ามี VRAM ว่างเท่าไร" } },
      { k: { en: "Not measured", th: "ยังไม่ได้วัด" }, v: { en: "The current IQ2_XS model, upstream v0.1.37 / v0.1.38, or upstream's own low-RAM modes.", th: "โมเดล IQ2_XS ปัจจุบัน, upstream v0.1.37 / v0.1.38 หรือโหมด low-RAM ของ upstream เอง" } },
    ],
    source: [
      { label: "Issue #45", href: issue(45) },
      { label: "Issue #4", href: issue(4) },
    ],
  },
  {
    id: "decode",
    label: { en: "Decode speed on the daily profile", th: "ความเร็ว decode ของโปรไฟล์ประจำวัน" },
    figures: [
      { value: "59.0 / 58.9", unit: "tok/s", caption: { en: "Strata-xeno, two runs", th: "Strata-xeno สองรอบ" } },
      { value: "41.0 / 42.5", unit: "tok/s", caption: { en: "upstream v0.1.37, layer split, two runs", th: "upstream v0.1.37 แบบ layer split สองรอบ" } },
    ],
    conditions: [
      { k: { en: "What was measured", th: "สิ่งที่วัด" }, v: { en: "Decode speed of a 400-token answer to a short prompt.", th: "ความเร็ว decode ของคำตอบ 400 โทเคนต่อ prompt สั้น" } },
      { k: { en: "Model and mode", th: "โมเดลและโหมด" }, v: { en: "Swift 1.5 IQ2_XS, serve mode, 262,144-token context.", th: "Swift 1.5 IQ2_XS, โหมด serve, context 262,144 โทเคน" } },
      { k: { en: "Machine", th: "เครื่อง" }, v: { en: "Intel Core i5-13500, 48 GB, RTX 5060 Ti ×4 + RTX 4070 SUPER ×16, Windows 11.", th: "Intel Core i5-13500, 48 GB, RTX 5060 Ti ×4 + RTX 4070 SUPER ×16, Windows 11" } },
      { k: { en: "Arms", th: "ฝั่งที่เทียบ" }, v: { en: "Strata-xeno's daily profile as served at the time (the 4070 as an exclusive expert tier, before PRs #113 and #115), against a pristine upstream v0.1.37 build with its layer split. Same script, same session, 2026-10-03.", th: "โปรไฟล์ประจำวันของ Strata-xeno ตอนนั้น (4070 เป็นชั้น expert แบบเฉพาะ ก่อน PR #113 และ #115) เทียบกับ upstream v0.1.37 ที่ build ใหม่พร้อม layer split ของมัน สคริปต์เดียวกัน เซสชันเดียวกัน 2026-10-03" } },
      { k: { en: "Same run, other side", th: "ในการรันเดียวกัน อีกด้าน" }, v: { en: "Upstream's layer split read a 34K-token prompt faster (1,220–1,616 vs 843–877 tok/s). This figure is about decode only. On Q2_0 at 16K context the same comparison was a tie.", th: "layer split ของ upstream อ่าน prompt 34K โทเคนได้เร็วกว่า (1,220–1,616 เทียบ 843–877 tok/s) ตัวเลขนี้พูดถึง decode เท่านั้น และบน Q2_0 ที่ context 16K การเทียบแบบเดียวกันเสมอกัน" } },
      { k: { en: "Caveat", th: "ข้อควรระวัง" }, v: { en: "Serve-mode numbers carry an open slow-state issue (#57).", th: "ตัวเลขโหมด serve มีปัญหา slow-state ที่ยังเปิดอยู่ (#57)" } },
    ],
    source: [
      { label: "Issue #112", href: issue(112) },
      { label: "PR #114", href: pr(114) },
    ],
  },
];

export type Count = { value: string; label: L; method: string };

/** Counts taken from the code at the revision stamped on the page. */
export const COUNTS: Count[] = [
  { value: "925", label: { en: "Python tests in serve/", th: "Python test ใน serve/" }, method: "grep -c '^\\s*def test_' over serve/test_*.py" },
  { value: "450", label: { en: "web-app unit tests (bun test)", th: "unit test ของเว็บแอป (bun test)" }, method: "serve/ui, 30 test files" },
  { value: "56", label: { en: "browser checks (end to end)", th: "การตรวจในเบราว์เซอร์ (end to end)" }, method: "serve/ui/e2e/checks.mjs" },
  { value: "40", label: { en: "engine test targets (xeno_*)", th: "test target ของเอนจิน (xeno_*)" }, method: "CMake targets added by the fork" },
  { value: "30", label: { en: "new server modules", th: "โมดูลฝั่งเซิร์ฟเวอร์ใหม่" }, method: "serve/*.py, non-test, absent from upstream" },
  { value: "28", label: { en: "new HTTP routes", th: "HTTP route ใหม่" }, method: "do_GET / do_POST pairs in serve/server.py" },
];

/**
 * The comparison charts. Every bar is a figure that is written out above, with the conditions it was measured under; the
 * differences are plain subtraction of two of those figures. A scale always starts at zero.
 */
export type ChartId = "ram" | "decode" | "capacity";

export const CHARTS: Record<ChartId, ChartSpec> = {
  ram: {
    id: "ram",
    title: { en: "RAM in use while the model runs", th: "RAM ที่ใช้อยู่ขณะโมเดลทำงาน" },
    unit: "GiB",
    better: "lower",
    groups: [
      { en: "Two GPUs", th: "สอง GPU" },
      { en: "One GPU", th: "หนึ่ง GPU" },
    ],
    series: [
      { name: { en: "Strata-xeno", th: "Strata-xeno" }, tone: "mine", values: [16.5, 23.2] },
      { name: { en: "upstream v0.1.26", th: "upstream v0.1.26" }, tone: "theirs", values: [35.3, 33.2] },
    ],
    max: 40,
    step: 10,
    deltas: [
      { en: "18.8 GiB less", th: "น้อยกว่า 18.8 GiB" },
      { en: "10.0 GiB less", th: "น้อยกว่า 10.0 GiB" },
    ],
    caption: {
      en: "Resident RAM (process working set), Q2_0, same session (2026-09-30), two runs per side, one PC. Upstream as its own setup chooses it. Measured on that version only.",
      th: "RAM ที่ใช้อยู่จริง (working set ของโปรเซส) Q2_0 เซสชันเดียวกัน (2026-09-30) สองรอบต่อฝั่ง พีซีเครื่องเดียว upstream ตามที่ setup ของมันเลือกเอง วัดกับเวอร์ชันนั้นเท่านั้น",
    },
  },
  decode: {
    id: "decode",
    title: { en: "Decode speed, two runs of each", th: "ความเร็ว decode สองรอบต่อฝั่ง" },
    unit: "tok/s",
    better: "higher",
    groups: [
      { en: "Run 1", th: "รอบที่ 1" },
      { en: "Run 2", th: "รอบที่ 2" },
    ],
    series: [
      { name: { en: "Strata-xeno", th: "Strata-xeno" }, tone: "mine", values: [59.0, 58.9] },
      { name: { en: "upstream v0.1.37", th: "upstream v0.1.37" }, tone: "theirs", values: [41.0, 42.5] },
    ],
    max: 70,
    step: 10,
    deltas: [
      { en: "+18.0 tok/s", th: "+18.0 tok/s" },
      { en: "+16.4 tok/s", th: "+16.4 tok/s" },
    ],
    caption: {
      en: "IQ2_XS, serve mode, 262,144-token context, a 400-token answer; same script, same session (2026-10-03), one PC. Decode only, not prompt reading.",
      th: "IQ2_XS โหมด serve context 262,144 โทเคน คำตอบ 400 โทเคน สคริปต์เดียวกัน เซสชันเดียวกัน (2026-10-03) พีซีเครื่องเดียว เฉพาะ decode ไม่ใช่การอ่าน prompt",
    },
  },
  capacity: {
    id: "capacity",
    title: { en: "Decode speed as the RAM cache grows (relative)", th: "ความเร็ว decode เมื่อ RAM cache ใหญ่ขึ้น (สัมพัทธ์)" },
    unit: "",
    better: "higher",
    groups: [
      { en: "RAM cache 6 GiB", th: "RAM cache 6 GiB" },
      { en: "RAM cache 12 GiB", th: "RAM cache 12 GiB" },
    ],
    series: [{ name: { en: "Capacity mode, 6 GiB = 100", th: "Capacity mode, 6 GiB = 100" }, tone: "mine", values: [100, 151.5], labels: ["100", "about 151–152"] }],
    max: 160,
    step: 40,
    digits: 0,
    caption: {
      en: "IQ3_XXS (about 40 GiB of experts), both GPUs, 512 tokens, a 48 GB PC. Only the percentage is recorded (about 51–52 %), so the bars are relative. No comparison with other low-RAM modes is recorded, so none is made.",
      th: "IQ3_XXS (expert ราว 40 GiB) สอง GPU 512 โทเคน พีซี 48 GB บันทึกไว้เฉพาะเปอร์เซ็นต์ (ราว 51–52 %) แท่งจึงเป็นค่าสัมพัทธ์ ไม่มีบันทึกการเทียบกับโหมด low-RAM อื่น จึงไม่เทียบ",
    },
  },
};
