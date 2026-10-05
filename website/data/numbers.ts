import type { ChartSpec } from "@/components/charts/Chart";
import type { L } from "@/lib/i18n";
import { issue, LINKS, pr } from "./sources";

export type Measured = {
  id: string;
  label: L;
  /** The headline figures, each with its own unit. */
  figures: { value: string; unit: string; caption: L }[];
  /** The conditions, in full, shown with the number. Never shortened away. */
  conditions: { k: L; v: L }[];
  source: { label: string; href: string }[];
};

/** The arms and the session of the two speed-by-length cards (#165), written once. */
const SPEED_BY_LENGTH_ARMS = {
  k: { en: "Arms", th: "ฝั่งที่เทียบ" },
  v: {
    en: "Strata-xeno as served (D2x: the 4070 as an expert tier and a split of the prompt path). Upstream v0.1.38, a pristine build, with the flags its own setup.py writes for this model (profile, expert cache auto, prefill auto, spec 4, int8 KV, 262,144-token context with KV streaming at 32,768) in its two two-GPU forms: its layer split, with 2,560 MiB of VRAM kept free to protect the display card; and the second GPU as its peer expert tier (no P2P between these cards, so its prompt path stays on the primary; STRATA_ARENA_PIN_GIB=8, without which the peer failed to start). Same pack, same draft layer. No layer or expert is read from the SSD in Strata-xeno's runs (every log: nvme loads 0).",
    th: "Strata-xeno ตามที่ serve อยู่ (D2x: 4070 เป็นชั้น expert และแบ่งงานอ่าน prompt) เทียบกับ upstream v0.1.38 ที่ build ใหม่โดยไม่แก้ ใช้ flag ที่ setup.py ของมันเขียนให้โมเดลนี้ (profile, expert cache auto, prefill auto, spec 4, KV int8, context 262,144 โทเคนพร้อม KV streaming ที่ 32,768) ในสองแบบที่ใช้สอง GPU: layer split ที่กัน VRAM ว่างไว้ 2,560 MiB เพื่อรักษาการ์ดจอ และ GPU ที่สองเป็นชั้น expert แบบ peer (การ์ดคู่นี้ไม่มี P2P การอ่าน prompt จึงอยู่บนการ์ดหลัก; STRATA_ARENA_PIN_GIB=8 ถ้าไม่ตั้ง peer เริ่มไม่ขึ้น) pack และ draft layer เดียวกัน ในการรันของ Strata-xeno ไม่มี layer หรือ expert ใดถูกอ่านจาก SSD (ทุก log: nvme loads 0)",
  },
};
const SPEED_BY_LENGTH_SESSION = {
  k: { en: "Session", th: "เซสชัน" },
  v: {
    en: "One session on 2026-10-04, serve mode, a fresh server boot for every run; at each length the arms in the order Strata-xeno, layer split, peer, peer, layer split, Strata-xeno. Swift 1.5 Qwen3.8 Flash-Next IQ2_XS. Intel Core i5-13500, 48 GB DDR5, RTX 5060 Ti 16 GB (PCIe 4.0 x4) + RTX 4070 SUPER 12 GB (x16, the display), Windows 11.",
    th: "เซสชันเดียว 2026-10-04 โหมด serve บูตเซิร์ฟเวอร์ใหม่ทุกรอบ ทุกความยาวสลับลำดับ Strata-xeno, layer split, peer, peer, layer split, Strata-xeno โมเดล Swift 1.5 Qwen3.8 Flash-Next IQ2_XS เครื่อง Intel Core i5-13500, DDR5 48 GB, RTX 5060 Ti 16 GB (PCIe 4.0 x4) + RTX 4070 SUPER 12 GB (x16 การ์ดจอ), Windows 11",
  },
};
const LENGTH_GROUPS: L[] = ["1K", "4K", "32K", "64K", "128K"].map((n) => ({ en: n, th: n }));
const SPEED_BY_LENGTH_SOURCE = [
  { label: "Issue #165", href: issue(165) },
  { label: "bench/results/2026-10-04-speed-xeno-vs-upstream-0138", href: `${LINKS.fork}/tree/main/bench/results/2026-10-04-speed-xeno-vs-upstream-0138` },
];

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
      { k: { en: "Not measured", th: "ยังไม่ได้วัด" }, v: { en: "This Q2_0 run does not cover the IQ2_XS model or later upstream versions: IQ2_XS against v0.1.38 is its own card (vs v0.1.38: RAM). Upstream's own low-RAM modes are not measured.", th: "การรัน Q2_0 นี้ไม่ครอบคลุมโมเดล IQ2_XS หรือ upstream รุ่นหลัง: IQ2_XS เทียบ v0.1.38 อยู่ในการ์ดของมันเอง (เทียบ v0.1.38: RAM) ยังไม่ได้วัดโหมด low-RAM ของ upstream เอง" } },
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
  {
    id: "v038-prompt",
    label: { en: "Reading the prompt, against upstream v0.1.38", th: "การอ่าน prompt เทียบกับ upstream v0.1.38" },
    figures: [
      { value: "11.1", unit: "s", caption: { en: "Strata-xeno, six follow-up turns", th: "Strata-xeno หก turn ต่อเนื่อง" } },
      { value: "13.2", unit: "s", caption: { en: "upstream v0.1.38, its fastest form here (layer split)", th: "upstream v0.1.38 แบบที่เร็วที่สุดที่นี่ (layer split)" } },
    ],
    conditions: [
      { k: { en: "What was measured", th: "สิ่งที่วัด" }, v: { en: "Time to read the new part of each turn, as the server logs it; the six turns together, and the first 9,932-token prompt on its own. Mean of two runs.", th: "เวลาอ่านส่วนใหม่ของแต่ละ turn ตามที่เซิร์ฟเวอร์บันทึก รวมหก turn และ prompt แรก 9,932 โทเคนแยกต่างหาก ค่าเฉลี่ยสองรอบ" } },
      { k: { en: "Arms", th: "ฝั่งที่เทียบ" }, v: { en: "Strata-xeno as served (D2x: the 4070 as an expert tier and a split of the prompt path). Upstream v0.1.38, a pristine build, with the flags its own setup.py writes for this model (profile, expert cache auto, prefill auto, spec 4, int8 KV, 262,144-token context with KV streaming at 32,768) in three forms: one GPU; the second GPU as its peer expert tier (no P2P between these cards, so its prompt path stays on the primary; STRATA_ARENA_PIN_GIB=8, without which the peer failed to start); its layer split, with 2,560 MiB of VRAM kept free to protect the display card (its default is 700, on both cards). Dynamic experts (upstream: the adaptive tier) on in every arm. Same pack, same draft layer.", th: "Strata-xeno ตามที่ serve อยู่ (D2x: 4070 เป็นชั้น expert และแบ่งงานอ่าน prompt) เทียบกับ upstream v0.1.38 ที่ build ใหม่โดยไม่แก้ ใช้ flag ที่ setup.py ของมันเขียนให้โมเดลนี้ (profile, expert cache auto, prefill auto, spec 4, KV int8, context 262,144 โทเคนพร้อม KV streaming ที่ 32,768) สามแบบ: GPU เดียว; GPU ที่สองเป็นชั้น expert แบบ peer (การ์ดคู่นี้ไม่มี P2P การอ่าน prompt จึงอยู่บนการ์ดหลัก; STRATA_ARENA_PIN_GIB=8 ถ้าไม่ตั้ง peer เริ่มไม่ขึ้น); และ layer split ที่กัน VRAM ว่างไว้ 2,560 MiB เพื่อรักษาการ์ดจอ (ค่าเริ่มต้นของมันคือ 700 และใช้กับทั้งสองการ์ด) เปิด dynamic experts (upstream: adaptive tier) ทุกฝั่ง pack และ draft layer เดียวกัน" } },
      { k: { en: "Session", th: "เซสชัน" }, v: { en: "One session on 2026-10-04, the arms in the order base, layer split, peer, one GPU, then back; two runs per arm. Claude Code request shape: a 9,932-token conversation, then six turns that each add 408–2,816 tokens.", th: "เซสชันเดียว 2026-10-04 สลับลำดับ base, layer split, peer, GPU เดียว แล้วย้อนกลับ สองรอบต่อฝั่ง รูปแบบ request ของ Claude Code: บทสนทนา 9,932 โทเคน แล้วหก turn ที่เพิ่ม 408–2,816 โทเคนต่อครั้ง" } },
      { k: { en: "Model", th: "โมเดล" }, v: { en: "Swift 1.5 Qwen3.8 Flash-Next IQ2_XS, serve mode.", th: "Swift 1.5 Qwen3.8 Flash-Next IQ2_XS โหมด serve" } },
      { k: { en: "Machine", th: "เครื่อง" }, v: { en: "Intel Core i5-13500, 48 GB DDR5, RTX 5060 Ti 16 GB (×4) + RTX 4070 SUPER 12 GB (×16, the display), Windows 11.", th: "Intel Core i5-13500, DDR5 48 GB, RTX 5060 Ti 16 GB (×4) + RTX 4070 SUPER 12 GB (×16 การ์ดจอ), Windows 11" } },
      { k: { en: "Same run, other side", th: "ในการรันเดียวกัน อีกด้าน" }, v: { en: "On the long first prompt upstream's layer split tied (7.96 vs 8.01 s). Its peer and one-GPU forms took about twice as long on everything.", th: "กับ prompt แรกที่ยาว layer split ของ upstream เสมอกัน (7.96 เทียบ 8.01 วินาที) ส่วนแบบ peer และแบบ GPU เดียวใช้เวลาราวสองเท่าในทุกส่วน" } },
    ],
    source: [
      { label: "Issue #163", href: issue(163) },
      { label: "Issue #136", href: issue(136) },
    ],
  },
  {
    id: "v038-decode",
    label: { en: "Decode speed, against upstream v0.1.38", th: "ความเร็ว decode เทียบกับ upstream v0.1.38" },
    figures: [
      { value: "71.7", unit: "tok/s", caption: { en: "Strata-xeno, six 200-token replies", th: "Strata-xeno คำตอบ 200 โทเคนหกครั้ง" } },
      { value: "72.3", unit: "tok/s", caption: { en: "upstream v0.1.38 with its peer tier (a tie)", th: "upstream v0.1.38 กับชั้น peer (เสมอกัน)" } },
    ],
    conditions: [
      { k: { en: "What was measured", th: "สิ่งที่วัด" }, v: { en: "Tokens per second while writing: the six 200-token replies of the Claude Code turns together, and a separate 400-token answer to a short prompt. Mean of two runs.", th: "โทเคนต่อวินาทีขณะเขียนคำตอบ: คำตอบ 200 โทเคนของหก turn รวมกัน และคำตอบ 400 โทเคนต่อ prompt สั้นแยกต่างหาก ค่าเฉลี่ยสองรอบ" } },
      { k: { en: "Arms", th: "ฝั่งที่เทียบ" }, v: { en: "Strata-xeno as served (D2x: the 4070 as an expert tier and a split of the prompt path). Upstream v0.1.38, a pristine build, with the flags its own setup.py writes for this model (profile, expert cache auto, prefill auto, spec 4, int8 KV, 262,144-token context with KV streaming at 32,768) in three forms: one GPU; the second GPU as its peer expert tier (no P2P between these cards, so its prompt path stays on the primary; STRATA_ARENA_PIN_GIB=8, without which the peer failed to start); its layer split, with 2,560 MiB of VRAM kept free to protect the display card (its default is 700, on both cards). Dynamic experts (upstream: the adaptive tier) on in every arm. Same pack, same draft layer.", th: "Strata-xeno ตามที่ serve อยู่ (D2x: 4070 เป็นชั้น expert และแบ่งงานอ่าน prompt) เทียบกับ upstream v0.1.38 ที่ build ใหม่โดยไม่แก้ ใช้ flag ที่ setup.py ของมันเขียนให้โมเดลนี้ (profile, expert cache auto, prefill auto, spec 4, KV int8, context 262,144 โทเคนพร้อม KV streaming ที่ 32,768) สามแบบ: GPU เดียว; GPU ที่สองเป็นชั้น expert แบบ peer (การ์ดคู่นี้ไม่มี P2P การอ่าน prompt จึงอยู่บนการ์ดหลัก; STRATA_ARENA_PIN_GIB=8 ถ้าไม่ตั้ง peer เริ่มไม่ขึ้น); และ layer split ที่กัน VRAM ว่างไว้ 2,560 MiB เพื่อรักษาการ์ดจอ (ค่าเริ่มต้นของมันคือ 700 และใช้กับทั้งสองการ์ด) เปิด dynamic experts (upstream: adaptive tier) ทุกฝั่ง pack และ draft layer เดียวกัน" } },
      { k: { en: "Session", th: "เซสชัน" }, v: { en: "One session on 2026-10-04, the arms in the order base, layer split, peer, one GPU, then back; two runs per arm. Claude Code request shape: a 9,932-token conversation, then six turns that each add 408–2,816 tokens.", th: "เซสชันเดียว 2026-10-04 สลับลำดับ base, layer split, peer, GPU เดียว แล้วย้อนกลับ สองรอบต่อฝั่ง รูปแบบ request ของ Claude Code: บทสนทนา 9,932 โทเคน แล้วหก turn ที่เพิ่ม 408–2,816 โทเคนต่อครั้ง" } },
      { k: { en: "Machine", th: "เครื่อง" }, v: { en: "Intel Core i5-13500, 48 GB DDR5, RTX 5060 Ti 16 GB (×4) + RTX 4070 SUPER 12 GB (×16, the display), Windows 11.", th: "Intel Core i5-13500, DDR5 48 GB, RTX 5060 Ti 16 GB (×4) + RTX 4070 SUPER 12 GB (×16 การ์ดจอ), Windows 11" } },
      { k: { en: "Read it as", th: "อ่านอย่างไร" }, v: { en: "A tie with upstream's peer tier on the turns (73.8 / 70.7 vs 71.7 / 71.7); ahead of its layer split and one-GPU forms by 10–20 %. Greedy, speculative decoding with the MTP draft layer in every arm.", th: "เสมอกับชั้น peer ของ upstream ใน turn (73.8 / 70.7 เทียบ 71.7 / 71.7) และเร็วกว่าแบบ layer split และ GPU เดียว 10–20 % ทุกฝั่งเป็น greedy พร้อม speculative decoding ด้วย MTP draft layer" } },
    ],
    source: [
      { label: "Issue #163", href: issue(163) },
      { label: "Issue #136", href: issue(136) },
    ],
  },
  {
    id: "v038-ram",
    label: { en: "RAM, against upstream v0.1.38", th: "RAM เทียบกับ upstream v0.1.38" },
    figures: [
      { value: "23.7", unit: "GiB", caption: { en: "Strata-xeno; 15.3 GB of RAM still free", th: "Strata-xeno; RAM ยังว่าง 15.3 GB" } },
      { value: "36.9–39.8", unit: "GiB", caption: { en: "upstream v0.1.38; free RAM down to 0.01–0.6 GB", th: "upstream v0.1.38; RAM ว่างเหลือ 0.01–0.6 GB" } },
    ],
    conditions: [
      { k: { en: "What was measured", th: "สิ่งที่วัด" }, v: { en: "The server process's working set (resident RAM) at its sampled peak, with the system's free RAM, commit and hard page-ins, every 10 seconds through load, prompt and decode. One run per arm, 4–7 samples each, so the peaks are approximate.", th: "working set ของโปรเซสเซิร์ฟเวอร์ (RAM ที่ใช้อยู่จริง) ที่จุดสูงสุดที่เก็บได้ พร้อม RAM ว่าง commit และ hard page-in ของระบบ ทุก 10 วินาทีตลอดการโหลด การอ่าน prompt และ decode หนึ่งรอบต่อฝั่ง ฝั่งละ 4–7 จุด ค่าสูงสุดจึงเป็นค่าประมาณ" } },
      { k: { en: "Arms", th: "ฝั่งที่เทียบ" }, v: { en: "Strata-xeno as served (D2x: the 4070 as an expert tier and a split of the prompt path). Upstream v0.1.38, a pristine build, with the flags its own setup.py writes for this model (profile, expert cache auto, prefill auto, spec 4, int8 KV, 262,144-token context with KV streaming at 32,768) in three forms: one GPU; the second GPU as its peer expert tier (no P2P between these cards, so its prompt path stays on the primary; STRATA_ARENA_PIN_GIB=8, without which the peer failed to start); its layer split, with 2,560 MiB of VRAM kept free to protect the display card (its default is 700, on both cards). Dynamic experts (upstream: the adaptive tier) on in every arm. Same pack, same draft layer.", th: "Strata-xeno ตามที่ serve อยู่ (D2x: 4070 เป็นชั้น expert และแบ่งงานอ่าน prompt) เทียบกับ upstream v0.1.38 ที่ build ใหม่โดยไม่แก้ ใช้ flag ที่ setup.py ของมันเขียนให้โมเดลนี้ (profile, expert cache auto, prefill auto, spec 4, KV int8, context 262,144 โทเคนพร้อม KV streaming ที่ 32,768) สามแบบ: GPU เดียว; GPU ที่สองเป็นชั้น expert แบบ peer (การ์ดคู่นี้ไม่มี P2P การอ่าน prompt จึงอยู่บนการ์ดหลัก; STRATA_ARENA_PIN_GIB=8 ถ้าไม่ตั้ง peer เริ่มไม่ขึ้น); และ layer split ที่กัน VRAM ว่างไว้ 2,560 MiB เพื่อรักษาการ์ดจอ (ค่าเริ่มต้นของมันคือ 700 และใช้กับทั้งสองการ์ด) เปิด dynamic experts (upstream: adaptive tier) ทุกฝั่ง pack และ draft layer เดียวกัน" } },
      { k: { en: "Machine", th: "เครื่อง" }, v: { en: "Intel Core i5-13500, 48 GB DDR5, RTX 5060 Ti 16 GB (×4) + RTX 4070 SUPER 12 GB (×16, the display), Windows 11.", th: "Intel Core i5-13500, DDR5 48 GB, RTX 5060 Ti 16 GB (×4) + RTX 4070 SUPER 12 GB (×16 การ์ดจอ), Windows 11" } },
      { k: { en: "What it means", th: "หมายความว่าอะไร" }, v: { en: "Upstream keeps every expert in RAM and lists this model at 48 GB, exactly this PC: free RAM ran out and the system paged from disk while serving (median 2,245–3,179 page-ins a second against Strata-xeno's 116). Dynamic experts hold part of them on the GPUs instead.", th: "upstream เก็บ expert ทุกตัวไว้ใน RAM และระบุโมเดลนี้ไว้ที่ 48 GB พอดีกับพีซีเครื่องนี้ RAM ว่างจึงหมดและระบบต้องอ่านหน้าจาก disk ระหว่าง serve (median 2,245–3,179 ครั้งต่อวินาที เทียบกับ 116 ของ Strata-xeno) ส่วน dynamic experts ให้ GPU ถือ expert ส่วนหนึ่งแทน" } },
    ],
    source: [
      { label: "Issue #163", href: issue(163) },
      { label: "Issue #136", href: issue(136) },
    ],
  },
  {
    id: "v038-len-prompt",
    label: { en: "Reading the prompt by length, against upstream v0.1.38", th: "การอ่าน prompt ตามความยาว เทียบกับ upstream v0.1.38" },
    figures: [
      { value: "1,021", unit: "tok/s", caption: { en: "Strata-xeno, 4K prompt", th: "Strata-xeno prompt 4K" } },
      { value: "768", unit: "tok/s", caption: { en: "upstream v0.1.38, its fastest form at 4K (layer split)", th: "upstream v0.1.38 แบบที่เร็วที่สุดที่ 4K (layer split)" } },
      { value: "1,508", unit: "tok/s", caption: { en: "Strata-xeno, 128K prompt", th: "Strata-xeno prompt 128K" } },
      { value: "1,690", unit: "tok/s", caption: { en: "upstream v0.1.38, layer split, 128K: ahead here", th: "upstream v0.1.38 layer split ที่ 128K: เร็วกว่าที่นี่" } },
    ],
    conditions: [
      { k: { en: "What was measured", th: "สิ่งที่วัด" }, v: { en: "Prompt tokens read per second, as the server logs it, for one code-agent prompt per length (1,064 / 4,136 / 32,808 / 65,575 / 131,112 tokens of C++ and CUDA source with a task). Mean of two runs.", th: "จำนวนโทเคน prompt ที่อ่านได้ต่อวินาทีตามที่เซิร์ฟเวอร์บันทึก prompt แบบ code-agent หนึ่งชุดต่อความยาว (1,064 / 4,136 / 32,808 / 65,575 / 131,112 โทเคน เป็นซอร์ส C++ และ CUDA พร้อมงานที่สั่ง) ค่าเฉลี่ยสองรอบ" } },
      SPEED_BY_LENGTH_ARMS,
      SPEED_BY_LENGTH_SESSION,
      { k: { en: "Read it as", th: "อ่านอย่างไร" }, v: { en: "Strata-xeno reads 1K–4K prompts 33–47 % faster than upstream's fastest form. From 32K up, upstream's layer split reads 6.6–12.9 % faster, in both runs at every length. The two runs of a cell differ by a median of 2.7 %.", th: "Strata-xeno อ่าน prompt 1K–4K เร็วกว่าแบบที่เร็วที่สุดของ upstream 33–47 % ตั้งแต่ 32K ขึ้นไป layer split ของ upstream อ่านเร็วกว่า 6.6–12.9 % ทั้งสองรอบในทุกความยาว สองรอบของช่องเดียวกันต่างกันโดย median 2.7 %" } },
      { k: { en: "Upstream's own figures", th: "ตัวเลขของ upstream เอง" }, v: { en: "Upstream's table for this model on an RTX 5070 (PCIe 5.0 x16), Ryzen 5 7600, 64 GB is 534 / 1,256 / 2,092 / 1,754 / 1,752 tok/s. Another machine and one-shot runs, so it is context, not a pairing.", th: "ตารางของ upstream สำหรับโมเดลนี้บน RTX 5070 (PCIe 5.0 x16), Ryzen 5 7600, 64 GB คือ 534 / 1,256 / 2,092 / 1,754 / 1,752 tok/s เป็นเครื่องอื่นและรันแบบ one-shot จึงใช้เป็นบริบท ไม่ใช่การเทียบคู่" } },
    ],
    source: SPEED_BY_LENGTH_SOURCE,
  },
  {
    id: "v038-len-decode",
    label: { en: "Decode speed by prompt length, against upstream v0.1.38", th: "ความเร็ว decode ตามความยาว prompt เทียบกับ upstream v0.1.38" },
    figures: [
      { value: "72.7", unit: "tok/s", caption: { en: "Strata-xeno, after a 4K prompt", th: "Strata-xeno หลัง prompt 4K" } },
      { value: "50.0", unit: "tok/s", caption: { en: "upstream v0.1.38, its fastest form at 4K (peer tier)", th: "upstream v0.1.38 แบบที่เร็วที่สุดที่ 4K (ชั้น peer)" } },
      { value: "62.8", unit: "tok/s", caption: { en: "Strata-xeno, after a 128K prompt", th: "Strata-xeno หลัง prompt 128K" } },
      { value: "46.9", unit: "tok/s", caption: { en: "upstream v0.1.38, its fastest form at 128K (peer tier)", th: "upstream v0.1.38 แบบที่เร็วที่สุดที่ 128K (ชั้น peer)" } },
    ],
    conditions: [
      { k: { en: "What was measured", th: "สิ่งที่วัด" }, v: { en: "Output tokens per second while writing a 256-token answer, greedy, with the MTP draft layer, after one code-agent prompt per length (1K to 128K). Mean of two runs.", th: "จำนวนโทเคนที่เขียนได้ต่อวินาทีขณะเขียนคำตอบ 256 โทเคน แบบ greedy พร้อม MTP draft layer หลัง prompt แบบ code-agent หนึ่งชุดต่อความยาว (1K ถึง 128K) ค่าเฉลี่ยสองรอบ" } },
      SPEED_BY_LENGTH_ARMS,
      SPEED_BY_LENGTH_SESSION,
      { k: { en: "Read it as", th: "อ่านอย่างไร" }, v: { en: "Ahead of upstream's fastest form (its peer tier) at every length, by 30–45 %, and of its layer split by 61–82 %. Speed moves with the share of drafts accepted (Strata-xeno 0.65–0.80, upstream 0.62–0.95); the two runs of a cell differ by a median of 5 %.", th: "เร็วกว่าแบบที่เร็วที่สุดของ upstream (ชั้น peer) ทุกความยาว 30–45 % และเร็วกว่า layer split ของมัน 61–82 % ความเร็วขึ้นกับสัดส่วน draft ที่ยอมรับ (Strata-xeno 0.65–0.80, upstream 0.62–0.95) สองรอบของช่องเดียวกันต่างกันโดย median 5 %" } },
      { k: { en: "Upstream's own figures", th: "ตัวเลขของ upstream เอง" }, v: { en: "Upstream's table for this model on an RTX 5070 (PCIe 5.0 x16), Ryzen 5 7600, 64 GB is 79.6 / 78.6 / 76.3 / 63.7 / 62.7 tok/s. Another machine and one-shot runs, so it is context, not a pairing.", th: "ตารางของ upstream สำหรับโมเดลนี้บน RTX 5070 (PCIe 5.0 x16), Ryzen 5 7600, 64 GB คือ 79.6 / 78.6 / 76.3 / 63.7 / 62.7 tok/s เป็นเครื่องอื่นและรันแบบ one-shot จึงใช้เป็นบริบท ไม่ใช่การเทียบคู่" } },
    ],
    source: SPEED_BY_LENGTH_SOURCE,
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
export type ChartId =
  | "ram"
  | "decode"
  | "capacity"
  | "v038-prompt"
  | "v038-decode"
  | "v038-ram"
  | "v038-len-prompt"
  | "v038-len-decode";

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
  "v038-prompt": {
    id: "v038-prompt",
    title: { en: "Reading the prompt (seconds, mean of two runs)", th: "การอ่าน prompt (วินาที ค่าเฉลี่ยสองรอบ)" },
    unit: "s",
    better: "lower",
    groups: [
      { en: "First prompt, 9,932 tokens", th: "prompt แรก 9,932 โทเคน" },
      { en: "Six follow-up turns, together", th: "หก turn ต่อเนื่องรวมกัน" },
    ],
    series: [
      { name: { en: "Strata-xeno", th: "Strata-xeno" }, tone: "mine", values: [8.01, 11.06] },
      { name: { en: "upstream v0.1.38, layer split", th: "upstream v0.1.38 layer split" }, tone: "theirs", values: [7.96, 13.18] },
      { name: { en: "upstream v0.1.38, peer tier", th: "upstream v0.1.38 ชั้น peer" }, tone: "theirs", values: [13.87, 23.82] },
      { name: { en: "upstream v0.1.38, one GPU", th: "upstream v0.1.38 GPU เดียว" }, tone: "theirs", values: [13.1, 23.67] },
    ],
    max: 25,
    step: 5,
    digits: 1,
    deltas: [
      { en: "a tie with the layer split", th: "เสมอกับ layer split" },
      { en: "2.1 s less than the fastest upstream", th: "น้อยกว่า upstream ที่เร็วที่สุด 2.1 วินาที" },
    ],
    caption: {
      en: "Swift 1.5 IQ2_XS, serve mode, Claude Code request shape; same session (2026-10-04), two runs per arm, one PC; dynamic experts on in every arm.",
      th: "Swift 1.5 IQ2_XS โหมด serve รูปแบบ request ของ Claude Code เซสชันเดียวกัน (2026-10-04) สองรอบต่อฝั่ง พีซีเครื่องเดียว เปิด dynamic experts ทุกฝั่ง",
    },
  },
  "v038-decode": {
    id: "v038-decode",
    title: { en: "Decode speed (tok/s, mean of two runs)", th: "ความเร็ว decode (tok/s ค่าเฉลี่ยสองรอบ)" },
    unit: "tok/s",
    better: "higher",
    groups: [
      { en: "Six 200-token replies", th: "คำตอบ 200 โทเคนหกครั้ง" },
      { en: "A 400-token answer", th: "คำตอบ 400 โทเคน" },
    ],
    series: [
      { name: { en: "Strata-xeno", th: "Strata-xeno" }, tone: "mine", values: [71.7, 75.2] },
      { name: { en: "upstream v0.1.38, layer split", th: "upstream v0.1.38 layer split" }, tone: "theirs", values: [63.1, 60.6] },
      { name: { en: "upstream v0.1.38, peer tier", th: "upstream v0.1.38 ชั้น peer" }, tone: "theirs", values: [72.3, 67.0] },
      { name: { en: "upstream v0.1.38, one GPU", th: "upstream v0.1.38 GPU เดียว" }, tone: "theirs", values: [57.8, 52.4] },
    ],
    max: 80,
    step: 20,
    deltas: [
      { en: "a tie with the peer tier", th: "เสมอกับชั้น peer" },
      { en: "+8.2 tok/s over the fastest upstream", th: "+8.2 tok/s เหนือ upstream ที่เร็วที่สุด" },
    ],
    caption: {
      en: "Swift 1.5 IQ2_XS, serve mode, greedy with the MTP draft layer; same session (2026-10-04), two runs per arm, one PC; dynamic experts on in every arm.",
      th: "Swift 1.5 IQ2_XS โหมด serve greedy พร้อม MTP draft layer เซสชันเดียวกัน (2026-10-04) สองรอบต่อฝั่ง พีซีเครื่องเดียว เปิด dynamic experts ทุกฝั่ง",
    },
  },
  "v038-ram": {
    id: "v038-ram",
    title: { en: "RAM in use while serving (GiB, sampled peak)", th: "RAM ที่ใช้ขณะ serve (GiB จุดสูงสุดที่เก็บได้)" },
    unit: "GiB",
    better: "lower",
    groups: [{ en: "Working set of the server process", th: "working set ของโปรเซสเซิร์ฟเวอร์" }],
    series: [
      { name: { en: "Strata-xeno", th: "Strata-xeno" }, tone: "mine", values: [23.7] },
      { name: { en: "upstream v0.1.38, layer split", th: "upstream v0.1.38 layer split" }, tone: "theirs", values: [37.5] },
      { name: { en: "upstream v0.1.38, peer tier", th: "upstream v0.1.38 ชั้น peer" }, tone: "theirs", values: [39.8] },
      { name: { en: "upstream v0.1.38, one GPU", th: "upstream v0.1.38 GPU เดียว" }, tone: "theirs", values: [36.9] },
    ],
    max: 40,
    step: 10,
    deltas: [{ en: "13.2 GiB less than the least upstream", th: "น้อยกว่า upstream ที่ใช้น้อยที่สุด 13.2 GiB" }],
    caption: {
      en: "One run per arm, sampled every 10 s (4–7 samples), so the peaks are approximate. 48 GB PC: upstream ran it out of free RAM and paged from disk; Strata-xeno left 15.3 GB free.",
      th: "หนึ่งรอบต่อฝั่ง เก็บทุก 10 วินาที (4–7 จุด) ค่าสูงสุดจึงเป็นค่าประมาณ พีซี 48 GB: upstream ทำให้ RAM ว่างหมดและต้องอ่านหน้าจาก disk ส่วน Strata-xeno เหลือ RAM ว่าง 15.3 GB",
    },
  },
  "v038-len-prompt": {
    id: "v038-len-prompt",
    title: { en: "Reading the prompt, by prompt length (tok/s, mean of two runs)", th: "การอ่าน prompt ตามความยาว prompt (tok/s ค่าเฉลี่ยสองรอบ)" },
    unit: "tok/s",
    better: "higher",
    groups: LENGTH_GROUPS,
    series: [
      { name: { en: "Strata-xeno", th: "Strata-xeno" }, tone: "mine", values: [416.1, 1021.2, 1469.4, 1536.0, 1508.3] },
      { name: { en: "upstream v0.1.38, layer split", th: "upstream v0.1.38 layer split" }, tone: "theirs", values: [283.8, 767.7, 1565.8, 1733.6, 1690.1] },
      { name: { en: "upstream v0.1.38, peer tier", th: "upstream v0.1.38 ชั้น peer" }, tone: "theirs", values: [212.3, 605.0, 1055.0, 1073.4, 1048.3] },
    ],
    max: 2000,
    step: 500,
    digits: 0,
    deltas: [
      { en: "+132.3 tok/s over the fastest upstream", th: "+132.3 tok/s เหนือ upstream ที่เร็วที่สุด" },
      { en: "+253.5 tok/s over the fastest upstream", th: "+253.5 tok/s เหนือ upstream ที่เร็วที่สุด" },
      { en: "upstream's layer split +96.4 tok/s", th: "layer split ของ upstream +96.4 tok/s" },
      { en: "upstream's layer split +197.6 tok/s", th: "layer split ของ upstream +197.6 tok/s" },
      { en: "upstream's layer split +181.8 tok/s", th: "layer split ของ upstream +181.8 tok/s" },
    ],
    caption: {
      en: "Swift 1.5 IQ2_XS, serve mode, one code-agent prompt per length; same session (2026-10-04), two runs per arm, one PC. From 32K up, upstream's layer split reads faster.",
      th: "Swift 1.5 IQ2_XS โหมด serve prompt แบบ code-agent หนึ่งชุดต่อความยาว เซสชันเดียวกัน (2026-10-04) สองรอบต่อฝั่ง พีซีเครื่องเดียว ตั้งแต่ 32K ขึ้นไป layer split ของ upstream อ่านเร็วกว่า",
    },
  },
  "v038-len-decode": {
    id: "v038-len-decode",
    title: { en: "Decode speed, by prompt length (tok/s, mean of two runs)", th: "ความเร็ว decode ตามความยาว prompt (tok/s ค่าเฉลี่ยสองรอบ)" },
    unit: "tok/s",
    better: "higher",
    groups: LENGTH_GROUPS,
    series: [
      { name: { en: "Strata-xeno", th: "Strata-xeno" }, tone: "mine", values: [69.2, 72.7, 63.2, 58.0, 62.8] },
      { name: { en: "upstream v0.1.38, layer split", th: "upstream v0.1.38 layer split" }, tone: "theirs", values: [40.4, 40.0, 39.3, 34.8, 37.8] },
      { name: { en: "upstream v0.1.38, peer tier", th: "upstream v0.1.38 ชั้น peer" }, tone: "theirs", values: [53.2, 50.0, 46.1, 44.2, 46.9] },
    ],
    max: 80,
    step: 20,
    deltas: [
      { en: "+16.0 tok/s over the fastest upstream", th: "+16.0 tok/s เหนือ upstream ที่เร็วที่สุด" },
      { en: "+22.7 tok/s over the fastest upstream", th: "+22.7 tok/s เหนือ upstream ที่เร็วที่สุด" },
      { en: "+17.1 tok/s over the fastest upstream", th: "+17.1 tok/s เหนือ upstream ที่เร็วที่สุด" },
      { en: "+13.8 tok/s over the fastest upstream", th: "+13.8 tok/s เหนือ upstream ที่เร็วที่สุด" },
      { en: "+15.9 tok/s over the fastest upstream", th: "+15.9 tok/s เหนือ upstream ที่เร็วที่สุด" },
    ],
    caption: {
      en: "Swift 1.5 IQ2_XS, serve mode, 256 tokens after one code-agent prompt per length, greedy with the MTP draft layer; same session (2026-10-04), two runs per arm, one PC.",
      th: "Swift 1.5 IQ2_XS โหมด serve 256 โทเคนหลัง prompt แบบ code-agent หนึ่งชุดต่อความยาว greedy พร้อม MTP draft layer เซสชันเดียวกัน (2026-10-04) สองรอบต่อฝั่ง พีซีเครื่องเดียว",
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
