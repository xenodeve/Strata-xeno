"use client";

import { LINKS, REV } from "@/data/sources";
import { T } from "@/lib/i18n";

const THIRD: { name: string; lic: string; use: { en: string; th: string } }[] = [
  { name: "Inter · Anuphan · JetBrains Mono", lic: "SIL OFL 1.1", use: { en: "Typefaces of the app, and of this site", th: "ฟอนต์ของแอปและของเว็บนี้" } },
  { name: "Instrument Serif", lic: "SIL OFL 1.1", use: { en: "The big words (the two canvas effects near the end of the home page)", th: "คำตัวใหญ่ (เอฟเฟกต์ canvas สองอย่างท้ายหน้าแรก)" } },
  { name: "Hugeicons (free)", lic: "MIT", use: { en: "Icons in the app", th: "ไอคอนในแอป" } },
  { name: "thinking-orbs · bot-avatars (Libraries.dev, Jakub Antalik)", lic: "MIT", use: { en: "The orbs and the 18 avatars, vendored in the app. The orbs on this site's screens are the same library (thinking-orbs 0.3.2, from npm); the avatars on the first screen are the vendored bot-avatars", th: "orb และ avatar 18 แบบ ที่ฝังไว้ในแอป orb บนหน้าจอของเว็บนี้ใช้ไลบรารีเดียวกัน (thinking-orbs 0.3.2 จาก npm) ส่วน avatar ในหน้าแรกคือ bot-avatars ที่ฝังไว้" } },
  { name: "@yogesharc/thinking-orbs", lic: "MIT", use: { en: "The compacting orb", th: "orb ตอนย่อบทสนทนา" } },
  { name: "highlight.js", lic: "BSD-3-Clause", use: { en: "Code colouring in answers, and on the screens of this site", th: "สีของโค้ดในคำตอบ และบนหน้าจอของเว็บนี้" } },
  { name: "@lobehub/icons-static-svg (LobeHub)", lic: "MIT", use: { en: "The marks of the other coding apps on the Import and Memory screens (Claude Code, Claude Desktop, Codex, Antigravity, Gemini CLI, Cursor). Each mark belongs to its maker and is shown only to say which app is meant; the shared ~/.agents folder's glyph is drawn here", th: "ไอคอนของแอปเขียนโค้ดอื่นบนหน้าจอ Import และ Memory (Claude Code, Claude Desktop, Codex, Antigravity, Gemini CLI, Cursor) เครื่องหมายแต่ละอันเป็นของเจ้าของมัน แสดงเพียงเพื่อบอกว่าหมายถึงแอปไหน ส่วนไอคอนโฟลเดอร์ ~/.agents วาดเอง" } },
  { name: "Monokai Classic (palette by Wimer Hazenberg)", lic: "colour scheme", use: { en: "The colours of every piece of code on this site; the app itself uses the same palette for code in its dark theme", th: "สีของโค้ดทุกจุดบนเว็บนี้ แอปเองก็ใช้ชุดสีเดียวกันกับโค้ดในธีมมืด" } },
  { name: "Three.js · Next.js · React", lic: "MIT", use: { en: "This site", th: "เว็บนี้" } },
  { name: "Octicons (the GitHub mark)", lic: "MIT", use: { en: "The GitHub button", th: "ปุ่ม GitHub" } },
  { name: "React Bits · TechText, WarpText, PixelSwap, Particles, Aurora, ShapeWaves, LatticeLoader and BranchedMenu", lic: "MIT with the Commons Clause", use: { en: "The two canvas words, the pixel swap, the drifting points, the aurora, the band of shapes and the lattice of dots beside a thought and the branched menu of the topics (both are what the app itself uses), supplied by the project owner and used as they came", th: "คำ canvas สองคำ pixel swap จุดแสงที่ลอยอยู่ แสงเหนือ แถบรูปทรง ตารางจุดข้างความคิด และเมนูแตกกิ่งของหัวข้อ (ทั้งสองอย่างแอปเองก็ใช้) ที่เจ้าของโปรเจกต์ส่งมาและใช้ตามที่ได้รับ" } },
  { name: "vgpu", lic: "MIT", use: { en: "The WebGPU library under ShapeWaves", th: "ไลบรารี WebGPU ที่ ShapeWaves ใช้" } },
  { name: "ogl", lic: "Unlicense", use: { en: "The WebGL library under WarpText, Particles and Aurora", th: "ไลบรารี WebGL ที่ WarpText, Particles และ Aurora ใช้" } },
];

export function CreditsPanel() {
  return (
    <article className="topic">
      <header className="topic__head">
        <h2 className="h3">
          <T v={{ en: "Credits & method", th: "เครดิตและวิธีวัด" }} />
        </h2>
      </header>
      <div>
        <div className="credits">
          <div>
            <h3 className="h4"><T v={{ en: "Built on Strata", th: "สร้างบน Strata" }} /></h3>
            <p>
              <T
                v={{
                  en: "Strata is the work of Niko1221 and the Strata contributors, released under the MIT licence. Strata-xeno is a fork of it. The engine it extends, the API it extends and the Legacy web app (serve/web, shown here unchanged and still served at /classic/) are upstream's work.",
                  th: "Strata เป็นผลงานของ Niko1221 และผู้ร่วมพัฒนา Strata เผยแพร่ภายใต้สัญญาอนุญาต MIT และ Strata-xeno เป็น fork ของมัน เอนจินที่ขยายต่อ API ที่ขยายต่อ และเว็บแอป Legacy (serve/web แสดงในหน้านี้แบบไม่แก้ไข และยังเปิดที่ /classic/) เป็นผลงานของ upstream",
                }}
              />
            </p>
            <p className="links mono">
              <a href={LINKS.upstream} rel="noopener">Niko1221/Strata ↗</a>
              <a href={LINKS.upstreamLicense} rel="noopener">MIT licence ↗</a>
              <a href={LINKS.fork} rel="noopener">xenodeve/Strata-xeno ↗</a>
              <a href={LINKS.forkLicense} rel="noopener">MIT licence ↗</a>
            </p>
          </div>

          <div>
            <h3 className="h4"><T v={{ en: "Third-party parts of the web app", th: "ส่วนของบุคคลที่สามในเว็บแอป" }} /></h3>
            <ul className="third">
              {THIRD.map((x) => (
                <li key={x.name}>
                  <b>{x.name}</b> <span className="mono">{x.lic}</span>
                  <small><T v={x.use} /></small>
                </li>
              ))}
            </ul>
            <p className="note">
              <T
                v={{
                  en: "The app also uses components that are licensed for use inside the app only (React Bits, MIT with the Commons Clause) or whose licence text is not recorded yet (transitions.dev). This site bundles none of the app's own code: the live demo runs the real app from a Strata-xeno checkout on the same PC, and the screens are rebuilt by hand; their motion (durations, curves, distances, how a figure turns or a thought is written) follows the numbers in the app's own stylesheets and components, written again in this site's own code. It does include eight React Bits components that the project owner supplied (TechText, WarpText, PixelSwap, Particles, Aurora, ShapeWaves, LatticeLoader, BranchedMenu), used as supplied and listed above with their licence. The app's own list is in REFERENCES.md.",
                  th: "แอปยังใช้ component ที่อนุญาตให้ใช้ภายในแอปเท่านั้น (React Bits, MIT พร้อม Commons Clause) หรือยังไม่ได้บันทึกข้อความสัญญาอนุญาต (transitions.dev) เว็บนี้ไม่ฝังโค้ดของแอปเอง: demo สดรันแอปจริงจาก checkout ของ Strata-xeno บนเครื่องเดียวกัน และหน้าจอสร้างใหม่ด้วยมือ โดยการเคลื่อนไหว (ระยะเวลา เส้นโค้ง ระยะทาง วิธีที่ตัวเลขหมุนหรือความคิดถูกเขียน) ทำตามตัวเลขในสไตล์ชีตและ component ของแอปจริง เขียนใหม่ด้วยโค้ดของเว็บนี้เอง แต่มี component ของ React Bits แปดตัวที่เจ้าของโปรเจกต์ส่งมา (TechText, WarpText, PixelSwap, Particles, Aurora, ShapeWaves, LatticeLoader, BranchedMenu) ใช้ตามที่ได้รับและระบุไว้ข้างบนพร้อมสัญญาอนุญาต รายการของแอปอยู่ใน REFERENCES.md",
                }}
              />{" "}
              <a href={LINKS.uiReferences} rel="noopener" className="mono">REFERENCES.md ↗</a>
            </p>
          </div>

          <div>
            <h3 className="h4"><T v={{ en: "Method and limits", th: "วิธีวัดและข้อจำกัด" }} /></h3>
            <ul className="plain">
              <li>
                <T
                  v={{
                    en: `Written against ${REV.fork.repo} @ ${REV.fork.sha} (${REV.fork.date}, ${REV.fork.note}); upstream ${REV.upstream.repo} @ ${REV.upstream.sha} (${REV.upstream.note}); merge base ${REV.base.sha} (${REV.base.note}).`,
                    th: `เขียนอ้างอิงจาก ${REV.fork.repo} @ ${REV.fork.sha} (${REV.fork.date}, ${REV.fork.note}); upstream ${REV.upstream.repo} @ ${REV.upstream.sha} (${REV.upstream.note}); merge base ${REV.base.sha} (${REV.base.note})`,
                  }}
                />
              </li>
              <li>
                <T
                  v={{
                    en: "Every measured figure comes from one developer's PC: Windows 11, an Intel Core i5-13500, 48 GB, RTX 5060 Ti + RTX 4070 SUPER. The engine work is validated on Windows and NVIDIA.",
                    th: "ตัวเลขที่วัดทุกตัวมาจากพีซีของนักพัฒนาหนึ่งเครื่อง: Windows 11, Intel Core i5-13500, 48 GB, RTX 5060 Ti + RTX 4070 SUPER งานของเอนจินตรวจสอบบน Windows และ NVIDIA",
                  }}
                />
              </li>
              <li>
                <T
                  v={{
                    en: "Comparisons are against the upstream version named next to each figure. Upstream v0.1.38, released the same day, was not compared.",
                    th: "การเทียบเป็นกับเวอร์ชัน upstream ที่ระบุข้างตัวเลขแต่ละตัว upstream v0.1.38 ที่ออกวันเดียวกันยังไม่ได้เทียบ",
                  }}
                />
              </li>
              <li>
                <T
                  v={{
                    en: "The screens on this site are rebuilt for it, by hand, from one real run of the real web app (a real model, Swift 1.5 IQ2_XS, and a throwaway demo project made for this page). The words are the app's own; the figures are what that run showed, once, on one PC, and are not a benchmark. The original captures are kept in the repository (docs/reference-shots). Nothing of anyone's own chats, imported skills or paths appears.",
                    th: "หน้าจอบนเว็บนี้สร้างใหม่ด้วยมือจากการรันเว็บแอปจริงหนึ่งครั้ง (โมเดลจริง Swift 1.5 IQ2_XS และโปรเจกต์ตัวอย่างที่สร้างขึ้นเพื่อหน้านี้) ข้อความเป็นของแอปเอง ส่วนตัวเลขคือที่การรันนั้นแสดง ครั้งเดียว บนเครื่องเดียว และไม่ใช่ benchmark ภาพต้นฉบับเก็บไว้ใน repository (docs/reference-shots) ไม่มีแชต skill ที่นำเข้า หรือพาธส่วนตัวของใครปรากฏ",
                  }}
                />
              </li>
              <li>
                <T
                  v={{
                    en: "Web access and image / PDF reading in the app had not been tried with a real model when the new app was merged (PR #104).",
                    th: "การเข้าเว็บและการอ่านรูป/PDF ในแอปยังไม่ได้ลองกับโมเดลจริงตอน merge แอปใหม่ (PR #104)",
                  }}
                />
              </li>
              <li>
                <T
                  v={{
                    en: "The before / after slider is written for this site. It behaves like a drag-or-hover comparison slider and contains no code from React Bits Pro, which needs a licence key.",
                    th: "สไลเดอร์ก่อน / หลังเขียนขึ้นเองสำหรับเว็บนี้ ทำงานแบบสไลเดอร์เปรียบเทียบที่ลากหรือชี้เมาส์ได้ และไม่มีโค้ดจาก React Bits Pro ซึ่งต้องใช้ license key",
                  }}
                />
              </li>
              <li>
                <T
                  v={{
                    en: "This site's own motion is written by hand (CSS and small scripts). The short films were made with Motion from real screenshots of the app.",
                    th: "การเคลื่อนไหวของเว็บนี้เขียนเองด้วย CSS ส่วนฟิล์มสั้นทำด้วย Motion จากภาพหน้าจอ",
                  }}
                />
              </li>
            </ul>
          </div>
        </div>

      </div>
    </article>
  );
}
