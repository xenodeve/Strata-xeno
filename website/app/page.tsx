import { existsSync } from "node:fs";
import { join } from "node:path";
import { Footer } from "@/components/Footer";
import { Nav } from "@/components/Nav";
import { Compare } from "@/components/home/Compare";
import { Demo } from "@/components/home/Demo";
import { Film, type FilmItem } from "@/components/home/Film";
import { Hero } from "@/components/home/Hero";
import { Stats } from "@/components/home/Stats";
import { Gather } from "@/components/home/Gather";
import { Statement } from "@/components/home/Statement";
import { Story } from "@/components/home/Story";
import { Strata } from "@/components/home/Strata";
import { Tools } from "@/components/home/Tools";
import { Cta } from "@/components/home/Cta";

export default function Page() {
  // the short films are shown only when they have been rendered and added to public/film/
  const has = (f: string) => existsSync(join(process.cwd(), "public", "film", f));
  const films: FilmItem[] = [];
  if (has("strata-xeno-showreel.mp4"))
    films.push({
      id: "showreel",
      src: "/film/strata-xeno-showreel.mp4",
      poster: "/film/strata-xeno-showreel.jpg",
      tab: { en: "The showreel", th: "โชว์รีล" },
      length: { en: "10 seconds.", th: "10 วินาที" },
      aria: "A 10-second motion-graphics film of Strata-xeno: its name, the nine moods, the coding tools that ask first, and the memory that gives RAM back. It has on-screen text and no speech.",
      caption: {
        en: "Made with Motion from the app's real screenshots. Plays by itself, muted; use the controls for sound.",
        th: "ทำด้วย Motion จากภาพหน้าจอจริงของแอป เล่นเองแบบปิดเสียง ใช้ปุ่มควบคุมเพื่อเปิดเสียง",
      },
    });
  if (has("strata-xeno-marks.mp4"))
    films.push({
      id: "marks",
      src: "/film/strata-xeno-marks.mp4",
      poster: "/film/strata-xeno-marks.jpg",
      tab: { en: "The nine marks", th: "ทั้งเก้ามาร์ก" },
      length: { en: "24 seconds.", th: "24 วินาที" },
      aria: "A 24-second film of the app's nine moods and avatars. It has on-screen captions and no speech.",
      caption: {
        en: "Made with Motion from the app's real screenshots. Plays by itself, muted; use the controls for sound.",
        th: "ทำด้วย Motion จากภาพหน้าจอจริงของแอป เล่นเองแบบปิดเสียง ใช้ปุ่มควบคุมเพื่อเปิดเสียง",
      },
    });

  return (
    <>
      <Nav />
      <main id="main">
        <Hero />
        <Story />
        <Tools />
        <Demo />
        <Compare />
        {films.length > 0 && <Film films={films} />}
        <Strata />
        <Stats />
        <Gather />
        <Statement />
        <Cta />
      </main>
      <Footer />
    </>
  );
}
