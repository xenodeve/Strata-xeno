"use client";

import { useEffect, useState } from "react";
import { STATUS_LABEL, type Feature } from "@/data/features";
import { T, useT } from "@/lib/i18n";
import { CHARTS } from "@/data/numbers";
import { Chart } from "../charts/Chart";
import { AppScene, type SceneId } from "../scenes";
import { HARNESS_LABEL, HarnessIcon, type HarnessId } from "../HarnessIcon";

const APPS: HarnessId[] = ["claude-code", "codex", "antigravity", "gemini-cli", "cursor", "claude-desktop", "agents"];

/** A schematic, not a screenshot: the Import pages list what is on the developer's own PC. */
function ImportDiagram() {
  return (
    <figure className="importviz">
      <div className="importviz__from">
        <p className="mono importviz__h"><T v={{ en: "Your other apps", th: "แอปอื่นของคุณ" }} /></p>
        <ul>
          {APPS.map((a) => (
            <li key={a} className="mono">
              <HarnessIcon id={a} size={22} />
              {HARNESS_LABEL[a]}
            </li>
          ))}
        </ul>
      </div>
      <div className="importviz__arrow" aria-hidden="true">
        <span />
      </div>
      <div className="importviz__to">
        <p className="mono importviz__h">Strata-xeno</p>
        <dl>
          <div>
            <dt>Skills</dt>
            <dd><T v={{ en: "in use · switch an app off", th: "ใช้ได้ทันที · ปิดทีละแอปได้" }} /></dd>
          </div>
          <div>
            <dt>MCP servers</dt>
            <dd><T v={{ en: "press Import on one", th: "กด Import ที่ตัวที่ต้องการ" }} /></dd>
          </div>
          <div>
            <dt>Memory</dt>
            <dd><T v={{ en: "off until you switch it on", th: "ปิดจนกว่าคุณจะเปิด" }} /></dd>
          </div>
        </dl>
      </div>
      <figcaption className="cap mono">
        <T v={{ en: "Schematic. Read-only scan; Rescan any time.", th: "แผนภาพอธิบาย สแกนแบบอ่านอย่างเดียว สแกนใหม่ได้ทุกเมื่อ" }} />
      </figcaption>
    </figure>
  );
}

function TextPanel({ f }: { f: Feature }) {
  return (
    <div className="textpanel">
      <p className="textpanel__q">
        <T v={f.summary} />
      </p>
    </div>
  );
}

/** Scenes that are one narrow column (a side panel, a card, a dialog) sit in a narrower frame than the full pages. */
const NARROW = new Set<SceneId>(["permission", "modes", "status", "git", "plan", "memory", "context", "project", "marks"]);

function Media({ f }: { f: Feature }) {
  const t = useT();
  const [i, setI] = useState(0);
  useEffect(() => setI(0), [f.id]);
  if (f.id === "import") return <ImportDiagram />;
  if (!f.scenes?.length) return <TextPanel f={f} />;
  const many = f.scenes.length > 1;
  const sc = f.scenes[i];
  return (
    <div className="media">
      <AppScene key={f.id + i + (sc.lang ?? "")} id={sc.scene} alt={sc.alt} lang={sc.lang} glide className={"media__scene" + (NARROW.has(sc.scene) ? " media__scene--narrow" : "")} />
      {many && (
        <div className="thumbs" role="group" aria-label={t({ en: "Screens of this feature", th: "หน้าจอของฟีเจอร์นี้" })}>
          {f.scenes.map((s, k) => (
            <button key={s.scene} type="button" className="thumbs__b mono" aria-pressed={k === i} onClick={() => setI(k)}>
              {s.label ? t(s.label) : k + 1}
            </button>
          ))}
        </div>
      )}
      <p className="cap mono">
        {t(sc.alt)} <T v={{ en: "Rebuilt from one real run; the figures are what that run showed.", th: "สร้างใหม่จากการรันจริงหนึ่งครั้ง ตัวเลขคือที่การรันนั้นแสดง" }} />
      </p>
    </div>
  );
}

export function FeaturePanel({ f }: { f: Feature }) {
  const t = useT();
  return (
    <article className="topic">
      <header className="topic__head">
        <p className="mono topic__meta">
          <span className={`badge badge--${f.status}`}>
            <T v={STATUS_LABEL[f.status]} />
          </span>
        </p>
        <h2 className="h3">
          <T v={f.title} />
        </h2>
        <p className="topic__sum">
          <T v={f.summary} />
        </p>
      </header>

      <Media f={f} />

      {f.chart && <Chart spec={CHARTS[f.chart]} />}

      <ul className="points">
        {f.points.map((p) => (
          <li key={p.en}>
            <T v={p} />
          </li>
        ))}
      </ul>

      {f.note && (
        <p className="note">
          <T v={f.note} />
        </p>
      )}

      <ul className="evidence" aria-label={t({ en: "Evidence", th: "หลักฐาน" })}>
        {f.evidence.map((e) => (
          <li key={e.href}>
            <a className="mono" href={e.href} rel="noopener">
              {e.label} <span aria-hidden="true">↗</span>
            </a>
          </li>
        ))}
      </ul>
    </article>
  );
}
