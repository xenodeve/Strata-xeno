"use client";

import BranchedMenu from "../text/BranchedMenu";
import { GradientWaveText } from "../text/Fx";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { FEATURES } from "@/data/features";
import { MEASURED } from "@/data/numbers";
import { ALL_TOPICS, GROUPS, type Topic } from "@/data/topics";
import { T, useT } from "@/lib/i18n";
import { Reveal } from "../Reveal";
import { CreditsPanel } from "./CreditsPanel";
import { FeaturePanel } from "./FeaturePanel";
import { Icon } from "./icons";
import { CountsPanel, MeasuredCard } from "./NumbersPanel";
import { RegisterPanel } from "./RegisterPanel";

function readHash(): string {
  const h = decodeURIComponent(window.location.hash.replace(/^#\/?/, ""));
  return ALL_TOPICS.some((t) => t.id === h) ? h : ALL_TOPICS[0].id;
}

/** The tree of topics: the supplied React Bits BranchedMenu (a line per group, a curved branch to each topic, the open one drawn in). */
function Tree({ current, onPick }: { current: string; onPick: (id: string) => void }) {
  const t = useT();
  const items = useMemo(
    () =>
      GROUPS.map((g) => ({
        value: g.id,
        label: t(g.label),
        children: g.topics.map((x) => ({ value: x.id, label: t(x.label), icon: <Icon name={x.icon} /> })),
      })),
    [t],
  );
  const wrap = useRef<HTMLDivElement>(null);
  // a label that is longer than its row is carried across it (and back), so it can be read whole instead of being cut off
  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const measure = () => {
      el.querySelectorAll<HTMLElement>(".branched-menu__label").forEach((l) => {
        const text = l.firstElementChild as HTMLElement | null;
        if (!text) return;
        const over = Math.ceil(text.scrollWidth - l.clientWidth);
        l.style.setProperty("--shift", `${Math.max(0, over)}px`);
        l.style.setProperty("--dur", `${(3.2 + Math.max(0, over) * 0.045).toFixed(2)}s`); // a longer way takes longer
        l.toggleAttribute("data-marquee", over > 1);
      });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    void document.fonts?.ready.then(measure);
    return () => ro.disconnect();
  }, [items]);
  return (
    <div ref={wrap}>
    <BranchedMenu
      items={items}
      defaultOpen={GROUPS.map((_, i) => i)} /* every group open, as the app's own Settings menu is */
      active={current}
      onSelect={(id) => onPick(id)}
      ariaLabel={t({ en: "Topics", th: "หัวข้อ" })}
      color="var(--ink)"
      accentColor="var(--ink)"
      lineColor="color-mix(in srgb, var(--ink) 24%, transparent)"
      width={300}
      fontSize={14}
      rowHeight={38}
    />
    </div>
  );
}

function Panel({ topic }: { topic: Topic }) {
  if (topic.kind === "feature") {
    const f = FEATURES.find((x) => x.id === topic.id);
    return f ? <FeaturePanel f={f} /> : null;
  }
  if (topic.kind === "measured") {
    const m = MEASURED.find((x) => x.id === topic.id);
    return m ? (
      <div className="topic">
        <MeasuredCard m={m} />
      </div>
    ) : null;
  }
  if (topic.kind === "counts") return <CountsPanel />;
  if (topic.kind === "register") return <RegisterPanel />;
  return <CreditsPanel />;
}

export function DetailsApp() {
  const t = useT();
  const [id, setId] = useState(ALL_TOPICS[0].id);
  const [menu, setMenu] = useState(false);
  const panel = useRef<HTMLDivElement>(null);
  const first = useRef(true);

  useEffect(() => {
    setId(readHash());
    const on = () => setId(readHash());
    window.addEventListener("hashchange", on);
    window.addEventListener("popstate", on);
    return () => {
      window.removeEventListener("hashchange", on);
      window.removeEventListener("popstate", on);
    };
  }, []);

  const pick = useCallback((next: string) => {
    setId(next);
    setMenu(false);
    history.pushState(null, "", `#${next}`);
  }, []);

  // after a change the panel is brought into view if it has scrolled away (small screens)
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    const el = panel.current;
    if (el && el.getBoundingClientRect().top < 60) el.scrollIntoView({ block: "start", behavior: "smooth" });
  }, [id]);

  const at = ALL_TOPICS.findIndex((x) => x.id === id);
  const topic = ALL_TOPICS[at] ?? ALL_TOPICS[0];
  const prev = ALL_TOPICS[at - 1];
  const next = ALL_TOPICS[at + 1];

  return (
    <div className="wrap details">
      <Reveal as="header" className="details__head">
        <h1 className="display display--sm">
          <GradientWaveText>
            <T v={{ en: "Details", th: "รายละเอียด" }} />
          </GradientWaveText>
        </h1>
        <p className="lede">
          <T
            v={{
              en: "Everything Strata-xeno adds, topic by topic. The screens move: they are rebuilt from one real run. Every claim links to the code.",
              th: "ทุกสิ่งที่ Strata-xeno เพิ่ม ทีละหัวข้อ หน้าจอขยับได้ สร้างใหม่จากการรันจริงหนึ่งครั้ง ทุกคำกล่าวอ้างมีลิงก์ไปยังโค้ด",
            }}
          />
        </p>
      </Reveal>

      <div className="details__grid">
        <aside className="details__side">
          <details className="treem" open={menu} onToggle={(e) => setMenu((e.currentTarget as HTMLDetailsElement).open)}>
            <summary>
              <Icon name={topic.icon} />
              <span>
                <T v={topic.label} />
              </span>
              <span className="treem__chev" aria-hidden="true">
                ▾
              </span>
            </summary>
            {/* built when it is opened: the menu measures its rows, and in a closed drop-down they have no size yet */}
            {menu && <Tree current={id} onPick={pick} />}
          </details>
          <div className="details__treewide">
            <Tree current={id} onPick={pick} />
          </div>
        </aside>

        <div className="details__panel" ref={panel}>
          <div key={topic.id} className="swap">
            <Panel topic={topic} />
            <nav className="steps" aria-label={t({ en: "Previous and next topic", th: "หัวข้อก่อนหน้าและถัดไป" })}>
              {prev ? (
                <button type="button" className="steps__b" onClick={() => pick(prev.id)}>
                  <span className="mono">← <T v={{ en: "Previous", th: "ก่อนหน้า" }} /></span>
                  <b>
                    <T v={prev.label} />
                  </b>
                </button>
              ) : (
                <span />
              )}
              {next && (
                <button type="button" className="steps__b steps__b--next" onClick={() => pick(next.id)}>
                  <span className="mono">
                    <T v={{ en: "Next", th: "ถัดไป" }} /> →
                  </span>
                  <b>
                    <T v={next.label} />
                  </b>
                </button>
              )}
            </nav>
          </div>
        </div>
      </div>
    </div>
  );
}
