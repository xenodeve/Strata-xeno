> **Status: superseded in part.** This is the first design (one long page). The developer then asked for something far more minimal, with smooth
> motion, a carousel, the app's own avatars as the story, and the deep detail on a separate Settings-style page; later also a live demo of the real app,
> scroll-played before / after sections, a self-starting film, a cinematic first screen and a comparison slider. What was built is described in
> `README.md`; what was found and changed on the way is in `docs/ROUND-LOG.md`. The constraints in this file (no external publishing, no benchmark
> without its conditions, shipped / experimental / planned kept apart, nothing private shown, upstream credited) still hold.

# Strata-xeno showcase — design spec

Date: 2026-10-03. Status: approved in conversation by the developer (design, defaults, and "take the real product images to Motion"); the written spec is followed straight by the build, as the developer approved ("เอาตามนั้น").

## 1. Purpose and audience

A public-facing website (Next.js, App Router) that lets a developer who runs LLMs on their own PC understand within ten seconds what **Strata-xeno** is, see the **real** web app, and see what the fork adds, with every claim tied to a source. It is a presentation of the system: it leads with what is proven and good, credits upstream, and puts the conditions of every number next to the number.

Not in scope: publishing the site anywhere (nothing is deployed), a hosted demo that calls a model, any telemetry that looks live.

## 2. Revisions the content is based on

| | |
|---|---|
| Fork | `xenodeve/Strata-xeno` `origin/main` `73649e0` (PR #120), 2026-10-03 14:45 +0700 |
| Upstream | `Niko1221/Strata` `upstream/main` `99f3dbd` (v0.1.38 released 2026-10-03) |
| Merge base | `db4f91a` (upstream v0.1.37); fork 380 ahead, upstream 56 ahead |
| Legacy frontend | `serve/web`, byte-identical at both revisions; the fork serves it at `/classic/` |
| New frontend | `serve/ui` (React 19 + Vite + Tailwind 4), merged by PR #104; `/` serves it unless the config says `"ui": "classic"` (the switch awaits developer confirmation, #110) |

Research (all read-only, in the session scratchpad, not shipped): `ledger-research.md`, `perf-api-research.md`, `issues-survey.md`, `feature-inventory.md`.

## 3. Art direction: "Core Sample"

Strata means layers, and the system has real layers: expert weights live in VRAM of the primary GPU, VRAM of a second GPU, system RAM, and (capacity mode) NVMe. The signature is a **core sample**: a vertical column of those strata drawn in dots (the dot language of the app's own thinking orb), which the visitor can turn and which answers one question — *what does a GPU owning an expert do to RAM?*

Rejected: a timeline-ribbon scene (reads as fake telemetry), an enlarged orb (looks like the app, tells nothing about the advantage).

- **Modern Swiss:** 12-column grid, a single baseline unit (8 px), numbered sections 01–06, flush-left ragged-right, tabular figures, hierarchy by size and weight rather than colour.
- **Editorial minimalism:** large measure of empty space, one idea per screen, reading width 64–72 ch, long-form captions in the margin column.
- **Palette** (from the real app's tokens): `#f5f5f7` / `#131316` light, `#0b0b0d` / `#f1f1f3` dark, accent `#0a62d6` (dark `#5aa2ff`) used only for "status"; no gradients, no glass.
- **Type:** Inter Variable (display + text), Anuphan Variable (Thai), JetBrains Mono Variable (technical labels, numbers' conditions). Same families as the app, self-hosted via `@fontsource-variable`.
- **Easing / durations:** the app's `cubic-bezier(0.23, 1, 0.32, 1)`; 160 / 320 / 600 ms; stagger 38 ms.

## 4. Page structure (single page `/`, bilingual EN / ไทย)

1. **Hero** — headline, one plain sentence (what it is), three CTAs (See the app → `#app`, What it adds → `#adds`, GitHub), the core sample (lazy, below the fold on mobile). The text and CTAs are HTML and never wait for 3D or motion.
2. **01 The app** — real screenshots of the new web app (taken from a real running D2x server, Swift 1.5 IQ2_XS, on the developer's PC; the demo project is a throwaway folder). Tabs: Chat with coding tools, Projects, Dashboard, Live, Requests, Hardware, Settings. Each image opens in a full-size viewer (scroll / zoom, keyboard, Esc). Callouts tie each screen to a claim and to the code that backs it. A **Legacy ↔ New** comparison (upstream's `serve/web` vs `serve/ui`) with a draggable and keyboard-operable divider. The product film made with Motion sits here.
3. **02 The strata** — the 3D core sample plus the explanation of dynamic experts. A VRAM slider changes how many experts the GPU strata own and how many stay in the RAM stratum. Labelled "Illustration of the principle — not telemetry". A reader view (static SVG + text) is always present beside it.
4. **03 What it adds** — 5 headline items, each with evidence and conditions: the app and its coding tools; Claude-Code compatibility of the API; dynamic experts (RAM falls as VRAM grows); decode speed on the daily config; measurement tooling and engineering depth (+ capacity mode as "runs a model larger than RAM").
5. **04 Numbers** — every measured number with its full caption (hardware, model + quant, context, prompt, mode, upstream version, same-session, source link). Only numbers that survive the rules in §6.
6. **05 Everything it adds (explorer)** — filterable, searchable register built from the inventory and issue survey: category, status (shipped / experimental / off by default), evidence links, conditions. Not a long table: collapsed rows, keyboard navigable, deep-linkable.
7. **06 Credits & method** — upstream credit and MIT licence, third-party licences relevant to the UI, the revision stamp, how numbers were measured, what has not been re-measured.

## 5. 3D design

- Three.js (no react-three-fiber), loaded with `import()` after the section is near the viewport (IntersectionObserver), `Points` with a round-dot shader, ~6–9 k points in one draw call, DPR capped at 1.5, orthographic-feel perspective camera.
- Four strata slabs (GPU 1, GPU 2, RAM, NVMe) with labels as HTML overlays (so they are real text). The slider moves the boundary: more GPU capacity → experts rise into the GPU strata, the RAM stratum thins.
- Interaction: horizontal drag / swipe rotates (`touch-action: pan-y`, so vertical swipes scroll the page), keyboard arrows rotate, slider is a native `<input type=range>`. No scroll hijacking.
- Render loop runs only while visible and tab-visible; one static frame is rendered on change when `prefers-reduced-motion` is set; no auto-rotation then.
- Fallback: no WebGL, or `?webgl=0` → the same strata as SVG with the slider still working; content and CTAs are unaffected.
- Honesty: caption "Illustration of the principle. Expert counts and sizes are schematic."

## 6. Content rules (what may be claimed)

- Positive framing: the site presents what Strata-xeno **adds**; it does not show a scoreboard against upstream and does not list upstream's advantages on the narrative pages. It never claims "faster than upstream" without a named configuration, never uses cross-session or retracted numbers, never presents private commit as RAM, never says D2x fits the 42 GB budget, never says dynamic experts is unique (upstream's `--resident-experts` exists), never says capacity mode beats the Low-RAM mode (no same-condition measurement is recorded), never lists benchmarks, testimonials or user counts that do not exist.
- Allowed headline numbers, each with its mandatory caption (from `perf-api-research.md` R.6 / D.1): RAM 16.5 vs 35.3 GiB (two GPUs) and 23.2 vs 33.2 GiB (one GPU), Q2_0, vs upstream v0.1.26, same session; decode 59.0 / 58.9 vs 41.0 / 42.5 tok/s on IQ2_XS, serve mode, 262K, vs upstream v0.1.37 layer split, same session, with the full condition text (including that the same run read a 34K prompt faster on upstream's layer split).
- API claims only for the fork-only items verified in code and tests: `stop_sequences`, `signature_delta`, `disable_parallel_tool_use`, document/PDF blocks, documents or images inside `tool_result`, 529 `overloaded_error` while loading, billing-header strip, `/health` 503, server-side cache slots, request ordering. Upstream had `/v1/messages` first; the site says the fork *extends* it.
- Status words: shipped / experimental / off by default. Planned items are not shown as features.
- A short "Method & limits" block states revisions, one-PC measurement, Windows/NVIDIA validation, and that upstream v0.1.38 is not compared.
- Attribution: Strata is upstream's (Niko1221 and contributors, MIT); the Legacy frontend is upstream's work and is shown as such.

## 7. Real-UI rules

- Screenshots come from the real server running the real model (D2x profile copy on its own port with its own history folder), not the mock server's fake answers. Nothing from the developer's own chats, imported skills, MCP servers, API keys, home paths or history is shown; the demo project is a throwaway folder created for this.
- Interactive components of the app are **not** redistributed: `REFERENCES.md` marks React Bits (MIT + Commons Clause, "inside this app only") and transitions.dev (no licence text) as unsettled. The showcase uses pictures and a recorded film; an embedded live demo is deferred until the licences are confirmed.
- Images are labelled with what they are (real UI, throwaway project, this PC).

## 8. Motion

- Motion language: one easing, three durations, 38 ms stagger, opacity + 8–12 px translate for reveals, no scroll hijacking, no looping decoration, every effect has a reduced-motion state (instant or fade only).
- The Motion MCP available here (Motion, mcp.motion.so) is a **video generator**, not a web-animation design tool. It is used for what it does: a short product film from the real screenshots (uploaded with `upload_asset`, rendered with `create_video`, no credits purchased). The web's own motion (CSS / Web Animations API) is written by hand and is reported as such.

## 9. Accessibility and performance

- Semantic landmarks and headings, skip link, visible focus, keyboard-operable tabs / compare divider / viewer / explorer, `aria-live` for the slider's readout, contrast ≥ 4.5:1 for text, language attributes per block, tap targets ≥ 44 px on touch.
- Fonts self-hosted with `font-display: swap`, images served by `next/image` (AVIF/WebP), explicit dimensions (no layout shift), 3D lazy-loaded, nothing blocks first paint.
- Metadata, Open Graph / Twitter image (generated from a real screenshot), `lang`, canonical-less (not deployed).
- Verification by real runs: Playwright at 1440 / 1024 / 390 px, reduced-motion emulation, WebGL-off fallback, keyboard walk-through, console clean, `next build`, type check; Lighthouse only if it can be run here, otherwise reported as not measured.

## 10. Files

```
app/layout.tsx, app/page.tsx, app/globals.css, app/opengraph-image.*
components/  Hero, TheApp, Viewer, Compare, Strata (3D client), StrataFallback, Adds, Numbers, Explorer, Credits, Nav, LangToggle
data/        content.ts (EN/TH copy), numbers.ts, register.ts (explorer rows), sources.ts
lib/         i18n, motion helpers, three scene
public/shots/ real screenshots (WebP + PNG originals)
public/film/ the Motion film (if rendered)
docs/        this spec, ROUND-LOG.md (critique rounds), RESEARCH-NOTES.md (what was verified)
```

## 11. Build and review plan

1. Scaffold Next.js, tokens, layout, fonts, i18n.
2. Content data from the research (every row sourced).
3. Sections in order: Hero → The app (+ viewer, compare) → Strata (3D + fallback) → Adds / Numbers → Explorer → Credits.
4. Three critique rounds with real screenshots: (1) first impression, art direction, narrative, relation to the real UI, clarity of what it adds; (2) typography, spacing, composition, showcase, responsive desktop/mobile; (3) interaction, motion, keyboard/touch, reduced motion, WebGL fallback, performance, correctness of every number. Fix after each; log in `docs/ROUND-LOG.md`.
5. Deliver: how to run, files, Motion usage list, screenshots, build/type results, limits and what could not be checked.
