# Strata-xeno showcase

A showcase site for [Strata-xeno](https://github.com/xenodeve/Strata-xeno), a fork of [Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors.
It lives in the `website/` folder of the Strata-xeno repository and is deployed to Vercel from there (Root Directory: `website`). The live demo needs the real server running on the same machine, so on the deployed site that section says how to start it; run the site locally to try the real app.

Next.js 16 (App Router) · React 19 · TypeScript · three.js · ogl · vgpu (WebGPU) · plain CSS. English by default, Thai with the toggle; light, dark or system theme.

## Run it

```sh
npm install
npm run dev          # http://localhost:3000   (or: npm run build && npm start)
```

The page works without anything else. To switch on **Try the real app** (the live demo), start the real Strata-xeno server in its no-GPU demo mode in a second terminal:

```sh
STRATA_SRC=<your Strata-xeno checkout> npm run demo      # PowerShell: $env:STRATA_SRC="..."; npm run demo
```

It runs the repo's own mock server (`serve/ui/dev/mock_server.py`): a script answers instead of the model, the engine's statistics are fixtures, and the history lives in an empty temporary folder, so nothing of anyone's own chats, skills or paths can appear. The site reaches it through a small guarded proxy (`app/demo/[[...path]]/route.ts`): localhost only, and anything that changes something must come from this site's own pages. Without the backend the section says how to start it. (The framed app keeps its own state in the visitor's browser under `strata.*`, on the same origin as the site.)

**On the deployed site** the same demo runs without anyone starting anything: the repo root has `api/index.py`, a Vercel (Python/FastAPI) function that starts that same mock server inside itself and passes requests to it. The website project sets `DEMO_BACKEND` to that function's address, and `next.config.ts` then rewrites `/demo/*` to it. The function is a short allowlist, not a pass-through: it serves the app's files and the read-only status pages, accepts a POST only for the chat and the answers to its permission, question, steer and cancel cards, never serves the pages that list folders, files, git or notes, and replaces the chat's working folder with a temp folder of its own. What differs from a PC: the Hardware page shows the small cloud machine, not a GPU; the history lives in the function's temp folder and goes with the instance; the demo is public.

Other commands: `npm run typecheck`. `NEXT_DIST_DIR=<folder>` builds to another folder, so a build does not disturb a running dev server.

## What is on the page

| | |
|---|---|
| First screen | The app's own live avatar walks through the nine moods by itself; the ring around it is the nine, the glow takes the colour of the one that is on, four bands of dots drift behind and soft points in the moods' colours float around it. Pick any of the 18 shapes to keep one. The headline's second half carries a slow colour wave. |
| Nine moods | One mood per scroll step, each with a piece of the app's interface that plays (typing, tool cards, a permission question, a plan being ticked, the hardware filling). The scroll is the browser's own; buttons, arrow keys and swipes scroll to the matching place. |
| Tools band | A line that says what the chat can do one verb at a time (a rolling word), and two strips: the fifteen tools and the nine moods. |
| Try the real app | The real web app, running (see above). |
| Classic to new | An app window that leans toward the pointer. The classic screen plays first; scrolling breaks it into pixels from the left edge into the new screen, which starts to play as it lands (the chat first, then the monitor). Callouts appear as the parts they point at arrive. Click or Enter swaps by hand; scrolling hands it back. |
| The film | Two short films made with Motion from the app's real screenshots (a 10-second showreel and a 24-second film of the marks). Nothing is downloaded while you scroll past: a film is fetched only after half of it has stayed on screen for a moment, then plays from memory, muted. |
| The strata | A 3D core sample (one draw call). It is a schematic of where the expert memory sits, not telemetry. `?webgl=0` shows the same facts as a reader. |
| Numbers | Two measurements as comparison charts (scale from zero, the direction that is better, the difference, the conditions under the chart). The same charts are on the Numbers topic of `/details`; the RAM chart is also on Dynamic experts and a relative chart on Capacity mode. |
| Ask first. | A dark band where points wait scattered and gather into the words when the pointer comes over (or, on a touch screen, when the band is on screen); they drift a little, are pushed away by the pointer and take a colour while they are pushed. The component (`components/text/ParticleText.tsx`) is this site's own code, written from the usage of the one that was supplied (its props); the supplied source was not available. |
| The shapes (on the Details page, under the topics) | A dark band of dots that become squares, circles and triangles, with the name cut out of the field; a pointer sends ripples through it (WebGPU; a still band where there is none). |
| Nothing leaves it / the name | Two big words made of canvases: one bends around the pointer (WebGL) over a slow aurora, one can be picked apart letter by letter over drifting points. |
| `/details` | The deep detail, laid out like the app's Settings: a tree of topics with connectors and icons, one panel per topic (`#coding-tools`, `#dynamic-experts`, `#credits`, ...). Every screen in it is a scene that plays. |

Text moves too: words rise out of masks, a highlighter sweeps behind key words, a line scrambles into place, a strip of tools drifts. Nothing moves with reduced motion, and a page without scripts shows the finished text.

## The screens are scenes, not pictures

Every screen of the app on this site (components/scenes) is rebuilt by hand in React and CSS, from one real run of the real web app, and played from a clock: a scene is a function of the time since it started, so it is exact, cheap and needs no animation library. The words are the app's own, copied from the run; the figures are what that run showed, once, on one PC, and are not a benchmark. None of the app's own component code is used, but the motion follows it: every duration, curve, distance and stagger is read from the app's stylesheets and components (`components/scenes/motion.ts` lists each one and the rule it comes from), the orbs are the app's own library (thinking-orbs, from npm), the loader beside a thought is the React Bits LatticeLoader the app uses, a figure that changes pops its digits or turns its reels as the app's does, a thought is written in a window that steps down, a section opens through grid rows, and a box that grows glides to its new height. The classic scenes move only as the classic UI moves (it has almost no motion). All code on the site is highlighted by highlight.js in Monokai Classic, drawn on its own dark ground (`app/monokai.css`). The original captures that the scenes follow are kept in `docs/reference-shots/` (they are not served).

## Key files

- `app/page.tsx`, `app/details/page.tsx`, `app/layout.tsx`, `app/globals.css` (tokens come from the real app: greys, accent, one easing, durations 160 / 320 / 600 ms), `app/effects.css` (text effects, charts, canvases), `app/scenes.css` (the app's light skin for the scenes)
- `components/home/` (`Hero`, `Story`, `Tools`, `Demo`, `Compare`, `Film`, `Strata`, `Stats`, `Statement`, `Cta`), `components/details/`
- `components/scenes/` (`Scene.tsx` the clock and the frame (and the height glide), `motion.ts` the app's timings and curves, `parts.tsx` the moving parts (enter, collapse, pop, reels, matrix, thought line, reason stream), `code.tsx` highlighted code, `kit.tsx` the small parts and the orb, `*-parts.tsx` the parts each group of scenes needs, `chat.tsx`, `monitor.tsx`, `panels.tsx`, `settings.tsx`, `legacy.tsx`, `index.tsx` the registry)
- `components/text/` (`Fx.tsx` the text effects, `TechText.tsx` / `WarpText.tsx` / `Particles.tsx` / `Aurora.tsx` / `ShapeWaves.tsx` / `LatticeLoader.tsx` the canvases and the loader, `BranchedMenu.tsx` the topic tree of `/details`, `ParticleText.tsx` the word of points, `Words.tsx` their wrappers), `components/PixelSwap.tsx`, `components/charts/Chart.tsx`
- `lib/strata.ts` + `lib/strataScene.ts` (3D)
- `data/` (all copy that is data: moods, topics, features, numbers and the charts, the issue / PR register with links)
- `vendor/bot-avatars/` (MIT, with its licence), `public/film/` (the two Motion films and their posters)

## Screenshots

Captured from the production build in a real browser (Chromium via Playwright), in `docs/screenshots/`: `desktop-1-hero`, `desktop-2-moods` (a scene typing its answer), `desktop-3-live-demo` (the real app, the agent asking before it acts), `desktop-4-compare-pixelswap` (the classic screen, which has finished streaming its answer, with its code in Monokai Classic), `desktop-5-strata-3d`, `desktop-7-numbers-charts`, `desktop-8-aurora-statement`, `desktop-9-shapes-band` (the WebGPU band at the foot of the Details page), `mobile-1-hero`, `mobile-2-compare-pixelswap`, `mobile-3-moods`, `mobile-4-numbers-chart`, `mobile-5-shapes-band`. Scenes are caught mid-play, so a frame shows one moment of each.

## Key files

- `app/page.tsx`, `app/details/page.tsx`, `app/layout.tsx`, `app/globals.css` (tokens come from the real app: greys, accent, one easing, durations 160 / 320 / 600 ms), `app/effects.css` (text effects, charts, canvases), `app/scenes.css` (the app's light skin for the scenes)
- `components/home/` (`Hero`, `Story`, `Tools`, `Demo`, `Compare`, `Film`, `Strata`, `Stats`, `Statement`, `Cta`), `components/details/`
- `components/scenes/` (`Scene.tsx` the clock and the frame (and the height glide), `motion.ts` the app's timings and curves, `parts.tsx` the moving parts (enter, collapse, pop, reels, matrix, thought line, reason stream), `code.tsx` highlighted code, `kit.tsx` the small parts and the orb, `*-parts.tsx` the parts each group of scenes needs, `chat.tsx`, `monitor.tsx`, `panels.tsx`, `settings.tsx`, `legacy.tsx`, `index.tsx` the registry)
- `components/text/` (`Fx.tsx` the text effects, `TechText.tsx` / `WarpText.tsx` / `Particles.tsx` / `Aurora.tsx` / `ShapeWaves.tsx` / `LatticeLoader.tsx` the canvases and the loader, `BranchedMenu.tsx` the topic tree of `/details`, `Words.tsx` their wrappers), `components/PixelSwap.tsx`, `components/charts/Chart.tsx`
- `lib/strata.ts` + `lib/strataScene.ts` (3D)
- `data/` (all copy that is data: moods, topics, features, numbers and the charts, the issue / PR register with links)
- `vendor/bot-avatars/` (MIT, with its licence), `public/film/` (the two Motion films and their posters)

## Screenshots

Captured from a production build in a real browser (Chromium via Playwright), in `docs/screenshots/`.

## Limits worth knowing

- The measurements are from one PC, each with its conditions; none is compared across sessions. Upstream v0.1.38 was not compared. A chart only draws figures that the Numbers section gives.
- Frame rate, Lighthouse and a real phone were not measured while building this (the capture browser here runs at about one frame per second). Checked instead: build, types, axe-core, layout at 390 / 1024 / 1440 px, keyboard, touch-style pointer events, reduced motion, WebGL off, layout shift on a throttled phone.
- The demo's model is a script. Web access and image / PDF reading in the app had not been tried with a real model when the new app was merged.
- Sound is not implemented.
- The films are fetched by the page and played from memory, so a download manager that watches video elements has no address to catch. This was not tested with a download manager; if one still pops up, set the Play button to be the only way to start a film.
- Eight React Bits components (TechText, WarpText, PixelSwap, Particles, Aurora, ShapeWaves, LatticeLoader, BranchedMenu) were supplied by the project owner and are used as supplied (MIT with the Commons Clause); the Comparison Slider from React Bits Pro was not used (it needs a licence key).

Credits and method: `/details#credits`.
