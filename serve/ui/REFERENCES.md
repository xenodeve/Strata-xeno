# serve/ui - licences and references

What ships in `dist/` and under which licence. Full texts are in `licenses/`.

| What | Licence | Where |
|---|---|---|
| Inter (variable, latin) | SIL OFL 1.1 | `licenses/Inter-OFL.txt`, via `@fontsource-variable/inter` |
| Anuphan (variable, thai + latin) | SIL OFL 1.1 | `licenses/Anuphan-OFL.txt`, via `@fontsource-variable/anuphan` |
| JetBrains Mono (variable, latin) | SIL OFL 1.1 | `licenses/JetBrainsMono-OFL.txt`, via `@fontsource-variable/jetbrains-mono` |
| Hugeicons free icons | MIT (file checked in the package, 2026-10-01) | `licenses/Hugeicons-MIT.txt`, `@hugeicons/core-free-icons` + `@hugeicons/react` |

SF Pro / SF Thai are named first in the font stack and used only where the OS has them. They are never shipped (Apple licence).

## Ship-gate exceptions

- `design-ship-gate` check 1 (at most two font families): three ship (Inter, Anuphan for Thai, JetBrains Mono for code). Decided in the UI handoff (Q11), so the gate's number gives way; mono is used only for code and measurements.
- Checks 2 (Open Graph), 4 and 6 (hero) are for static landing pages and do not apply to this app.

## React Bits (MIT + Commons Clause: used inside this app only, never redistributed as components)

- `PromptBar` (the chat's composer, `src/components/PromptBar.tsx` + `src/prompt-bar.css`): adapted. Our tokens replace the fixed colours, the arrow-to-stop morph is a small rAF tween instead of the `motion` package, the sparks at the top thinking level run only while the field is focused, nothing is being written and the tab is visible, the menu holds this app's own actions (attach, new chat, save as Markdown), the effort slider is `reasoning_effort`, and the focus-then-open order is fixed (the original closed a menu it had just opened when the field was not focused). Pasted by the developer on 2026-10-01.
- `BranchedMenu` (the Settings page's menu of sections and topics, and the chat sidebar's Projects and Recents, `src/components/BranchedMenu.tsx` + `src/branched-menu.css`): adapted from code the developer pasted on 2026-10-02 **without naming its source or licence**; the HugeIcons idiom and the `Libraries.dev`/React Bits style suggest it is a React Bits component, so it is listed here under the same terms (MIT + Commons Clause, used inside this app only) **until the source is confirmed - to verify**. Changes: TypeScript; controlled (the address says which topic is in use and a link opens its section); the app's tokens for colour and its ease; `inert` on a folded section; one look (no size props); for the sidebar also an optional icon, a trailing control, a custom body (the rename field), a disabled look, an ellipsis for long labels, an empty section and the full width of the column.
- `LatticeLoader` and `ThoughtLine` (a pair, for the agent's thinking: `src/components/thought.tsx` + `src/thought.css`, used in `pages/chat/Messages.tsx`): adapted. The `motion` package is gone (the breathing is CSS, the shimmer is the original's CSS), the lattice keeps only the 3 x 3 patterns and has no label or timer of its own, the line's open state belongs to the chat (open while it thinks if the setting asks, closed once the answer starts, the user's click wins) and its trace is the reasoning text, the lattice's pattern follows what the agent is doing (`latticePattern` in `src/lib/orbs.ts`, tested: thinking = orbit, a lookup = ripple, a tool call = snake, the next step after a tool = spiral) and it ends as a tick, or a cross when the reply failed. Pasted by the developer on 2026-10-01.

- **bot-avatars 0.1.2 (Libraries.dev, Jakub Antalik): MIT**, `licenses/BotAvatars-MIT.txt`, vendored into `src/vendor/bot-avatars/` (the built module and its types, unchanged). A choice for what stands for a status (About, or the button in the header, which goes round four: Orbs; Orbs + Loading, the default: the lattice of dots by the thinking as the app had it and orbs elsewhere (the lattice marks the thinking in Loading only too); Loading only, a lattice or matrix pattern for each of the nine forms and no orb; Avatar; kept in `localStorage` `strata.avatar`). Which avatar is a second choice in About (`strata.avatar.type`): by status (each of the nine orb forms has a bot shape of its own, `src/lib/avatar.ts`, tested), one of the eighteen for every status, or Random (one drawn for each place when the page opens). At work it hops, at rest (an idle server) it looks around, dormant (an unloaded model) or paused it sleeps. Loaded only when chosen (a lazy import, its own script in `dist/assets`), so with the orbs nothing extra is downloaded. Small ones use the lighter shading and do not follow the pointer.
- **transitions.dev (Jakub Antalik): "Number pop-in", "Reasoning stream", "Plus to menu morph", "Spinning counter" and "Matrix dot loader"**, in the CLI's free list, pulled with `npx transitions-dev add <name>` on the developer's instruction on 2026-10-01 (the developer's "Dropdown menu morph" is `plus-menu-morph` there; `menu-dropdown` is a different one). The files the CLI wrote carry **no licence text**, so the terms are not recorded here: check them on transitions.dev before the app is shared beyond this PC. The CSS is in `src/styles.css` and `src/prompt-bar.css` (namespaced `.t-*`, tokens as variables, the reduced-motion guards kept). Changes: a digit that was already there is not animated; the stream is a window onto text that grows, with native scrolling, not the demo's looping ticker; the morph is anchored at the bottom left and sized for three rows; the reel is one text node of fifty lines, shorter and in em. The matrix: the dot and the gap follow the size it is shown at, rounded dots, our ink for its two colours; it is one of the two loader families of the loading style (with the lattice of dots), see `src/lib/avatar.ts` `loaderFor`. Used: Pop for figures that only grow (token counts, elapsed), the reels for figures that go up and down (speeds, loads, temperatures), the stream for the thinking window, the morph for the + menu.
- `playwright-core` 1.63.0 (Microsoft, Apache-2.0): a **dev dependency only**, it drives the browser checks in `serve/ui/e2e/` (`bun run e2e`) and is not part of the built app.

## Reference only, own implementation (no code copied)

Paid or unclear-licence work that informed the design. Recorded here as the handoff requires; add each one the
moment it is looked at.

(none yet)

## Developing without the GPUs

`python serve/ui/dev/mock_server.py [port]` (default 8099) serves the app at `/next/` with no model: hardware, routes and history are real,
the answer and the engine's STATS / prefill chunks are labelled fixtures. `serve/ui/fixtures/` holds a real `/metrics` and `/health` recorded from
the daily server on 2026-10-01 (idle, one card visible: the S0 bug). The daily server on :8091 is never touched.

## Build

`cd serve/ui && bun install && bun run build` writes `dist/` and `dist/source-hash.txt`. `dist/` is committed so the
runtime needs no Node; `python -m unittest serve.test_ui` fails when `src/` changed and `dist/` did not.

| thinking-orbs 0.3.2 (Libraries.dev, Jakub Antalik) | MIT | `licenses/ThinkingOrbs-MIT.txt`, vendored into `src/vendor/thinking-orbs/`. Used in, by what the orb shows (`src/lib/orbs.ts`, tested): the server (idle = the searching globe turning slowly at every display frame, reading the prompt = listening, writing = composing at the pace of tok/s, queued or not answering = connecting, unloaded = shaping, turning slowly; no orb is a frozen picture, only reduced motion stills them), a GPU card (idle = the same slow searching globe at 30 frames a second, working, working hard = weaving, held back by a limit = solving), an assistant message (waiting = breathing, thinking = solving, a tool running = connecting or searching, the next step after a tool = weaving, writing = composing), the empty chat while you type (listening), pages that wait for data (connecting, working), the timeline file reading, an MCP server starting. The status line's shimmer is the library web demo's `t-shimmer`, re-written in CSS. `haplollc/ThinkingOrbs` (a SwiftUI port of the same designs, MIT) was read for its state-to-design recipe and not used: a Swift package cannot run in a page. Two local patches to the vendored library, marked `xeno` in `index.es.js` and `types.d.ts`: an `fps` prop (frames are skipped, the loop still ticks) and a `scale` prop (the tuned size-64 orb drawn 2.5 times bigger on the empty chat, as vectors: the canvas is larger, the geometry is not stretched); the empty chat's orb shows a different one of the nine forms every 5 s while the server is idle and nobody types (`nextDesign`, tested), and the real state otherwise; re-apply it if the library is updated. A change of design is a 450 ms dissolve (the old form swells and blurs away, the new one settles in) in `styles.css`. |
