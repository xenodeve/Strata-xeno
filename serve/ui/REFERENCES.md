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
- `LatticeLoader` and `ThoughtLine` (a pair, for the agent's thinking: `src/components/thought.tsx` + `src/thought.css`, used in `pages/chat/Messages.tsx`): adapted. The `motion` package is gone (the breathing is CSS, the shimmer is the original's CSS), the lattice keeps only the 3 x 3 patterns and has no label or timer of its own, the line's open state belongs to the chat (open while it thinks if the setting asks, closed once the answer starts, the user's click wins) and its trace is the reasoning text, the lattice's pattern follows what the agent is doing (`latticePattern` in `src/lib/orbs.ts`, tested: thinking = orbit, a lookup = ripple, a tool call = snake, the next step after a tool = spiral) and it ends as a tick, or a cross when the reply failed. Pasted by the developer on 2026-10-01.

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

| thinking-orbs 0.3.2 (Libraries.dev, Jakub Antalik) | MIT | `licenses/ThinkingOrbs-MIT.txt`, vendored into `src/vendor/thinking-orbs/`. Used in, by what the orb shows (`src/lib/orbs.ts`, tested): the server (idle = the searching globe turning slowly at 15 frames a second, reading the prompt = listening, writing = composing at the pace of tok/s, queued or not answering = connecting, unloaded = shaping), a GPU card (idle = the same slow searching globe, working, working hard = weaving, held back by a limit = solving), an assistant message (waiting = breathing, thinking = solving, a tool running = connecting or searching, the next step after a tool = weaving, writing = composing), the empty chat while you type (listening), pages that wait for data (connecting, working), the timeline file reading, an MCP server starting. The status line's shimmer is the library web demo's `t-shimmer`, re-written in CSS. `haplollc/ThinkingOrbs` (a SwiftUI port of the same designs, MIT) was read for its state-to-design recipe and not used: a Swift package cannot run in a page. One local patch to the vendored library: an `fps` prop (frames are skipped, the loop still ticks) in `index.es.js` and `types.d.ts`, marked `xeno`; re-apply it if the library is updated. A change of design is a 450 ms dissolve (the old form swells and blurs away, the new one settles in) in `styles.css`. |
