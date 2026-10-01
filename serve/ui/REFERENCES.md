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

## Reference only, own implementation (no code copied)

Paid or unclear-licence work that informed the design. Recorded here as the handoff requires; add each one the
moment it is looked at.

(none yet)

## Build

`cd serve/ui && bun install && bun run build` writes `dist/` and `dist/source-hash.txt`. `dist/` is committed so the
runtime needs no Node; `python -m unittest serve.test_ui` fails when `src/` changed and `dist/` did not.
