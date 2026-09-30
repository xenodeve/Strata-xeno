# CLAUDE.md

The rules for every coding agent on this repository live in `AGENTS.md`, so Claude Code and Codex follow one file.

The six rules agents miss most, all in full in `AGENTS.md`:

1. **Run everything in the background,** with no window on the developer's display.
2. **Update the issue as soon as a piece of work is done,** and open a new issue the moment a problem (perf, bug,
   instrument fault) is found, so knowledge and history do not disappear with the context.
3. **Do not hunt for where the time goes one point at a time.** Record one run with `STRATA_TIMELINE=<file>` and read
   it with `python tests/xeno/perf/timeline.py <file>`. It shows every thread and GPU lane on one clock, and why each
   one waited. A stage it does not show gets a timeline span, not an ad-hoc timer.
4. **Do not guess a fix.** A fix proposed from a guess wastes the build and the A/B when the guess is wrong. Measure
   the exact mechanism first, splitting any average by its cause, and take the fix's ceiling from those spans. Design
   the fix from that number. Three guesses of 2026-09-30 were overturned this way (#44).
5. **Keep `docs/BLUEPRINT.md` current.** Read it before a structural change, and update it in the same commit when a
   change touches a process, the request path, the engine protocol, expert placement, the checkpoints, a flag / config
   key / environment variable, a diagnostic, or the served engine's branch. Work not yet in its baseline goes into
   its "Not yet in the baseline" table. `tools/hooks/commit-msg` enforces the part a diff shows (flags, GEN keys,
   `STRATA_*` variables, config keys, new source files); the way out is a `Blueprint: n/a - <why>` trailer.
6. **Every upstream merge ends with a merge report.** Once the merge is finished and verified, write
   `docs/reports/<date>-merge-upstream-<version>.md` as its blueprint and checkpoint: what moved, each conflict and
   the side kept, the hazards checked, the bugs the merge introduced, the verification with numbers, and what is left.
   Link it from the tracking issue and the blueprint's revision log. A merge without it is not finished.

@AGENTS.md
