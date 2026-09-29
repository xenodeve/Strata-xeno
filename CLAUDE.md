# CLAUDE.md

The rules for every coding agent on this repository live in `AGENTS.md`, so Claude Code and Codex follow one file.

The four rules agents miss most, all in full in `AGENTS.md`:

1. **Run everything in the background,** with no window on the developer's display.
2. **Update the issue as soon as a piece of work is done,** and open a new issue the moment a problem (perf, bug,
   instrument fault) is found, so knowledge and history do not disappear with the context.
3. **Do not hunt for where the time goes one point at a time.** Record one run with `STRATA_TIMELINE=<file>` and read
   it with `python tests/xeno/perf/timeline.py <file>`. It shows every thread and GPU lane on one clock, and why each
   one waited. A stage it does not show gets a timeline span, not an ad-hoc timer.
4. **Do not guess a fix.** A fix proposed from a guess wastes the build and the A/B when the guess is wrong. Measure
   the exact mechanism first, splitting any average by its cause, and take the fix's ceiling from those spans. Design
   the fix from that number. Three guesses of 2026-09-30 were overturned this way (#44).

@AGENTS.md
