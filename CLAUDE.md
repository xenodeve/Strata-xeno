# CLAUDE.md

The rules for every coding agent on this repository live in `AGENTS.md`, so Claude Code and Codex follow one file.

**The goal is the lowest latency in the system and the software** (developer, 2026-10-04): rank work by the time it takes off a
real Claude Code turn on the served D2x profile, fix root causes rather than work around them, and name the issue that removes any
latency a change adds. In full: `AGENTS.md` → "The goal".

The nine rules agents miss most, all in full in `AGENTS.md`:

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
7. **Test with the smallest run that answers the question.** Skip suites the model nearly aces. Run a small first pass, then repeat only
   the misses the feature can reach; when that satisfies, check the trade-off on a sample of right answers, fix, and go round again. No whole suites unless a
   whole-suite number is what is asked for. Run tracks in parallel and never stop to wait for one. When a scorer was wrong, fix it and
   score the stored replies again before running anything twice.
8. **Time spent is the cost.** No machine sits idle unnoticed, a job is checked a minute after it starts, a scorer is gated on real
   replies, a run stops the moment its question has an answer, and what repeats is scripted.
9. **Idle means scrutinize the tests and the mechanism.** When both tracks wait on a run, run `/scrutinize` on the test procedure and the
   mechanism under work. Most results of 2026-11-02 came from bugs in the system, so a number is not evidence until its code has been read.

## CLAUDE.md and AGENTS.md mirror each other (developer, 2026-11-02)

A rule added to one goes into the other in the same commit, word for word, so no agent reads less than another.

## Test with the smallest run that answers the question (developer, 2026-11-02)

"Testing must be as efficient and as short as it can be; a whole suite is not needed." A test exists to answer one question, so it stops
as soon as that question has an answer.

- **Do not pick a suite the model nearly aces.** Read its published score first. A high score means easy questions the model already gets,
  and they cost the same and show nothing.
- **First pass, then only the misses.** Run the smallest first pass that shows a result, aimed at what the change can reach. Then repeat
  only what it missed, with `tools/igc/select_rows.py`.
- **When the result on the misses satisfies, check the trade-off** on questions that were right, by a sample. If there is one, fix it and
  go round the same loop. A gain on misses alone is not a claim: failures can only go up.
- **Drop misses the feature cannot reach.** A low score does not make every miss useful: a wrong fact or a misread question has nothing for
  MIP to compute. Test only what the feature under work can change, and say how many were left out and by what rule.
- **A wrong scorer does not cost the run.** Bench records keep the raw reply, so fix the scorer, then score the same records again, by the
  new scorer or by hand, and use them. Run again only when the text itself cannot answer. Say which numbers changed.
- **A whole suite runs only when a whole-suite number is the thing asked for.**
- **Run tracks in parallel and never stop to wait for one.** While one track runs, advance the other: prepare the next queue, write the
  reader, read partial results, fix what they show. An idle machine is lost time.
- **Rule A/Bs run on a machine that repeats itself.** A Colab A100 with every expert in VRAM gave identical text on 49 of 49 questions, so
  one round is enough and a difference belongs to the rule. This PC gives 1 of 49: paths part before the rule. Seconds are not compared
  across machines, and time claims stay on this PC.

## Time spent is the cost (developer, 2026-11-02)

"The time we lose at each point is a cost we really pay." Each of these cost hours or credits once, so each is now a rule:

- **No machine sits idle unnoticed.** One queue with one worker on a remote machine, never scripts chained to wait on each other (one waited
  on itself for 10 minutes). The watcher reports the moment the queue runs dry, and both machines always have the next job.
- **Check a job started,** a minute after starting it, not when its result is due.
- **Gate a scorer on real replies first.** After the first rows of a pass, read how many replies gave no answer and what the misses look
  like. A score far from any published figure is an alarm, not a result.
- **Stop a run the moment its question has an answer,** and stop a running arm whose result no longer matters. Keep what it wrote.
- **Measure a machine's speed on one question before giving it a queue.** Two sessions once ran an hour for numbers nothing used.
- **Script what repeats:** retries, uploads, scoring, setup. Hand-typed steps fail differently each time.
- **Write code with backslashes through the file tool,** not a shell heredoc: three commits broke on lost escapes.

## Idle means scrutinize the tests and the mechanism (developer, 2026-11-02)

"Through this whole session most results came from bugs in the system, not from a test or a score we could use." A number is not evidence
until the code that produced it has been read, so time spent waiting goes here first.

- **When both tracks wait on a run, run `/scrutinize`,** on the test procedure and on the mechanism under work: row builders, selection, reply
  caps, scorers, readers, runners, queues, and the feature's own path. Fix the bug, and improve the procedure that let it through.
- **Ask what else could have produced a number** before reading it as a result. Read one cut reply, one refusal and one miss first.
- **Prove the path ran.** Work is not done until a run's log or output shows it firing. A test that stops at a refusal has tested nothing.
- **Try to break it.** Run adversarial inputs against a reference, and compare against a control that needs none of the feature.
- **A fix lands with the check that would have caught it,** and stored replies are scored again where they can answer.

Each of these looked like a result on 2026-11-02: a reply cap turned 93.3 % into 63.3 % and 68.3 % into 23.3 %; a reader that needed one phrase
reported 45 %; a bench check missed a cycle by 5 tokens and ran 31 minutes to the end of the context; MIP in slots was reported done while a
boot refusal kept it from running, and then the bench did not read the slots' lines; the sandbox refused every `class Solution`; the close
came in the same delivery as the output, so the model never read it.

@AGENTS.md
