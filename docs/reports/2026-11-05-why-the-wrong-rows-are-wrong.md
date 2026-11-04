# Why the 19 rows that every run got wrong are wrong (2026-11-05, #298)

The developer asked: take the wrong rows and read why. Does the model not believe our value, is there reasoning our software does not cover, or did the model set up a wrong hypothesis and never fix it?

## What was read, and what was not

- **Rows**: the 19 of the 48 `lb1-rows` questions that were wrong in all five A100 runs (`ctrlb`, `ctrl`, `inline`, `logit8`, `inlinesup`; Qwen3.5-4B bf16, thinking, T 0.6, cap 20,000 tokens). 17 of the 19 were cut at the cap in every run, 2 ended in a loop in the `inline` run. Over the five runs, 20 to 24 of the wrong replies per run were cut at the cap (`ctrlb` 22 of 23 wrong, `inline` 20 of 27, with 7 more in a loop), so **a wrong row here is mostly a row that never gave an answer**.
- **What an auditor saw**, per row and for the two arms `ctrlb` (the model alone) and `inline` (our software puts the exact value in after an `=`): the question, the first 2,500 characters of the reply, up to 6 windows of +-500 characters around mentions of the gold answer (the first 3 and last 3), 3 windows around re-check words, and the last 5,000 characters (`scratchpad/lb2/extract/<row>__<arm>.txt`, cut by `mkextract.py`). **The middle of every reply, from about 5 % to 80 %, was not read by anyone.**
- **Who read**: five general-purpose subagents, 3 or 4 rows each, with a fixed rubric (below). They checked the mathematics themselves (several brute-forced the problem). This session spot-checked about 20 of their quotes against the files: all found (one differs only in LaTeX spacing). The labels are judgements and were not re-derived.
- The gold-mention counts that were given to the auditors are not reliable alone: in `c3cc0058` the gold 315 is an angle sum, in `de81e6c1` the 8 mentions of 243 are an unrelated intermediate.

Rubric: NOT_TRUSTED (a correct value, from the software or the model, was in the reply, and the model kept re-checking and never committed), UNCOVERED (the stuck part is reasoning no arithmetic expression covers), WRONG_HYPOTHESIS_KEPT (a wrong set-up carried on), TOKENS_ONLY (a sound path unfinished at the cap), OTHER.

## The 19 rows

Primary label per arm; the position is where in the reply the gold result first appeared as a result the model had reached.

| row | question kind | `ctrlb` | `inline` (software) | gold reached |
|---|---|---|---|---|
| b162bd2e | 4x4 determinant | NOT_TRUSTED | NOT_TRUSTED | both, inline at 22 % (software had put -320 in) |
| e6d6ebe2 | grid colouring count | TOKENS_ONLY | NOT_TRUSTED | inline only, 54 % ("So the answer should be 902"), then 28,000 characters of small-case checks |
| 41e8ed05 | geometry, min of BD^2 | NOT_TRUSTED | NOT_TRUSTED | both (480; software put 120 and 480 in) |
| afac20e6 | counting sequences | NOT_TRUSTED | NOT_TRUSTED | both (62 many times; `ctrlb` then builds a wrong check against it) |
| 28a85ac4 | angle chasing | UNCOVERED | NOT_TRUSTED | inline only, 86 % (336) |
| 4d9e52e5 | quartic factorisation | OTHER (slips) + WRONG_HYPOTHESIS | NOT_TRUSTED | inline only, 50 % (62) |
| 7068e4c9 | perimeter, 4 parts | WRONG_HYPOTHESIS (late fix) + NOT_TRUSTED | NOT_TRUSTED | both; `ctrlb` at 94 %, inline at 45 % |
| 72e8ae47 | residue counting | TOKENS_ONLY + NOT_TRUSTED | NOT_TRUSTED + UNCOVERED | both; `ctrlb` at 94 %, inline at 80 % |
| add52d77 | integer count by a turning condition | NOT_TRUSTED | NOT_TRUSTED + WRONG_HYPOTHESIS | both (503); inline wrote "= 168 + 84 = 252" after the software had put 168 |
| 652451e2 | intersections of two curves | UNCOVERED + TOKENS_ONLY | UNCOVERED + WRONG_HYPOTHESIS | no; inline wrote y(0.125) = 1 where the software had put 0.172 |
| c3cc0058 | chords / rectangles in a 12-gon | UNCOVERED + WRONG_HYPOTHESIS | UNCOVERED + WRONG_HYPOTHESIS | no (the 315 is an angle sum) |
| 75a0f5a3 | area of reflected polygons | NOT_TRUSTED (13/18 three times) | UNCOVERED (geometry) | `ctrlb` only |
| 25af3417 | tied minima of a quartic | WRONG_HYPOTHESIS + UNCOVERED | UNCOVERED + WRONG_HYPOTHESIS | no (real algebra errors kept) |
| 4e97db9c | chord-crossing probability | UNCOVERED + TOKENS_ONLY | UNCOVERED + TOKENS_ONLY | no |
| e940587c | pairings count | TOKENS_ONLY | OTHER (a loop, "stuck") + UNCOVERED | no |
| ab7dcc84 | counting by cases (parity DP) | UNCOVERED + TOKENS_ONLY | UNCOVERED + TOKENS_ONLY | no |
| 459ffbe4 | probability, rolls and stickers | TOKENS_ONLY + UNCOVERED | OTHER (a loop on a restated sentence, "stuck" at 7,852 tokens) | no; the software put in no value |
| 75a4b681 | extremal graph orientation | TOKENS_ONLY + UNCOVERED | OTHER (a 3x3 detour) + TOKENS_ONLY | `ctrlb` reached 3896 at 94 %, the mod-1000 step still open |
| de81e6c1 | lattice paths | WRONG_HYPOTHESIS | WRONG_HYPOTHESIS | no (the end of the path is taken as (10,1) only; N = 19683 + 39366 = 243^2) |

## Count, `inline` arm (the software acting), 19 rows

| primary | rows | which |
|---|---:|---|
| NOT_TRUSTED (the right result was in the reply and the model kept re-checking) | 9 | b162bd2e, e6d6ebe2, 41e8ed05, afac20e6, 28a85ac4, 4d9e52e5, 7068e4c9, 72e8ae47, add52d77 |
| UNCOVERED (counting, geometry, symbolic algebra, no `=` to compute) | 8 | 652451e2, c3cc0058, 75a0f5a3, 25af3417, 4e97db9c, ab7dcc84, 459ffbe4, e940587c |
| WRONG_HYPOTHESIS_KEPT | 1 | de81e6c1 (and as a second label in c3cc0058, 25af3417, 652451e2, add52d77) |
| OTHER / TOKENS_ONLY | 1 | 75a4b681 |

In `ctrlb`, the model alone, the gold result was reached in 8 of the 19 (b162bd2e, 41e8ed05, afac20e6, 7068e4c9, 72e8ae47, add52d77, 75a0f5a3, 75a4b681's 3896), more of them late (at 94 % in three).

## What it says

1. **The software's values were correct in every list the auditors could see, and in no case did the model contradict one by saying it was wrong.** The doubt in the NOT_TRUSTED rows is about the structure ("did I miss a constraint?", "is the factor 3 right?", "re-read the question"), not about a number. No value from an `=` can settle that, so this is not what a value hand-off fixes.
2. **9 of 19 rows had the right result in the reply and no commit.** This is a ceiling of 9 rows of 48 (19 points) for a rule that commits when the result is reached; it is the ceiling only: the software does not know the gold, and a rule that closes early would also close right-looking wrong results (earlier closes lost right answers on LiveCodeBench in #279).
3. **8 of 19 are reasoning the software does not cover**: counting by cases, geometry, symbolic algebra. Here the values it puts in are trivial sums that are not on the critical path.
4. **Three instances of the model not using a value the software gave**: 652451e2 (0.172 against the model's 1), add52d77 (168 followed by "+ 84 = 252"), and a possible one in 28a85ac4 ("= 336 + 48 + 216 = 600?"). Few, and each is a different slip.
5. **One loop the software did not cause or catch**: 459ffbe4 `inline` repeated a six-line block verbatim to the end; the loop-close mechanisms were not on in this arm.
6. The replies are dense (about 2.2 characters per token), so a correct derivation is often still in progress at 20,000 tokens: 72e8ae47 and 75a4b681 reached the right result at 94 %.

## Not shown

- Whether the right result in the NOT_TRUSTED rows was reached before or after the model had already begun to doubt it: the middle 5-80 % of the replies was not read.
- Whether the same split holds on the 29 rows that were not wrong in every run, or in the 16 that were wrong in some runs only.
- That committing at first sight would have helped: not tested. A test would take the NOT_TRUSTED rows' first "answer is X" position, close there, and measure on a sample of right rows what it costs.
