# What each MIP mechanism was measured to do to the answer (2026-11-04, #298)

The developer asked for the data we already have on how each mechanism affects the answer. Sources: the stored arms in the scratchpad (`records.json`, read and
paired row by row by this session: the DeepMind lines), and the tracker comments of #279, #281, #287, #288, #296, #298 as read by a subagent (everything marked
[N/ID] is `https://github.com/xenodeve/Strata-xeno/issues/N#issuecomment-ID`; second-hand here, opened again before it is quoted outside this file).
Not read: #276 (cited through a memory note), the PRD's R12/R13 (second-hand through #281).

## Read this first: how far a difference can be trusted

- **The model alone differs from itself.** On this PC two greedy base runs on the same 160 DeepMind rows with identical recorded settings (`pc-base`, `pc4-base`)
  got 114 and 118 right; the text differs on 58 rows, the score on 10 (gained 3, lost 7); the scorer was not the cause (both score the same now).
  Anything under about 10 rows of 160 is not a finding. (`dm-base` is another configuration: no settings file, text equal to the other two on only 20 rows.)
- **Rows picked from the base's misses regress to the mean.** A sampled model gets some of its own misses right on a second try: GPQA gpl1 6/7 and gpl2 3/7 with
  no tool and no MIP [288/6593229141], against 3/7 with the mechanism on.
- **Rules written after reading the failures of the arm before, on the same 160 rows** (the hand-off arms) are tuned to those rows.
- **Scorers moved results** (list at the end). Arms before the scorer fix are scored again before they are compared.
- Machines are not compared: this PC does not repeat itself; a Colab A100 with every expert in VRAM did (49 of 49).

## Per mechanism

| # | mechanism | what was measured | isolated? |
|---|---|---|---|
| 1 | forced block + close after a print (Gyro-M unless stated) | 49 questions x2 rounds, paired, same session: 73 -> 77 of 98 right, tokens 0.508x, "no measured gain" (p 0.29) [279/6558540134]. LCB hard 20 rows, A100 paired: plain 17, block@512+close 14, +read 15, block@2048+close 13 - "every close lost right answers" [279/6567322075]. 4B, AIME25+HMMT+MATH long, 82 rows, T 0.6: 58 -> 56, tokens 0.93x, the model opened a block itself 2 times [296/6601741033]. HumanEval long: "the forced block does nothing" [279/6560191478] | yes (Gyro-M); the close is inside it |
| 2 | stop-checking note, repeat note, third-repeat close | no measurement of the note's words. Removing the close: tokens +43.7 %, right 40 -> 37 on 49 cases, but 48 of 49 pairs parted before the close [281/6558614962]. 3364 on the A100 lost to the third-repeat close "cutting off a hypothesis being tested" [279/6568582984]. The same-exe `--igc-close-repeat 3` against `0` was queued; no result found | no |
| 3 | hint at a tedious calculation (`--hint-calc`) and its trust line | 4B, 10 paired rows 3 -> 5 [296/6605187999] and 15 paired 4 -> 5 (gained 2, lost 1) [296/6606176648], both declared not counted: the model's blocks never ran, outputs were made up. Flash-Next: a prompt hint "does not work on this model", 0 blocks [279/6555605141]. The trust line alone was never isolated; after it the 4B "still re-checks" on one answer [296/6604411865]. Our H-trust arms: a line that vouches for a value ("exact and final", " (exact)") raised the re-check word by 22-35 points against a bare new line [298/6629560608] | no |
| 4 | inline value at `=` | 4B thinking, 10 paired rows: 5 -> 2, stuck 5 -> 7, "did not help, dropped" [296/6603813402]. 4B no-think DeepMind: `--inline-calc` 90 -> 80 (gained 5, lost 15), `--inline-plain` 85 -> 81, "worse than base, so it is cut" [296/6616350128]. Gyro-M inside a bundle: LCB 17 -> 18 [279/6568582984]; GSM-Hard 177 against 178, tokens -22 % (memory note, #276 not read) | alone: no gain |
| 5 | check-hand correction | DeepMind 160, paired: base 121 -> 116 (tokens 0.99x), "mostly right" corrections but the model redid the work by hand [296/6617628007]. On top of whole: 138 against 138, fired once [296/6618889490]. On reasoning misses: corrected 4 of 14 wrong hand values [296/6620529703] | alone: no gain |
| 6 | loop hint / loop close | 3 paired rows 1 -> 2, 0.37x [296/6606176648]; loops given a program 3 of 5 against 0 of 3 (pc5-full, stopped by a box bug and the 30K cap) [296/6620529703]; Flash-Next only inside a bundle | no |
| 7 | whole-expression hint | DeepMind 160 paired: `--whole-hint` 121 -> 127 at 0.41x tokens (L4) [296/6617628007]; `--whole-plain` 118 -> 138 at 0.11x (5060 Ti): add_sub 33 -> 37, div 29 -> 27, mul_big 21 -> 40, mul_div 35 -> 34 [296/6618864288]. This session, same rows: gained 32 lost 12 against pc4-base | yes (with inline: the value is what it computes) |
| 8 | hand-off | same 160 rows, greedy, no think: 138 (whole-plain) -> pc6 146 -> pc7 143 (reverted) -> pc8 151 -> pc9 156 with the fraction close and the question's value; tokens 0.11x -> 0.05x [298/6623524689] ... [298/6624038999]. This session: pc9 gained 39, lost 1 against pc4-base | each step is a bundle; tuned on these rows |
| 9 | native tool format / the tool route | 4B `--native-tool`: 9 paired 4 -> 4, no `<tool_call>` in 9 answers [296/6603918364]; with `--system`: 4 -> 3, calls on 2 of 8 rows, both right [296/6604411865]. Gyro-M tool route: GSM8K 60 -> 60 with 0 calls, MATH L5 57 -> 59 (neither gained row called the tool), HLE 4 -> 5 [288/6578681777], [288/6580280815]. Sampled ON against OFF: MATH long 21 -> 21, GPQA 12 -> 11, SuperGPQA 11 -> 10, HLE 4 -> 4: "no right answers gained, fewer tokens on three of the four sets, and no call ever made inside the thinking" [288/6594660367] ... [288/6600685063] | ON/OFF pairs bundle MIP + tool + sandbox |
| 10 | fine-tuning for block use | no result posted (the FT arms were running when account a hit 0 CU) [296/6606176648] | - |
| 11 | the hypothesis loop | no row saw a real mismatch and then a pass [288/6571077073]; of 9 wrong -> right rows over 15 paired arms the call decided 1 [288/6591858637] | no |
| - | residual-stream write of a digit | 4B layer 31: fix 6 of 13, control 0 of 167; Qwen3.6: only 2 wrong rows [298/6625995583] | yes |
| - | forcing tokens through the sampler | the same greedy text as inserting them in 25 of 30 points, 25 of 30 for the same insertion at two batch sizes [298/6628946779] | yes |
| - | taking the re-check starters out of the next 60 steps | re-check word within 300 chars lower by 9 and 11 points (30 -> 21 %, 51 -> 40 %), inside the noise (+-15) [298/6629560608] | yes |

## What the data says

- The one mechanism family with a gain beyond the noise is **the whole-expression hint with the software's value (7), then the hand-off rules (8)**, on **no-think
  arithmetic** (DeepMind), at 0.05-0.11x tokens. It does not reach LiveBench (the question reader read 0 of 568 questions [298/6628267003]).
- On **thinking** (the case for LiveBench, AIME) no mechanism was shown to raise the number right: inline 5 -> 2, native tool 4 -> 4, forced block 58 -> 56 and the
  A100 pairs below, hint-calc 5 against 3-4 on blocks that never ran. What moved is **tokens** (0.5-0.8x on easy rows).
- Never isolated: the trust line, loop hint/close, the repeat note, fine-tuning. Run only inside bundles: 2, 4 on Flash-Next, 6, 8 (pc9), 9, 11.

## Contradictions the tracker holds

- Mechanism 1: 77 against 73 of 98 (math, 0.508x) against 14 against 17 of 20 (LCB, A100, paired).
- Mechanism 3: 10 paired rows 5 against 3, then the same arm on 15 rows 5 against 4 with 2 gained and 1 lost.
- DeepMind base: 121 (L4) against 118 (5060 Ti); this session's repeats on the PC: 114 and 118.
- HLE, tool route: 4 -> 5 by rule and reader; sampled ON 5 of 10 by rule, 4 of 10 by the reader.

## Results later found wrong (what moved)

- [279] the C1 table compared old long runs: real plain 3/3 at 1,511-2,558 tokens. A reply cap of 1024 turned MMLU-Pro 63.3 % into 93.3 % and SuperGPQA 23.3 % into 68.3 % [279/6564023383]. The interpreter refused classes: 0 of 16 LCB programs ran, 15 of 16 after the fix [279/6564180518]. The close came in the same delivery as the output [279/6564754473]. Reader re-score: MIP values 39 -> 37 [279/6569648606].
- [288] a hint bug lost 4 rows (17 -> 21 after the fix) [288/6584335557]; the HLE rule false positive moved ON 5/10 to 4/10 [288/6600685063]; a bench bug cost one mph2 row [288/6584975377].
- [296] the DeepMind scorer fix moved 6 rows wrong -> right; a scorer that accepts the same number in another form moved r1-ctrl 20 -> 21/36; the box bug failed 20 of 45 blocks in 0.0 s and the 30K cap ended 6 of 10 and 6 of 7 at length (pc5 stopped).
- [298] digit-confidence AUROC 0.984 -> 0.899 (one trace held 177 of 201 wrong values) -> 0.885 (10 traces) -> 0.705 / 0.668 (46 traces). This session: lb1 scored AMPS_Hard in an environment without lark, so every radical-form answer was wrong: base 18 -> 22 of 48 when scored again.
