# tools/igc/probe: how much of a model's computing can MIP see, and what a guideline does to it (#298)

Research scripts, run on a Colab A100 (they need a GPU and the model weights); each one writes what it measured and prints the table the
tracker comment quotes. Inputs are stored traces (tir4b `--ids` files, or a record plus its question). Nothing here is part of the engine.

| script | question | numbers it prints |
|---|---|---|
| `detect_cov2.py` | for three kinds of event (a computation starts, a re-check starts, a loop starts) how many does each signal see before the model writes it? also wrong values and the end of the prompt | per kind and window k: AUROC, onsets caught out of all onsets at a fixed alarm rate, median lead, precision |
| `h_trust.py` | does the model re-check a value software gave it more than one it wrote? | re-check rate by who wrote it and by the model's own probability of the value |
| `htab.py` | the same, as an intervention at one point: software's value inserted against the model writing its own; also forcing the tokens through a logits processor against inserting them | re-check and recompute rates per arm; greedy agreement of the two ways to force |
| `htab2.py` | two guideline arms at those points: the re-check starters taken out for 60 steps; a short guideline sentence after the value | the same table, with A and B read back |
| `htab3.py` | the controls for `htab2.py`: the value then only a newline (N), and the value then ` (exact)` and a newline (G) | the same table, with A, B and E read back |
| `numnoise.py` | control for `htab.py` part 1: the same insertion at two batch sizes | how often bf16 noise alone parts the greedy text |

`detect_cov2.py` has three modes: cross-validated by trace (the default); `--train-traces control.jsonl --traces arm/ids.jsonl` fits every detector once on the control traces and reads each arm
(the folder of its ids.jsonl) with them, so every arm gets every detector; and `--check`, which tests the detectors against the real decode before any result is read: the state at
token t predicts token t+1 (93.5 % on the 4B, 91.1 % on Qwen3.6, 0.7 % if shifted), the start-token masses sit where the next token is a starter, event token indexes point at the text,
what the extractors miss (a pure-numeric `=` line with no event: 2 of 454 on the control rows), a detector fitted on labels shifted by 2-4 tokens does worse on the true labels
than one fitted on the true ones, and shuffled labels give 0.5. Events inside the software's own spans (tir4b `--ids`) are not the model's and are dropped.

`detect_cov2.py --dry` runs the whole pipeline on random features on a CPU (no model), to test the code.
Ground truth comes from the text alone with `tools/igc/mathexpr.py`; the loop rule (ten lines in a row that already appeared) was set against
the runner's own `stuck` label on the 36 control rows. The scripts on a VM expect `mathexpr.py` next to them in `/content`.
