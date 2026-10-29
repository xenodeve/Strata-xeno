# I2: balanced native-embedding development

The small pilots fit training data but still fail exact operand extraction on
known diagnostics. Query removal alone failed too. This next experiment tests
whether substantially broader, explicitly factored supervision can train the
same query-free readout. It does not declare data scarcity the sole root cause
or silently turn the pilot into a reproduction of the IGC paper.

## Frozen data and native input

`balanced_corpus.dataset()` freezes seed 20261030 and 4,480 rows:
4,096 fit rows (1,024 per no-call/add/subtract/multiply class) and three distinct
128-row development axes. Canonical row SHA256:
`193f34b29fef6e884a4074a2e20db858f4467e2cc138365c76995d39ed8edf62`.

All values have signed magnitude at most 99. Absolute values
`3,8,17,29,43,61,83,97` are withheld from fitting. All decimal digit classes in
every target slot nevertheless occur in fitting. The axes are seen values/new
templates, new values/seen templates, and new values/new templates. A new-value
sample has at least one withheld operand. Development pairs are disjoint across
axes and from fitting. The original pilot and untouched evaluation64 operand
pairs are excluded from every new row. English/Thai templates, sign, length and
three expression styles are explicit; expression style is independently sampled.

Use the selected deployed tokenizer and derive the constant prefix/suffix from
the completed real 64-request cohort. Verify that the same wrapper and tokenizer
recreate every collected prompt token ID exactly. Preserve the answer-anchor
exclusion and last-80-position window. Refuse preparation if an operand falls
outside that window. No full-model state or final mixed query is fabricated.

For this embedding-only seam, the native lookup is independent of prior tokens.
Prepare a bounded cache of only the required native Q8_0 embedding rows with the
existing task-owned GPU control. Verify all four HC streams against independent
FP16-scale/int8 block decoding. This is a frozen activation cache, not a learned
token-ID shortcut: only those actual F32 activation vectors enter the input head.
Stored row indices select the frozen cache for minibatches; they are not model
features. This preparation does not execute the language-model body and is not a
claim about later contextual activation seams.

## Frozen learning recipe

Keep the 63,492-parameter query-free `InputReadout` architecture, 80 learned
relative positions, masks, per-stream LayerNorm/projection, key/value attention,
nine queries and categorical sign/length/little-endian digit heads. No raw text,
token IDs, oracle operands or arithmetic answers enter the neural network.
Supervision contains only the declared ready/operator/operand labels.

Fit CPU Torch with four threads, deterministic operations, seed 20261030, AdamW
lr 0.001 and weight decay 0.01. Use 30 epochs, batch 64, deterministic seeded
per-epoch permutations and a 600-second training/660-second outer bound. Persist
the final epoch only, not a development-selected checkpoint. Log fit loss every
five epochs. No hyperparameter or threshold changes after seeing predictions.
Do not backpropagate through or update Gyro weights; no output mapper is trained.

## Rubric and stop

Evaluate training and each development axis only after the frozen fit. Report
readiness/operator, canonical exact tuples and operands, nonpadding digits,
sign/length, swaps/reversals, invalid encodings and false/missed calls separately.
Correct arithmetic on wrong operands is not an accepted invocation. No repair,
normalization or host parser may convert a wrong neural prediction into success.

Fresh evaluation eligibility requires all 3,072 training positives exact, all
96 positives on each development axis exact and no false calls on any axis.
If it fails, retain the fresh evaluation64 untouched and preserve the failed
weights/results. A failed run does not authorize extending its epochs or changing
the frozen data. A further experiment needs its own declared protocol.

If eligible, lock the weights before the original fresh evaluation64. Its finite
two-digit acceptance is all 48 positive tuples exact and no false calls on
16 negatives. This is neither a 62-digit gate nor an output-integration or
daily-use claim. No model downloads, new third-party execution, served changes,
deployment or default activation are part of this experiment.

After input quality, continue output mapping and committed-state integration.
Once the integrated path is usable, its immediate next step remains actual Hub
end-to-end quality, latency, token/context and resource benchmarking.
