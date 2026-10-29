# I2: remove the measured final-query shortcut

This is a new declared development arm after the three-seam protocol completed.
Do not edit or rerun its frozen arms and do not relabel their negative result.
All three got 0/12 exact known-diagnostic positive tuples. The fresh evaluation64
was not consumed. This 48-fit pilot is not a reproduction or refutation of IGC's
10,000-sample experiment, nor evidence that final states contain no operands.

## Evidence and falsifiable hypothesis

The embedding arm fits 36/36 training positive tuples. Rotating observed embedding
states while retaining their original positional features still gives 36/36;
zeroing all states gives 21/36. Zeroing or rotating only the standardized final
query gives 0/36. Complete state/position pair rotation is an invariant control,
with maximum logit difference 2.862e-6. Therefore the final query is necessary
for this trained readout's perfect fit; ordered embedding copying has not been
demonstrated. This does not prove that removing it alone fixes generalization.

The hypothesis is that removing this input route forces this small readout to
use the observed token states and makes exact operand extraction possible on at
least part of the known diagnostic set. Disproof is failure to fit positives or
continued zero diagnostic tuples. A failed arm does not justify more epochs or
an unreported architecture change.

## Frozen arm

Use the already completed `igc-i2-early-fit-01` cohort, embedding stage, same
native executable/source/corpus identities as the three-seam experiment. Keep
all 48 fit examples, sixteen known diagnostics, masks, 80 learned positional
features, input projection, key/value and categorical output heads unchanged.
Replace only final-vector query conditioning with nine learned queries. Remove
the unused context projection from the model; no raw token IDs, text, oracle
operands or answers enter the input head. Keep the existing canonical little-
endian sign/length/digit target convention. Do not silently reverse subtraction.

Keep seed20261029, AdamW lr0.001/weight decay0.01, 300 full-batch epochs, four
CPU threads, deterministic operations and 180s fit/210s outer bounds. Persist
the state, architecture/query mode, all predictions and protocol hash. The final-
query arm already recorded is the control; no duplicate overfit run is needed.

Score readiness/operator, sign/length, nonpadding digits, each operand, exact
tuple, false/missed calls and invalid/noncanonical predictions separately. Audit
coverage and window/anchor identities from saved data. The query-free structural
test must show that changing the final query cannot change output.

Eligibility remains all36 training positives exact and at least one known-
diagnostic exact tuple. Otherwise keep fresh evaluation64 untouched and close
this arm as failed. If eligible, freeze this one arm before its one evaluation;
the existing all48 positive tuples/no false calls on16 negative feasibility gate
still applies. No serving integration, output mapper, 62-digit coverage, benefit
or default claim follows merely from fitting this arm.
