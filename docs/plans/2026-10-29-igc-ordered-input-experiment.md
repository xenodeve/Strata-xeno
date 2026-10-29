# I2 separate left-aligned digit-view diagnostic

The frozen contextual checkpoint exposes a concrete failure mechanism: two-digit
operands are all correct in the inspected development splits, but single-digit
ones/padding generalization fails. Unused tens are already fully supervised; no
mask fix is justified. Keep that checkpoint/predictions/ABI validation unchanged.

Test a separately named neural representation, not a silent repair: most
significant digit first for either operand length, second digit or explicit PAD10.
Keep operator/sign/length heads and the contextual query-free readout body. Enlarge
only the two second-digit heads from10 to11 classes (+130 parameters). Common
parameters and their initial values remain the legacy initialization. All digit
heads remain supervised for all positive rows, including PAD. The canonical
executor ABI remains little-endian decimal with unused bytes0. Convert only a
fully canonical neural tuple; missing PAD, leading zero and negative zero fail.
The old diagnostic artifact schema does not accept this new head shape.

Use the existing qualified expanded native embedding cache first, with the exact
10,240 fit and384 already-known development examples. This isolates the view
change against the saved expanded contextual embedding control, without another
23-minute first-attention fit or a new model capture. Seed20261031, same 70 epochs,
batch64, AdamW0.001/0.01, global cosine to0.00001, deterministic F32/math SDP,
TF32 off, four CPU threads, learned queries only. Final weights only. Deadline600s,
outer720s. Pin cache, old preparation protocol, this protocol, training source and
state. Do not evaluate acceptance64.

Report each existing split separately: routing/false calls, exact operands and
tuple, active digits versus PAD consistency, canonical rejection count. Positive
numeric gains cannot close I2 while unseen negative intent triggers persist.
This is one mechanism test, not a broad architecture win or a deployment. The
embedding comparison is not the same feature arm as the final first-attention
checkpoint; do not causally compare them as if only digit view changed.
