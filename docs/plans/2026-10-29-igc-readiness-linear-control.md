# Readiness capacity control after the frozen MLP diagnostic

The300-epoch final-state MLP correctly classifies111/112 known development rows
and all208 fit rows. There are zero false calls on64 development negatives; one
positive `(19) - (19)` is missed. No threshold or historical labels change.
This is insufficient to establish the cause of the missed call. Test a simpler
linear readout as a capacity control, not a claimed fix or a new release corpus.

Use the identical320 captured vectors, train-only normalization, seed20261107,
300 full-batch AdamW steps atlr0.001/weight_decay0.01, four CPU threads and final
checkpoint rubric. Only replace2560→128→2 GELU with2560→2 (5,122 parameters).
No development fitting, threshold selection, extra training samples, checkpoint
selection or language-model gradients. The original capture protocol and MLP
source/results stay pinned; this protocol binds only the new fit. Full development
success remains required for this diagnostic. Fresh acceptance64 remains unused.
