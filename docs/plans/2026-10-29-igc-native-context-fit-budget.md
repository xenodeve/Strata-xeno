# Native-context fit: measured time-budget correction

The first native-feature fit is stopped as incomplete, never promoted to a
quality result. At35 epochs it reports689.1047s; intervals5..35 remain about
19.3-19.6s/epoch. The declared70 epochs therefore cannot fit the1,200s inner
budget (or1,260s outer including cache hashing). Its log/source/protocol identity
and controlled-stop process evidence are retained; no partial checkpoint is used.

Retry from the same initial seed20261031 for all70 epochs, using the same exact
context F32 cache, expanded corpus,96,964-parameter architecture, shuffle,
batch64, optimizer and global cosine schedule. Only the explicit offline fit
timeout becomes1,800s, with1,980s outer allowance for hashing/evaluation/cleanup.
This is a time-bound correction, not a smaller workload, model/configuration
change, serving limit change or benchmark optimization.

Use `--activation-cache context --fit-timeout 1800`. Historical defaults stay
600s for embedding and1,200s for context when the option is absent. Reject invalid
budgets, booleans and explicit overrides on the historical embedding path.
Source hashes and the explicit budget are recorded. Compare available printed
loss checkpoints to the interrupted run, but judge only final weights and the
unchanged development rubric. Fresh acceptance64 remains untouched.

Cache preparation stays bound to the immutable
`2026-10-29-igc-native-context-input-experiment.md`; pass it through
`--cache-protocol`, and use this document for `--protocol`. The fitting recipe in
the cache protocol remains normative except this enumerated timeout override.
