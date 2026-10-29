# Bounded early activations from the actual batched prompt path

Issue #276, live-controller prerequisite. The actual Hub uses automatic batched and
wave prefill. Do not disable it or manufacture early features from final R/KV.

Survey and change inventory:

- `core/igc_prefill_snapshot.hpp/.cpp`: opt-in per-lane tail snapshot, two 80-row
  F32 buffers. Save after embedding broadcast and layer-zero attention HC write,
  emit only from the ordered chunk tail. No arithmetic or tensor readback.
- `tests/igc/prefill_snapshot_test.cpp` and native CMake: tail bounds, mixed-chunk
  refusal, consumer failure, repeated emission and two-stream/context ownership.
- `prefill/prefill.hpp`, `src/prefill/prefill.cpp`: diagnostic-only consumer and
  startup allocation/release, actual D2D capture sites, ordered tail emission.
  Existing `on_chunk` waits for the previous wave tail; reuse that ordering.
- `program/igc_learned_diagnostic.hpp`, `generate.cpp`: attach both primary wave
  lanes before initialization; accept completed prompt segments separately from
  staging feature rows. Preserve MTP callbacks and exact accepted-input ownership.
- `docs/BLUEPRINT.md`, report and Issue #276 comment: measured requested storage,
  actually qualified producer paths and remaining full-Hub validation debt.

Each wave lane owns 6,553,600 requested device bytes for two 80 x 4 x 2560 snapshots.
This is additional device storage, not a new host expert stager. The actual Hub's
existing stager/buffers remain unchanged. The combined two-lane estimate adds
13,107,200 bytes before driver/graph overhead. Profile cold/warm/tail costs after
correctness; do not claim that the transfers are free or activate by default.

Component tests qualify source-buffer lifetime and ordered consumption without
loading a model. They cannot establish actual batched-versus-serial activation
fidelity, full-engine memory fit, learned extraction quality or speed. Those need
the unchanged full Gyro 262K / 5-worker Hub route when resources admit it. Retain the
missing-feature bypass until that producer is present and qualified.
