# Bound learned-controller inputs

Issue #276. The live controller needs four existing diagnostic neural assets:
embedding input, first-attention input, final-state readiness and late result
output. Bundle only weights and their model/tokenizer/file identities. Do not
introduce a prompt lookup table, operands, checked answers or serialized calls.

Change inventory: `core/igc_controller_bundle.hpp/.cpp` validates the fixed host
manifest and loads all four networks transactionally using the existing adapter
reader; `tests/igc/controller_bundle_probe.cpp`, host CMake and Python tests cover
valid roles, production refusal before file access, exact byte/header/nonzero
digest bounds, changed digest, wrong role, missing files and preserved old output
after failed replacement. Update Blueprint/evidence with the tested component.

Format v1 is exactly204 bytes: magic `IGCLAT01`, little-endian version1,32-byte
model digest,32-byte tokenizer digest and four32-byte artifact digests. Fixed
sibling filenames are `input-embedding.bin`, `input-attention.bin`,
`readiness.bin`, `output.bin`; required kinds1,1,3,2. No paths or executable
payload in the manifest. Refuse nonregular/reparse leaf files and production use.
Load into a temporary bundle and publish only when every digest/schema/kind
passes. The runtime must additionally bind actual model/tokenizer receipts before
CUDA preparation; successful parsing establishes compatibility metadata, not
learned semantic quality or safe selection on arbitrary prompts.

The rejected partial-prefill test remains in local evidence: Prefill requires a
real successor whenever a stage ends before all48 layers. Do not relax that core
invariant for the observer test. Qualify the batched producer through the full
Hub engine with the selected profile; no surrogate stage or forced-cold baseline.
