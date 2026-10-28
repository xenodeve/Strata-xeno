# Reviewed Qwenputer compiler subset

Pinned to divagr18/qwenputer@ca13310e2f9f788d1dd3feeea8213c1b0c23913c.
UPSTREAM.json records original Git blobs and SHA256 for every unmodified file.
LICENSE is the upstream MIT notice. No model, firmware binary, HF loader,
merged-checkpoint emitter, or Triton executor is included.

This source is for offline circuit compilation and test oracles. Production
inference must consume a validated data-only artifact through native C++/CUDA.
Execution approval is separate from vendoring. Picolibc/libgcc firmware notices
and reproducible firmware-build provenance are not established by this license.
