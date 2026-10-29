# Bound learned-controller assets

Issue #276; [contract](../plans/2026-10-29-igc-controller-bundle.md).
The new host loader binds the four already qualified diagnostic neural assets in
a fixed204-byte manifest: embedding input, first-attention input, readiness and
late result output. It contains only model/tokenizer/artifact digests, no paths,
operands, answers, prompt tables, code or invocation records. Fixed sibling files
must parse with their expected kinds1,1,3,2. Production is refused before file
access. The old bundle remains byte-identical if a replacement fails, including
failure after three networks have loaded into temporary storage.

The host build passes MSVC `/W4 /WX /utf-8`. The built behavioral stub first
failed on the valid-bundle test with `controller_bundle_unimplemented` (exit2),
not a missing executable. Red probe SHA256
`ad33700e3c9831e5c0abdaed95a11ef775cf196f6a099a2c35cf33a8026ac016`.
Final probe SHA256
`0038bc8eb5c453f8c55f98874432904112a792c54fa7fadb8daece2bf4a53b72`.

Final `igc-controller-bundle-green-run-03.log`:14 tests pass, zero skips, failures
or errors. Five new tests cover valid roles, production refusal, exact manifest
size/magic/version and all six nonzero digests, directory refusal, wrong family/
file digest, wrong role, missing network and transactional preservation. Existing
adapter/readiness tests cover finite tensor/schema/offset/positive-scale checks,
resigned malformed artifacts, native host kind1/2/3 compatibility and production
refusal. Run02's one unselected native-reader skip is retained; Run03 explicitly
selects the rebuilt reader and runs that test. No GPU/model inference is involved.

Prepared actual learned assets total4,947,912 bytes including the manifest:

| role | state SHA256 | artifact SHA256 |
|---|---|---|
| embedding input | `72a0020591a7fb59f0f74e53ab1eadaee5a31ec6ae3b45744c20e3dea99cdefd` | `1b094c86b2d198a9328722a879537b71fd5b97cd25a109553e92b70baaa4c649` |
| first-attention input | `519fbc196993c94676f4031f7bd45ad9abf661f8b7b4fd3b9d6358204b5294e2` | `5df29e817fdbad1ada7eedf7737bf3fc305cd45b9fb343d63deae59d887f782c` |
| readiness | `0ca4f0b8cc66a0f876d86623dd95ecef828bf9927c9fd9a55e2021b3d1497264` | `f65622f14b3134aa3259bc5f2f1356db2b8a7e963b31e084b06665d740fc3b9f` |
| output | `241ef7032aed2bb9b3e7a3bc8bddd990c963c1d182c63ea968136e1e863c0337` | `43cf5c8d341bf91966ffc6fa22aba67eb3e04be6a3dcfd694af4b21fb1c62c2c` |

Manifest SHA256
`3e853f3190cfb710b3da6f4513f5f981c6af272c533a410fbeb63406390a4fa5`.
The native reader loads this exact bundle successfully. Its local preparation
script, source bindings and result are in `igc-controller-bundle-prepare-01.log`
and `igc-live-controller-bundle-01/report.json`. The runtime still must bind the
actual model/tokenizer receipts before CUDA preparation. Parsing success is not
learned quality, fresh acceptance, live model selection or a training result.

## Batched-path qualification status

An attempted first-stage Prefill probe failed at initialization with
`prefill: the stage's layer range is wrong`. Source `prefill.cpp` requires a real
successor when a stage ends before all48 layers. That setup failure is not the
producer's behavioral red and does not diagnose a model defect. The test-only
partial-stage changes were archived locally and removed; the invariant was not
relaxed and no dummy successor was substituted. The committed serial probe was
rebuilt and its two real-weight tests pass again (`igc-prefix-restored-validation-01`;
new executable SHA256
`7435ba0f4c26f4a01aa5aba76c7d5b81d4bb816ed8c89110f055e37c3fd0229d`).

The batched producer must be qualified through the full actual-Hub engine. Current
resource sampling found no stale task executables. Available system commitment
was51,020,406,784 bytes; the previous same-profile engine+Vision run sampled
48,869,707,776 bytes combined private commit at peak. That leaves a narrow margin
before new batched staging, not proof of another allocation failure. No new full
model load, memory cap, setting change or unrelated process termination occurred.
Independent controller work continues; price/verify the actual transient demand
and recheck resources before launching. This diagnostic admission is separate
from the unrestricted Hub baseline.

This is a host asset-binding component only. Live learned controller, batched/wave
ownership, quality and end-to-end benchmarking remain open. No Linux/HIP/SYCL
execution or full security audit is claimed; malformed-data tests and narrow
manual trust-boundary review cover this preparatory loader. No serving defaults
changed and no artifacts were uploaded.
