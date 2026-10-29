# I2: native early-observation fidelity checkpoint

I2 remains open. The final-HC attention input head extracted zero of 12 exact
positive known-diagnostic tuples despite fitting the training examples. This
checkpoint adds actual early native observations so the next controlled experiment
can test the seam rather than assuming the final representation preserves digits.

## Change and boundaries

The existing opt-in first-window dump now enables two first-stage verifier buffers
before `Verifier::init`: the HC embedding broadcast and the output of the first
native `pre(layer0)` before its MoE result is combined. The graph copies each
actual tensor into an owned buffer; the serial prompt reader copies its selected
rows before the next window/commit. Fusions and Hub flags remain unchanged.
This second point is called `native_first_attention_output`, not a reconstructed
or purported complete Hugging Face layer-one state.

Each archive retains the maximum 80-row bound, token/position/source identity and
missing-cache mask. Distinct magic and metadata prevent cross-loading final-HC,
embedding and attention archives. The default path has neither additional buffers
nor copy nodes. That code property is not a measured disabled-parity claim.
Pipelined, layer-split and batch-MTP modes do not enable early capture. The existing
batched prefill callback still captures only final-HC; unobserved early rows stay
absent. These modes and real missing-cache cases remain validation debt.

The optional `igc_embedding_probe` loads the existing Q8_0 token-embedding table,
gathers only supplied IDs, broadcasts to four HC streams and repeats bitwise.
It is a fidelity oracle for these observations, not a token-based replacement for
the learned input head or a production calculator path.

## Verification

- Full CUDA engine build passed with architectures `89;120`. Strict CPU capture
  contract passed, including separate stage fixtures. Five Python archive tests
  passed with the independently generated C++ fixtures. Four attention input/mask/
  gradient/stage-identity tests passed. Red failures named the absent stage API
  before implementation. HIP and SYCL were not built.
- Native embedding control: two tests passed in 1.240 seconds. All selected Q8_0
  rows matched independent FP16-scale/int8 block decoding and all four HC streams.
  Malformed IDs were rejected before loading model data.
- Real Hub-profile smoke: two requests, 65 and 72 prompt tokens, captured 64 and
  71 pre-anchor rows at all three stages, source `1` (serial verifier), no missing
  rows. Both API requests succeeded. Owned cleanup drained; served configuration
  and the original learned profile remained byte-identical.
- All 135 captured embedding rows matched the native control exactly, covering
  56 unique token IDs. The first 45 equal-prefix rows were bitwise identical between
  requests at embedding, first-attention and final-HC stages. This does not explain
  earlier cohort prefix variation or establish all cache/placement modes.

Candidate SHA256:
`ae6c485fb6b765ae4005183bac1712508b801dbfb1da35ee46bae50966597d7d`.
Embedding control SHA256:
`cfce4c2146630b897f71d45273fdadcabd6a4730b394b71acaa078238ab50c1e`.
Selected profile SHA256:
`06ef847e063bb9643c7bf3f486e2ffdadc29737493268238c21f59188d5f6c5b`.

Private evidence folder `igc-early-smoke-01` contains `result.json`, all three
stage archives, metadata/token identities and `fidelity.json`; wrapper logs are
`igc-early-smoke-collection.log`, `igc-early-fidelity.log`,
`igc-early-native-build.log` and `igc-early-stage-contract.log`.

The instrumented run took 26.906 seconds, ready after 18.781 seconds. Minimum
sampled system commit headroom was 13,709,135,872 bytes and physical available was
9,812,213,760 bytes. The owned job had only kill-on-close (`0x2000`); no process/job
private-commit limit was configured. These are diagnostic observations, not Hub
latency or overhead measurements. No feature was deployed or made default.

## Next gate

The [frozen early-input protocol](../plans/2026-10-29-igc-early-input-experiment.md)
compares three stages from one 64-request cohort with the same input-head
architecture, seed and fit budget. The prior sixteen samples are known diagnostic
data. The new evaluation64 stays untouched unless a frozen arm meets the declared
development eligibility rule. Input extraction, output mapping and integrated
result use are separate gates; no usable integrated IGC or end-to-end speedup is
claimed here.
