# I3 native result-mapper forward qualification

Implement the existing frozen `ResultMapper` on CUDA without Torch in the runtime.
Use the strict existing kind2 artifact and explicit diagnostic opt-in. This is a
numerical/integration prerequisite, not permission to serve an ineligible pilot.
The standalone target builds architectures89;120; it does not move the full engine
or change serving configuration. No new third-party execution is involved.

Load weights once after host artifact validation. Bind one workspace to one
explicit created stream, enforce count1..64, and allocate only at load. `run`
uses device mixed activation, typed exact result and expected invocation/anchor
arrays. Validate canonical integer/status/ownership on device, encode left-aligned
numeric features, reproduce LayerNorm/query/result attention/gate/residual, then
publish only if the whole row is finite. Upstream failure/malformed/nonfinite
leaves original activation bytes intact, with distinct diagnostic statuses.
Hard-off performs no module/device work. No parser, host arithmetic, text-result
format or token-ID mapping enters native forward.

Before first comparison, freeze tolerances: mixed output max absolute1e-3 and
relative1e-4 against deterministic F32 Torch on the trained diagnostic421-position
archive. This permits different F32 reduction order, not a claim of bit equality
for learned enabled outputs. All output logits/argmax and close margins must be
checked through the qualified native head before in-engine use; disagreements
must be reported, never overwritten. Disabled and failure bytes must be exact.

Test batch1/16/64, repeated A/A and regular CUDA graph relaunch at stable addresses,
wrong IDs/anchor, noncanonical digits, upstream failures, NaN/overflow activations,
and source/file/shape bounds before upload. Check applicable sanitizers and keep
WHILE-path historical debt distinct. Record device weights/scratch/transfer costs
but make no speed claim from this qualification. Caller must drain its bound
stream before changing weights, destroying buffers or resetting a session.

Next is native model injection/free generation with controlled known operands,
then I2 plus model-selected calls and transaction/MTP ownership. This step alone
cannot close I3/I4 or trigger Hub default activation.
