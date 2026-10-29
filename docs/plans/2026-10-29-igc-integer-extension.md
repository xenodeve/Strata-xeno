# I1b: bounded exact integer extensions

Issue #276. Continue independent numeric work while full-model admission is
blocked. This is part of I1b, not completion of rational/decimal support or a
claim that the learned mapper can select these operations.

## Compatibility boundary

Keep the existing `igc_exact.hpp`, its ABI1, kernels, learned selection and output
mapping unchanged. Add a separate `integer_extended` namespace, ABI2 typed API
and optional CUDA test library, off by default and not linked into the current
controller. Its 160-byte request / 88-byte result reuse the canonical 62-digit
integer representation, with distinct types/version/opcodes. ABI1 and ABI2
entrypoints reject the other's records without a numerical value. Future model
adapters must explicitly declare/support this version; no automatic fallback.

Operations are floor division, modulo, gcd and integer-valued powers. Floor and
modulo follow the existing independent oracle's Python semantics: quotient rounds
toward negative infinity, remainder has the divisor's sign, and a = b*q + r.
Gcd is nonnegative, including gcd(0,0)=0. Power uses 0**0=1; unit bases support
arbitrary signed integer exponents, zero to a negative power is a domain error,
and non-unit negative powers are unsupported pending rational support. Overflow
and domain errors have no numeric value. Never clip, wrap or substitute zero.

## Bounded execution

Use digit arithmetic only on the device. Long division has at most 62 input
digits and nine trial subtractions per quotient digit. Gcd uses Euclid with at
most 8*62+2 iterations: fewer than 4*62 magnitude bits, and every two Euclidean
iterations at least halve the larger input. Power handles 0, +1 and -1 first; other
nonnegative exponents of 206 or more overflow because 2**206 exceeds 62 decimal
digits. Exponents below that fit eight binary exponentiation steps. Multiply
uses a bounded 124-digit intermediate and checks final range. No host arithmetic,
token parser, allocation, synchronization or unbounded loop in the executor.

## Test and delivery sequence

1. Add the isolated typed seam and one literal floor test; observe red on an
   unimplemented device operation, then implement only that slice.
2. Add modulo, gcd and power slices with literal signed/boundary cases and red
   controls. Freeze an independent standard-library integer oracle corpus,
   separate from any learned-model acceptance set.
3. Test malformed/cross-version/domain/zero/overflow records, repeated execution,
   reused output, graph1/64 and maximum-digit inputs; compare oracle results
   offline. No oracle is an input to native execution.
4. Build CUDA89;120, run primary GPU controls/sanitizers and a secondary-device
   smoke when display headroom permits. Preserve ABI1 test/output behavior.
5. Update Blueprint/report and bilingual Issue #276 comment with actual evidence,
   hashes, numerical contracts, costs and unrun gates. This is functionality
   qualification, not an end-to-end speed claim or Hub default change.

The next live-model gate retains priority when resources admit it. Do not run
heavy jobs together or alter the selected Hub profile, system settings, process
limits or unrelated processes to make capacity appear sufficient.
