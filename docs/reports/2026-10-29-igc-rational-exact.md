# I1b exact rational and decimal CUDA qualification

Issue #276. This qualifies the separate typed numerical component. It does not
establish learned selection, result use, live-model parity or practical Hub speed.
The existing integer ABI1/ABI2 and current controller are unchanged.

## Semantics and implementation

Reference contracts were read from the pinned [upstream calc.c](https://github.com/divagr18/qwenputer/blob/ca13310e2f9f788d1dd3feeea8213c1b0c23913c/experimental/riscv/chipformer/rv/env/calc/calc.c)
and the independent `tools/qwenputer/oracle.py`. Rationals support exact +,-,*,/;
decimals support exact +,-,* as coefficient times a power of ten. No new upstream
code or firmware was executed. This is first-party software arithmetic on CUDA,
not the secondary compiled FFN/GDN path and not a host calculator.

`core/igc_rational_exact.hpp` / `kernels/igc_rational_exact.cu` define ABI3 with a
304-byte request and 152-byte result. Rational domain 2 accepts signed denominators
and unreduced inputs with scale zero. Decimal domain 3 requires denominator +1
and scale 0..61. Input integers have canonical sign/length/unused digits and at
most 62 decimal digits. Results have reduced positive denominator, with zero 0/1.
The output is an exact rational, without a particular textual display spelling.

There is no implicit rounding. Rounded decimal conversion would require explicit
precision/mode and is not part of this contract. Scales above 61, decimal division,
mixed input formats and rational powers/floor/mod/gcd are unsupported here; the
existing separately typed integer operations keep their semantics. No owner
policy choice is needed for the settled exact operations. Future rounding policy
remains unselected rather than being guessed.

Malformed/version/reserved/zero-owner, unsupported, mathematical domain, final
overflow and internal fault statuses are distinct. Failures carry no numeric
value, including when output memory is reused. Synchronous pointer/count/stream
or overlap refusal returns false; its caller must not consume a prior output as
the refused operation's answer. Deferred CUDA failures require stopping/fencing,
not resuming from an assumed complete result; fault injection is not claimed.

Cross-products need 124 digits, signed sums 125. Fixed 126-digit work buffers and
bounded division/GCD reduce before checking final numerator/denominator against
62 digits. Euclid has a conservative 1,010-step cap, allowing an initial swap
before the usual every-two-step halving argument. This intentionally accepts
values that fit after reduction even when upstream's smaller intermediate
working capacity would refuse them. Historical secondary-path refusals remain
unchanged; no byte-for-byte firmware-error parity is claimed for this new ABI.
There is no host math/parser, float arithmetic, explicit device allocation or
sync in execution. Error strings may allocate on the host. Read-only review
identified the earlier unqualified no-allocation wording; this clarification
changes only documentation/comments, not the qualified kernel or ABI.

## Test-first and exactness evidence

Observed red assertions cover 1/3+1/6, the separate subtraction/multiplication/
division literals, decimal 0.1+0.2 and missing overlap rejection. The corresponding
implementations pass. An auxiliary red-run driver initially expected the wrong
wrapper exit code; the actual multiply assertion failed as intended, its log is
preserved, and the remaining divide control uses the corrected expectation.

Representative native CUDA results:

| Input | Expected and actual |
|---|---|
| 1/3 + 1/6 | 1/2 |
| 1/3 - 1/2 | -1/6 |
| (-2/3) * (9/-4) | 3/2 |
| (2/3) / (-4/9) | -3/2 |
| decimal 0.1 + 0.2 | 3/10 |
| (max62/max62) + (max62/max62) | 2/1 after 125-digit intermediate reduction |
| equal huge fractions subtracted | 0/1 |
| either canonical denominator zero | DomainError, no numeric value |

The frozen independent Python `Fraction` corpus has 1,403 records: 850 successes,
375 overflow, 12 domain errors, 157 malformed and 9 unsupported. It includes
signed/unreduced/range crosses, separately varied operand lengths, exact decimal
scales, cancellation, metadata and malformed records. Explicit unit controls
cover each canonical zero denominator and division by a zero fraction, final
numerator/denominator overflow and reused outputs. The reviewer noted the
canonical-zero coverage gap while these unit additions were underway; the final
run exercises them. Read-only review found no important numerical defect.

The native file mode reads only requests and writes results. Expected records
come from the independent stdlib oracle and are compared AFTER execution; they
are not inputs to the native process. The corpus is unrelated to sealed learned
acceptance64, which remains unopened.

## Qualification

Windows, CUDA 13.3.73 / driver 621.01, primary RTX 5060 Ti; Release 89;120 builds.
All rows below use the frozen 1,403-record corpus and agree byte for byte:

| Mode | Result |
|---|---|
| Primary direct batches 1 and 64 | 1,403/1,403 each |
| Primary graph batch 1 repeated twice | 1,403/1,403 |
| Primary graph batch 64 repeated 8 times | 1,403/1,403 |
| Primary memcheck/initcheck/synccheck/racecheck, graph batch 64 repeated twice | 1,403/1,403 each, zero errors/hazards |
| Root-built graph batch 64 repeated twice | 1,403/1,403 |
| Secondary RTX 4070 SUPER graph batch 64 repeated twice | 1,403/1,403 |

Separate ABI-refusal tests pass normally and under memcheck: ABI1/ABI2 reject
ABI3 records, and ABI3 rejects old version tags, without numerical values. Root
CMake builds all three numeric test targets together with the existing head
subtree enabled; root literals/refusal and old integer regression pass. The
standalone rational executable is the one qualified by all four sanitizer tools;
root/secondary coverage is regular. Both exes contain sm89 and sm120 cubins.
Secondary admission observed 9,623 MiB free display-device memory.

The caller's requested 64-row input/output storage is 29,184 device bytes,
excluding CUDA context, runtime and kernel stack. `cuobjdump` reports kernel
stack 8,296 bytes / 168 registers on sm89 and 8,416 / 255 on sm120. These compiler
resources are not measured resident peaks and must not be multiplied into an
invented allocation claim. This is functional qualification, not a performance
benchmark or profiler-off ABBA. HIP/SYCL/Linux and device-fault injection are unrun.

Frozen references:

- Requests `b54911b607fac273f8a989ba37da186fe4b194eaa05754fc47485ef2a49541ef`.
- Expected and actual `2d347a058a440d708e7d483ba5012c0fd4f70a40b129ef0bf8f588f0ea9f9aaa`.
- Standalone test `e07fb01a8c1133fcab858858757093885cfbcd1154ded74bad6f82939af78842`.
- Root test `8a2eb8881a686192ff7af9118ba18370ca5b733ddd17b2d48ad3cb94dc6097f1`.
- Compatibility test `a227ca892f08b2d7806665ad5acf8ebd0c6d128ad7c5b4a70a7b6b9b6539676d`.

Private receipts `igc-rational-validation-01.json` and
`igc-rational-root-qualification-01.json` retain commands, hashes and outcomes;
all earlier red/control logs remain. The root receipt pins current source and
frozen executable copies. `STRATA_BUILD_IGC_RATIONAL_TESTS` defaults OFF; the
component is not linked into the current learned controller and no artifact is
made eligible for ABI3. Candidate engine hash remains
`4cb668aac953b0712cb82095f8acdc41ed727e56294d457e6ab6542df87c5871`.
No Hub/profile/system/default changes or full-model run occurred here. Live
validation still precedes major repeated-trigger development and benchmarks.
Pure arithmetic graph repetition does not prove once-per-accepted-event
invocation ownership; connecting this ABI to that ledger remains separate work.
