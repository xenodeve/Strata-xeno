# I1b exact rational and decimal component

Issue #276, continuation after integer component `430e2081` and controller
timeline `d89ef4dd`. This work is independent of the blocked live-model gate.

## Semantics inspected

The pinned upstream `calc.c` at `ca13310e` defines exact reduced fractions for
addition, subtraction, multiplication and division. Its decimal coefficient/scale
path supports exact addition/subtraction/multiplication and does not mix fraction
and decimal operands. `tools/qwenputer/oracle.py` independently evaluates decimal
literals with `Fraction(original_source_token)`, bounds normalized numerator and
denominator to 62 digits, and does not promise a decimal display spelling.
Reuse those mathematical contracts; no upstream code/firmware execution is needed.

## Separate typed contract

Experimental `rational_exact` ABI3 does not modify ABI1 or integer-only ABI2.
The 304-byte request has a 32-byte owner/position header and two 136-byte operands:
signed numerator/coefficient, signed denominator, uint32 scale and reserved zero.
All input integers use the existing canonical 62-digit representation.

Rational domain 2 requires scale zero; denominators may be signed and fractions
unreduced. Decimal domain 3 means coefficient * 10^-scale, with denominator +1
and scale 0..61, whose power-of-ten denominator fits the 62-digit domain. Larger
scales are unsupported, not silently rounded. Decimal division/mixed formats are
unsupported in this scope. The 152-byte result always holds a reduced exact
rational with positive denominator; zero becomes 0/1. Every failure leaves both
integer lengths zero. Division by zero is DomainError; malformed, unsupported,
overflow and internal fault remain distinct. Header/operand reserved fields must
be zero. Range is checked AFTER reduction, permitting cancelling intermediates
above 62 digits. This deliberately avoids the upstream firmware's conservative
intermediate-capacity refusals; it does not change the frozen secondary evidence.

For example, decimal 0.1 + 0.2 returns exact 3/10. Rounded decimal formatting is a
different operation and is not added. A future conversion must specify precision
and rounding mode; the reviewed contracts do not choose half-even, toward-zero
or another policy. That open future policy does not block exact arithmetic.
Rational powers/floor/modulo/gcd are not silently introduced. Qualified integer
operations, including unsupported non-unit negative powers, keep their contracts.

## Bounds and delivery

Input cross-products fit 124 digits and signed sums 125. Use 126-digit work
storage, bounded decimal long division with at most nine subtractions per digit,
and at most 1,010 Euclidean iterations. An initial a<b swap may occur; afterward
every two steps halve the larger magnitude. Fewer than 504 magnitude bits plus
the initial swap fit the conservative bound. Reduce before the final range check.
There is no recursion, host arithmetic/parser, float conversion, explicit device
allocation or synchronization in execution. Error strings may allocate on the
host. The caller owns disjoint device arrays, count 1..64
and an explicit created stream, then fences and handles deferred device errors.

Test first: 1/3+1/6, signed rational operations, decimal coefficient/scale and
overlap controls. Freeze independent Fraction corpus bytes; the native CLI takes
requests only and comparison happens offline. Qualify direct/graph/repeated/
reused output, all ABI refusals, primary sanitizers, CUDA 89;120 root coexistence
and secondary smoke. Review source, update Blueprint/report and publish bilingual
phase-piece evidence after a local commit. The build option defaults OFF and no
mapper/artifact binds this ABI yet. No model, Hub/default or performance claim.
