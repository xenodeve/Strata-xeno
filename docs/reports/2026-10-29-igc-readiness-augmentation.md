# Additional readiness fit coverage: negative result

The frozen [augmentation](../plans/2026-10-29-igc-readiness-augmentation.md)
collects416/416 new fit-only real final-anchor vectors in297.750s, unchanged
Hub/Vision/GPU/profile and no memory cap. Initial data preparation exhausts its
lhs-zero quota before any capture; inventory proves15 unused lhs-zero and19
rhs-zero pairs. Using both orientations retains all disjointness and24 zero-pair
quota. A regression test covers this and all pair/template ownership.

With624 fit rows and the original112 development rows, the unchanged MLP and
300-step recipe fit624/624. Development becomes110/112:46/48 positives, zero
false calls on64 negatives. It still misses `(19) - (19)` and additionally the
development multiplication pair `development-op3-00-ready1`. The previous MLP
was111/112. More coverage is not an improvement; retain the earlier weights and
all original reports. No threshold adjustment, fresh acceptance or promotion.

State `77f15bfb91c04382ff352e4b5d4c33a0b8db07c7813a94f790dabb9fb16b119a`;
additional corpus `e7e5bf2fcda56fedad2b47b6966ab967243a5abf2f1a9239bc62710825d2b7c5`.
0.859s is offline fitting/evaluation, not inference latency. The loader independently
checks each completed capture, then concatenates only disjoint fit rows with
training-only normalization. Private evidence: `igc-readiness-augmentation-01/`,
`igc-readiness-augmented-fit-01/` and DONE0 logs.11 corpus/accounting/collector
tests pass. Language weights, serving defaults and input acceptance64 unchanged.

## Operand diagnosis using saved predictions only

An offline intervention replaces only predicted digit values with literal-token
values at each saved attention argmax, preserving predicted operator/sign/length
and rejecting nonnumeric source positions. It is a **diagnostic oracle**, not an
activation mapper or a proposed host decoder. Known-development embedding exact
tuples drop50→39,77→29 and33→26 per96, respectively. First-attention states give
zero under literal copying although their actual readout scores96/50/67: their
contextual information is not confined to literal digit locations. This rejects
simple pointer-value substitution as the next implementation. It does not prove
that every future pointer-trained architecture fails.25 and21 new-template digit
fields have the correct source position yet the learned value is wrong; thus
wrong position alone also cannot explain the errors.

Diagnostic source `44e93c1fef514858c6a2bfb77fae06a464616e512561b3e2596a6a81e6d7fdd4`,
saved details `c7a94a1950ff9366c9047abddf6dae047f7a210597880c6e3588d4bd7cc5e69e`;
private `igc-pointer-value-ceiling-01/{report,cases}.json` and sibling source script.
No historical predictions are changed.

A separate read-only join of the existing aligned embedding and first-attention
predictions finds exact nine-field agreement on49/96,50/96 and31/96 positives;
all130 agreements are correct in these **already inspected** development cases.
No negative has an agreed nonzero tuple. This motivates a future bounded selective
controller experiment, not an independent acceptance pass or proof that agreement
guarantees correctness. Native-forward qualification is the next integration work;
I2 readiness/generalization, ownership and full-model quality gates remain open.
