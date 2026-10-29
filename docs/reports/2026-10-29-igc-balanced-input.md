# I2: balanced native-embedding input development

The balanced input-head baseline completed but failed its quality gate. I2 is
still open, the fresh evaluation64 remains untouched, and no learned integration
or model speedup is claimed.

## Data and fidelity

The [frozen protocol](../plans/2026-10-29-igc-balanced-input-experiment.md) uses
4,096 fit rows and three independent 128-row development axes, seed 20261030,
canonical row SHA256
`193f34b29fef6e884a4074a2e20db858f4467e2cc138365c76995d39ed8edf62`.
All four readiness/operator classes are balanced. Eight absolute operand values
are withheld from fitting; all decimal target digit classes are nevertheless
covered. Every new pair excludes the pilot and untouched evaluation pairs.

The selected tokenizer and one constant wrapper recreated all 64 real collected
prompt token sequences exactly. Every generated expression is inside the last
80 pre-anchor states; maximum distance from anchor is 26. The task-owned native
embedding control gathered 141 unique Q8_0 rows. All four HC streams matched
independent FP16-scale/int8 block decoding exactly. Native activation-cache
SHA256: `f9d24a283fbd63dff335c517090dcb21493e8e2385761460068ea1496f4c46be`.

Only these frozen F32 activations enter the learned head. Cache indices select
rows for minibatches and are not network features; a row-renumbering test leaves
network inputs identical. This seam does not require executing the frozen Gyro
body, and establishes no later-layer activation fidelity. The model and Hub
settings remain unchanged. Preparation evidence is
`igc-balanced-embedding-prepare.log` and `igc-balanced-embeddings-01/report.json`.

## Fixed baseline result

The existing query-free 63,492-parameter readout ran 30 epochs, batch 64,
AdamW 0.001/0.01, deterministic CPU Torch with four threads. It took 209.468 s
including final evaluation. Mean epoch loss dropped from 12.113136 to 1.805301.
It did not fit every training tuple, so the eligibility gate failed before any
fresh evaluation. No checkpoint was selected on development scores.

| split | correct operator | exact tuple | false calls |
|---|---:|---:|---:|
| fit | 4096/4096 | 1457/3072 | 0/1024 |
| seen values/new templates | 112/128 | 20/96 | 16/32 |
| new values/seen templates | 128/128 | 6/96 | 0/32 |
| new values/new templates | 112/128 | 5/96 | 16/32 |

The unseen English negative label template accounts for all 16 false calls on
each template-new axis; the Thai negative template has none. These are observed
counts, not a claim about either language generally. New-value axes have 23/24
left-digit reversals, versus one on the seen-value axis. Signs mostly match;
digit/length copying and false triggering still prevent usable invocation.
Invalid encodings remain failures rather than being repaired.

Frozen weights SHA256:
`827a41e837db3515c386509f538069b1dfe2b442fbf9f0363a588e6526d0c0a1`.
Peak process private commitment from the recorded lifetime peak was
1,913,376,768 bytes. Training was CPU-only and used the small native activation
cache; it did not load the full model into a training framework. This is a
resource observation, not a serving latency or quality-benefit comparison.
Full evidence: `igc-balanced-input-fit.log` and
`igc-balanced-input-fit-01/{report,predictions}.json`.

## Next controlled arm

The [contextual protocol](../plans/2026-10-29-igc-contextual-input-experiment.md)
keeps this exact data/order/seed/30-epoch budget and adds one trained contextual
block before learned-query pooling. Its common initial weights are tested equal
to this baseline. Its result is not known at this checkpoint. It is input-head
development, not an additional layer in Gyro or a production host parser.

Thirteen current input/cache/corpus/diagnostic tests pass. Native cache indices
are range-checked before gathering, padding stays masked, and the new block has
finite gradients and excludes padded values. CUDA engine code is unchanged from
the earlier build; HIP/SYCL and disabled-token parity remain stated debt.
No output mapper, committed inference integration, deployment or new default.
