# I1b integer primitives: separate typed CUDA component

Issue #276. This is the integer portion of I1b: floor division, modulo, gcd and
integer-valued powers. It is not completion of I1b rational/decimal support,
learned routing, live Gyro-M integration or practical-use benchmarking.

## Implementation and compatibility

`strata::igc::integer_extended::execute` accepts a separate ABI2 request/result
pair (160/88 bytes), integer domain 1, up to 64 disjoint caller-owned device rows,
and an explicit created CUDA stream. Canonical operands have at most 62 decimal
digits. No input text, expected answer or host arithmetic enters the executor.
One CUDA thread owns each row; buffers, arithmetic and loops have fixed bounds.
The call is asynchronous; the caller fences and handles deferred errors.

The existing ABI1 types, kernels and learned controller are unchanged. The new
`STRATA_BUILD_IGC_INTEGER_EXTENDED_TESTS` option is OFF by default; the component
is not linked into the current model controller. Its test alone links the
unchanged ABI1 kernel for cross-version and original multiply controls. Both
entrypoints reject foreign-version records without a numeric value. A future
adapter must explicitly declare the new ABI and operations before using them.

Numerical contracts:

- Floor division rounds toward negative infinity. Modulo has the divisor's
  sign, obeying `a = b*q + r` and `abs(r) < abs(b)`.
- Gcd is nonnegative, including `gcd(0,0)=0`.
- Powers define `0**0=1`. Bases 0, +1, -1 are handled before inspecting large
  exponents. Negative powers of non-unit bases are unsupported pending the
  rational extension; zero to a negative exponent is a domain error.
- Divide/modulo by zero are domain errors. Overflow, malformed, unsupported and
  fault results carry no numeric value, including on reused output storage.
  Values never wrap or clip. Input/output overlap is refused before launch.

Decimal long division uses at most 62 input digits and nine trial subtractions
per quotient digit. Euclid is capped at 498 iterations: inputs have fewer than
248 bits, and every two Euclidean steps at least halve the larger magnitude.
Non-unit powers at exponent 206 or above overflow the 62-digit domain; smaller
exponents use at most 8 binary steps and checked 124-digit intermediate products.
The per-test requested input/output storage at 64 rows is 15,872 device bytes;
this excludes CUDA context, runtime and kernel local storage and is not a
measured process peak or a residency claim for the language model.

## Test-first evidence

Successfully rebuilt red controls fail at the new floor, modulo, gcd, power and
overlap contracts. Each slice then passes after its corresponding implementation.
The early modulo build had an unqualified test constant; that compile failure
and an accidental old-binary run are retained and excluded from red/green claims.
Additional literal controls cover signed divisors, exact division, zero,62-digit
limits, huge unit-base exponents, domain/unsupported/overflow failures, disjoint
buffers, created streams and ABI1 multiplication `31 * -24 = -744`.

The native test CLI receives only frozen request bytes and writes raw results.
`tools/igc/integer_extended_oracle.py` uses independent Python big integers,
`divmod`, `math.gcd` and `pow` to construct expected results before native runs;
comparison happens offline afterward. Expected values are never a native input.
These new numeric corpora are separate from the sealed learned acceptance64.

Corpus01 has 1,271 requests: 957 successful values, 80 malformed, 39 unsupported,
165 overflow and 30 domain errors. It passes byte for byte in direct batches 1/64,
graph batches 1 repeated 2 times, and graph batches 64 repeated 64 times. Input
SHA256 `b11f82ffee70f7a0ff9d986ee55e103e9ca5be45174ec6e7b5cb3a7d7a4f7cb1`;
expected/output `79f6bbcd8d34a54c642ba0812091dc55362008ad3f55adf986cd75925af899df`.

A read-only review found no important numerical/compatibility defect but noted
that the random full-range divisor biased zero quotients. Corpus01 remains
unchanged, with its original generator preserved. A separate corpus02 varies
both operand lengths independently and retains signed/range cross products,
consecutive Fibonacci Euclid cases, powers and malformed records. Its 1,271
requests are pinned by
`d2db9a7ceed8a24cd5a498885380a8aee7867e313f703531aa3f5cfa60cddf31`;
expected results `fa8d2ab98c2dfa4e7090de4f2e902d3e97a07c227f1133b9b9c112a115f09bf7`.
Corpus02 passes all 1,271 records byte for byte in each final qualification mode
below. The test executable SHA256 is
`71a8fa2718aa1f0f0d1faffc61e6dae69729bad1fa9a12c6dfbada04703e3fb3`.

## Qualification status

The isolated Release build contains CUDA 89;120. Windows CUDA 13.3.73 / driver 621.01
and the primary RTX 5060 Ti are the current target. The first sandboxed memcheck
reported `No attachable process found` and reached the 150-second owned-job
cleanup. It is not a sanitizer pass. Repeating the same executable and corpus01
memcheck in the native execution context completes with zero errors and exact
oracle output. The tested discriminator is execution context, without changing
the binary, driver, system settings or CUDA workload. This separates that attach
failure from the successful native-context instrumentation; it does not count
the failed run as coverage.

Final corpus02 qualification uses the native execution context:

| Mode | Result |
|---|---|
| Direct batch 1 and batch 64 | 1,271/1,271 each, exact bytes |
| Graph batch 1, repeat 2 | 1,271/1,271, exact bytes |
| Graph batch 64, repeat 64 | 1,271/1,271, exact bytes |
| Memcheck, graph batch 64 repeat 2 | 1,271/1,271, zero errors |
| Initcheck, graph batch 64 repeat 2 | 1,271/1,271, zero errors |
| Synccheck, graph batch 64 repeat 2 | 1,271/1,271, zero errors |
| Racecheck, graph batch 64 repeat 2 | 1,271/1,271, zero hazards |
| Secondary RTX 4070 SUPER, graph batch 64 repeat 2 | 1,271/1,271, exact bytes |

The secondary smoke was admitted with 9,364 MiB free display-device memory;
it does not load the model or represent a dual-GPU model benchmark. Sanitizer
scope is this component; no inference about full-engine memory safety follows.
HIP/SYCL/Linux are unrun. No quality, latency or Hub-default claim.

Root CMake integration builds `igc_integer_extended_test` and `igc_exact_run`
together with the existing native head test subtree enabled. The root-built
literal/cross-version controls and full corpus02 graph batch 64 repeat 2 pass.
`cuobjdump --list-elf` confirms both sm89 and sm120 cubins in both standalone and
root-built test executables. Root test SHA256:
`de2d140bada8cecdf823101858d29a1ca29c966f0b53f2a0862a191620646895`.
The standalone binary is the one qualified by all four sanitizer tools above.

An initial unquoted PowerShell
`-D` path split into two arguments, configuring the wrong existing dependency
as the source project. Its failed build is preserved and excluded. The corrected
explicit source and quoted path use a fresh isolated directory. Existing Hub
and candidate engine binaries are untouched.

Reproduction uses the existing installed toolchain and local dependency source:

```powershell
cmake -S tests/igc/integer_extended -B <task-build> -G Ninja "-DCMAKE_CUDA_ARCHITECTURES=89;120" -DCMAKE_BUILD_TYPE=Release
cmake --build <task-build> --target igc_integer_extended_test --parallel 4
python -B -m tools.igc.integer_extended_oracle generate <new-corpus-directory>
# Inside the task headless wrapper, with the selected device and a timeout:
igc_integer_extended_test.exe <new-corpus-directory>/requests.bin <actual.bin> 64 2
python -B -m tools.igc.integer_extended_oracle verify <new-corpus-directory> <actual.bin>
```

The wrapper supplies the installed CUDA bin directories. On PowerShell, quote
the entire `-D` argument when its value is a drive path or a semicolon list.
Do not reuse a directory belonging to a different source project. Raw oracle
comparison is outside native execution; no times here are a latency benchmark.

Private receipts use the `igc-integer-extended-*` prefix under the existing P0
evidence root; `validation-01.json` retains all four corpus01 regular modes and
the failed attach attempt; `validation-02.json` retains the earlier build; `validation-03.json` records all eight successful
corpus02 primary modes. The secondary smoke has a separate log and raw output.
Commands are bounded through the task headless
wrapper. The live learned-model gate is still separately blocked by measured
system commitment headroom; no profile, system settings or artificial model
commit limit was changed to make the numeric component run.

Final source/binary receipts are in `root-qualification-02.json`. A trailing
blank line in the test was removed before commit; the rebuilt final binaries
were requalified, retaining the earlier logs and corpus. The candidate engine
SHA256 remains `74494c52e952eff3ba36eb7495112ba92136db65128aa65df874a65fe1d1a7e3`.
