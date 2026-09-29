# P-core ordering and dual architecture build

Engine issue: xenodeve/Strata-xeno#2. Measurement issue: xenodeve/Qwen3.8-Flash-Next-Tuning#11. Base `f679806`.

Windows `PROCESSOR_RELATIONSHIP.EfficiencyClass` assigns a higher value to a faster physical core ([Microsoft](https://learn.microsoft.com/windows/win32/api/winnt/ns-winnt-processor_relationship)). `physical_cores` now sorts descending by this class before reserving the first core for the host. `--pool-workers N` takes the first N remaining physical cores, so N=5 uses only P-cores on this i5-13500 and N>=6 explicitly includes E-cores. The automatic worker count remains 13; the rough sweep below does not justify changing it to 5.

The machine reports 14 physical cores: six class-1 P-cores and eight class-0 E-cores. `xeno_pool_core_policy.exe` checks Windows topology against the returned affinity order and checks auto=13, P-only=5, explicit E=6. Before the sorting work, its initial auto-P-only expectation was red at actual 13; that expectation was revised after the Strata sweep showed E-cores can help its dynamic worker pool.

| Check | Result |
|---|---|
| CMake `-DCMAKE_CUDA_ARCHITECTURES="89;120"`, `strata` target | Built successfully for both architectures. No model run on the 4070 SUPER yet. |
| `STRATA_BUILD_XENO_TESTS=ON`, `ctest -R xeno_` | Four tests passed: physical core ordering, Q2_0 AVX2 versus AVX-VNNI bit-exact row outputs at both model widths, native VNNI dispatch, and forced AVX2. The test target is not built by default. |
| Normal CPU feature detection | AVX-VNNI yes, AVX-512 no on the target i5-13500. |
| `STRATA_FORCE_AVX2=1` | AVX-VNNI no on the target i5-13500. |
| Startup log | Native Q2_0 prints the selected AVX-512, AVX-VNNI or AVX2 tier, replacing an inaccurate AVX2 message. |

Rough single runs on 5060 Ti, fixed 256-token long prompt, spec 4, 5,000 cache slots, no adaptive swaps, PCIe miss share 0, with background CPU noise. These are separate boots, not a precise ABBA comparison:

| Pool workers | Decode tok/s |
|---:|---:|
| 0 (auto=13) | 39.31 (another preceding run: 40.95) |
| 4 | 39.28 |
| 6 | 37.61 |
| 10 | 41.08 |
| 13 | 41.43 |

Commands and logs are at `%TEMP%/strata-afk-runs/pool-sweep`. The sweep does not establish an optimal count; it contradicts transferring EXL3's "E-core workers always slow the layer" conclusion directly to Strata. The default stays at 13 pending a quieter paired measurement.

The first policy test expected a P-only default and failed at 13 workers. The later sweep showed 10 and 13 workers faster than 4 and the auto P-only experiment in these rough runs, so the implementation retains the old auto count and makes ordering explicit. Passing `--pool-workers 5` chooses only P-cores; larger explicit counts admit E-cores in order. The deterministic Q2_0 parity test was added after review found that an ISA selection check alone did not verify kernel arithmetic.
