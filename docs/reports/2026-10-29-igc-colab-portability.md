# Bounded input-adapter portability pilot

Issue #276. A separate authorized Colab worker ran the immutable synthetic
activation package on one Tesla T4, then stopped it and verified zero assignments
and zero active consumption rate. This implementer prepared the package and read
the returned raw reports; it did not start another cloud job. No installation,
full-model upload, user prompt/source upload or acceptance64 consumption occurred.

The ZIP is 2,222,586 bytes, SHA256
`95ec874af493f41b28a37adda3d0f0911679e892fb0982ce17cb384441543738`.
It contains a bounded native embedding table, synthetic indices/masks, canonical
training labels and nullable alignment, split identities, saved reference
predictions, a frozen input readout, and standalone first-party NumPy/PyTorch
code. It contains no tokenizer or textual corpus. The sealed acceptance set is
excluded. Source bundle SHA256:
`091b471f22e0a1927bc3e7b926f62971d5b68ceea952974ec25e4ff30bef675f`.

Commands: `python pilot.py --device cpu --check-only --output check-cpu`, then
`python pilot.py --device cuda --steps 32 --max-seconds 120 --output pilot32`.
This is the first32 updates of the existing70-epoch schedule, not a converged fit.

Both local RTX5060Ti and remote T4 used PyTorch2.11.0+cu130. NumPy was2.5.2 locally
and2.1.3 remotely. The reference check preserved384/384 nine-field decisions on
both. Post32-step prediction arrays were byte-identical, including the inspected
development metrics. The32 losses differed by at most3.814697265625e-6; trained
weight arrays were not byte-identical. These distinctions are retained.

| observed scope | local repeat02 (s) | T4 pilot (s) |
|---|---:|---:|
| preparation inside Python | 2.6478225 | 6.660673944 |
|32 training updates | 1.745453900 | 1.444363831 |
| Python main, after imports | 4.468779500 | 8.192609942 |
| outer process, including import/startup | 7.3013347 | 12.681 |
| cloud allocation/transfer/setup/run/download/cleanup | n/a | 55.2650020 |

Both reported928,696,832 peak PyTorch-allocated bytes; this excludes CUDA context
and other allocator/device memory. The returned ZIP is362,858 bytes, SHA256
`b1f34e41aa2e89ca50e47fef13dda69110b3c2e07bc473a3bc419ea51ace5f65`.
Raw evidence: the worker's `result.json`, `comparison.json` and downloaded
`pilot32/report.json`; local `colab-input-local-gpu32-02/report.json` and its
outer-process timing record. Full private paths stay in the local checkpoint.

This one bounded pilot establishes portability of the cached-input experiment,
not a statistically qualified speed improvement, a70-epoch quality result,
whole-model training support or Hub performance. It does not fix I2 generalization.
Small jobs remain local because transfer/setup dominated this pilot. Consider
Colab for a separately planned longer or parallel training workload; early output
adapter training still requires its actual differentiable native-head path and
cannot be inferred from ordinary PyTorch input training.
