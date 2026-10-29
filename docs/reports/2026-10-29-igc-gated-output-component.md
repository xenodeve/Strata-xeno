# I3 gated result-to-activation component

Tracker [#276](https://github.com/xenodeve/Strata-xeno/issues/276).
[Contract](../plans/2026-10-29-igc-late-output-mapper-component.md).

`tools/igc/output_mapping.py` independently implements categorical result
attention and a learned per-position gated residual at the late native head input.
Its offline encoder binds ABI1 version/status/invocation/consumed position and
canonical62-digit signed integers. It preserves the exact ABI's little-endian
digits and presents a left-aligned neural view. No result text, tokenizer IDs,
chosen answer tokens or host arithmetic enter the mapper.

Hard off returns the original tensor without running modules. Error statuses
carry no numeric value and hard-bypass the residual, including failed rows beside
successful rows. An attention mask prevents padding from changing the proposal;
the mask test explicitly uses nonzero proposal weights, so zero initialization
cannot hide a broken branch. Six CPU boundary/mask/ownership/gradient tests pass.

`igc-output-mapper-native-gradient.log` adds the seventh test on real CUDA using
the qualified native-head DLL and actual cached Gyro mixed activation. Only the
new mapper trains. One supervised step changes native-forward CE
21.0040245 to19.6906242 with finite nonzero proposal gradients, leaving the base
activation unchanged. Its oracle result is42 and target is only the numeric symbol
4: this proves the composed result-residual/native-head loss seam, not full-number
understanding, learned input extraction, an arithmetic execution or model benefit.

I3 is not complete. Full-sequence native teacher-forced output-position data,
trained faithful-result use, absent/corrupted result controls and integrated model
evaluation remain. This late-head variant has no backward through HC/GDN/MoE or
attention and is not the paper's early-layer adapter. Production tensor encoding,
native mapper execution and committed exactly-once injection are not implemented
by this Python training component. No serving dependency or default change.
