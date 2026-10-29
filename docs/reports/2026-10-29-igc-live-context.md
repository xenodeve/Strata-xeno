# Actual device activation history and retained input ownership

Issue #276; [staged change inventory](../plans/2026-10-29-igc-live-context.md).
This completes the bounded context-history and serial device-consumer components.
It does not complete learned model integration, batched-prefill support or I2/I3.

`ContextWindow` stores80 native embedding rows and80 first-attention rows in fixed
GPU rings, with two right-aligned input buffers and an80-byte mask. Requested
device allocation is13,107,280 bytes; host canonical-token identity storage is
reserved for at most262,144 IDs. No tensor data goes to the host in this component.
Observations are staged; explicit acceptance advances only the contiguous
consumed-input prefix. True compatible prefix reuse preserves available rows;
unverified token/scope lineage and evicted rewind rows bypass IGC without forcing
a cold language prefill. Generated accepted inputs can extend the next request's
cache. The caller must also verify image/control/model identity before allowing
reuse. Missing features clear the device mask, preventing a stale eligible window.

Created producer streams enqueue D2D copies with event ordering. The caller fences
model consumers before resetting/releasing the cache. Only the newest80 staged
rows are kept; arbitrary concurrent/out-of-order producers have not been qualified.
The planned batched/wave controller must serialize metadata and preserve newest
position ownership before using this path. This is not a snapshot format.

`Verifier::visit_early_activations` exposes the existing genuine embedding and
first-attention buffers to a first-party device consumer after plain `run`, on
the owning stream. It validates retained count and emits no rejected suffix.
There is no graph host callback, tensor round trip or numerical operator change.
Ordinary builds do not allocate these buffers unless early observation is enabled.

## Evidence

- Built allocation stub failed at `context_allocate_contract`, exit2; SHA256
  `154a9d1500e3421b38a807432a539ea113dc6b5ec83cdf1b9da171c9c4cafcc2`.
  Device-ring tests then cover wrap, large chunk, right alignment, two streams,
  staged-versus-accepted rows, missing second tap, rewind, changed prefix/scope,
  generated-input continuation and fixed allocation. A new stale-mask test first
  failed at `unavailable_mask_cleared`; clearing the mask on bypass fixes it.
- Context test SHA256
  `2bd7a3dcb8bc9b853497bc7d4484875af0f559e7fb69a565450d4eb2ba8d74a8`.
  Memcheck/initcheck/synccheck each report zero errors; racecheck reports zero
  hazards/errors/warnings. Each tool exits0. The exact command/source/binary hashes
  and logs are in `igc-context-validation-01/report.json`.
- The initial test incorrectly gathered before an already-consumed last token,
  after that token had evicted the oldest desired row. The unchanged cache passes
  after correcting the test to the engine's actual boundary: prefill retains the
  first n-1 inputs; the last prompt token is the pending first decoder anchor.
  Missing rewind data remains a tested bypass; it was not fabricated or hidden.
- The built verifier consumer stub failed on real Gyro weights at
  `prefix_device_tap_visit`, exit2; SHA256
  `6b9576bf89ae5b210c8e8b0e084c328f6ecf839709bba077a30a7bb927195200`.
- Final CUDA89;120 first-stage probe SHA256
  `095770d16c98c1ebde1f6f4bd2baa812c45af65b76131dd4649b3d61570c901c`.
  Two tests pass with zero skips: malformed input before allocation, then actual
  Gyro requests A/B/A with reset repeats and window sizes1/8. Both device taps are
  bitwise equal to the old host observation. The retained first-attention ring is
  bitwise equal to all corresponding saved full-model reference rows; right-aligned
  masks and clamped retained-count descriptors also pass. This uses full48-layer
  geometry and the real [0,1) native verifier stage, not a shortened fake model.
  No language answer is sampled. See `igc-live-context-validation-01/report.json`.
- A first combined-probe build lacked its explicit `<array>` include and failed
  compilation. Build02 corrects that test include; the failed build log remains.

GetPerformanceInfo before final real-weight qualification reported51,357,945,856
bytes available system commitment and26,142,728,192 physical bytes available.
These are preflight samples, not peaks. The isolated first-layer probe's8GiB
commit/4GiB physical preflight is a diagnostic safety check, not a Hub restriction
or a process hard cap. The selected Hub profile and served executable are unchanged.

No performance claim or end-to-end quality pass follows from these checks. The
next work is actual batched early-row ownership and the learned controller hook,
followed by free generation and ownership/cancellation tests. The fresh acceptance64
remains sealed. HIP/SYCL builds are unrun debt; no equivalent integration for those
backends is claimed. Broader I2 quality and previously failed disabled raw-token
parity remain open rather than being relabeled as component success.
