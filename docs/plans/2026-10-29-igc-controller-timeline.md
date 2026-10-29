# IGC controller host timeline scopes

Issue #276, G8 measurement prerequisite. The existing timeline reports saved
activation observation and generic startup/decode phases, but has no explicit
learned-controller lifecycle report. Add the analyzer tests first, then host
scopes where the actual wrapper already performs work.

Record literal names under `igc.controller`: load_host, bind, prepare, begin,
stage, cancel, audit and release. Report attempts, inclusive host milliseconds
and maximum host duration per name. Failed/refused calls are still attempts;
counts must not become successful invocation counts. Never sum overlapping
host/GPU/parent scopes into a synthetic total.

Do not put a host span inside the captured head and call it device execution.
Actual arithmetic/readout/output kernels need a separate compatible GPU timing
measurement when the live path is profiled. These scopes change no numerical
operation, ownership, allocation policy, graph addresses, ABI, engine protocol
or Hub setting. With the feature hard-off the wrapper is not created; with
profiling off its existing optional calls use the normal timeline branch.

Validation: synthetic hand-computed analyzer/CLI tests and unchanged-trace
behavior, full CUDA 89;120 build, actual wrapper lifecycle controls on old saved
activations with timeline OFF/ON, and a real trace parsed/exported by the normal
analyzer. The wrapper test flushes after its controller/stream objects are
released so teardown is visible. Model content binding is a sequential read,
not a language-model allocation. Keep the fresh acceptance64 closed. No latency
improvement or profiler overhead claim from a single OFF/ON sequence; practical
Hub performance and hard-off raw-token parity remain their own gates.
