#pragma once
#include "strata/core/qwenputer.hpp"

namespace strata::qwenputer {
enum class RunReason : std::uint32_t { Ready, Budget, Output, Success, Limit, Cancelled, Fault, Deadline };
struct RuntimeLimits {
    // Every circuit evaluation performed by this Runtime, including its prefix.
    std::uint32_t steps = 200000, output = 4096, chunk = 256;
    // Cumulative wall time from prepare, never renewed by chunk relaunches.
    // The host checks chunk boundaries; one bounded chunk may still be in flight.
    std::uint32_t timeout_ms = 120000;
    bool record_trace = false; // bounded diagnostic ring, never language context
};
struct ChunkResult {
    RunReason reason = RunReason::Ready;
    std::uint32_t error = 0, chunk_steps = 0, total_steps = 0;
    std::uint32_t internal_steps = 0, trace_steps = 0, output_total = 0;
    std::vector<std::uint32_t> text;
    std::vector<Input> trace;
};

// Low-level native runner. The module controller supplies trusted enrollment,
// role/grammar/ownership checks; this API validates capture and execution bounds.
// Artifact, Executor and the supplied stream outlive the Runtime; the Artifact
// stays immutable.
// The runner binds to the Executor's uploaded artifact digest. No model caches
// are accessible. Lifecycle calls are serialized; cancellation may run in parallel
// with an enqueued chunk, with the Runtime kept alive until that caller finishes.
class Runtime {
public:
    Runtime();
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    bool init(Executor&, const Artifact&, RuntimeLimits, cudaStream_t, std::string& error);
    // Complete Text TCO/capture/TCC, one trusted GO seed for program tests, or an
    // empty prefix to continue the executor's already-owned TCC prediction.
    bool prepare(std::span<const Input>, bool reset, std::string& error);
    bool enqueue_chunk(std::uint32_t steps, std::uint32_t output, std::string& error);
    bool collect(ChunkResult&, std::string& error);
    bool request_cancel(std::string& error);
    std::uint64_t allocated_bytes() const;
    std::uint64_t graph_allocation_delta() const;
    std::size_t body_nodes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::qwenputer
