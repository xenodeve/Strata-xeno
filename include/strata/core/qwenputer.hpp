#pragma once
#include "strata/core/qwenputer_artifact.hpp"
#include <cuda_runtime.h>
#include <memory>

namespace strata::qwenputer {
struct Input { TokenKind kind; std::uint32_t id; };
struct StepStatus {
    std::uint32_t error = 0, owned = 0;
    Input next{};
};

// One primary-device circuit state. Language-model caches are never passed to it.
// All allocation/upload happens in init; enqueue_step is graph-capturable.
class Executor {
public:
    Executor();
    ~Executor();
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    bool init(const Artifact&, cudaStream_t, std::string& error);
    bool enqueue_reset(cudaStream_t, std::string& error);
    bool enqueue_step(cudaStream_t, std::string& error);
    // Read-only diagnostic of the exact embedding used by enqueue_step.
    bool enqueue_feature_probe(std::uint32_t index, TokenKind, std::uint32_t base,
                               std::uint32_t count, float* output, cudaStream_t, std::string& error);
    Input* device_input() const;
    StepStatus* device_status() const;
    float* device_residual() const;
    float* device_state() const;
    float* device_lags() const;
    std::uint64_t allocated_bytes() const;
    const Digest* artifact_digest() const;
private:
    friend class StateImage;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// An immutable checkpoint until capture() replaces it. Only circuit S, lag,
// current input and prediction status are copied; no language-model state or
// graph handles are present. All operations enqueue copies on one explicit
// created stream that outlives the image; default-stream aliases are refused.
// The caller fences completion before committing state. A failed capture cannot
// be restored. Any restore failure invalidates the caller's circuit session.
class StateImage {
public:
    StateImage();
    ~StateImage();
    StateImage(const StateImage&) = delete;
    StateImage& operator=(const StateImage&) = delete;
    bool init(const Executor&, cudaStream_t, std::string& error);
    bool capture(const Executor&, cudaStream_t, std::string& error);
    bool restore(Executor&, cudaStream_t, std::string& error) const;
    std::uint64_t allocated_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::qwenputer
