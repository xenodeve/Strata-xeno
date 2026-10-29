#pragma once
#include "strata/core/qwenputer_call.hpp"
#include "strata/core/qwenputer_ownership.hpp"
#include "strata/core/qwenputer_runner.hpp"
#include <atomic>
#include <functional>

namespace strata::qwenputer {
// A language-window transaction over an existing native circuit. The caller
// owns the immutable artifact, executor and stream. Lifecycle is serialized.
// The commit callback MUST perform the language verifier commit and its fence;
// it receives the exact consumed-input count selected by the shadow trial.
class Transaction {
public:
    using LanguageCommit = std::function<bool(int, std::string&)>;
    Transaction() = default;
    ~Transaction();
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    bool init(Executor&, const Artifact&, cudaStream_t, RuntimeLimits,
              std::span<const std::uint32_t> stop_ids, std::string& error);
    bool reset(std::string& error);
    bool begin_assistant(std::uint64_t live_from, bool enrolled, std::string& error);
    bool read_history(std::span<const std::uint32_t>, std::string& error);
    bool trial(std::span<const std::uint32_t>, int normal_keep, WindowDecision&, std::string& error);
    bool commit(const LanguageCommit&, std::string& error);
    bool abort_trial(std::string& error);
    // Stages the whole result until native Success. It never invokes language
    // decode, a reference or host arithmetic. Failure invalidates the session.
    bool run_owned(const std::atomic_bool* cancel, std::vector<std::uint32_t>& result,
                   RunReason& reason, std::string& error);
    bool valid() const { return artifact_ && !failed_; }
    bool owned() const { return pending_.kind == CallKind::Eligible || pending_.kind == CallKind::Invalid; }
    std::uint64_t position() const { return position_; }
    std::uint64_t text_steps() const { return text_steps_; }
    std::size_t replay_remaining() const { return ahead_.size() - ahead_cursor_; }
private:
    bool invalidate(std::string& error);
    bool step(std::size_t slot, std::string& error);
    bool fence(std::string& error);
    bool check_steps(std::size_t count, std::string& error);
    Executor* executor_ = nullptr;
    const Artifact* artifact_ = nullptr;
    cudaStream_t stream_ = nullptr;
    StateImage base_;
    Runtime runtime_;
    RuntimeLimits limits_{};
    CallParser parser_;
    std::vector<Input> tokens_;
    std::vector<std::uint32_t> ahead_;
    std::size_t ahead_cursor_ = 0;
    std::uint64_t position_ = 0, live_from_ = 0, text_steps_ = 0;
    WindowDecision decision_{};
    CallEvent pending_{};
    Input* host_inputs_ = nullptr;
    StepStatus* host_status_ = nullptr;
    int device_ = -1;
    bool trial_active_ = false, failed_ = false;
};
} // namespace strata::qwenputer
