#include "strata/core/qwenputer_transaction.hpp"
#include "strata/timeline.hpp"
#include <algorithm>
#include <exception>
#include <limits>

namespace strata::qwenputer {
namespace {
constexpr std::size_t max_window = 256;
bool checked(cudaError_t result, std::string& error) {
    if (result == cudaSuccess) return true;
    error = cudaGetErrorString(result); return false;
}
}
Transaction::~Transaction() {
    int previous = -1;
    cudaGetDevice(&previous);
    if (device_ >= 0) cudaSetDevice(device_);
    if (stream_) cudaStreamSynchronize(stream_);
    if (host_inputs_) cudaFreeHost(host_inputs_);
    if (host_status_) cudaFreeHost(host_status_);
    if (previous >= 0 && previous != device_) cudaSetDevice(previous);
}
bool Transaction::invalidate(std::string& error) {
    failed_ = true; if (error.empty()) error = "native_transaction_invalid"; return false;
}
bool Transaction::fence(std::string& error) {
    int current = -1;
    if (!checked(cudaGetDevice(&current), error) || current != device_) {
        error = "transaction_device"; return invalidate(error);
    }
    return checked(cudaStreamSynchronize(stream_), error) || invalidate(error);
}
bool Transaction::step(std::size_t slot, std::string& error) {
    if (!checked(cudaMemcpyAsync(executor_->device_input(), host_inputs_ + slot, sizeof(Input), cudaMemcpyHostToDevice, stream_), error)
            || !executor_->enqueue_step(stream_, error)
            || !checked(cudaMemcpyAsync(host_status_ + slot, executor_->device_status(), sizeof(StepStatus), cudaMemcpyDeviceToHost, stream_), error))
        return invalidate(error);
    ++text_steps_; return true;
}
bool Transaction::check_steps(std::size_t count, std::string& error) {
    if (!fence(error)) return false;
    for (std::size_t i = 0; i < count; ++i) if (host_status_[i].error) {
        error = "native_text_step_fault"; return invalidate(error);
    }
    return true;
}
bool Transaction::init(Executor& executor, const Artifact& artifact, cudaStream_t stream, RuntimeLimits limits,
                       std::span<const std::uint32_t> stops, std::string& error) try {
    if (artifact_ || failed_) { error = "transaction_already_initialized"; return false; }
    if (!executor.artifact_digest() || *executor.artifact_digest()!=artifact.digest) {
        error="executor_artifact_binding"; return invalidate(error);
    }
    if (!stream || stream==cudaStreamLegacy || stream==cudaStreamPerThread) {
        error="transaction_explicit_stream_required"; return invalidate(error);
    }
    if (!parser_.init(artifact, stops, error) || !runtime_.init(executor, artifact, limits, stream, error)
            || !base_.init(executor, stream, error)) return invalidate(error);
    if (!checked(cudaGetDevice(&device_), error)) return invalidate(error);
    executor_ = &executor; artifact_ = &artifact; stream_ = stream; limits_ = limits;
    if (!checked(cudaHostAlloc(reinterpret_cast<void**>(&host_inputs_), max_window * sizeof(Input), cudaHostAllocDefault), error)
            || !checked(cudaHostAlloc(reinterpret_cast<void**>(&host_status_), max_window * sizeof(StepStatus), cudaHostAllocDefault), error))
        return invalidate(error);
    return reset(error);
} catch (const std::exception& e) { error=e.what(); return invalidate(error); }
bool Transaction::reset(std::string& error) {
    if (!valid() || trial_active_) { error = "transaction_busy_or_invalid"; return false; }
    if (!executor_->enqueue_reset(stream_, error) || !fence(error)) return invalidate(error);
    parser_.stop(); pending_ = {}; ahead_.clear(); ahead_cursor_ = 0;
    position_ = text_steps_ = 0; live_from_ = std::numeric_limits<std::uint64_t>::max();
    error.clear(); return true;
}
bool Transaction::begin_assistant(std::uint64_t live_from, bool enrolled, std::string& error) {
    if (!valid() || trial_active_ || owned() || replay_remaining()) { error="transaction_busy_or_invalid"; return false; }
    parser_.begin_assistant(enrolled); live_from_ = live_from; error.clear(); return true;
}
bool Transaction::read_history(std::span<const std::uint32_t> ids, std::string& error) try {
    timeline::Span span("qwenputer.history", static_cast<std::int64_t>(ids.size()));
    if (!valid() || trial_active_ || owned() || replay_remaining() || ids.size() > 1048576
            || position_ > std::numeric_limits<std::uint64_t>::max() - ids.size()) { error="transaction_history_bounds"; return false; }
    for (auto id : ids) if (id >= artifact_->config[2]) { error="call_token_id"; return false; }
    parser_.stop();
    for (std::size_t start=0; start<ids.size(); start+=max_window) {
        const auto count=std::min(max_window, ids.size()-start);
        for (std::size_t i=0; i<count; ++i) host_inputs_[i]={TokenKind::Text,ids[start+i]};
        for (std::size_t i=0; i<count; ++i) if (!step(i,error)) return false;
        if (!check_steps(count,error)) return false;
        position_+=count;
    }
    error.clear(); return true;
} catch (const std::exception& e) { error=e.what(); return invalidate(error); }
bool Transaction::trial(std::span<const std::uint32_t> ids, int normal_keep, WindowDecision& output, std::string& error) try {
    timeline::Span span("qwenputer.shadow_trial",static_cast<std::int64_t>(ids.size()),normal_keep);
    if (!valid() || trial_active_ || owned() || ids.empty() || ids.size()>max_window || normal_keep<1
            || static_cast<std::size_t>(normal_keep)>ids.size()
            || position_>std::numeric_limits<std::uint64_t>::max()-ids.size()) { error="transaction_window_bounds"; return false; }
    for (auto id : ids) if (id>=artifact_->config[2]) { error="call_token_id"; return false; }
    tokens_.clear(); for (auto id:ids) tokens_.push_back({TokenKind::Text,id});
    if (!base_.capture(*executor_,stream_,error)) return invalidate(error);
    auto shadow=parser_; auto cursor=ahead_cursor_; int takeover=-1;
    std::size_t evaluated=0;
    for (std::size_t i=0; i<ids.size(); ++i) {
        if (cursor<ahead_.size()) {
            if (ahead_[cursor++]!=ids[i]) { error="native_replay_identity"; return invalidate(error); }
            continue;
        }
        host_inputs_[evaluated]=tokens_[i];
        if (!step(evaluated++,error)) return false;
        CallEvent event;
        if (!shadow.consume(ids[i],position_+i>=live_from_,event,error)) return invalidate(error);
        if (event.kind==CallKind::Eligible || event.kind==CallKind::Invalid) { takeover=static_cast<int>(i); break; }
    }
    if (!check_steps(evaluated,error)) return false;
    if (!decide_window(static_cast<int>(ids.size()),normal_keep,takeover,decision_,error)) return invalidate(error);
    output=decision_; trial_active_=true; error.clear(); return true;
} catch (const std::exception& e) { error=e.what(); return invalidate(error); }
bool Transaction::abort_trial(std::string& error) {
    if (!valid() || !trial_active_) { error="transaction_no_trial"; return false; }
    if (!base_.restore(*executor_,stream_,error) || !fence(error)) return invalidate(error);
    trial_active_=false; tokens_.clear(); error.clear(); return true;
}
bool Transaction::commit(const LanguageCommit& language_commit, std::string& error) try {
    timeline::Span span("qwenputer.accepted_replay",decision_.consumed_inputs);
    if (!valid() || !trial_active_ || !language_commit) { error="transaction_no_trial"; return false; }
    if (!language_commit(decision_.consumed_inputs,error)) return invalidate(error);
    if (!base_.restore(*executor_,stream_,error)) return invalidate(error);
    auto accepted=parser_; auto cursor=ahead_cursor_; CallEvent last; std::size_t evaluated=0;
    for (int i=0; i<decision_.consumed_inputs; ++i) {
        const auto id=tokens_[static_cast<std::size_t>(i)].id;
        if (cursor<ahead_.size()) {
            if (ahead_[cursor++]!=id) { error="native_replay_identity"; return invalidate(error); }
            continue;
        }
        host_inputs_[evaluated]={TokenKind::Text,id};
        if (!step(evaluated++,error) || !accepted.consume(id,position_+static_cast<std::uint64_t>(i)>=live_from_,last,error))
            return invalidate(error);
    }
    if (!check_steps(evaluated,error)) return false;
    if (decision_.takeover_row>=0 && last.kind!=CallKind::Eligible && last.kind!=CallKind::Invalid) {
        error="native_grammar_replay_mismatch"; return invalidate(error);
    }
    if (decision_.takeover_row>=0 && last.kind==CallKind::Eligible
            && (!evaluated || !host_status_[evaluated-1].owned)) { error="native_owner_mismatch"; return invalidate(error); }
    if (decision_.takeover_row>=0) pending_=last;
    parser_=std::move(accepted); ahead_cursor_=cursor; position_+=decision_.consumed_inputs;
    if (ahead_cursor_==ahead_.size()) { ahead_.clear(); ahead_cursor_=0; }
    tokens_.clear(); trial_active_=false; error.clear(); return true;
} catch (const std::exception& e) { error=e.what(); return invalidate(error); }
bool Transaction::run_owned(const std::atomic_bool* cancel, std::vector<std::uint32_t>& output, RunReason& reason, std::string& error) try {
    output.clear(); reason=RunReason::Fault;
    if (!valid() || trial_active_ || !owned() || replay_remaining()) { error="transaction_not_owned"; return false; }
    if (pending_.kind==CallKind::Invalid) { error=pending_.refusal; return invalidate(error); }
    if (cancel && cancel->load(std::memory_order_relaxed)) { reason=RunReason::Cancelled; error="cancelled"; return invalidate(error); }
    if (!runtime_.prepare({},false,error)) return invalidate(error);
    std::vector<std::uint32_t> staged;
    for (;;) {
        if (cancel && cancel->load(std::memory_order_relaxed) && !runtime_.request_cancel(error)) return invalidate(error);
        if (!runtime_.enqueue_chunk(std::min(limits_.chunk,128u),128,error)) return invalidate(error);
        ChunkResult result;
        if (!runtime_.collect(result,error)) return invalidate(error);
        reason=result.reason;
        if (staged.size()+result.text.size()>limits_.output) { error="native_output_bound"; return invalidate(error); }
        staged.insert(staged.end(),result.text.begin(),result.text.end());
        if (reason==RunReason::Success) {
            ahead_=staged; ahead_cursor_=0; pending_={}; output=std::move(staged); error.clear(); return true;
        }
        if (reason!=RunReason::Budget && reason!=RunReason::Output) {
            error="native_terminal_"+std::to_string(static_cast<unsigned>(reason)); return invalidate(error);
        }
    }
} catch (const std::exception& e) { error=e.what(); return invalidate(error); }
} // namespace strata::qwenputer
