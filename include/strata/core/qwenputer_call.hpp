#pragma once
#include "strata/core/qwenputer_artifact.hpp"

namespace strata::qwenputer {
enum class CallKind { None, External, Eligible, Invalid };
struct CallEvent {
    CallKind kind = CallKind::None;
    std::string expression, refusal;
};

// Copyable CPU grammar shadow; it performs no arithmetic. The immutable bound
// artifact outlives it. Only live, explicitly enrolled assistant tokens enter a
// capture. History, internal result replay and tokens after stop never do.
class CallParser {
public:
    bool init(const Artifact&, std::span<const std::uint32_t> stop_ids, std::string& error);
    void begin_assistant(bool enrolled);
    void stop();
    bool consume(std::uint32_t id, bool live_assistant, CallEvent&, std::string& error);
private:
    struct Outer {
        unsigned indent = 0, marker_run = 0, fence_run = 0, ticks = 0, inline_ticks = 0;
        unsigned utf_pending = 0, utf_value = 0, utf_min = 0;
        std::uint8_t marker = 0, fence = 0;
        std::uint32_t quote = 0;
        bool line_prefix = true, block_quote = false, opening = false;
        bool closing_tail = true, escaped = false, previous_word = false;
        void point(std::uint32_t);
        void byte(std::uint8_t);
        void boundary();
        bool eligible() const;
    } outer_;
    const Artifact* artifact_ = nullptr;
    std::vector<std::uint32_t> stop_ids_;
    std::vector<std::uint8_t> body_;
    std::string invalid_;
    unsigned capture_tokens_ = 0;
    bool active_ = false, enrolled_ = false, capturing_ = false;
};
} // namespace strata::qwenputer
