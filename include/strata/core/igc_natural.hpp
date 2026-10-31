#pragma once
// Firing on natural language: a number the model starts in plain words is checked against the sidecar's proposal.
// The two leading digits stay the model's and must agree, which is the gate measured on reasoning traces (#276). A
// later digit that differs is replaced with the computed value. A number closing an expression the parser reads is the
// parser's; one it could not read, like `*13,000 =`, is checked here.

#include "strata/core/igc_inline.hpp"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace strata::igc {

template <class Model>
class NaturalFirer {
public:
    static constexpr int32_t kImStart = 248045, kImEnd = 248046, kUser = 872, kNewline = 198;
    static constexpr int kGateDigits = 2;

    explicit NaturalFirer(Model& model, float threshold = .5f) : model_(model), threshold_(threshold) {}

    // The question's tokens from a prompt, as training saw them: no template, then a line break.
    void begin(std::span<const int64_t> prompt) {
        model_.reset();
        tokens_.clear();
        table_.clear();
        digits_.clear();
        expected_.clear();
        active_ = true;
        previous_comma_ = false;
        size_t start = 0, end = prompt.size();
        for (size_t i = prompt.size(); i-- > 2;)
            if (prompt[i - 2] == kImStart && prompt[i - 1] == kUser && prompt[i] == kNewline) { start = i + 1; break; }
        for (size_t i = start; i < prompt.size(); ++i)
            if (prompt[i] == kImEnd) { end = i; break; }
        for (size_t i = start; i < end && active_; ++i) feed(static_cast<int32_t>(prompt[i]));
        if (active_) feed(kNewline);
    }

    // One accepted token. Returns the digits to deliver after `token` when it fires; `token` may be replaced.
    std::vector<int32_t> step(int32_t& token) {
        if (!active_) return {};
        if (!digit(token)) {
            expected_.clear();
            feed(token);
            return {};
        }
        if (digits_.empty()) propose();
        std::vector<int32_t> rest;
        if (!expected_.empty()) {
            const size_t at = digits_.size();
            if (at >= expected_.size()) expected_.clear();
            else if (expected_[at] != token) {
                if (at < kGateDigits) expected_.clear();
                else {
                    token = expected_[at];
                    rest.assign(expected_.begin() + static_cast<std::ptrdiff_t>(at) + 1, expected_.end());
                    ++fires_;
                }
            }
        }
        feed(token);
        return rest;
    }

    int64_t fires() const { return fires_; }
    int64_t proposals() const { return proposals_; }
    float last_confidence() const { return last_confidence_; }

private:
    struct Entry {
        std::optional<int64_t> value;
        int end = 0;
    };
    static constexpr int kMaxEntries = 64;

    static bool digit(int32_t token) { return token >= 15 && token <= 24; }

    // After `a op b =` the parser has the number, when it can read the expression.
    bool parser_owns() const {
        size_t i = tokens_.size();
        while (i > 0 && (tokens_[i - 1] == 220 || tokens_[i - 1] == 3 || tokens_[i - 1] == 393)) --i;
        if (i == 0 || (tokens_[i - 1] != 28 && tokens_[i - 1] != 283)) return false;
        return parse_inline(std::span<const int32_t>(tokens_).first(i)).has_value();
    }

    void propose() {
        expected_.clear();
        if (table_.size() < 1 || parser_owns()) return;
        std::vector<int> ends;
        const size_t first = table_.size() > kMaxEntries ? table_.size() - kMaxEntries : 0;
        for (size_t i = first; i < table_.size(); ++i) ends.push_back(table_[i].end);
        const auto proposal = model_.propose(ends);
        ++proposals_;
        last_confidence_ = proposal.confidence;
        if (proposal.confidence < threshold_) return;
        const auto& a = table_[first + proposal.left].value;
        const auto& b = table_[first + proposal.right].value;
        if (!a || !b) return;
        const std::optional<int64_t> value = apply(proposal.operation, *a, *b);
        if (!value || *value < 0) return;
        for (const char c : std::to_string(*value)) expected_.push_back(15 + (c - '0'));
    }

    static std::optional<int64_t> apply(int operation, int64_t a, int64_t b) {
        constexpr int64_t kLimit = 1000000000;
        if (a > kLimit || b > kLimit) return std::nullopt;
        switch (operation) {
            case 0: return a + b;
            case 1: return a - b;
            case 2: return a * b;
            case 3: if (b != 0 && a % b == 0) return a / b; return std::nullopt;
            default: return std::nullopt;
        }
    }

    void feed(int32_t token) {
        if (model_.tokens() >= model_.capacity()) { active_ = false; return; }
        const bool is_digit = digit(token);
        if (!is_digit && !digits_.empty()) close();
        if (is_digit) {
            if (digits_.empty() && previous_comma_ && !table_.empty() && table_.back().value) {
                // `1,764`: the part before the comma is the same number
                for (char c : std::to_string(*table_.back().value)) digits_.push_back(15 + (c - '0'));
                table_.pop_back();
            }
            digits_.push_back(token);
        }
        previous_comma_ = token == 11 && !tokens_.empty() && digit(tokens_.back());
        decimal_ = is_digit ? decimal_ : (token == 13 && !tokens_.empty() && digit(tokens_.back()));
        tokens_.push_back(token);
        model_.push(token);
        last_end_ = model_.tokens() - 1;
    }

    void close() {
        int64_t value = 0;
        bool fits = digits_.size() <= 15 && !decimal_;
        if (fits) for (int32_t d : digits_) value = value * 10 + (d - 15);
        Entry entry;
        if (fits) entry.value = value;
        entry.end = last_end_;
        table_.push_back(entry);
        digits_.clear();
    }

    Model& model_;
    float threshold_;
    std::vector<int32_t> tokens_, digits_, expected_;
    std::vector<Entry> table_;
    bool active_ = false, previous_comma_ = false, decimal_ = false;
    int last_end_ = 0;
    int64_t fires_ = 0, proposals_ = 0;
    float last_confidence_ = 0;
};

}  // namespace strata::igc
