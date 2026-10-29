#include "strata/core/qwenputer_call.hpp"
#include <algorithm>
#include <string_view>

namespace strata::qwenputer {
void CallParser::Outer::boundary() {
    if (!fence && marker_run) {
        if (marker_run >= 3) { fence = marker; fence_run = marker_run; opening = true; }
        else if (marker == '`') ticks += marker_run;
        marker = 0; marker_run = 0; line_prefix = false;
    }
    if (ticks) {
        if (!inline_ticks) inline_ticks = ticks;
        else if (inline_ticks == ticks) inline_ticks = 0;
        ticks = 0;
    }
}
bool CallParser::Outer::eligible() const {
    return !fence && !quote && !inline_ticks && !block_quote && !utf_pending && !escaped && indent < 4;
}
void CallParser::Outer::point(std::uint32_t c) {
    if (c == '\n') {
        if (fence) {
            if (!opening && marker == fence && marker_run >= fence_run && closing_tail) {
                fence = 0; fence_run = 0;
            }
        } else boundary();
        indent = marker_run = ticks = 0; marker = 0; line_prefix = true;
        block_quote = opening = escaped = previous_word = false; closing_tail = true;
        return;
    }
    if (fence) {
        if (opening) return;
        if (line_prefix) {
            if (!marker_run && c == ' ' && indent < 4) { ++indent; return; }
            if (indent <= 3 && c == fence && (!marker || marker == fence)) {
                marker = fence; ++marker_run; return;
            }
            line_prefix = false;
        }
        if (c != ' ' && c != '\t') closing_tail = false;
        return;
    }
    if (line_prefix) {
        if (!marker_run && c == '\t') { indent = 4; return; }
        if (!marker_run && c == ' ' && indent < 4) { ++indent; return; }
        if (indent <= 3 && (c == '`' || c == '~') && (!marker || c == marker)) {
            marker = static_cast<std::uint8_t>(c); ++marker_run; return;
        }
        boundary(); line_prefix = false;
        if (fence) return;
        if (c == '>' && indent <= 3) { block_quote = true; return; }
    }
    if (c == '`' && !quote && !escaped) { ++ticks; return; }
    boundary();
    if (inline_ticks) return;
    const bool word = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
    if (escaped) { escaped = false; previous_word = word; return; }
    if (c == '\\') { escaped = true; previous_word = false; return; }
    if (quote) {
        if (quote == c) quote = 0;
    } else if (c == '"' || (c == '\'' && !previous_word)) quote = c;
    else if (c == 0x201c) quote = 0x201d;
    else if (c == 0x2018) quote = 0x2019;
    else if (c == 0xab) quote = 0xbb;
    previous_word = word;
}
void CallParser::Outer::byte(std::uint8_t c) {
    if (utf_pending) {
        if (c < 0x80 || c > 0xbf) { utf_pending = 0; quote = 0xffffffff; return; }
        utf_value = (utf_value << 6) | (c & 63);
        if (!--utf_pending) {
            if (utf_value < utf_min || utf_value > 0x10ffff || (utf_value >= 0xd800 && utf_value <= 0xdfff)) quote = 0xffffffff;
            else point(utf_value);
        }
    } else if (c < 0x80) point(c);
    else if (c >= 0xc2 && c <= 0xdf) { utf_pending = 1; utf_value = c & 31; utf_min = 0x80; }
    else if (c >= 0xe0 && c <= 0xef) { utf_pending = 2; utf_value = c & 15; utf_min = 0x800; }
    else if (c >= 0xf0 && c <= 0xf4) { utf_pending = 3; utf_value = c & 7; utf_min = 0x10000; }
    else quote = 0xffffffff;
}

bool CallParser::init(const Artifact& a, std::span<const std::uint32_t> stops, std::string& error) {
    if (artifact_) { error = "call_already_initialized"; return false; }
    if (!a.has_full_text() || a.text_lengths.size() != a.config[2] || a.text_types.size() != a.config[2]
            || a.config[7] >= a.config[2] || a.config[8] >= a.config[2] || a.config[7] == a.config[8]) {
        error = "call_full_lexicon_required"; return false;
    }
    for (auto id : stops) if (id >= a.config[2] || id == a.config[7] || id == a.config[8]) {
        error = "call_stop_id"; return false;
    }
    stop_ids_.assign(stops.begin(), stops.end()); artifact_ = &a; error.clear(); return true;
}
void CallParser::begin_assistant(bool enrolled) {
    outer_ = {}; body_.clear(); invalid_.clear(); capture_tokens_ = 0;
    capturing_ = false; active_ = true; enrolled_ = enrolled;
}
void CallParser::stop() {
    active_ = capturing_ = false; body_.clear(); invalid_.clear(); capture_tokens_ = 0;
}
bool CallParser::consume(std::uint32_t id, bool live, CallEvent& event, std::string& error) {
    if (!artifact_ || id >= artifact_->config[2]) { error = "call_token_id"; return false; }
    event = {}; error.clear();
    if (!live || !active_) return true;
    if (std::find(stop_ids_.begin(), stop_ids_.end(), id) != stop_ids_.end()) { stop(); return true; }
    const auto& a = *artifact_;
    if (id == a.config[7]) {
        if (capturing_) { invalid_ = "capture_nested_call"; return true; }
        outer_.boundary();
        if (enrolled_ && outer_.eligible()) {
            capturing_ = true; capture_tokens_ = 0; body_.clear(); invalid_.clear(); return true;
        }
    } else if (id == a.config[8] && capturing_) {
        capturing_ = false;
        std::string_view raw(reinterpret_cast<const char*>(body_.data()), body_.size());
        auto candidate = raw;
        while (candidate.starts_with('\n')) candidate.remove_prefix(1);
        if (!candidate.starts_with("<function=calc>")) { event.kind = CallKind::External; return true; }
        if (raw.starts_with('\n')) raw.remove_prefix(1);
        event.kind = CallKind::Invalid;
        if (!invalid_.empty()) { event.refusal = invalid_; return true; }
        if (raw.ends_with('\n')) raw.remove_suffix(1);
        constexpr std::string_view prefix = "<function=calc><parameter=expression>", suffix = "</parameter></function>";
        if (!raw.starts_with(prefix) || !raw.ends_with(suffix) || raw.size() <= prefix.size() + suffix.size()) {
            event.refusal = "native_call_shape"; return true;
        }
        raw.remove_prefix(prefix.size()); raw.remove_suffix(suffix.size());
        for (char c : raw) if (c != '\n') event.expression += c;
        if (event.expression.empty() || event.expression.size() > 255) { event.refusal = "expression_bytes"; return true; }
        for (char c : event.expression) if (std::string_view("0123456789+*/%^()., gcd-").find(c) == std::string_view::npos) {
            event.refusal = "expression_syntax"; event.expression.clear(); return true;
        }
        event.kind = CallKind::Eligible; return true;
    }
    const auto raw = a.token_bytes(id);
    if (capturing_) {
        ++capture_tokens_;
        if (capture_tokens_ > 80) invalid_ = "capture_token_count";
        else if (a.text_types[id] != 1) invalid_ = "capture_reserved_token";
        else if (raw.size() > 16) invalid_ = "capture_token_width";
        else if (std::find(raw.begin(), raw.end(), 0) != raw.end()) invalid_ = "capture_nul";
        if (raw.size() <= 1280 - body_.size()) body_.insert(body_.end(), raw.begin(), raw.end());
        else invalid_ = "capture_byte_count";
    } else for (auto c : raw) outer_.byte(c);
    return true;
}
} // namespace strata::qwenputer
