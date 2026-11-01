#pragma once
// A ```python block in the reasoning runs at its closing fence, and its output follows as forced tokens (#279).
// The engine has no tokenizer, so the pack's vocab.json is read here, and only when the sandbox is on.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>
#include "strata/core/igc_sandbox.hpp"

namespace strata::igc::sandbox {

inline constexpr std::size_t kShownOutput = 1024;   // one forced token per byte: a long output costs seconds of decode
inline constexpr int kMaxRuns = 32;
inline constexpr int kMaxBlockTokens = 1024;   // the model can drop its program and reason on inside the open block: one ran to 36607 characters
// Told plainly: on a bare `[nothing printed]` the model decided it cannot execute anything, and looped on that.
inline constexpr std::string_view kNothingPrinted = "[it ran and printed nothing: call your function and print the result]";
// 8 of 32 imports were sympy: told only that it is missing, the model went on by hand for 1820 to 9189 tokens (#279).
// The model's own words before it answers: 4 of 66 replies after a close were the program again, with no sentence (#279).
inline constexpr std::string_view kCloseLine = "Need final with enough detail.\n";
inline constexpr std::string_view kOnlyModules = "[only math, fractions, itertools, functools and collections are here: rewrite with those]";

// The GPT-2 byte table: printable bytes stand for themselves, the others sit from code point 256 up, in order.
inline bool byte_is_itself(uint32_t b) { return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255); }
inline int byte_of_code_point(uint32_t point) {
    if (point < 256) return byte_is_itself(point) ? (int) point : -1;
    uint32_t left = point - 256;
    for (uint32_t b = 0; b < 256; ++b)
        if (!byte_is_itself(b) && !left--) return (int) b;
    return -1;
}

// `Need a short Python program here.`, a blank line and the opening fence. The model does not reach for a program by
// itself: a hint in the prompt gave 0 blocks in thinking on the 4 long questions, and 12 blocks after it on easy ones.
inline constexpr std::array<int32_t, 11> kOpenBlock = {22916, 264, 2716, 12654, 1957, 1532, 13, 271, 71093, 12305, 198};

inline std::string output_block(std::string_view body) {
    std::string block = "\n```output\n";
    block += body;
    block += "\n```\n\n";
    return block;
}

class TokenText {
public:
    bool load(const std::filesystem::path& vocab, std::string& error) {
        std::ifstream file(vocab, std::ios::binary);
        if (!file) {
            error = "igc sandbox: cannot open " + vocab.string();
            return false;
        }
        std::ostringstream read;
        read << file.rdbuf();
        return parse(read.str(), error);
    }
    // The vocabulary is one flat object of string to integer, and the file is ours.
    bool parse(const std::string& json, std::string& error) {
        byte_token_.fill(-1);
        std::size_t at = json.find('{');
        while (at != std::string::npos && at < json.size()) {
            at = json.find_first_of("\"}", at);
            if (at == std::string::npos || json[at] == '}') break;
            std::string key;
            if (!read_key(json, at, key)) {
                error = "igc sandbox: bad key in vocab.json";
                return false;
            }
            at = json.find(':', at);
            if (at == std::string::npos) break;
            int64_t id = 0;
            for (++at; at < json.size() && (json[at] == ' ' || json[at] == '\n' || json[at] == '\r'); ++at) {}
            bool digits = false;
            for (; at < json.size() && json[at] >= '0' && json[at] <= '9'; ++at) {
                id = id * 10 + (json[at] - '0');
                digits = true;
            }
            if (!digits || id < 0 || id > 16 * 1024 * 1024) {
                error = "igc sandbox: bad id in vocab.json";
                return false;
            }
            if ((std::size_t) id >= text_.size()) text_.resize((std::size_t) id + 1);
            if (key.size() == 1) byte_token_[(uint8_t) key[0]] = (int32_t) id;
            longest_ = std::max(longest_, key.size());
            ids_.emplace(key, (int32_t) id);
            text_[(std::size_t) id] = std::move(key);
        }
        for (const int32_t token : byte_token_)
            if (token < 0) {
                error = "igc sandbox: vocab.json lacks a single-byte token";
                return false;
            }
        return true;
    }
    std::string_view text(int32_t token) const {
        return token >= 0 && (std::size_t) token < text_.size() ? std::string_view(text_[(std::size_t) token]) : std::string_view();
    }
    // Longest match, not the merges: close enough for the model to read a fence as a fence. Three single backticks did
    // not close a block to it, and it wrote the closing fence again and again.
    std::vector<int32_t> encode(std::string_view bytes) const {
        std::vector<int32_t> tokens;
        tokens.reserve(bytes.size());
        for (std::size_t at = 0; at < bytes.size();) {
            std::size_t width = std::min(longest_, bytes.size() - at);
            for (; width > 1; --width) {
                const auto found = ids_.find(std::string(bytes.substr(at, width)));
                if (found != ids_.end()) {
                    tokens.push_back(found->second);
                    break;
                }
            }
            if (width <= 1) tokens.push_back(byte_token_[(uint8_t) bytes[at]]);
            at += std::max<std::size_t>(width, 1);
        }
        return tokens;
    }

private:
    static bool read_key(const std::string& json, std::size_t& at, std::string& key) {
        for (++at; at < json.size() && json[at] != '"'; ++at) {
            uint32_t point = (uint8_t) json[at];
            if (json[at] == '\\') {
                if (++at >= json.size()) return false;
                switch (json[at]) {
                    case 'u': {
                        if (at + 4 >= json.size()) return false;
                        point = (uint32_t) std::stoul(json.substr(at + 1, 4), nullptr, 16);
                        at += 4;
                        break;
                    }
                    case 'n': point = '\n'; break;
                    case 'r': point = '\r'; break;
                    case 't': point = '\t'; break;
                    default: point = (uint8_t) json[at];
                }
            } else if (point >= 0x80) {   // UTF-8 in the file
                const int more = point >= 0xF0 ? 3 : point >= 0xE0 ? 2 : 1;
                point &= 0x3F >> more;
                for (int i = 0; i < more; ++i) {
                    if (++at >= json.size()) return false;
                    point = (point << 6) | ((uint8_t) json[at] & 0x3F);
                }
            }
            const int byte = byte_of_code_point(point);
            if (byte >= 0) key.push_back((char) byte);
        }
        return at < json.size();
    }
    std::vector<std::string> text_;
    std::unordered_map<std::string, int32_t> ids_;
    std::size_t longest_ = 1;
    std::array<int32_t, 256> byte_token_{};
};

// Finds the block in the bytes of the reasoning, since the tokenizer can glue a fence to what stands before it.
// A bare ``` stays out: all 83 fences in 7420 reasonings have no language, and they draft the answer.
class Fence {
public:
    std::optional<std::string> step(std::string_view bytes) {
        std::size_t from = inside_ && text_.size() > 3 ? text_.size() - 3 : 0;
        text_ += bytes;
        if (!inside_) {
            for (const std::string_view open : {"```python\n", "```py\n", "```Python\n"}) {
                const std::size_t at = text_.find(open);
                if (at == std::string::npos) continue;
                text_.erase(0, at + open.size() - 1);   // the block's own newline: an empty block closes on it
                inside_ = true;
                from = 0;
                break;
            }
            if (!inside_) {
                if (text_.size() > 16) text_.erase(0, text_.size() - 16);
                return std::nullopt;
            }
        }
        const std::size_t close = text_.find("\n```", from);
        if (close == std::string::npos) return std::nullopt;
        std::string program = text_.substr(0, close + 1);
        text_.clear();
        inside_ = false;
        return program;
    }
    bool inside() const { return inside_; }
    void leave() {
        text_.clear();
        inside_ = false;
    }
    // The opening fence was forced, so this class never read it.
    void enter() {
        text_ = "\n";
        inside_ = true;
    }

private:
    std::string text_;
    bool inside_ = false;
};

// Blocks of one reasoning share their names, as notebook cells do: each run replays the blocks that finished and drops
// what they printed. The language has no clock or randomness, so a replay gives the same state.
class Session {
public:
    // 200M steps is about 0.4 s, the time of some 30 decoded tokens, and a program stopped short costs a rewrite. The default 2M is 4 ms.
    explicit Session(Limits limits = {}) : limits_(limits) {
        limits_.output_bytes = std::max(limits_.output_bytes, std::size_t(64 * 1024));
        limits_.steps = std::max<uint64_t>(limits_.steps, 200000000);
    }
    bool spent() const { return runs_ >= kMaxRuns; }
    // How many times this program was run before, counted so a repeat can end the thinking.
    int seen(std::string_view program) const { return (int) std::count(programs_.begin(), programs_.end(), program); }
    int runs() const { return runs_; }
    bool printed() const { return printed_last_; }   // the last run finished and said something
    uint64_t steps() const { return steps_; }
    // What the reasoning reads next, right after the closing fence.
    std::string run(std::string_view program, const std::atomic<bool>* cancel = nullptr) {
        ++runs_;
        programs_.emplace_back(program);
        std::string source = done_;
        source += program;
        // A fresh stack: the interpreter's budget of 640 KiB counts from run(), and the decode loop already sits deep in its own.
        Result result;
        std::thread([&] { result = sandbox::run(source, limits_, cancel); }).join();
        steps_ += result.steps;
        std::string body = result.output.size() > printed_ ? result.output.substr(printed_) : std::string();
        bool cut = result.truncated;
        if (body.size() > kShownOutput) {
            body.resize(kShownOutput);
            cut = true;
        }
        if (cut) body += "\n[output cut]";
        printed_last_ = result.status == Status::Ok && !body.empty();
        if (result.status == Status::Ok) {
            done_ = std::move(source);
            printed_ = result.output.size();
            if (body.empty()) body = kNothingPrinted;
        } else {
            if (!body.empty()) body += '\n';
            body += result.error;
            if (result.error.starts_with("ModuleNotFoundError")) {
                body += '\n';
                body += kOnlyModules;
            }
        }
        while (!body.empty() && body.back() == '\n') body.pop_back();
        return output_block(body);
    }

private:
    Limits limits_;
    std::string done_;
    std::vector<std::string> programs_;
    std::size_t printed_ = 0;
    bool printed_last_ = false;
    int runs_ = 0;
    uint64_t steps_ = 0;
};
}  // namespace strata::igc::sandbox
