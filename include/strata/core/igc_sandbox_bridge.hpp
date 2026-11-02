#pragma once
#include <functional>
// A ```python block in the reasoning runs at its closing fence, and its output follows as forced tokens (#279).
// The engine has no tokenizer, so the pack's vocab.json is read here, and only when the sandbox is on.
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
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
inline constexpr int kMaxRuns = 128;
inline constexpr int kMaxBlockTokens = 2048;   // the model can drop its program and reason on inside the open block: one ran to 36607 characters
// Told plainly: on a bare `[nothing printed]` the model decided it cannot execute anything, and looped on that.
inline constexpr std::string_view kNothingPrinted = "[it ran and printed nothing: call your function and print the result]";
// 8 of 32 imports were sympy: told only that it is missing, the model went on by hand for 1820 to 9189 tokens (#279).
// The model's own words before it answers: 4 of 66 replies after a close were the program again, with no sentence (#279).
inline constexpr std::string_view kCloseLine = "Need final with enough detail.\n";
inline constexpr std::string_view kOnlyModules = "[only math, fractions, itertools, functools, collections, heapq, bisect and sys are here: rewrite with those]";
// Told to call its function, the model did not in 36 of 36 blocks, with a whole draft in 17 of 19 scored (#279). So the call is opened
// for it, with the name: given a bare `print(`, 12 of 22 runs wrote `print(1)`, and the thinking closed on that.
inline constexpr std::string_view kCallHead = "print(";
// The call as a test: the comment says where the second argument comes from, because `expect` is in no prompt.
inline constexpr std::string_view kExpectHead = "# result, then the expected output from the question\nexpect(";
// Said in the model's voice after a mismatch. Self-Debugging: a test result with the model's own explanation did best.
inline constexpr std::string_view kMismatchLine = "Wait, this does not match, so my approach is wrong somewhere. Why?\n";
// Before the thinking ends, where the model has its answer and nothing has been cut short. START's own words came back as a comment in an
// empty block, 8 times over, on the question that had 4 drafts written (#288): so the line says what to run, and what the block is for.
// `Let` and not `Wait`: at this point the first raised a small model's score and the second lowered it (2606.16934).
inline constexpr std::string_view kGateLine = "Let me run my code on the examples before I answer.\n\n```python\n# the whole program I am about to answer with\n";
// With a draft already defined the block only calls it.
inline constexpr std::string_view kGateCallLine = "Let me run my code on the examples before I answer.\n\n```python\n";
// A match is not proof, and a model tests to confirm: asked to look for what breaks it, it finds more (Failing to Falsify, 2604.02485).
inline constexpr std::string_view kMatchLine = "The examples match. What input would break this?\n";
// An oracle written from the statement before any idea of how to solve it. 40 % of the reasoning is work a new small program could settle, and it comes in no
// word a rule can find: `Wait` stands 0.04 times in 1,000 tokens, and windows with a phrase are no likelier to be that work than windows without (#288). A forced
// block in the old words held brute force 3 times in 181, because it never asked for it.
inline constexpr std::string_view kBruteHead = "def brute(";
inline constexpr std::string_view kBruteLine = "Let me write a brute force straight from the statement and run it on the examples.\n\n```python\ndef brute(";
// A draft that passes the examples, set against the oracle where the examples stop.
inline constexpr std::string_view kCompareHead = "import random\n";
inline constexpr std::string_view kCompareLine = "The examples match. Let me compare it with the brute force on random small inputs.\n\n```python\nimport random\n";
inline constexpr int kMaxGates = 3;
// stdin holds only what the program fed itself, so a program that read nothing would look like a program that did nothing.
inline constexpr std::string_view kNoStdin = "[stdin was empty: put feed(\"...\") before the code, with an example from the question]";
inline constexpr std::string_view kBlockOpen = "```python\n";

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
    // A bare fence holds the model's own draft: 19 of 20 hard questions wrote one, and 66 % of their tokens came after the first (#288).
    void read_drafts(bool on) { drafts_ = on; }
    std::optional<std::string> step(std::string_view bytes) {
        std::size_t from = inside_ && text_.size() > 3 ? text_.size() - 3 : 0;
        text_ += bytes;
        if (!inside_ && drafts_) {
            if (!open_any()) return std::nullopt;
            from = 0;
        }
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
    // The opening fence was forced, so this class never read it, nor `head` after it.
    void enter(std::string_view head = {}) {
        text_ = "\n";
        text_ += head;
        inside_ = true;
        kind_ = Kind::Code;
    }
    // Whether what just closed is a program to run. Another language never is, and a bare fence only when it holds a whole draft.
    bool runs(std::string_view program) const {
        if (kind_ == Kind::Other) return false;
        if (kind_ == Kind::Code) return true;
        return (program.find("\ndef ") != std::string_view::npos || program.find("\nclass ") != std::string_view::npos)
            && std::count(program.begin(), program.end(), '\n') >= 4;
    }
    bool code() const { return kind_ == Kind::Code; }   // the model means to run it, so a block too long to run is told so

private:
    enum class Kind { Code, Bare, Other };
    // Pairs every fence, so a closing fence is never read as an opening one. A fence in a sentence, ```python ...```, opens nothing.
    bool open_any() {
        for (std::size_t at = text_.find("```"); at != std::string::npos; at = text_.find("```", at + 3)) {
            std::size_t word = at + 3;
            while (word < text_.size() && word - at < 19 && (std::isalnum((unsigned char) text_[word]) || text_[word] == '+' || text_[word] == '#' || text_[word] == '-')) ++word;
            if (word == text_.size()) {   // the line is not finished yet
                text_.erase(0, at);
                return false;
            }
            if (text_[word] != '\n') continue;
            const std::string_view language = std::string_view(text_).substr(at + 3, word - at - 3);
            kind_ = language.empty() ? Kind::Bare : language == "python" || language == "py" || language == "Python" ? Kind::Code : Kind::Other;
            text_.erase(0, word);   // the block's own newline: an empty block closes on it
            inside_ = true;
            return true;
        }
        if (text_.size() > 16) text_.erase(0, text_.size() - 16);
        return false;
    }
    std::string text_;
    bool inside_ = false, drafts_ = false;
    Kind kind_ = Kind::Code;
};

// The function to call: the last one the block defined at its top level, or a class's last method when the block is a
// `class Solution`. Helpers come first, so this is the one the block is for.
inline std::string last_def(std::string_view program) {
    std::string name, in_class;
    for (std::size_t at = 0; at < program.size();) {
        std::size_t end = program.find('\n', at);
        if (end == std::string_view::npos) end = program.size();
        const std::string_view line = program.substr(at, end - at);
        const std::size_t paren = line.find('(');
        if (line.starts_with("def ") && paren != std::string_view::npos) {
            name = std::string(line.substr(4, paren - 4));
            in_class.clear();
        } else if (line.starts_with("class ")) {
            in_class = std::string(line.substr(6, line.find_first_of("(:") - 6));
            name.clear();
        } else if (!in_class.empty() && line.starts_with("    def ") && paren != std::string_view::npos && !line.starts_with("    def __")) {
            name = in_class + "()." + std::string(line.substr(8, paren - 8));
        }
        at = end + 1;
    }
    return name;
}
// What the forced block opens with, after its fence.
inline std::string call_head(std::string_view program, bool expect = false) {
    std::string head(expect ? kExpectHead : kCallHead);
    const std::string name = last_def(program);
    if (!name.empty()) head += name + "(";
    return head;
}

// Whether a program is the oracle or a test of it, so that a pass here says nothing of the model's answer.
inline bool about_the_oracle(std::string_view program) {
    return last_def(program) == "brute" || program.find("expect(brute(") != std::string_view::npos;
}

// A block that bound nothing leaves nothing to call.
inline bool binds_a_name(std::string_view program) {
    return program.find("def ") != std::string_view::npos || program.find('=') != std::string_view::npos;
}

// `None` is an unfinished draft called anyway, not a result: the close took 2 of the 4 rows it lost on one (#279).
inline bool says_something(std::string_view body) {
    while (body.starts_with("None\n")) body.remove_prefix(5);
    return !body.empty();
}

// A run replays the blocks before it, so the interpreter counts from the first of them, and a 2-line block reported
// `SyntaxError (line 23)` (#279). The model sees one block: the number is rebased, and the line itself says what failed.
inline std::string name_the_line(std::string_view error, std::string_view source, int replayed) {
    const std::size_t at = error.rfind(" (line ");
    if (at == std::string_view::npos || error.back() != ')') return std::string(error);
    int line = 0;
    for (const char c : error.substr(at + 7, error.size() - at - 8)) {
        if (c < '0' || c > '9') return std::string(error);
        line = line * 10 + (c - '0');
    }
    std::size_t from = 0;
    for (int i = 1; i < line && from != std::string_view::npos; ++i) {
        from = source.find('\n', from);
        if (from != std::string_view::npos) ++from;
    }
    std::string_view text = from == std::string_view::npos ? std::string_view() : source.substr(from, source.find('\n', from) - from);
    text.remove_prefix(std::min(text.find_first_not_of(" \t"), text.size()));
    std::string named(error.substr(0, at));
    named += line > replayed ? " (line " + std::to_string(line - replayed) + ")" : std::string(" (in an earlier block)");
    if (!text.empty()) {
        named += "\n    ";
        named += text;
    }
    return named;
}

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
    bool quiet() const { return quiet_last_; }       // the last run finished and said nothing
    bool mismatch() const { return mismatch_last_; } // the last run's `expect` found a wrong result
    bool matched() const { return matched_last_; }   // the last run's every `expect` held
    uint64_t steps() const { return steps_; }
    // Real Python, for what the interpreter refuses: (code, a fresh kernel, body, ok), false when nothing answered. The refusal itself
    // chooses the tier, and a turn that moved stays there, because the kernel holds its names.
    using Exec = std::function<bool(std::string_view, bool, std::string&, bool&)>;
    void set_exec(Exec exec) { exec_ = std::move(exec); }
    // What the reasoning reads next, right after the closing fence.
    std::string run(std::string_view program, const std::atomic<bool>* cancel = nullptr) {
        ++runs_;
        programs_.emplace_back(program);
        if (outside_) return run_outside(program);
        const int replayed = (int) std::count(done_.begin(), done_.end(), '\n') + (program.starts_with('\n') ? 1 : 0);
        std::string source = done_;
        source += program;
        // A fresh stack: the interpreter's budget of 640 KiB counts from run(), and the decode loop already sits deep in its own.
        Result result;
        std::thread([&] { result = sandbox::run(source, limits_, cancel); }).join();
        steps_ += result.steps;
        if (exec_ && (result.status == Status::Unsupported || result.error.starts_with("ModuleNotFoundError"))) {
            outside_ = true;
            return run_outside(source);
        }
        std::string body = result.output.size() > printed_ ? result.output.substr(printed_) : std::string();
        bool cut = result.truncated;
        if (body.find_first_not_of(" \t\r\n") == std::string::npos) body.clear();   // `print()` says nothing
        if (body.size() > kShownOutput) {
            body.resize(kShownOutput);
            cut = true;
        }
        if (cut) body += "\n[output cut]";
        printed_last_ = result.status == Status::Ok && says_something(body);
        quiet_last_ = result.status == Status::Ok && body.empty() && !result.stdin_short;
        mismatch_last_ = result.status == Status::Ok && body.find("MISMATCH: ") != std::string::npos;
        matched_last_ = result.status == Status::Ok && !mismatch_last_ && body.starts_with("match: ");
        if (result.status == Status::Ok) {
            done_ = std::move(source);
            printed_ = result.output.size();
            if (body.empty() && result.stdin_short) body = kNoStdin;
            else if (body.empty()) body = kNothingPrinted;
        } else {
            if (!body.empty()) body += '\n';
            body += name_the_line(result.error, source, replayed);
            if (result.error.starts_with("ModuleNotFoundError")) {
                body += '\n';
                body += kOnlyModules;
            }
            if (result.stdin_short) {
                body += '\n';
                body += kNoStdin;
            }
        }
        while (!body.empty() && body.back() == '\n') body.pop_back();
        return output_block(body);
    }

private:
    std::string run_outside(std::string_view code) {
        std::string body;
        bool ok = false;
        const bool fresh = !kernel_used_;
        kernel_used_ = true;
        if (!exec_(code, fresh, body, ok)) {
            body = "[the sandbox did not answer]";
            ok = false;
        }
        if (body.find_first_not_of(" \t\r\n") == std::string::npos) body.clear();
        printed_last_ = ok && says_something(body);
        quiet_last_ = ok && body.empty();
        mismatch_last_ = ok && body.find("MISMATCH: ") != std::string::npos;
        matched_last_ = ok && !mismatch_last_ && body.starts_with("match: ");
        if (body.size() > kShownOutput) {
            body.resize(kShownOutput);
            body += "\n[output cut]";
        }
        if (ok && body.empty()) body = kNothingPrinted;
        while (!body.empty() && body.back() == '\n') body.pop_back();
        return output_block(body);
    }
    Exec exec_;
    bool outside_ = false;
    bool kernel_used_ = false;
    Limits limits_;
    std::string done_;
    std::vector<std::string> programs_;
    std::size_t printed_ = 0;
    bool printed_last_ = false;
    bool quiet_last_ = false;
    bool mismatch_last_ = false;
    bool matched_last_ = false;
    int runs_ = 0;
    uint64_t steps_ = 0;
};
}  // namespace strata::igc::sandbox
