#include "strata/core/qwenputer_call.hpp"
#include <iostream>
#include <fstream>
#include <iterator>
#include <cstring>
#include <string_view>
using namespace strata::qwenputer;
using namespace std::string_literals;
static int failures = 0;
static void check(bool ok, const char* name) { if (!ok) { ++failures; std::cerr << "FAIL " << name << '\n'; } }
static Artifact fixture() {
    Artifact a;
    a.config[2] = 260; a.config[7] = 256; a.config[8] = 257;
    for (unsigned q = 0; q < 260; ++q) {
        std::string raw = q < 256 ? std::string(1, static_cast<char>(q)) : q == 259 ? std::string(32, '~') : "special";
        a.text_offsets.push_back(static_cast<std::uint32_t>(a.text_bytes.size()));
        a.text_bytes.insert(a.text_bytes.end(), raw.begin(), raw.end());
        a.text_lengths.push_back(static_cast<std::uint32_t>(raw.size()));
        a.text_types.push_back(q < 256 || q == 259 ? 1 : 4);
    }
    a.text_offsets.push_back(static_cast<std::uint32_t>(a.text_bytes.size()));
    return a;
}
static CallEvent feed(CallParser& p, std::string_view s, bool live = true) {
    std::string error; CallEvent event;
    for (unsigned char c : s) check(p.consume(c, live, event, error), "consume bytes");
    return event;
}
static CallEvent call(CallParser& p, std::string_view body = "<function=calc><parameter=expression>2+3</parameter></function>", bool live = true) {
    std::string error; CallEvent event;
    check(p.consume(256, live, event, error), "open call");
    feed(p, body, live);
    check(p.consume(257, live, event, error), "close call");
    return event;
}
int main(int argc, char** argv) {
    auto a = fixture(); CallParser parser; std::string error;
    const std::uint32_t stops[]{258};
    if (!parser.init(a, stops, error)) { std::cerr << "FAIL init: " << error << '\n'; return 1; }
    parser.begin_assistant(true);
    auto event = call(parser); check(event.kind == CallKind::Eligible && event.expression == "2+3", "live enrolled calc");
    parser.begin_assistant(false); check(call(parser).kind == CallKind::None, "not enrolled");
    parser.begin_assistant(true); check(call(parser, "<function=calc><parameter=expression>2+3</parameter></function>", false).kind == CallKind::None, "history cannot execute");
    parser.begin_assistant(true); check(call(parser, "<function=other><parameter=x>2</parameter></function>").kind == CallKind::External, "external name stays external");
    for (std::string_view prefix : {"```xml\n", "~~~xml\n", "> ", "  > ", "    ", "\t", "`", "\"", "'", "\xe2\x80\x9c", "\xc0", "\\"}) {
        parser.begin_assistant(true); feed(parser, prefix);
        check(call(parser).kind == CallKind::None, "quoted or fenced call ignored");
    }
    for (std::string_view prefix : {"```xml\nexample\n```\n", "~~~\nx\n~~~~\n", "`x` ", "\"x\" ", "don't ", "\xe2\x80\x9c\xe2\x80\x9d "}) {
        parser.begin_assistant(true); feed(parser, prefix);
        check(call(parser).kind == CallKind::Eligible, "closed outer region allows call");
    }
    parser.begin_assistant(true); check(parser.consume(259, true, event, error), "long token"); feed(parser, "\n");
    check(call(parser).kind == CallKind::None, "full 32-byte fence remains active");
    feed(parser, "\n" + std::string(31, '~') + "\n"); check(call(parser).kind == CallKind::None, "short close cannot end long fence");
    feed(parser, std::string(32, '~') + "\n"); check(call(parser).kind == CallKind::None, "close in the middle of a line is not a fence");
    feed(parser, "\n" + std::string(32, '~') + "\n"); check(call(parser).kind == CallKind::Eligible, "complete long fence closes");
    for (const auto& body : std::vector<std::string>{"<function=calc>bad</function>", "<function=calc><parameter=expression>__import__(x)</parameter></function>", "<function=calc><parameter=expression>2\0+3</parameter></function>"s}) {
        parser.begin_assistant(true); check(call(parser, body).kind == CallKind::Invalid, "malformed native call refused");
    }
    parser.begin_assistant(true);
    check(call(parser, "\n\n<function=calc><parameter=expression>2+3</parameter></function>").kind == CallKind::Invalid, "malformed native framing stays internal refusal");
    parser.begin_assistant(true);
    check(parser.consume(256, true, event, error), "nested capture open");
    feed(parser, "<function=calc><parameter=expression>2");
    check(parser.consume(256, true, event, error), "nested opener refused without rearming");
    feed(parser, "+3</parameter></function>");
    check(parser.consume(257, true, event, error) && event.kind == CallKind::Invalid && event.refusal == "capture_nested_call", "nested call cannot execute");
    parser.begin_assistant(true);
    check(parser.consume(256, true, event, error), "wide capture open");
    feed(parser, "<function=calc><parameter=expression>");
    check(parser.consume(259, true, event, error), "wide captured token"); feed(parser, "</parameter></function>");
    check(parser.consume(257, true, event, error) && event.kind == CallKind::Invalid && event.refusal == "capture_token_width", "wide capture cannot silently truncate");
    parser.begin_assistant(true);
    check(call(parser, "<function=calc><parameter=expression>" + std::string(90, '1') + "</parameter></function>").kind == CallKind::Invalid, "capture token limit");
    parser.begin_assistant(true); feed(parser, "normal "); auto shadow = parser;
    check(call(shadow).kind == CallKind::Eligible, "shadow can trial a takeover");
    check(call(parser).kind == CallKind::Eligible, "discarded shadow leaves base intact");
    parser.begin_assistant(true); check(parser.consume(258, true, event, error), "EOS");
    check(call(parser).kind == CallKind::None, "post EOS cannot execute");
    check(!parser.consume(999, true, event, error) && error == "call_token_id", "unchecked id refused");
    auto legacy = a; legacy.text_offsets.clear(); legacy.text_bytes.clear();
    CallParser refused; check(!refused.init(legacy, stops, error), "truncated legacy lexicon cannot enroll");
    if (argc == 3) {
        std::ifstream file(argv[2], std::ios::binary);
        const std::vector<std::uint8_t> raw{std::istreambuf_iterator<char>(file), {}};
        if (raw.size() != 192) { check(false, "actual binding fixture"); return 1; }
        Binding binding; std::memcpy(binding.digests.data(), raw.data(), raw.size());
        Artifact real; MemoryPlan memory;
        check(load_artifact(argv[1], binding, {}, real, memory, error), "load actual lexicon artifact");
        CallParser actual;
        if (!actual.init(real, {}, error)) { check(false, "actual parser init"); return 1; }
        auto actual_feed = [&](std::string_view s) {
            for (unsigned char c : s) check(actual.consume(static_cast<std::uint32_t>(real.printable[c]), true, event, error), "actual byte token");
        };
        auto actual_call = [&]() {
            check(actual.consume(real.config[7], true, event, error), "actual opener");
            actual_feed("<function=calc><parameter=expression>2+3</parameter></function>");
            check(actual.consume(real.config[8], true, event, error), "actual closer");
            return event.kind;
        };
        actual.begin_assistant(true); check(actual_call() == CallKind::Eligible, "actual binding native call");
        actual.begin_assistant(true);
        check(actual.consume(63434, true, event, error), "actual 32-tilde token"); actual_feed("\n");
        check(actual_call() == CallKind::None, "actual complete long fence ignores call");
        actual_feed("\n" + std::string(32, '~') + "\n");
        check(actual_call() == CallKind::Eligible, "actual complete long fence closes");
    }
    std::cout << (failures ? "FAIL " : "PASS ") << failures << " failures\n"; return failures ? 1 : 0;
}
