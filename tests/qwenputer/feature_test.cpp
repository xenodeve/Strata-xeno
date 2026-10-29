#include "strata/core/qwenputer_artifact.hpp"
#include <fstream>
#include <iostream>
#include <iterator>

using namespace strata::qwenputer;
static int failures = 0;
static void check(bool value, const char* name) {
    if (!value) { std::cerr << "FAIL " << name << '\n'; ++failures; }
}
static std::vector<std::uint8_t> read(const char* path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
static std::string hex(const Digest& digest) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (auto b : digest) { result += digits[b >> 4]; result += digits[b & 15]; }
    return result;
}
int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "provide artifact and binding\n"; return 2; }
    const auto identity = read(argv[2]);
    if (identity.size() != 192) return 2;
    Binding binding;
    for (int i = 0; i < 192; ++i) binding.digests[i / 32][i % 32] = identity[i];
    Artifact a;
    MemoryPlan plan;
    std::string error;
    if (!load_artifact(argv[1], binding, {}, a, plan, error)) {
        std::cerr << "FAIL load " << error << '\n'; return 1;
    }
    float value = -999;
    check(feature_value(a, {0,2,0,0}, TokenKind::Trace, 32767, value) && value == 1, "family boundary");
    check(feature_value(a, {0,3,0,14}, TokenKind::Trace, 16384, value) && value == 1, "payload bit");
    check(feature_value(a, {0,5,5,0}, TokenKind::Trace, 1, value) && value == 128, "code payload order");
    check(feature_value(a, {0,8,0,0}, TokenKind::Text, 0, value) && value == 0x1p-40f, "tau exact");
    check(feature_value(a, {0,9,0,0}, TokenKind::Text, a.config[7], value) && value == 1, "tool open");
    check(feature_value(a, {0,11,0,6}, TokenKind::Text, static_cast<std::uint32_t>(a.printable[65]), value)
          && value == 1, "text byte projection");
    value = -999;
    check(!feature_value(a, {0,1,0,0}, TokenKind::Trace, 230324, value) && value == -999, "invalid trace unchanged");
    check(!feature_value(a, {0,1,0,0}, TokenKind::Text, a.config[2], value), "invalid text");
    check(!feature_value(a, {0,99,0,0}, TokenKind::Text, 0, value), "unknown feature");
    Digest digest{};
    check(feature_digest(a, digest, error), "full feature enumeration");
    check(hex(digest) == "190676489d4adacf6ee6a78d80f9fcbd1a2bc98eced7210c5881f567762478f6",
          "263254200 values match original compiler proof");
    std::cout << (failures ? "FAIL " : "PASS ") << failures << " failures; digest=" << hex(digest) << '\n';
    return failures ? 1 : 0;
}
