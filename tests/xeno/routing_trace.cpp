// tests/xeno/routing_trace.cpp - #85: the --dump-routing format with commit and format tags.
//
// The N0 simulator (#87) rebuilds verify windows and accepted / rejected positions from these records, and
// tools/make_profile.py must keep reading the file as before.  So the test pins the exact bytes of each record
// shape, then reads a whole written file back.
#include "strata/core/routing_trace.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace rt = strata::core::routing_trace;

static int g_fail = 0;
#define CHECK(c)                                                                                   \
    do {                                                                                           \
        if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

static std::vector<unsigned char> bytes_of(const std::string& path) {
    std::vector<unsigned char> b;
    if (std::FILE* f = std::fopen(path.c_str(), "rb")) {
        unsigned char buf[4096];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
        std::fclose(f);
    }
    return b;
}

static int32_t i32_at(const std::vector<unsigned char>& b, size_t at) {
    int32_t v = 0;
    if (at + 4 <= b.size()) std::memcpy(&v, b.data() + at, 4);
    return v;
}

int main() {
    const std::string path = std::string(std::getenv("TEMP") ? std::getenv("TEMP") : ".") + "/xeno_routing_trace.bin";

    // 1. byte layout: a commit tag is 8 + 3*4 + 3*4 bytes, ints then zero floats
    {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        rt::write_commit(f, 7, 4, 2);
        std::fclose(f);
        const auto b = bytes_of(path);
        CHECK(b.size() == 8 + 12 + 12);
        CHECK(i32_at(b, 0) == rt::kTagCommit);
        CHECK(i32_at(b, 4) == 3);
        CHECK(i32_at(b, 8) == 7 && i32_at(b, 12) == 4 && i32_at(b, 16) == 2);
        bool zero = b.size() == 32;
        for (size_t i = 20; i < b.size(); ++i) zero = zero && b[i] == 0;
        CHECK(zero);
    }

    // 2. a route record without weights carries zeros; with weights, the weights
    {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        const int32_t ids[2] = {511, 3};
        const float w[2] = {0.75f, 0.25f};
        rt::write_route(f, 47, 2, ids, nullptr);
        rt::write_route(f, 0, 2, ids, w);
        std::fclose(f);
        std::vector<rt::Record> r;
        CHECK(rt::read_all(path, r));
        CHECK(r.size() == 2);
        if (r.size() == 2) {
            CHECK(r[0].layer == 47 && r[0].ids == std::vector<int32_t>({511, 3}));
            CHECK(r[0].weights == std::vector<float>({0.0f, 0.0f}));
            CHECK(r[1].layer == 0 && r[1].weights == std::vector<float>({0.75f, 0.25f}));
        }
    }

    // 3. a whole window round-trips in order: format, then layer-major positions, then the commit
    {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        rt::write_format(f);
        for (int32_t layer = 0; layer < 2; ++layer)
            for (int32_t pos = 0; pos < 3; ++pos) {
                const int32_t ids[1] = {layer * 100 + pos};
                rt::write_route(f, layer, 1, ids, nullptr);
            }
        rt::write_commit(f, 0, 3, 1);
        std::fclose(f);
        std::vector<rt::Record> r;
        CHECK(rt::read_all(path, r));
        CHECK(r.size() == 1 + 6 + 1);
        if (r.size() == 8) {
            CHECK(r[0].layer == rt::kTagFormat && r[0].ids == std::vector<int32_t>({rt::kFormatVersion}));
            CHECK(r[1].layer == 0 && r[1].ids[0] == 0);
            CHECK(r[3].layer == 0 && r[3].ids[0] == 2);
            CHECK(r[4].layer == 1 && r[4].ids[0] == 100);
            CHECK(r[7].layer == rt::kTagCommit && r[7].ids == std::vector<int32_t>({0, 3, 1}));
        }
    }

    // 4. a truncated record is an error, not a silently shorter trace
    {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        rt::write_commit(f, 1, 2, 1);
        const int32_t half[2] = {3, 10};   // a route header that promises 10 ids and delivers none
        std::fwrite(half, sizeof half, 1, f);
        std::fclose(f);
        std::vector<rt::Record> r;
        CHECK(!rt::read_all(path, r));
    }

    // 5. a partial record header at the very end (a crash mid-write) is a truncation too
    {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        rt::write_commit(f, 1, 2, 1);
        const int32_t stub = -1;
        std::fwrite(&stub, sizeof stub, 1, f);   // 4 of the next record's 8 header bytes
        std::fclose(f);
        std::vector<rt::Record> r;
        CHECK(!rt::read_all(path, r));
    }

    // 6. #86: request, phase, GPU-owned and boot-resident tags round-trip; an empty id list is still a record
    {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        rt::write_owned(f, std::vector<int32_t>{3, 512 * 47 + 511});
        rt::write_boot(f, std::vector<int32_t>{});
        rt::write_request(f, 12);
        rt::write_phase(f, 0);
        std::fclose(f);
        std::vector<rt::Record> r;
        CHECK(rt::read_all(path, r));
        CHECK(r.size() == 4);
        if (r.size() == 4) {
            CHECK(r[0].layer == rt::kTagOwned && r[0].ids == std::vector<int32_t>({3, 512 * 47 + 511}));
            CHECK(r[1].layer == rt::kTagBoot && r[1].ids.empty());
            CHECK(r[2].layer == rt::kTagRequest && r[2].ids == std::vector<int32_t>({12}));
            CHECK(r[3].layer == rt::kTagPhase && r[3].ids == std::vector<int32_t>({0}));
        }
    }

    // 7. a long id list (more than the writer's 64-float padding block) keeps its zero padding exact
    {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        std::vector<int32_t> many(1000);
        for (int32_t i = 0; i < 1000; ++i) many[(size_t) i] = i;
        rt::write_owned(f, many);
        std::fclose(f);
        std::vector<rt::Record> r;
        CHECK(rt::read_all(path, r));
        CHECK(r.size() == 1 && r[0].ids == many && r[0].weights == std::vector<float>(1000, 0.0f));
    }

    std::remove(path.c_str());
    if (g_fail == 0) std::printf("xeno_routing_trace: PASS\n");
    return g_fail == 0 ? 0 : 1;
}
