// #33: GPU lanes of the pipeline timeline (strata/timeline_gpu.hpp).  A chain of marks around three device memsets of
// known relative size must come back as three spans, in order, inside the host interval that enqueued and waited for
// them, with the largest memset the longest; a span between two events must land on its own lane.
#include "strata/timeline_gpu.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c, ...)                     \
    do {                                  \
        if (!(c)) {                       \
            std::printf("FAIL %s: ", #c); \
            std::printf(__VA_ARGS__);     \
            std::printf("\n");            \
            ++fails;                      \
        }                                 \
    } while (0)

struct Rec { std::string name; int tid; double ts, dur; };

// the few fields this test needs, from the records timeline.cpp writes (one object per line)
static std::vector<Rec> spans(const std::string& text) {
    std::vector<Rec> out;
    std::istringstream in(text);
    std::string line;
    auto num = [&](const std::string& l, const char* key) {
        const size_t p = l.find(key);
        return p == std::string::npos ? -1.0 : std::atof(l.c_str() + p + std::strlen(key));
    };
    while (std::getline(in, line)) {
        if (line.find("\"ph\":\"X\"") == std::string::npos) continue;
        const size_t n0 = line.find("\"name\":\"") + 8;
        out.push_back({line.substr(n0, line.find('"', n0) - n0), (int) num(line, "\"tid\":"), num(line, "\"ts\":"),
                       num(line, "\"dur\":")});
    }
    return out;
}

int main() {
    char path[512];
    std::snprintf(path, sizeof path, "%s/strata-timeline-gpu-test.json", std::getenv("TEMP") ? std::getenv("TEMP") : ".");
#if defined(_WIN32)
    _putenv_s("STRATA_TIMELINE", path);
#else
    setenv("STRATA_TIMELINE", path, 1);
#endif
    namespace tl = strata::timeline;
    cudaSetDevice(0);
    cudaStream_t s = nullptr, c = nullptr;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking);
    const size_t mb = 1 << 20;
    void* buf = nullptr;
    if (cudaMalloc(&buf, 512 * mb) != cudaSuccess) { std::printf("no device memory\n"); return 1; }
    tl::GpuClock clk;
    CHECK(clk.record(s) == nullptr, "no event before the anchor");
    clk.anchor(s);
    const int compute = tl::lane("gpu test compute"), copy = tl::lane("gpu test copy");
    const double h0 = tl::now_us();
    clk.mark(compute, "small", s, 1);
    cudaMemsetAsync(buf, 1, 16 * mb, s);
    clk.mark(compute, "large", s, 2);
    cudaMemsetAsync(buf, 2, 512 * mb, s);
    clk.mark(compute, "medium", s, 3);
    cudaMemsetAsync(buf, 3, 128 * mb, s);
    clk.mark(compute, nullptr, s);
    cudaEvent_t e0 = clk.record(c);
    cudaMemsetAsync(buf, 4, 64 * mb, c);
    cudaEvent_t e1 = clk.record(c);
    clk.span(copy, "copy", e0, e1, 9);
    cudaStreamSynchronize(s);
    cudaStreamSynchronize(c);
    const double h1 = tl::now_us();
    clk.resolve(true);
    tl::flush();
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::vector<Rec> r = spans(ss.str());
    std::vector<Rec> ch, cp;
    for (const Rec& x : r) {
        if (x.tid == compute) ch.push_back(x);
        if (x.tid == copy) cp.push_back(x);
    }
    CHECK(ch.size() == 3, "three compute spans (%zu)", ch.size());
    CHECK(cp.size() == 1 && cp[0].name == "copy", "one copy span on its own lane (%zu)", cp.size());
    if (ch.size() == 3) {
        CHECK(ch[0].name == "small" && ch[1].name == "large" && ch[2].name == "medium", "in mark order: %s %s %s",
              ch[0].name.c_str(), ch[1].name.c_str(), ch[2].name.c_str());
        CHECK(ch[1].dur > ch[2].dur && ch[2].dur > ch[0].dur, "durations follow the sizes: %.1f %.1f %.1f us",
              ch[0].dur, ch[1].dur, ch[2].dur);
        for (int i = 0; i < 2; ++i)
            CHECK(ch[(size_t) i + 1].ts >= ch[(size_t) i].ts + ch[(size_t) i].dur - 1.0, "span %d ends before %d starts", i, i + 1);
        // on the host clock: inside [enqueue, synchronized], allowing the anchor's own error
        CHECK(ch[0].ts >= h0 - 50.0 && ch[2].ts + ch[2].dur <= h1 + 50.0,
              "placed on the host clock: %.1f..%.1f within %.1f..%.1f", ch[0].ts, ch[2].ts + ch[2].dur, h0, h1);
        std::printf("spans: small %.1f us, large %.1f us, medium %.1f us; copy %.1f us\n", ch[0].dur, ch[1].dur,
                    ch[2].dur, cp.empty() ? 0.0 : cp[0].dur);
    }
    {   // #178: split_send's two relay spans share the event between them.  As two span() calls it was recycled twice
        // and record() handed it to two spans; as a mark chain it is recycled once, after its last use - also when a
        // resolve(false) lands between the two spans' completions (the second one still pending)
        void* big = nullptr;
        cudaMalloc(&big, 256 * mb);
        clk.mark(copy, "first", c);
        cudaMemsetAsync(big, 5, 1 * mb, c);
        clk.mark(copy, "second", c);
        cudaMemsetAsync(big, 6, 256 * mb, c);   // long: the second span is still running at the first resolve
        clk.mark(copy, nullptr, c);
        clk.resolve(false);
        cudaEvent_t x = clk.record(s), y = clk.record(s), z = clk.record(s);
        CHECK(x != y && y != z && x != z, "no event handed out twice (%p %p %p)", (void*) x, (void*) y, (void*) z);
        cudaStreamSynchronize(c);
        cudaStreamSynchronize(s);
        clk.resolve(true);
        cudaFree(big);
    }
    cudaFree(buf);
    std::remove(path);
    if (fails == 0) std::printf("timeline_gpu: ok\n");
    return fails == 0 ? 0 : 1;
}
