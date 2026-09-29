// #33: the pipeline timeline (strata/timeline.hpp) - every thread's spans and the virtual (GPU) lanes reach the file,
// a second flush appends only the new events, and the file parses as a Chrome trace once closed.
#include "strata/timeline.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static int fails = 0;
#define CHECK(c, ...)                                   \
    do {                                                \
        if (!(c)) {                                     \
            std::printf("FAIL %s: ", #c);               \
            std::printf(__VA_ARGS__);                   \
            std::printf("\n");                          \
            ++fails;                                    \
        }                                               \
    } while (0)

static size_t count(const std::string& s, const std::string& what) {
    size_t n = 0;
    for (size_t p = s.find(what); p != std::string::npos; p = s.find(what, p + 1)) ++n;
    return n;
}

static std::string slurp(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main() {
    char path[512];
    std::snprintf(path, sizeof path, "%s/strata-timeline-test-%d.json",
                  std::getenv("TEMP") ? std::getenv("TEMP") : ".", (int) std::rand());
#if defined(_WIN32)
    _putenv_s("STRATA_TIMELINE", path);
#else
    setenv("STRATA_TIMELINE", path, 1);
#endif
    namespace tl = strata::timeline;
    CHECK(tl::enabled(), "STRATA_TIMELINE is set");
    tl::name_thread("main");
    const double t0 = tl::now_us();
    CHECK(t0 > 0, "the clock runs (%f)", t0);
    tl::complete("setup", t0, t0 + 5, 7, 8);
    const int gpu = tl::lane("gpu0 compute");
    CHECK(gpu == tl::lane("gpu0 compute"), "a lane name maps to one lane");
    tl::complete_on(gpu, "gemm", t0 + 1, t0 + 3, 3, -1);
    tl::instant("marker", 1);
    {
        tl::Span s("scoped", 42);
    }
    std::vector<std::thread> ts;
    for (int w = 0; w < 4; ++w)
        ts.emplace_back([w] {
            char name[32];
            std::snprintf(name, sizeof name, "worker %d", w);
            tl::name_thread(name);
            for (int i = 0; i < 1000; ++i) {
                const double a = tl::now_us();
                tl::complete("job", a, a + 1, i, w);
            }
        });
    for (auto& t : ts) t.join();
    tl::flush();
    std::string first = slurp(path);
    CHECK(!first.empty() && first[0] == '[', "the file opens a JSON array");
    // 1 setup + 1 gemm + 1 scoped + 4000 jobs
    CHECK(count(first, "\"ph\":\"X\"") == 4003, "complete events after the first flush: %zu",
          count(first, "\"ph\":\"X\""));
    CHECK(count(first, "\"ph\":\"i\"") == 1, "instant events: %zu", count(first, "\"ph\":\"i\""));
    CHECK(count(first, "\"name\":\"thread_name\"") == 6, "lane names (main, gpu0 compute, 4 workers): %zu",
          count(first, "\"name\":\"thread_name\""));
    CHECK(count(first, "\"worker 3\"") == 1, "a worker's name is written once");
    CHECK(count(first, "\"a\":42") >= 1, "a span's argument reaches the file");

    tl::complete("after", tl::now_us(), tl::now_us() + 1);
    // a thread made per window (the adaptive tier's, the copy issuer's per chunk) takes over the finished lane of
    // the same name: 100 of them must not become 100 rows
    for (int i = 0; i < 100; ++i)
        std::thread([] {
            tl::name_thread("per-window thread");
            tl::complete("adapt", tl::now_us(), tl::now_us() + 1);
        }).join();
    tl::flush();
    std::string second = slurp(path);
    CHECK(second.compare(0, first.size(), first) == 0, "a second flush appends");
    CHECK(count(second, "\"ph\":\"X\"") == 4104, "only the new events are appended: %zu", count(second, "\"ph\":\"X\""));
    CHECK(count(second, "\"name\":\"thread_name\"") == 7, "names are not repeated, a reused lane is one: %zu",
          count(second, "\"name\":\"thread_name\""));
    // every record ends with ",\n": closing the array makes it valid JSON (the analyzer does exactly this)
    CHECK(second.size() >= 2 && second.compare(second.size() - 2, 2, ",\n") == 0, "records end with a comma");
    std::remove(path);
    if (fails == 0) std::printf("timeline_core: ok\n");
    return fails == 0 ? 0 : 1;
}
