// tests/xeno/nvme_overlap.cpp - #95: materialize_begin / materialize_end split a layer's NVMe-tier load so the CPU pool
// can run the layer's resident experts while the reads are in flight.  The split must leave exactly the state
// materialize_batch leaves (resident set, host bytes, load count), the batch's experts must not look resident between
// begin and end (a job would read a slot whose bytes have not landed), and the other residents stay readable.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() /
        ("strata-nvme-overlap-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(dir);
    const uint64_t blob = strata::kernels::cpu::BLOB;
    const int layers = 2, experts = 3;
    auto mark = [](int l, int e) { return (uint8_t) (0x11 * (l * 3 + e) + 1); };
    {
        std::ofstream out(dir / "experts.bin", std::ios::binary);
        std::vector<uint8_t> b(blob);
        for (int l = 0; l < layers; ++l)
            for (int e = 0; e < experts; ++e) {
                for (uint64_t i = 0; i < blob; ++i) b[i] = (uint8_t) (mark(l, e) + i);
                out.write((const char*) b.data(), (std::streamsize) blob);
            }
    }
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(dir.string(), layers, experts, err)) {
        std::fprintf(stderr, "layout: %s\n", err.c_str()); return 1;
    }
    // room for three experts; order (0,0), (1,0), (1,1): layer 0 has one resident, layer 1 two
    auto make = [&](std::unique_ptr<strata::core::ArenaExpertSource>& s) -> bool {
        s = std::make_unique<strata::core::ArenaExpertSource>();
        if (!s->open(dir.string(), layers, experts, 1, err, /*pin_for_cuda=*/false, /*defer_load=*/true)) return false;
        s->set_capacity(3 * blob, {0, 3, 4, 1, 2, 5});
        return s->load_rest(1, err);
    };
    std::unique_ptr<strata::core::ArenaExpertSource> a, b;
    if (!make(a) || !make(b)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    int bad = 0;
    auto expect = [&](bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++bad; } };

    const int32_t miss[2] = {1, 2};
    expect(a->materialize_batch(0, miss, 2, err), "materialize_batch");

    expect(b->materialize_begin(0, miss, 2, err), "materialize_begin");
    expect(!b->resident(0, 1) && b->blob(0, 1) == nullptr, "in flight: (0,1) is not resident before end");
    expect(!b->resident(0, 2) && b->blob(0, 2) == nullptr, "in flight: (0,2) is not resident before end");
    expect(b->blob(0, 0) && b->blob(0, 0)[3] == (uint8_t) (mark(0, 0) + 3), "a resident expert stays readable");
    expect(b->materialize_end(err), "materialize_end");

    for (int l = 0; l < layers; ++l)
        for (int e = 0; e < experts; ++e)
            expect(a->resident(l, e) == b->resident(l, e), "the same resident set as materialize_batch");
    expect(a->host_cache_bytes() == b->host_cache_bytes(), "the same host bytes");
    expect(a->nvme_loads() == b->nvme_loads() && b->nvme_loads() == 2, "two loads each");
    {   // where a miss's time goes: every stage counted, the read and the copy always nonzero after a load
        const auto s = b->nvme_stages();
        expect(s.evict_ms >= 0 && s.commit_ms >= 0 && s.submit_ms >= 0, "the stage times are counted");
        expect(s.wait_ms + s.copy_ms > 0 && s.copy_ms > 0, "a load waits for and copies its bytes");
    }
    expect(b->blob(0, 2) && b->blob(0, 2)[blob - 1] == (uint8_t) (mark(0, 2) + blob - 1), "the bytes landed");

    // nothing to read: begin and end are no-ops; end without begin too
    const int32_t resident_only[1] = {1};
    expect(b->materialize_begin(0, resident_only, 1, err) && b->materialize_end(err), "a resident expert: no-op");
    expect(b->materialize_end(err), "end without begin");
    expect(b->nvme_loads() == 2, "no extra load");

    a.reset();
    b.reset();
    std::error_code ec;
    fs::remove_all(dir, ec);

    // a dispatch that fails between materialize_begin and materialize_end must still release the host tier: a held
    // host_mu_ turns a loud failure into a silent hang of the next hold()/materialize() (the adapt thread's join).
    // Window: (0,1) is a miss (its read goes in flight), then an out-of-range id fails the dispatch before the pool.
    // A native Q8_0 pack: its activation quantizer runs on AVX-2 CPUs (the Q2 layout's is AVX-512 only).
    const fs::path ndir = dir.string() + "-native";
    fs::create_directory(ndir);
    strata::kernels::cpu::NativeFmt f;
    if (!strata::kernels::cpu::native_fmt(8, 8, strata::kernels::cpu::H, strata::kernels::cpu::FF, f, err)) {
        std::fprintf(stderr, "native_fmt: %s\n", err.c_str()); return 1;
    }
    {
        std::ofstream txt(ndir / "native_experts.txt");
        for (int l = 0; l < layers; ++l) txt << l << " 8 8 " << (uint64_t) l * experts * f.bytes << " " << f.bytes << "\n";
        std::ofstream out(ndir / "experts.bin", std::ios::binary);
        std::vector<uint8_t> z(f.bytes, 0);
        for (int i = 0; i < layers * experts; ++i) out.write((const char*) z.data(), (std::streamsize) f.bytes);
    }
    if (!strata::kernels::cpu::expert_layout_load(ndir.string(), layers, experts, err)) {
        std::fprintf(stderr, "native layout: %s\n", err.c_str()); return 1;
    }
    auto c = std::make_unique<strata::core::ArenaExpertSource>();
    if (!c->open(ndir.string(), layers, experts, 1, err, /*pin_for_cuda=*/false, /*defer_load=*/true)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    c->set_capacity(3 * f.bytes, {0, 3, 4, 1, 2, 5});
    if (!c->load_rest(1, err)) { std::fprintf(stderr, "load_rest: %s\n", err.c_str()); return 1; }
    strata::core::ExpertDispatch d;
    d.src = c.get();
    d.n_expert = experts;
    d.layers = 0;
    const int32_t ids[2] = {1, experts + 4};
    std::vector<float> x((size_t) strata::kernels::cpu::H, 0.5f), y(2 * (size_t) strata::kernels::cpu::H);
    strata::core::expert_pool_dispatch_multi(d, x.data(), ids, 1, 2, y.data());
    expect(d.failed, "the out-of-range id fails the dispatch");
    std::atomic<bool> got{false};
    std::thread([&] { c->hold(1, 0); got = true; }).detach();
    for (int i = 0; i < 200 && !got; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (!got) { std::fprintf(stderr, "FAIL: the failed dispatch left the host tier locked\n"); std::_Exit(1); }
    c->release_hold(1, 0);
    expect(c->materialize_begin(0, miss, 2, err) && c->materialize_end(err), "a new batch can start after the failure");
    c.reset();

    // the overlapped dispatch scores the host tier exactly as the serial one (materialize_batch): blob() counts a use, so
    // one lookup per routed expert per window - a second lookup would change which expert the next miss evicts.
    // Boot: (0,0), (1,0), (1,1) resident at score 1.  Window 1 (layer 0, experts 0 and 1): (0,1) misses and evicts (1,0)
    // (the lower index of a tie); (0,0) and (0,1) both end at 2.  Window 2 (layer 1, expert 0): the miss evicts the
    // lower-index layer-0 expert of that tie, (0,0) - with a double count (0,0) would be at 3 and (0,1) would go.
    auto s2 = std::make_unique<strata::core::ArenaExpertSource>();
    if (!s2->open(ndir.string(), layers, experts, 1, err, /*pin_for_cuda=*/false, /*defer_load=*/true)) {
        std::fprintf(stderr, "open: %s\n", err.c_str()); return 1;
    }
    s2->set_capacity(3 * f.bytes, {0, 3, 4, 1, 2, 5});
    if (!s2->load_rest(1, err)) { std::fprintf(stderr, "load_rest: %s\n", err.c_str()); return 1; }
    strata::kernels::cpu::ExpertPool pool(1, /*pin=*/false);
    strata::core::ExpertDispatch d2;
    d2.src = s2.get();
    d2.pool = &pool;
    d2.n_expert = experts;
    const int32_t w1[2] = {0, 1}, w2[1] = {0};
    d2.layers = 0;
    strata::core::expert_pool_dispatch_multi(d2, x.data(), w1, 1, 2, y.data());
    expect(!d2.failed, "window 1 dispatches");
    expect(s2->resident(0, 0) && s2->resident(0, 1) && !s2->resident(1, 0), "window 1: (0,1) in, (1,0) out");
    d2.layers = 1;
    strata::core::expert_pool_dispatch_multi(d2, x.data(), w2, 1, 1, y.data());
    expect(!d2.failed, "window 2 dispatches");
    expect(s2->resident(1, 0) && !s2->resident(0, 0) && s2->resident(0, 1), "window 2 evicts (0,0): one use per lookup");
    s2.reset();
    fs::remove_all(ndir, ec);
    if (bad == 0) std::printf("xeno_nvme_overlap: PASS\n");
    return bad == 0 ? 0 : 1;
}
