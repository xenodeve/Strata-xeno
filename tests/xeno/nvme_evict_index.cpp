// tests/xeno/nvme_evict_index.cpp - #97: the eviction victim (lowest decayed score outside the loading layer, held
// and GPU-owned experts skipped, ties to the lowest index) comes from per-layer cached minima instead of a scan of
// every expert per eviction (16 scans of 24,576 experts per decode round).  A differential: the same random sequence
// of loads, reads (blob counts a use), holds, decays and copy-homes on a source with the cache and on one with the
// full scan (STRATA_NVME_EVICT_SCAN=1) must leave the same resident set after every step.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() /
        ("strata-evict-index-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(dir);
    const uint64_t blob = strata::kernels::cpu::BLOB;
    const int layers = 4, experts = 6;
    {
        std::ofstream out(dir / "experts.bin", std::ios::binary);
        std::vector<uint8_t> b(blob, 0x3c);
        for (int i = 0; i < layers * experts; ++i) out.write((const char*) b.data(), (std::streamsize) blob);
    }
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(dir.string(), layers, experts, err)) {
        std::fprintf(stderr, "layout: %s\n", err.c_str()); return 1;
    }
    auto make = [&](std::unique_ptr<strata::core::ArenaExpertSource>& s) -> bool {
        s = std::make_unique<strata::core::ArenaExpertSource>();
        if (!s->open(dir.string(), layers, experts, 1, err, /*pin_for_cuda=*/false, /*defer_load=*/true)) return false;
        if (!s->release_host_copy(3, 5, err)) return false;   // one GPU-owned expert, as at a placement-first boot
        s->set_capacity(7 * blob, {0, 6, 12, 1, 7, 13, 2, 8, 14, 3});
        return s->load_rest(1, err);
    };
    _putenv_s("STRATA_NVME_EVICT_SCAN", "1");
    std::unique_ptr<strata::core::ArenaExpertSource> scan, cached;
    if (!make(scan)) { std::fprintf(stderr, "open scan: %s\n", err.c_str()); return 1; }
    _putenv_s("STRATA_NVME_EVICT_SCAN", "");
    if (!make(cached)) { std::fprintf(stderr, "open cached: %s\n", err.c_str()); return 1; }

    int bad = 0;
    std::mt19937 rng(97);
    std::vector<std::pair<int, int>> held;
    bool homed = false;
    for (int step = 0; step < 3000 && bad < 5; ++step) {
        const int op = (int) (rng() % 100);
        const int l = (int) (rng() % layers), e = (int) (rng() % experts);
        if (op < 45) {                               // a layer's misses, 1-3 experts
            int32_t ids[3];
            const int n = 1 + (int) (rng() % 3);
            for (int k = 0; k < n; ++k) ids[k] = (int32_t) (rng() % experts);
            const bool a = scan->materialize_batch(l, ids, n, err), b = cached->materialize_batch(l, ids, n, err);
            if (a != b) { std::fprintf(stderr, "FAIL step %d: load result differs\n", step); ++bad; }
        } else if (op < 75) {                        // a read counts a use
            const bool a = scan->blob(l, e) != nullptr, b = cached->blob(l, e) != nullptr;
            if (a != b) { std::fprintf(stderr, "FAIL step %d: blob differs\n", step); ++bad; }
        } else if (op < 82) {                        // hold, or release the oldest hold
            if (held.size() < 3 && scan->resident(l, e)) {
                scan->hold(l, e); cached->hold(l, e); held.emplace_back(l, e);
            } else if (!held.empty()) {
                scan->release_hold(held.front().first, held.front().second);
                cached->release_hold(held.front().first, held.front().second);
                held.erase(held.begin());
            }
        } else if (op < 92) {
            scan->decay_scores(); cached->decay_scores();
        } else if (op < 96) {
            scan->trim(l); cached->trim(l);
        } else if (!homed) {                          // the GPU-owned expert comes home once (a paired swap)
            uint8_t* a = scan->recommit_host_copy(3, 5, err);
            uint8_t* b = cached->recommit_host_copy(3, 5, err);
            if (a != nullptr && b != nullptr) {
                scan->publish_host_copy(3, 5); cached->publish_host_copy(3, 5);
                scan->admit_home(3, 5); cached->admit_home(3, 5);
                homed = true;
            }
        }
        for (int ll = 0; ll < layers; ++ll)
            for (int ee = 0; ee < experts; ++ee)
                if (scan->resident(ll, ee) != cached->resident(ll, ee)) {
                    std::fprintf(stderr, "FAIL step %d (op %d): (%d,%d) scan %d cached %d\n", step, op, ll, ee,
                                 (int) scan->resident(ll, ee), (int) cached->resident(ll, ee));
                    ++bad;
                }
    }
    if (scan->nvme_loads() != cached->nvme_loads()) { std::fprintf(stderr, "FAIL: loads differ\n"); ++bad; }
    scan.reset();
    cached.reset();
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (bad == 0) std::printf("xeno_nvme_evict_index: PASS\n");
    return bad == 0 ? 0 : 1;
}
