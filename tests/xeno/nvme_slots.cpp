// tests/xeno/nvme_slots.cpp - capacity mode's host tier as a pool of committed slots.  A load used to commit the
// expert's fixed arena pages (first touch: ~430 demand-zero faults) and an eviction to decommit the victim's; on the
// decode path those were half of a miss's cost (copy 0.35 ms, evict 0.27 ms per load).  With slots a load overwrites
// the victim's slot: nothing is committed or decommitted after boot.  What must hold:
//   - eviction order, host bytes and loads are the byte-accounted ones (the slots are only the backing store);
//   - every resident expert reads its own bytes after many evict/load cycles (a reused slot is fully rewritten);
//   - a copy-home (a GPU-owned expert coming back) gets a slot, and release_host_copy gives it back;
//   - acquire() holds an expert in one step, so no eviction can reuse its slot between the lookup and the hold;
//   - STRATA_NVME_SLOTS=0 keeps the fixed addresses (the A/B arm).
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

int main() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() /
        ("strata-nvme-slots-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(dir);
    const uint64_t blob = strata::kernels::cpu::BLOB;
    const int layers = 3, experts = 4;
    auto mark = [](int l, int e) { return (uint8_t) (0x13 * (l * 4 + e) + 7); };
    {
        std::ofstream out(dir / "experts.bin", std::ios::binary);
        std::vector<uint8_t> b(blob);
        for (int l = 0; l < layers; ++l)
            for (int e = 0; e < experts; ++e) {
                for (uint64_t i = 0; i < blob; ++i) b[i] = (uint8_t) (mark(l, e) + i * 31);
                out.write((const char*) b.data(), (std::streamsize) blob);
            }
    }
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(dir.string(), layers, experts, err)) {
        std::fprintf(stderr, "layout: %s\n", err.c_str()); return 1;
    }
    int bad = 0;
    auto expect = [&](bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++bad; } };
    auto intact = [&](strata::core::ArenaExpertSource& s, int l, int e) {
        const uint8_t* p = s.blob(l, e);
        if (p == nullptr) return false;
        for (uint64_t i = 0; i < blob; i += 4093)
            if (p[i] != (uint8_t) (mark(l, e) + i * 31)) return false;
        return p[blob - 1] == (uint8_t) (mark(l, e) + (blob - 1) * 31);
    };
    // GPU-owned (2,3) like a placement-first boot; room for 4 host experts, order (0,0) (0,1) (1,0) (1,1) first
    auto make = [&](std::unique_ptr<strata::core::ArenaExpertSource>& s) -> bool {
        s = std::make_unique<strata::core::ArenaExpertSource>();
        if (!s->open(dir.string(), layers, experts, 1, err, /*pin_for_cuda=*/false, /*defer_load=*/true)) return false;
        if (!s->release_host_copy(2, 3, err)) return false;
        s->set_capacity(4 * blob, {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 10});
        return s->load_rest(1, err);
    };

    _putenv_s("STRATA_NVME_SLOTS", "0");
    std::unique_ptr<strata::core::ArenaExpertSource> fixed, slots;
    if (!make(fixed)) { std::fprintf(stderr, "open fixed: %s\n", err.c_str()); return 1; }
    _putenv_s("STRATA_NVME_SLOTS", "");
    _putenv_s("STRATA_NVME_SLACK_MIB", "2");   // one idle slot: the decommit path runs too
    if (!make(slots)) { std::fprintf(stderr, "open slots: %s\n", err.c_str()); return 1; }
    expect(fixed->host_slots() == 0, "STRATA_NVME_SLOTS=0: fixed addresses");
    expect(slots->host_slots() == 4, "capacity mode: one committed slot per resident expert after boot");

    // the same miss sequence on both: residency, bytes and loads agree; every resident expert reads its own bytes
    const int seq[][2] = {{0, 2}, {1, 3}, {2, 0}, {0, 3}, {2, 1}, {1, 2}, {0, 0}, {2, 2}, {1, 0}, {0, 1}, {2, 0}, {1, 1}};
    for (auto& m : seq) {
        const int32_t one[1] = {m[1]};
        expect(fixed->materialize_batch(m[0], one, 1, err), "fixed load");
        expect(slots->materialize_batch(m[0], one, 1, err), "slot load");
        for (int l = 0; l < layers; ++l)
            for (int e = 0; e < experts; ++e) {
                expect(fixed->resident(l, e) == slots->resident(l, e), "the same resident set");
                // both arms read (blob() counts a use, so both must, or their scores part ways)
                if (slots->resident(l, e)) {
                    expect(intact(*slots, l, e), "a resident slot holds its own expert");
                    expect(intact(*fixed, l, e), "the fixed arm reads the same expert");
                }
            }
        expect(fixed->host_cache_bytes() == slots->host_cache_bytes(), "the same host bytes");
    }
    expect(fixed->nvme_loads() == slots->nvme_loads(), "the same loads");
    expect(slots->host_idle_bytes() <= (2ull << 20), "idle slots stay within the slack");
    expect(slots->host_slots() >= 4, "the slots of the resident experts stay committed");
    expect(slots->nvme_stages().commit_ms == 0.0, "a slot load commits nothing");

    // copy-home: the GPU-owned (2,3) comes back into a slot, then goes out again and frees it
    uint8_t* home = slots->recommit_host_copy(2, 3, err);
    expect(home != nullptr, "copy-home gets a slot");
    if (home != nullptr) {
        for (uint64_t i = 0; i < blob; ++i) home[i] = (uint8_t) (mark(2, 3) + i * 31);
        slots->publish_host_copy(2, 3);
        slots->admit_home(2, 3);
        expect(intact(*slots, 2, 3), "the homed expert reads its bytes");
        expect(slots->host_cache_bytes() <= 4 * blob, "admit_home trims to the cap");
        expect(slots->release_host_copy(2, 3, err), "release gives the slot back");
        expect(slots->blob(2, 3) == nullptr, "a released expert is GPU-owned again");
    }

    // acquire: resident -> held at once; a miss is loaded and held; held experts survive every eviction
    const uint8_t* a1 = slots->acquire(0, 2, err);
    expect(a1 != nullptr && intact(*slots, 0, 2), "acquire loads and holds a miss");
    for (int e = 0; e < experts; ++e) {
        const int32_t one[1] = {e};
        expect(slots->materialize_batch(1, one, 1, err), "load around the held expert");
    }
    expect(slots->resident(0, 2) && intact(*slots, 0, 2), "a held expert's slot is never reused");
    slots->release_hold(0, 2);

    fixed.reset();
    slots.reset();
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (bad == 0) std::printf("xeno_nvme_slots: PASS\n");
    return bad == 0 ? 0 : 1;
}
