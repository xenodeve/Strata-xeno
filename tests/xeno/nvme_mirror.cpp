// #62 (#11 mechanism 2): byte-identical copies of the expert source on more drives.  With a mirror, a layer's NVMe
// misses are spread over the copies, one expert's ranges on one copy; the bytes are the same either way.  A copy
// whose bytes differ from its source is refused (a wrong mirror is a wrong model, not noise); a directory without
// the file leaves that source on one drive.
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {
constexpr int LAYERS = 2, EXPERTS = 6;
uint8_t mark(int l, int e) { return (uint8_t) (0x13 * (l * EXPERTS + e) + 1); }

// a canonical pack of LAYERS x EXPERTS blobs whose bytes name their expert
void write_pack(const fs::path& dir) {
    fs::create_directories(dir);
    const uint64_t blob = strata::kernels::cpu::BLOB;
    std::ofstream out(dir / "experts.bin", std::ios::binary);
    std::vector<uint8_t> b(blob);
    for (int l = 0; l < LAYERS; ++l)
        for (int e = 0; e < EXPERTS; ++e) {
            for (uint64_t i = 0; i < blob; ++i) b[i] = (uint8_t) (mark(l, e) + i);
            out.write((const char*) b.data(), (std::streamsize) blob);
        }
}

// a source in capacity mode with only expert (0,0) in RAM: everything else is read from NVMe
bool open_source(strata::core::ArenaExpertSource& s, const fs::path& dir, std::string& err) {
    if (!strata::kernels::cpu::expert_layout_load(dir.string(), LAYERS, EXPERTS, err)) return false;
    if (!s.open(dir.string(), LAYERS, EXPERTS, 1, err, /*pin_for_cuda=*/false, /*defer_load=*/true)) return false;
    std::vector<int32_t> order;
    for (int i = 0; i < LAYERS * EXPERTS; ++i) order.push_back(i);
    s.set_capacity(strata::kernels::cpu::BLOB, order);
    return s.load_rest(1, err);
}
}  // namespace

int main() {
    const fs::path root = fs::temp_directory_path() /
        ("strata-mirror-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const fs::path a = root / "a", b = root / "b", bad = root / "bad", none = root / "none";
    write_pack(a);
    fs::create_directories(b);
    fs::copy_file(a / "experts.bin", b / "experts.bin");
    fs::create_directories(bad);
    fs::copy_file(a / "experts.bin", bad / "experts.bin");
    {   // one flipped byte in the first page, which every identity check reads
        std::fstream f(bad / "experts.bin", std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(100);
        f.put((char) 0x5a);
    }
    fs::create_directories(none);
    const uint64_t blob = strata::kernels::cpu::BLOB;
    int fails = 0;
    auto expect = [&](bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++fails; } };
    const int32_t miss[] = {1, 2, 3, 4, 5};
    std::string err;

    {   // a good mirror: right bytes, and both copies serve
        strata::core::ArenaExpertSource s;
        if (!open_source(s, a, err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
        s.add_mirror(b.string());
        expect(s.materialize_batch(0, miss, 5, err), "materialize with a mirror");
        bool bytes_ok = true;
        for (int e : miss) {
            const uint8_t* p = s.blob(0, e);
            bytes_ok &= p != nullptr && p[0] == mark(0, e) && p[blob - 1] == (uint8_t) (mark(0, e) + blob - 1);
        }
        expect(bytes_ok, "every expert read through the mirror has its own bytes");
        const auto st = s.nvme_file_stats();
        int serving = 0;
        int64_t reads = 0;
        for (const auto& f : st) { serving += f.reads > 0; reads += f.reads; }
        expect(st.size() == 2 && serving == 2, "both copies received reads");
        expect(reads == 5, "one read per expert across the copies");
        // equal blobs, least-queued first, ties to the source: source, copy, source, copy, source
        expect(st.size() == 2 && st[0].reads == 3 && st[1].reads == 2, "the split is 3 / 2, source first");
        expect(st.size() == 2 && st[1].batches == 1 && st[1].max_us > 0, "the copy's batch and read latency are timed");
        // #82: a decode layer misses ~0.3 experts, so most batches hold one; the tie must not always go to the source.
        // Each batch starts its tie-break one copy further on: two single-expert batches use both copies.
        const int32_t one_a[1] = {1}, one_b[1] = {2};
        expect(s.materialize_batch(1, one_a, 1, err) && s.materialize_batch(1, one_b, 1, err), "two single loads");
        const auto st2 = s.nvme_file_stats();
        expect(st2.size() == 2 && st2[0].reads == 4 && st2[1].reads == 3, "single-expert batches alternate the copies");
        for (const auto& f : st)
            std::printf("  %s: %lld experts, %llu bytes\n", f.path.c_str(), (long long) f.reads,
                        (unsigned long long) f.bytes);
    }
    {   // a copy that differs is refused, loudly
        strata::core::ArenaExpertSource s;
        if (!open_source(s, a, err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
        s.add_mirror(bad.string());
        err.clear();
        expect(!s.materialize_batch(0, miss, 5, err) && err.find("mirror") != std::string::npos,
               "a mismatching mirror fails the read and says so");
    }
    {   // a copy that differs only near its end is refused too (the last MiB is always compared)
        const fs::path tail = root / "tail";
        fs::create_directories(tail);
        fs::copy_file(a / "experts.bin", tail / "experts.bin");
        const uint64_t size = fs::file_size(tail / "experts.bin");
        std::fstream f(tail / "experts.bin", std::ios::in | std::ios::out | std::ios::binary);
        f.seekp((std::streamoff) (size - 1000));
        f.put((char) 0x5a);
        f.close();
        strata::core::ArenaExpertSource s;
        if (!open_source(s, a, err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
        s.add_mirror(tail.string());
        err.clear();
        expect(!s.materialize_batch(0, miss, 5, err) && err.find("mirror") != std::string::npos,
               "a mirror differing in its last MiB is refused");
    }
    {   // a directory without the file: the source stays on one drive
        strata::core::ArenaExpertSource s;
        if (!open_source(s, a, err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
        s.add_mirror(none.string());
        expect(s.materialize_batch(1, miss, 5, err), "materialize without a usable mirror");
        const auto st = s.nvme_file_stats();
        expect(st.size() == 1 && st[0].reads == 5, "one copy served everything");
        expect(s.blob(1, 3) && s.blob(1, 3)[7] == (uint8_t) (mark(1, 3) + 7), "bytes without a mirror");
    }
    std::error_code ec;
    fs::remove_all(root, ec);
    std::printf(fails ? "nvme_mirror: %d failures\n" : "nvme_mirror: ok\n", fails);
    return fails ? 1 : 0;
}
