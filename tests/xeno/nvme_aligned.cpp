// tests/xeno/nvme_aligned.cpp - #81: an aligned expert pack (`--expert-pack`, written by strata --write-expert-pack) stores
// each expert's blob contiguous (gate | up | down, the arena's layout) at a 4 KiB-aligned offset, padded to a 4 KiB
// stride.  A capacity-mode load then reads one aligned request per expert straight into its slab slot (the slot is
// page-aligned and its stride is the pack's): no bounce copy, one request instead of a GGUF's three role slices.
//
// The pack's format, which --write-expert-pack writes and ArenaExpertSource::set_expert_pack checks:
//   bytes 0..7   "STRAPACK"
//   then u32 version (1), u32 n_layers, u32 n_expert, u32 alignment (4096)
//   then per layer: u64 offset, u64 stride, u64 blob bytes
//   the experts: layer l, expert x at offset[l] + x * stride[l] (blob bytes, then zeros to the stride)
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static uint8_t role_byte(int l, int r, int x, uint64_t i) { return (uint8_t) (31 * l + 17 * r + 7 * x + i * 13 + 1); }

int main() {
    const fs::path dir = fs::temp_directory_path() /
        ("strata-nvme-aligned-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(dir);
    std::string err;
    const int layers = 2, experts = 4;
    strata::kernels::cpu::NativeFmt f;
    if (!strata::kernels::cpu::native_fmt(8, 8, strata::kernels::cpu::H, strata::kernels::cpu::FF, f, err)) {
        std::fprintf(stderr, "native_fmt: %s\n", err.c_str()); return 1;
    }
    const uint64_t blob = f.bytes, per[3] = {f.up_off, f.up_off, blob - f.down_off}, at[3] = {0, f.up_off, f.down_off};
    // a GGUF-shaped file: per layer three role tensors, each the n_expert slices back to back, at odd offsets
    const fs::path gguf = dir / "model-00001-of-00001.gguf";
    uint64_t go[layers][3];
    {
        std::ofstream g(gguf, std::ios::binary);
        std::vector<uint8_t> pad(777, 0xEE);
        g.write((const char*) pad.data(), (std::streamsize) pad.size());
        for (int l = 0; l < layers; ++l)
            for (int r = 0; r < 3; ++r) {
                go[l][r] = (uint64_t) g.tellp();
                std::vector<uint8_t> t(per[r]);
                for (int x = 0; x < experts; ++x) {
                    for (uint64_t i = 0; i < per[r]; ++i) t[i] = role_byte(l, r, x, i);
                    g.write((const char*) t.data(), (std::streamsize) t.size());
                }
                g.write((const char*) pad.data(), 123);
            }
    }
    {
        std::ofstream txt(dir / "native_experts.txt");
        for (int l = 0; l < layers; ++l)
            txt << l << " 8 8 " << (uint64_t) l * experts * blob << " " << blob << " " << go[l][0] << " " << go[l][1]
                << " " << go[l][2] << "\n";
    }
    // the aligned pack
    const uint64_t A = 4096, stride = (blob + A - 1) / A * A;
    const fs::path pack = dir / "experts.aligned";
    {
        std::ofstream p(pack, std::ios::binary);
        std::vector<uint8_t> head(A, 0);
        std::memcpy(head.data(), "STRAPACK", 8);
        const uint32_t h32[4] = {1, (uint32_t) layers, (uint32_t) experts, (uint32_t) A};
        std::memcpy(head.data() + 8, h32, sizeof h32);
        for (int l = 0; l < layers; ++l) {
            const uint64_t row[3] = {A + (uint64_t) l * experts * stride, stride, blob};
            std::memcpy(head.data() + 24 + l * 24, row, sizeof row);
        }
        p.write((const char*) head.data(), (std::streamsize) A);
        std::vector<uint8_t> e(stride);
        for (int l = 0; l < layers; ++l)
            for (int x = 0; x < experts; ++x) {
                std::fill(e.begin(), e.end(), 0);
                for (int r = 0; r < 3; ++r)
                    for (uint64_t i = 0; i < per[r]; ++i) e[at[r] + i] = role_byte(l, r, x, i);
                p.write((const char*) e.data(), (std::streamsize) stride);
            }
    }
    if (!strata::kernels::cpu::expert_layout_load(dir.string(), layers, experts, err)) {
        std::fprintf(stderr, "layout: %s\n", err.c_str()); return 1;
    }
    int bad = 0;
    auto expect = [&](bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++bad; } };
    auto make = [&](std::unique_ptr<strata::core::ArenaExpertSource>& s, bool with_pack) -> bool {
        s = std::make_unique<strata::core::ArenaExpertSource>();
        s->set_gguf(gguf.string());
        if (!s->open(dir.string(), layers, experts, 1, err, /*pin_for_cuda=*/false, /*defer_load=*/true)) return false;
        if (with_pack && !s->set_expert_pack(pack.string(), err)) return false;
        s->set_capacity(2 * blob, {0, 4});
        return s->load_rest(1, err);
    };
    std::unique_ptr<strata::core::ArenaExpertSource> plain, packed;
    if (!make(plain, false)) { std::fprintf(stderr, "open plain: %s\n", err.c_str()); return 1; }
    if (!make(packed, true)) { std::fprintf(stderr, "open packed: %s\n", err.c_str()); return 1; }

    const int32_t miss[3] = {1, 2, 3};
    expect(plain->materialize_batch(0, miss, 3, err), "plain load");
    expect(packed->materialize_batch(0, miss, 3, err), "packed load");
    for (int x = 1; x < 4; ++x) {
        const uint8_t* a = plain->blob(0, x);
        const uint8_t* b = packed->blob(0, x);
        expect(a != nullptr && b != nullptr && std::memcmp(a, b, blob) == 0, "the pack's bytes are the GGUF's");
        expect(b != nullptr && b[at[2] + 5] == role_byte(0, 2, x, 5), "the down slice lands at its offset");
    }
    const auto st = packed->nvme_file_stats();
    expect(st.size() == 1 && st[0].path == pack.string() && st[0].reads == 3, "one read per expert, from the pack");
    expect(packed->nvme_stages().copy_ms == 0.0, "slab slots are read straight into: nothing copied");
    expect(plain->nvme_stages().copy_ms > 0.0, "the GGUF path copies from the bounce buffer");

    // read_experts_to's destinations are not slab slots (the tail file's check and build, the lent-slot refill): a
    // pack's reads land exactly in them - an unaligned one through the bounce buffer, none past its blob (2026-10-02
    // /code-review: `direct` was decided from the pack and the slab alone, so these reads failed or overran)
    {
        auto want = [&](int l, int x) {
            std::vector<uint8_t> e(blob, 0);
            for (int r = 0; r < 3; ++r)
                for (uint64_t i = 0; i < per[r]; ++i) e[at[r] + i] = role_byte(l, r, x, i);
            return e;
        };
        std::vector<uint8_t> raw(2 * stride + 3 * A, 0xCD);
        uint8_t* al = (uint8_t*) (((uintptr_t) raw.data() + A - 1) & ~(uintptr_t) (A - 1));
        uint8_t* dsts[2] = {al, al + blob + 64};   // page-aligned, then unaligned, 64 bytes of guard between
        const int32_t ls[2] = {1, 1}, xs[2] = {0, 2};
        err.clear();
        expect(packed->read_experts_to(ls, xs, 2, dsts, err), "read_experts_to from the pack into plain buffers");
        expect(std::memcmp(dsts[0], want(1, 0).data(), blob) == 0, "an aligned plain buffer gets its expert");
        expect(std::memcmp(dsts[1], want(1, 2).data(), blob) == 0, "an unaligned plain buffer gets its expert");
        bool guard = true;
        for (int i = 0; i < 64; ++i) guard &= al[blob + i] == 0xCD;
        expect(guard, "nothing is written past a plain buffer's blob");
    }

    // the engine writes the pack itself (`strata --write-expert-pack`): byte for byte the format above
    const fs::path written = dir / "written.aligned";
    expect(plain->write_expert_pack(written.string(), err), "write_expert_pack");
    {
        std::ifstream a(pack, std::ios::binary), b(written, std::ios::binary);
        const std::string sa((std::istreambuf_iterator<char>(a)), {}), sb((std::istreambuf_iterator<char>(b)), {});
        expect(!sa.empty() && sa == sb, "the written pack is the format, byte for byte");
    }

    // a pack of another geometry is refused, loudly (the sources hold the pack open: close them first)
    plain.reset();
    packed.reset();
    {
        std::fstream p(pack, std::ios::in | std::ios::out | std::ios::binary);
        const uint32_t wrong = experts + 1;
        p.seekp(16);
        p.write((const char*) &wrong, 4);
    }
    strata::core::ArenaExpertSource other;
    other.set_gguf(gguf.string());
    expect(other.open(dir.string(), layers, experts, 1, err, false, true), "open");
    err.clear();
    expect(!other.set_expert_pack(pack.string(), err) && err.find("expert pack") != std::string::npos,
           "a pack for another geometry is refused");
    {   // the right geometry, another alignment: refused too (its strides were written for that alignment)
        std::fstream p(pack, std::ios::in | std::ios::out | std::ios::binary);
        const uint32_t right = experts, wrong = 8192;
        p.seekp(16);
        p.write((const char*) &right, 4);
        p.write((const char*) &wrong, 4);
    }
    err.clear();
    expect(!other.set_expert_pack(pack.string(), err) && err.find("alignment") != std::string::npos,
           "a pack of another alignment is refused");
    other.close();

    std::error_code ec;
    fs::remove_all(dir, ec);
    if (bad == 0) std::printf("xeno_nvme_aligned: PASS\n");
    return bad == 0 ? 0 : 1;
}
