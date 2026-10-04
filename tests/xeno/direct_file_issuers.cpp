// #139: a reader asks for its issuing-thread count at open (the tail refill asks for 1); STRATA_IO_THREADS, when set,
// overrides every file (the operator's A/B lever).  Reads through one issuer still land byte for byte.
#include "strata/platform/direct_file.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>   // _putenv_s
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using strata::platform::Completion;
using strata::platform::DirectFile;

namespace {
int bad = 0;
void check(bool ok, const char* what) {
    std::printf("%s - %s\n", what, ok ? "PASS" : "FAIL");
    bad += !ok;
}
uint8_t pattern(size_t i) { return (uint8_t) (i * 131 + (i >> 12)); }
}  // namespace

int main() {
    const std::string path = (std::filesystem::temp_directory_path() / "strata-direct-issuers-test.bin").string();
    constexpr uint32_t kPiece = 1u << 20;   // four 1 MiB pieces of pattern()
    constexpr int kPieces = 4;
    {
        std::vector<uint8_t> bytes((size_t) kPiece * kPieces);
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = pattern(i);
        std::ofstream(path, std::ios::binary).write((const char*) bytes.data(), (std::streamsize) bytes.size());
    }
    std::string err;
    {   // no STRATA_IO_THREADS: the caller's count, else the default (4 on Windows)
        _putenv_s("STRATA_IO_THREADS", "");
        DirectFile a, b;
        check(a.open(path, err, 1) && a.issuers() == 1, "open(path, err, 1) runs one issuer");
        check(b.open(path, err) && b.issuers() == 4, "open(path, err) runs the default 4");
    }
    {   // STRATA_IO_THREADS overrides every file, the asked count too: the A/B lever back to a pool
        _putenv_s("STRATA_IO_THREADS", "3");
        DirectFile a, b;
        check(a.open(path, err, 1) && a.issuers() == 3, "STRATA_IO_THREADS=3 overrides open(path, err, 1)");
        check(b.open(path, err) && b.issuers() == 3, "STRATA_IO_THREADS=3 sets open(path, err)");
        _putenv_s("STRATA_IO_THREADS", "");
    }
    DirectFile f;
    check(f.open(path, err, 1) && f.issuers() == 1, "one issuer for the read below");
    auto* buf = (uint8_t*) DirectFile::alloc_aligned((size_t) kPiece * kPieces);
    bool ok = buf != nullptr;
    for (int q = 0; ok && q < kPieces; ++q) ok = f.submit((uint64_t) q * kPiece, buf + (size_t) q * kPiece, kPiece, (uint64_t) q, err);
    int got = 0;
    while (ok && got < kPieces) {
        Completion c[8];
        const int n = f.wait(c, 8, 5000);
        if (n <= 0) { ok = false; break; }
        for (int i = 0; i < n; ++i) {
            ok = ok && c[i].ok && c[i].bytes == kPiece && c[i].tag < (uint64_t) kPieces;
            ++got;
        }
    }
    for (size_t i = 0; ok && i < (size_t) kPiece * kPieces; ++i) ok = buf[i] == pattern(i);
    check(ok, "one issuer reads four 1 MiB pieces byte for byte");
    DirectFile::free_aligned(buf);
    f.close();
    std::filesystem::remove(path);
    return bad == 0 ? 0 : 1;
}
