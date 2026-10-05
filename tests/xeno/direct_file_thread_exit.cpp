// #185: a thread that read through DirectFile::for_this_thread() must be able to exit.  The prompt path's stager
// threads read lent tail experts through it and end with their tier; a handle destroyed at thread exit closed its
// issuing threads under the Windows loader lock and the join never returned (the D2x hang of #184).
//   1. a reader that calls release_this_thread() on its way out exits, and its handle is closed (its file can be
//      deleted, which an open handle refuses on Windows) - the stager's path, so a tier rebuilt again and again keeps
//      no handle or issuing thread per old stager thread;
//   2. a reader that does not release still exits (it keeps its handle; nothing runs at its exit).
// A reader thread is joined on a helper; past the deadline the test fails and ends the process (it cannot join it).
#include "strata/platform/direct_file.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>   // _putenv_s, _Exit
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#if defined(_WIN32)
#include <windows.h>
#endif

using strata::platform::Completion;
using strata::platform::DirectFile;

namespace {
int bad = 0;
void check(bool ok, const char* what) {
    std::printf("%s - %s\n", what, ok ? "PASS" : "FAIL");
    std::fflush(stdout);
    bad += !ok;
}
[[noreturn]] void end(int code) {
    std::fflush(stdout);
#if defined(_WIN32)
    TerminateProcess(GetCurrentProcess(), (UINT) code);   // ExitProcess would need the loader lock a stuck thread holds
#endif
    std::_Exit(code);
}
std::string make_file(const char* name) {
    const std::string path = (std::filesystem::temp_directory_path() / name).string();
    std::ofstream(path, std::ios::binary).write(std::string(4096, 'a').data(), 4096);
    return path;
}
// one 4 KiB read through this thread's handle and buffer, then (release) give them back
bool read_once(const std::string& path, bool release) {
    DirectFile& f = DirectFile::for_this_thread();
    std::string err;
    void* b = DirectFile::buffer_for_this_thread(4096);
    Completion c;
    const bool ok = f.open(path, err, 1) && b != nullptr && f.submit(0, b, 4096, 0, err) && f.wait(&c, 1, 5000) == 1 &&
                    c.ok && c.bytes == 4096 && ((const char*) b)[4095] == 'a';
    if (release) DirectFile::release_this_thread();
    return ok;
}
// run `body` on a thread and join it within 10 s
bool joins(const std::function<void()>& body) {
    std::thread reader(body);
    std::atomic<bool> joined{false};
    std::thread joiner([&] { reader.join(); joined = true; });
    const auto t0 = std::chrono::steady_clock::now();
    while (!joined && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10))
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (!joined) return false;
    joiner.join();
    return true;
}
}  // namespace

int main() {
#if defined(_WIN32)
    _putenv_s("STRATA_IO_THREADS", "");
#endif
    {
        const std::string path = make_file("strata-direct-thread-exit-released.bin");
        bool ok = false;
        const bool j = joins([&] { ok = read_once(path, true); });
        check(ok, "a released reader read its 4 KiB through its own handle and buffer");
        check(j, "a reader that released its handle exits and joins within 10 s");
        if (!j) end(1);
        std::error_code ec;
        check(std::filesystem::remove(path, ec) && !ec, "its handle is closed: the file it read can be deleted");
    }
    {
        const std::string path = make_file("strata-direct-thread-exit-kept.bin");
        bool ok = false;
        const bool j = joins([&] { ok = read_once(path, false); });
        check(ok, "a reader that keeps its handle read its 4 KiB");
        check(j, "a reader that keeps its handle still exits and joins within 10 s (nothing runs at its exit)");
        if (!j) end(1);
        std::error_code ec;
        std::filesystem::remove(path, ec);   // may fail: that reader's handle stays open by design
    }
    return bad == 0 ? 0 : 1;
}
