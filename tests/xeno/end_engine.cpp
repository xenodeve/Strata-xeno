// #203: the engine's failure paths end it through core::end_engine.  A display-VRAM breach ended it with _Exit
// while a verify window's kernel spun on a host flag: the process sat in its exit for minutes with the 5060 at 100 %.
// end_engine must arm the kill deadline, release the GPU's spin waits (#267), then end the process with the code given
// and without running exit-time handlers (TerminateProcess, not ExitProcess).  The test runs itself as a child
// ("child" argument) and checks the child's output and exit code.
#include "strata/core/end_engine.hpp"
#include "strata/core/progress.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#if defined(_WIN32)
#include <windows.h>
#endif

namespace {
void armed() { std::printf("ARMED\n"); std::fflush(stdout); }
void released(std::FILE*) { std::printf("RELEASED\n"); std::fflush(stdout); }
void at_exit() { std::printf("ATEXIT\n"); std::fflush(stdout); }
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "child") {
        std::atexit(at_exit);
        strata::core::set_end_engine_arm(armed);
        strata::core::release_gpu_fn().store(released);
        strata::core::end_engine(7, "strata: test end (#203)");
    }
#if defined(_WIN32)
    std::string cmd = std::string("\"") + argv[0] + "\" child";
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE rd = nullptr, wr = nullptr;
    CreatePipe(&rd, &wr, &sa, 0);
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    si.hStdError = wr;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        std::printf("cannot start the child - FAIL\n");
        return 1;
    }
    CloseHandle(wr);
    std::string out;
    char buf[256];
    DWORD n = 0;
    while (ReadFile(rd, buf, sizeof buf, &n, nullptr) && n > 0) out.append(buf, n);
    const bool ended = WaitForSingleObject(pi.hProcess, 10000) == WAIT_OBJECT_0;
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    int bad = 0;
    auto check = [&](bool ok, const char* what) { std::printf("%s - %s\n", what, ok ? "PASS" : "FAIL"); bad += !ok; };
    const size_t a = out.find("ARMED"), r = out.find("RELEASED");
    check(ended && code == 7, "the child ends with the code given");
    check(out.find("strata: test end (#203)") != std::string::npos, "it prints why");
    check(a != std::string::npos && r != std::string::npos && a < r, "it arms the deadline, then releases the GPU's waits");
    check(out.find("ATEXIT") == std::string::npos, "no exit-time handler runs (not ExitProcess)");
    return bad == 0 ? 0 : 1;
#else
    return 0;
#endif
}
