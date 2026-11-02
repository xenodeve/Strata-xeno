// #203: the engine's failure paths end it through core::end_engine.  A display-VRAM breach ended it with _Exit while
// a verify window's kernel spun on a host flag: the process sat in its exit for minutes.  The test runs itself as a
// child in three ways and checks each child's output, exit code and that it ended at all (a hung child is killed):
//   plain    - the deadline is armed, the GPU's waits are released, the reason is the LAST "strata:" line (the server
//              shows the last one as the cause) and no exit-time handler runs;
//   lock     - another thread holds the loader lock for good (a thread_local's destructor that joins a thread, #185):
//              ExitProcess would wait for that lock forever, end_engine must not;
//   deadline - the release itself never returns: the deadline started at startup ends the process;
//   return   - main returns with the deadline started and never armed, as a refused boot does: exit-time destruction
//              must not wait on the deadline thread (glibc's condition_variable destructor waited for its waiter, and a
//              refused MIP bundle left the engine hung for good on Linux, #292).
// lock is Windows only (the loader lock is Windows'); the others run everywhere.
#include "strata/core/end_engine.hpp"
#include "strata/core/progress.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#if defined(_WIN32)
#include <windows.h>
#else
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
void released(std::FILE* f) { std::fprintf(f, "strata: released (test)\n"); std::fflush(f); }
void released_never(std::FILE*) { for (;;) std::this_thread::sleep_for(std::chrono::seconds(1)); }
void at_exit() { std::printf("ATEXIT\n"); std::fflush(stdout); }

struct LoaderLockHolder {   // #185's shape: run at thread exit, under the loader lock, it waits for a thread that
    ~LoaderLockHolder() {   // cannot start without that lock
        std::thread t([] {});
        t.join();
    }
};

int child(const std::string& mode) {
    std::atexit(at_exit);
    strata::core::start_end_engine_deadline(1500);
    if (mode == "return") {   // as late as a refused boot returns: the deadline thread is waiting by then
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        return 5;
    }
    strata::core::release_gpu_fn().store(mode == "deadline" ? released_never : released);
    if (mode == "lock") {
        std::thread([] {
            thread_local LoaderLockHolder h;
            volatile const void* p = &h;
            (void) p;
        }).detach();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));   // its exit now holds the loader lock
    }
    strata::core::end_engine(7, "strata: test end (#203)");
}

#if defined(_WIN32)
struct Run { bool ended; DWORD code; std::string out; };
Run run_child(const char* self, const char* mode) {
    std::string cmd = std::string("\"") + self + "\" " + mode;
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
    Run r{false, 0, ""};
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) return r;
    CloseHandle(wr);
    r.ended = WaitForSingleObject(pi.hProcess, 8000) == WAIT_OBJECT_0;
    if (!r.ended) TerminateProcess(pi.hProcess, 99);   // hung: end it so the pipe closes
    char buf[256];
    DWORD n = 0;
    while (ReadFile(rd, buf, sizeof buf, &n, nullptr) && n > 0) r.out.append(buf, n);
    GetExitCodeProcess(pi.hProcess, &r.code);
    CloseHandle(rd);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return r;
}
#else
struct Run { bool ended; int code; std::string out; };
Run run_child(const char* self, const char* mode) {
    int fd[2];
    Run r{false, -1, ""};
    if (pipe(fd) != 0) return r;
    const pid_t pid = fork();
    if (pid == 0) {
        dup2(fd[1], 1);
        dup2(fd[1], 2);
        close(fd[0]);
        close(fd[1]);
        execl(self, self, mode, (char*) nullptr);
        _exit(127);
    }
    close(fd[1]);
    int status = 0;
    for (int i = 0; i < 80 && !r.ended; ++i) {
        if (waitpid(pid, &status, WNOHANG) == pid) r.ended = true;
        else std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!r.ended) {   // hung: end it so the pipe closes
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
    }
    char buf[256];
    ssize_t n = 0;
    while ((n = read(fd[0], buf, sizeof buf)) > 0) r.out.append(buf, (size_t) n);
    close(fd[0]);
    if (WIFEXITED(status)) r.code = WEXITSTATUS(status);
    return r;
}
#endif
std::string last_strata_line(const std::string& out) {
    std::string last;
    size_t pos = 0;
    while (pos < out.size()) {
        size_t e = out.find('\n', pos);
        if (e == std::string::npos) e = out.size();
        std::string line = out.substr(pos, e - pos);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("strata", 0) == 0) last = line;
        pos = e + 1;
    }
    return last;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) return child(argv[1]);
    int bad = 0;
    auto check = [&](bool ok, const char* what) { std::printf("%s - %s\n", what, ok ? "PASS" : "FAIL"); bad += !ok; };
    const Run plain = run_child(argv[0], "plain");
    check(plain.ended && plain.code == 7, "plain: ends with the code given");
    check(plain.out.find("strata: released (test)") != std::string::npos, "plain: releases the GPU's waits");
    check(last_strata_line(plain.out).find("test end (#203)") != std::string::npos,
          "plain: the reason is the last strata line");
    check(plain.out.find("ATEXIT") == std::string::npos, "plain: no exit-time handler runs");
#if defined(_WIN32)
    const Run lock = run_child(argv[0], "lock");
    check(lock.ended && lock.code == 7, "lock: ends with the loader lock held by another thread");
#endif
    const Run dl = run_child(argv[0], "deadline");
    check(dl.ended && dl.code == 7, "deadline: a release that never returns is ended by the deadline");
    const Run ret = run_child(argv[0], "return");
    check(ret.ended && ret.code == 5, "return: a refused boot that returns from main ends (#292)");
    check(ret.out.find("ATEXIT") != std::string::npos, "return: the ordinary exit still runs its handlers");
    return bad == 0 ? 0 : 1;
}
