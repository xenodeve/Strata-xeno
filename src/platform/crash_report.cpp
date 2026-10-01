// src/platform/crash_report.cpp - see include/strata/platform/crash_report.hpp.
#include "strata/platform/crash_report.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>

namespace {

// DbgHelp is loaded on demand (no link dependency), as the stall dump in generate.cpp does
struct Dbg {
    HMODULE h = nullptr;
    decltype(&SymInitialize) init = nullptr;
    decltype(&SymSetOptions) opts = nullptr;
    decltype(&SymFromAddr) from_addr = nullptr;
    decltype(&SymGetLineFromAddr64) line = nullptr;
    decltype(&StackWalk64) walk = nullptr;
    decltype(&SymFunctionTableAccess64) table = nullptr;
    decltype(&SymGetModuleBase64) base = nullptr;
    decltype(&MiniDumpWriteDump) dump = nullptr;
};

void symbol_of(Dbg& d, HANDLE proc, DWORD64 a, char* out, size_t n) {
    out[0] = 0;
    HMODULE mod = nullptr;
    char mpath[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR) a, &mod) && mod)
        GetModuleFileNameA(mod, mpath, MAX_PATH);
    const char* mname = std::strrchr(mpath, '\\') ? std::strrchr(mpath, '\\') + 1 : mpath;
    const DWORD64 rva = mod ? a - (DWORD64) mod : 0;
    alignas(SYMBOL_INFO) char sbuf[sizeof(SYMBOL_INFO) + 512] = {};
    auto* sym = (SYMBOL_INFO*) sbuf;
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 511;
    DWORD64 disp = 0;
    IMAGEHLP_LINE64 ln = {};
    ln.SizeOfStruct = sizeof ln;
    DWORD ldisp = 0;
    if (d.from_addr && d.from_addr(proc, a, &disp, sym)) {
        if (d.line && d.line(proc, a, &ldisp, &ln))
            std::snprintf(out, n, "%s+0x%llx %s+0x%llx (%s:%lu)", mname, (unsigned long long) rva, sym->Name,
                          (unsigned long long) disp, ln.FileName, (unsigned long) ln.LineNumber);
        else
            std::snprintf(out, n, "%s+0x%llx %s+0x%llx", mname, (unsigned long long) rva, sym->Name,
                          (unsigned long long) disp);
    } else {
        std::snprintf(out, n, "%s+0x%llx", mname, (unsigned long long) rva);
    }
}

LONG WINAPI on_crash(EXCEPTION_POINTERS* ep) {
    static volatile LONG once = 0;
    if (InterlockedExchange(&once, 1) != 0) return EXCEPTION_CONTINUE_SEARCH;
    const EXCEPTION_RECORD* er = ep->ExceptionRecord;
    std::fprintf(stderr, "strata crash: exception 0x%08lx at %p, thread %lu", (unsigned long) er->ExceptionCode,
                 er->ExceptionAddress, (unsigned long) GetCurrentThreadId());
    if (er->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && er->NumberParameters >= 2)
        std::fprintf(stderr, " (%s address 0x%llx)", er->ExceptionInformation[0] == 0 ? "read of" :
                     er->ExceptionInformation[0] == 1 ? "write to" : "execute at",
                     (unsigned long long) er->ExceptionInformation[1]);
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
    Dbg d;
    d.h = LoadLibraryA("dbghelp.dll");
    if (d.h) {
        d.init = (decltype(d.init)) GetProcAddress(d.h, "SymInitialize");
        d.opts = (decltype(d.opts)) GetProcAddress(d.h, "SymSetOptions");
        d.from_addr = (decltype(d.from_addr)) GetProcAddress(d.h, "SymFromAddr");
        d.line = (decltype(d.line)) GetProcAddress(d.h, "SymGetLineFromAddr64");
        d.walk = (decltype(d.walk)) GetProcAddress(d.h, "StackWalk64");
        d.table = (decltype(d.table)) GetProcAddress(d.h, "SymFunctionTableAccess64");
        d.base = (decltype(d.base)) GetProcAddress(d.h, "SymGetModuleBase64");
        d.dump = (decltype(d.dump)) GetProcAddress(d.h, "MiniDumpWriteDump");
    }
    HANDLE proc = GetCurrentProcess();
    // the .pdb beside the build is found from the path the exe records; a copied exe also looks beside itself
    if (d.opts) d.opts(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    if (d.init) d.init(proc, nullptr, TRUE);
    if (d.walk && d.table && d.base) {
        CONTEXT ctx = *ep->ContextRecord;
        STACKFRAME64 sf = {};
        sf.AddrPC.Offset = ctx.Rip;
        sf.AddrPC.Mode = AddrModeFlat;
        sf.AddrFrame.Offset = ctx.Rbp;
        sf.AddrFrame.Mode = AddrModeFlat;
        sf.AddrStack.Offset = ctx.Rsp;
        sf.AddrStack.Mode = AddrModeFlat;
        char where[1024];
        for (int i = 0; i < 48; ++i) {
            if (!d.walk(IMAGE_FILE_MACHINE_AMD64, proc, GetCurrentThread(), &sf, &ctx, nullptr, d.table, d.base,
                        nullptr) || sf.AddrPC.Offset == 0)
                break;
            symbol_of(d, proc, sf.AddrPC.Offset, where, sizeof where);
            std::fprintf(stderr, "strata crash:   #%02d %s\n", i, where);
        }
    }
    if (d.dump) {
        char path[64];
        std::snprintf(path, sizeof path, "strata-crash-%lu.dmp", (unsigned long) GetCurrentProcessId());
        HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) {
            MINIDUMP_EXCEPTION_INFORMATION mei = {GetCurrentThreadId(), ep, FALSE};
            const BOOL ok = d.dump(proc, GetCurrentProcessId(), h,
                                   (MINIDUMP_TYPE) (MiniDumpWithThreadInfo | MiniDumpWithIndirectlyReferencedMemory),
                                   &mei, nullptr, nullptr);
            CloseHandle(h);
            char full[MAX_PATH];
            if (!GetFullPathNameA(path, MAX_PATH, full, nullptr)) std::snprintf(full, sizeof full, "%s", path);
            std::fprintf(stderr, "strata crash: %s %s\n", ok ? "wrote" : "could not write", full);
        }
    }
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

}  // namespace

namespace strata::platform {
void install_crash_report() {
    SetUnhandledExceptionFilter(on_crash);
    if (const char* t = std::getenv("STRATA_TEST_CRASH"); t != nullptr && t[0] == '1') {
        volatile int* p = nullptr;
        *p = 42;   // the reporter's own test: a write to address 0
    }
}
}  // namespace strata::platform

#else
namespace strata::platform {
void install_crash_report() {}
}  // namespace strata::platform
#endif
