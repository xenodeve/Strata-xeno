// include/strata/platform/crash_report.hpp - where a crashed engine was (#62 crash, 2026-10-01).
//
// A served engine died with 0xC0000005 after a 19,939-token prompt and ~700 decoded tokens. It left only its exit
// code: WER keeps at most DumpCount (10) files in %LOCALAPPDATA%\CrashDumps, and the folder was full. This prints
// the exception, the faulting module + offset and a symbolized stack of the crashing thread to stderr (the engine
// log), and writes strata-crash-<pid>.dmp in the working directory. The exception then continues to the default
// handler, so the exit code and WER are unchanged. Windows only; elsewhere a no-op.
#pragma once

namespace strata::platform {

/// Install once, at the top of main(). STRATA_TEST_CRASH=1 then crashes on purpose (the reporter's own test).
void install_crash_report();

}  // namespace strata::platform
