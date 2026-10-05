// include/strata/core/end_engine.hpp - #203: the one way the engine ends itself from a failure path.
#pragma once

namespace strata::core {

/// A function that arms the kill deadline: the watchdog registers one at startup (a thread created then, which
/// TerminateProcess-es a set time after it is armed - a thread created on the failure path could not start, #185).
void set_end_engine_arm(void (*arm)());

/// #203: end the engine now, with `code`, after printing `why` (a "strata:" line the server shows as the cause).
/// Arms the kill deadline first, releases the GPU's spin waits (#267: a verify window's kernel waiting on a host flag
/// nobody will raise keeps the GPU busy and the process from going away), then TerminateProcess - never ExitProcess,
/// whose DLL detach waits for a GPU still running a kernel and needs the loader lock (#185).
[[noreturn]] void end_engine(int code, const char* why);

}  // namespace strata::core
