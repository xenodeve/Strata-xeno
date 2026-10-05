// include/strata/core/end_engine.hpp - #203: how the engine ends itself from a failure path.
#pragma once

namespace strata::core {

/// Start the kill deadline once, at startup: a thread that waits until a failure path arms it, then ends the process
/// `ms` later whatever else is stuck.  It must exist before the failure: a thread created then cannot start while the
/// loader lock is held (#185).  A second call does nothing.
void start_end_engine_deadline(int ms);

/// Start the deadline's clock now (end_engine does it too; a path that does slow work first - the watchdog's report -
/// arms it before that work).
void arm_end_engine();

/// #203: end the engine now, with `code`.  Prints `why` (when given) first and again last - the server shows the last
/// "strata" line as the cause - arms the deadline, releases the GPU's spin waits (#267: a kernel waiting on a host flag
/// nobody will raise keeps the GPU busy and the process from going away), then TerminateProcess: never ExitProcess,
/// whose DLL detach waits for a GPU still running a kernel and needs the loader lock (#185).  The watchdog, the
/// secondary monitor, the CPU pool's stalls and the serve loop's fatal errors end the engine through it.
[[noreturn]] void end_engine(int code, const char* why);

}  // namespace strata::core
