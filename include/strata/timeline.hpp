// include/strata/timeline.hpp - #33: one timeline of the whole pipeline.
//
// STRATA_TIMELINE=<file> records every host thread's spans (the prompt path's layers and copy issues, the verify
// window's waits, the CPU pool's phases and workers, the 4070's launches and waits) and the GPU lanes (device
// times of events, placed on the host clock by strata/timeline_gpu.hpp) into one Chrome trace JSON file.  Open it
// in ui.perfetto.dev, or read the budget with tests/xeno/perf/timeline.py.
//
// Unset, every call is one predictable branch on a static bool.  Names must be string literals (they are stored as
// pointers); what varies goes in the two integer arguments (a layer, an entry index, a count).
//
// The clock is std::chrono::steady_clock in microseconds since its epoch: QueryPerformanceCounter on Windows, the
// same timebase as Python's time.perf_counter(), so serve/server.py's spans line up with the engine's.
#pragma once

#include <chrono>
#include <cstdint>

namespace strata::timeline {

/// STRATA_TIMELINE names a file (read once).
bool enabled();
/// The timeline clock, in microseconds.
double now_us();
inline double us(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::micro>(t.time_since_epoch()).count();
}
/// The calling thread's lane name in the viewer (copied).
void name_thread(const char* name);
/// A span of the calling thread.
void complete(const char* name, double t0_us, double t1_us, int64_t a = -1, int64_t b = -1);
inline void complete(const char* name, std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1,
                     int64_t a = -1, int64_t b = -1) {
    if (enabled()) complete(name, us(t0), us(t1), a, b);
}
/// A virtual lane (a GPU stream, a copy engine): created on first use, the same id for the same name.
int lane(const char* name);
/// A span on a virtual lane (from any thread).
void complete_on(int lane, const char* name, double t0_us, double t1_us, int64_t a = -1, int64_t b = -1);
/// A point event of the calling thread.
void instant(const char* name, int64_t a = -1, int64_t b = -1);
/// Appends every event recorded since the last flush to the file.  Safe to call from any thread, repeatedly (the
/// engine flushes after each request and at exit).  The file is a JSON array left open: each record ends with
/// ",\n", and a reader closes it (tests/xeno/perf/timeline.py load()).
void flush();

/// A scoped span of the calling thread.
struct Span {
    const char* name;
    int64_t a, b;
    double t0;
    explicit Span(const char* n, int64_t a_ = -1, int64_t b_ = -1) : name(n), a(a_), b(b_), t0(enabled() ? now_us() : 0) {}
    ~Span() {
        if (enabled()) complete(name, t0, now_us(), a, b);
    }
    Span(const Span&) = delete;
    Span& operator=(const Span&) = delete;
};

}  // namespace strata::timeline
