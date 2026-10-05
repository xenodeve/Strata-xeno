// include/strata/platform/direct_file.hpp - plan v0.3 P1/P2: unbuffered asynchronous file reads.
//
// The n-gram table (26.8 GiB) stays on the SSD and must never occupy RAM, including the OS file cache. A memory
// map cannot promise that; an unbuffered read can. On Windows this is FILE_FLAG_NO_BUFFERING | OVERLAPPED with
// an I/O completion port; on Linux, O_DIRECT preads.  Both issue the reads from a small thread pool (submit only
// queues), so many reads are in flight: STRATA_IO_THREADS sets its size (default 4 on Windows, 16 on Linux).
//
// Contract of every read: offset, length and buffer address are multiples of `alignment()` (4096 here).
// Reads past end of file return the bytes that exist; `Completion::bytes` says how many.
#pragma once

#include <cstdint>
#include <string>

namespace strata::platform {

struct Completion {
    uint64_t tag = 0;      ///< the caller's tag from `submit`
    uint32_t bytes = 0;    ///< bytes transferred (short only at end of file)
    bool ok = false;
};

class DirectFile {
public:
    DirectFile();
    ~DirectFile();
    DirectFile(const DirectFile&) = delete;
    DirectFile& operator=(const DirectFile&) = delete;

    /// `issuers`: the threads that issue the reads when STRATA_IO_THREADS is not set (it overrides every file);
    /// <= 0 = the platform default.  #139: a reader of a few large contiguous pieces asks for 1 (measured: #139).
    bool open(const std::string& path, std::string& err, int issuers = 0);
    void close();
    bool is_open() const;
    uint64_t size() const;
    int issuers() const;   ///< the open file's issuing threads
    static constexpr uint32_t alignment() { return 4096; }

    /// Queue one read. Returns false (and sets `err`) if the request could not be queued; a queued request
    /// always produces exactly one completion.
    bool submit(uint64_t offset, void* buffer, uint32_t length, uint64_t tag, std::string& err);

    /// Wait for up to `max` completions; returns how many were written to `out`. `timeout_ms` < 0 waits
    /// forever; 0 polls. A `wake()` shows up as one completion with tag `WAKE_TAG`.
    int wait(Completion* out, int max, int timeout_ms);

    /// Make a thread blocked in `wait` return (thread-safe). Used to hand it newly queued work.
    static constexpr uint64_t WAKE_TAG = ~0ull;
    void wake();

    /// #185: this thread's own handle and aligned buffer, for a reader called from threads that come and go (a
    /// prompt tier's stager).  A thread's exit never destroys them: a thread_local's destructor runs inside
    /// LdrShutdownThread with the Windows loader lock held, and close() joins the issuing threads, whose own exit
    /// needs that lock.  A thread that ends before the process calls release_this_thread() on its way out (it closes
    /// the handle and frees the buffer in an ordinary scope); one that does not keeps them to the process's end.
    /// One handle per thread, whatever it is opened on: a thread that reads two files needs its own DirectFiles.
    static DirectFile& for_this_thread();
    static void* buffer_for_this_thread(size_t bytes);   ///< aligned, at least `bytes`, kept between calls
    static void release_this_thread();

    /// Allocation helpers for aligned read buffers (page-aligned, never touched by the cache manager).
    static void* alloc_aligned(size_t bytes);
    static void free_aligned(void* p);

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

/// Monotonic time in microseconds, for latency accounting.
double now_us();

}  // namespace strata::platform
