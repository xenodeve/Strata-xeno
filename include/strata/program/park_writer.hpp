// strata/program/park_writer.hpp - #145: one background write of a parked conversation cache slot.  A slot switch
// wrote the parked conversation's checkpoints to its file on the serve thread (127-497 ms of every switch); the write
// now runs here, and join() - before anything reads the slot's file or vectors again - returns its result.
#pragma once

#include <functional>
#include <thread>
#include <utility>

namespace strata::program {

class ParkWriter {
public:
    ParkWriter() = default;
    ParkWriter(const ParkWriter&) = delete;
    ParkWriter& operator=(const ParkWriter&) = delete;
    ~ParkWriter() { (void) join(); }

    /// Run `write` on a thread (a write still running is joined first).
    void start(std::function<bool()> write) {
        (void) join();
        ok_ = false;
        thread_ = std::thread([this, w = std::move(write)] { ok_ = w(); });
    }
    /// Wait for the write; true when there was none or it succeeded.  The writer is idle afterwards.
    bool join() {
        if (!thread_.joinable()) return true;
        thread_.join();
        return ok_;
    }
    bool busy() const { return thread_.joinable(); }

private:
    std::thread thread_;
    bool ok_ = true;   // written by the thread, read after its join
};

}  // namespace strata::program
