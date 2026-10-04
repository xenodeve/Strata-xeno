// #145: a parked cache slot's checkpoints are written to its file on a thread, off the request's critical path (the
// write was 127-497 ms of every slot switch).  ParkWriter runs one write; join() returns its result and is the only
// way back to the slot's file and vectors; a destroyed writer joins first (a parked slot outliving the serve loop).
#include "strata/program/park_writer.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using strata::program::ParkWriter;

namespace {
int bad = 0;
void check(bool ok, const char* what) {
    std::printf("%s - %s\n", what, ok ? "PASS" : "FAIL");
    bad += !ok;
}
}  // namespace

int main() {
    {
        ParkWriter w;
        check(w.join(), "an idle writer joins as success");
    }
    {
        ParkWriter w;
        std::atomic<bool> ran{false};
        w.start([&] { std::this_thread::sleep_for(std::chrono::milliseconds(50)); ran = true; return true; });
        check(w.busy(), "a started write is busy until joined");
        check(w.join() && ran.load(), "join waits for the write and returns its result");
        check(!w.busy() && w.join(), "join again: idle, success");
    }
    {
        ParkWriter w;
        w.start([] { return false; });
        check(!w.join(), "a failed write joins as failure");
        check(w.join(), "after that the writer is idle again");
    }
    {
        ParkWriter w;
        std::atomic<int> order{0};
        w.start([&] { std::this_thread::sleep_for(std::chrono::milliseconds(30)); order = 1; return true; });
        w.start([&] { const bool first_done = order.load() == 1; order = 2; return first_done; });
        check(w.join() && order.load() == 2, "a second start waits for the first write");
    }
    std::atomic<bool> done{false};
    {
        ParkWriter w;
        w.start([&] { std::this_thread::sleep_for(std::chrono::milliseconds(30)); done = true; return true; });
    }   // destroyed while writing: joins, no std::terminate
    check(done.load(), "a destroyed writer joins its write");
    return bad == 0 ? 0 : 1;
}
