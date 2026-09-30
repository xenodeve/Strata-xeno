// include/strata/timeline_gpu.hpp - #33: GPU lanes of the pipeline timeline (see strata/timeline.hpp).
//
// Device times come from timing events recorded on the streams; anchor() ties them to the host clock by recording one
// event on an idle stream between two host clock reads (the error, half that interval, is a few microseconds and is
// written to the timeline as the "anchor error us" span argument).  Events are recycled once resolved.
//
// A GpuClock belongs to one device and is used from one host thread at a time: its events are created on the device
// that is current at the first record, and every stream it records on must be on that device.
#pragma once

#include "strata/timeline.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace strata::timeline {

class GpuClock {
public:
    GpuClock() = default;
    GpuClock(const GpuClock&) = delete;
    GpuClock& operator=(const GpuClock&) = delete;
    ~GpuClock() {   // also the events of spans never resolved and of open chains (an early return of a run)
        std::vector<cudaEvent_t> all = free_;
        for (const Pending& p : pending_) {
            all.push_back(p.e0);
            all.push_back(p.e1);
        }
        for (const auto& kv : chains_) all.push_back(kv.second.ev);
        std::sort(all.begin(), all.end());
        all.erase(std::unique(all.begin(), all.end()), all.end());
        for (cudaEvent_t e : all)
            if (e != nullptr) cudaEventDestroy(e);
        if (anchor_ev_) cudaEventDestroy(anchor_ev_);
    }

    /// Ties the device clock to the host clock.  Synchronizes `s` (call it where that is free: a run's start).
    void anchor(cudaStream_t s) {
        if (!enabled()) return;
        resolve(true);
        if (anchor_ev_ == nullptr && cudaEventCreate(&anchor_ev_) != cudaSuccess) { cudaGetLastError(); return; }
        cudaStreamSynchronize(s);
        // WDDM batches submissions: without the query the record reaches the GPU only when the synchronize flushes
        // it, and the device time sits at the end of the interval (4 ms off in xeno_timeline_gpu).  The narrowest of
        // a few tries bounds the error.
        cudaEvent_t tmp = nullptr;
        if (cudaEventCreate(&tmp) != cudaSuccess) { cudaGetLastError(); return; }
        double best = 1e30, best_t0 = 0, best_t1 = 0;
        for (int i = 0; i < 8; ++i) {
            cudaEvent_t e = i == 0 ? anchor_ev_ : tmp;
            const double t0 = now_us();
            cudaEventRecord(e, s);
            (void) cudaStreamQuery(s);
            cudaEventSynchronize(e);
            const double t1 = now_us();
            if (t1 - t0 < best) {
                best = t1 - t0; best_t0 = t0; best_t1 = t1;
                if (i > 0) std::swap(anchor_ev_, tmp);
            }
        }
        cudaEventDestroy(tmp);
        anchor_us_ = 0.5 * (best_t0 + best_t1);
        anchored_ = true;
        const int l = lane("timeline anchors");
        complete_on(l, "anchor", best_t0, best_t1, (int64_t) best, -1);
    }

    /// A timing event recorded on `s` now (nullptr when the timeline is off or not anchored).
    cudaEvent_t record(cudaStream_t s) {
        if (!enabled() || !anchored_) return nullptr;
        cudaEvent_t e = nullptr;
        if (!free_.empty()) { e = free_.back(); free_.pop_back(); }
        else if (cudaEventCreate(&e) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        cudaEventRecord(e, s);
        return e;
    }

    /// A span between two events from record() on the same device; both are recycled once it is resolved.
    void span(int lane_id, const char* name, cudaEvent_t e0, cudaEvent_t e1, int64_t a = -1, int64_t b = -1) {
        if (e0 == nullptr || e1 == nullptr) return;
        pending_.push_back({lane_id, name, e0, e1, a, b, true, true});
    }

    /// Consecutive marks on one lane: the time from a mark to the next is charged to the first one's name (a gap in
    /// which the stream waits lands on the phase that waited).  name == nullptr closes the chain.
    void mark(int lane_id, const char* name, cudaStream_t s, int64_t a = -1, int64_t b = -1) {
        if (!enabled() || !anchored_) return;
        Chain& c = chains_[lane_id];
        cudaEvent_t e = name != nullptr || c.ev != nullptr ? record(s) : nullptr;
        if (e == nullptr) return;   // could not record: the open mark stays open (a pending span may end on it)
        if (c.ev != nullptr) pending_.push_back({lane_id, c.name, c.ev, e, c.a, c.b, true, name == nullptr});
        if (name == nullptr) {
            if (c.ev == nullptr && e != nullptr) free_.push_back(e);
            c = Chain{};
        } else {
            c = Chain{name, e, a, b};
        }
    }

    /// Writes every completed span to the timeline (wait: all of them; the caller has synchronized or will wait).
    void resolve(bool wait) {
        size_t i = 0;
        for (; i < pending_.size(); ++i) {
            Pending& p = pending_[i];
            if (wait) cudaEventSynchronize(p.e1);
            else if (cudaEventQuery(p.e1) != cudaSuccess) { cudaGetLastError(); break; }
            float m0 = 0.f, m1 = 0.f;
            if (cudaEventElapsedTime(&m0, anchor_ev_, p.e0) == cudaSuccess &&
                cudaEventElapsedTime(&m1, anchor_ev_, p.e1) == cudaSuccess)
                complete_on(p.lane, p.name, anchor_us_ + 1000.0 * m0, anchor_us_ + 1000.0 * m1, p.a, p.b);
            else
                cudaGetLastError();
            if (p.free0) free_.push_back(p.e0);
            if (p.free1) free_.push_back(p.e1);
        }
        pending_.erase(pending_.begin(), pending_.begin() + (std::ptrdiff_t) i);
    }

private:
    struct Pending {
        int lane;
        const char* name;
        cudaEvent_t e0, e1;
        int64_t a, b;
        bool free0, free1;
    };
    struct Chain {
        const char* name = nullptr;
        cudaEvent_t ev = nullptr;
        int64_t a = -1, b = -1;
    };
    cudaEvent_t anchor_ev_ = nullptr;
    double anchor_us_ = 0;
    bool anchored_ = false;
    std::vector<cudaEvent_t> free_;
    std::vector<Pending> pending_;
    std::map<int, Chain> chains_;
};

}  // namespace strata::timeline
