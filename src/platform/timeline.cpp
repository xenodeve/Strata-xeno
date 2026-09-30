// src/platform/timeline.cpp - see include/strata/timeline.hpp.
#include "strata/timeline.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#define STRATA_GETPID _getpid
#else
#include <unistd.h>
#define STRATA_GETPID getpid
#endif

namespace strata::timeline {
namespace {

struct Ev {
    const char* name;
    double t0, t1;
    int64_t a, b;
    char ph;   // 'X' span, 'i' instant
};

// One per thread and one per virtual lane.  The owner appends under `mu` (uncontended except during a flush);
// name, named_out and retired belong to the registry and are read and written under Registry::mu only.
struct Lane {
    int tid = 0;
    std::string name;
    bool named_out = false;   // its thread_name record is in the file
    bool retired = false;     // its thread has exited: a new thread of the same name takes it over
    std::mutex mu;
    std::vector<Ev> ev;
};

struct Registry {
    std::mutex mu;
    std::vector<std::unique_ptr<Lane>> lanes;   // never freed: a finished thread's events are still flushed
    std::mutex file_mu;
    bool file_open = false;
    std::atomic<int64_t> pending{0}, dropped{0};   // events recorded and not flushed yet; events refused
    int64_t max_events = 0;
    const char* path = nullptr;
    int pid = 0;
};

Registry& reg() {
    static Registry* r = [] {
        auto* x = new Registry;   // leaked on purpose: threads may still record while statics are destroyed
        x->path = std::getenv("STRATA_TIMELINE");
        if (x->path != nullptr && *x->path == 0) x->path = nullptr;
        const char* m = std::getenv("STRATA_TIMELINE_MAX_EVENTS");
        x->max_events = m ? std::atoll(m) : 8'000'000;   // unflushed events (~0.4 GB); beyond, counted, not kept
        x->pid = (int) STRATA_GETPID();
        return x;
    }();
    return *r;
}

Lane* new_lane_locked(Registry& r, std::string name) {   // r.mu held
    r.lanes.push_back(std::make_unique<Lane>());
    Lane* l = r.lanes.back().get();
    l->tid = (int) r.lanes.size();
    l->name = std::move(name);
    return l;
}

// The calling thread's lane, retired when the thread exits so that a thread made per window or per chunk (the
// adaptive tier's, the copy issuer's) reuses the row of its predecessor instead of adding one each time.
struct Holder {
    Lane* l = nullptr;
    ~Holder() {
        if (l == nullptr) return;
        std::lock_guard<std::mutex> lk(reg().mu);
        l->retired = true;
    }
};
thread_local Holder t_holder;

Lane* my_lane() {
    if (t_holder.l == nullptr) {
        Registry& r = reg();
        std::lock_guard<std::mutex> lk(r.mu);
        t_holder.l = new_lane_locked(r, "");
    }
    return t_holder.l;
}

void push(Lane* l, const Ev& e) {
    Registry& r = reg();
    if (r.pending.fetch_add(1, std::memory_order_relaxed) >= r.max_events) {
        r.pending.fetch_sub(1, std::memory_order_relaxed);
        r.dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::lock_guard<std::mutex> lk(l->mu);
    l->ev.push_back(e);
}

void escape(std::string& out, const char* s) {
    for (; *s; ++s) {
        const char c = *s;
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if ((unsigned char) c < 0x20) out += ' ';
        else out += c;
    }
}

}  // namespace

namespace detail {
std::atomic<int> g_state{0};
bool init_enabled() {
    const bool on = reg().path != nullptr;
    g_state.store(on ? 2 : 1, std::memory_order_relaxed);
    return on;
}
}  // namespace detail

double now_us() { return us(std::chrono::steady_clock::now()); }

void name_thread(const char* name) {
    if (!enabled()) return;
    const std::string nm = name ? name : "";
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    if (t_holder.l == nullptr && !nm.empty())
        for (auto& l : r.lanes)
            if (l->retired && l->name == nm) {
                l->retired = false;
                t_holder.l = l.get();
                return;
            }
    if (t_holder.l == nullptr) t_holder.l = new_lane_locked(r, nm);
    else t_holder.l->name = nm;
}

void complete(const char* name, double t0_us, double t1_us, int64_t a, int64_t b) {
    if (!enabled()) return;
    push(my_lane(), {name, t0_us, t1_us, a, b, 'X'});
}

int lane(const char* name) {
    if (!enabled()) return 0;
    Registry& r = reg();
    std::lock_guard<std::mutex> lk(r.mu);
    for (auto& l : r.lanes)
        if (l->name == name) return l->tid;
    return new_lane_locked(r, name)->tid;
}

void complete_on(int lane_id, const char* name, double t0_us, double t1_us, int64_t a, int64_t b) {
    if (!enabled()) return;
    Registry& r = reg();
    Lane* l = nullptr;
    {
        std::lock_guard<std::mutex> lk(r.mu);
        if (lane_id >= 1 && lane_id <= (int) r.lanes.size()) l = r.lanes[(size_t) lane_id - 1].get();
    }
    if (l != nullptr) push(l, {name, t0_us, t1_us, a, b, 'X'});
}

void instant(const char* name, int64_t a, int64_t b) {
    if (!enabled()) return;
    const double t = now_us();
    push(my_lane(), {name, t, t, a, b, 'i'});
}

void flush() {
    if (!enabled()) return;
    Registry& r = reg();
    std::lock_guard<std::mutex> fk(r.file_mu);
    std::vector<Lane*> lanes;
    std::vector<std::string> names_now;   // a lane's name, when its thread_name record is due in this flush
    {
        std::lock_guard<std::mutex> lk(r.mu);
        for (auto& l : r.lanes) {
            lanes.push_back(l.get());
            const bool due = !l->named_out && !l->name.empty();
            names_now.push_back(due ? l->name : std::string());
            if (due) l->named_out = true;
        }
    }
    std::FILE* f = std::fopen(r.path, r.file_open ? "ab" : "wb");
    if (f == nullptr) return;
    std::string out;
    if (!r.file_open) {
        out += "[\n";
        char b[256];
        std::snprintf(b, sizeof b,
                      "{\"ph\":\"M\",\"pid\":%d,\"tid\":0,\"name\":\"process_name\",\"args\":{\"name\":\"strata %d\"}},\n",
                      r.pid, r.pid);
        out += b;
        r.file_open = true;
    }
    for (size_t li = 0; li < lanes.size(); ++li) {
        Lane* l = lanes[li];
        std::vector<Ev> ev;
        {
            std::lock_guard<std::mutex> lk(l->mu);
            ev.swap(l->ev);
        }
        r.pending.fetch_sub((int64_t) ev.size(), std::memory_order_relaxed);
        const std::string& name = names_now[li];
        char b[320];
        if (!name.empty()) {
            out += "{\"ph\":\"M\",\"pid\":";
            out += std::to_string(r.pid);
            out += ",\"tid\":";
            out += std::to_string(l->tid);
            out += ",\"name\":\"thread_name\",\"args\":{\"name\":\"";
            escape(out, name.c_str());
            out += "\"}},\n";
        }
        for (const Ev& e : ev) {
            int n;
            if (e.ph == 'X')
                n = std::snprintf(b, sizeof b, "{\"ph\":\"X\",\"pid\":%d,\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f,\"name\":\"", r.pid,
                                  l->tid, e.t0, e.t1 - e.t0);
            else
                n = std::snprintf(b, sizeof b, "{\"ph\":\"i\",\"s\":\"t\",\"pid\":%d,\"tid\":%d,\"ts\":%.3f,\"name\":\"", r.pid,
                                  l->tid, e.t0);
            out.append(b, (size_t) n);
            escape(out, e.name ? e.name : "?");
            out += '"';
            if (e.a != -1 || e.b != -1) {
                n = std::snprintf(b, sizeof b, ",\"args\":{\"a\":%lld,\"b\":%lld}", (long long) e.a, (long long) e.b);
                out.append(b, (size_t) n);
            }
            out += "},\n";
            if (out.size() > (1u << 20)) { std::fwrite(out.data(), 1, out.size(), f); out.clear(); }
        }
    }
    if (const int64_t d = r.dropped.load(); d > 0) {
        char b[160];
        std::snprintf(b, sizeof b, "{\"ph\":\"i\",\"s\":\"g\",\"pid\":%d,\"tid\":0,\"ts\":%.3f,\"name\":\"dropped events\","
                                   "\"args\":{\"a\":%lld,\"b\":-1}},\n", r.pid, now_us(), (long long) d);
        out += b;
    }
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
}

}  // namespace strata::timeline
