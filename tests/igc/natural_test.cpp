// Firing rules of igc_natural.hpp with a stub proposer; no weights or GPU.
#include "strata/core/igc_natural.hpp"
#include "strata/core/igc_sidecar.hpp"
#include <cassert>
#include <iostream>

using strata::igc::NaturalFirer;
using strata::igc::SidecarProposal;

struct Stub {
    int count = 0, cap = 512, calls = 0;
    SidecarProposal next;
    std::vector<int> last_ends;
    void reset() { count = 0; }
    void push(int32_t) { ++count; }
    int tokens() const { return count; }
    int capacity() const { return cap; }
    SidecarProposal propose(const std::vector<int>& ends) { ++calls; last_ends = ends; return next; }
};

static int32_t d(int digit) { return 15 + digit; }

static std::vector<int64_t> prompt(std::vector<int64_t> user) {
    std::vector<int64_t> out{248045, 872, 198};
    out.insert(out.end(), user.begin(), user.end());
    out.insert(out.end(), {248046, 198, 248045, 77091, 198, 248068, 198});
    return out;
}

// the question holds 12 and 34; a word token separates them
static std::vector<int32_t> run(NaturalFirer<Stub>& firer, std::vector<int32_t> tokens, std::vector<int32_t>* written = nullptr) {
    std::vector<int32_t> rest;
    for (int32_t token : tokens) {
        auto queued = firer.step(token);
        if (written) written->push_back(token);
        if (!queued.empty()) { rest = queued; break; }
    }
    return rest;
}

int main() {
    Stub stub;
    stub.next.operation = 2; stub.next.left = 0; stub.next.right = 1; stub.next.confidence = .9f;   // 12 * 34 = 408
    {   // a later digit that differs is replaced and the rest follows
        NaturalFirer<Stub> firer(stub);
        firer.begin(prompt({d(1), d(2), 1000, d(3), d(4), 1000}));
        assert(stub.count > 0);
        std::vector<int32_t> written;
        const auto rest = run(firer, {1001, d(4), d(0), d(9)}, &written);
        assert(written.back() == d(8));
        assert(rest.empty());
        assert(firer.fires() == 1);
        assert((stub.last_ends == std::vector<int>{1, 4}));
    }
    stub.next.operation = 2; stub.next.left = 1; stub.next.right = 1;   // 34 * 34 = 1156
    {   // the rest of a longer value is delivered
        NaturalFirer<Stub> firer(stub);
        firer.begin(prompt({d(1), d(2), 1000, d(3), d(4), 1000}));
        const auto rest = run(firer, {1001, d(1), d(1), d(9), d(9)});
        assert((rest == std::vector<int32_t>{d(6)}));
        assert(firer.fires() == 1);
    }
    stub.next.left = 0; stub.next.right = 1;
    {   // a leading digit that differs is the model's: never fires
        NaturalFirer<Stub> firer(stub);
        firer.begin(prompt({d(1), d(2), 1000, d(3), d(4), 1000}));
        const auto rest = run(firer, {1001, d(4), d(1), d(9)});
        assert(rest.empty());
        assert(firer.fires() == 0);
    }
    {   // the same value changes nothing
        NaturalFirer<Stub> firer(stub);
        firer.begin(prompt({d(1), d(2), 1000, d(3), d(4), 1000}));
        std::vector<int32_t> written;
        run(firer, {1001, d(4), d(0), d(8), 1002}, &written);
        assert((written == std::vector<int32_t>{1001, d(4), d(0), d(8), 1002}));
        assert(firer.fires() == 0);
    }
    {   // low confidence never fires
        NaturalFirer<Stub> firer(stub, 1.f);
        firer.begin(prompt({d(1), d(2), 1000, d(3), d(4), 1000}));
        run(firer, {1001, d(4), d(0), d(9)});
        assert(firer.fires() == 0);
    }
    {   // a number closing a written expression belongs to the parser
        NaturalFirer<Stub> firer(stub);
        firer.begin(prompt({d(1), d(2), 1000, d(3), d(4), 1000}));
        const int before = stub.calls;
        run(firer, {d(1), d(2), 9, d(3), d(4), 28, d(4), d(0), d(9)});
        assert(firer.fires() == 0);
        assert(stub.calls == before + 2);   // proposed for the operands, not the result
    }
    {   // a comma continues the number, so 1,200 is one entry
        NaturalFirer<Stub> firer(stub);
        firer.begin(prompt({d(1), 11, d(2), d(0), d(0), 1000, d(3), 1000}));
        stub.next.left = 0; stub.next.right = 1;   // 1200 * 3 = 3600
        const auto rest = run(firer, {1001, d(3), d(6), d(1), d(9)});
        assert(firer.fires() == 1);
        assert((rest == std::vector<int32_t>{d(0)}));
        assert(stub.last_ends.size() == 2);
    }
    {   // past its capacity the sidecar goes quiet
        Stub small; small.cap = 8; small.next = stub.next;
        NaturalFirer<Stub> firer(small);
        firer.begin(prompt({d(1), d(2), 1000, d(3), d(4), 1000}));
        run(firer, {1001, 1001, 1001, 1001, 1001, d(4), d(0), d(9)});
        assert(firer.fires() == 0);
    }
    std::cout << "OK\n";
    return 0;
}
