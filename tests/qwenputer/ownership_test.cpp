#include "strata/core/qwenputer_ownership.hpp"
#include <iostream>
#include <initializer_list>

using namespace strata::qwenputer;
int main() {
    int failures = 0;
    auto check = [&](bool condition, const char* name) {
        if (!condition) {
            if (++failures <= 20) std::cerr << "FAIL " << name << '\n';
        }
    };
    std::string error;
    for (int inputs = 1; inputs <= 32; ++inputs) {
        for (int keep = 1; keep <= inputs; ++keep) {
            for (int row = -1; row < inputs; ++row) {
                WindowDecision decision;
                check(decide_window(inputs, keep, row, decision, error), "valid_window_decision");
                check(error.empty(), "valid_decision_clears_error");
                const bool owned = row >= 0 && row < keep;
                check(decision.consumed_inputs == (owned ? row + 1 : keep), "accepted_input_prefix");
                check(decision.visible_model_outputs == (owned ? row : keep), "no_language_bonus_at_takeover");
                check(decision.takeover_row == (owned ? row : -1), "rejected_suffix_has_no_takeover");
                check(decision.consumed_inputs <= keep, "never_commit_rejected_input");
            }
        }
    }
    for (const auto invalid : {WindowDecision{0, 1, -1}, WindowDecision{2, 0, -1},
                               WindowDecision{2, 3, -1}, WindowDecision{2, 1, -2},
                               WindowDecision{2, 1, 2}}) {
        WindowDecision decision{9, 8, 7};
        check(!decide_window(invalid.consumed_inputs, invalid.visible_model_outputs,
                             invalid.takeover_row, decision, error), "invalid_boundary_refused");
        check(decision.consumed_inputs == 9 && decision.visible_model_outputs == 8
              && decision.takeover_row == 7, "invalid_boundary_is_transactional");
    }
    if (failures) { std::cerr << "ownership_failures=" << failures << '\n'; return 1; }
    std::cout << "PASS ownership boundary: consumed input versus emitted next token\n";
    return 0;
}
