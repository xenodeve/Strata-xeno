#pragma once
#include <string>

namespace strata::qwenputer {
struct WindowDecision {
    int consumed_inputs = 0;
    int visible_model_outputs = 0;
    int takeover_row = -1;
};

// A verifier row consumes its INPUT and predicts the NEXT token. The next token
// from an eligible closing-call row belongs to the circuit. It must not leak out
// as a language-model bonus. A takeover in the rejected suffix has no effect.
// first_takeover_row comes from the trusted role/enrollment/grammar shadow trial.
inline bool decide_window(int inputs, int normal_keep, int first_takeover_row,
                          WindowDecision& result, std::string& error) {
    if (inputs < 1 || normal_keep < 1 || normal_keep > inputs
        || first_takeover_row < -1 || first_takeover_row >= inputs) {
        error = "ownership_window_bounds";
        return false;
    }
    WindowDecision decision;
    if (first_takeover_row >= 0 && first_takeover_row < normal_keep) {
        decision.consumed_inputs = first_takeover_row + 1;
        decision.visible_model_outputs = first_takeover_row;
        decision.takeover_row = first_takeover_row;
    } else {
        decision.consumed_inputs = normal_keep;
        decision.visible_model_outputs = normal_keep;
    }
    result = decision;
    error.clear();
    return true;
}
} // namespace strata::qwenputer
