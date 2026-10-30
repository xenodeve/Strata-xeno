#pragma once
#include "strata/core/igc_adapter.hpp"
#include <array>

namespace strata::igc {
// Diagnostic learned controller assets only. Role order: embedding input,
// first-attention input, final-state readiness, late result output. No operands,
// answers, prompt hashes, invocation records, code or user pointers.
struct ControllerBundle {
    adapter::Digest model{},tokenizer{},manifest{};
    std::array<adapter::Artifact,4> networks;
    // Manifest version 2: the attention readout alone selects; embedding is not run.
    bool single_readout=false;
};
// Fixed204-byte manifest and fixed sibling filenames; validate every network
// before publishing anything to `out`. Refuses production before file access.
bool load_controller_bundle(const std::filesystem::path& manifest,bool allow_diagnostic,
                            ControllerBundle& out,std::string& error);
}
