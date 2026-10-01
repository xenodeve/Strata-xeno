// include/strata/core/placement_formats.hpp - #11: which expert placements a pack's per-layer formats allow.
//
// The 4070 tier and the automatic exclusive-primary default need every layer in native Q2_0 (gate/up and down
// type 42): the 4070's kernels and the bit-exact pool-hit parity exist for Q2_0 only.  An explicit
// --exclusive-primary-experts (placement-first, and with it the --ram-cache-gib NVMe tier) only needs the primary
// GPU and the CPU pool to compute every layer's formats natively.
#pragma once

#include <algorithm>

namespace strata::core {

struct PlacementFormats {
    bool all_q2 = false;       ///< every layer is Q2_0 gate/up and Q2_0 down
    bool all_native = false;   ///< every layer's two formats are ones the native GPU and CPU kernels compute
};

/// `fmts`: the layout's per-layer formats (members gu_type, d_type); `supported(type)`: the native kernels'
/// type check (iq_supported in the engine).
template <class Fmts, class Supported>
PlacementFormats placement_formats(bool native_pack, const Fmts& fmts, Supported supported) {
    PlacementFormats p;
    if (!native_pack || fmts.empty()) return p;
    p.all_q2 = std::all_of(fmts.begin(), fmts.end(),
                           [](const auto& f) { return f.gu_type == 42 && f.d_type == 42; });
    p.all_native = std::all_of(fmts.begin(), fmts.end(),
                               [&](const auto& f) { return supported(f.gu_type) && supported(f.d_type); });
    return p;
}

/// Exclusive primary ownership: by default only on all-Q2_0 packs; requested explicitly, on any native pack.
inline bool exclusive_primary_formats_ok(const PlacementFormats& p, bool requested) {
    return p.all_q2 || (requested && p.all_native);
}

}  // namespace strata::core
