#pragma once
#include "strata/core/qwenputer_artifact.hpp"
#include <memory>

namespace strata::qwenputer {
struct InputProof {
    Digest model{}, tokenizer{};
};
// Content binding for the actual loaded sources, not a geometry/path assertion.
// On Windows verified input handles deny concurrent writes/deletion until this
// object is destroyed. Construct before model allocation; retain through use.
// The caller must enumerate every source covered by its identity contract. This
// class does not discover pack banks, MTP weights, build options or run settings.
class VerifiedInputs {
public:
    VerifiedInputs();
    ~VerifiedInputs();
    VerifiedInputs(const VerifiedInputs&) = delete;
    VerifiedInputs& operator=(const VerifiedInputs&) = delete;
    bool bind(const Binding&, const std::filesystem::path& tokenizer,
              std::span<const std::filesystem::path> model_shards, std::string& error);
    const InputProof* proof() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::qwenputer
