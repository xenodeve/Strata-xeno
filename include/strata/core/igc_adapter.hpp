#pragma once
#include "strata/core/igc_adapter_schema.hpp"
#include "strata/core/qwenputer_artifact.hpp"
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace strata::igc::adapter {
using Digest=strata::qwenputer::Digest;
struct Binding {Digest model{},tokenizer{},file{};};
struct Artifact {
    uint32_t kind=0,max_digits=0;
    bool diagnostic_only=true;
    Digest state{},protocol{},schema{};
    std::vector<uint64_t> offsets;
    std::vector<float> payload;
    std::span<const float> tensor(std::size_t slot) const {
        const auto definition=adapter_schema::get(kind);
        if(slot>=offsets.size() || slot>=definition.tensors.size())return {};
        if(offsets[slot]>payload.size() || definition.tensors[slot].count>payload.size()-offsets[slot])return {};
        return std::span(payload).subspan(static_cast<std::size_t>(offsets[slot]),
                                         static_cast<std::size_t>(definition.tensors[slot].count));
    }
};
bool parse(std::span<const uint8_t>,const Binding&,bool allow_diagnostic,Artifact&,std::string&);
bool load(const std::filesystem::path&,const Binding&,bool allow_diagnostic,Artifact&,std::string&);
} // namespace strata::igc::adapter
