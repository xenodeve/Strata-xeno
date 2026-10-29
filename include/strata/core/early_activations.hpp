#pragma once
#include <cstdint>
#include <string>

namespace strata::core {
enum class EarlyActivationTap { Embedding,FirstAttention };
struct EarlyActivationRows {
    EarlyActivationTap tap;
    const float* device=nullptr;
    int64_t count=0,width=0,pos0=0;
    void* stream=nullptr;
};
// First-party consumer only. Enqueue copies/work on the supplied owning stream
// before returning; never retain borrowed source addresses into a later window.
// This host notification is outside CUDA graph execution, not a graph host node.
struct EarlyActivationConsumer {
    void* user=nullptr;
    bool (*consume)(void*,const EarlyActivationRows&,std::string&)=nullptr;
};
}
