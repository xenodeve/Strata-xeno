#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace strata::igc {
enum class ContextTap : uint32_t { Embedding=0,FirstAttention=1 };
// Bounded actual activation history. Host metadata tracks causal prefix identity;
// tensors never leave the device. configure once; caller fences model consumers
// before begin/release. observe stages rows; only accept makes positions owned.
class ContextWindow {
public:
    static constexpr int rows=80,width=4*2560,max_tokens=262144;
    ContextWindow()=default;
    ~ContextWindow();
    ContextWindow(const ContextWindow&)=delete;
    ContextWindow& operator=(const ContextWindow&)=delete;
    bool allocate(void* created_stream,std::string&);
    // reused is the engine's verified retained INPUT prefix length. Images,
    // model/control-vector identity must also match for may_reuse to be true.
    // Unverifiable lineage clears feature availability; it never forces a cold
    // language-model prefill or manufactures features from KV state.
    bool begin(std::span<const int64_t> prefix,int64_t reused,bool may_reuse,std::string&);
    // Source is contiguous actual [count,width] native state on a created
    // producer stream. Enqueues bounded D2D copies, retaining at most80 rows.
    bool observe(ContextTap,int64_t pos0,int64_t count,const float* device,void* producer_stream,std::string&);
    // Accept exactly the contiguous consumed-input prefix, never emitted next
    // tokens or a speculative rejected suffix. Missing features stay missing.
    bool accept(int64_t pos0,std::span<const int64_t> tokens,std::string&);
    // Only the full pre-anchor window is eligible. false with error empty means
    // unavailable features, so the caller bypasses IGC and keeps baseline decode.
    bool gather(int64_t anchor,void* created_stream,std::string&);
    const float* states(ContextTap)const;
    const uint8_t* mask()const;
    std::size_t device_bytes()const;
    void invalidate();
private:
    struct Impl;Impl* impl_=nullptr;
};
}
