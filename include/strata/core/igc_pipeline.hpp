#pragma once
#include "strata/core/igc_controller_bundle.hpp"
#include "strata/core/igc_invocation.hpp"
namespace strata::igc {
// Diagnostic first-anchor latent route, not a session/snapshot owner. The caller
// binds actual model/tokenizer receipts and owns/fences source activation rows.
class LatentPipeline {
public:
    LatentPipeline()=default;
    ~LatentPipeline();
    LatentPipeline(const LatentPipeline&)=delete;
    LatentPipeline& operator=(const LatentPipeline&)=delete;
    bool initialize(const ControllerBundle&,void* created_stream,std::string&);
    // Once per request, outside graph capture. Source[80,4,2560], mask[80]; no
    // token IDs or operands. Host owner metadata remains alive until next stage.
    // available=false arms no invocation and cannot retain an earlier answer.
    bool stage_inputs(const float* embedding,const float* attention,const uint8_t* mask,
                      uint64_t invocation,uint64_t anchor,bool available,void* stream,bool enabled,std::string&);
    // Graph-capturable. Reads actual final-HC mixed row0 for readiness, selects
    // typed fields, executes once, retains result, maps it into all mixed rows.
    bool head(float* mixed,uint32_t rows,void* stream,bool enabled,std::string&);
    // Outside capture, once the result has been delivered: later heads pass through
    // unchanged, captured graphs included. Audit keeps reporting the invocation.
    bool end_output(void* stream,std::string&);
    // Optional module-only trial, outside capture on the owning stream. Lazy
    // bounded checkpoint storage is retained for reuse; never touches LM state.
    // Caller discards rejected output rows and fences borrowed inputs. Partial
    // acceptance requires abort, then replay only the accepted input prefix.
    bool begin_trial(void* stream,bool enabled,std::string&);
    bool finish_trial(bool keep,void* stream,bool enabled,std::string&);
    // Offline/diagnostic request-end readback only, never a per-step decision.
    bool audit(Invocation&,uint32_t& head_status,void* stream,std::string&);
    std::size_t device_bytes()const;
private:
    struct Impl;Impl* impl_=nullptr;
};
}
