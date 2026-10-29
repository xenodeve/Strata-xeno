#include "strata/core/igc_pipeline.hpp"
#include "strata/core/igc_input.hpp"
#include "strata/core/igc_readiness.hpp"
#include "strata/core/igc_dispatch.hpp"
#include "strata/core/igc_output.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <memory>
namespace strata::igc {
namespace {
struct Buffers {
    float embedding[52],attention[52],ready[2];
    uint32_t embedding_status,attention_status,ready_status,selection;
    uint64_t ids[64],anchors[64],arm_id,arm_anchor;
    ExactRequest request;
    Invocation invocation;
    ExactResult results[64];
    uint32_t head_status[64];
};
static_assert(std::is_standard_layout_v<Buffers> && std::is_trivially_copyable_v<Buffers>);
bool checked(cudaError_t c,std::string& error){if(c==cudaSuccess)return true;error=cudaGetErrorString(c);return false;}
bool created(void* raw){const auto s=static_cast<cudaStream_t>(raw);return s && s!=cudaStreamLegacy && s!=cudaStreamPerThread;}
bool outside_capture(cudaStream_t stream,std::string& error){
    cudaStreamCaptureStatus status;
    if(!checked(cudaStreamIsCapturing(stream,&status),error))return false;
    if(status!=cudaStreamCaptureStatusNone){error="pipeline_outside_capture_required";return false;}
    return true;
}
}
struct LatentPipeline::Impl {
    InputMapper primary,secondary;ReadinessMapper readiness;OutputMapper output;
    cudaStream_t stream=nullptr;Buffers* data=nullptr;Buffers* checkpoint=nullptr;
    bool trial=false,failed=false;
    std::array<uint64_t,64> ids{},anchors{};
    uint64_t arm_id=0,arm_anchor=0;
    ~Impl(){if(stream)cudaStreamSynchronize(stream);if(checkpoint)cudaFree(checkpoint);if(data)cudaFree(data);}
};
LatentPipeline::~LatentPipeline(){delete impl_;}
bool LatentPipeline::initialize(const ControllerBundle& bundle,void* raw,std::string& error)try{
    if(impl_ || !created(raw)){error="pipeline_initialize_arguments";return false;}
    auto next=std::make_unique<Impl>();next->stream=static_cast<cudaStream_t>(raw);
    if(!outside_capture(next->stream,error))return false;
    constexpr std::array<uint32_t,4> kinds={1,1,3,2};
    for(std::size_t i=0;i<kinds.size();++i)if(bundle.networks[i].kind!=kinds[i] || !bundle.networks[i].diagnostic_only){error="pipeline_network_roles";return false;}
    if(!next->primary.load(bundle.networks[0],raw,1,error)
            || !next->secondary.load(bundle.networks[1],raw,1,error)
            || !next->readiness.load(bundle.networks[2],raw,1,error)
            || !next->output.load(bundle.networks[3],raw,error)
            || !checked(cudaMalloc(reinterpret_cast<void**>(&next->data),sizeof(Buffers)),error)
            || !checked(cudaMemsetAsync(next->data,0,sizeof(Buffers),next->stream),error))return false;
    auto* d=next->data;
    if(!arm_invocations(&d->invocation,&d->arm_id,&d->arm_anchor,1,raw,true,error)
            || !broadcast_invocation(&d->invocation,d->results,64,raw,true,error)
            || !checked(cudaStreamSynchronize(next->stream),error))return false;
    impl_=next.release();error.clear();return true;
}catch(const std::exception& e){error=std::string("pipeline_initialize:")+e.what();return false;}
bool LatentPipeline::stage_inputs(const float* embedding,const float* attention,const uint8_t* mask,
                                  uint64_t id,uint64_t anchor,bool available,void* raw,bool enabled,std::string& error){
    if(!enabled){error.clear();return true;}
    if(!impl_ || impl_->failed || raw!=impl_->stream || !id || anchor>=262144 || (available && (!embedding || !attention || !mask))){error="pipeline_stage_arguments";return false;}
    auto& m=*impl_;auto* d=m.data;
    if(m.trial){error="pipeline_stage_during_trial";return false;}
    if(!outside_capture(m.stream,error))return false;
    m.ids.fill(id);m.anchors.fill(anchor);m.arm_id=available?id:0;m.arm_anchor=anchor;
    // These host buffers belong to Impl, remaining alive through graph execution.
    // Caller fences its prior request before staging the next one.
    if(!checked(cudaMemcpyAsync(d->ids,m.ids.data(),sizeof(d->ids),cudaMemcpyHostToDevice,m.stream),error)
            || !checked(cudaMemcpyAsync(d->anchors,m.anchors.data(),sizeof(d->anchors),cudaMemcpyHostToDevice,m.stream),error)
            || !checked(cudaMemcpyAsync(&d->arm_id,&m.arm_id,8,cudaMemcpyHostToDevice,m.stream),error)
            || !checked(cudaMemcpyAsync(&d->arm_anchor,&m.arm_anchor,8,cudaMemcpyHostToDevice,m.stream),error)
            || !checked(cudaMemsetAsync(&d->embedding_status,0xff,4,m.stream),error)
            || !checked(cudaMemsetAsync(&d->attention_status,0xff,4,m.stream),error)
            || !arm_invocations(&d->invocation,&d->arm_id,&d->arm_anchor,1,raw,true,error))return false;
    if(available && (!m.primary.run(embedding,mask,d->embedding,&d->embedding_status,1,raw,true,error)
                    || !m.secondary.run(attention,mask,d->attention,&d->attention_status,1,raw,true,error)))return false;
    error.clear();return true;
}
bool LatentPipeline::head(float* mixed,uint32_t rows,void* raw,bool enabled,std::string& error){
    if(!enabled){error.clear();return true;}
    if(!impl_ || impl_->failed || raw!=impl_->stream || !mixed || !rows || rows>64){error="pipeline_head_arguments";return false;}
    auto& m=*impl_;auto* d=m.data;
    const SelectionInputs inputs{d->embedding,d->attention,d->ready,&d->embedding_status,&d->attention_status,&d->ready_status,d->ids,d->anchors};
    return m.readiness.run(mixed,d->ready,&d->ready_status,1,raw,true,error)
        && assemble_selective_exact(inputs,&d->request,&d->selection,1,raw,true,error)
        && execute_invocations(&d->invocation,&d->request,&d->selection,1,raw,true,error)
        && broadcast_invocation(&d->invocation,d->results,rows,raw,true,error)
        && m.output.run(mixed,d->results,d->ids,d->anchors,d->head_status,rows,raw,true,error);
}
bool LatentPipeline::audit(Invocation& out,uint32_t& status,void* raw,std::string& error){
    if(!impl_ || impl_->failed || raw!=impl_->stream){error="pipeline_audit_arguments";return false;}
    if(!outside_capture(impl_->stream,error))return false;
    return checked(cudaMemcpyAsync(&out,&impl_->data->invocation,sizeof(out),cudaMemcpyDeviceToHost,impl_->stream),error)
        && checked(cudaMemcpyAsync(&status,impl_->data->head_status,4,cudaMemcpyDeviceToHost,impl_->stream),error)
        && checked(cudaStreamSynchronize(impl_->stream),error);
}
bool LatentPipeline::begin_trial(void* raw,bool enabled,std::string& error){
    if(!enabled){error.clear();return true;}
    if(!impl_ || impl_->failed || impl_->trial || raw!=impl_->stream){error="pipeline_trial_begin_arguments";return false;}
    auto& m=*impl_;
    if(!outside_capture(m.stream,error))return false;
    if(!m.checkpoint && !checked(cudaMalloc(reinterpret_cast<void**>(&m.checkpoint),sizeof(Buffers)),error))return false;
    if(!checked(cudaMemcpyAsync(m.checkpoint,m.data,sizeof(Buffers),cudaMemcpyDeviceToDevice,m.stream),error)){
        m.failed=true;return false;
    }
    m.trial=true;error.clear();return true;
}
bool LatentPipeline::finish_trial(bool keep,void* raw,bool enabled,std::string& error){
    if(!enabled){error.clear();return true;}
    if(!impl_ || impl_->failed || !impl_->trial || raw!=impl_->stream){error="pipeline_trial_finish_arguments";return false;}
    auto& m=*impl_;
    if(!outside_capture(m.stream,error))return false;
    if(!keep && !checked(cudaMemcpyAsync(m.data,m.checkpoint,sizeof(Buffers),cudaMemcpyDeviceToDevice,m.stream),error)){
        m.failed=true;return false;
    }
    m.trial=false;error.clear();return true;
}
std::size_t LatentPipeline::device_bytes()const{return impl_?(impl_->checkpoint?2:1)*sizeof(Buffers)+impl_->primary.device_bytes()+impl_->secondary.device_bytes()+impl_->readiness.device_bytes()+impl_->output.device_bytes():0;}
}
