#include "strata/core/igc_prefill_snapshot.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <memory>
namespace strata::igc {
namespace {
constexpr int64_t width=4*2560,rows=80,max_tokens=262144;
constexpr size_t plane=rows*width;
bool checked(cudaError_t c,std::string& e){if(c==cudaSuccess)return true;e=cudaGetErrorString(c);return false;}
bool created(void* raw){const auto s=static_cast<cudaStream_t>(raw);return s && s!=cudaStreamLegacy && s!=cudaStreamPerThread;}
}
struct PrefillSnapshot::Impl {
    cudaStream_t stream=nullptr;float* data=nullptr;int device=-1;
    int64_t pos0=0,count=0,start=0,take=0;bool embedding=false,attention=false;
    ~Impl(){
        int previous=-1;cudaGetDevice(&previous);
        if(device>=0 && device!=previous)cudaSetDevice(device);
        if(stream)cudaStreamSynchronize(stream);
        if(data)cudaFree(data);
        if(previous>=0 && device!=previous)cudaSetDevice(previous);
    }
    void clear(){embedding=false;attention=false;}
};
PrefillSnapshot::~PrefillSnapshot(){delete impl_;}
bool PrefillSnapshot::allocate(void* raw,std::string& e)try{
    if(impl_ || !created(raw)){e="snapshot_allocate_arguments";return false;}
    auto next=std::make_unique<Impl>();next->stream=static_cast<cudaStream_t>(raw);
    cudaStreamCaptureStatus status;
    if(!checked(cudaStreamIsCapturing(next->stream,&status),e))return false;
    if(status!=cudaStreamCaptureStatusNone){e="snapshot_allocate_during_capture";return false;}
    if(!checked(cudaGetDevice(&next->device),e) || !checked(cudaMalloc(&next->data,2*plane*sizeof(float)),e))return false;
    impl_=next.release();e.clear();return true;
}catch(const std::exception& error){e=error.what();return false;}
bool PrefillSnapshot::stage(core::EarlyActivationTap tap,const float* source,int64_t pos0,int64_t count,void* raw,std::string& e){
    const bool first=tap==core::EarlyActivationTap::Embedding,second=tap==core::EarlyActivationTap::FirstAttention;
    if(!impl_ || raw!=impl_->stream || !source || pos0<0 || count<1 || count>max_tokens-pos0 || (!first && !second)){
        if(impl_)impl_->clear();e="snapshot_stage_arguments";return false;
    }
    auto& m=*impl_;
    if(first){m.clear();m.pos0=pos0;m.count=count;m.take=std::min(count,rows);m.start=pos0+count-m.take;}
    else if(!m.embedding || m.attention || pos0!=m.pos0 || count!=m.count){m.clear();e="snapshot_mixed_or_duplicate_chunk";return false;}
    auto* target=m.data+(second?plane:0);
    if(!checked(cudaMemcpyAsync(target,source+(count-m.take)*width,m.take*width*sizeof(float),cudaMemcpyDeviceToDevice,m.stream),e)){
        m.clear();return false;
    }
    if(first)m.embedding=true;else m.attention=true;
    e.clear();return true;
}
bool PrefillSnapshot::emit(const core::EarlyActivationConsumer& consumer,std::string& e){
    if(!impl_ || !impl_->embedding || !impl_->attention || !consumer.user || !consumer.consume){
        if(impl_)impl_->clear();e="snapshot_emit_arguments";return false;
    }
    auto& m=*impl_;m.clear(); // A failed consumer cannot publish this chunk again.
    for(const auto tap:{core::EarlyActivationTap::Embedding,core::EarlyActivationTap::FirstAttention}){
        const core::EarlyActivationRows value{tap,m.data+(tap==core::EarlyActivationTap::FirstAttention?plane:0),m.take,width,m.start,m.stream};
        if(!consumer.consume(consumer.user,value,e)){if(e.empty())e="snapshot_consumer_failed";return false;}
    }
    e.clear();return true;
}
std::size_t PrefillSnapshot::device_bytes()const{return impl_?2*plane*sizeof(float):0;}
}
