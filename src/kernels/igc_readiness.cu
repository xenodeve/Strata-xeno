#include "strata/core/igc_readiness.hpp"
#include <cuda_runtime.h>
#include <cmath>
namespace strata::igc {
namespace {
constexpr int N=2560,H=128;
struct Weights {const float* p[6]{};};
struct Scratch {float *normalized,*hidden,*logits;};
__global__ void prepare(const float* mixed,uint32_t* status,Weights w,Scratch s) {
    const int b=blockIdx.x,t=threadIdx.x;if(t==0)status[b]=0;__syncthreads();
    for(int j=t;j<N;j+=blockDim.x){
        const float input=mixed[b*N+j],value=(input-w.p[0][j])/w.p[1][j];
        s.normalized[b*N+j]=value;if(!isfinite(input)||!isfinite(value))atomicExch(status+b,2u);
    }
}
__global__ void hidden(uint32_t* status,Weights w,Scratch s) {
    const int b=blockIdx.x,t=threadIdx.x;__shared__ uint32_t bad;
    if(t==0)bad=status[b];__syncthreads();if(bad)return;
    if(t<H){float sum=0;for(int j=0;j<N;++j)sum=fmaf(s.normalized[b*N+j],w.p[2][t*N+j],sum);
        sum+=w.p[3][t];const float value=.5f*sum*(1+erff(sum*.7071067811865475244f));
        s.hidden[b*H+t]=value;if(!isfinite(value))atomicExch(status+b,2u);}
}
__global__ void finish(uint32_t* status,Weights w,Scratch s) {
    const int b=blockIdx.x,t=threadIdx.x;__shared__ uint32_t bad;
    if(t==0)bad=status[b];__syncthreads();if(bad)return;
    if(t<2){float sum=0;for(int j=0;j<H;++j)sum=fmaf(s.hidden[b*H+j],w.p[4][t*H+j],sum);
        float value=sum+w.p[5][t];s.logits[b*2+t]=value;if(!isfinite(value))atomicExch(status+b,2u);}
}
__global__ void publish(float* output,const uint32_t* status,Scratch s,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count*2 && !status[i/2])output[i]=s.logits[i];
}
bool explicit_stream(void* s){return s && s!=cudaStreamLegacy && s!=cudaStreamPerThread;}
}
struct ReadinessMapper::Impl {
    float* storage=nullptr;Weights weights{};Scratch scratch{};std::size_t bytes=0;
    uint32_t capacity=0;cudaStream_t stream=nullptr;
    ~Impl(){if(stream)cudaStreamSynchronize(stream);if(storage)cudaFree(storage);}
};
ReadinessMapper::~ReadinessMapper(){delete impl_;}
std::size_t ReadinessMapper::device_bytes()const{return impl_?impl_->bytes:0;}
bool ReadinessMapper::load(const adapter::Artifact& a,void* stream,uint32_t capacity,std::string& error) {
    if(impl_ || a.kind!=3 || !explicit_stream(stream) || !capacity || capacity>64 || a.offsets.size()!=6 || a.payload.size()>512*1024){error="readiness_load_arguments";return false;}
    const auto schema=adapter_schema::get(3);
    for(int j=0;j<6;++j)if(a.tensor(j).size()!=schema.tensors[j].count){error="readiness_tensor_schema";return false;}
    for(float value:a.payload)if(!std::isfinite(value)){error="readiness_weight_nonfinite";return false;}
    for(float value:a.tensor(1))if(value<.0001f){error="readiness_scale";return false;}
    auto* p=new Impl;p->stream=static_cast<cudaStream_t>(stream);p->capacity=capacity;
    p->bytes=(a.payload.size()+capacity*(N+H+2))*sizeof(float);
    auto status=cudaMalloc(reinterpret_cast<void**>(&p->storage),p->bytes);
    if(status==cudaSuccess)status=cudaMemcpyAsync(p->storage,a.payload.data(),a.payload.size()*4,cudaMemcpyHostToDevice,p->stream);
    if(status==cudaSuccess)status=cudaStreamSynchronize(p->stream);
    if(status!=cudaSuccess){error=cudaGetErrorString(status);delete p;return false;}
    for(int j=0;j<6;++j)p->weights.p[j]=p->storage+a.offsets[j];
    p->scratch.normalized=p->storage+a.payload.size();p->scratch.hidden=p->scratch.normalized+capacity*N;
    p->scratch.logits=p->scratch.hidden+capacity*H;
    impl_=p;error.clear();return true;
}
bool ReadinessMapper::run(const float* mixed,float* logits,uint32_t* status,uint32_t count,void* stream,bool enabled,std::string& error) {
    if(!enabled){error.clear();return true;}
    if(!impl_ || !mixed || !logits || !status || !count || count>impl_->capacity || stream!=impl_->stream){error="readiness_run_arguments";return false;}
    const auto st=impl_->stream;const auto w=impl_->weights;const auto s=impl_->scratch;
    prepare<<<count,256,0,st>>>(mixed,status,w,s);hidden<<<count,128,0,st>>>(status,w,s);finish<<<count,32,0,st>>>(status,w,s);
    publish<<<(count*2+127)/128,128,0,st>>>(logits,status,s,count);
    const auto result=cudaPeekAtLastError();if(result!=cudaSuccess){error=cudaGetErrorString(result);return false;}
    error.clear();return true;
}
}
