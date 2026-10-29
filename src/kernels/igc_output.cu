#include "strata/core/igc_output.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace strata::igc {
namespace {
constexpr int N=2560,D=128,L=62,B=64;
struct Weights {const float* p[15]{};};
struct Scratch {float *norm,*tokens,*keys,*values,*query,*result,*gate,*proposal;};
__device__ float sum_block(float x,float* shared) {
    shared[threadIdx.x]=x;__syncthreads();
    for(int step=blockDim.x/2;step;step/=2) {
        if(threadIdx.x<step)shared[threadIdx.x]+=shared[threadIdx.x+step];
        __syncthreads();
    }
    const float result=shared[0];
    // Every thread must consume the reduction before the next invocation can
    // overwrite this same shared buffer. The final reduction barrier alone
    // protects the writes, not the subsequent reads across warps.
    __syncthreads();
    return result;
}
__device__ uint32_t validate(const ExactResult& r,uint64_t id,uint64_t anchor) {
    if(r.version!=1 || !id || r.invocation!=id || r.position!=anchor || static_cast<uint32_t>(r.status)>4)return 2;
    if(r.status!=Status::Success) {
        const auto* bytes=reinterpret_cast<const uint8_t*>(&r.value);
        for(int i=0;i<64;++i)if(bytes[i])return 2;
        return 1;
    }
    const auto& v=r.value;
    if(!v.length || v.length>L || v.negative>1)return 2;
    for(int i=0;i<L;++i)if(v.digits[i]>9 || (i>=v.length && v.digits[i]))return 2;
    if((v.length>1 && !v.digits[v.length-1]) || (v.length==1 && !v.digits[0] && v.negative))return 2;
    return 0;
}
__global__ void prepare(const float* x,const ExactResult* r,const uint64_t* ids,const uint64_t* anchors,
                        uint32_t* status,Weights w,Scratch s) {
    const int row=blockIdx.x;__shared__ float sums[256];__shared__ float mean,inverse;
    if(threadIdx.x==0)status[row]=validate(r[row],ids[row],anchors[row]);
    __syncthreads();if(status[row])return;
    float bad=0,sum=0;
    for(int i=threadIdx.x;i<N;i+=blockDim.x){float v=x[row*N+i];bad+=!isfinite(v);sum+=v;}
    if(sum_block(bad,sums)>0){if(threadIdx.x==0)status[row]=3;return;}
    const float total=sum_block(sum,sums);if(threadIdx.x==0)mean=total/N;__syncthreads();
    float squared=0;for(int i=threadIdx.x;i<N;i+=blockDim.x){float v=x[row*N+i]-mean;squared+=v*v;}
    const float variance=sum_block(squared,sums)/N;
    if(threadIdx.x==0){inverse=rsqrtf(variance+1e-5f);if(!isfinite(inverse) || !isfinite(mean))status[row]=3;}
    __syncthreads();if(status[row])return;
    for(int i=threadIdx.x;i<N;i+=blockDim.x)s.norm[row*N+i]=(x[row*N+i]-mean)*inverse*w.p[0][i]+w.p[1][i];
}
__global__ void result_tokens(const ExactResult* records,const uint32_t* status,Weights w,Scratch s,int count) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count*L*D)return;
    const int neuron=index%D,position=(index/D)%L,row=index/(L*D);if(status[row])return;
    const auto& v=records[row].value;
    // PAD features are all zero, exactly as in the training encoder. Positions
    // still exist but are masked out of attention scores below.
    float value=w.p[5][neuron]+w.p[6][position*D+neuron];
    if(position<v.length)value+=w.p[4][neuron*12+v.digits[v.length-1-position]]+w.p[4][neuron*12+11]*v.negative;
    s.tokens[index]=value;
}
__global__ void linear(const float* input,const float* weight,const float* bias,float* output,
                       const uint32_t* status,int rows_per_sample,int width,int out,int rows) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=rows*out)return;
    const int neuron=index%out,row=index/out;if(status[row/rows_per_sample])return;
    float sum=0;for(int i=0;i<width;++i)sum=fmaf(input[row*width+i],weight[neuron*width+i],sum);
    output[index]=sum+bias[neuron];
}
__global__ void attend(const ExactResult* records,uint32_t* status,Scratch s) {
    int row=blockIdx.x;if(status[row])return;
    __shared__ float scores[L];const int t=threadIdx.x;
    if(t<L){
        float sum=0;for(int i=0;i<D;++i)sum=fmaf(s.query[row*D+i],s.keys[(row*L+t)*D+i],sum);
        scores[t]=t<records[row].value.length?sum/sqrtf(float(D)):-INFINITY;
    }
    __syncthreads();
    if(t==0){
        float maximum=-INFINITY;for(int i=0;i<L;++i)maximum=fmaxf(maximum,scores[i]);
        float total=0;for(int i=0;i<L;++i){scores[i]=expf(scores[i]-maximum);total+=scores[i];}
        if(!isfinite(total) || total<=0)status[row]=3;
        else for(int i=0;i<L;++i)scores[i]/=total;
    }
    __syncthreads();if(status[row])return;
    if(t<D){float sum=0;for(int j=0;j<L;++j)sum=fmaf(scores[j],s.values[(row*L+j)*D+t],sum);s.result[row*D+t]=sum;}
}
__global__ void gate(uint32_t* status,Weights w,Scratch s) {
    int row=blockIdx.x;if(status[row])return;__shared__ float sums[256];float sum=0;
    for(int i=threadIdx.x;i<N+D;i+=blockDim.x)sum=fmaf(w.p[13][i],i<N?s.norm[row*N+i]:s.result[row*D+i-N],sum);
    float total=sum_block(sum,sums)+w.p[14][0];
    if(threadIdx.x==0){s.gate[row]=1/(1+expf(-total));if(!isfinite(total))status[row]=3;}
}
__global__ void propose(const float* original,uint32_t* status,Weights w,Scratch s,int count) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count*N)return;
    int row=index/N,neuron=index%N;if(atomicAdd(status+row,0u))return;
    float sum=0;for(int i=0;i<D;++i)sum=fmaf(s.result[row*D+i],w.p[11][neuron*D+i],sum);
    float value=original[index]+s.gate[row]*(sum+w.p[12][neuron]);s.proposal[index]=value;
    if(!isfinite(value))atomicExch(status+row,3u);
}
__global__ void publish(float* output,const uint32_t* status,Scratch s,int count) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index<count*N && !status[index/N])output[index]=s.proposal[index];
}
bool explicit_stream(void* value){return value && value!=cudaStreamLegacy && value!=cudaStreamPerThread;}
}
struct OutputMapper::Impl {
    float* storage=nullptr;std::size_t bytes=0;Weights weights{};Scratch scratch{};cudaStream_t stream=nullptr;
    ~Impl(){if(stream)cudaStreamSynchronize(stream);if(storage)cudaFree(storage);}
};
OutputMapper::~OutputMapper(){delete impl_;}
std::size_t OutputMapper::device_bytes()const{return impl_?impl_->bytes:0;}
bool OutputMapper::load(const adapter::Artifact& artifact,void* stream,std::string& error) {
    if(impl_ || artifact.kind!=2 || !explicit_stream(stream) || artifact.offsets.size()!=15 || artifact.payload.size()>2*1024*1024){error="output_load_arguments";return false;}
    const auto definition=adapter_schema::get(2);
    for(std::size_t i=0;i<15;++i)if(artifact.tensor(i).size()!=definition.tensors[i].count){error="output_tensor_schema";return false;}
    for(float value:artifact.payload)if(!std::isfinite(value)){error="output_weight_nonfinite";return false;}
    auto* p=new Impl;p->stream=static_cast<cudaStream_t>(stream);
    const std::size_t scratch=B*(2*N+3*L*D+2*D+1);p->bytes=(artifact.payload.size()+scratch)*sizeof(float);
    auto status=cudaMalloc(reinterpret_cast<void**>(&p->storage),p->bytes);
    if(status==cudaSuccess)status=cudaMemcpyAsync(p->storage,artifact.payload.data(),artifact.payload.size()*4,cudaMemcpyHostToDevice,p->stream);
    if(status==cudaSuccess)status=cudaStreamSynchronize(p->stream); // startup only; source lifetime fenced
    if(status!=cudaSuccess){error=cudaGetErrorString(status);delete p;return false;}
    for(std::size_t i=0;i<15;++i)p->weights.p[i]=p->storage+artifact.offsets[i];
    float* cursor=p->storage+artifact.payload.size();
    auto take=[&](std::size_t count){float* result=cursor;cursor+=count;return result;};
    p->scratch.norm=take(B*N);p->scratch.tokens=take(B*L*D);p->scratch.keys=take(B*L*D);p->scratch.values=take(B*L*D);
    p->scratch.query=take(B*D);p->scratch.result=take(B*D);p->scratch.gate=take(B);p->scratch.proposal=take(B*N);
    impl_=p;error.clear();return true;
}
bool OutputMapper::run(float* mixed,const ExactResult* results,const uint64_t* ids,const uint64_t* anchors,
                       uint32_t* status,uint32_t count,void* stream,bool enabled,std::string& error) {
    if(!enabled){error.clear();return true;} // hard bypass, no argument/device work
    if(!impl_ || !mixed || !results || !ids || !anchors || !status || !count || count>B || stream!=impl_->stream){error="output_run_arguments";return false;}
    const auto s=impl_->scratch;const auto w=impl_->weights;const auto st=impl_->stream;
    prepare<<<count,256,0,st>>>(mixed,results,ids,anchors,status,w,s);
    result_tokens<<<(count*L*D+127)/128,128,0,st>>>(results,status,w,s,count);
    linear<<<(count*D+127)/128,128,0,st>>>(s.norm,w.p[2],w.p[3],s.query,status,1,N,D,count);
    linear<<<(count*L*D+127)/128,128,0,st>>>(s.tokens,w.p[7],w.p[8],s.keys,status,L,D,D,count*L);
    linear<<<(count*L*D+127)/128,128,0,st>>>(s.tokens,w.p[9],w.p[10],s.values,status,L,D,D,count*L);
    attend<<<count,128,0,st>>>(results,status,s);
    gate<<<count,256,0,st>>>(status,w,s);
    propose<<<(count*N+127)/128,128,0,st>>>(mixed,status,w,s,count);
    publish<<<(count*N+127)/128,128,0,st>>>(mixed,status,s,count);
    const auto result=cudaPeekAtLastError();if(result!=cudaSuccess){error=cudaGetErrorString(result);return false;}
    error.clear();return true;
}
} // namespace strata::igc
