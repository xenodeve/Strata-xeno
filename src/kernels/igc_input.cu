#include "strata/core/igc_input.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>

namespace strata::igc {
namespace {
constexpr int N=2560,L=80,D=64,H=4,F=128,O=52;
struct Weights {const float* p[40]{};};
struct Scratch {float *tokens,*qkv,*attended,*ff,*keys,*values,*logits;};
__device__ float reduce(float x,float* shared) {
    shared[threadIdx.x]=x;__syncthreads();
    for(int s=blockDim.x/2;s;s/=2){if(threadIdx.x<s)shared[threadIdx.x]+=shared[threadIdx.x+s];__syncthreads();}
    float result=shared[0];__syncthreads();return result;
}
__device__ bool failed(uint32_t* status,int sample) {
    __shared__ uint32_t value;
    if(threadIdx.x==0)value=atomicAdd(status+sample,0u);
    __syncthreads();const bool result=value!=0;__syncthreads();return result;
}
__device__ bool normalize(float* row,int n,const float* weight,const float* bias,float* sums) {
    float sum=0;for(int j=threadIdx.x;j<n;j+=blockDim.x)sum+=row[j];
    const float mean=reduce(sum,sums)/n;
    float squared=0;for(int j=threadIdx.x;j<n;j+=blockDim.x){float v=row[j]-mean;squared+=v*v;}
    const float variance=reduce(squared,sums)/n,inverse=rsqrtf(variance+1e-5f);
    if(!isfinite(mean) || !isfinite(variance) || !isfinite(inverse))return false;
    for(int j=threadIdx.x;j<n;j+=blockDim.x)row[j]=(row[j]-mean)*inverse*weight[j]+bias[j];
    __syncthreads();return true;
}
__global__ void validate(const float* states,const uint8_t* valid,uint32_t* status) {
    const int b=blockIdx.x;__shared__ float sums[256];float mask_bad=0,seen=0,bad=0;
    for(int p=threadIdx.x;p<L;p+=blockDim.x){uint8_t v=valid[b*L+p];mask_bad+=v>1;seen+=v==1;}
    const float masks=reduce(mask_bad,sums),observed=reduce(seen,sums);
    for(int j=threadIdx.x;j<L*H*N;j+=blockDim.x)if(valid[b*L+j/(H*N)]==1)bad+=!isfinite(states[b*L*H*N+j]);
    const float nonfinite=reduce(bad,sums);
    if(threadIdx.x==0)status[b]=(masks || !observed)?1u:nonfinite?2u:0u;
}
__global__ void project(const float* states,const uint8_t* valid,uint32_t* status,Weights w,Scratch s) {
    const int flat=blockIdx.x,h=flat%H,p=(flat/H)%L,b=flat/(H*L);
    if(failed(status,b))return;
    __shared__ float row[N],sums[256];
    for(int j=threadIdx.x;j<N;j+=blockDim.x)row[j]=valid[b*L+p]?states[flat*N+j]:0;
    __syncthreads();
    if(!normalize(row,N,w.p[1],w.p[2],sums)){if(threadIdx.x==0)atomicExch(status+b,2u);return;}
    if(threadIdx.x<16){
        const int k=threadIdx.x;float sum=0;
        for(int j=0;j<N;++j)sum=fmaf(row[j],w.p[3][k*N+j],sum);
        s.tokens[(b*L+p)*D+h*16+k]=sum+w.p[4][k]+w.p[5][p*D+h*16+k];
    }
}
__global__ void qkv(uint32_t* status,Weights w,Scratch s) {
    const int row_index=blockIdx.x,b=row_index/L;if(failed(status,b))return;
    __shared__ float row[D],sums[256];
    if(threadIdx.x<D)row[threadIdx.x]=s.tokens[row_index*D+threadIdx.x];__syncthreads();
    if(!normalize(row,D,w.p[36],w.p[37],sums)){if(threadIdx.x==0)atomicExch(status+b,2u);return;}
    if(threadIdx.x<3*D){int k=threadIdx.x;float sum=0;for(int j=0;j<D;++j)sum=fmaf(row[j],w.p[28][k*D+j],sum);
        s.qkv[row_index*3*D+k]=sum+w.p[29][k];}
}
__global__ void self_attention(const uint8_t* valid,uint32_t* status,Scratch s) {
    const int p=blockIdx.x%L,b=blockIdx.x/L,h=blockIdx.y,t=threadIdx.x;
    if(failed(status,b))return;__shared__ float scores[L];__shared__ uint32_t bad;
    if(t<L){float sum=0;for(int j=0;j<16;++j)sum=fmaf(s.qkv[(b*L+p)*3*D+h*16+j],s.qkv[(b*L+t)*3*D+D+h*16+j],sum);
        scores[t]=valid[b*L+t]?sum/4:-INFINITY;}
    __syncthreads();
    if(t==0){float maximum=-INFINITY;for(int j=0;j<L;++j)maximum=fmaxf(maximum,scores[j]);
        float total=0;for(int j=0;j<L;++j){scores[j]=expf(scores[j]-maximum);total+=scores[j];}
        bad=!isfinite(total) || total<=0;if(bad)atomicExch(status+b,2u);else for(int j=0;j<L;++j)scores[j]/=total;}
    __syncthreads();if(bad)return;
    if(t<16){float sum=0;for(int j=0;j<L;++j)sum=fmaf(scores[j],s.qkv[(b*L+j)*3*D+2*D+h*16+t],sum);
        s.attended[(b*L+p)*D+h*16+t]=sum;}
}
__global__ void feedforward_prepare(uint32_t* status,Weights w,Scratch s) {
    const int index=blockIdx.x,b=index/L,t=threadIdx.x;if(failed(status,b))return;
    __shared__ float row[D],sums[256];
    if(t<D){float sum=0;for(int j=0;j<D;++j)sum=fmaf(s.attended[index*D+j],w.p[30][t*D+j],sum);
        row[t]=s.tokens[index*D+t]+sum+w.p[31][t];s.tokens[index*D+t]=row[t];}
    __syncthreads();
    if(!normalize(row,D,w.p[38],w.p[39],sums)){if(t==0)atomicExch(status+b,2u);return;}
    if(t<F){float sum=0;for(int j=0;j<D;++j)sum=fmaf(row[j],w.p[32][t*D+j],sum);
        sum+=w.p[33][t];s.ff[index*F+t]=.5f*sum*(1+erff(sum*.7071067811865475244f));}
}
__global__ void feedforward_finish(uint32_t* status,Weights w,Scratch s) {
    const int index=blockIdx.x,b=index/L,t=threadIdx.x;if(failed(status,b))return;
    if(t<D){float sum=0;for(int j=0;j<F;++j)sum=fmaf(s.ff[index*F+j],w.p[34][t*F+j],sum);
        s.tokens[index*D+t]+=sum+w.p[35][t];}
}
__global__ void key_value(uint32_t* status,Weights w,Scratch s) {
    const int index=blockIdx.x,b=index/L,t=threadIdx.x;if(failed(status,b))return;
    if(t<D){float key=0,value=0;for(int j=0;j<D;++j){float v=s.tokens[index*D+j];
            key=fmaf(v,w.p[6][t*D+j],key);value=fmaf(v,w.p[8][t*D+j],value);}
        s.keys[index*D+t]=key+w.p[7][t];s.values[index*D+t]=value+w.p[9][t];}
}
__global__ void head(const uint8_t* valid,uint32_t* status,Weights w,Scratch s) {
    const int b=blockIdx.x,h=blockIdx.y,t=threadIdx.x;if(failed(status,b))return;
    __shared__ float scores[L],result[D];__shared__ uint32_t bad;
    if(t<L){float sum=0;for(int j=0;j<D;++j)sum=fmaf(w.p[0][h*D+j],s.keys[(b*L+t)*D+j],sum);
        scores[t]=valid[b*L+t]?sum/8:-INFINITY;}
    __syncthreads();
    if(t==0){float maximum=-INFINITY;for(int j=0;j<L;++j)maximum=fmaxf(maximum,scores[j]);
        float total=0;for(int j=0;j<L;++j){scores[j]=expf(scores[j]-maximum);total+=scores[j];}
        bad=!isfinite(total)||total<=0;if(bad)atomicExch(status+b,2u);else for(int j=0;j<L;++j)scores[j]/=total;}
    __syncthreads();if(bad)return;
    if(t<D){float sum=0;for(int j=0;j<L;++j)sum=fmaf(scores[j],s.values[(b*L+j)*D+t],sum);result[t]=sum+w.p[0][h*D+t];}
    __syncthreads();
    const int width=h==0?4:h<5?2:10,offset=h==0?0:h<5?4+(h-1)*2:12+(h-5)*10;
    if(t<width){float sum=0;for(int j=0;j<D;++j)sum=fmaf(result[j],w.p[10+h*2][t*D+j],sum);
        float value=sum+w.p[11+h*2][t];s.logits[b*O+offset+t]=value;if(!isfinite(value))atomicExch(status+b,2u);}
}
__global__ void publish(float* output,const uint32_t* status,Scratch s,int count) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<count*O && !status[index/O])output[index]=s.logits[index];
}
bool explicit_stream(void* p){return p && p!=cudaStreamLegacy && p!=cudaStreamPerThread;}
}
struct InputMapper::Impl {
    float* storage=nullptr;Weights weights{};Scratch scratch{};std::size_t bytes=0;
    uint32_t capacity=0;cudaStream_t stream=nullptr;
    ~Impl(){if(stream)cudaStreamSynchronize(stream);if(storage)cudaFree(storage);}
};
InputMapper::~InputMapper(){delete impl_;}
std::size_t InputMapper::device_bytes()const{return impl_?impl_->bytes:0;}
bool InputMapper::load(const adapter::Artifact& a,void* stream,uint32_t capacity,std::string& error) {
    if(impl_ || a.kind!=1 || !explicit_stream(stream) || !capacity || capacity>64 || a.offsets.size()!=40 || a.payload.size()>1024*1024){error="input_load_arguments";return false;}
    const auto schema=adapter_schema::get(1);
    for(int j=0;j<40;++j)if(a.tensor(j).size()!=schema.tensors[j].count){error="input_tensor_schema";return false;}
    for(float value:a.payload)if(!std::isfinite(value)){error="input_weight_nonfinite";return false;}
    auto* p=new Impl;p->stream=static_cast<cudaStream_t>(stream);p->capacity=capacity;
    p->bytes=(a.payload.size()+capacity*(L*(D+3*D+D+F+D+D)+O))*sizeof(float);
    auto status=cudaMalloc(reinterpret_cast<void**>(&p->storage),p->bytes);
    if(status==cudaSuccess)status=cudaMemcpyAsync(p->storage,a.payload.data(),a.payload.size()*4,cudaMemcpyHostToDevice,p->stream);
    if(status==cudaSuccess)status=cudaStreamSynchronize(p->stream);
    if(status!=cudaSuccess){error=cudaGetErrorString(status);delete p;return false;}
    for(int j=0;j<40;++j)p->weights.p[j]=p->storage+a.offsets[j];
    float* cursor=p->storage+a.payload.size();auto take=[&](std::size_t count){float* result=cursor;cursor+=count;return result;};
    p->scratch.tokens=take(capacity*L*D);p->scratch.qkv=take(capacity*L*3*D);p->scratch.attended=take(capacity*L*D);
    p->scratch.ff=take(capacity*L*F);p->scratch.keys=take(capacity*L*D);p->scratch.values=take(capacity*L*D);p->scratch.logits=take(capacity*O);
    impl_=p;error.clear();return true;
}
bool InputMapper::run(const float* states,const uint8_t* valid,float* logits,uint32_t* status,uint32_t count,void* stream,bool enabled,std::string& error) {
    if(!enabled){error.clear();return true;}
    if(!impl_ || !states || !valid || !logits || !status || !count || count>impl_->capacity || stream!=impl_->stream){error="input_run_arguments";return false;}
    const auto st=impl_->stream;const auto s=impl_->scratch;const auto w=impl_->weights;
    validate<<<count,256,0,st>>>(states,valid,status);
    project<<<count*L*H,256,0,st>>>(states,valid,status,w,s);
    qkv<<<count*L,256,0,st>>>(status,w,s);
    self_attention<<<dim3(count*L,4),128,0,st>>>(valid,status,s);
    feedforward_prepare<<<count*L,256,0,st>>>(status,w,s);
    feedforward_finish<<<count*L,128,0,st>>>(status,w,s);
    key_value<<<count*L,128,0,st>>>(status,w,s);
    head<<<dim3(count,9),128,0,st>>>(valid,status,w,s);
    publish<<<(count*O+127)/128,128,0,st>>>(logits,status,s,count);
    const auto result=cudaPeekAtLastError();if(result!=cudaSuccess){error=cudaGetErrorString(result);return false;}
    error.clear();return true;
}
}
