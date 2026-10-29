#include "strata/core/igc_dispatch.hpp"
#include <cuda_runtime.h>
namespace strata::igc {
namespace {
__device__ int argmax(const float* values,int count) {
    int best=0;for(int i=1;i<count;++i)if(values[i]>values[best])best=i;return best;
}
__device__ bool canonical(int sign,int length,int ones,int tens) {
    return !(length==0 && tens!=0) && !(length==1 && tens==0)
        && !(sign && length==0 && ones==0);
}
__global__ void assemble(SelectionInputs in,ExactRequest* output,uint32_t* status,uint32_t count) {
    const auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    ExactRequest q{};q.domain=0;q.operation=static_cast<Operation>(0);
    q.invocation=in.invocation[i];q.position=in.anchor[i]<262144?in.anchor[i]:0;
    uint32_t code=0;
    if(!q.invocation || in.anchor[i]>=262144 || in.readiness_status[i])code=2;
    else {
        const float* ready=in.readiness+i*2;
        if(!isfinite(ready[0]) || !isfinite(ready[1]))code=3;
        else if(ready[0]>=ready[1])code=1;
        else if(in.primary_status[i] || in.secondary_status[i])code=2;
        else {
            const float* a=in.primary+i*52;const float* b=in.secondary+i*52;
            for(int j=0;j<52;++j)if(!isfinite(a[j]) || !isfinite(b[j])){code=3;break;}
            if(!code){
                const int sizes[9]={4,2,2,2,2,10,10,10,10};int fields[9]{},offset=0;
                for(int h=0;h<9;++h){fields[h]=argmax(a+offset,sizes[h]);
                    if(fields[h]!=argmax(b+offset,sizes[h]))code=4;offset+=sizes[h];}
                if(!code){
                    if(!fields[0])code=1;
                    else if(!canonical(fields[1],fields[3],fields[5],fields[6])
                            || !canonical(fields[2],fields[4],fields[7],fields[8]))code=5;
                    else {
                        q.domain=1;q.operation=static_cast<Operation>(fields[0]);
                        q.lhs.length=uint8_t(fields[3]+1);q.lhs.negative=uint8_t(fields[1]);
                        q.lhs.digits[0]=uint8_t(fields[5]);q.lhs.digits[1]=uint8_t(fields[6]);
                        q.rhs.length=uint8_t(fields[4]+1);q.rhs.negative=uint8_t(fields[2]);
                        q.rhs.digits[0]=uint8_t(fields[7]);q.rhs.digits[1]=uint8_t(fields[8]);
                    }
                }
            }
        }
    }
    // Inactive requests are deliberately overwritten too: no previous success
    // can survive a no-call, disagreement or failed neural observation.
    output[i]=q;status[i]=code;
}
}
bool assemble_selective_exact(const SelectionInputs& in,ExactRequest* output,uint32_t* status,uint32_t count,
                              void* stream,bool enabled,std::string& error) {
    if(!enabled){error.clear();return true;}
    if(!in.primary || !in.secondary || !in.readiness || !in.primary_status || !in.secondary_status
            || !in.readiness_status || !in.invocation || !in.anchor || !output || !status || !count || count>64
            || !stream || stream==cudaStreamLegacy || stream==cudaStreamPerThread){error="dispatch_arguments";return false;}
    assemble<<<1,64,0,static_cast<cudaStream_t>(stream)>>>(in,output,status,count);
    const auto result=cudaPeekAtLastError();if(result!=cudaSuccess){error=cudaGetErrorString(result);return false;}
    error.clear();return true;
}
}
