#include "strata/core/igc_context.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata::igc;
static void require(bool yes,const std::string& why){if(!yes)throw std::runtime_error(why);}
static void check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
struct Stream{cudaStream_t s=nullptr;Stream(){check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){if(s){cudaStreamSynchronize(s);cudaStreamDestroy(s);}}};
struct Buffer{float* p=nullptr;explicit Buffer(size_t n){check(cudaMalloc(&p,n*4));}~Buffer(){if(p)cudaFree(p);}};
int main()try{
    Stream decoder,producer;std::string error;ContextWindow window;
    require(window.allocate(decoder.s,error),"context_allocate_contract");
    require(window.device_bytes()==4ull*80*10240*4+80,"bounded_device_bytes");
    std::vector<int64_t> prefix(192);for(int i=0;i<192;++i)prefix[i]=100+i;
    require(window.begin(prefix,0,false,error),error);
    require(!window.gather(191,decoder.s,error) && error.empty(),"unobserved_prefix_bypasses");
    Buffer source(192ull*ContextWindow::width);
    auto stage=[&](int64_t p,int count,ContextTap tap,cudaStream_t stream){
        std::vector<float> data(size_t(count)*ContextWindow::width);
        for(int t=0;t<count;++t)for(int k=0;k<ContextWindow::width;++k)data[size_t(t)*ContextWindow::width+k]=float((p+t)*20000+k+int(tap)*10000000);
        check(cudaMemcpyAsync(source.p,data.data(),data.size()*4,cudaMemcpyHostToDevice,stream));
        require(window.observe(tap,p,count,source.p,stream,error),error);
        // Host upload storage may be reused only after the upload's stream ends.
        check(cudaStreamSynchronize(stream));
    };
    auto verify=[&](int anchor){
        require(window.gather(anchor,decoder.s,error),error);
        std::vector<uint8_t> mask(80);check(cudaMemcpyAsync(mask.data(),window.mask(),80,cudaMemcpyDeviceToHost,decoder.s));
        const int n=std::min(anchor,80),first=anchor-n;
        std::vector<float> data(80ull*ContextWindow::width);
        for(auto tap:{ContextTap::Embedding,ContextTap::FirstAttention}){
            check(cudaMemcpyAsync(data.data(),window.states(tap),data.size()*4,cudaMemcpyDeviceToHost,decoder.s));check(cudaStreamSynchronize(decoder.s));
            for(int t=0;t<80;++t){require(mask[t]==(t>=80-n),"right_aligned_mask");if(t<80-n)continue;
                for(int k=0;k<ContextWindow::width;++k)require(data[size_t(t)*ContextWindow::width+k]==float((first+t-(80-n))*20000+k+int(tap)*10000000),"genuine_position_and_tap_bytes");
            }
        }
    };
    // Stage both taps without accepting: no model-owned prefix yet.
    stage(0,32,ContextTap::Embedding,producer.s);stage(0,32,ContextTap::FirstAttention,producer.s);
    require(!window.gather(31,decoder.s,error) && error.empty(),"staged_is_not_owned");
    require(window.accept(0,{prefix.data(),32},error),error);verify(32);
    require(!window.accept(34,{prefix.data()+34,1},error),"noncontiguous_accept_refused");
    // True retained prefix survives a request continuation, with stream crossing.
    require(window.begin(prefix,32,true,error),error);
    stage(32,64,ContextTap::Embedding,decoder.s);stage(32,64,ContextTap::FirstAttention,producer.s);
    require(window.accept(32,{prefix.data()+32,64},error),error);verify(96);
    // Speculative source contains rejected rows, but only the first3 are copied
    // and accepted. They cannot displace the old accepted tail with rejected data.
    stage(96,3,ContextTap::Embedding,producer.s);stage(96,3,ContextTap::FirstAttention,decoder.s);
    require(window.accept(96,{prefix.data()+96,3},error),error);verify(99);
    // Missing second tap fails closed; it must not pair with a stale ring slot.
    stage(99,1,ContextTap::Embedding,producer.s);require(window.accept(99,{prefix.data()+99,1},error),error);
    require(!window.gather(100,decoder.s,error) && error.empty(),"missing_tap_bypasses");
    std::vector<uint8_t> unavailable(80);
    check(cudaMemcpyAsync(unavailable.data(),window.mask(),80,cudaMemcpyDeviceToHost,decoder.s));check(cudaStreamSynchronize(decoder.s));
    require(std::all_of(unavailable.begin(),unavailable.end(),[](uint8_t b){return b==0;}),"unavailable_mask_cleared");
    stage(99,1,ContextTap::FirstAttention,decoder.s);verify(100);
    // A checkpoint rewind can reuse only the rows actually retained in the ring.
    require(window.begin(prefix,95,true,error),error);
    require(!window.gather(95,decoder.s,error) && error.empty(),"evicted_rewind_rows_bypass");
    stage(15,5,ContextTap::Embedding,producer.s);stage(15,5,ContextTap::FirstAttention,producer.s);verify(95);
    require(!window.gather(100,decoder.s,error) && error.empty(),"rewound_suffix_not_owned");
    // A changed source prefix invalidates all features even if reuse was claimed.
    auto changed=prefix;changed[0]++;
    require(window.begin(changed,95,true,error),error);
    require(!window.gather(95,decoder.s,error) && error.empty(),"mismatched_lineage_bypasses");
    require(window.begin(prefix,0,false,error),error);
    // A whole large batched chunk needs only its last80 rows and fixed storage.
    stage(0,191,ContextTap::Embedding,producer.s);stage(0,191,ContextTap::FirstAttention,producer.s);
    require(window.accept(0,{prefix.data(),191},error),error);verify(191);
    require(window.device_bytes()==4ull*80*10240*4+80,"fixed_allocation_after_wrap");
    stage(191,2,ContextTap::Embedding,producer.s);stage(191,2,ContextTap::FirstAttention,decoder.s);
    const int64_t generated[2]={prefix[191],900};require(window.accept(191,generated,error),error);
    auto continued=prefix;continued.push_back(900);continued.push_back(901);
    require(window.begin(continued,193,true,error),error);verify(193);
    // User/slot/image/control changes explicitly deny reuse without an LM reset.
    require(window.begin(prefix,191,false,error),error);
    require(!window.gather(191,decoder.s,error) && error.empty(),"scope_change_bypasses");
    require(!window.observe(ContextTap::Embedding,-1,1,source.p,producer.s,error),"bad_position_refused");
    require(!window.observe(ContextTap::Embedding,0,1,source.p,cudaStreamLegacy,error),"default_stream_refused");
    require(!window.begin({},0,false,error),"empty_prefix_refused");
    std::cout<<"PASS actual-state context ring ownership/right-align/cache/rewind/missing/scope/cross-stream\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
