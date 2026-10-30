#include "strata/program/igc_learned_diagnostic.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
using strata::program::IgcLearnedDiagnostic;
static void require(bool v,const std::string& why){if(!v)throw std::runtime_error(why);}
static void check(cudaError_t c){require(c==cudaSuccess,cudaGetErrorString(c));}
int main(int argc,char** argv)try{
    require(argc==6 || argc==7,"bundle tokenizer model plan fresh-root [cancel|missing|uncommitted]");
    const std::string mode=argc==7?argv[6]:"normal";bool refused=false;
    strata::program::igc_coherent::Plan plan;std::string e;
    require(strata::program::igc_coherent::load_plan(argv[4],plan,e),e);
    cudaStream_t stream=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    IgcLearnedDiagnostic d;require(d.load_host(argv[1],e),e);require(d.bind(argv[2],{argv[3]},e),e);
    require(d.observe_only(argv[4],argv[5],e),e);auto hook=d.hook();require(hook.prepare(hook.user,stream,e),e);
    require(d.device_bytes()==4ull*80*10240*4+80+(2560+248320)*4+2560/32*36,"observer bounded context/snapshot/replay scratch, no learned pipeline");
    float *source=nullptr,*mixed=nullptr;check(cudaMalloc(&source,256ull*10240*4));check(cudaMalloc(&mixed,2560*4));
    void* toy_weights=nullptr;
    if(mode=="normal") {
        constexpr size_t bytes=uint64_t(2560/32)*34*248320;size_t free=0,total=0;check(cudaMemGetInfo(&free,&total));
        require(free>bytes+(512ull<<20),"toy head memory admission");check(cudaMalloc(&toy_weights,bytes));check(cudaMemsetAsync(toy_weights,0,bytes,stream));
        std::array<uint8_t,34> one{};one[1]=0x3c;one[2]=1; // Q8_0 scale1, first coefficient1.
        check(cudaMemcpyAsync(static_cast<uint8_t*>(toy_weights)+17*uint64_t(2560/32)*34,one.data(),one.size(),cudaMemcpyHostToDevice,stream));
        check(cudaStreamSynchronize(stream));
    }
    // Constant127 has exactly representable Q8_1 scale1 and integer127.
    std::vector<float> base(2560,127.0f),actual(2560);
    check(cudaMemcpyAsync(mixed,base.data(),base.size()*4,cudaMemcpyHostToDevice,stream));
    // Capture before any begin/stage. The observer head is a stable no-op; a
    // cached T1 graph cannot bake in a host arming decision or inject old output.
    cudaGraph_t graph=nullptr;cudaGraphExec_t instance=nullptr;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));require(hook.run(hook.user,mixed,1,stream,e),e);
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));
    for(size_t i=0;i<plan.records.size();++i){
        const auto& r=plan.records[i];require(d.begin(r.tokens,0,false,true,e),e);
        if(i==0 && mode=="missing") {
            require(!d.stage(e) && e=="coherent_missing_cached_features","missing prefix must fail without manufactured states");
            require(d.cancel(e),e);refused=true;break;
        }
        auto consumer=d.prefill_consumer();const auto anchor=static_cast<int64_t>(r.tokens.size()-1);
        auto observe=[&](int64_t pos,int64_t n){
            std::vector<float> data(static_cast<size_t>(n)*10240);
            for(auto tap:{strata::core::EarlyActivationTap::Embedding,strata::core::EarlyActivationTap::FirstAttention}){
                for(int64_t j=0;j<n;++j)std::fill_n(data.data()+j*10240,10240,float(r.tokens[static_cast<size_t>(pos+j)])+(tap==strata::core::EarlyActivationTap::FirstAttention?1000.0f:0.0f));
                check(cudaMemcpyAsync(source,data.data(),data.size()*4,cudaMemcpyHostToDevice,stream));
                const strata::core::EarlyActivationRows rows{tap,source,n,10240,pos,stream};require(consumer.consume(consumer.user,rows,e),e);
                check(cudaStreamSynchronize(stream));
            }
        };
        if(!r.role)observe(std::max<int64_t>(0,anchor-80),std::min<int64_t>(anchor,80));
        require(d.accept_prefill(0,{r.tokens.data(),static_cast<size_t>(anchor)},e),e);
        require(d.stage(e),e);
        if(i==0 && (mode=="cancel" || mode=="uncommitted")) {
            if(mode=="cancel")require(d.cancel(e),e);
            bool called=false;
            require(!d.export_observation(mode!="uncommitted",[&](auto,auto){called=true;return true;},e),"cancelled/incomplete export refused");
            require(!called && !std::filesystem::exists(std::filesystem::path(argv[5])/"0.bin"),"no stale final copy or success file");
            require(!d.begin(r.tokens,0,false,true,e),"failed archive cannot resume another owner");refused=true;break;
        }
        if(!r.role)observe(anchor,1);
        require(d.accept_prefill(anchor,{r.tokens.data()+anchor,1},e),e);
        check(cudaGraphLaunch(instance,stream));
        require(d.export_observation(true,[&](std::span<float> m,std::span<float> logits){
            check(cudaMemcpyAsync(actual.data(),mixed,actual.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
            require(actual==base,"observer graph preserves mixed bytes");std::copy(actual.begin(),actual.end(),m.begin());
            std::fill(logits.begin(),logits.end(),0.0f);logits[17]=127.0f;
            if(i==0) {
                require(d.prove_head(8,toy_weights,logits,e),e);
                logits[17]=128.0f;require(!d.prove_head(8,toy_weights,logits,e),"incorrect replay reference must fail");
                logits[17]=127.0f;require(d.prove_head(8,toy_weights,logits,e),e);
            }
            return true;
        },e),e);
    }
    if(!refused)require(!d.begin(plan.records.front().tokens,0,false,true,e),"exhausted plan cannot publish another owner");
    check(cudaGraphExecDestroy(instance));check(cudaGraphDestroy(graph));hook.release(hook.user);
    strata::program::igc_coherent::Writer duplicate;
    require(!duplicate.open(argv[4],argv[5],plan.model,plan.tokenizer,e),"existing/partial native archive cannot be resumed or overwritten");
    require(hook.prepare(hook.user,stream,e),e);
    require(!d.begin(plan.records.front().tokens,0,false,true,e),"release/reprepare cannot revive an exhausted or failed archive");
    hook.release(hook.user);
    if(toy_weights)check(cudaFree(toy_weights));
    check(cudaFree(source));check(cudaFree(mixed));check(cudaStreamDestroy(stream));
    std::cout<<"PASS synthetic coherent observer mode="<<mode<<" no-injection graph/ring-wrap/owned-output/export; no language-model inference\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
