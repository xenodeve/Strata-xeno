#include "strata/program/igc_learned_diagnostic.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using strata::program::IgcLearnedDiagnostic;
using namespace strata::igc;
static void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
static void check(cudaError_t c){require(c==cudaSuccess,cudaGetErrorString(c));}
struct Stream{cudaStream_t s=nullptr;Stream(){check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){if(s){cudaStreamSynchronize(s);cudaStreamDestroy(s);}}};
struct Buffer{float* p=nullptr;explicit Buffer(size_t count){check(cudaMalloc(&p,count*4));}~Buffer(){if(p)cudaFree(p);}};
int main(int argc,char** argv)try{
    require(argc==6,"bundle tokenizer model prefix-fixture activation-fixture required");
    constexpr size_t width=4*2560,plane=80*width,record_bytes=16+2*plane*4+80+2560*4;
    std::ifstream pf(argv[4],std::ios::binary);std::array<char,8> magic{};uint32_t count=0;
    pf.read(magic.data(),8);pf.read(reinterpret_cast<char*>(&count),4);
    require(pf.good() && !std::memcmp(magic.data(),"IGCLPX01",8) && count>1 && count<=262144
        && std::filesystem::file_size(argv[4])==12+uint64_t(count)*8,"canonical_prefix_fixture");
    std::vector<int64_t> prefix(count);pf.read(reinterpret_cast<char*>(prefix.data()),prefix.size()*8);require(pf.good(),"prefix_read");
    std::ifstream af(argv[5],std::ios::binary);std::array<uint8_t,76> header{};af.read(reinterpret_cast<char*>(header.data()),header.size());
    uint32_t rows=0;std::memcpy(&rows,header.data()+8,4);
    require(af.good() && !std::memcmp(header.data(),"IGCLPF01",8) && rows==1
        && std::filesystem::file_size(argv[5])==76+record_bytes,"one_actual_activation_record");
    uint64_t owner=0,anchor=0;af.read(reinterpret_cast<char*>(&owner),8);af.read(reinterpret_cast<char*>(&anchor),8);
    require(owner && anchor+1==prefix.size() && anchor<=80,"matching_short_prefix_anchor");
    std::vector<float> states(2*plane),base(2560),actual(base.size());std::array<uint8_t,80> mask{};
    af.read(reinterpret_cast<char*>(states.data()),states.size()*4);af.read(reinterpret_cast<char*>(mask.data()),80);af.read(reinterpret_cast<char*>(base.data()),base.size()*4);require(af.good(),"activation_read");
    for(size_t j=0;j<80;++j)require(mask[j]==(j>=80-anchor),"canonical_feature_window");
    Stream stream;std::string error;IgcLearnedDiagnostic diagnostic;auto hook=diagnostic.hook();
    require(diagnostic.load_host(argv[1],error),error);
    require(!hook.prepare(hook.user,stream.s,error) && diagnostic.device_bytes()==0,"unbound_prepare_refused");
    std::cout<<"START read-only model/tokenizer content bind; no language-model allocation\n"<<std::flush;
    require(diagnostic.bind(argv[2],{std::filesystem::path(argv[3])},error),error);
    std::cout<<"DONE content bind\n"<<std::flush;
    require(hook.prepare(hook.user,stream.s,error),error);
    Buffer source(2*plane),mixed(2560);check(cudaMemcpyAsync(source.p,states.data(),states.size()*4,cudaMemcpyHostToDevice,stream.s));
    require(diagnostic.begin(prefix,0,false,true,error),error);
    auto consumer=diagnostic.prefill_consumer();
    auto populate=[&]{
    for(auto tap:{strata::core::EarlyActivationTap::Embedding,strata::core::EarlyActivationTap::FirstAttention}){
        const auto offset=tap==strata::core::EarlyActivationTap::Embedding?0:plane;
        const strata::core::EarlyActivationRows value{tap,source.p+offset+(80-anchor)*width,static_cast<int64_t>(anchor),width,0,stream.s};
        require(consumer.consume(consumer.user,value,error),error);
    }
    require(diagnostic.accept_prefill(0,std::span<const int64_t>(prefix.data(),anchor),error),error);
    };
    populate();
    require(diagnostic.stage(error),error);
    cudaGraph_t graph=nullptr;cudaGraphExec_t instance=nullptr;
    auto run=[&](bool captured=false){check(cudaMemcpyAsync(mixed.p,base.data(),base.size()*4,cudaMemcpyHostToDevice,stream.s));
        if(captured)check(cudaGraphLaunch(instance,stream.s));else require(hook.run(hook.user,mixed.p,1,stream.s,error),error);};
    uint32_t last_status=~uint32_t{0};
    auto read=[&](Invocation& invocation){require(diagnostic.audit(invocation,last_status,error),error);
        check(cudaMemcpyAsync(actual.data(),mixed.p,actual.size()*4,cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));};
    run();Invocation result;read(result);
    require(result.evaluations==1 && result.result.status==Status::Success && result.result.value.length==3
        && result.result.value.negative==1 && result.result.value.digits[0]==4 && result.result.value.digits[1]==4
        && result.result.value.digits[2]==7,"wrapper_native_literal_minus744");
    require(std::memcmp(actual.data(),base.data(),base.size()*4)!=0,"wrapper_applies_output_mapper");
    const auto expected=actual;uint64_t previous=result.id;
    check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));require(hook.run(hook.user,mixed.p,1,stream.s,error),error);
    check(cudaStreamEndCapture(stream.s,&graph));check(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));
    check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));
    const bool canceled_in_capture=diagnostic.cancel(error);
    cudaGraph_t rejected_capture=nullptr;check(cudaStreamEndCapture(stream.s,&rejected_capture));check(cudaGraphDestroy(rejected_capture));
    require(!canceled_in_capture,"cancel_during_capture_must_be_refused");
    // Do not launch another head after failed revocation. Retry only after
    // ending this known capture refusal, before resuming any model execution.
    require(diagnostic.cancel(error),error);read(result);
    require(last_status!=0,"cancel_must_revoke_success_audit_before_another_head");
    run(true);read(result);
    require(result.evaluations==0 && result.result.status!=Status::Success && result.result.value.length==0
        && !std::memcmp(actual.data(),base.data(),base.size()*4),"cancel_must_clear_retained_result_before_next_head");
    require(diagnostic.cancel(error),"idempotent_cancel");
    require(!diagnostic.stage(error),"cancel_closes_stage_until_new_begin");
    auto success=[&]{read(result);require(result.evaluations==1 && result.result.status==Status::Success && result.id>previous
        && !std::memcmp(actual.data(),expected.data(),actual.size()*4),"new_owned_request_matches_native_result");previous=result.id;};
    auto bypass=[&]{read(result);require(result.evaluations==0 && result.result.status!=Status::Success && result.result.value.length==0
        && !std::memcmp(actual.data(),base.data(),actual.size()*4),"unavailable_request_cannot_inject_an_old_answer");};
    require(diagnostic.begin(prefix,0,false,true,error),error);
    require(diagnostic.cancel(error),"cancel_before_first_stage");
    require(!diagnostic.stage(error),"prestage_cancel_closes_staging");run(true);bypass();
    // Fresh real rows followed by a genuine retained-prefix continuation.
    require(diagnostic.begin(prefix,0,false,true,error),error);populate();require(diagnostic.stage(error),error);run(true);success();
    require(diagnostic.begin(prefix,anchor,true,true,error),error);require(diagnostic.stage(error),error);run(true);success();
    // Identical token count is insufficient when the engine denies lineage.
    require(diagnostic.begin(prefix,anchor,false,true,error),error);require(diagnostic.stage(error),error);run(true);bypass();
    require(diagnostic.begin(prefix,0,false,true,error),error);populate();require(diagnostic.stage(error),error);run(true);success();
    auto changed=prefix;changed[0]=(changed[0]+1)%248320;
    require(diagnostic.begin(changed,anchor,true,true,error),error);require(diagnostic.stage(error),error);run(true);bypass();
    require(diagnostic.begin(prefix,0,false,true,error),error);populate();require(diagnostic.stage(error),error);run(true);success();
    require(diagnostic.begin(prefix,0,false,false,error),error);require(diagnostic.stage(error),error);run(true);bypass();
    check(cudaGraphExecDestroy(instance));check(cudaGraphDestroy(graph));
    hook.release(hook.user);require(diagnostic.device_bytes()==0,"release_owned_device_buffers");
    hook.release(hook.user);require(diagnostic.device_bytes()==0,"double_release");
    require(hook.prepare(hook.user,stream.s,error),error);
    require(!diagnostic.stage(error),"stage_requires_new_begin_after_reprepare");
    require(diagnostic.begin(prefix,0,false,true,error),error);populate();require(diagnostic.stage(error),error);run();success();
    hook.release(hook.user);
    std::cout<<"PASS actual learned wrapper native result/cancel/audit/cache/lineage/graph/reprepare; no language-model inference\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
