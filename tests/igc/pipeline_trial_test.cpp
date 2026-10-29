#include "strata/core/igc_pipeline.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <array>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <vector>
using namespace strata::igc;
static void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
static void check(cudaError_t c){require(c==cudaSuccess,cudaGetErrorString(c));}
struct Stream{cudaStream_t s=nullptr;Stream(){check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){if(s){cudaStreamSynchronize(s);cudaStreamDestroy(s);}}};
struct Buffer{float* p=nullptr;explicit Buffer(size_t count){check(cudaMalloc(&p,count*4));}~Buffer(){if(p)cudaFree(p);}};
static void executed_trials(LatentPipeline& pipeline,const ControllerBundle& bundle,cudaStream_t stream,const std::filesystem::path& path,uint32_t rows){
    constexpr size_t plane=80ull*4*2560,record_bytes=16+2*plane*4+80+2560*4;
    std::ifstream file(path,std::ios::binary);std::array<uint8_t,76> header{};file.read(reinterpret_cast<char*>(header.data()),header.size());
    uint32_t count=0;std::memcpy(&count,header.data()+8,4);
    require(file.good() && !std::memcmp(header.data(),"IGCLPF01",8) && count==3
        && std::filesystem::file_size(path)==76+count*record_bytes
        && !std::memcmp(header.data()+12,bundle.model.data(),32) && !std::memcmp(header.data()+44,bundle.tokenizer.data(),32),"three_frozen_diagnostic_records_required");
    Buffer states(2*plane),mask(20),mixed(64*2560);Stream foreign;
    std::vector<float> host_states(2*plane),base(64*2560),actual(base.size()),expected(base.size());std::array<uint8_t,80> valid{};
    cudaGraph_t graph=nullptr;cudaGraphExec_t instance=nullptr;std::string error;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    require(pipeline.head(mixed.p,rows,stream,true,error),error);
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));
    const auto bounded_bytes=pipeline.device_bytes();
    for(uint32_t row=0;row<count;++row){
        uint64_t id=0,anchor=0;file.read(reinterpret_cast<char*>(&id),8);file.read(reinterpret_cast<char*>(&anchor),8);
        file.read(reinterpret_cast<char*>(host_states.data()),host_states.size()*4);file.read(reinterpret_cast<char*>(valid.data()),80);file.read(reinterpret_cast<char*>(base.data()),2560*4);
        require(file.good() && id && anchor<262144,"trial_fixture_record");
        for(int j=1;j<64;++j)std::copy_n(base.data(),2560,base.data()+j*2560);
        check(cudaMemcpyAsync(states.p,host_states.data(),host_states.size()*4,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(mask.p,valid.data(),80,cudaMemcpyHostToDevice,stream));
        auto stage=[&](uint64_t owner,bool available){require(pipeline.stage_inputs(available?states.p:nullptr,available?states.p+plane:nullptr,
            available?reinterpret_cast<uint8_t*>(mask.p):nullptr,owner,anchor,available,stream,true,error),error);};
        auto head=[&](bool captured){check(cudaMemcpyAsync(mixed.p,base.data(),base.size()*4,cudaMemcpyHostToDevice,stream));
            if(captured)check(cudaGraphLaunch(instance,stream));else require(pipeline.head(mixed.p,rows,stream,true,error),error);};
        auto read=[&](Invocation& value,uint32_t& status){require(pipeline.audit(value,status,stream,error),error);
            check(cudaMemcpyAsync(actual.data(),mixed.p,actual.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));};
        stage(id,true);head(false);Invocation reference;uint32_t reference_status=0;read(reference,reference_status);expected=actual;
        require(reference.attempts==1 && reference.selection_status==(row==0?4u:row==1?0u:1u),"frozen_selection_status");
        if(row==1){const auto& value=reference.result.value;
            require(reference.evaluations==1 && reference.result.status==Status::Success && value.length==3 && value.negative==1
                && value.digits[0]==4 && value.digits[1]==4 && value.digits[2]==7,"literal_exact_minus744");}
        else require(reference.evaluations==0 && reference.result.status!=Status::Success && reference.result.value.length==0,"failure_is_valueless");
        stage(id,true);Invocation before;uint32_t before_status=0;read(before,before_status);
        require(before.armed==1 && before.attempts==0,"base_is_pending");
        require(pipeline.begin_trial(stream,true,error),error);
        require(!pipeline.begin_trial(stream,true,error),"nested_trial_refused");
        require(!pipeline.finish_trial(false,foreign.s,true,error),"foreign_finish_cannot_discard_trial");
        require(pipeline.finish_trial(false,nullptr,false,error),"hard_off_does_not_finish_trial");
        require(!pipeline.begin_trial(stream,true,error),"hard_off_preserves_active_trial");
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        require(!pipeline.finish_trial(true,stream,true,error),"finish_inside_capture_refused");
        cudaGraph_t empty=nullptr;check(cudaStreamEndCapture(stream,&empty));check(cudaGraphDestroy(empty));
        head(true);Invocation candidate;uint32_t candidate_status=0;read(candidate,candidate_status);
        require(!std::memcmp(&candidate,&reference,sizeof(reference)) && candidate_status==reference_status
            && !std::memcmp(actual.data(),expected.data(),actual.size()*4),"trial_executes_same_real_pipeline");
        require(pipeline.finish_trial(false,stream,true,error),error);
        Invocation restored;uint32_t restored_status=0;read(restored,restored_status);
        require(!std::memcmp(&restored,&before,sizeof(before)) && restored_status==before_status,"abort_restores_pending_state_and_head_status");
        require(!pipeline.finish_trial(false,stream,true,error),"double_finish_refused");
        // Rejected language rows are discarded by their caller. Recreate the
        // original mixed input before replay; module rollback cannot undo it.
        require(pipeline.begin_trial(stream,true,error),error);head(false);read(candidate,candidate_status);
        require(!std::memcmp(&candidate,&reference,sizeof(reference)) && !std::memcmp(actual.data(),expected.data(),actual.size()*4),"accepted_replay_from_base");
        require(pipeline.finish_trial(true,stream,true,error),error);
        for(int repeat=0;repeat<3;++repeat){head(true);read(candidate,candidate_status);
            require(!std::memcmp(&candidate,&reference,sizeof(reference)) && !std::memcmp(actual.data(),expected.data(),actual.size()*4),"commit_and_graph_repeats_do_not_execute_twice");}
        // A canceled/changed request gets a new owner and unavailable context;
        // old success cannot survive the existing request-entry reset.
        stage(id+1000,false);head(true);read(candidate,candidate_status);
        require(candidate.attempts==0 && candidate.evaluations==0 && candidate.result.status!=Status::Success
            && candidate.result.value.length==0 && !std::memcmp(actual.data(),base.data(),base.size()*4),"request_reset_has_no_stale_answer");
        stage(id+1000,true);head(true);read(candidate,candidate_status);
        require(candidate.id==id+1000 && candidate.result.invocation==id+1000 && candidate.evaluations==reference.evaluations,"next_owner_reexecutes_its_own_attempt");
        require(pipeline.device_bytes()==bounded_bytes,"checkpoint_storage_reused");
    }
    require(file.peek()==std::char_traits<char>::eof(),"trial_fixture_exhausted");
    check(cudaGraphExecDestroy(instance));check(cudaGraphDestroy(graph));
    std::cout<<"PASS three actual-activation trials statuses4/0/1, abort/replay/commit graph_rows="<<rows<<", reset/new owner, stable storage\n";
}
int main(int argc,char** argv)try{
    require(argc==2 || argc==3 || (argc==4 && !std::strcmp(argv[3],"--single-row")),"bundle [three-record fixture [--single-row]] required");
    Stream stream;std::string error;ControllerBundle bundle;LatentPipeline pipeline;
    require(pipeline.begin_trial(nullptr,false,error) && pipeline.finish_trial(false,nullptr,false,error),"uninitialized_hard_off_noop");
    require(!pipeline.begin_trial(stream.s,true,error) && !pipeline.finish_trial(true,stream.s,true,error),"uninitialized_trial_refused");
    require(load_controller_bundle(argv[1],true,bundle,error),error);
    require(pipeline.initialize(bundle,stream.s,error),error);
    require(pipeline.stage_inputs(nullptr,nullptr,nullptr,7,31,false,stream.s,true,error),error);
    Invocation before;uint32_t status=0;require(pipeline.audit(before,status,stream.s,error),error);
    const auto original_bytes=pipeline.device_bytes();
    const bool started=pipeline.begin_trial(stream.s,true,error);
    require(started,"trial_begin:"+error);
    require(pipeline.device_bytes()>original_bytes && pipeline.device_bytes()-original_bytes<65536,"bounded_lazy_checkpoint");
    require(!pipeline.stage_inputs(nullptr,nullptr,nullptr,8,32,false,stream.s,true,error),"new_request_during_trial_must_be_refused");
    require(pipeline.finish_trial(false,stream.s,true,error),error);
    Invocation after;uint32_t after_status=0;require(pipeline.audit(after,after_status,stream.s,error),error);
    require(!std::memcmp(&before,&after,sizeof(before)) && status==after_status,"aborted_trial_restores_base");
    std::cout<<"PASS pipeline trial base checkpoint/abort requested_extra_bytes="<<pipeline.device_bytes()-original_bytes<<'\n';
    if(argc>=3)executed_trials(pipeline,bundle,stream.s,argv[2],argc==4?1u:64u);
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
