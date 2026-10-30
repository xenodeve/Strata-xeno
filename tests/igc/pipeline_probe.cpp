#include "strata/core/igc_pipeline.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <array>
#include <fstream>
#include <filesystem>
#include <algorithm>
static void require(bool v,const std::string& e){if(!v)throw std::runtime_error(e);}
static void check(cudaError_t c){require(c==cudaSuccess,cudaGetErrorString(c));}
struct Stream{cudaStream_t s=nullptr;Stream(){check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){if(s){cudaStreamSynchronize(s);cudaStreamDestroy(s);}}};
struct Buffer{float* p=nullptr;explicit Buffer(size_t n){check(cudaMalloc(&p,n*4));}~Buffer(){if(p)cudaFree(p);}};
static void replay(strata::igc::LatentPipeline& pipeline,const strata::igc::ControllerBundle& bundle,cudaStream_t stream,
                    const std::filesystem::path& input,const std::filesystem::path& output){
    constexpr size_t plane=80ull*4*2560,record_bytes=16+2*plane*4+80+2560*4;
    require(std::filesystem::is_regular_file(std::filesystem::symlink_status(input)),"pipeline_input_file");
    std::ifstream file(input,std::ios::binary);std::array<uint8_t,76> header{};file.read(reinterpret_cast<char*>(header.data()),header.size());
    uint32_t count=0;std::memcpy(&count,header.data()+8,4);
    require(file.good() && !std::memcmp(header.data(),"IGCLPF01",8) && count>0 && count<=64
        && std::filesystem::file_size(input)==76+uint64_t(count)*record_bytes
        && !std::memcmp(header.data()+12,bundle.model.data(),32) && !std::memcmp(header.data()+44,bundle.tokenizer.data(),32),"pipeline_input_header");
    require(!std::filesystem::exists(output),"pipeline_output_exists");
    std::ofstream saved(output,std::ios::binary);saved.write("IGCLPO01",8);saved.write(reinterpret_cast<char*>(&count),4);
    Buffer device_states(2*plane),device_mask(20),device_mixed(64*2560);
    std::vector<float> states(2*plane),mixed(64*2560),after(64*2560);std::array<uint8_t,80> mask{};
    std::string error;cudaGraph_t graph=nullptr;cudaGraphExec_t instance=nullptr;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));require(pipeline.head(device_mixed.p,1,stream,true,error),error);
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));
    uint32_t selected=0;bool broadcast_checked=false;
    for(uint32_t row=0;row<count;++row){
        uint64_t id=0,anchor=0;file.read(reinterpret_cast<char*>(&id),8);file.read(reinterpret_cast<char*>(&anchor),8);
        file.read(reinterpret_cast<char*>(states.data()),states.size()*4);file.read(reinterpret_cast<char*>(mask.data()),80);file.read(reinterpret_cast<char*>(mixed.data()),2560*4);
        require(file.good() && id && anchor<262144 && std::all_of(mask.begin(),mask.end(),[](uint8_t v){return v<=1;}),"pipeline_input_record");
        const auto kept=std::min<uint64_t>(80,anchor);bool available=kept>0;
        for(size_t j=0;j<80;++j)available&=mask[j]==(j>=80-kept);
        for(int j=1;j<64;++j)std::copy_n(mixed.data(),2560,mixed.data()+j*2560);
        check(cudaMemcpyAsync(device_states.p,states.data(),states.size()*4,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(device_mask.p,mask.data(),80,cudaMemcpyHostToDevice,stream));
        strata::igc::Invocation first{};uint32_t first_status=0;std::array<float,2560> first_output{};
        for(int mode=0;mode<2;++mode){
            require(pipeline.stage_inputs(device_states.p,device_states.p+plane,reinterpret_cast<uint8_t*>(device_mask.p),id,anchor,available,stream,true,error),error);
            for(int repeat=0;repeat<2;++repeat){
                check(cudaMemcpyAsync(device_mixed.p,mixed.data(),mixed.size()*4,cudaMemcpyHostToDevice,stream));
                if(mode)check(cudaGraphLaunch(instance,stream));else require(pipeline.head(device_mixed.p,1,stream,true,error),error);
                strata::igc::Invocation state;uint32_t status=0;require(pipeline.audit(state,status,stream,error),error);
                check(cudaMemcpyAsync(after.data(),device_mixed.p,2560*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
                if(mode==0 && repeat==0){first=state;first_status=status;std::copy_n(after.data(),2560,first_output.data());}
                else require(!std::memcmp(&state,&first,sizeof(first)) && status==first_status && !std::memcmp(after.data(),first_output.data(),2560*4),"pipeline_normal_graph_repeat_mismatch");
            }
        }
        if(!broadcast_checked && first.result.status==strata::igc::Status::Success){
            cudaGraph_t many=nullptr;cudaGraphExec_t many_exec=nullptr;
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));require(pipeline.head(device_mixed.p,64,stream,true,error),error);
            check(cudaStreamEndCapture(stream,&many));check(cudaGraphInstantiate(&many_exec,many,nullptr,nullptr,0));
            check(cudaMemcpyAsync(device_mixed.p,mixed.data(),mixed.size()*4,cudaMemcpyHostToDevice,stream));check(cudaGraphLaunch(many_exec,stream));
            strata::igc::Invocation state;uint32_t status=0;require(pipeline.audit(state,status,stream,error),error);
            require(!std::memcmp(&state,&first,sizeof(first)) && status==first_status,"pipeline_broadcast_reexecuted");
            check(cudaMemcpyAsync(after.data(),device_mixed.p,after.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
            for(int j=0;j<64;++j)require(!std::memcmp(after.data()+j*2560,first_output.data(),2560*4),"pipeline_broadcast_output_rows");
            check(cudaGraphExecDestroy(many_exec));check(cudaGraphDestroy(many));broadcast_checked=true;
        }
        selected+=first.evaluations;
        // After end_output the head stays unchanged, the audit keeps the answer and nothing executes again.
        if(first.result.status==strata::igc::Status::Success){
            require(pipeline.end_output(stream,error),error);
            check(cudaMemcpyAsync(device_mixed.p,mixed.data(),mixed.size()*4,cudaMemcpyHostToDevice,stream));check(cudaGraphLaunch(instance,stream));
            strata::igc::Invocation ended;uint32_t ended_status=0;require(pipeline.audit(ended,ended_status,stream,error),error);
            check(cudaMemcpyAsync(after.data(),device_mixed.p,2560*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
            require(!std::memcmp(&ended,&first,sizeof(first)) && ended_status==first_status && !std::memcmp(after.data(),mixed.data(),2560*4),"pipeline_end_output");
        }
        saved.write(reinterpret_cast<const char*>(&first),sizeof(first));saved.write(reinterpret_cast<const char*>(&first_status),4);saved.write(reinterpret_cast<const char*>(first_output.data()),2560*4);
    }
    require(file.peek()==std::char_traits<char>::eof() && saved.good(),"pipeline_record_completion");
    check(cudaGraphExecDestroy(instance));check(cudaGraphDestroy(graph));
    require(broadcast_checked && selected>0,"pipeline_replay_missing_positive_coverage");
    // Clear a real successful history using unavailable context; no stale answer.
    require(pipeline.stage_inputs(nullptr,nullptr,nullptr,999,7,false,stream,true,error),error);
    check(cudaMemcpyAsync(device_mixed.p,mixed.data(),mixed.size()*4,cudaMemcpyHostToDevice,stream));require(pipeline.head(device_mixed.p,64,stream,true,error),error);
    strata::igc::Invocation cleared;uint32_t status=0;require(pipeline.audit(cleared,status,stream,error),error);
    check(cudaMemcpyAsync(after.data(),device_mixed.p,after.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
    require(cleared.attempts==0 && cleared.evaluations==0 && cleared.result.status!=strata::igc::Status::Success && cleared.result.value.length==0
        && !std::memcmp(after.data(),mixed.data(),after.size()*4),"pipeline_restage_unavailable_stale_answer");
    // Even if a faulty producer labels an empty mask available, NN status must
    // propagate through selection to a valueless result and unchanged head.
    check(cudaMemsetAsync(device_mask.p,0,80,stream));
    require(pipeline.stage_inputs(device_states.p,device_states.p+plane,reinterpret_cast<uint8_t*>(device_mask.p),1000,7,true,stream,true,error),error);
    check(cudaMemcpyAsync(device_mixed.p,mixed.data(),mixed.size()*4,cudaMemcpyHostToDevice,stream));require(pipeline.head(device_mixed.p,64,stream,true,error),error);
    require(pipeline.audit(cleared,status,stream,error),error);
    check(cudaMemcpyAsync(after.data(),device_mixed.p,after.size()*4,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));
    require(cleared.attempts==1 && cleared.evaluations==0 && cleared.result.status!=strata::igc::Status::Success && cleared.result.value.length==0
        && !std::memcmp(after.data(),mixed.data(),after.size()*4),"pipeline_bad_mask_stale_answer");
    std::cout<<"PASS coherent activation replay records="<<count<<" selected="<<selected<<" normal_graph_repeat=bitwise broadcast64=bitwise stale_reset=pass\n";
}
int main(int argc,char** argv)try{
    require(argc==2 || argc==4,"bundle [coherent-input output] required");std::string error;
    Stream stream;
    strata::igc::LatentPipeline pipeline;
    require(pipeline.head(nullptr,0,nullptr,false,error),"pipeline_hard_off_head");
    require(pipeline.stage_inputs(nullptr,nullptr,nullptr,0,0,false,nullptr,false,error),"pipeline_hard_off_stage");
    strata::igc::ControllerBundle bundle;require(strata::igc::load_controller_bundle(argv[1],true,bundle,error),error);
    const bool initialized=pipeline.initialize(bundle,stream.s,error);
    require(initialized,"pipeline_initialize_contract:"+error);
    require(pipeline.device_bytes()>0 && pipeline.device_bytes()<32*1024*1024,"pipeline_bounded_storage");
    require(!pipeline.initialize(bundle,stream.s,error),"pipeline_reinit_refused");
    Buffer mixed(64*2560);std::vector<float> original(64*2560),actual(original.size());
    for(size_t i=0;i<original.size();++i)original[i]=float(int(i%17)-8)*.03f;
    for(uint32_t count:{1u,64u}){
        require(pipeline.stage_inputs(nullptr,nullptr,nullptr,9,31,false,stream.s,true,error),error);
        check(cudaMemcpyAsync(mixed.p,original.data(),original.size()*4,cudaMemcpyHostToDevice,stream.s));check(cudaStreamSynchronize(stream.s));
        cudaGraph_t graph=nullptr;cudaGraphExec_t instance=nullptr;
        check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));
        require(pipeline.head(mixed.p,count,stream.s,true,error),error);
        check(cudaStreamEndCapture(stream.s,&graph));check(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));
        for(int repeat=0;repeat<3;++repeat){check(cudaGraphLaunch(instance,stream.s));
            strata::igc::Invocation state;uint32_t status=0;require(pipeline.audit(state,status,stream.s,error),error);
            require(state.attempts==0 && state.evaluations==0 && state.result.status!=strata::igc::Status::Success && state.result.value.length==0,"pipeline_missing_no_answer");
            check(cudaMemcpyAsync(actual.data(),mixed.p,original.size()*4,cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
            require(!std::memcmp(actual.data(),original.data(),original.size()*4),"pipeline_missing_preserves_head_bytes");
        }
        check(cudaGraphExecDestroy(instance));check(cudaGraphDestroy(graph));
    }
    std::cout<<"PASS pipeline missing-context graph1/64 hard-off bounded requested_bytes="<<pipeline.device_bytes()<<'\n';
    if(argc==4)replay(pipeline,bundle,stream.s,argv[2],argv[3]);
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
