#include "strata/core/igc_invocation.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata::igc;
static void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
static void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
struct Buffer {void* p=nullptr;explicit Buffer(size_t n){check(cudaMalloc(&p,n));}~Buffer(){if(p)cudaFree(p);}template<class T>T* as(){return static_cast<T*>(p);}};
struct Stream {cudaStream_t s=nullptr;Stream(){check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){if(s){cudaStreamSynchronize(s);cudaStreamDestroy(s);}}};
// Independent fixed test oracle; never linked to production execution.
static Integer integer(int v){Integer r;r.negative=v<0;unsigned a=v<0?-v:v;do{r.digits[r.length++]=uint8_t(a%10);a/=10;}while(a);return r;}
int main()try{
    std::string error;
    require(arm_invocations(nullptr,nullptr,nullptr,0,nullptr,false,error),"hard_off_arm");
    require(execute_invocations(nullptr,nullptr,nullptr,0,nullptr,false,error),"hard_off_execute");
    require(broadcast_invocation(nullptr,nullptr,0,nullptr,false,error),"hard_off_broadcast");
    constexpr int count=64;Stream stream;
    Buffer state(count*sizeof(Invocation)),request(count*sizeof(ExactRequest)),selection(count*4),ids(count*8),positions(count*8),published(count*sizeof(ExactResult));
    std::array<uint64_t,count> id{},pos{};std::array<uint32_t,count> code{};
    std::array<ExactRequest,count> q{};std::array<Invocation,count> out{},first{};
    for(int i=0;i<count;++i){
        id[i]=100+i;pos[i]=80+i;q[i].invocation=id[i];q[i].position=pos[i];
        q[i].operation=Operation::Multiply;q[i].lhs=integer(-12);q[i].rhs=integer(64);
        switch(i%8){case 1:code[i]=1;break;case 2:code[i]=4;break;case 3:++q[i].invocation;break;
            case 4:++q[i].position;break;case 5:q[i].lhs.negative=2;break;case 6:id[i]=0;break;
            case 7:q[i].lhs=integer(0);break;}
    }
    auto upload=[&](Buffer& b,const auto& values){check(cudaMemcpyAsync(b.p,values.data(),sizeof(values),cudaMemcpyHostToDevice,stream.s));};
    auto read=[&]{check(cudaMemcpyAsync(out.data(),state.p,sizeof(out),cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));};
    upload(ids,id);upload(positions,pos);upload(request,q);upload(selection,code);
    require(!arm_invocations(state.as<Invocation>(),ids.as<uint64_t>(),positions.as<uint64_t>(),65,stream.s,true,error),"count_refused");
    require(!execute_invocations(state.as<Invocation>(),nullptr,selection.as<uint32_t>(),count,stream.s,true,error),"null_refused");
    require(!arm_invocations(state.as<Invocation>(),ids.as<uint64_t>(),positions.as<uint64_t>(),count,cudaStreamLegacy,true,error),"default_stream_refused");
    auto arm=[&]{require(arm_invocations(state.as<Invocation>(),ids.as<uint64_t>(),positions.as<uint64_t>(),count,stream.s,true,error),error.c_str());};
    auto execute=[&]{require(execute_invocations(state.as<Invocation>(),request.as<ExactRequest>(),selection.as<uint32_t>(),count,stream.s,true,error),error.c_str());};
    auto verify=[&]{
        for(int i=0;i<count;++i){const auto& s=out[i];require(s.id==id[i] && s.position==pos[i],"retained_owner");
            require(s.armed==0 && s.attempts==(i%8==6?0u:1u),"attempt_once");
            if(i%8!=6){require(!std::memcmp(&s.request,&q[i],sizeof(q[i])),"retained_attempted_request");require(s.selection_status==code[i],"retained_selection_reason");}
            const bool success=i%8==0 || i%8==7,called=success || i%8==5;
            require(s.evaluations==unsigned(called),"actual_evaluations");
            auto terminal=success?InvocationStatus::Applied:i%8==1?InvocationStatus::NoCall:i%8==3||i%8==4?InvocationStatus::OwnerMismatch:InvocationStatus::Rejected;
            require(s.terminal==terminal,"terminal_reason");
            require(s.result.invocation==id[i] && s.result.position==pos[i],"result_owner");
            if(success){auto value=integer(i%8==7?0:-768);require(s.result.status==Status::Success && !std::memcmp(&value,&s.result.value,sizeof(value)),"exact_value");}
            else{Integer empty;require(s.result.status!=Status::Success && !std::memcmp(&empty,&s.result.value,sizeof(empty)),"no_false_value");}
        }
    };
    for(int graph_mode=0;graph_mode<2;++graph_mode){
        arm();execute();read();verify();first=out;
        cudaGraph_t graph=nullptr;cudaGraphExec_t instance=nullptr;
        if(graph_mode){check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));execute();check(cudaStreamEndCapture(stream.s,&graph));check(cudaGraphInstantiate(&instance,graph,nullptr,nullptr,0));}
        // Change every request into a different, otherwise valid calculation.
        auto changed=q;for(auto& r:changed){r.lhs=integer(99);r.rhs=integer(99);}upload(request,changed);
        for(int repeat=0;repeat<5;++repeat){if(instance)check(cudaGraphLaunch(instance,stream.s));else execute();}
        read();require(!std::memcmp(out.data(),first.data(),sizeof(out)),"repeat_preserves_terminal_result");
        // A new arm must clear old successes before any new calculation.
        arm();read();for(int i=0;i<count;++i){Integer empty;require(out[i].attempts==0 && out[i].evaluations==0 && out[i].result.status!=Status::Success && !std::memcmp(&out[i].result.value,&empty,sizeof(empty)),"rearm_clears_answer");}
        upload(request,q);if(instance)check(cudaGraphLaunch(instance,stream.s));else execute();read();verify();
        // Publishing repeats a retained result, not the arithmetic evaluation.
        require(broadcast_invocation(state.as<Invocation>(),published.as<ExactResult>(),count,stream.s,true,error),error.c_str());
        std::array<ExactResult,count> copies{};check(cudaMemcpyAsync(copies.data(),published.p,sizeof(copies),cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
        for(const auto& r:copies)require(!std::memcmp(&r,&out[0].result,sizeof(r)),"broadcast_retained_result");
        read();verify();
        // Hard off with valid memory must leave it byte-identical too.
        require(arm_invocations(state.as<Invocation>(),ids.as<uint64_t>(),positions.as<uint64_t>(),count,stream.s,false,error),"off_arm");
        require(execute_invocations(state.as<Invocation>(),request.as<ExactRequest>(),selection.as<uint32_t>(),count,stream.s,false,error),"off_execute");
        read();require(!std::memcmp(out.data(),first.data(),sizeof(out)),"hard_off_bytes");
        if(instance)check(cudaGraphExecDestroy(instance));if(graph)check(cudaGraphDestroy(graph));
    }
    // First accepted-window state can be trialled without mutating the base.
    Buffer shadow(sizeof(out));check(cudaMemcpyAsync(shadow.p,state.p,sizeof(out),cudaMemcpyDeviceToDevice,stream.s));
    require(arm_invocations(shadow.as<Invocation>(),ids.as<uint64_t>(),positions.as<uint64_t>(),count,stream.s,true,error),"shadow_arm");
    require(execute_invocations(shadow.as<Invocation>(),request.as<ExactRequest>(),selection.as<uint32_t>(),count,stream.s,true,error),"shadow_execute");
    read();require(!std::memcmp(out.data(),first.data(),sizeof(out)),"shadow_isolation");
    require(broadcast_invocation(state.as<Invocation>()+1,published.as<ExactResult>(),count,stream.s,true,error),"failure_broadcast");
    std::array<ExactResult,count> failures{};check(cudaMemcpyAsync(failures.data(),published.p,sizeof(failures),cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
    for(const auto& r:failures){Integer empty;require(r.status!=Status::Success && !std::memcmp(&r.value,&empty,sizeof(empty)),"broadcast_no_stale_success");}
    require(arm_invocations(state.as<Invocation>(),ids.as<uint64_t>(),positions.as<uint64_t>(),1,stream.s,true,error),"single_arm");
    require(execute_invocations(state.as<Invocation>(),request.as<ExactRequest>(),selection.as<uint32_t>(),1,stream.s,true,error),"single_execute");
    require(broadcast_invocation(state.as<Invocation>(),published.as<ExactResult>(),1,stream.s,true,error),"single_broadcast");
    read();verify();
    std::cout<<"PASS owned invocation1/64 normal/graph repeated=once failure/reset/hard-off/shadow/broadcast\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
