#include "strata/core/qwenputer_runner.hpp"
#include "strata/timeline.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>
using namespace strata::qwenputer;
static void ck(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
template<class T> static std::vector<T> read(const char* path) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file)throw std::runtime_error("fixture_io");
    const auto size=std::size_t(file.tellg());
    if(size%sizeof(T)||size>(1ull<<30))throw std::runtime_error("fixture_size");
    std::vector<T> data(size/sizeof(T));file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(data.data()),std::streamsize(size)))throw std::runtime_error("fixture_io");
    return data;
}
int main(int argc,char** argv)try {
    struct Flush{~Flush(){strata::timeline::flush();}}flush;
    if(argc<6){std::cerr<<"artifact binding expected-inputs final-state chunk-steps [test options]\n";return 2;}
    RuntimeLimits limits;limits.record_trace=true;
    unsigned output_budget=256,repeats=1;RunReason expected_reason=RunReason::Success;
    bool zero_first=false,zero_output=false,cancel_before=false,cancel_during=false,resume=false,bad_prefix=false;
    bool device_checks=false,artifact_checks=false,retain_state=false;unsigned between_ms=0;
    const char* initial_path=nullptr;const char* dump_path=nullptr;const char* repeat_reference=nullptr;
    for(int i=6;i<argc;++i) {
        const std::string option=argv[i];
        if(option=="--zero-first")zero_first=true;
        else if(option=="--zero-output-first")zero_output=true;
        else if(option=="--cancel-before")cancel_before=true;
        else if(option=="--cancel-during")cancel_during=true;
        else if(option=="--resume")resume=true;
        else if(option=="--bad-prefix")bad_prefix=true;
        else if(option=="--device-checks")device_checks=true;
        else if(option=="--artifact-checks")artifact_checks=true;
        else if(option=="--retain-state")retain_state=true;
        else if(i+1<argc&&option=="--timeout-ms")limits.timeout_ms=std::stoul(argv[++i]);
        else if(i+1<argc&&option=="--between-ms")between_ms=std::stoul(argv[++i]);
        else if(i+1<argc&&option=="--max-steps")limits.steps=std::stoul(argv[++i]);
        else if(i+1<argc&&option=="--max-output")limits.output=std::stoul(argv[++i]);
        else if(i+1<argc&&option=="--output-chunk")output_budget=std::stoul(argv[++i]);
        else if(i+1<argc&&option=="--expect")expected_reason=RunReason(std::stoul(argv[++i]));
        else if(i+1<argc&&option=="--repeat")repeats=std::stoul(argv[++i]);
        else if(i+1<argc&&option=="--initial")initial_path=argv[++i];
        else if(i+1<argc&&option=="--dump")dump_path=argv[++i];
        else if(i+1<argc&&option=="--repeat-reference")repeat_reference=argv[++i];
        else throw std::runtime_error("test_option");
    }
    if(repeats<1||repeats>5||output_budget<1||output_budget>4096)throw std::runtime_error("test_bounds");
    const auto bytes=read<std::uint8_t>(argv[2]);if(bytes.size()!=sizeof(Binding))return 2;
    Binding binding;std::memcpy(&binding,bytes.data(),bytes.size());
    Artifact a;MemoryPlan memory;std::string error;
    if(!load_artifact(argv[1],binding,{},a,memory,error))throw std::runtime_error(error);
    auto expected=read<Input>(argv[3]);const auto final=read<float>(argv[4]);
    if(repeat_reference&&(!retain_state||repeats!=2))throw std::runtime_error("repeat_reference_needs_two_retained_runs");
    const auto budget=std::stoul(argv[5]);
    if(expected.empty()||expected.size()>200000||budget<1||budget>256||final.size()!=a.config[6])
        throw std::runtime_error("fixture_geometry");
    std::size_t prefix=0;
    while(prefix<expected.size()&&expected[prefix].kind==TokenKind::Text)++prefix;
    if(!prefix)prefix=1; // explicit GO seed for low-level program tests
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    {
        Executor executor;if(!executor.init(a,stream,error))throw std::runtime_error(error);
        if(artifact_checks) {
            auto other=a;other.digest[0]^=1;
            Runtime wrong;
            if(wrong.init(executor,other,limits,stream,error)||error!="runner_artifact_mismatch"||wrong.allocated_bytes())
                throw std::runtime_error("mismatched_artifact_accepted");
            std::cout<<"mismatched_artifact_refused_before_allocation=1\n";
        }
        if(device_checks) {
            int primary,count;ck(cudaGetDevice(&primary));ck(cudaGetDeviceCount(&count));
            if(count<2)throw std::runtime_error("device_check_needs_two_devices");
            ck(cudaSetDevice((primary+1)%count));
            cudaStream_t other;ck(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking));
            {Runtime wrong;if(wrong.init(executor,a,limits,other,error)||error!="runner_device_mismatch")
                throw std::runtime_error("wrong_device_not_refused");}
            ck(cudaSetDevice(primary));
            {Runtime wrong;if(wrong.init(executor,a,limits,other,error)||error!="runner_stream_device_mismatch")
                throw std::runtime_error("wrong_stream_not_refused");}
            ck(cudaSetDevice((primary+1)%count));ck(cudaStreamDestroy(other));ck(cudaSetDevice(primary));
            std::cout<<"wrong_device_and_stream_refused_before_allocation=1\n";
        }
        Runtime runtime;
        if(!runtime.init(executor,a,limits,stream,error))throw std::runtime_error("runner_init:"+error);
        for(unsigned repeat=0;repeat<repeats;++repeat) {
        if(repeat&&repeat_reference) {
            auto next=read<Input>(repeat_reference);
            if(next.size()<prefix||next.size()>200000)throw std::runtime_error("repeat_reference_geometry");
            for(std::size_t i=0;i<prefix;++i)
                if(next[i].kind!=expected[i].kind||next[i].id!=expected[i].id)
                    throw std::runtime_error("repeat_reference_changed_supplied_prefix");
            expected=std::move(next); // Future tokens remain host-only expected data.
        }
        // Only the language prefix is uploaded. Expected future trace/state stays
        // on the host and is consulted after autonomous execution completes.
        if(resume) {
            if(prefix<2)throw std::runtime_error("resume_needs_language_prefix");
            if(!executor.enqueue_reset(stream,error))throw std::runtime_error(error);
            for(std::size_t i=0;i<prefix;++i) {
                ck(cudaMemcpyAsync(executor.device_input(),&expected[i],sizeof(Input),cudaMemcpyHostToDevice,stream));
                if(!executor.enqueue_step(stream,error))throw std::runtime_error(error);
            }
        }
        if(!runtime.prepare(resume?std::span<const Input>{}:std::span(expected).first(prefix),
                            !resume&&(!retain_state||!repeat),error))throw std::runtime_error(error);
        if(initial_path) {
            auto initial=read<float>(initial_path);if(initial.size()!=a.config[6])throw std::runtime_error("initial_geometry");
            ck(cudaMemcpyAsync(executor.device_state(),initial.data(),initial.size()*sizeof(float),cudaMemcpyHostToDevice,stream));
            ck(cudaStreamSynchronize(stream));
        }
        if(zero_first) {
            ChunkResult zero;
            if(!runtime.enqueue_chunk(0,output_budget,error)||!runtime.collect(zero,error))throw std::runtime_error(error);
            if(zero.reason!=RunReason::Budget||zero.total_steps||!zero.text.empty()||!zero.trace.empty())
                throw std::runtime_error("zero_chunk_moved_state");
        }
        if(zero_output) {
            ChunkResult zero;
            if(!runtime.enqueue_chunk(unsigned(budget),0,error)||!runtime.collect(zero,error))throw std::runtime_error(error);
            if(zero.reason!=RunReason::Output||zero.total_steps||!zero.text.empty()||!zero.trace.empty())
                throw std::runtime_error("zero_output_moved_state");
        }
        if(cancel_before&&!runtime.request_cancel(error))throw std::runtime_error(error);
        std::vector<Input> actual;std::vector<unsigned> text;
        ChunkResult result;unsigned launches=0,max_in_launch=0;
        do {
            if(++launches>200002)throw std::runtime_error("host_chunk_bound");
            if(launches>1&&between_ms)std::this_thread::sleep_for(std::chrono::milliseconds(between_ms));
            if(!runtime.enqueue_chunk(unsigned(budget),output_budget,error))throw std::runtime_error(error);
            if(cancel_during&&launches==1) {
                const auto query=cudaStreamQuery(stream);
                if(query!=cudaSuccess&&query!=cudaErrorNotReady)ck(query);
                cudaGetLastError();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                if(!runtime.request_cancel(error))throw std::runtime_error(error);
            }
            if(!runtime.collect(result,error))throw std::runtime_error(error);
            max_in_launch=std::max(max_in_launch,result.chunk_steps);
            actual.insert(actual.end(),result.trace.begin(),result.trace.end());
            text.insert(text.end(),result.text.begin(),result.text.end());
        }while(result.reason==RunReason::Budget||result.reason==RunReason::Output);
        if(result.reason!=expected_reason)throw std::runtime_error("runner_terminal:"+std::to_string(unsigned(result.reason)));
        if(result.total_steps>limits.steps)throw std::runtime_error("global_step_limit_exceeded");
        const std::size_t offset=resume?prefix:0;
        if(result.reason==RunReason::Success&&actual.size()!=expected.size()-offset)
            throw std::runtime_error("trace_length:"+std::to_string(actual.size()));
        if(result.reason!=RunReason::Success&&!result.text.empty())throw std::runtime_error("failure_published_partial_result");
        if(actual.size()>expected.size()-offset)throw std::runtime_error("trace_overrun");
        std::size_t mismatches=0;
        for(std::size_t i=0;i<actual.size();++i)
            if(actual[i].kind!=expected[i+offset].kind||actual[i].id!=expected[i+offset].id)++mismatches;
        std::vector<float> state(final.size());
        ck(cudaMemcpy(state.data(),executor.device_state(),state.size()*sizeof(float),cudaMemcpyDeviceToHost));
        if(result.reason==RunReason::Success) {
            for(std::size_t i=0;i<state.size();++i)if(state[i]!=final[i])++mismatches;
            std::vector<unsigned> expected_text;
            for(std::size_t i=prefix;i<expected.size();++i)
                if(expected[i].kind==TokenKind::Text)expected_text.push_back(expected[i].id);
            if(text!=expected_text)throw std::runtime_error("fifo_text_mismatch");
        }
        if(cancel_before&&result.total_steps)throw std::runtime_error("cancel_before_executed");
        if(cancel_during&&!result.total_steps)throw std::runtime_error("cancel_not_in_flight");
        if((cancel_before||cancel_during)&&result.total_steps>=expected.size())throw std::runtime_error("late_cancel");
        std::cout<<"free_running_steps="<<actual.size()<<" supplied_prefix="<<prefix<<" launches="<<launches
                 <<" max_steps_per_launch="<<max_in_launch<<" mismatches="<<mismatches
                 <<" runtime_buffers="<<runtime.allocated_bytes()<<" graph_code_allocation_delta="<<runtime.graph_allocation_delta()
                 <<" body_nodes="<<runtime.body_nodes()<<" terminal="<<unsigned(result.reason)<<" error="<<result.error<<'\n';
        if(result.reason==RunReason::Success) {
            std::cout<<"actual_text=";
            for(auto q:text){if(q>=a.printable_index.size())throw std::runtime_error("text_id");
                const auto p=a.printable_index[q];if(p>=0&&p<256)std::cout<<char(p);}
        }
        std::cout<<'\n';if(mismatches)return 1;
        if(dump_path) {
            std::vector<float> lags(memory.lag_each/sizeof(float));
            ck(cudaMemcpy(lags.data(),executor.device_lags(),memory.lag_each,cudaMemcpyDeviceToHost));
            std::ofstream dump(dump_path,std::ios::binary);
            dump.write(reinterpret_cast<const char*>(lags.data()),std::streamsize(memory.lag_each));
            if(!dump)throw std::runtime_error("dump_io");
        }
        if(bad_prefix) {
            std::vector<Input> malformed(expected.begin(),expected.begin()+prefix);
            malformed[0]={TokenKind::Text,a.config[8]};
            if(runtime.prepare(malformed,true,error)||error!="prefix_boundary")throw std::runtime_error("bad_prefix_accepted");
            std::vector<float> after(state.size());ck(cudaMemcpy(after.data(),executor.device_state(),after.size()*sizeof(float),cudaMemcpyDeviceToHost));
            if(after!=state)throw std::runtime_error("bad_prefix_mutated_machine_state");
        }
        }
    }
    ck(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
