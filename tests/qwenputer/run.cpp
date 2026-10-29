#include "strata/core/qwenputer_runner.hpp"
#include "strata/timeline.hpp"
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif
using namespace strata::qwenputer;
static void ck(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
template<class T>static std::vector<T> read(const char* path,std::size_t limit) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);if(!file)throw std::runtime_error("input_io");
    const auto n=std::size_t(file.tellg());if(n>limit||n%sizeof(T))throw std::runtime_error("input_size");
    std::vector<T> data(n/sizeof(T));file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(data.data()),std::streamsize(n)))throw std::runtime_error("input_io");
    return data;
}
static std::string hex(const Digest& digest) {
    const char* chars="0123456789abcdef";std::string result;
    for(auto b:digest){result+=chars[b>>4];result+=chars[b&15];}return result;
}
int main(int argc,char** argv)try {
    struct Flush{~Flush(){strata::timeline::flush();}}flush;
    if(argc!=4){std::cerr<<"artifact binding prefix-inputs (no oracle arguments)\n";return 2;}
    const auto identity=read<std::uint8_t>(argv[2],192);if(identity.size()!=192)return 2;
    Binding binding;std::memcpy(&binding,identity.data(),192);
    Artifact artifact;MemoryPlan memory;std::string error;
    if(!load_artifact(argv[1],binding,{},artifact,memory,error))throw std::runtime_error(error);
    const auto prefix=read<Input>(argv[3],82*sizeof(Input));
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    std::size_t free_before,total,free_executor,free_runtime,free_execution;
    ck(cudaMemGetInfo(&free_before,&total));
    {
        Executor executor;if(!executor.init(artifact,stream,error))throw std::runtime_error(error);
        ck(cudaMemGetInfo(&free_executor,&total));
        Runtime runtime;if(!runtime.init(executor,artifact,{},stream,error))throw std::runtime_error(error);
        ck(cudaMemGetInfo(&free_runtime,&total));
        if(!runtime.prepare(prefix,true,error))throw std::runtime_error(error);
        ChunkResult result;std::vector<unsigned> text;unsigned chunks=0;
        do {
            if(++chunks>4096)throw std::runtime_error("host_chunk_bound");
            if(!runtime.enqueue_chunk(128,256,error)||!runtime.collect(result,error))throw std::runtime_error(error);
            text.insert(text.end(),result.text.begin(),result.text.end());
        }while(result.reason==RunReason::Budget||result.reason==RunReason::Output);
        if(result.reason!=RunReason::Success)text.clear();
        std::vector<std::uint8_t> state(memory.state_each);
        ck(cudaMemcpy(state.data(),executor.device_state(),state.size(),cudaMemcpyDeviceToHost));
        const auto state_digest=hex(sha256(state));
        ck(cudaMemGetInfo(&free_execution,&total));
        std::uint64_t private_bytes=0,private_peak=0,working_peak=0;bool process_memory_available=false;
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX counters{};counters.cb=sizeof(counters);
        if(!GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),sizeof(counters)))
            throw std::runtime_error("process_memory_observation");
        private_bytes=counters.PrivateUsage;private_peak=counters.PeakPagefileUsage;working_peak=counters.PeakWorkingSetSize;
        process_memory_available=true;
#endif
        std::cout<<"{\"kind\":\"native-execution\",\"evidence_level\":\"observed-execution\",\"reason\":"
                 <<unsigned(result.reason)<<",\"error\":"<<result.error<<",\"steps\":"<<result.total_steps
                 <<",\"internal_steps\":"<<result.internal_steps<<",\"trace_steps\":"<<result.trace_steps
                 <<",\"chunks\":"<<chunks<<",\"artifact_sha256\":\""<<hex(artifact.digest)
                 <<"\",\"state_sha256\":\""<<state_digest<<"\",\"owned_gpu_bytes\":"
                 <<executor.allocated_bytes()+runtime.allocated_bytes()
                 <<",\"owned_device_buffer_peak_bytes\":"<<executor.allocated_bytes()+runtime.allocated_bytes()
                 <<",\"process_private_commit_bytes\":"<<private_bytes<<",\"process_private_commit_peak_bytes\":"<<private_peak
                 <<",\"process_memory_available\":"<<(process_memory_available?"true":"false")
                 <<",\"working_set_peak_bytes\":"<<working_peak<<",\"cuda_free_before_executor_bytes\":"<<free_before
                 <<",\"cuda_total_bytes\":"<<total
                 <<",\"cuda_free_after_executor_bytes\":"<<free_executor<<",\"cuda_free_after_runtime_bytes\":"<<free_runtime
                 <<",\"cuda_free_after_execution_bytes\":"<<free_execution<<",\"text_ids\":[";
        for(std::size_t i=0;i<text.size();++i){if(i)std::cout<<',';std::cout<<text[i];}
        std::cout<<"]}\n";
    }
    ck(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
