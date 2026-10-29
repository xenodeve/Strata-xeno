#include "strata/core/igc_exact.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t v){if(v!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(v));}
struct Buffer{void* p=nullptr;explicit Buffer(std::size_t bytes){check(cudaMalloc(&p,bytes));}~Buffer(){if(p)cudaFree(p);}template<class T>T* as(){return static_cast<T*>(p);}};
struct Stream{cudaStream_t value=nullptr;Stream(){check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}~Stream(){if(value){cudaStreamSynchronize(value);cudaStreamDestroy(value);}}};
}
int main(int argc,char** argv)try{
    if(argc!=4)throw std::runtime_error("two-request-file output-prefix count");
    const int count=std::stoi(argv[3]);if(count<1 || count>64)throw std::runtime_error("count");
    std::array<strata::igc::ExactRequest,2> input;
    std::ifstream file(argv[1],std::ios::binary);file.read(reinterpret_cast<char*>(input.data()),sizeof(input));
    if(!file || file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("request_bytes");
    Stream stream;Buffer request(sizeof(input[0])),results(count*sizeof(strata::igc::ExactResult));std::string error;
    if(strata::igc::broadcast_exact_result(results.as<strata::igc::ExactResult>(),0,stream.value,error)
        || strata::igc::broadcast_exact_result(results.as<strata::igc::ExactResult>(),65,stream.value,error)
        || strata::igc::broadcast_exact_result(nullptr,count,stream.value,error)
        || strata::igc::broadcast_exact_result(results.as<strata::igc::ExactResult>(),count,nullptr,error))throw std::runtime_error("bad_api_admitted");
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    check(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));
    if(!strata::igc::execute_exact(request.as<strata::igc::ExactRequest>(),results.as<strata::igc::ExactResult>(),1,stream.value,error)
            || !strata::igc::broadcast_exact_result(results.as<strata::igc::ExactResult>(),count,stream.value,error))throw std::runtime_error(error);
    check(cudaStreamEndCapture(stream.value,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    std::vector<strata::igc::ExactResult> first(count),host(count);
    for(int repeat=0;repeat<3;++repeat){
        check(cudaMemcpyAsync(request.p,&input[repeat==1?1:0],sizeof(input[0]),cudaMemcpyHostToDevice,stream.value));
        check(cudaGraphLaunch(executable,stream.value));
        check(cudaMemcpyAsync(host.data(),results.p,host.size()*sizeof(host[0]),cudaMemcpyDeviceToHost,stream.value));check(cudaStreamSynchronize(stream.value));
        for(int i=1;i<count;++i)if(std::memcmp(&host[0],&host[i],sizeof(host[0])))throw std::runtime_error("broadcast_mismatch");
        if(!repeat)first=host;
        if(repeat==2 && std::memcmp(first.data(),host.data(),host.size()*sizeof(host[0])))throw std::runtime_error("reset_mismatch");
        const auto path=std::string(argv[2])+"."+std::to_string(repeat)+".bin";
        if(std::filesystem::exists(path))throw std::runtime_error("output_exists");
        std::ofstream output(path,std::ios::binary);output.write(reinterpret_cast<const char*>(host.data()),static_cast<std::streamsize>(host.size()*sizeof(host[0])));output.close();if(!output)throw std::runtime_error("output_write");
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
    std::cout<<"{\"count\":"<<count<<",\"graph\":true,\"reset_bitwise\":true}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
