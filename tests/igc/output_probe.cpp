#include "strata/core/igc_output.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
strata::igc::adapter::Digest digest(const char* text){
    std::string value(text);if(value.size()!=64)throw std::runtime_error("digest_length");
    auto digit=[](char c){if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::runtime_error("digest_hex");};
    strata::igc::adapter::Digest result{};for(std::size_t i=0;i<32;++i)result[i]=uint8_t(digit(value[i*2])*16+digit(value[i*2+1]));return result;
}
struct Buffer {void* p=nullptr;explicit Buffer(std::size_t n){check(cudaMalloc(&p,n));}~Buffer(){if(p)cudaFree(p);}template<class T>T* as(){return static_cast<T*>(p);}};
struct Stream {cudaStream_t s=nullptr;Stream(){check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){if(s){cudaStreamSynchronize(s);cudaStreamDestroy(s);}}};
template<class T>void read(std::ifstream& f,std::vector<T>& v){f.read(reinterpret_cast<char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(T)));if(!f)throw std::runtime_error("input_truncated");}
template<class T>void write(const std::string& path,const std::vector<T>& v){
    if(std::filesystem::exists(path))throw std::runtime_error("output_exists");
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(T)));f.close();if(!f)throw std::runtime_error("output_write");
}
}
int main(int argc,char** argv)try{
    if(argc!=8)throw std::runtime_error("artifact model_sha tokenizer_sha file_sha input output_prefix mode[normal|off|graph]");
    const std::string mode(argv[7]);if(mode!="normal" && mode!="off" && mode!="graph")throw std::runtime_error("mode");
    strata::igc::adapter::Artifact artifact;std::string error;
    if(!strata::igc::adapter::load(argv[1],{digest(argv[2]),digest(argv[3]),digest(argv[4])},true,artifact,error))throw std::runtime_error(error);
    // This executable is explicitly diagnostic-only; its caller must not treat
    // successful parsing as production eligibility.
    std::ifstream input(argv[5],std::ios::binary);char magic[8]{};uint32_t count=0;
    input.read(magic,8);input.read(reinterpret_cast<char*>(&count),4);
    if(!input || std::memcmp(magic,"IGCOIN01",8) || !count || count>64)throw std::runtime_error("input_header");
    std::vector<float> original(count*2560),actual(original.size()),first(original.size());
    std::vector<strata::igc::ExactResult> results(count);
    std::vector<uint64_t> ids(count),anchors(count);std::vector<uint32_t> status(count),first_status(count);
    read(input,original);read(input,results);read(input,ids);read(input,anchors);
    if(input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("input_trailing");
    Stream stream;Buffer base(original.size()*4),mixed(original.size()*4),device_results(count*88),device_ids(count*8),device_anchors(count*8),device_status(count*4);
    check(cudaMemcpy(base.p,original.data(),original.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_results.p,results.data(),count*88,cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_ids.p,ids.data(),count*8,cudaMemcpyHostToDevice));check(cudaMemcpy(device_anchors.p,anchors.data(),count*8,cudaMemcpyHostToDevice));
    strata::igc::OutputMapper mapper;
    if(mode!="off" && !mapper.load(artifact,stream.s,error))throw std::runtime_error(error);
    auto invoke=[&](){
        check(cudaMemcpyAsync(mixed.p,base.p,original.size()*4,cudaMemcpyDeviceToDevice,stream.s));
        check(cudaMemsetAsync(device_status.p,0xff,count*4,stream.s));
        if(!mapper.run(mixed.as<float>(),device_results.as<strata::igc::ExactResult>(),device_ids.as<uint64_t>(),device_anchors.as<uint64_t>(),device_status.as<uint32_t>(),count,stream.s,mode!="off",error))throw std::runtime_error(error);
    };
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    if(mode=="graph"){
        check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));invoke();check(cudaStreamEndCapture(stream.s,&graph));
        check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    }
    for(int repeat=0;repeat<3;++repeat){
        if(executable)check(cudaGraphLaunch(executable,stream.s));else invoke();
        check(cudaMemcpyAsync(actual.data(),mixed.p,actual.size()*4,cudaMemcpyDeviceToHost,stream.s));
        check(cudaMemcpyAsync(status.data(),device_status.p,count*4,cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
        if(!repeat){first=actual;first_status=status;}
        else if(std::memcmp(first.data(),actual.data(),actual.size()*4) || first_status!=status)throw std::runtime_error("repeat_mismatch");
    }
    if(executable)check(cudaGraphExecDestroy(executable));if(graph)check(cudaGraphDestroy(graph));
    write(std::string(argv[6])+".mixed.bin",actual);write(std::string(argv[6])+".status.bin",status);
    std::cout<<"{\"count\":"<<count<<",\"repeat_bitwise\":true,\"graph\":"<<(mode=="graph"?"true":"false")
             <<",\"device_weights_scratch_bytes\":"<<mapper.device_bytes()<<",\"diagnostic_only\":true}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
