#include "strata/core/igc_readiness.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
void check(cudaError_t v){if(v!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(v));}
auto digest(const char* p){strata::igc::adapter::Digest r{};std::string v(p);if(v.size()!=64)throw std::runtime_error("digest_length");
    auto digit=[](char c){if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::runtime_error("digest_hex");};
    for(int i=0;i<32;++i)r[i]=uint8_t(digit(v[i*2])*16+digit(v[i*2+1]));return r;}
struct Buffer {void* p=nullptr;explicit Buffer(std::size_t n){check(cudaMalloc(&p,n));}~Buffer(){if(p)cudaFree(p);}template<class T>T* as(){return static_cast<T*>(p);}};
struct Stream {cudaStream_t s=nullptr;Stream(){check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){if(s){cudaStreamSynchronize(s);cudaStreamDestroy(s);}}};
template<class T>void read(std::ifstream& f,std::vector<T>& v){f.read(reinterpret_cast<char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(T)));if(!f)throw std::runtime_error("input_truncated");}
template<class T>void write(const std::string& path,const std::vector<T>& v){if(std::filesystem::exists(path))throw std::runtime_error("output_exists");
    std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(T)));f.close();if(!f)throw std::runtime_error("output_write");}
}
int main(int argc,char** argv)try{
    if(argc!=8)throw std::runtime_error("artifact model-sha tokenizer-sha file-sha inputs output-prefix mode");
    const std::string mode=argv[7];if(mode!="normal"&&mode!="graph"&&mode!="off")throw std::runtime_error("probe_mode");
    std::ifstream input(argv[5],std::ios::binary);char magic[8];uint32_t count=0;
    input.read(magic,8);input.read(reinterpret_cast<char*>(&count),4);
    if(!input || std::memcmp(magic,"IGCRIN01",8) || !count || count>64)throw std::runtime_error("probe_input_header");
    const auto expected=12ull+uint64_t(count)*2560*4;
    if(std::filesystem::file_size(argv[5])!=expected)throw std::runtime_error("probe_input_bytes");
    std::vector<float> states(std::size_t(count)*2560),actual(count*2,42.125f),first;
    std::vector<uint32_t> status(count,0x12345678),first_status;
    read(input,states);
    strata::igc::adapter::Artifact a;std::string error;
    strata::igc::adapter::Binding binding{digest(argv[2]),digest(argv[3]),digest(argv[4])};
    if(!strata::igc::adapter::load(argv[1],binding,true,a,error))throw std::runtime_error(error);
    Stream stream;strata::igc::ReadinessMapper mapper;
    if(!mapper.run(nullptr,nullptr,nullptr,0,nullptr,false,error) || mapper.device_bytes())throw std::runtime_error("hard_off_contract");
    if(mapper.load(a,nullptr,count,error) || mapper.load(a,stream.s,0,error) || mapper.load(a,stream.s,65,error))throw std::runtime_error("load_arguments_accepted");
    if(mode!="off" && (!mapper.load(a,stream.s,count,error) || mapper.load(a,stream.s,count,error)))throw std::runtime_error("load_contract "+error);
    Buffer x(states.size()*4),out(actual.size()*4),codes(status.size()*4);
    check(cudaMemcpyAsync(x.p,states.data(),states.size()*4,cudaMemcpyHostToDevice,stream.s));
    check(cudaMemcpyAsync(out.p,actual.data(),actual.size()*4,cudaMemcpyHostToDevice,stream.s));
    check(cudaMemcpyAsync(codes.p,status.data(),status.size()*4,cudaMemcpyHostToDevice,stream.s));
    if(mode!="off" && mapper.run(x.as<float>(),out.as<float>(),codes.as<uint32_t>(),count+1,stream.s,true,error))throw std::runtime_error("run_capacity_accepted");
    auto invoke=[&]{if(!mapper.run(x.as<float>(),out.as<float>(),codes.as<uint32_t>(),count,stream.s,mode!="off",error))throw std::runtime_error(error);};
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    if(mode=="graph"){check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));invoke();check(cudaStreamEndCapture(stream.s,&graph));check(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));}
    bool b_diff=false;
    for(int repeat=0;repeat<5;++repeat){
        const auto before=actual;
        if(repeat==2)check(cudaMemsetAsync(x.p,0,states.size()*4,stream.s));
        if(repeat==3)check(cudaMemcpyAsync(x.p,states.data(),states.size()*4,cudaMemcpyHostToDevice,stream.s));
        if(exec)check(cudaGraphLaunch(exec,stream.s));else invoke();
        check(cudaMemcpyAsync(actual.data(),out.p,actual.size()*4,cudaMemcpyDeviceToHost,stream.s));
        check(cudaMemcpyAsync(status.data(),codes.p,status.size()*4,cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
        // Failure preserves this invocation's caller buffer, which can contain
        // B's valid result when the next A input is nonfinite. It need not equal
        // the first A's sentinel from before that intervening successful call.
        for(uint32_t row=0;row<count;++row)if(status[row] && std::memcmp(before.data()+row*2,actual.data()+row*2,2*4))
            throw std::runtime_error("failure_publication");
        if(!repeat){first=actual;first_status=status;}
        else if(repeat==2)b_diff=std::memcmp(first.data(),actual.data(),actual.size()*4)!=0;
        else {
            if(first_status!=status)throw std::runtime_error("repeat_status_mismatch");
            for(uint32_t row=0;row<count;++row)if(!status[row] && std::memcmp(first.data()+row*2,actual.data()+row*2,2*4))
                throw std::runtime_error("valid_repeat_mismatch");
        }
    }
    if(exec)check(cudaGraphExecDestroy(exec));if(graph)check(cudaGraphDestroy(graph));
    // Publish the initial A observation; all later calls independently checked
    // valid repeat parity and failure preservation against their own before-image.
    write(std::string(argv[6])+".logits.bin",first);write(std::string(argv[6])+".status.bin",first_status);
    std::cout<<"{\"count\":"<<count<<",\"valid_repeats_bitwise\":true,\"failure_bytes_preserved\":true,\"alternating_input_changed\":"<<(b_diff?"true":"false")
        <<",\"device_weights_scratch_bytes\":"<<mapper.device_bytes()<<",\"diagnostic_only\":true}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
