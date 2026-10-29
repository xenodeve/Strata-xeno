#include "strata/core/igc_rational_exact.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
namespace re=strata::igc::rational_exact;
namespace ie=strata::igc::integer_extended;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
struct Device {
    void* input=nullptr;void* output=nullptr;cudaStream_t stream=nullptr;
    ~Device(){if(stream)cudaStreamSynchronize(stream);if(input)cudaFree(input);if(output)cudaFree(output);if(stream)cudaStreamDestroy(stream);}
};
int main()try{
    Device device;check(cudaStreamCreateWithFlags(&device.stream,cudaStreamNonBlocking));
    check(cudaMalloc(&device.input,sizeof(re::Request)));check(cudaMalloc(&device.output,sizeof(re::Result)));
    re::Request request;request.invocation=31;request.position=65;
    request.lhs.numerator.length=request.lhs.denominator.length=1;
    request.rhs.numerator.length=request.rhs.denominator.length=1;
    request.lhs.numerator.digits[0]=request.rhs.numerator.digits[0]=1;
    request.lhs.denominator.digits[0]=3;request.rhs.denominator.digits[0]=6;
    check(cudaMemcpyAsync(device.input,&request,sizeof(request),cudaMemcpyHostToDevice,device.stream));
    std::string error;strata::igc::Integer absent{};
    require(strata::igc::execute_exact(static_cast<strata::igc::ExactRequest*>(device.input),
        static_cast<strata::igc::ExactResult*>(device.output),1,device.stream,error),error.c_str());
    strata::igc::ExactResult one;
    check(cudaMemcpyAsync(&one,device.output,sizeof(one),cudaMemcpyDeviceToHost,device.stream));check(cudaStreamSynchronize(device.stream));
    require(one.version==1 && one.status==strata::igc::Status::Malformed && one.invocation==31 && one.position==65
        && !std::memcmp(&one.value,&absent,sizeof(absent)),"abi1_refuses_abi3");
    require(ie::execute(static_cast<ie::Request*>(device.input),static_cast<ie::Result*>(device.output),1,device.stream,error),error.c_str());
    ie::Result two;
    check(cudaMemcpyAsync(&two,device.output,sizeof(two),cudaMemcpyDeviceToHost,device.stream));check(cudaStreamSynchronize(device.stream));
    require(two.version==2 && two.status==ie::Status::Malformed && two.invocation==31 && two.position==65
        && !std::memcmp(&two.value,&absent,sizeof(absent)),"abi2_refuses_abi3");
    for(uint32_t version:{1u,2u}){
        request.version=version;
        check(cudaMemcpyAsync(device.input,&request,sizeof(request),cudaMemcpyHostToDevice,device.stream));
        require(re::execute(static_cast<re::Request*>(device.input),static_cast<re::Result*>(device.output),1,device.stream,error),error.c_str());
        re::Result three;const re::Rational absent_fraction{};
        check(cudaMemcpyAsync(&three,device.output,sizeof(three),cudaMemcpyDeviceToHost,device.stream));check(cudaStreamSynchronize(device.stream));
        require(three.version==3 && three.status==re::Status::Malformed && three.invocation==31 && three.position==65
            && !std::memcmp(&three.value,&absent_fraction,sizeof(absent_fraction)),"abi3_refuses_older_version");
    }
    std::cout<<"PASS ABI1/ABI2/ABI3 record refusal, no numeric value\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
