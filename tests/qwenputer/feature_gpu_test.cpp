#include "strata/core/qwenputer.hpp"
#include <bit>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
using namespace strata::qwenputer;
static void ck(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
int main(int argc, char** argv) try {
    if(argc!=3)return 2;
    std::ifstream file(argv[2],std::ios::binary);
    std::vector<char> bytes{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
    if(bytes.size()!=sizeof(Binding))return 2;
    Binding binding;std::memcpy(&binding,bytes.data(),bytes.size());
    Artifact a;MemoryPlan memory;std::string error;
    if(!load_artifact(argv[1],binding,{},a,memory,error))throw std::runtime_error(error);
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    Executor executor;if(!executor.init(a,stream,error))throw std::runtime_error(error);
    std::vector<float> before(a.config[6]),after(a.config[6]);
    ck(cudaMemcpy(before.data(),executor.device_state(),before.size()*sizeof(float),cudaMemcpyDeviceToHost));
    float* values=nullptr;ck(cudaMalloc(&values,262144*sizeof(float)));
    std::vector<float> actual(262144);
    std::uint64_t checked=0;
    for(std::uint32_t f=0;f<a.features.size();++f)for(std::uint32_t kind=0;kind<2;++kind) {
        const auto n=a.config[kind==0?1:2];
        if(!executor.enqueue_feature_probe(f,static_cast<TokenKind>(kind),0,n,values,stream,error))
            throw std::runtime_error("feature_probe:"+error);
        ck(cudaMemcpyAsync(actual.data(),values,n*sizeof(float),cudaMemcpyDeviceToHost,stream));
        ck(cudaStreamSynchronize(stream));
        for(std::uint32_t i=0;i<n;++i) {
            float expected;
            if(!feature_value(a,a.features[f],static_cast<TokenKind>(kind),i,expected)
                || std::bit_cast<std::uint32_t>(expected)!=std::bit_cast<std::uint32_t>(actual[i])) {
                std::cerr<<"FAIL feature="<<f<<" kind="<<kind<<" id="<<i<<'\n';return 1;
            }
            ++checked;
        }
    }
    if(executor.enqueue_feature_probe(0,TokenKind::Trace,230323,2,values,stream,error))
        throw std::runtime_error("invalid probe range accepted");
    ck(cudaMemcpy(after.data(),executor.device_state(),after.size()*sizeof(float),cudaMemcpyDeviceToHost));
    if(std::memcmp(before.data(),after.data(),before.size()*sizeof(float)))
        throw std::runtime_error("read-only embedding probe changed machine state");
    ck(cudaFree(values));ck(cudaStreamDestroy(stream));
    std::cout<<"PASS native GPU embedding bit-identical to verified CPU: "<<checked<<" values\n";
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
