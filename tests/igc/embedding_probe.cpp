// Input-seam control only: the engine's own embedding and HC broadcast.
#include "strata/core/native_head.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

static void check(cudaError_t code) {
    if (code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
struct Device {
    strata::core::NativeEmbed embedding;
    cudaStream_t stream=nullptr;
    int32_t* ids=nullptr;
    float *word=nullptr,*residual=nullptr;
    ~Device() {
        if (stream) cudaStreamSynchronize(stream);
        if (residual) cudaFree(residual);
        if (word) cudaFree(word);
        if (ids) cudaFree(ids);
        if (stream) cudaStreamDestroy(stream);
    }
};
int main(int argc,char** argv) try {
    if (argc!=4) throw std::runtime_error("model token-ids.bin output-f32.bin");
    std::ifstream input(argv[2],std::ios::binary|std::ios::ate);
    if (!input || input.tellg()<=0 || input.tellg()>1024*4 || input.tellg()%4)
        throw std::runtime_error("embedding_token_bytes");
    const auto bytes=input.tellg();
    std::vector<int32_t> ids(static_cast<std::size_t>(bytes)/4);
    input.seekg(0);input.read(reinterpret_cast<char*>(ids.data()),bytes);
    if (!input) throw std::runtime_error("embedding_token_read");
    for (auto id:ids) if (id<0 || id>=248320) throw std::runtime_error("embedding_token_id");
    Device device;
    check(cudaStreamCreateWithFlags(&device.stream,cudaStreamNonBlocking));
    std::string error;
    if (!device.embedding.load({argv[1]},2560,248320,error)) throw std::runtime_error(error);
    check(cudaMalloc(reinterpret_cast<void**>(&device.ids),ids.size()*4));
    check(cudaMalloc(reinterpret_cast<void**>(&device.word),ids.size()*2560*4));
    check(cudaMalloc(reinterpret_cast<void**>(&device.residual),ids.size()*10240*4));
    check(cudaMemcpyAsync(device.ids,ids.data(),ids.size()*4,cudaMemcpyHostToDevice,device.stream));
    std::vector<float> first(ids.size()*10240),repeated(first.size());
    for (auto* output:{&first,&repeated}) {
        device.embedding.gather_dev(device.ids,static_cast<int64_t>(ids.size()),device.word,device.stream);
        strata::kernels::broadcast_streams(device.word,device.residual,2560,4,static_cast<int>(ids.size()),device.stream);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(output->data(),device.residual,output->size()*4,cudaMemcpyDeviceToHost,device.stream));
        check(cudaStreamSynchronize(device.stream));
    }
    if (std::memcmp(first.data(),repeated.data(),first.size()*4)) throw std::runtime_error("embedding_repeat");
    std::ofstream output(argv[3],std::ios::binary);
    output.write(reinterpret_cast<const char*>(first.data()),static_cast<std::streamsize>(first.size()*4));
    output.close();if (!output) throw std::runtime_error("embedding_output");
    std::cout<<"PASS native embedding control rows="<<ids.size()<<" HC=4 D=2560 repeat=bitwise type="
             <<device.embedding.type()<<" table_bytes="<<device.embedding.bytes()<<'\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 2;
}
