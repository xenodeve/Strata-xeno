#include "strata/core/qwenputer.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace strata::qwenputer;
static void ck(cudaError_t result){if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));}
template<class T>static std::vector<T> read(const char* path){
    std::ifstream file(path,std::ios::binary|std::ios::ate);if(!file)throw std::runtime_error("fixture_io");
    const auto size=std::size_t(file.tellg());if(size%sizeof(T)||size>(256ull<<20))throw std::runtime_error("fixture_size");
    std::vector<T> out(size/sizeof(T));file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(out.data()),std::streamsize(size)))throw std::runtime_error("fixture_io");
    return out;
}
static std::vector<unsigned char> bytes(Executor& executor,const MemoryPlan& plan){
    std::vector<unsigned char> out(plan.state_each+plan.lag_each+sizeof(Input)+sizeof(StepStatus));
    std::size_t at=0;
    auto copy=[&](const void* source,std::size_t size){ck(cudaMemcpy(out.data()+at,source,size,cudaMemcpyDeviceToHost));at+=size;};
    copy(executor.device_state(),plan.state_each);copy(executor.device_lags(),plan.lag_each);
    copy(executor.device_input(),sizeof(Input));copy(executor.device_status(),sizeof(StepStatus));return out;
}
int main(int argc,char** argv)try{
    if(argc!=3)return 2;
    const auto raw=read<unsigned char>(argv[2]);if(raw.size()!=sizeof(Binding))return 2;
    Binding binding;std::memcpy(&binding,raw.data(),raw.size());Artifact artifact;MemoryPlan plan;std::string error;
    if(!load_artifact(argv[1],binding,{},artifact,plan,error))throw std::runtime_error(error);
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    {
        Executor first,second;
        if(!first.init(artifact,stream,error)||!second.init(artifact,stream,error))throw std::runtime_error(error);
        if(first.device_state()==second.device_state())throw std::runtime_error("relocation_not_tested");
        {
            StateImage implicit;
            if(implicit.init(first,nullptr,error)||error!="state_image_explicit_stream_required")
                throw std::runtime_error("implicit_stream_was_accepted");
        }
        {
            StateImage empty;
            if(empty.capture(first,stream,error)||empty.restore(first,stream,error)||empty.allocated_bytes())
                throw std::runtime_error("uninitialized_image_accepted");
        }
        StateImage image;
        if(!image.init(first,stream,error))throw std::runtime_error("image_init:"+error);
        ck(cudaStreamSynchronize(stream));const auto original=bytes(first,plan);const auto allocation=image.allocated_bytes();
        if(allocation!=original.size())throw std::runtime_error("image_budget");
        if(image.init(second,stream,error)||error!="state_image_initialized"||image.allocated_bytes()!=allocation)
            throw std::runtime_error("image_reinitialized");
        cudaStream_t other;ck(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking));
        if(image.capture(first,other,error)||error!="state_image_stream_mismatch"
           ||image.restore(second,other,error)||error!="state_image_stream_mismatch")
            throw std::runtime_error("foreign_stream_accepted");
        ck(cudaStreamDestroy(other));
        if(bytes(first,plan)!=original||bytes(second,plan)!=original)throw std::runtime_error("rejected_copy_mutated_state");
        {
            auto changed=artifact;changed.digest[0]^=1;Executor foreign;
            if(!foreign.init(changed,stream,error))throw std::runtime_error(error);
            const auto before=bytes(foreign,plan);
            if(image.capture(foreign,stream,error)||error!="state_image_artifact_mismatch"
               ||image.restore(foreign,stream,error)||error!="state_image_artifact_mismatch")
                throw std::runtime_error("foreign_artifact_accepted");
            if(bytes(foreign,plan)!=before)throw std::runtime_error("foreign_artifact_mutated");
        }
        int device,count;ck(cudaGetDevice(&device));ck(cudaGetDeviceCount(&count));
        if(count>1) {
            ck(cudaSetDevice((device+1)%count));
            if(image.capture(first,stream,error)||error!="state_image_device_mismatch"
               ||image.restore(second,stream,error)||error!="state_image_device_mismatch")
                throw std::runtime_error("foreign_device_accepted");
            ck(cudaSetDevice(device));
        }
        auto step=[&](Executor& e,unsigned id){Input input{TokenKind::Text,id};ck(cudaMemcpyAsync(e.device_input(),&input,sizeof(input),cudaMemcpyHostToDevice,stream));
            if(!e.enqueue_step(stream,error))throw std::runtime_error(error);ck(cudaStreamSynchronize(stream));};
        step(first,artifact.config[7]); // opening-call marker changes bridge state and history
        if(bytes(first,plan)==original)throw std::runtime_error("mutation_control_failed");
        if(!image.restore(first,stream,error))throw std::runtime_error(error);
        ck(cudaStreamSynchronize(stream));if(bytes(first,plan)!=original)throw std::runtime_error("rollback_incomplete");
        step(first,artifact.config[7]);
        if(!image.capture(first,stream,error))throw std::runtime_error(error);
        ck(cudaStreamSynchronize(stream));const auto captured=bytes(first,plan);
        step(first,artifact.config[9]);
        if(!image.restore(second,stream,error))throw std::runtime_error(error);
        ck(cudaStreamSynchronize(stream));if(bytes(second,plan)!=captured)throw std::runtime_error("relocated_restore_incomplete");
        if(!image.restore(first,stream,error))throw std::runtime_error(error);
        step(first,artifact.config[9]);step(second,artifact.config[9]);
        if(bytes(first,plan)!=bytes(second,plan))throw std::runtime_error("restored_continuation_diverged");
        if(image.allocated_bytes()!=allocation)throw std::runtime_error("capture_reallocated");
        std::cout<<"PASS circuit rollback, relocated restore and identical continuation; image_bytes="<<allocation<<'\n';
    }
    ck(cudaStreamDestroy(stream));return 0;
}catch(const std::exception& error){std::cerr<<"FAIL "<<error.what()<<'\n';return 1;}
