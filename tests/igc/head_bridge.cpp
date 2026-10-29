// Optional training diagnostic: exact native head forward, declared W^T STE.
#include "strata/core/native_head.hpp"
#include "head_vjp.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#if defined(_WIN32)
#define IGC_API extern "C" __declspec(dllexport)
#else
#define IGC_API extern "C" __attribute__((visibility("default")))
#endif
namespace {
constexpr int width=2560,vocabulary=248320;
thread_local char error[1025]{};
int fail(std::string_view message,int code=1) {
    const auto size=std::min<size_t>(message.size(),1024);
    std::memcpy(error,message.data(),size);error[size]=0;return code;
}
struct Head {
    strata::core::NativeHead native;
    int device=-1;
    cudaEvent_t event=nullptr;
    cudaStream_t last_stream=nullptr;
    bool pending=false,recorded=false,faulted=false;
    ~Head() {
        if (pending) {
            if (recorded) cudaEventSynchronize(event);
            else if (last_stream) cudaStreamSynchronize(last_stream);
        }
        if (event) cudaEventDestroy(event);
    }
};
struct Registry {
    std::mutex mutex;
    uint64_t next=1;
    std::unordered_map<uint64_t,std::unique_ptr<Head>> heads;
};
Registry& registry() {
    // Explicit close owns CUDA teardown. Never run CUDA in a DLL global dtor
    // after a framework has already torn down its context.
    static auto* value=new Registry;
    return *value;
}
bool pointer(const void* value,int device) {
    if (!value || reinterpret_cast<uintptr_t>(value)%16) {fail("cuda_head_pointer_alignment");return false;}
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes,value)!=cudaSuccess) {
        cudaGetLastError();fail("cuda_head_device_pointer");return false;
    }
    if (attributes.type!=cudaMemoryTypeDevice || attributes.device!=device) {
        fail("cuda_head_device_pointer");return false;
    }
    return true;
}
bool disjoint(const void* a,size_t an,const void* b,size_t bn) {
    auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x<=std::numeric_limits<uintptr_t>::max()-an && y<=std::numeric_limits<uintptr_t>::max()-bn &&
           (x+an<=y || y+bn<=x);
}
int execute(uint64_t id,const float* input,float* output,uint32_t count,void* stream,bool backward) {
    auto& r=registry();std::scoped_lock lock(r.mutex);
    auto found=r.heads.find(id);
    if (found==r.heads.end()) return fail("cuda_head_handle");
    auto& h=*found->second;
    if (h.faulted) return fail("cuda_head_faulted",2);
    int device=-1;
    if (cudaGetDevice(&device)!=cudaSuccess || device!=h.device) return fail("cuda_head_current_device");
    if (!stream || !count || count>64) return fail("cuda_head_stream_or_batch");
    if (!pointer(input,device) || !pointer(output,device)) return 1;
    const size_t in_width=backward?vocabulary:width,out_width=backward?width:vocabulary;
    if (!disjoint(input,count*in_width*4,output,count*out_width*4)) return fail("cuda_head_overlap");
    auto cs=static_cast<cudaStream_t>(stream);
    unsigned flags=0;
    if (cudaStreamGetFlags(cs,&flags)!=cudaSuccess) {cudaGetLastError();return fail("cuda_head_stream");}
    auto fault=[&](std::string_view message){
        h.faulted=true;
        h.recorded=cudaEventRecord(h.event,cs)==cudaSuccess;
        return fail(message,2);
    };
    if (h.pending && cudaStreamWaitEvent(cs,h.event,0)!=cudaSuccess) return fault("cuda_head_dependency");
    h.pending=true;h.recorded=false;h.last_stream=cs;
    try {
        for (uint32_t i=0;i<count;++i) {
            if (backward) igc_head_weight_vjp(h.native.weights(),input+i*in_width,output+i*out_width,width,vocabulary,stream);
            else {
                std::string detail;
                if (!h.native.run(input+i*in_width,output+i*out_width,stream,detail)) return fault(detail);
            }
        }
    } catch (const std::exception& detail) {return fault(detail.what());}
    if (cudaEventRecord(h.event,cs)!=cudaSuccess) return fault("cuda_head_event_record");
    h.recorded=true;error[0]=0;return 0;
}
}
IGC_API uint32_t igc_head_bridge_abi() {return 1;}
IGC_API const char* igc_head_error() {return error;}
IGC_API uint64_t igc_head_open(const char* model) {
    try {
        auto& r=registry();std::scoped_lock lock(r.mutex);
        if (!model || !*model || std::strlen(model)>32768) {fail("cuda_head_model_path");return 0;}
        if (r.heads.size()>=2 || !r.next) {fail("cuda_head_handle_limit");return 0;}
        auto head=std::make_unique<Head>();
        if (cudaGetDevice(&head->device)!=cudaSuccess) {fail("cuda_head_device");return 0;}
        std::string detail;
        if (!head->native.load({model},width,vocabulary,detail)) {fail(detail);return 0;}
        if (head->native.type()!=8 || head->native.weight_bytes()!=675430400) {fail("cuda_head_q8_geometry");return 0;}
        if (cudaEventCreateWithFlags(&head->event,cudaEventDisableTiming)!=cudaSuccess) {fail("cuda_head_event");return 0;}
        auto id=r.next++;r.heads.emplace(id,std::move(head));error[0]=0;return id;
    } catch (const std::exception& detail) {fail(detail.what());return 0;}
    catch (...) {fail("cuda_head_open_exception");return 0;}
}
IGC_API int igc_head_project(uint64_t id,const float* input,float* output,uint32_t count,void* stream) {
    try {return execute(id,input,output,count,stream,false);}
    catch (const std::exception& detail) {return fail(detail.what(),2);}
    catch (...) {return fail("cuda_head_project_exception",2);}
}
IGC_API int igc_head_vjp(uint64_t id,const float* input,float* output,uint32_t count,void* stream) {
    try {return execute(id,input,output,count,stream,true);}
    catch (const std::exception& detail) {return fail(detail.what(),2);}
    catch (...) {return fail("cuda_head_vjp_exception",2);}
}
IGC_API int igc_head_close(uint64_t id) {
    try {
    auto& r=registry();std::scoped_lock lock(r.mutex);
    auto found=r.heads.find(id);
    if (found==r.heads.end()) return fail("cuda_head_handle");
    auto& h=*found->second;int device=-1;
    if (cudaGetDevice(&device)!=cudaSuccess || device!=h.device) return fail("cuda_head_current_device");
    auto status=!h.pending?cudaSuccess:h.recorded?cudaEventSynchronize(h.event):cudaStreamSynchronize(h.last_stream);
    h.pending=false;r.heads.erase(found);
    if (status!=cudaSuccess) return fail("cuda_head_close_fence",2);
    error[0]=0;return 0;
    } catch (const std::exception& detail) {return fail(detail.what(),2);}
    catch (...) {return fail("cuda_head_close_exception",2);}
}
