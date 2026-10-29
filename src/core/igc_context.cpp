#include "strata/core/igc_context.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <memory>
#include <vector>

namespace strata::igc {
namespace {
constexpr std::size_t row_bytes=ContextWindow::width*sizeof(float);
constexpr std::size_t plane_floats=ContextWindow::rows*ContextWindow::width;
bool stream_ok(void* raw){const auto s=static_cast<cudaStream_t>(raw);return s && s!=cudaStreamLegacy && s!=cudaStreamPerThread;}
bool checked(cudaError_t c,std::string& error){if(c==cudaSuccess)return true;error=cudaGetErrorString(c);return false;}
bool tokens_ok(std::span<const int64_t> tokens){return std::all_of(tokens.begin(),tokens.end(),[](int64_t t){return t>=0 && t<248320;});}
}
struct ContextWindow::Impl {
    float* storage=nullptr;uint8_t* valid=nullptr;
    cudaEvent_t last=nullptr;bool recorded=false;
    std::array<std::array<int64_t,rows>,2> positions;
    std::vector<int64_t> prefix;
    int64_t owned=0;
    Impl(){for(auto& p:positions)p.fill(-1);}
    ~Impl(){if(recorded)cudaEventSynchronize(last);if(storage)cudaFree(storage);if(valid)cudaFree(valid);if(last)cudaEventDestroy(last);}
    float* ring(int k){return storage+k*plane_floats;}
    float* output(int k){return storage+(k+2)*plane_floats;}
    void clear(){for(auto& p:positions)p.fill(-1);}
    bool wait(cudaStream_t s,std::string& e){return !recorded || checked(cudaStreamWaitEvent(s,last,0),e);}
    bool publish(cudaStream_t s,std::string& e){if(!checked(cudaEventRecord(last,s),e)){clear();return false;}recorded=true;return true;}
    bool bypass(cudaStream_t s,std::string& e){
        // Also invalidate the device mask, so an accidental downstream neural
        // launch cannot silently reuse a previous eligible input window.
        if(wait(s,e) && checked(cudaMemsetAsync(valid,0,rows,s),e) && publish(s,e))e.clear();
        return false;
    }
};
ContextWindow::~ContextWindow(){delete impl_;}
bool ContextWindow::allocate(void* raw,std::string& error)try{
    if(impl_ || !stream_ok(raw)){error="context_allocate_arguments";return false;}
    auto next=std::make_unique<Impl>();const auto s=static_cast<cudaStream_t>(raw);
    // Fixed device storage; no allocation occurs while observing/gathering.
    if(!checked(cudaMalloc(&next->storage,4*plane_floats*sizeof(float)),error)
            || !checked(cudaMalloc(&next->valid,rows),error)
            || !checked(cudaEventCreateWithFlags(&next->last,cudaEventDisableTiming),error))return false;
    next->prefix.reserve(max_tokens);
    if(!checked(cudaMemsetAsync(next->storage,0,4*plane_floats*sizeof(float),s),error)
            || !checked(cudaMemsetAsync(next->valid,0,rows,s),error)
            || !next->publish(s,error))return false;
    impl_=next.release();error.clear();return true;
}catch(const std::exception& e){error=std::string("context_allocate:")+e.what();return false;}
bool ContextWindow::begin(std::span<const int64_t> prefix,int64_t reused,bool may_reuse,std::string& error)try{
    if(!impl_ || prefix.empty() || prefix.size()>max_tokens || reused<0 || reused>=static_cast<int64_t>(prefix.size()) || !tokens_ok(prefix)){
        invalidate();error="context_prefix_arguments";return false;
    }
    auto& m=*impl_;
    const bool compatible=may_reuse && reused<=m.owned && reused<=static_cast<int64_t>(m.prefix.size())
        && std::equal(prefix.begin(),prefix.begin()+reused,m.prefix.begin());
    if(!compatible)m.clear();
    else for(auto& plane:m.positions)for(auto& p:plane)if(p>=reused)p=-1;
    m.prefix.assign(prefix.begin(),prefix.end());m.owned=reused;
    error.clear();return true;
}catch(const std::exception& e){invalidate();error=std::string("context_prefix:")+e.what();return false;}
bool ContextWindow::observe(ContextTap tap,int64_t pos0,int64_t count,const float* source,void* raw,std::string& error){
    const int kind=static_cast<int>(tap);
    if(!impl_ || impl_->prefix.empty() || kind<0 || kind>1 || pos0<0 || count<1 || count>max_tokens-pos0 || !source || !stream_ok(raw)){
        error="context_observe_arguments";return false;
    }
    auto& m=*impl_;const auto s=static_cast<cudaStream_t>(raw);
    const int64_t take=std::min<int64_t>(count,rows),skip=count-take,start=pos0+skip;
    const int slot=static_cast<int>(start%rows),first=static_cast<int>(std::min<int64_t>(take,rows-slot));
    if(!m.wait(s,error)
            || !checked(cudaMemcpyAsync(m.ring(kind)+slot*width,source+skip*width,first*row_bytes,cudaMemcpyDeviceToDevice,s),error)
            || (take>first && !checked(cudaMemcpyAsync(m.ring(kind),source+(skip+first)*width,(take-first)*row_bytes,cudaMemcpyDeviceToDevice,s),error))
            || !m.publish(s,error)){m.clear();return false;}
    for(int64_t p=start;p<pos0+count;++p)m.positions[kind][p%rows]=p;
    error.clear();return true;
}
bool ContextWindow::accept(int64_t pos0,std::span<const int64_t> tokens,std::string& error)try{
    if(!impl_ || pos0!=impl_->owned || tokens.empty() || pos0<0 || tokens.size()>static_cast<std::size_t>(max_tokens-pos0) || !tokens_ok(tokens)){
        error="context_accept_arguments";return false;
    }
    auto& m=*impl_;
    const auto overlap=std::min<std::size_t>(tokens.size(),m.prefix.size()-static_cast<std::size_t>(pos0));
    if(!std::equal(tokens.begin(),tokens.begin()+overlap,m.prefix.begin()+pos0)){
        invalidate();error="context_accept_prefix_mismatch";return false;
    }
    m.prefix.insert(m.prefix.end(),tokens.begin()+overlap,tokens.end());m.owned+=static_cast<int64_t>(tokens.size());
    error.clear();return true;
}catch(const std::exception& e){invalidate();error=std::string("context_accept:")+e.what();return false;}
bool ContextWindow::gather(int64_t anchor,void* raw,std::string& error){
    if(!impl_ || !stream_ok(raw) || anchor<0 || anchor>=max_tokens){error="context_gather_arguments";return false;}
    error.clear();auto& m=*impl_;
    const auto s=static_cast<cudaStream_t>(raw);
    if(anchor==0 || anchor>m.owned || anchor>=static_cast<int64_t>(m.prefix.size()))return m.bypass(s,error);
    const int64_t n=std::min<int64_t>(anchor,rows),start=anchor-n;
    for(int64_t p=start;p<anchor;++p)for(const auto& plane:m.positions)if(plane[p%rows]!=p)return m.bypass(s,error);
    const int slot=static_cast<int>(start%rows),first=static_cast<int>(std::min<int64_t>(n,rows-slot));
    if(!m.wait(s,error) || !checked(cudaMemsetAsync(m.valid,0,rows,s),error)
            || !checked(cudaMemsetAsync(m.valid+rows-n,1,n,s),error))return false;
    for(int k=0;k<2;++k){
        auto* dest=m.output(k)+(rows-n)*width;
        if(!checked(cudaMemcpyAsync(dest,m.ring(k)+slot*width,first*row_bytes,cudaMemcpyDeviceToDevice,s),error)
                || (n>first && !checked(cudaMemcpyAsync(dest+first*width,m.ring(k),(n-first)*row_bytes,cudaMemcpyDeviceToDevice,s),error)))return false;
    }
    return m.publish(s,error);
}
const float* ContextWindow::states(ContextTap tap)const{const int k=static_cast<int>(tap);return impl_ && k>=0 && k<2?impl_->output(k):nullptr;}
const uint8_t* ContextWindow::mask()const{return impl_?impl_->valid:nullptr;}
std::size_t ContextWindow::device_bytes()const{return impl_?4*plane_floats*sizeof(float)+rows:0;}
void ContextWindow::invalidate(){if(impl_){impl_->clear();impl_->prefix.clear();impl_->owned=0;}}
}
