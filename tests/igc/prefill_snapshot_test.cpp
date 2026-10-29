#include "strata/core/igc_prefill_snapshot.hpp"
#include "strata/core/igc_context.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
static void require(bool ok,const std::string& e){if(!ok)throw std::runtime_error(e);}
static void check(cudaError_t c){require(c==cudaSuccess,cudaGetErrorString(c));}
struct Stream {cudaStream_t s=nullptr;Stream(){check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){if(s){cudaStreamSynchronize(s);cudaStreamDestroy(s);}}};
struct Buffer {float* p=nullptr;explicit Buffer(size_t n){check(cudaMalloc(&p,n*4));}~Buffer(){if(p)cudaFree(p);}};
int main()try{
    using namespace strata;
    constexpr int width=igc::ContextWindow::width,count=137;
    Stream a,b,read;igc::PrefillSnapshot first,second;igc::ContextWindow context;std::string e;
    const bool allocated=first.allocate(a.s,e);
    require(allocated,"snapshot_allocate:"+e);require(second.allocate(b.s,e),e);
    require(first.device_bytes()==6553600 && second.device_bytes()==6553600,"snapshot_fixed_tail_storage");
    require(!first.allocate(a.s,e),"snapshot_reinit");require(context.allocate(read.s,e),e);
    std::vector<int64_t> prefix(300);for(int i=0;i<300;++i)prefix[i]=100+i;
    require(context.begin(prefix,0,false,e),e);Buffer source(count*width);
    std::vector<float> values(count*width);int callbacks=0;
    const core::EarlyActivationConsumer consumer{&callbacks,
        [](void* raw,const core::EarlyActivationRows& rows,std::string& err){
            ++*static_cast<int*>(raw);err="consumer_failure";return false;}};
    require(!first.emit(consumer,e) && callbacks==0,"snapshot_incomplete_emit");
    const core::EarlyActivationConsumer accept{&context,
        [](void* raw,const core::EarlyActivationRows& r,std::string& err){
            if(r.count!=80 || r.width!=width || (r.pos0!=57 && r.pos0!=194)){err="snapshot_descriptor";return false;}
            return static_cast<igc::ContextWindow*>(raw)->observe(r.tap==core::EarlyActivationTap::Embedding
                ?igc::ContextTap::Embedding:igc::ContextTap::FirstAttention,r.pos0,r.count,r.device,r.stream,err);}};
    auto stage=[&](igc::PrefillSnapshot& snapshot,cudaStream_t stream,int pos,int n){
        for(int tap=0;tap<2;++tap){
            for(int t=0;t<count;++t)for(int j=0;j<width;++j)values[t*width+j]=float(pos+t+tap*1000)+float(j%3)*.25f;
            check(cudaMemcpyAsync(source.p,values.data(),values.size()*4,cudaMemcpyHostToDevice,stream));
            require(snapshot.stage(tap?core::EarlyActivationTap::FirstAttention:core::EarlyActivationTap::Embedding,source.p,pos,n,stream,e),e);
            // Host source storage is reused; fence its upload before rewriting.
            check(cudaStreamSynchronize(stream));
        }
    };
    stage(first,a.s,0,count);require(first.emit(accept,e),e);
    // Reuse of the original source is ordered after first lane consumption.
    check(cudaStreamSynchronize(a.s));stage(second,b.s,137,count);require(second.emit(accept,e),e);
    // Wave snapshots alone do not grant input ownership; its successful part
    // must accept the full input span before a subsequent window/stage can use it.
    require(!context.gather(274,read.s,e) && e.empty(),"snapshot_unaccepted_input_exposed");
    require(context.accept(0,std::span<const int64_t>(prefix.data(),274),e),e);
    require(context.gather(274,read.s,e),e);
    std::vector<float> got(80*width);std::array<uint8_t,80> mask;
    for(int tap=0;tap<2;++tap){
        check(cudaMemcpyAsync(got.data(),context.states(tap?igc::ContextTap::FirstAttention:igc::ContextTap::Embedding),got.size()*4,cudaMemcpyDeviceToHost,read.s));
        check(cudaStreamSynchronize(read.s));
        for(int t=0;t<80;++t)for(int j=0;j<width;++j)require(got[t*width+j]==float(194+t+tap*1000)+float(j%3)*.25f,"snapshot_source_lifetime_or_order");
    }
    check(cudaMemcpyAsync(mask.data(),context.mask(),80,cudaMemcpyDeviceToHost,read.s));check(cudaStreamSynchronize(read.s));
    require(std::all_of(mask.begin(),mask.end(),[](auto v){return v==1;}),"snapshot_complete_window");
    require(!second.emit(accept,e),"snapshot_duplicate_emit");
    struct Boundary {int64_t first,take;int callbacks=0;std::vector<float> values;};
    for(int n:{1,80,81}){
        stage(first,a.s,10,n);Boundary boundary{10+n-std::min(n,80),std::min(n,80)};
        boundary.values.resize(static_cast<size_t>(boundary.take)*width);
        const core::EarlyActivationConsumer edge{&boundary,
            [](void* raw,const core::EarlyActivationRows& r,std::string& err){
                auto& b=*static_cast<Boundary*>(raw);
                if(r.pos0!=b.first || r.count!=b.take || r.width!=width){err="snapshot_edge_descriptor";return false;}
                ++b.callbacks;
                check(cudaMemcpyAsync(b.values.data(),r.device,b.values.size()*4,cudaMemcpyDeviceToHost,static_cast<cudaStream_t>(r.stream)));
                check(cudaStreamSynchronize(static_cast<cudaStream_t>(r.stream)));
                const int bias=r.tap==core::EarlyActivationTap::FirstAttention?1000:0;
                for(int64_t t=0;t<r.count;++t)for(int j=0;j<width;++j)
                    require(b.values[t*width+j]==float(b.first+t+bias)+float(j%3)*.25f,"snapshot_edge_values");
                return true;
            }};
        require(first.emit(edge,e) && boundary.callbacks==2,"snapshot_edge_publish");
    }
    require(!first.stage(core::EarlyActivationTap::Embedding,source.p,0,0,a.s,e),"snapshot_empty_source");
    require(!first.stage(core::EarlyActivationTap::Embedding,source.p,262143,2,a.s,e),"snapshot_source_bounds");
    require(!first.stage(core::EarlyActivationTap::Embedding,source.p,0,count,b.s,e),"snapshot_wrong_stream");
    stage(first,a.s,0,count);require(!first.emit(consumer,e) && callbacks==1,"snapshot_failed_consumer");
    require(!first.emit(accept,e),"snapshot_failed_publish_reused");
    require(first.stage(core::EarlyActivationTap::Embedding,source.p,0,count,a.s,e),e);
    require(!first.stage(core::EarlyActivationTap::FirstAttention,source.p,1,count,a.s,e),"snapshot_mixed_chunk");
    require(!first.emit(accept,e),"snapshot_mixed_chunk_published");
    std::cout<<"PASS per-lane bounded tails, cross-stream context, source reuse, bounds and failed/duplicate publication\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
