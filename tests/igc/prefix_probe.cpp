// Offline contextual-feature extractor, not a shortened language model.
// Keep full geometry and run the engine's genuine [0,1) verifier stage.
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/igc_context.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/program/igc_observation.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace fs=std::filesystem;
using namespace strata::core;
using Capture=strata::program::igc_observation::Capture;
static void require(bool ok,const std::string& message) {if (!ok) throw std::runtime_error(message);}
static void check(cudaError_t e) {require(e==cudaSuccess,cudaGetErrorString(e));}
static std::vector<std::vector<int32_t>> read_input(const fs::path& path) {
    static_assert(std::endian::native==std::endian::little);
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    require(file.good() && file.tellg()>=12 && file.tellg()<=66*1024,"prefix_input_bytes");
    const auto bytes=static_cast<std::size_t>(file.tellg());
    std::vector<char> data(bytes);file.seekg(0);file.read(data.data(),static_cast<std::streamsize>(bytes));
    require(file.good() && !std::memcmp(data.data(),"IGCPFX01",8),"prefix_input_magic");
    std::size_t offset=8;
    auto get=[&]() {require(offset+4<=bytes,"prefix_input_short");uint32_t n;
        std::memcpy(&n,data.data()+offset,4);offset+=4;return n;};
    const auto n=get();require(n>=1 && n<=64,"prefix_input_count");
    std::vector<std::vector<int32_t>> rows;
    for (uint32_t i=0;i<n;++i) {
        const auto count=get();require(count>=2 && count<=256,"prefix_input_tokens");
        std::vector<int32_t> tokens;
        for (uint32_t j=0;j<count;++j) {const auto token=get();require(token<248320,"prefix_input_vocabulary");tokens.push_back(static_cast<int32_t>(token));}
        rows.push_back(std::move(tokens));
    }
    require(offset==bytes,"prefix_input_trailing");return rows;
}
struct Owned {
    ModelGeometry geometry;
    WeightTable weights;
    NativeDense dense;
    NativeEmbed embedding;
    FileExpertSource experts;
    SessionState session;
    std::unique_ptr<Verifier> verifier;
    cudaStream_t stream=nullptr;
    std::vector<void*> memory;
    bool session_ready=false;
    uint64_t device_bytes=0;
    void* allocate(uint64_t bytes) {void* p=nullptr;check(cudaMalloc(&p,bytes));memory.push_back(p);device_bytes+=bytes;return p;}
    ~Owned() {
        if (stream) cudaStreamSynchronize(stream);
        verifier.reset();
        set_native_embed(nullptr);
        if (session_ready) session_release(session);
        for (auto i=memory.rbegin();i!=memory.rend();++i) cudaFree(*i);
        if (stream) cudaStreamDestroy(stream);
    }
};
int main(int argc,char** argv) try {
    require(argc==6,"model pack prefixes.bin output-directory window(1..8)");
    const auto requests=read_input(argv[3]); // Refuse malformed data before CUDA/model access.
    require(std::strlen(argv[5])==1 && argv[5][0]>='1' && argv[5][0]<='8',"prefix_input_window");
    const int window=argv[5][0]-'0';
    const fs::path output(argv[4]);require(fs::is_directory(output),"prefix_output_directory");
    for (std::size_t i=0;i<requests.size();++i)
        require(!fs::exists(output/(std::to_string(i)+".attention.bin")),"prefix_output_exists");
    const char* ar=std::getenv("STRATA_VERIFY_ALL_RESIDENT");
    require(!ar || std::atoi(ar)!=0,"prefix_all_resident_required");
    Owned owner;std::string error;const auto& g=owner.geometry;
    check(cudaStreamCreateWithFlags(&owner.stream,cudaStreamNonBlocking));
    require(strata::kernels::cpu::expert_layout_load(argv[2],g.n_layers,g.n_expert,error),error);
    const auto& layout=strata::kernels::cpu::expert_layout();
    require(layout.native && layout.blob_bytes(0)>0,"prefix_native_expert_layout");
    std::set<std::string> skip;
    require(NativeDense::served_names({argv[1]},false,skip,error),error);
    // Existing loader remains the sole index/layout validator. This walk only
    // selects resident names, retaining every skipped row's metadata.
    std::ifstream index(fs::path(argv[2])/"index.txt");require(index.good(),"prefix_weight_index");
    std::string line;int lines=0;
    while (std::getline(index,line)) {
        require(++lines<=8192 && line.size()<=4096,"prefix_weight_index_bounds");
        if (line.empty() || line[0]=='#') continue;
        const auto name=line.substr(0,line.find(' '));
        if (!name.starts_with("blk.0.")) skip.insert(name);
    }
    require(index.eof(),"prefix_weight_index_read");
    uint64_t pool=0;require(WeightTable::pool_bytes(argv[2],pool,error,&skip),error);
    require(pool<=2ull*1024*1024*1024,"prefix_canonical_pool_bounds");
    require(owner.weights.load(argv[2],owner.allocate(std::max<uint64_t>(pool,256)),std::max<uint64_t>(pool,256),error,&skip),error);
    require(owner.dense.load({argv[1]},owner.weights,error,false,0,1),error);
    require(owner.embedding.load({argv[1]},g.n_embd,248320,error),error);
    set_native_embed(&owner.embedding);
    layer_set_native_bf16(true);layer_set_fused_gr(true);
    strata::kernels::native_gdn_set_enabled(true);
    strata::kernels::native_qsa_indexer_set_enabled(true);
    require(check_layer(owner.weights,g,0,error),error);
    owner.experts.set_gguf(argv[1]);
    require(owner.experts.open(argv[2],g.n_layers,g.n_expert,error),error);
    const uint64_t blob=layout.max_blob;
    require(blob>0 && blob<=16*1024*1024 && blob*g.n_expert<=4ull*1024*1024*1024,"prefix_expert_arena_bounds");
    auto* cache=static_cast<uint8_t*>(owner.allocate(blob*g.n_expert));
    std::vector<int32_t> resident(static_cast<std::size_t>(g.n_layers*g.n_expert),-1);
    for (int64_t e=0;e<g.n_expert;++e) {
        const uint8_t* source=owner.experts.blob(0,e);
        require(source!=nullptr,"prefix_expert_source_blob");
        // Synchronous upload because the source buffer is reused immediately.
        check(cudaMemcpy(cache+e*blob,source,static_cast<std::size_t>(layout.blob_bytes(0)),cudaMemcpyHostToDevice));
        resident[static_cast<std::size_t>(e)]=static_cast<int32_t>(e);
    }
    auto* table=static_cast<int32_t*>(owner.allocate(resident.size()*4));
    check(cudaMemcpy(table,resident.data(),resident.size()*4,cudaMemcpyHostToDevice));
    const uint64_t session_bytes=strata::core::session_bytes(g,256,10,0,1);
    require(session_bytes<=256ull*1024*1024,"prefix_session_bounds");
    session_init(g,256,10,owner.allocate(session_bytes),owner.session,0,1);owner.session_ready=true;
    auto* hand=static_cast<float*>(owner.allocate(8*Verifier::handoff_floats(g)*4));
    VerifyHits hits;hits.d_res=table;hits.h_res=resident.data();hits.cache_base=cache;
    hits.n_slots=g.n_expert;hits.blob=static_cast<int64_t>(blob);
    owner.verifier=std::make_unique<Verifier>();auto& verifier=*owner.verifier;
    verifier.set_stream(owner.stream);verifier.set_stage(0,1,nullptr,hand);
    require(verifier.enable_early_observation(true),"prefix_observation_configuration");
    require(verifier.init(owner.weights,g,owner.session,hits,nullptr,8,error),error);
    strata::igc::ContextWindow history;require(history.allocate(owner.stream,error),error);
    struct DeviceRows {
        float* data[2]{};cudaStream_t stream=nullptr;int64_t position=0;int count=0,seen=0;
        strata::igc::ContextWindow* history=nullptr;
    } copied;
    copied.stream=owner.stream;copied.history=&history;
    for(auto& p:copied.data)p=static_cast<float*>(owner.allocate(8*10240*4));
    EarlyActivationConsumer consumer{&copied,[](void* user,const EarlyActivationRows& row,std::string& error){
        auto& c=*static_cast<DeviceRows*>(user);const int kind=static_cast<int>(row.tap);
        if(kind<0 || kind>1 || row.count!=c.count || row.width!=10240 || row.pos0!=c.position || row.stream!=c.stream || !row.device){error="prefix_device_tap_descriptor";return false;}
        c.seen|=1<<kind;
        const auto status=cudaMemcpyAsync(c.data[kind],row.device,row.count*row.width*4,cudaMemcpyDeviceToDevice,c.stream);
        if(status!=cudaSuccess){error=cudaGetErrorString(status);return false;}
        return c.history->observe(static_cast<strata::igc::ContextTap>(kind),row.pos0,row.count,row.device,row.stream,error);
    }};
    std::vector<float> buffer(10240);
    for (std::size_t i=0;i<requests.size();++i) {
        const auto& tokens=requests[i];std::vector<int64_t> prefix(tokens.begin(),tokens.end());
        std::vector<float> first;
        for (int repetition=0;repetition<2;++repetition) {
            require(verifier.wait_commit(error),error);
            session_zero(owner.session,g,nullptr,owner.stream);check(cudaStreamSynchronize(owner.stream));
            require(history.begin(prefix,0,false,error),error);
            Capture capture;require(capture.begin(prefix,error),error);
            std::vector<float> current;
            for (int p=0;p<static_cast<int>(tokens.size())-1;) {
                const int n=std::min(window,static_cast<int>(tokens.size())-1-p);int32_t unused[8]{};
                require(verifier.run(n,tokens.data()+p,p,nullptr,nullptr,unused,error),error);
                copied.position=p;copied.count=n;copied.seen=0;
                require(verifier.visit_early_activations(n,consumer,error),"prefix_device_tap_visit");
                require(copied.seen==3,"prefix_device_both_taps");
                for(auto tap:{Verifier::ObservationTap::Embedding,Verifier::ObservationTap::FirstAttention}){
                    std::vector<float> device_copy(static_cast<size_t>(n)*10240),legacy(10240);
                    check(cudaMemcpyAsync(device_copy.data(),copied.data[static_cast<int>(tap)],device_copy.size()*4,cudaMemcpyDeviceToHost,owner.stream));
                    check(cudaStreamSynchronize(owner.stream));
                    for(int t=0;t<n;++t){require(verifier.copy_observation_tap(tap,t,legacy.data()),"prefix_legacy_tap");
                        require(!std::memcmp(legacy.data(),device_copy.data()+t*10240,10240*4),"prefix_device_vs_host_bitwise");}
                }
                require(!verifier.visit_early_activations(n+1,consumer,error),"prefix_rejected_suffix_bound");
                require(!verifier.visit_early_activations(0,consumer,error),"prefix_empty_retention_bound");
                if(n>1){copied.count=n-1;copied.seen=0;require(verifier.visit_early_activations(n-1,consumer,error),"prefix_clamped_tap_visit");require(copied.seen==3,"prefix_clamped_both_taps");}
                require(capture.observe(p,n,1,[&](int64_t row,std::span<float> values){
                    return verifier.copy_observation_tap(Verifier::ObservationTap::FirstAttention,static_cast<int>(row),values.data());},error),error);
                for (int t=0;t<n;++t) {
                    require(verifier.copy_observation_tap(Verifier::ObservationTap::FirstAttention,t,buffer.data()),"prefix_tap_copy");
                    current.insert(current.end(),buffer.begin(),buffer.end());
                }
                require(verifier.commit(n,error) && verifier.wait_commit(error),error);
                std::vector<int64_t> accepted(tokens.begin()+p,tokens.begin()+p+n);
                require(history.accept(p,accepted,error),error);p+=n;
            }
            require(history.gather(static_cast<int64_t>(tokens.size())-1,owner.stream,error),error);
            std::vector<float> ring(80ull*10240);std::array<uint8_t,80> mask{};
            check(cudaMemcpyAsync(ring.data(),history.states(strata::igc::ContextTap::FirstAttention),ring.size()*4,cudaMemcpyDeviceToHost,owner.stream));
            check(cudaMemcpyAsync(mask.data(),history.mask(),80,cudaMemcpyDeviceToHost,owner.stream));check(cudaStreamSynchronize(owner.stream));
            const size_t valid=std::min<size_t>(80,tokens.size()-1),pad=80-valid;
            for(size_t r=0;r<80;++r)require(mask[r]==(r>=pad),"prefix_ring_valid_mask");
            require(!std::memcmp(ring.data()+pad*10240,current.data()+current.size()-valid*10240,valid*10240*4),"prefix_ring_native_bitwise");
            if (repetition==0) first=std::move(current);
            else require(current.size()==first.size() && !std::memcmp(current.data(),first.data(),current.size()*sizeof(float)),"prefix_reset_repeat");
            if (repetition==1) require(capture.write((output/(std::to_string(i)+".attention.bin")).string(),error,Capture::Stage::FirstAttention),error);
        }
        std::cout<<"prefix request="<<i<<" tokens="<<tokens.size()<<" reset_repeat=bitwise\n"<<std::flush;
    }
    require(verifier.wait_commit(error),error);check(cudaStreamSynchronize(owner.stream));
    std::cout<<"PASS native prefix device_taps=bitwise context_ring=bitwise retained_input_bounds=pass requests="<<requests.size()<<" window="<<window
             <<" geometry_layers="<<g.n_layers<<" stage=[0,1) owned_device_bytes="<<owner.device_bytes
             <<" native_dense_bytes="<<owner.dense.weight_bytes()<<" embedding_mapped_bytes="<<owner.embedding.bytes()
             <<" context_device_bytes="<<history.device_bytes()<<'\n';
    return 0;
} catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
