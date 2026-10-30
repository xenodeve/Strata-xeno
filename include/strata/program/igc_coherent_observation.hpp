#pragma once
// Diagnostic input-only plan and bounded staging writer. No model/executor calls.
#include "strata/core/qwenputer_artifact.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace strata::program::igc_coherent {
using strata::qwenputer::Digest;
inline constexpr uint32_t width=4*2560,rows=80,vocab=248320;
struct Record {uint32_t role=0,base=0,step=0;std::vector<int64_t> tokens;};
struct Plan {Digest model{},tokenizer{},digest{};std::vector<Record> records;};
struct Provenance {
    uint64_t reused=0,accepted=0,previous_owner=0,fence=0;
    uint32_t may_reuse=0,cache=0,streams=1,flags=31,source=0,head_fidelity=0;
};
inline void put32(std::vector<uint8_t>& out,uint32_t v){for(int j=0;j<4;++j)out.push_back(static_cast<uint8_t>(v>>(j*8)));}
inline void put64(std::vector<uint8_t>& out,uint64_t v){for(int j=0;j<8;++j)out.push_back(static_cast<uint8_t>(v>>(j*8)));}
inline bool fail(std::string& e,const char* why){e=why;return false;}
inline bool parse_plan(std::span<const uint8_t> raw,Plan& out,std::string& e){
    out={};
    if(raw.size()<112 || raw.size()>3*1024*1024 || std::memcmp(raw.data(),"IGCCPL02",8))return fail(e,"coherent_plan_header");
    const auto hash=strata::qwenputer::sha256(raw.first(raw.size()-32));
    if(!std::equal(hash.begin(),hash.end(),raw.end()-32))return fail(e,"coherent_plan_digest");
    size_t at=8;
    auto get=[&](){uint32_t v=0;for(int j=0;j<4;++j)v|=uint32_t(raw[at++])<<(j*8);return v;};
    if(get()!=2)return fail(e,"coherent_plan_version");
    const auto count=get();if(!count || count>2048)return fail(e,"coherent_plan_count");
    Plan next;next.digest=hash;
    std::copy_n(raw.begin()+16,32,next.model.begin());std::copy_n(raw.begin()+48,32,next.tokenizer.begin());at=80;
    bool output=false;std::vector<uint32_t> steps(count,0);std::vector<size_t> previous(count,0);
    for(uint32_t i=0;i<count;++i){
        if(at+16>raw.size()-32)return fail(e,"coherent_plan_truncated");
        Record r;r.role=get();r.base=get();r.step=get();const auto n=get();
        if(r.role>1 || !n || n>256 || at+size_t(n)*4>raw.size()-32)return fail(e,"coherent_plan_record");
        for(uint32_t j=0;j<n;++j){auto t=get();if(t>=vocab)return fail(e,"coherent_plan_token");r.tokens.push_back(t);}
        if(!r.role){if(output || r.base!=i || r.step)return fail(e,"coherent_anchor_order");}
        else {
            output=true;
            if(r.base>=i || next.records[r.base].role || !r.step || r.step>31 || r.step!=steps[r.base]+1)
                return fail(e,"coherent_output_parent");
            const auto& base=next.records[r.base].tokens;
            if(r.tokens.size()!=base.size()+r.step || !std::equal(base.begin(),base.end(),r.tokens.begin()))return fail(e,"coherent_output_prefix");
            if(r.step>1){const auto& p=next.records[previous[r.base]].tokens;
                if(!std::equal(p.begin(),p.end(),r.tokens.begin()))return fail(e,"coherent_output_sequence");}
            steps[r.base]=r.step;previous[r.base]=i;
        }
        next.records.push_back(std::move(r));
    }
    if(at!=raw.size()-32)return fail(e,"coherent_plan_trailing");
    out=std::move(next);e.clear();return true;
}
inline bool load_plan(const std::filesystem::path& path,Plan& out,std::string& e)try{
    if(!std::filesystem::is_regular_file(std::filesystem::symlink_status(path)))return fail(e,"coherent_plan_file");
    const auto n=std::filesystem::file_size(path);if(n>3*1024*1024)return fail(e,"coherent_plan_file_size");
    std::vector<uint8_t> raw(static_cast<size_t>(n));std::ifstream f(path,std::ios::binary);
    if(!f.read(reinterpret_cast<char*>(raw.data()),static_cast<std::streamsize>(n)) || f.peek()!=std::char_traits<char>::eof())return fail(e,"coherent_plan_read");
    return parse_plan(raw,out,e);
}catch(const std::exception& x){e=x.what();return false;}

class Writer {
    Plan plan_;std::filesystem::path root_;size_t next_=0;uint64_t last_owner_=0;bool ready_=false,bad_=false;
public:
    bool open(const std::filesystem::path& plan,const std::filesystem::path& root,const Digest& model,const Digest& tokenizer,std::string& e)try{
        if(ready_ || !load_plan(plan,plan_,e))return false;
        if(plan_.model!=model || plan_.tokenizer!=tokenizer)return fail(e,"coherent_plan_binding");
        // A partial/previous archive is deliberately not resumable in v2. Never
        // overwrite old evidence or infer ownership from a filename/index alone.
        if(!std::filesystem::create_directory(root))return fail(e,"coherent_archive_exists_or_partial");
        root_=root;ready_=true;e.clear();return true;
    }catch(const std::exception& x){e=x.what();return false;}
    const Record* current()const{return ready_ && !bad_ && next_<plan_.records.size()?&plan_.records[next_]:nullptr;}
    size_t ordinal()const{return next_;}
    uint64_t previous_owner()const{return last_owner_;}
    bool begin(std::span<const int64_t> tokens,std::string& e){
        const auto* r=current();
        if(!r || r->tokens.size()!=tokens.size() || !std::equal(tokens.begin(),tokens.end(),r->tokens.begin())){bad_=true;return fail(e,"coherent_request_order_or_prefix");}
        e.clear();return true;
    }
    void cancel(){bad_=true;}
    bool publish(uint64_t owner,std::span<const int64_t> tokens,std::span<const float> embedding,
            std::span<const float> attention,std::span<const uint8_t> mask,std::span<const float> mixed,
            std::span<const float> logits,const Provenance& provenance,bool committed,std::string& e)try{
        if(!committed || owner<=last_owner_ || !begin(tokens,e)){bad_=true;return fail(e,"coherent_uncommitted_or_stale_owner");}
        const auto& r=*current();const uint32_t n=r.role?0:static_cast<uint32_t>(std::min<size_t>(tokens.size()-1,rows));
        if(provenance.reused>=tokens.size() || provenance.accepted!=tokens.size() || provenance.previous_owner!=last_owner_
                || provenance.fence!=owner || provenance.may_reuse>1 || provenance.cache!=(provenance.reused?(provenance.may_reuse?1u:2u):0u)
                || !(provenance.streams&1) || provenance.streams>7 || provenance.flags!=31 || provenance.source>1 || provenance.head_fidelity>1){
            bad_=true;return fail(e,"coherent_capture_provenance");}
        if(provenance.may_reuse && provenance.reused && (!next_ || provenance.reused>plan_.records[next_-1].tokens.size()
                || !std::equal(tokens.begin(),tokens.begin()+provenance.reused,plan_.records[next_-1].tokens.begin()))){
            bad_=true;return fail(e,"coherent_capture_prefix_lineage");}
        if(mixed.size()!=2560 || logits.size()!=vocab || mask.size()!=rows
                || embedding.size()!=(r.role?0:rows*width) || attention.size()!=embedding.size()){
            bad_=true;return fail(e,"coherent_capture_geometry");}
        for(uint32_t j=0;j<rows;++j)if(mask[j]!=(n && j>=rows-n)){bad_=true;return fail(e,"coherent_capture_missing_mask");}
        auto finite=[](std::span<const float> p){return std::all_of(p.begin(),p.end(),[](float v){return std::isfinite(v);});};
        if(!finite(embedding) || !finite(attention) || !finite(mixed) || !finite(logits)){bad_=true;return fail(e,"coherent_capture_nonfinite");}
        std::vector<uint8_t> raw;raw.reserve(8*1024*1024);
        raw.insert(raw.end(),{'I','G','C','C','O','H','0','3'});
        for(auto v:{uint32_t{3},r.role,static_cast<uint32_t>(next_),r.base,r.step,static_cast<uint32_t>(tokens.size()),n})put32(raw,v);
        put64(raw,owner);
        for(const auto& d:{plan_.model,plan_.tokenizer,plan_.digest})raw.insert(raw.end(),d.begin(),d.end());
        for(auto v:{provenance.reused,provenance.accepted,provenance.previous_owner,provenance.fence})put64(raw,v);
        for(auto v:{provenance.may_reuse,provenance.cache,provenance.streams,provenance.flags,provenance.source,provenance.head_fidelity})put32(raw,v);
        for(auto t:tokens)put32(raw,static_cast<uint32_t>(t));raw.insert(raw.end(),mask.begin(),mask.end());
        auto append=[&](std::span<const float> p){const auto* a=reinterpret_cast<const uint8_t*>(p.data());if(!p.empty())raw.insert(raw.end(),a,a+p.size_bytes());};
        append(embedding);append(attention);append(mixed);append(logits);
        const auto hash=strata::qwenputer::sha256(raw);raw.insert(raw.end(),hash.begin(),hash.end());
        const auto name=std::to_string(next_);auto pending=root_/(name+".pending"),done=root_/(name+".bin");
        if(std::filesystem::exists(pending) || std::filesystem::exists(done)){bad_=true;return fail(e,"coherent_capture_exists");}
        std::ofstream f(pending,std::ios::binary);f.write(reinterpret_cast<const char*>(raw.data()),static_cast<std::streamsize>(raw.size()));f.close();
        if(!f){bad_=true;return fail(e,"coherent_capture_write");}
        std::filesystem::rename(pending,done);last_owner_=owner;++next_;e.clear();return true;
    }catch(const std::exception& x){bad_=true;e=x.what();return false;}
};
} // namespace strata::program::igc_coherent
