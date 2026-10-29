#pragma once
#include "strata/core/igc_output.hpp"
#include "strata/core/qwenputer_binding.hpp"
#include "strata/core/verify.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <vector>

namespace strata::program {
// Controlled-input I3 experiment only. The host matches a complete prefix hash;
// it never parses arithmetic or computes a result. No learned I2 claim.
class IgcOutputDiagnostic {
    struct Record {strata::igc::adapter::Digest prefix{};strata::igc::ExactResult result{};};
    strata::igc::adapter::Artifact artifact_;
    strata::igc::adapter::Binding binding_{};
    strata::qwenputer::VerifiedInputs inputs_;
    std::vector<Record> records_;
    std::unique_ptr<strata::igc::OutputMapper> mapper_;
    strata::igc::ExactResult* results_=nullptr;
    uint64_t *ids_=nullptr,*anchors_=nullptr;
    uint32_t* status_=nullptr;
    cudaStream_t stream_=nullptr;
    bool matched_=false;
    bool cuda_ok(cudaError_t value,std::string& error) {
        if(value==cudaSuccess)return true;error=cudaGetErrorString(value);return false;
    }
    bool prepare(void* stream,std::string& error) {
        if(mapper_ || !inputs_.proof()){error="igc diagnostic prepare/binding";return false;}
        stream_=static_cast<cudaStream_t>(stream);
        mapper_=std::make_unique<strata::igc::OutputMapper>();
        if(!mapper_->load(artifact_,stream,error))return false;
        if(!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&results_),64*sizeof(*results_)),error)
                || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&ids_),64*sizeof(*ids_)),error)
                || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&anchors_),64*sizeof(*anchors_)),error)
                || !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&status_),64*sizeof(*status_)),error))return false;
        return stage({},error);
    }
public:
    ~IgcOutputDiagnostic(){release();}
    IgcOutputDiagnostic()=default;
    IgcOutputDiagnostic(const IgcOutputDiagnostic&)=delete;
    IgcOutputDiagnostic& operator=(const IgcOutputDiagnostic&)=delete;
    bool load_host(const std::filesystem::path& artifact,const std::filesystem::path& fixture,std::string& error)try{
        if(!records_.empty() || !std::filesystem::is_regular_file(std::filesystem::symlink_status(fixture))){error="igc fixture file";return false;}
        const auto size=std::filesystem::file_size(fixture);
        if(size<228 || size>108+64*120){error="igc fixture bytes";return false;}
        std::vector<uint8_t> raw(static_cast<std::size_t>(size));std::ifstream file(fixture,std::ios::binary);
        file.read(reinterpret_cast<char*>(raw.data()),static_cast<std::streamsize>(raw.size()));
        uint32_t count=0;std::memcpy(&count,raw.data()+8,4);
        if(!file || file.peek()!=std::char_traits<char>::eof() || std::memcmp(raw.data(),"IGCOFX01",8)
                || !count || count>64 || size!=108+uint64_t(count)*120){error="igc fixture header";return false;}
        std::copy_n(raw.data()+12,32,binding_.model.begin());std::copy_n(raw.data()+44,32,binding_.tokenizer.begin());std::copy_n(raw.data()+76,32,binding_.file.begin());
        std::set<strata::igc::adapter::Digest> seen;
        for(uint32_t i=0;i<count;++i){
            Record record;std::memcpy(record.prefix.data(),raw.data()+108+i*120,32);
            std::memcpy(&record.result,raw.data()+140+i*120,88);
            const auto& r=record.result;const auto& v=r.value;
            bool valid=r.version==1 && r.status==strata::igc::Status::Success && r.invocation && r.position<262144
                && v.length>=1 && v.length<=62 && v.negative<=1;
            for(int j=0;j<62;++j)valid=valid && v.digits[j]<=9 && (j<v.length || !v.digits[j]);
            if(valid)valid=!(v.length>1 && !v.digits[v.length-1]) && !(v.length==1 && !v.digits[0] && v.negative);
            if(!valid || !seen.insert(record.prefix).second){error="igc fixture canonical result/duplicate";return false;}
            records_.push_back(record);
        }
        if(!strata::igc::adapter::load(artifact,binding_,true,artifact_,error) || artifact_.kind!=2){error="igc diagnostic artifact:"+error;return false;}
        return true;
    }catch(const std::exception& e){error=std::string("igc fixture:")+e.what();return false;}
    bool bind(const std::filesystem::path& tokenizer,const std::vector<std::filesystem::path>& models,std::string& error){
        strata::qwenputer::Binding expected;expected.digests[4]=binding_.model;expected.digests[5]=binding_.tokenizer;
        return inputs_.bind(expected,tokenizer,models,error);
    }
    std::size_t entries()const{return records_.size();}
    strata::core::Verifier::DiagnosticHeadTransform hook(){
        return {this,
            [](void* self,void* stream,std::string& error){return static_cast<IgcOutputDiagnostic*>(self)->prepare(stream,error);},
            [](void* self,float* mixed,int count,void* stream,std::string& error){
                auto* p=static_cast<IgcOutputDiagnostic*>(self);
                return p->mapper_->run(mixed,p->results_,p->ids_,p->anchors_,p->status_,static_cast<uint32_t>(count),stream,true,error);},
            [](void* self){static_cast<IgcOutputDiagnostic*>(self)->release();}};
    }
    bool select(const std::vector<int64_t>& prefix,strata::igc::ExactResult& chosen,bool& matched,std::string& error)const{
        chosen={};chosen.status=strata::igc::Status::Fault;chosen.invocation=1;matched=false;
        if(!prefix.empty()){
            if(prefix.size()>262144){error="igc diagnostic prefix bound";return false;}
            std::vector<int32_t> values;values.reserve(prefix.size());
            for(int64_t token:prefix){if(token<0 || token>=248320){error="igc diagnostic prefix token";return false;}values.push_back(static_cast<int32_t>(token));}
            auto hash=strata::qwenputer::sha256({reinterpret_cast<const uint8_t*>(values.data()),values.size()*4});
            for(const auto& record:records_)if(record.prefix==hash){
                if(record.result.position!=prefix.size()-1){error="igc fixture prefix/anchor mismatch";return false;}
                chosen=record.result;matched=true;break;
            }
        }
        return true;
    }
    bool stage(const std::vector<int64_t>& prefix,std::string& error){
        if(!stream_ || !results_ || !ids_ || !anchors_){error="igc diagnostic not prepared";return false;}
        strata::igc::ExactResult chosen;
        if(!select(prefix,chosen,matched_,error))return false;
        std::array<strata::igc::ExactResult,64> results;results.fill(chosen);
        std::array<uint64_t,64> ids,anchors;ids.fill(chosen.invocation);anchors.fill(chosen.position);
        // Diagnostic request-entry copies only, never a per-token answer bridge.
        return cuda_ok(cudaMemcpyAsync(results_,results.data(),sizeof(results),cudaMemcpyHostToDevice,stream_),error)
            && cuda_ok(cudaMemcpyAsync(ids_,ids.data(),sizeof(ids),cudaMemcpyHostToDevice,stream_),error)
            && cuda_ok(cudaMemcpyAsync(anchors_,anchors.data(),sizeof(anchors),cudaMemcpyHostToDevice,stream_),error)
            && cuda_ok(cudaStreamSynchronize(stream_),error);
    }
    bool check(int count,std::string& error){
        if(count<1 || count>64){error="igc diagnostic status bound";return false;}
        std::array<uint32_t,64> values{};
        if(!cuda_ok(cudaMemcpyAsync(values.data(),status_,count*4,cudaMemcpyDeviceToHost,stream_),error)
                || !cuda_ok(cudaStreamSynchronize(stream_),error))return false;
        for(int i=0;i<count;++i)if(values[i]!=(matched_?0u:1u)){error="igc diagnostic rejected output row";return false;}
        return true;
    }
    void release(){
        if(stream_)cudaStreamSynchronize(stream_);
        mapper_.reset();
        if(results_)cudaFree(results_);if(ids_)cudaFree(ids_);if(anchors_)cudaFree(anchors_);if(status_)cudaFree(status_);
        results_=nullptr;ids_=nullptr;anchors_=nullptr;status_=nullptr;stream_=nullptr;
    }
};
} // namespace strata::program
