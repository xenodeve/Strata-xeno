#pragma once
#include "strata/core/igc_context.hpp"
#include "strata/core/igc_pipeline.hpp"
#include "strata/core/qwenputer_binding.hpp"
#include "strata/core/verify.hpp"
#include "strata/timeline.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace strata::program {
// First-anchor learned diagnostic, distinct from the oracle-input comparator.
// Runtime receipts bind the actual loaded model/tokenizer before allocation.
class IgcLearnedDiagnostic {
    strata::igc::ControllerBundle bundle_;
    strata::qwenputer::VerifiedInputs inputs_;
    std::unique_ptr<strata::igc::ContextWindow> context_;
    std::unique_ptr<strata::igc::LatentPipeline> pipeline_;
    cudaStream_t stream_=nullptr;
    uint64_t invocation_=0,anchor_=0;
    bool loaded_=false,active_=false,staged_=false;

    bool prepare(void* raw,std::string& error) {
        strata::timeline::Span span("igc.controller.prepare");
        if(!loaded_ || !inputs_.proof() || pipeline_ || !raw){error="igc learned prepare/binding";return false;}
        stream_=static_cast<cudaStream_t>(raw);
        context_=std::make_unique<strata::igc::ContextWindow>();
        pipeline_=std::make_unique<strata::igc::LatentPipeline>();
        return context_->allocate(raw,error) && pipeline_->initialize(bundle_,raw,error);
    }
public:
    IgcLearnedDiagnostic()=default;
    ~IgcLearnedDiagnostic(){release();}
    IgcLearnedDiagnostic(const IgcLearnedDiagnostic&)=delete;
    IgcLearnedDiagnostic& operator=(const IgcLearnedDiagnostic&)=delete;

    bool load_host(const std::filesystem::path& path,std::string& error) {
        strata::timeline::Span span("igc.controller.load_host");
        if(loaded_){error="igc learned bundle already loaded";return false;}
        loaded_=strata::igc::load_controller_bundle(path,true,bundle_,error);
        return loaded_;
    }
    bool bind(const std::filesystem::path& tokenizer,const std::vector<std::filesystem::path>& models,std::string& error) {
        strata::timeline::Span span("igc.controller.bind",static_cast<int64_t>(models.size()));
        if(!loaded_){error="igc learned bind before load";return false;}
        strata::qwenputer::Binding expected;
        expected.digests[4]=bundle_.model;expected.digests[5]=bundle_.tokenizer;
        return inputs_.bind(expected,tokenizer,models,error);
    }
    strata::core::Verifier::DiagnosticHeadTransform hook() {
        return {this,
            [](void* self,void* stream,std::string& error){return static_cast<IgcLearnedDiagnostic*>(self)->prepare(stream,error);},
            [](void* self,float* mixed,int count,void* stream,std::string& error){
                auto* p=static_cast<IgcLearnedDiagnostic*>(self);
                return p->pipeline_->head(mixed,static_cast<uint32_t>(count),stream,true,error);},
            [](void* self){static_cast<IgcLearnedDiagnostic*>(self)->release();}};
    }
    // Caller has waited for the previous commit and verified the retained live
    // prefix lineage. Clear the prior answer before ANY prompt head can run.
    bool begin(std::span<const int64_t> prefix,int64_t reused,bool may_reuse,bool eligible,std::string& error) {
        strata::timeline::Span span("igc.controller.begin",static_cast<int64_t>(prefix.size()),eligible);
        if(!pipeline_ || prefix.empty() || prefix.size()>strata::igc::ContextWindow::max_tokens
                || invocation_==std::numeric_limits<uint64_t>::max()) {error="igc learned begin arguments";return false;}
        ++invocation_;anchor_=prefix.size()-1;active_=eligible;staged_=false;
        if(!pipeline_->stage_inputs(nullptr,nullptr,nullptr,invocation_,anchor_,false,stream_,true,error))return false;
        if(!context_->begin(prefix,reused,may_reuse && eligible,error))return false;
        if(!eligible)context_->invalidate();
        return true;
    }
    // Both taps are borrowed from this completed verifier window. Accept only
    // its committed INPUT prefix, not the emitted token or rejected MTP suffix.
    bool capture(strata::core::Verifier& verifier,int64_t pos0,std::span<const int64_t> accepted,std::string& error) {
        if(!active_)return true;
        if(!context_ || accepted.empty() || accepted.size()>64){error="igc learned accepted window";return false;}
        struct Capture {strata::igc::ContextWindow* context;int64_t pos0,count;};
        Capture capture{context_.get(),pos0,static_cast<int64_t>(accepted.size())};
        const strata::core::EarlyActivationConsumer consumer{&capture,
            [](void* raw,const strata::core::EarlyActivationRows& rows,std::string& e) {
                auto& c=*static_cast<Capture*>(raw);
                if(rows.pos0!=c.pos0 || rows.count!=c.count || rows.width!=strata::igc::ContextWindow::width){e="igc learned source rows";return false;}
                const auto tap=rows.tap==strata::core::EarlyActivationTap::Embedding
                    ?strata::igc::ContextTap::Embedding:strata::igc::ContextTap::FirstAttention;
                return c.context->observe(tap,rows.pos0,rows.count,rows.device,rows.stream,e);
            }};
        return verifier.visit_early_activations(static_cast<int>(accepted.size()),consumer,error)
            && context_->accept(pos0,accepted,error);
    }
    // Existing prefill wave tails serialize these callbacks. Copy on the
    // producer's stream; full segment input ownership is accepted after run().
    strata::core::EarlyActivationConsumer prefill_consumer() {
        return {this,[](void* raw,const strata::core::EarlyActivationRows& rows,std::string& error) {
            auto& self=*static_cast<IgcLearnedDiagnostic*>(raw);
            if(!self.active_)return true;
            if(!self.context_ || rows.width!=strata::igc::ContextWindow::width || rows.count<1 || rows.count>80){error="igc learned prefill rows";return false;}
            const auto tap=rows.tap==strata::core::EarlyActivationTap::Embedding
                ?strata::igc::ContextTap::Embedding:strata::igc::ContextTap::FirstAttention;
            return self.context_->observe(tap,rows.pos0,rows.count,rows.device,rows.stream,error);
        }};
    }
    // Missing rows remain unavailable; never fabricate features from final R/KV.
    bool accept_prefill(int64_t pos0,std::span<const int64_t> accepted,std::string& error) {
        return !active_ || context_->accept(pos0,accepted,error);
    }
    bool stage(std::string& error) {
        strata::timeline::Span span("igc.controller.stage",active_,static_cast<int64_t>(anchor_));
        if(!pipeline_ || !invocation_ || staged_){error="igc learned stage lifetime";return false;}
        bool available=false;
        if(active_) {available=context_->gather(static_cast<int64_t>(anchor_),stream_,error);if(!error.empty())return false;}
        if(!pipeline_->stage_inputs(context_->states(strata::igc::ContextTap::Embedding),
                context_->states(strata::igc::ContextTap::FirstAttention),context_->mask(),
                invocation_,anchor_,available,stream_,true,error))return false;
        staged_=true;return true;
    }
    bool audit(strata::igc::Invocation& record,uint32_t& status,std::string& error) {
        strata::timeline::Span span("igc.controller.audit");
        if(!pipeline_){error="igc learned audit before prepare";return false;}
        return pipeline_->audit(record,status,stream_,error);
    }
    // Outside capture after the owner's prior work; clearing context alone
    // cannot revoke the retained device result used by already captured heads.
    bool cancel(std::string& error) {
        strata::timeline::Span span("igc.controller.cancel");
        if(!pipeline_ || !invocation_){error="igc learned cancel lifetime";return false;}
        context_->invalidate();active_=false;staged_=true;
        return pipeline_->stage_inputs(nullptr,nullptr,nullptr,invocation_,anchor_,false,stream_,true,error);
    }
    std::size_t device_bytes()const {
        return (context_?context_->device_bytes():0)+(pipeline_?pipeline_->device_bytes():0);
    }
    void release() {
        strata::timeline::Span span("igc.controller.release");
        if(stream_)cudaStreamSynchronize(stream_);
        pipeline_.reset();context_.reset();stream_=nullptr;active_=false;staged_=true;
    }
};
}
