#pragma once
#include "strata/core/igc_context.hpp"
#include "strata/core/igc_pipeline.hpp"
#include "strata/core/qwenputer_binding.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/native_head.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/program/igc_coherent_observation.hpp"
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
    static constexpr int64_t kThinkClose=248069; // packs/gyro-m/tokenizer/vocab.json `</think>`
    strata::igc::ControllerBundle bundle_;
    strata::qwenputer::VerifiedInputs inputs_;
    std::unique_ptr<strata::igc::ContextWindow> context_;
    std::unique_ptr<strata::igc::LatentPipeline> pipeline_;
    std::unique_ptr<igc_coherent::Writer> observation_;
    std::vector<int64_t> observed_prefix_;
    igc_coherent::Provenance provenance_;
    std::array<void*,2> producer_streams_{};
    float *observation_snapshot_=nullptr,*observation_logits_=nullptr;
    void* observation_q8_=nullptr;
    cudaStream_t stream_=nullptr;
    uint64_t invocation_=0,anchor_=0;
    bool loaded_=false,active_=false,staged_=false,invocation_known_=false,invoked_=false,bridge_=false;
    std::vector<int32_t> bridge_tokens_;std::size_t bridge_at_=0;
    bool note_stream(void* stream,std::string& error) {
        if(!observation_ || stream==stream_)return true;
        for(size_t i=0;i<producer_streams_.size();++i)if(producer_streams_[i]==stream || !producer_streams_[i]) {
            producer_streams_[i]=stream;provenance_.streams|=2u<<i;return true;
        }
        error="coherent_unknown_producer_stream";return false;
    }

    bool prepare(void* raw,std::string& error) {
        strata::timeline::Span span("igc.controller.prepare");
        if(!loaded_ || !inputs_.proof() || context_ || pipeline_ || !raw){error="igc learned prepare/binding";return false;}
        stream_=static_cast<cudaStream_t>(raw);
        context_=std::make_unique<strata::igc::ContextWindow>();
        if(observation_) {
            if(!context_->allocate(raw,error))return false;
            if(cudaMalloc(&observation_snapshot_,2560*4)!=cudaSuccess
                    || cudaMalloc(&observation_logits_,248320*4)!=cudaSuccess
                    || cudaMalloc(&observation_q8_,strata::kernels::native_q8_1_bytes(2560))!=cudaSuccess){error="coherent_fidelity_buffers";return false;}
            return true;
        }
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
    bool observe_only(const std::filesystem::path& plan,const std::filesystem::path& root,std::string& error) {
        if(!inputs_.proof() || context_ || observation_){error="coherent_configure_lifetime";return false;}
        auto writer=std::make_unique<igc_coherent::Writer>();
        if(!writer->open(plan,root,inputs_.proof()->model,inputs_.proof()->tokenizer,error))return false;
        observation_=std::move(writer);return true;
    }
    bool observing()const{return bool(observation_);}
    strata::core::Verifier::DiagnosticHeadTransform hook() {
        return {this,
            [](void* self,void* stream,std::string& error){return static_cast<IgcLearnedDiagnostic*>(self)->prepare(stream,error);},
            [](void* self,float* mixed,int count,void* stream,std::string& error){
                auto* p=static_cast<IgcLearnedDiagnostic*>(self);
                // Stable pre-hook snapshot node, even before begin/stage and
                // during prompt graph capture. Never mutate the model input.
                if(p->observation_){
                    if(stream!=p->stream_ || !mixed || count<1 || count>64){error="coherent_head_arguments";return false;}
                    const auto status=cudaMemcpyAsync(p->observation_snapshot_,mixed,2560*4,cudaMemcpyDeviceToDevice,p->stream_);
                    if(status!=cudaSuccess){error=cudaGetErrorString(status);return false;}error.clear();return true;
                }
                return p->pipeline_->head(mixed,static_cast<uint32_t>(count),stream,true,error);},
            [](void* self){static_cast<IgcLearnedDiagnostic*>(self)->release();}};
    }
    // Caller has waited for the previous commit and verified the retained live
    // prefix lineage. Clear the prior answer before ANY prompt head can run.
    bool begin(std::span<const int64_t> prefix,int64_t reused,bool may_reuse,bool eligible,std::string& error) {
        strata::timeline::Span span("igc.controller.begin",static_cast<int64_t>(prefix.size()),eligible);
        if(!context_ || (!pipeline_ && !observation_) || prefix.empty() || prefix.size()>strata::igc::ContextWindow::max_tokens
                || invocation_==std::numeric_limits<uint64_t>::max()) {error="igc learned begin arguments";return false;}
        if(observation_ && !eligible){observation_->cancel();error="coherent_ineligible_request";return false;}
        if(observation_ && !observation_->begin(prefix,error))return false;
        ++invocation_;anchor_=prefix.size()-1;active_=eligible;staged_=false;
        if(observation_) {
            observed_prefix_.assign(prefix.begin(),prefix.end());producer_streams_.fill(nullptr);
            provenance_={};provenance_.reused=static_cast<uint64_t>(reused);provenance_.accepted=provenance_.reused;
            provenance_.previous_owner=observation_->previous_owner();provenance_.fence=invocation_;
            provenance_.may_reuse=may_reuse;provenance_.cache=reused?(may_reuse?1:2):0;
        }
        if(pipeline_ && !pipeline_->stage_inputs(nullptr,nullptr,nullptr,invocation_,anchor_,false,stream_,true,error))return false;
        if(!context_->begin(prefix,reused,may_reuse && eligible,error))return false;
        if(!eligible)context_->invalidate();
        return true;
    }
    // Both taps are borrowed from this completed verifier window. Accept only
    // its committed INPUT prefix, not the emitted token or rejected MTP suffix.
    bool capture(strata::core::Verifier& verifier,int64_t pos0,std::span<const int64_t> accepted,std::string& error) {
        if(!active_)return true;
        if(!context_ || accepted.empty() || accepted.size()>64){error="igc learned accepted window";return false;}
        struct Capture {IgcLearnedDiagnostic* owner;strata::igc::ContextWindow* context;int64_t pos0,count;};
        Capture capture{this,context_.get(),pos0,static_cast<int64_t>(accepted.size())};
        const strata::core::EarlyActivationConsumer consumer{&capture,
            [](void* raw,const strata::core::EarlyActivationRows& rows,std::string& e) {
                auto& c=*static_cast<Capture*>(raw);
                if(rows.pos0!=c.pos0 || rows.count!=c.count || rows.width!=strata::igc::ContextWindow::width){e="igc learned source rows";return false;}
                const auto tap=rows.tap==strata::core::EarlyActivationTap::Embedding
                    ?strata::igc::ContextTap::Embedding:strata::igc::ContextTap::FirstAttention;
                return c.owner->note_stream(rows.stream,e) && c.context->observe(tap,rows.pos0,rows.count,rows.device,rows.stream,e);
            }};
        const bool ok=verifier.visit_early_activations(static_cast<int>(accepted.size()),consumer,error)
            && context_->accept(pos0,accepted,error);
        if(ok && observation_)provenance_.accepted=static_cast<uint64_t>(pos0+accepted.size());
        return ok;
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
            return self.note_stream(rows.stream,error) && self.context_->observe(tap,rows.pos0,rows.count,rows.device,rows.stream,error);
        }};
    }
    // Missing rows remain unavailable; never fabricate features from final R/KV.
    bool accept_prefill(int64_t pos0,std::span<const int64_t> accepted,std::string& error) {
        if(!active_)return true;
        const bool ok=context_->accept(pos0,accepted,error);
        if(ok && observation_)provenance_.accepted=static_cast<uint64_t>(pos0+accepted.size());
        return ok;
    }
    bool stage(std::string& error) {
        strata::timeline::Span span("igc.controller.stage",active_,static_cast<int64_t>(anchor_));
        if(!context_ || (!pipeline_ && !observation_) || !invocation_ || staged_){error="igc learned stage lifetime";return false;}
        if(observation_) {
            const auto* record=observation_->current();
            if(!record){error="coherent_stage_plan";return false;}
            // Gather before run/commit overwrites the oldest ring slot. Output
            // rows need only their owned final state, not a fabricated tail.
            if(!record->role && !context_->gather(static_cast<int64_t>(anchor_),stream_,error)) {
                if(error.empty())error="coherent_missing_cached_features";return false;
            }
            staged_=true;error.clear();return true;
        }
        bool available=false;
        if(active_) {available=context_->gather(static_cast<int64_t>(anchor_),stream_,error);if(!error.empty())return false;}
        if(!pipeline_->stage_inputs(context_->states(strata::igc::ContextTap::Embedding),
                context_->states(strata::igc::ContextTap::FirstAttention),context_->mask(),
                invocation_,anchor_,available,stream_,true,error))return false;
        staged_=true;invocation_known_=false;return true;
    }
    // Only after wait_commit, before another run. The diagnostic head above is
    // hard no-op, so copy_head_input is the genuine pre-injection mixed row.
    bool prove_head(int type,const void* weights,std::span<const float> expected,std::string& error) {
        if(!observation_ || type!=8 || !weights || expected.size()!=248320 || !observation_snapshot_){error="coherent_head_fidelity_arguments";return false;}
        try {
            strata::kernels::native_quantize_q8_1(observation_snapshot_,observation_q8_,2560,1,stream_);
            strata::kernels::native_mmvq(type,weights,observation_q8_,observation_logits_,2560,248320,1,stream_);
            std::vector<float> replay(248320);
            if(cudaMemcpyAsync(replay.data(),observation_logits_,replay.size()*4,cudaMemcpyDeviceToHost,stream_)!=cudaSuccess
                    || cudaStreamSynchronize(stream_)!=cudaSuccess || std::memcmp(replay.data(),expected.data(),expected.size_bytes())) {
                error="coherent_head_fidelity_mismatch";return false;
            }
            error.clear();return true;
        }catch(const std::exception& e){error=e.what();return false;}
    }
    template<class CopyFinal> bool export_observation(bool committed,CopyFinal&& copy_final,std::string& error,const strata::core::NativeHead* head=nullptr) {
        strata::timeline::Span span("igc.coherent.publish");
        if(!observation_ || !staged_ || !active_ || !committed || !observation_->current()) {
            if(observation_)observation_->cancel();error="coherent_incomplete_request";return false;
        }
        const bool anchor=observation_->current()->role==0;
        std::vector<float> embedding(anchor?80*4*2560:0),attention(embedding.size()),mixed(2560),logits(248320);
        std::array<uint8_t,80> mask{};
        if(anchor) {
            if(cudaMemcpyAsync(mask.data(),context_->mask(),80,cudaMemcpyDeviceToHost,stream_)!=cudaSuccess
                    || cudaMemcpyAsync(embedding.data(),context_->states(strata::igc::ContextTap::Embedding),embedding.size()*4,cudaMemcpyDeviceToHost,stream_)!=cudaSuccess
                    || cudaMemcpyAsync(attention.data(),context_->states(strata::igc::ContextTap::FirstAttention),attention.size()*4,cudaMemcpyDeviceToHost,stream_)!=cudaSuccess
                    || cudaStreamSynchronize(stream_)!=cudaSuccess){observation_->cancel();error="coherent_copy_tails";return false;}
            // Unobserved padding is not an archive of a previous request.
            for(size_t j=0;j<80;++j)if(!mask[j]) {
                std::fill_n(embedding.data()+j*4*2560,4*2560,0.0f);
                std::fill_n(attention.data()+j*4*2560,4*2560,0.0f);
            }
        }
        if(!copy_final(std::span<float>(mixed),std::span<float>(logits))) {
            observation_->cancel();error="coherent_copy_final";return false;
        }
        std::array<float,2560> prehook{};
        if(cudaMemcpyAsync(prehook.data(),observation_snapshot_,sizeof(prehook),cudaMemcpyDeviceToHost,stream_)!=cudaSuccess
                || cudaStreamSynchronize(stream_)!=cudaSuccess || std::memcmp(prehook.data(),mixed.data(),sizeof(prehook))) {
            observation_->cancel();error="coherent_prehook_fidelity_mismatch";return false;
        }
        provenance_.source=head?1u:0u;
        if(head && observation_->ordinal()<4) {
            if(!prove_head(head->type(),head->weights(),logits,error)){observation_->cancel();return false;}
            provenance_.head_fidelity=1;
        }
        const bool ok=observation_->publish(invocation_,observed_prefix_,embedding,attention,mask,mixed,logits,provenance_,true,error);
        active_=false;return ok;
    }
    bool finish_observation(strata::core::Verifier& verifier,const strata::core::NativeHead& head,bool committed,std::string& error) {
        return export_observation(committed,[&](std::span<float> mixed,std::span<float> logits) {
            return verifier.copy_head_input(0,nullptr,mixed.data()) && verifier.copy_logits(0,logits.data());
        },error,&head);
    }
    // The model copies its own text after the thinking closes, so injection stops there.
    // One token a round while a result is delivered: a longer window would run the
    // heads past `</think>` before the host can end injection. Reads back once a request.
    bool delivering(std::string& error) {
        if(!pipeline_ || observation_ || !staged_)return false;
        if(!invocation_known_) {
            strata::igc::Invocation record;uint32_t status=0;
            if(!pipeline_->audit(record,status,stream_,error))return false;
            invoked_=record.evaluations>0 && record.result.status==strata::igc::Status::Success;invocation_known_=true;
            bridge_tokens_.clear();bridge_at_=0;
            if(bridge_ && invoked_)bridge_tokens_=result_tokens(record.result.value);
        }
        if(bridge_)return bridge_at_<bridge_tokens_.size();
        return invoked_ && !pipeline_->output_ended();
    }
    // Token-result bridge: the executed value, as the answer's own tokens (packs/gyro-m).
    static std::vector<int32_t> result_tokens(const strata::igc::Integer& v) {
        std::vector<int32_t> digits;
        for(int i=v.length-1;i>=0;--i)digits.push_back(15+v.digits[i]);
        std::vector<int32_t> out{760,1067,369,v.negative?471:220};
        out.insert(out.end(),digits.begin(),digits.end());
        out.insert(out.end(),{13,198,248069,271});
        if(v.negative)out.push_back(12);
        out.insert(out.end(),digits.begin(),digits.end());
        out.push_back(248046);
        return out;
    }
    void set_bridge(bool on){bridge_=on;}
    // One-token rounds only: replaces the sampled token while a bridged result is delivered.
    void bridge_sample(int32_t& token) {
        if(bridge_ && bridge_at_<bridge_tokens_.size())token=bridge_tokens_[bridge_at_++];
    }
    bool end_output_after(std::span<const int64_t> accepted,std::string& error) {
        if(!pipeline_ || observation_ || std::find(accepted.begin(),accepted.end(),kThinkClose)==accepted.end())return true;
        return pipeline_->end_output(stream_,error);
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
        if(!context_ || (!pipeline_ && !observation_) || !invocation_){error="igc learned cancel lifetime";return false;}
        context_->invalidate();active_=false;staged_=true;
        if(observation_){observation_->cancel();error.clear();return true;}
        return pipeline_->stage_inputs(nullptr,nullptr,nullptr,invocation_,anchor_,false,stream_,true,error);
    }
    std::size_t device_bytes()const {
        return (context_?context_->device_bytes():0)+(pipeline_?pipeline_->device_bytes():0)
            +(observation_snapshot_?2560*4:0)+(observation_logits_?248320*4:0)
            +(observation_q8_?strata::kernels::native_q8_1_bytes(2560):0);
    }
    void release() {
        strata::timeline::Span span("igc.controller.release");
        if(stream_)cudaStreamSynchronize(stream_);
        if(observation_snapshot_)cudaFree(observation_snapshot_);
        observation_snapshot_=nullptr;
        if(observation_logits_)cudaFree(observation_logits_);
        observation_logits_=nullptr;
        if(observation_q8_)cudaFree(observation_q8_);
        observation_q8_=nullptr;
        pipeline_.reset();context_.reset();stream_=nullptr;active_=false;staged_=true;
    }
};
}
