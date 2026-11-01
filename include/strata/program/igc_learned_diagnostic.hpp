#pragma once
#include "strata/core/igc_inline.hpp"
#include "strata/core/igc_natural.hpp"
#include "strata/core/igc_sidecar.hpp"
#include "strata/core/igc_sandbox_bridge.hpp"
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
    static constexpr int32_t kEndOfTurn=248046;
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
    std::vector<int32_t> generated_;
    bool thinking_closed_=false;   // generated_ is trimmed, so it cannot answer this
    std::vector<int64_t> inline_seen_;
    std::vector<int32_t> thinking_tail_;
    std::unique_ptr<strata::igc::sandbox::TokenText> sandbox_text_;
    strata::igc::sandbox::Fence sandbox_fence_;
    strata::igc::sandbox::Session sandbox_session_;
    int sandbox_after_=0,sandbox_thought_=0,sandbox_block_=0,bridged_unread_=0;
    bool sandbox_opened_=false,sandbox_close_=false,sandbox_close_line_=false,sandbox_call_=false,sandbox_called_=false;
    bool inline_notes_=true,inline_reply_=true;int close_repeat_=3;strata::igc::inline_tokens::SumNote sum_note_=strata::igc::inline_tokens::SumNote::Stop;strata::igc::inline_tokens::RepeatNote repeat_note_=strata::igc::inline_tokens::RepeatNote::Stop;
    std::vector<int64_t> prefix_;
    bool inline_exec_=false;int64_t inline_calls_=0;
    std::unique_ptr<strata::igc::Sidecar> sidecar_;std::unique_ptr<strata::igc::NaturalFirer<strata::igc::Sidecar>> natural_;
    strata::igc::ExactRequest* inline_request_=nullptr;strata::igc::ExactResult* inline_result_=nullptr;
    strata::igc::rational_exact::Request* inline_rational_request_=nullptr;strata::igc::rational_exact::Result* inline_rational_result_=nullptr;
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
        if(!context_->allocate(raw,error) || !pipeline_->initialize(bundle_,raw,error))return false;
        pipeline_->set_output_bypass(bridge_);
        return true;
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
        prefix_.assign(prefix.begin(),prefix.end());
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
        staged_=true;invocation_known_=false;generated_.clear();inline_seen_.clear();thinking_tail_.clear();thinking_closed_=strata::igc::inline_tokens::thinking_off(prefix_,kThinkClose);
        sandbox_fence_=strata::igc::sandbox::Fence{};sandbox_session_=strata::igc::sandbox::Session{};sandbox_thought_=0;sandbox_block_=0;sandbox_opened_=false;sandbox_called_=false;bridged_unread_=0;
        if(natural_)natural_->begin(prefix_);
        return true;
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
            if(bridge_ && invoked_) {
                if(strata::igc::written_in(prefix_,record.request.lhs) && strata::igc::written_in(prefix_,record.request.rhs))
                    bridge_tokens_=result_tokens(record.result.value);
                else if(!pipeline_->end_output(stream_,error))return false;
            }
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
    static int64_t number_key(const strata::igc::Integer& v) {
        int64_t key=v.length;
        for(int i=0;i<v.length;++i)key=key*1000003+v.digits[i];
        return key;
    }
    void set_bridge(bool on){bridge_=on;}
    void set_inline_exec(bool on){inline_exec_=on;}
    void set_inline_notes(bool on){inline_notes_=on;}
    void set_inline_reply(bool on){inline_reply_=on;}
    void set_sandbox_after(int tokens){sandbox_after_=tokens;}
    void set_sandbox_close(bool on){sandbox_close_=on;}
    void set_sandbox_close_line(bool on){sandbox_close_line_=on;}
    void set_sandbox_call(bool on){sandbox_call_=on;}
    bool set_sandbox(const std::filesystem::path& vocab,std::string& error) {
        sandbox_text_=std::make_unique<strata::igc::sandbox::TokenText>();
        return sandbox_text_->load(vocab,error);
    }
    void set_close_repeat(int n){close_repeat_=n;}
    void set_sum_note(int mode){sum_note_=static_cast<strata::igc::inline_tokens::SumNote>(mode);}
    void set_repeat_note(int mode){repeat_note_=static_cast<strata::igc::inline_tokens::RepeatNote>(mode);}
    bool set_sidecar(const std::string& path,std::string& error) {
        sidecar_=std::make_unique<strata::igc::Sidecar>();
        if(!sidecar_->load(path,error))return false;
        natural_=std::make_unique<strata::igc::NaturalFirer<strata::igc::Sidecar>>(*sidecar_);
        return true;
    }
    int64_t inline_calls()const{return inline_calls_;}
    // After one accepted token: an expression that just closed runs on the device and its
    // value is delivered next. The device buffers live as long as the process.
    bool inline_token(int32_t& token,bool& called,std::string& error) {
        called=false;
        if(!inline_exec_ || !pipeline_ || observation_ || !staged_)return true;
        // An end of turn inside thinking would leave the reader no answer after a call.
        if(token==kEndOfTurn && (!inline_seen_.empty() || sandbox_session_.runs()>0)
                && !thinking_closed_) {
            token=198;bridge_tokens_={kThinkClose,271};bridge_at_=0;
        }
        if(natural_) {
            strata::timeline::Span span("igc.sidecar");
            const int32_t sampled=token;
            auto rest=natural_->step(token);
            if(token!=sampled && bridge_at_>=bridge_tokens_.size()) {
                std::fprintf(stderr,"strata igc sidecar: fired, %d became %d, %d digits follow, confidence %.3f\n",sampled,token,static_cast<int>(rest.size()),natural_->last_confidence());
                bridge_tokens_=std::move(rest);bridge_at_=0;called=true;
            }
        }
        thinking_closed_=thinking_closed_||token==kThinkClose;
        generated_.push_back(token);
        if(generated_.size()>256)generated_.erase(generated_.begin(),generated_.begin()+128);
        // A window can take the last of a delivery at once, so bridge_at_ alone would read those tokens as the model's.
        if(bridged_unread_>0) {
            --bridged_unread_;
            return true;
        }
        if(bridge_at_<bridge_tokens_.size() || (thinking_closed_ && !inline_reply_))return true;
        if(!thinking_closed_) {
            // A loop holds no `=` for the repeat rule below; left alone it runs to the end of the context.
            thinking_tail_.push_back(token);
            if(thinking_tail_.size()>16384)thinking_tail_.erase(thinking_tail_.begin(),thinking_tail_.begin()+4096);
            if(thinking_tail_.size()%16==0 && strata::igc::inline_tokens::in_cycle(thinking_tail_,4096,8)) {
                std::fprintf(stderr,"strata igc loop: closed the thinking\n");
                bridge_tokens_={198,kThinkClose,271};bridge_at_=0;called=true;return true;
            }
        }
        if(sandbox_text_ && !thinking_closed_) {
            auto program=sandbox_fence_.step(sandbox_text_->text(token));
            if(program) {
                // The same program again is re-checking, as with a repeated expression: a loop of whole blocks is far longer
                // than in_cycle reaches, 2567 tokens on one question, and it ran to the end of the context.
                if(close_repeat_>0 && sandbox_session_.seen(*program)+1>=close_repeat_) {
                    std::fprintf(stderr,"strata igc sandbox: closed the thinking on a repeated program\n");
                    bridge_tokens_={198,kThinkClose,271};bridge_at_=0;called=true;return true;
                }
                strata::timeline::Span span("igc.sandbox");
                const auto text=sandbox_session_.spent()?strata::igc::sandbox::output_block("[no more runs]"):sandbox_session_.run(*program);
                std::fprintf(stderr,"strata igc sandbox: run %d of %d bytes, %d bytes back, %llu steps so far\n",sandbox_session_.runs(),static_cast<int>(program->size()),static_cast<int>(text.size()),static_cast<unsigned long long>(sandbox_session_.steps()));
                bridge_tokens_=sandbox_text_->encode(text);bridge_at_=0;called=true;
                if(sandbox_call_ && !sandbox_called_ && sandbox_session_.quiet() && strata::igc::sandbox::binds_a_name(*program)) {
                    std::fprintf(stderr,"strata igc sandbox: opened a call\n");
                    sandbox_called_=true;sandbox_block_=0;sandbox_fence_.enter(strata::igc::sandbox::kCallHead);
                    const auto open=sandbox_text_->encode(strata::igc::sandbox::kCallBlock);
                    bridge_tokens_.insert(bridge_tokens_.end(),open.begin(),open.end());
                }
                // Given the answer the model still went on for thousands of tokens to prove it.
                if(sandbox_close_ && sandbox_session_.printed()) {
                    if(sandbox_close_line_) {
                        const auto line=sandbox_text_->encode(strata::igc::sandbox::kCloseLine);
                        bridge_tokens_.insert(bridge_tokens_.end(),line.begin(),line.end());
                    }
                    bridge_tokens_.insert(bridge_tokens_.end(),{kThinkClose,271});
                }
                return true;
            }
            if(sandbox_fence_.inside()) {   // code, not arithmetic
                if(++sandbox_block_<strata::igc::sandbox::kMaxBlockTokens || (token!=198 && token!=271))return true;
                std::fprintf(stderr,"strata igc sandbox: stopped a block at %d tokens\n",sandbox_block_);
                sandbox_fence_.leave();sandbox_block_=0;
                bridge_tokens_=sandbox_text_->encode("```"+strata::igc::sandbox::output_block("[too long, not run]"));bridge_at_=0;called=true;return true;
            }
            sandbox_block_=0;
            // Long reasoning is where a program pays: open a block at the next line end, once. Opening one every N tokens
            // cut off a derivation that was going well, and made an exact loop.
            if(sandbox_after_>0 && !sandbox_opened_ && ++sandbox_thought_>=sandbox_after_ && (token==198 || token==271)) {
                std::fprintf(stderr,"strata igc sandbox: opened a block after %d tokens\n",sandbox_thought_);
                sandbox_opened_=true;sandbox_fence_.enter();
                bridge_tokens_.assign(strata::igc::sandbox::kOpenBlock.begin(),strata::igc::sandbox::kOpenBlock.end());bridge_at_=0;called=true;return true;
            }
        }
        const auto call=strata::igc::parse_inline(generated_);
        if(!call)return true;
        if(!inline_request_ && (cudaMalloc(&inline_request_,sizeof(*inline_request_))!=cudaSuccess
                || cudaMalloc(&inline_result_,sizeof(*inline_result_))!=cudaSuccess)){error="igc inline device buffers";return false;}
        std::vector<int32_t> value;
        if(call->exact_integers()) {
            std::vector<strata::igc::Integer> stack;
            for(const auto& step:call->program) {
                if(step.number){stack.push_back(step.value.coefficient);continue;}
                if(stack.size()<2){error="igc inline program";return false;}
                strata::igc::ExactRequest request;request.operation=static_cast<strata::igc::Operation>(step.operation);request.invocation=1;
                request.rhs=stack.back();stack.pop_back();request.lhs=stack.back();stack.pop_back();
                strata::igc::ExactResult result;
                if(step.operation==strata::igc::InlineOperation::Power) {
                    namespace extended=strata::igc::integer_extended;
                    extended::Request power;power.operation=extended::Operation::Power;power.invocation=1;
                    power.lhs=request.lhs;power.rhs=request.rhs;
                    extended::Result raised;
                    // ABI2 records have ABI1's size, so the same device buffers carry them.
                    auto* device_request=reinterpret_cast<extended::Request*>(inline_request_);
                    auto* device_result=reinterpret_cast<extended::Result*>(inline_result_);
                    if(cudaMemcpyAsync(device_request,&power,sizeof(power),cudaMemcpyHostToDevice,stream_)!=cudaSuccess
                            || !extended::execute(device_request,device_result,1,stream_,error)
                            || cudaMemcpyAsync(&raised,device_result,sizeof(raised),cudaMemcpyDeviceToHost,stream_)!=cudaSuccess
                            || cudaStreamSynchronize(stream_)!=cudaSuccess){if(error.empty())error="igc inline execution";return false;}
                    if(raised.status!=extended::Status::Success)return true;
                    stack.push_back(raised.value);continue;
                }
                if(cudaMemcpyAsync(inline_request_,&request,sizeof(request),cudaMemcpyHostToDevice,stream_)!=cudaSuccess
                        || !strata::igc::execute_exact(inline_request_,inline_result_,1,stream_,error)
                        || cudaMemcpyAsync(&result,inline_result_,sizeof(result),cudaMemcpyDeviceToHost,stream_)!=cudaSuccess
                        || cudaStreamSynchronize(stream_)!=cudaSuccess){if(error.empty())error="igc inline execution";return false;}
                if(result.status!=strata::igc::Status::Success)return true;
                stack.push_back(result.value);
            }
            if(stack.size()!=1){error="igc inline program";return false;}
            value=strata::igc::value_tokens(stack.back(),call->spaced);
        } else {
            namespace rational=strata::igc::rational_exact;
            if(!inline_rational_request_ && (cudaMalloc(&inline_rational_request_,sizeof(*inline_rational_request_))!=cudaSuccess
                    || cudaMalloc(&inline_rational_result_,sizeof(*inline_rational_result_))!=cudaSuccess)){error="igc inline device buffers";return false;}
            std::vector<rational::Operand> stack;
            for(const auto& step:call->program) {
                if(step.number) {
                    rational::Operand operand;operand.numerator=step.value.coefficient;
                    operand.denominator.length=static_cast<uint8_t>(step.value.scale+1);operand.denominator.digits[step.value.scale]=1;
                    stack.push_back(operand);continue;
                }
                if(stack.size()<2){error="igc inline program";return false;}
                if(step.operation==strata::igc::InlineOperation::Power)return true; // fractional powers stay the model's
                rational::Request request;request.operation=static_cast<rational::Operation>(step.operation);request.invocation=1;
                request.rhs=stack.back();stack.pop_back();request.lhs=stack.back();stack.pop_back();
                rational::Result result;
                if(cudaMemcpyAsync(inline_rational_request_,&request,sizeof(request),cudaMemcpyHostToDevice,stream_)!=cudaSuccess
                        || !rational::execute(inline_rational_request_,inline_rational_result_,1,stream_,error)
                        || cudaMemcpyAsync(&result,inline_rational_result_,sizeof(result),cudaMemcpyDeviceToHost,stream_)!=cudaSuccess
                        || cudaStreamSynchronize(stream_)!=cudaSuccess){if(error.empty())error="igc inline execution";return false;}
                if(result.status!=rational::Status::Success)return true;
                rational::Operand next;next.numerator=result.value.numerator;next.denominator=result.value.denominator;
                stack.push_back(next);
            }
            if(stack.size()!=1){error="igc inline program";return false;}
            rational::Rational final_value;final_value.numerator=stack.back().numerator;final_value.denominator=stack.back().denominator;
            const auto tokens=strata::igc::rational_tokens(final_value,call->spaced);
            if(!tokens)return true;
            value=*tokens;
        }
        bridge_tokens_=std::move(value);bridge_at_=0;
        // Inside reasoning the value says where it came from; unmarked, the model re-derives it.
        if(!thinking_closed_) {
            // A repeat means the model is re-checking: say so once, then close the thinking.
            int64_t key=0;
            for(const auto& step:call->program)
                key=key*1000003+(step.number?number_key(step.value.coefficient)*16+step.value.scale:static_cast<int64_t>(step.operation));
            const auto seen=std::count(inline_seen_.begin(),inline_seen_.end(),key);
            inline_seen_.push_back(key);
            // No closing `.` outside a sum: at least 16 % of the values continue, `2100 per week`, and the model writes its own stop.
            const auto& top=call->program.back();
            const bool sum=!top.number && (top.operation==strata::igc::InlineOperation::Add || top.operation==strata::igc::InlineOperation::Subtract);
            const auto after=strata::igc::inline_tokens::after_value(static_cast<int>(seen),inline_notes_,close_repeat_,static_cast<int32_t>(kThinkClose),sum?sum_note_:strata::igc::inline_tokens::SumNote::Open,repeat_note_);
            bridge_tokens_.insert(bridge_tokens_.end(),after.begin(),after.end());
        }
        ++inline_calls_;called=true;return true;
    }
    // One-token rounds only: replaces the sampled token while a bridged result is delivered.
    int bridge_left()const{return static_cast<int>(bridge_tokens_.size()-bridge_at_);}
    int32_t bridge_ahead(int i)const{return bridge_tokens_[bridge_at_+static_cast<size_t>(i)];}
    void bridge_sample(int32_t& token) {
        if(bridge_ && bridge_at_<bridge_tokens_.size()) {
            token=bridge_tokens_[bridge_at_++];
            ++bridged_unread_;
        }
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
