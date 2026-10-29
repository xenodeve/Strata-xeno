#include "strata/core/qwenputer_runner.hpp"
#include "strata/timeline_gpu.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace strata::qwenputer {
namespace {
constexpr unsigned go_token=230322,halt_token=230321,fault_start=229892;
struct Parameters {unsigned steps,output,epoch;};
struct Control {
    RunReason reason=RunReason::Ready;
    unsigned error=0,prefix_count=0,prefix_cursor=0,prefix_language=1,ready=0,halted=0;
    unsigned chunk_steps=0,total_steps=0,internal_steps=0,trace_steps=0,output_total=0;
    unsigned output_count=0,trace_count=0,cancel=0,epoch=0;
};
bool terminal(RunReason r){return r==RunReason::Success||r==RunReason::Limit||r==RunReason::Cancelled||r==RunReason::Fault||r==RunReason::Deadline;}
void ck(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
__device__ bool stopped(RunReason r){return r==RunReason::Success||r==RunReason::Limit||r==RunReason::Cancelled||r==RunReason::Fault||r==RunReason::Deadline;}
__device__ bool native_step(const Control* c){return !(c->prefix_language&&c->prefix_cursor<c->prefix_count);}
__global__ void begin_chunk(Control* c,const Parameters* p,const Input* prefix,Input* input,
                            const StepStatus* status,RuntimeLimits limits,cudaGraphConditionalHandle condition) {
    c->chunk_steps=0;c->output_count=0;c->trace_count=0;
    if(c->epoch!=p->epoch){c->reason=RunReason::Fault;c->error=0x200;}
    if(stopped(c->reason)){cudaGraphSetConditional(condition,0);return;}
    c->reason=RunReason::Ready;
    if(atomicAdd(&c->cancel,0u))c->reason=RunReason::Cancelled;
    else if(!p->steps)c->reason=RunReason::Budget;
    else if(!p->output)c->reason=RunReason::Output;
    if(c->reason!=RunReason::Ready){cudaGraphSetConditional(condition,0);return;}
    if(!c->ready) {
        if(c->prefix_cursor<c->prefix_count)*input=prefix[c->prefix_cursor];
        else if(status->owned)*input=status->next;
        else {c->reason=RunReason::Fault;c->error=0x201;}
        c->ready=1;
    }
    if(c->reason==RunReason::Ready&&c->total_steps>=limits.steps)c->reason=RunReason::Limit;
    cudaGraphSetConditional(condition,c->reason==RunReason::Ready);
}
__global__ void advance(Control* c,const Parameters* p,const Input* prefix,Input* input,
                        const StepStatus* status,unsigned* output,Input* trace,RuntimeLimits limits,
                        unsigned newline,cudaGraphConditionalHandle condition) {
    const Input consumed=*input;
    const bool internal=native_step(c);
    ++c->chunk_steps;++c->total_steps;
    if(internal){++c->internal_steps;if(consumed.kind==TokenKind::Trace)++c->trace_steps;}
    if(limits.record_trace) {
        if(c->trace_count>=limits.chunk){c->reason=RunReason::Fault;c->error=0x202;}
        else trace[c->trace_count++]=consumed;
    }
    if(status->error){c->reason=RunReason::Fault;c->error=status->error;}
    if(internal&&consumed.kind==TokenKind::Trace&&consumed.id>=fault_start&&consumed.id<fault_start+8) {
        c->reason=RunReason::Fault;c->error=0x100+(consumed.id-fault_start);
    }
    if(stopped(c->reason)){cudaGraphSetConditional(condition,0);return;}
    if(internal&&consumed.kind==TokenKind::Trace&&consumed.id==halt_token)c->halted=1;
    if(internal&&consumed.kind==TokenKind::Text) {
        if(c->output_total>=limits.output){c->reason=RunReason::Limit;cudaGraphSetConditional(condition,0);return;}
        output[c->output_count++]=consumed.id;++c->output_total;
    }
    if(c->prefix_cursor<c->prefix_count)++c->prefix_cursor;
    if(c->prefix_cursor<c->prefix_count)*input=prefix[c->prefix_cursor];
    else if(status->owned)*input=status->next;
    else if(c->halted&&consumed.kind==TokenKind::Text&&consumed.id==newline) {
        c->reason=RunReason::Success;c->ready=0;
    } else {c->reason=RunReason::Fault;c->error=0x203;}
    if(!stopped(c->reason)) {
        if(atomicAdd(&c->cancel,0u))c->reason=RunReason::Cancelled;
        else if(c->total_steps>=limits.steps)c->reason=RunReason::Limit;
        else if(c->output_count>=p->output)c->reason=RunReason::Output;
        else if(c->chunk_steps>=p->steps)c->reason=RunReason::Budget;
    }
    cudaGraphSetConditional(condition,c->reason==RunReason::Ready);
}
__global__ void cancel_run(Control* c){atomicExch(&c->cancel,1u);}
bool valid_utf8(const std::vector<unsigned char>& bytes) {
    unsigned pending=0,lo=0x80,hi=0xbf;
    for(auto b:bytes) {
        if(pending) {if(b<lo||b>hi)return false;--pending;lo=0x80;hi=0xbf;continue;}
        if(b<=0x7f)continue;
        if(b>=0xc2&&b<=0xdf)pending=1;
        else if(b>=0xe0&&b<=0xef){pending=2;if(b==0xe0)lo=0xa0;else if(b==0xed)hi=0x9f;}
        else if(b>=0xf0&&b<=0xf4){pending=3;if(b==0xf0)lo=0x90;else if(b==0xf4)hi=0x8f;}
        else return false;
    }
    return pending==0;
}
}

struct Runtime::Impl {
    Executor* executor=nullptr;const Artifact* artifact=nullptr;RuntimeLimits limits{};
    cudaStream_t stream=nullptr,cancel_stream=nullptr;
    cudaGraph_t graph=nullptr;cudaGraphExec_t instance=nullptr;cudaEvent_t finished=nullptr;
    cudaGraphConditionalHandle condition{};
    Control* control=nullptr;Parameters* parameters=nullptr;Input* prefix=nullptr;Input* trace=nullptr;unsigned* output=nullptr;
    std::vector<Input> host_prefix;
    Parameters host_parameters{};
    RunReason deadline_reason=RunReason::Deadline;
    std::chrono::steady_clock::time_point started{};
    unsigned epoch=0;int device=-1;bool prepared=false,inflight=false,ended=false;
    std::uint64_t bytes=0,graph_delta=0;std::size_t nodes=0;
    std::unique_ptr<timeline::GpuClock> clock=std::make_unique<timeline::GpuClock>();
    cudaEvent_t chunk_begin=nullptr,chunk_end=nullptr;
    ~Impl() {
        int previous=-1;cudaGetDevice(&previous);
        if(device>=0&&previous!=device)cudaSetDevice(device);
        if(stream)cudaStreamSynchronize(stream);
        if(cancel_stream)cudaStreamSynchronize(cancel_stream);
        clock.reset();
        if(instance)cudaGraphExecDestroy(instance);
        if(graph)cudaGraphDestroy(graph);
        if(finished)cudaEventDestroy(finished);
        if(cancel_stream)cudaStreamDestroy(cancel_stream);
        if(output)cudaFree(output);if(trace)cudaFree(trace);if(prefix)cudaFree(prefix);
        if(parameters)cudaFree(parameters);if(control)cudaFree(control);
        if(previous>=0&&previous!=device)cudaSetDevice(previous);
    }
    template<class T>void allocate(T*& pointer,std::size_t n) {
        if(n){ck(cudaMalloc(&pointer,n*sizeof(T)));bytes+=n*sizeof(T);}
    }
    void context()const {int current;ck(cudaGetDevice(&current));if(current!=device)throw std::runtime_error("runner_device_mismatch");}
    bool expired()const {
        return std::chrono::steady_clock::now()-started>=std::chrono::milliseconds(limits.timeout_ms);
    }
};
Runtime::Runtime()=default;
Runtime::~Runtime()=default;
bool Runtime::init(Executor& executor,const Artifact& artifact,RuntimeLimits limits,cudaStream_t stream,std::string& error)try {
    if(impl_){error="already_initialized";return false;}
    if(!executor.device_input()||!limits.chunk||limits.chunk>256||limits.steps>200000||limits.output>4096) {
        error="runner_limits";return false;
    }
    if(!executor.artifact_digest()||*executor.artifact_digest()!=artifact.digest) {
        error="runner_artifact_mismatch";return false;
    }
    auto p=std::make_unique<Impl>();p->executor=&executor;p->artifact=&artifact;p->limits=limits;
    ck(cudaGetDevice(&p->device));cudaPointerAttributes attributes{};ck(cudaPointerGetAttributes(&attributes,executor.device_state()));
    if(attributes.device!=p->device)throw std::runtime_error("runner_device_mismatch");
    int stream_device;ck(cudaStreamGetDevice(stream,&stream_device));
    if(stream_device!=p->device)throw std::runtime_error("runner_stream_device_mismatch");
    p->stream=stream;
    p->allocate(p->control,1);p->allocate(p->parameters,1);p->allocate(p->prefix,82);
    p->allocate(p->output,4096);if(limits.record_trace)p->allocate(p->trace,limits.chunk);
    ck(cudaMemsetAsync(p->control,0,sizeof(Control),stream));ck(cudaStreamSynchronize(stream));
    int least,greatest;ck(cudaDeviceGetStreamPriorityRange(&least,&greatest));
    ck(cudaStreamCreateWithPriority(&p->cancel_stream,cudaStreamNonBlocking,greatest));
    cancel_run<<<1,1,0,p->cancel_stream>>>(p->control);ck(cudaGetLastError());ck(cudaStreamSynchronize(p->cancel_stream));
    ck(cudaEventCreateWithFlags(&p->finished,cudaEventDisableTiming));
    std::size_t before,total,after;ck(cudaMemGetInfo(&before,&total));
    ck(cudaGraphCreate(&p->graph,0));ck(cudaGraphConditionalHandleCreate(&p->condition,p->graph,0,cudaGraphCondAssignDefault));
    auto* input=executor.device_input();auto* status=executor.device_status();
    void* args[]{&p->control,&p->parameters,&p->prefix,&input,&status,&p->limits,&p->condition};
    cudaKernelNodeParams kernel{};kernel.func=reinterpret_cast<void*>(begin_chunk);
    kernel.gridDim=dim3(1);kernel.blockDim=dim3(1);kernel.kernelParams=args;
    cudaGraphNode_t begin_node,while_node;ck(cudaGraphAddKernelNode(&begin_node,p->graph,nullptr,0,&kernel));
    cudaGraphNodeParams conditional{};conditional.type=cudaGraphNodeTypeConditional;
    conditional.conditional.handle=p->condition;conditional.conditional.type=cudaGraphCondTypeWhile;conditional.conditional.size=1;
    ck(cudaGraphAddNode(&while_node,p->graph,&begin_node,nullptr,1,&conditional));
    const auto body=conditional.conditional.phGraph_out[0];
    ck(cudaStreamBeginCaptureToGraph(stream,body,nullptr,nullptr,0,cudaStreamCaptureModeThreadLocal));
    const bool enqueued=executor.enqueue_step(stream,error);
    if(enqueued)advance<<<1,1,0,stream>>>(p->control,p->parameters,p->prefix,input,status,p->output,p->trace,limits,
                                        artifact.config[9],p->condition);
    const auto launch_error=cudaGetLastError();cudaGraph_t captured=nullptr;
    const auto capture_error=cudaStreamEndCapture(stream,&captured);
    if(!enqueued)throw std::runtime_error(error);
    ck(launch_error);ck(capture_error);if(captured!=body)throw std::runtime_error("capture_body_identity");
    ck(cudaGraphGetNodes(body,nullptr,&p->nodes));ck(cudaGraphInstantiate(&p->instance,p->graph,0));
    ck(cudaMemGetInfo(&after,&total));p->graph_delta=before>after?before-after:0;
    impl_=std::move(p);error.clear();return true;
}catch(const std::exception& e){error=e.what();return false;}

bool Runtime::prepare(std::span<const Input> prefix,bool reset,std::string& error)try {
    timeline::Span span("qwenputer.prepare");
    if(!impl_){error="uninitialized";return false;}auto& p=*impl_;p.context();
    if(p.inflight||(p.prepared&&!p.ended)){error="runner_busy";return false;}
    const auto& a=*p.artifact;
    const bool seed=prefix.size()==1&&prefix[0].kind==TokenKind::Trace&&prefix[0].id==go_token;
    if(prefix.size()>82||(prefix.empty()&&reset)){error="prefix_shape";return false;}
    if(!prefix.empty()&&!seed) {
        if(prefix.size()<2||prefix.front().kind!=TokenKind::Text||prefix.back().kind!=TokenKind::Text
            ||prefix.front().id!=a.config[7]||prefix.back().id!=a.config[8]){error="prefix_boundary";return false;}
        std::vector<unsigned char> bytes;
        for(std::size_t i=1;i+1<prefix.size();++i) {
            const auto t=prefix[i];
            if(t.kind!=TokenKind::Text||t.id>=a.config[2]||a.text_types[t.id]!=1||a.text_lengths[t.id]>16) {
                error="prefix_row";return false;
            }
            for(unsigned j=0;j<a.text_lengths[t.id];++j) {
                const auto b=a.text_rows[t.id][j];if(!b){error="prefix_nul";return false;}bytes.push_back(b);
            }
        }
        if(!valid_utf8(bytes)){error="prefix_utf8";return false;}
    }
    p.started=std::chrono::steady_clock::now();
    if(reset&&!p.executor->enqueue_reset(p.stream,error))return false;
    p.host_prefix.assign(prefix.begin(),prefix.end());
    if(!prefix.empty())ck(cudaMemcpyAsync(p.prefix,p.host_prefix.data(),prefix.size_bytes(),cudaMemcpyHostToDevice,p.stream));
    Control state;state.prefix_count=unsigned(prefix.size());state.prefix_language=seed?0:1;state.epoch=++p.epoch;
    ck(cudaMemcpyAsync(p.control,&state,sizeof(state),cudaMemcpyHostToDevice,p.stream));
    ck(cudaStreamSynchronize(p.stream));p.clock->anchor(p.stream);
    p.prepared=true;p.ended=false;error.clear();return true;
}catch(const std::exception& e){error=e.what();return false;}

bool Runtime::enqueue_chunk(std::uint32_t steps,std::uint32_t output,std::string& error)try {
    timeline::Span span("qwenputer.enqueue",steps,output);
    if(!impl_){error="uninitialized";return false;}auto& p=*impl_;p.context();
    if(!p.prepared||p.inflight||p.ended){error="runner_state";return false;}
    if(steps>p.limits.chunk||output>4096){error="chunk_limits";return false;}
    if(p.expired())ck(cudaMemcpyAsync(&p.control->reason,&p.deadline_reason,sizeof(RunReason),cudaMemcpyHostToDevice,p.stream));
    p.host_parameters={steps,output,p.epoch};
    ck(cudaMemcpyAsync(p.parameters,&p.host_parameters,sizeof(Parameters),cudaMemcpyHostToDevice,p.stream));
    p.chunk_begin=p.clock->record(p.stream);
    ck(cudaGraphLaunch(p.instance,p.stream));p.chunk_end=p.clock->record(p.stream);
    ck(cudaEventRecord(p.finished,p.stream));
    p.inflight=true;error.clear();return true;
}catch(const std::exception& e){if(impl_)impl_->ended=true;error=e.what();return false;}

bool Runtime::collect(ChunkResult& result,std::string& error)try {
    if(!impl_||!impl_->inflight){error="no_pending_chunk";return false;}auto& p=*impl_;p.context();
    {timeline::Span wait("qwenputer.wait");ck(cudaEventSynchronize(p.finished));}
    timeline::Span collect("qwenputer.collect");Control state;
    ck(cudaMemcpy(&state,p.control,sizeof(state),cudaMemcpyDeviceToHost));
    if(state.epoch!=p.epoch||state.output_count>4096||state.trace_count>p.limits.chunk
       ||unsigned(state.reason)>unsigned(RunReason::Deadline))
        throw std::runtime_error("chunk_result_bounds");
    // A blocking wait may finish after the cumulative deadline. Do not publish
    // that result as timely success, nor renew the deadline at the next chunk.
    if(p.expired()&&state.reason!=RunReason::Fault&&state.reason!=RunReason::Cancelled)state.reason=RunReason::Deadline;
    if(timeline::enabled())p.clock->span(timeline::lane("gpu0 qwenputer"),"qwenputer.chunk",
                                       p.chunk_begin,p.chunk_end,state.chunk_steps,state.epoch);
    p.clock->resolve(false);
    ChunkResult r;r.reason=state.reason;r.error=state.error;r.chunk_steps=state.chunk_steps;
    r.total_steps=state.total_steps;r.internal_steps=state.internal_steps;r.trace_steps=state.trace_steps;r.output_total=state.output_total;
    const bool failure=state.reason==RunReason::Fault||state.reason==RunReason::Limit||state.reason==RunReason::Cancelled||state.reason==RunReason::Deadline;
    if(!failure&&state.output_count){r.text.resize(state.output_count);ck(cudaMemcpy(r.text.data(),p.output,r.text.size()*4,cudaMemcpyDeviceToHost));}
    if(p.limits.record_trace&&state.trace_count){r.trace.resize(state.trace_count);ck(cudaMemcpy(r.trace.data(),p.trace,r.trace.size()*sizeof(Input),cudaMemcpyDeviceToHost));}
    p.inflight=false;p.ended=terminal(state.reason);result=std::move(r);error.clear();return true;
}catch(const std::exception& e){if(impl_){impl_->inflight=false;impl_->ended=true;}error=e.what();return false;}

bool Runtime::request_cancel(std::string& error)try {
    timeline::Span cancel("qwenputer.cancel");
    if(!impl_||!impl_->prepared){error="runner_state";return false;}auto& p=*impl_;
    int previous;ck(cudaGetDevice(&previous));if(previous!=p.device)ck(cudaSetDevice(p.device));
    cancel_run<<<1,1,0,p.cancel_stream>>>(p.control);const auto launch=cudaGetLastError();
    const auto wait=cudaStreamSynchronize(p.cancel_stream);if(previous!=p.device)cudaSetDevice(previous);
    ck(launch);ck(wait);error.clear();return true;
}catch(const std::exception& e){error=e.what();return false;}
std::uint64_t Runtime::allocated_bytes()const{return impl_?impl_->bytes:0;}
std::uint64_t Runtime::graph_allocation_delta()const{return impl_?impl_->graph_delta:0;}
std::size_t Runtime::body_nodes()const{return impl_?impl_->nodes:0;}
}
