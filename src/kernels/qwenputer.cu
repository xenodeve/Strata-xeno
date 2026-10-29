#include "strata/core/qwenputer.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace strata::qwenputer {
namespace {
struct Contribution { int kind, index, column, into; float weight; };
struct Group { int dim, start, count; };
struct View {
    std::uint32_t config[16]{};
    const Feature* features;
    const Expression* expressions;
    const Term* terms;
    const Channel* channels;
    const Mapping* maps;
    const FfnTerm* ffn_terms;
    const Ffn* ffns;
    const Head* heads;
    const Output* outputs;
    const InitialRow* initial;
    const int* printable;
    const int* printable_index;
    const std::uint8_t* text_rows;
    const int* keep;
    const int* value_channels;
    const int* gates;
    float *residual, *state, *lags, *ffn_values, *head_values;
    Input* input;
    StepStatus* status;
    int features_count, initial_count;
};
struct Stage {
    int level, slot;
    std::vector<int> nodes;
    std::vector<Group> groups;
    std::vector<Contribution> contributions;
    const int* device_nodes = nullptr;
    const Group* device_groups = nullptr;
    const Contribution* device_contributions = nullptr;
};
void check(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
__device__ float add(float a, float b) { return __fadd_rn(a, b); }
__device__ float mul(float a, float b) { return __fmul_rn(a, b); }
__device__ void fault(View v, unsigned bit) { atomicOr(&v.status->error, bit); }
__device__ float expression(View v, int id, const float* m) {
    const auto e = v.expressions[id];
    float sum = 0;
    for (int i = 0; i < e.count; ++i) {
        const auto t = v.terms[e.start + i];
        sum = add(sum, mul(t.coefficient, m[t.dim]));
    }
    return sum;
}
__device__ float channel(View v, int id, int slot) {
    const auto c = v.channels[id];
    float sum = 0;
    for (int lag = 0; lag < 4; ++lag) {
        if (c.taps[lag] != 0) {
            const auto* m = lag == 0 ? v.residual : v.lags + (slot * 3 + lag - 1) * v.config[0];
            sum = add(sum, mul(c.taps[lag], expression(v, c.expression, m)));
        }
    }
    return fmaxf(sum, 0);
}
__device__ float feature(View v, Feature f, Input t) {
    constexpr unsigned sizes[]{32768,32768,32768,4,65536,65536,512,8,16,16,128,256,1,1,1,1,1,1,1,1};
    const bool text = t.kind == TokenKind::Text;
    unsigned family = 0, payload = t.id;
    if (!text && f.opcode >= 2 && f.opcode <= 5)
        while (payload >= sizes[family]) { payload -= sizes[family]; ++family; }
    switch (f.opcode) {
    case 0: return 0;
    case 1: return 1;
    case 2: return !text && family == unsigned(f.p0) ? 1.f : 0.f;
    case 3: return !text && family == unsigned(f.p0) ? float((payload >> f.p1) & 1) : 0.f;
    case 4: return !text && ((family == 9 && payload == unsigned(f.p0))
                     || (family == 10 && int(payload >> 4) == f.p0 - 1)) ? 1.f : 0.f;
    case 5: {
        if (text) {
            const int p = v.printable_index[t.id];
            if (p < 0) return 0;
            family = 20; payload = unsigned(p);
        }
        return ((family | (payload << 5)) >> f.p0) & 1 ? 128.f : -128.f;
    }
    case 6: return text ? 1.f : 0.f;
    case 7: return text ? 0.f : 1.f;
    case 8: return text ? 0x1p-40f : -1.f;
    case 9: return text && t.id == v.config[7] ? 1.f : 0.f;
    case 10: return text && t.id == v.config[8] ? 1.f : 0.f;
    case 11: return text ? float((v.text_rows[t.id * 16 + f.p0] >> f.p1) & 1) : 0.f;
    default: return 0;
    }
}
__global__ void probe_features(View v, unsigned f, TokenKind kind, unsigned base,
                                unsigned count, float* output) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = feature(v, v.features[f], Input{kind,base+i});
}
__global__ void begin(View v) {
    if (threadIdx.x == 0) {
        v.status->owned = 0;
        const auto t = *v.input;
        if ((t.kind != TokenKind::Trace && t.kind != TokenKind::Text)
            || t.id >= v.config[t.kind == TokenKind::Trace ? 1 : 2]) fault(v, 1);
    }
    __syncthreads();
    if (v.status->error) return;
    for (unsigned d = threadIdx.x; d < v.config[0]; d += blockDim.x) v.residual[d] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < v.features_count; i += blockDim.x) {
        const auto f = v.features[i];
        v.residual[f.dim] = feature(v, f, *v.input);
    }
}
__global__ void initialize_rows(View v) {
    const int i = blockIdx.x;
    if (i < v.initial_count) {
        const auto& row = v.initial[i];
        v.state[row.head * 16384 + row.row * 128 + threadIdx.x] = row.values[threadIdx.x];
    }
}
__global__ void ffns(View v, const int* ids, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || v.status->error) return;
    const int id = ids[i];
    const auto f = v.ffns[id];
    float sum = 0;
    for (int k = 0; k < f.count; ++k) {
        const auto t = v.ffn_terms[f.start + k];
        const float a = expression(v, t.a, v.residual), b = expression(v, t.b, v.residual);
        sum = add(sum, mul(mul(t.coefficient, a), fmaxf(b, 0)));
    }
    if (!isfinite(sum)) fault(v, 2);
    v.ffn_values[id] = sum;
}
__global__ void heads(View v, const int* ids, int count, int slot) {
    __shared__ int active;
    if (threadIdx.x == 0) active = int(blockIdx.x) < count && !v.status->error;
    __syncthreads();
    if (!active) return;
    const int id = ids[blockIdx.x], col = threadIdx.x;
    const auto h = v.heads[id];
    __shared__ int rows[128], qcount, key, ok;
    __shared__ float decay, beta, refs[2], denominator;
    if (col == 0) {
        qcount = 0; key = -1; ok = 1;
        float first = 0;
        for (int i = 0; i < h.qcount; ++i) {
            const auto m = v.maps[h.qstart + i];
            const auto q = channel(v, m.index, slot);
            if (!isfinite(q)) ok = 0;
            if (q > 0) {
                if (qcount && q != first) ok = 0;
                first = q; rows[qcount++] = m.lane;
            }
        }
        for (int i = 0; i < h.kcount; ++i) {
            const auto m = v.maps[h.kstart + i];
            const auto k = channel(v, m.index, slot);
            if (!isfinite(k)) ok = 0;
            if (k > 0) { if (key >= 0) ok = 0; key = m.lane; }
        }
        const auto selector = expression(v, h.decay, v.residual);
        decay = selector == 0 ? 1.f : selector == 110 ? .5f : 0.f;
        if (selector != 0 && selector != 110 && selector != 238) ok = 0;
        beta = h.beta < 0 ? 1.f : expression(v, h.beta, v.residual);
        if (!(beta == 1 || (beta <= 0 && beta == floorf(beta)))) ok = 0;
        if (!ok) fault(v, 4);
    }
    __syncthreads();
    if (!ok) return;
    const int vc = v.value_channels[id * 128 + col], ze = v.gates[id * 128 + col];
    const float value = vc < 0 ? 0.f : channel(v, vc, slot);
    const float gate = ze < 0 ? 0.f : expression(v, ze, v.residual);
    float* state = v.state + h.state_offset;
    if (decay != 1)
        for (int row = 0; row < 128; ++row) state[row * 128 + col] = mul(state[row * 128 + col], decay);
    if (key >= 0 && beta > 0) state[key * 128 + col] = value;
    float sum = 0;
    for (int i = 0; i < qcount; ++i) sum = add(sum, state[rows[i] * 128 + col]);
    if (col < 2) refs[col] = sum;
    __syncthreads();
    if (col == 0) denominator = __fsqrt_rn(__fdiv_rn(add(mul(refs[0], refs[0]), mul(refs[1], refs[1])), 128.f));
    __syncthreads();
    if (!isfinite(value) || !isfinite(gate) || !isfinite(sum) || !isfinite(denominator)
        || (denominator == 0 && sum != 0)) fault(v, 8);
    const auto normalized = denominator == 0 ? 0.f : __fdiv_rn(sum, denominator);
    v.head_values[id * 128 + col] = mul(normalized, fmaxf(gate, 0));
}
__global__ void shift_history(View v, int slot) {
    const unsigned d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= v.config[0] || v.status->error) return;
    auto* hist = v.lags + slot * 3 * v.config[0] + d;
    hist[2 * v.config[0]] = hist[v.config[0]];
    hist[v.config[0]] = hist[0];
    hist[0] = v.residual[d];
}
__global__ void gather(View v, const Group* groups, const Contribution* values, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count || v.status->error) return;
    const auto g = groups[i];
    float result = v.residual[g.dim];
    for (int k = 0; k < g.count; ++k) {
        const auto c = values[g.start + k];
        const auto value = c.kind == 0 ? v.ffn_values[c.index] : mul(v.head_values[c.index * 128 + c.column], c.weight);
        result = c.into ? add(result, value) : value;
    }
    v.residual[g.dim] = result;
}
__global__ void predict(View v) {
    if (v.status->error) return;
    const float owner = v.residual[v.config[15]];
    if (owner == 0) return;
    if (owner != 1) { fault(v, 16); return; }
    unsigned code = 0;
    for (int bit = 0; bit < 21; ++bit) {
        const auto x = v.residual[v.keep[bit]];
        if (fabsf(x) != 0x1p-20f) { fault(v, 16); return; }
        if (x > 0) code |= 1u << bit;
    }
    const unsigned family = code & 31, payload = code >> 5;
    if (family == 20) {
        if (payload >= 260) { fault(v, 16); return; }
        v.status->next = {TokenKind::Text, unsigned(v.printable[payload])};
    } else {
        constexpr unsigned sizes[]{32768,32768,32768,4,65536,65536,512,8,16,16,128,256,1,1,1,1,1,1,1,1};
        if (family >= 20 || payload >= sizes[family]) { fault(v, 16); return; }
        unsigned id = payload;
        for (unsigned i = 0; i < family; ++i) id += sizes[i];
        v.status->next = {TokenKind::Trace, id};
    }
    v.status->owned = 1;
}
} // namespace

struct Executor::Impl {
    View v{};
    int device = -1, lag_slots = 0;
    std::uint64_t bytes = 0;
    std::vector<void*> allocations;
    std::vector<Stage> stages;
    ~Impl() {
        if (allocations.empty()) return;
        int previous = -1;
        cudaGetDevice(&previous);
        if (device >= 0 && previous != device) cudaSetDevice(device);
        for (auto p : allocations) cudaFree(p);
        if (previous >= 0 && previous != device) cudaSetDevice(previous);
    }
    template<class T> T* allocate(std::size_t n) {
        if (!n) return nullptr;
        if (n > (256ull << 20) / sizeof(T) || bytes + n * sizeof(T) > (256ull << 20))
            throw std::runtime_error("device_buffer_budget");
        T* result = nullptr;
        check(cudaMalloc(&result, n * sizeof(T)));
        try { allocations.push_back(result); }
        catch (...) { cudaFree(result); throw; }
        bytes += n * sizeof(T);
        return result;
    }
    template<class T> const T* upload(const std::vector<T>& data, cudaStream_t stream) {
        (void)stream;
        auto* p = allocate<T>(data.size());
        // Initialization is outside all captured/runtime execution. Complete each
        // upload before its host staging can unwind on an allocation failure.
        if (!data.empty()) check(cudaMemcpy(p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
        return p;
    }
    void schedule(const Artifact& a) {
        struct Op { int node, vi, level, kind, index; std::vector<int> reads, writes; };
        std::vector<Op> ops;
        auto expression_reads = [&](Op& op, int e) {
            const auto x = a.expressions[e];
            for (int i = 0; i < x.count; ++i) op.reads.push_back(a.terms[x.start + i].dim);
        };
        for (int i = 0; i < int(a.ffns.size()); ++i) {
            const auto& f = a.ffns[i];
            Op op{f.node,0,f.level,0,i,{}, {f.out}};
            for (int k = 0; k < f.count; ++k) {
                const auto& t = a.ffn_terms[f.start + k];
                expression_reads(op,t.a); expression_reads(op,t.b);
            }
            ops.push_back(std::move(op));
        }
        for (int i = 0; i < int(a.heads.size()); ++i) {
            const auto& h = a.heads[i];
            Op op{h.node,h.value_index,h.level,1,i,{},{}};
            const int starts[]{h.qstart,h.kstart,h.vstart,h.zstart}, counts[]{h.qcount,h.kcount,h.vcount,h.zcount};
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < counts[j]; ++k) {
                    const auto m = a.maps[starts[j] + k];
                    expression_reads(op,j == 3 ? m.index : a.channels[m.index].expression);
                }
            expression_reads(op,h.decay);
            if (h.beta >= 0) expression_reads(op,h.beta);
            for (int k = 0; k < h.ocount; ++k) op.writes.push_back(a.outputs[h.ostart + k].dim);
            ops.push_back(std::move(op));
        }
        std::sort(ops.begin(),ops.end(),[](const auto& x,const auto& y){return std::tie(x.node,x.vi)<std::tie(y.node,y.vi);});
        std::vector<std::vector<int>> dependencies(ops.size());
        std::vector<int> producer(a.dims.size(),-1);
        for (int i = 0; i < int(ops.size()); ++i) {
            auto& op = ops[i];
            for (auto d : op.reads) {
                const auto p = producer[d];
                if (p >= 0 && ops[p].level >= op.level) throw std::runtime_error("same_stage_dependency");
                dependencies[i].push_back(p);
            }
            for (auto d : op.writes) producer[d] = i;
        }
        std::vector<int> order(ops.size()); std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](int x,int y){return ops[x].level<ops[y].level;});
        std::fill(producer.begin(),producer.end(),-1);
        for (auto i : order) {
            const auto& op = ops[i];
            for (std::size_t k = 0; k < op.reads.size(); ++k)
                if (producer[op.reads[k]] != dependencies[i][k]) throw std::runtime_error("stage_reorder_dependency");
            for (auto d : op.writes) producer[d] = i;
        }
        std::map<int,std::vector<int>> by_level;
        for (auto i : order) by_level[ops[i].level].push_back(i);
        for (const auto& [level, indices] : by_level) {
            Stage stage{level, level % 2 ? lag_slots++ : -1, {}, {}, {}};
            std::map<int,std::vector<Contribution>> groups;
            for (auto i : indices) {
                const auto& op = ops[i]; stage.nodes.push_back(op.index);
                if (op.kind == 0) {
                    const auto& f = a.ffns[op.index];
                    groups[f.out].push_back({0,op.index,0,f.into,1});
                } else {
                    const auto& h = a.heads[op.index];
                    for (int k = 0; k < h.ocount; ++k) {
                        const auto& o = a.outputs[h.ostart + k];
                        groups[o.dim].push_back({1,op.index,o.column,h.into,o.weight});
                    }
                }
            }
            for (const auto& [dim, values] : groups) {
                stage.groups.push_back({dim,int(stage.contributions.size()),int(values.size())});
                stage.contributions.insert(stage.contributions.end(),values.begin(),values.end());
            }
            stages.push_back(std::move(stage));
        }
    }
};
Executor::Executor() = default;
Executor::~Executor() = default;
bool Executor::init(const Artifact& a, cudaStream_t stream, std::string& error) try {
    if (impl_) { error = "already_initialized"; return false; }
    if (a.binding.digests[0] != arithmetic_abi()) { error = "numeric_abi"; return false; }
    auto p = std::make_unique<Impl>();
    p->schedule(a); // No GPU allocation before schedule validation.
    check(cudaGetDevice(&p->device));
    auto& v = p->v;
    std::copy(a.config.begin(),a.config.end(),v.config);
    v.features = p->upload(a.features,stream); v.features_count = int(a.features.size());
    v.expressions = p->upload(a.expressions,stream); v.terms = p->upload(a.terms,stream);
    v.channels = p->upload(a.channels,stream); v.maps = p->upload(a.maps,stream);
    v.ffn_terms = p->upload(a.ffn_terms,stream); v.ffns = p->upload(a.ffns,stream);
    v.heads = p->upload(a.heads,stream); v.outputs = p->upload(a.outputs,stream);
    v.initial = p->upload(a.initial,stream); v.initial_count = int(a.initial.size());
    v.printable = p->upload(a.printable,stream); v.printable_index = p->upload(a.printable_index,stream);
    std::vector<std::uint8_t> rows(a.text_rows.size() * 16);
    for (std::size_t i=0;i<a.text_rows.size();++i) std::copy(a.text_rows[i].begin(),a.text_rows[i].end(),rows.begin()+i*16);
    v.text_rows = p->upload(rows,stream); v.keep = p->upload(a.keep,stream);
    std::vector<int> values(a.heads.size()*128,-1), gates(values.size(),-1);
    for (std::size_t i=0;i<a.heads.size();++i) {
        const auto& h=a.heads[i];
        for(int j=0;j<h.vcount;++j) {const auto m=a.maps[h.vstart+j];values[i*128+m.lane]=m.index;}
        for(int j=0;j<h.zcount;++j) {const auto m=a.maps[h.zstart+j];gates[i*128+m.lane]=m.index;}
    }
    v.value_channels=p->upload(values,stream);v.gates=p->upload(gates,stream);
    for(auto& stage:p->stages) {
        stage.device_nodes=p->upload(stage.nodes,stream);stage.device_groups=p->upload(stage.groups,stream);
        stage.device_contributions=p->upload(stage.contributions,stream);
    }
    v.residual=p->allocate<float>(a.config[0]);v.state=p->allocate<float>(a.config[6]);
    v.lags=p->allocate<float>(p->lag_slots*3*a.config[0]);
    v.ffn_values=p->allocate<float>(a.ffns.size());v.head_values=p->allocate<float>(a.heads.size()*128);
    v.input=p->allocate<Input>(1);v.status=p->allocate<StepStatus>(1);
    check(cudaStreamSynchronize(stream)); // Own all host staging lifetimes before return.
    impl_=std::move(p);
    if(!enqueue_reset(stream,error)) {impl_.reset();return false;}
    check(cudaStreamSynchronize(stream));error.clear();return true;
} catch(const std::exception& e) {impl_.reset();error=e.what();return false;}
bool Executor::enqueue_reset(cudaStream_t stream,std::string& error) try {
    if(!impl_) {error="uninitialized";return false;}
    auto& p=*impl_;const auto v=p.v;
    check(cudaMemsetAsync(v.state,0,std::size_t(v.config[6])*4,stream));
    check(cudaMemsetAsync(v.lags,0,std::size_t(p.lag_slots)*3*v.config[0]*4,stream));
    check(cudaMemsetAsync(v.residual,0,std::size_t(v.config[0])*4,stream));
    check(cudaMemsetAsync(v.status,0,sizeof(StepStatus),stream));
    check(cudaMemsetAsync(v.input,0,sizeof(Input),stream));
    if(v.initial_count) initialize_rows<<<v.initial_count,128,0,stream>>>(v);
    check(cudaGetLastError());error.clear();return true;
} catch(const std::exception& e) {error=e.what();return false;}
bool Executor::enqueue_step(cudaStream_t stream,std::string& error) try {
    if(!impl_) {error="uninitialized";return false;}
    const auto v=impl_->v;
    begin<<<1,256,0,stream>>>(v);
    for(const auto& s:impl_->stages) {
        if(s.level%2) {
            heads<<<unsigned(s.nodes.size()),128,0,stream>>>(v,s.device_nodes,int(s.nodes.size()),s.slot);
            shift_history<<<(v.config[0]+255)/256,256,0,stream>>>(v,s.slot);
        } else ffns<<<(unsigned(s.nodes.size())+127)/128,128,0,stream>>>(v,s.device_nodes,int(s.nodes.size()));
        if(!s.groups.empty()) gather<<<(unsigned(s.groups.size())+127)/128,128,0,stream>>>(v,s.device_groups,s.device_contributions,int(s.groups.size()));
    }
    predict<<<1,1,0,stream>>>(v);
    check(cudaGetLastError());error.clear();return true;
} catch(const std::exception& e) {error=e.what();return false;}
Input* Executor::device_input() const {return impl_?impl_->v.input:nullptr;}
StepStatus* Executor::device_status() const {return impl_?impl_->v.status:nullptr;}
float* Executor::device_residual() const {return impl_?impl_->v.residual:nullptr;}
float* Executor::device_state() const {return impl_?impl_->v.state:nullptr;}
float* Executor::device_lags() const {return impl_?impl_->v.lags:nullptr;}
std::uint64_t Executor::allocated_bytes() const {return impl_?impl_->bytes:0;}
bool Executor::enqueue_feature_probe(std::uint32_t index,TokenKind kind,std::uint32_t base,
                                    std::uint32_t count,float* output,cudaStream_t stream,std::string& error) try {
    if(!impl_) {error="uninitialized";return false;}
    const auto v=impl_->v;
    if(index>=unsigned(v.features_count) || (kind!=TokenKind::Trace && kind!=TokenKind::Text)
       || !output || !count || count>262144) {error="feature_probe_range";return false;}
    const auto limit=v.config[kind==TokenKind::Trace?1:2];
    if(base>=limit || count>limit-base) {error="feature_probe_range";return false;}
    probe_features<<<(count+255)/256,256,0,stream>>>(v,index,kind,base,count,output);
    check(cudaGetLastError());error.clear();return true;
}catch(const std::exception& e){error=e.what();return false;}
}
