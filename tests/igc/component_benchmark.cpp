// Measurement only: native input records in, observed outputs out. No oracle.
#include "strata/core/igc_exact.hpp"
#include "strata/core/qwenputer_runner.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

namespace qp = strata::qwenputer;
namespace igc = strata::igc;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b-a).count();
}
static void ck(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
template<class T> static void read(std::istream& file, T& value) {
    if (!file.read(reinterpret_cast<char*>(&value), sizeof(value))) throw std::runtime_error("corpus_short");
}
static std::string quoted(const std::string& value) {
    std::string out = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { out += '\\'; out += c; }
        else if (c < 32 || c >= 127) { out += "\\u00"; out += hex[c>>4]; out += hex[c&15]; }
        else out += c;
    }
    return out + '"';
}
struct Case { igc::ExactRequest request; std::vector<qp::Input> prefix; };
struct Sample {
    double wall=0, prepare=0, issue=0, wait=0;
    unsigned status=0, reason=0, chunks=0, steps=0, internal=0, trace=0;
    std::string actual;
};
struct Stream {
    cudaStream_t value=nullptr;
    ~Stream() { if (value) { cudaStreamSynchronize(value); cudaStreamDestroy(value); } }
};
struct ExactBuffers {
    igc::ExactRequest* input=nullptr;
    igc::ExactResult* output=nullptr;
    ~ExactBuffers() { if (output) cudaFree(output); if (input) cudaFree(input); }
};
static void emit(const char* kind, const char* arm, unsigned index, unsigned block,
                 unsigned slot, const Sample& s) {
    std::cout << "{\"kind\":" << quoted(kind) << ",\"arm\":" << quoted(arm)
              << ",\"case\":" << index << ",\"block\":" << block << ",\"slot\":" << slot
              << ",\"wall_ms\":" << s.wall << ",\"prepare_ms\":" << s.prepare
              << ",\"issue_ms\":" << s.issue << ",\"wait_ms\":" << s.wait
              << ",\"status\":" << s.status << ",\"native_reason\":" << s.reason
              << ",\"chunks\":" << s.chunks << ",\"steps\":" << s.steps
              << ",\"internal_steps\":" << s.internal << ",\"trace_steps\":" << s.trace
              << ",\"actual_raw\":" << quoted(s.actual) << "}\n" << std::flush;
}
int main(int argc, char** argv) try {
    if (argc != 6) throw std::runtime_error("artifact binding corpus paired|exact|qwenputer blocks");
    const std::string mode = argv[4];
    if (mode != "paired" && mode != "exact" && mode != "qwenputer") throw std::runtime_error("mode");
    const unsigned blocks = std::stoul(argv[5]);
    if (blocks < 1 || blocks > 4) throw std::runtime_error("blocks");
#ifdef _WIN32
    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) throw std::runtime_error("priority");
#endif
    const auto begin = Clock::now();
    std::ifstream file(argv[3], std::ios::binary | std::ios::ate);
    if (!file || file.tellg() > 8192) throw std::runtime_error("corpus_size");
    file.seekg(0);
    uint32_t magic=0, count=0;
    read(file, magic); read(file, count);
    if (magic != 0x31424349 || count < 1 || count > 8) throw std::runtime_error("corpus_header");
    std::vector<Case> cases(count);
    for (auto& c : cases) {
        uint32_t n=0; read(file, n); read(file, c.request);
        if (n < 2 || n > 82) throw std::runtime_error("prefix_count");
        c.prefix.resize(n);
        if (!file.read(reinterpret_cast<char*>(c.prefix.data()), n*sizeof(qp::Input))) throw std::runtime_error("prefix_read");
    }
    if (file.peek() != std::char_traits<char>::eof()) throw std::runtime_error("corpus_trailing");
    qp::Artifact artifact; qp::MemoryPlan memory; std::string error;
    if (mode != "exact") {
        qp::Binding binding;
        std::ifstream identity(argv[2], std::ios::binary | std::ios::ate);
        if (!identity || identity.tellg() != sizeof(binding)) throw std::runtime_error("binding_size");
        identity.seekg(0); read(identity, binding);
        if (!qp::load_artifact(argv[1], binding, {}, artifact, memory, error)) throw std::runtime_error(error);
    }
    const auto host_ready = Clock::now();
    Stream stream; ck(cudaStreamCreateWithFlags(&stream.value, cudaStreamNonBlocking));
    const auto context_ready = Clock::now();
    std::size_t total=0, free_context=0, free_ready=0, free_final=0;
    ck(cudaMemGetInfo(&free_context, &total));
    qp::Executor executor;
    qp::Runtime runtime;
    ExactBuffers exact;
    if (mode != "exact") {
        if (!executor.init(artifact, stream.value, error) || !runtime.init(executor, artifact, {}, stream.value, error))
            throw std::runtime_error(error);
    }
    if (mode != "qwenputer") {
        ck(cudaMalloc(reinterpret_cast<void**>(&exact.input), sizeof(igc::ExactRequest)));
        ck(cudaMalloc(reinterpret_cast<void**>(&exact.output), sizeof(igc::ExactResult)));
    }
    ck(cudaStreamSynchronize(stream.value));
    const auto ready = Clock::now();
    ck(cudaMemGetInfo(&free_ready, &total));
    std::cout << std::setprecision(12);
    std::cout << "{\"kind\":\"setup\",\"mode\":" << quoted(mode)
              << ",\"host_load_ms\":" << ms(begin,host_ready)
              << ",\"context_ms\":" << ms(host_ready,context_ready)
              << ",\"module_ms\":" << ms(context_ready,ready)
              << ",\"setup_ms\":" << ms(begin,ready)
              << ",\"qwenputer_owned_bytes\":" << executor.allocated_bytes()+runtime.allocated_bytes()
              << ",\"exact_owned_bytes\":" << (exact.input ? sizeof(igc::ExactRequest)+sizeof(igc::ExactResult) : 0)
              << ",\"graph_allocation_delta_bytes\":" << runtime.graph_allocation_delta()
              << ",\"cuda_total_bytes\":" << total << ",\"cuda_free_context_bytes\":" << free_context
              << ",\"cuda_free_ready_bytes\":" << free_ready << "}\n" << std::flush;
    auto execute = [&](const char* arm, const Case& c) {
        Sample s;
        const auto t0=Clock::now();
        if (std::strcmp(arm,"exact") == 0) {
            igc::ExactResult output;
            ck(cudaMemcpyAsync(exact.input,&c.request,sizeof(c.request),cudaMemcpyHostToDevice,stream.value));
            const auto t1=Clock::now();
            if (!igc::execute_exact(exact.input,exact.output,1,stream.value,error)) throw std::runtime_error(error);
            ck(cudaMemcpyAsync(&output,exact.output,sizeof(output),cudaMemcpyDeviceToHost,stream.value));
            const auto t2=Clock::now();
            ck(cudaStreamSynchronize(stream.value));
            const auto t3=Clock::now();
            s.wall=ms(t0,t3); s.prepare=ms(t0,t1); s.issue=ms(t1,t2); s.wait=ms(t2,t3);
            s.status=static_cast<unsigned>(output.status);
            if (s.status==0) {
                if (output.value.length < 1 || output.value.length > igc::kMaxDigits) throw std::runtime_error("result_shape");
                if (output.value.negative) s.actual+='-';
                for (int i=output.value.length-1;i>=0;--i) {
                    if (output.value.digits[i]>9) throw std::runtime_error("result_digit");
                    s.actual+=char('0'+output.value.digits[i]);
                }
            }
        } else {
            if (!runtime.prepare(c.prefix,true,error)) throw std::runtime_error(error);
            auto last=Clock::now(); s.prepare=ms(t0,last);
            qp::ChunkResult result; std::vector<uint32_t> text;
            do {
                if (++s.chunks>4096) throw std::runtime_error("chunk_bound");
                const auto issue_begin=Clock::now();
                if (!runtime.enqueue_chunk(128,256,error)) throw std::runtime_error(error);
                const auto issue_end=Clock::now();
                if (!runtime.collect(result,error)) throw std::runtime_error(error);
                last=Clock::now();
                s.issue+=ms(issue_begin,issue_end); s.wait+=ms(issue_end,last);
                text.insert(text.end(),result.text.begin(),result.text.end());
            } while (result.reason==qp::RunReason::Budget || result.reason==qp::RunReason::Output);
            // The complete text-ID result has returned before this interval ends.
            const auto t3=Clock::now(); s.wall=ms(t0,t3);
            s.reason=static_cast<unsigned>(result.reason);
            s.status=result.reason==qp::RunReason::Success ? 0 : 1;
            s.steps=result.total_steps; s.internal=result.internal_steps; s.trace=result.trace_steps;
            for (auto id : text) {
                const auto bytes=artifact.token_bytes(id);
                s.actual.append(reinterpret_cast<const char*>(bytes.data()),bytes.size());
            }
        }
        return s;
    };
    if (mode=="paired") {
        // Untimed by the Python summary, but recorded explicitly; each measured
        // Qwenputer invocation still resets the machine and replays its prefix.
        for (unsigned i=0;i<count;++i) for (auto arm : {"qwenputer","exact"})
            emit("warmup",arm,i,0,0,execute(arm,cases[i]));
        for (unsigned i=0;i<count;++i) for (unsigned b=0;b<blocks;++b) {
            const char* order[]{"qwenputer","exact","exact","qwenputer"};
            for (unsigned slot=0;slot<4;++slot) emit("sample",order[slot],i,b,slot,execute(order[slot],cases[i]));
        }
    } else {
        emit("cold",mode.c_str(),0,0,0,execute(mode.c_str(),cases[0]));
    }
    ck(cudaMemGetInfo(&free_final,&total));
    uint64_t private_now=0, private_peak=0, working_peak=0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{}; counters.cb=sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),sizeof(counters)))
        throw std::runtime_error("process_memory");
    private_now=counters.PrivateUsage; private_peak=counters.PeakPagefileUsage; working_peak=counters.PeakWorkingSetSize;
#endif
    std::cout << "{\"kind\":\"memory\",\"process_private_commit_bytes\":" << private_now
              << ",\"process_private_commit_peak_bytes\":" << private_peak
              << ",\"working_set_peak_bytes\":" << working_peak
              << ",\"cuda_free_final_bytes\":" << free_final << "}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "component_benchmark: " << error.what() << '\n'; return 2;
}
