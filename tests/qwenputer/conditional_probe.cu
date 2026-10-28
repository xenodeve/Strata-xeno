// P0 capability diagnostic only: not a Qwenputer circuit or speed benchmark.
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

static void check(cudaError_t rc, const char* op) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(rc));
}
#define CUDA_CHECK(call) check((call), #call)
enum Reason { Running, Complete, OutputBound, Cancelled };
struct Params { unsigned limit, capacity, increment, epoch; unsigned long long work_cycles; };
struct Result { unsigned steps, checksum, epoch; Reason reason; };

__global__ void initialize(const Params* p, Result* r, unsigned* cancel, cudaGraphConditionalHandle condition) {
    *cancel = 0;
    *r = {0, 0, p->epoch, p->limit == 0 ? Complete : (p->capacity == 0 ? OutputBound : Running)};
    cudaGraphSetConditional(condition, r->reason == Running);
}
__global__ void step(const Params* p, Result* r, unsigned* cancel, cudaGraphConditionalHandle condition) {
    if (atomicAdd(cancel, 0u)) {
        r->reason = Cancelled; cudaGraphSetConditional(condition, 0); return;
    }
    r->steps += p->increment; r->checksum += r->steps;
    // Bounded work makes a cancellation race observable without a long kernel.
    const auto start = clock64();
    while (clock64() - start < p->work_cycles) {}
    if (r->steps >= p->limit) r->reason = Complete;
    else if (r->steps >= p->capacity) r->reason = OutputBound;
    cudaGraphSetConditional(condition, r->reason == Running);
}
__global__ void request_cancel(unsigned* cancel) { atomicExch(cancel, 1u); }

static std::string quote(const std::string& s) {
    std::ostringstream out; out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
        else out << c;
    }
    out << '"'; return out.str();
}
static const char* reason_name(Reason r) {
    switch (r) { case Complete: return "complete"; case OutputBound: return "output_bound";
        case Cancelled: return "cancelled"; default: return "running"; }
}
struct Case { std::string name; Result actual; unsigned expected_steps, expected_checksum;
    bool ok; double elapsed_ms, cancel_ms; };

struct Executor {
    Params* params = nullptr; Result* result = nullptr; unsigned* cancel = nullptr;
    cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
    cudaStream_t stream = nullptr, control = nullptr; cudaEvent_t initialized = nullptr;
    cudaGraphConditionalHandle condition{};
    bool created = false, instantiated = false;
    ~Executor() {
        if (stream) cudaStreamSynchronize(stream);
        if (control) cudaStreamSynchronize(control);
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        if (initialized) cudaEventDestroy(initialized);
        if (control) cudaStreamDestroy(control);
        if (stream) cudaStreamDestroy(stream);
        if (cancel) cudaFree(cancel);
        if (result) cudaFree(result);
        if (params) cudaFree(params);
    }
    void build() {
        CUDA_CHECK(cudaMalloc(&params, sizeof(Params))); CUDA_CHECK(cudaMalloc(&result, sizeof(Result)));
        CUDA_CHECK(cudaMalloc(&cancel, sizeof(unsigned)));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        int low, high; CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&low, &high));
        CUDA_CHECK(cudaStreamCreateWithPriority(&control, cudaStreamNonBlocking, high));
        CUDA_CHECK(cudaEventCreateWithFlags(&initialized, cudaEventDisableTiming));
        request_cancel<<<1, 1, 0, control>>>(cancel); // Warm control kernel before the cancellation case.
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(control));
        CUDA_CHECK(cudaGraphCreate(&graph, 0));
        CUDA_CHECK(cudaGraphConditionalHandleCreate(&condition, graph, 0, cudaGraphCondAssignDefault));
        void* args[] = {&params, &result, &cancel, &condition};
        cudaKernelNodeParams init_params{};
        init_params.func = reinterpret_cast<void*>(initialize);
        init_params.gridDim = dim3(1); init_params.blockDim = dim3(1); init_params.kernelParams = args;
        cudaGraphNode_t init_node, event_node, while_node, step_node;
        CUDA_CHECK(cudaGraphAddKernelNode(&init_node, graph, nullptr, 0, &init_params));
        CUDA_CHECK(cudaGraphAddEventRecordNode(&event_node, graph, &init_node, 1, initialized));
        cudaGraphNodeParams conditional{}; conditional.type = cudaGraphNodeTypeConditional;
        conditional.conditional.handle = condition; conditional.conditional.type = cudaGraphCondTypeWhile;
        conditional.conditional.size = 1;
        CUDA_CHECK(cudaGraphAddNode(&while_node, graph, &event_node, nullptr, 1, &conditional));
        cudaKernelNodeParams step_params = init_params; step_params.func = reinterpret_cast<void*>(step);
        CUDA_CHECK(cudaGraphAddKernelNode(&step_node, conditional.conditional.phGraph_out[0], nullptr, 0, &step_params));
        created = true;
        CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0)); instantiated = true;
    }
    Case run(const char* name, Params p, unsigned expected_steps, unsigned expected_checksum,
             Reason expected_reason, bool do_cancel = false) {
        CUDA_CHECK(cudaMemcpyAsync(params, &p, sizeof(p), cudaMemcpyHostToDevice, stream));
        const auto start = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
        auto requested = start;
        if (do_cancel) {
            CUDA_CHECK(cudaEventSynchronize(initialized));
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            requested = std::chrono::steady_clock::now();
            request_cancel<<<1, 1, 0, control>>>(cancel); CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (do_cancel) CUDA_CHECK(cudaStreamSynchronize(control));
        const auto finish = std::chrono::steady_clock::now();
        Result r{}; CUDA_CHECK(cudaMemcpy(&r, result, sizeof(r), cudaMemcpyDeviceToHost));
        bool ok = r.epoch == p.epoch && r.reason == expected_reason;
        if (do_cancel) ok = ok && r.steps > 0 && r.steps < p.limit && r.checksum == r.steps * (r.steps + 1) / 2;
        else ok = ok && r.steps == expected_steps && r.checksum == expected_checksum;
        return {name, r, expected_steps, expected_checksum, ok,
                std::chrono::duration<double, std::milli>(finish - start).count(),
                do_cancel ? std::chrono::duration<double, std::milli>(finish - requested).count() : 0};
    }
};

int main(int argc, char** argv) {
    std::string output, error, device_name, device_uuid;
    int device = 0, device_count = 0, runtime = 0, driver = 0, major = 0, minor = 0;
    bool negative = false, created = false, instantiated = false, ok = false, context_switch_checked = false;
    size_t free_before = 0, free_after = 0, total = 0;
    std::vector<Case> cases;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json" && i + 1 < argc) output = argv[++i];
        else if (arg == "--device" && i + 1 < argc) device = std::stoi(argv[++i]);
        else if (arg == "--negative-control") negative = true;
        else { std::cerr << "unknown or incomplete argument: " << arg << '\n'; return 2; }
    }
    if (output.empty()) { std::cerr << "--json path is required\n"; return 2; }
    try {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        device_name = prop.name; major = prop.major; minor = prop.minor;
        std::ostringstream uuid;
        for (int i = 0; i < 16; ++i) uuid << std::hex << std::setw(2) << std::setfill('0') << (unsigned(prop.uuid.bytes[i]) & 255);
        device_uuid = uuid.str();
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime)); CUDA_CHECK(cudaDriverGetVersion(&driver));
        CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
        {
            Executor executor;
            try { executor.build(); }
            catch (...) { created = executor.created; instantiated = executor.instantiated; throw; }
            created = executor.created; instantiated = executor.instantiated;
            unsigned epoch = 0, increment = negative ? 2 : 1;
            auto run = [&](const char* name, unsigned limit, unsigned cap, unsigned n, unsigned sum, Reason reason) {
                cases.push_back(executor.run(name, {limit, cap, increment, ++epoch, 0}, n, sum, reason));
            };
            run("zero", 0, 4096, 0, 0, Complete); run("one", 1, 4096, 1, 1, Complete);
            run("n", 17, 4096, 17, 153, Complete); run("repeat_n", 5, 4096, 5, 15, Complete);
            run("repeat_zero", 0, 4096, 0, 0, Complete); run("output_bound", 17, 3, 3, 6, OutputBound);
            cases.push_back(executor.run("cancel", {4096, 4096, increment, ++epoch, 100000}, 0, 0, Cancelled, true));
            if (device_count > 1) {
                // No circuit or kernel runs on the display device. Check that a
                // host current-device change cannot leave this graph on stale state.
                CUDA_CHECK(cudaSetDevice((device + 1) % device_count));
                CUDA_CHECK(cudaSetDevice(device));
                cases.push_back(executor.run("after_device_switch", {1, 4096, 1, ++epoch, 0}, 1, 1, Complete));
                context_switch_checked = true;
            }
        }
        // Destroy and recreate the standalone graph/state, checking reset behavior.
        // Allocator addresses may be reused; this does not test relocation. Handles
        // are never cloned or reused across independently instantiated graphs.
        {
            Executor rebuilt; rebuilt.build();
            cases.push_back(rebuilt.run("rebuilt_one", {1, 4096, 1, 101, 0}, 1, 1, Complete));
            cases.push_back(rebuilt.run("rebuilt_zero", {0, 4096, 1, 102, 0}, 0, 0, Complete));
        }
        CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
        ok = true; for (const auto& c : cases) ok = ok && c.ok;
    } catch (const std::exception& ex) { error = ex.what(); }
    unsigned long long private_bytes = 0, peak_working = 0;
    std::string cudart_module;
#ifdef _WIN32
    if (auto module = GetModuleHandleW(L"cudart64_13.dll")) {
        wchar_t path[32768]{};
        const auto size = GetModuleFileNameW(module, path, 32768);
        if (size && size < 32768) {
            const auto bytes = WideCharToMultiByte(CP_UTF8, 0, path, size, nullptr, 0, nullptr, nullptr);
            cudart_module.resize(bytes);
            WideCharToMultiByte(CP_UTF8, 0, path, size, cudart_module.data(), bytes, nullptr, nullptr);
        }
    }
    PROCESS_MEMORY_COUNTERS_EX memory{}; memory.cb = sizeof(memory);
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof(memory))) {
        private_bytes = memory.PrivateUsage; peak_working = memory.PeakWorkingSetSize;
    }
#endif
    std::ofstream f(output);
    if (!f) { std::cerr << "cannot write report\n"; return 2; }
    f << std::boolalpha << std::setprecision(9) << "{\"ok\":" << ok
      << ",\"graph_created\":" << created << ",\"graph_instantiated\":" << instantiated
      << ",\"device_count\":" << device_count << ",\"context_switch_checked\":" << context_switch_checked
      << ",\"device\":" << device << ",\"device_name\":" << quote(device_name)
      << ",\"device_uuid_hex\":" << quote(device_uuid) << ",\"sm\":" << major * 10 + minor
      << ",\"runtime\":" << runtime << ",\"driver_api\":" << driver
      << ",\"headers\":" << CUDART_VERSION << ",\"cudart_module\":" << quote(cudart_module)
      << ",\"vram_free_before\":" << free_before << ",\"vram_free_after\":" << free_after
      << ",\"vram_total\":" << total << ",\"process_private_at_end\":" << private_bytes
      << ",\"peak_working_set\":" << peak_working << ",\"negative_control\":" << negative
      << ",\"error\":" << quote(error) << ",\"cases\":[";
    bool first = true;
    for (const auto& c : cases) {
        if (!first) f << ','; first = false;
        f << "{\"name\":" << quote(c.name) << ",\"steps\":" << c.actual.steps
          << ",\"expected_steps\":" << c.expected_steps << ",\"checksum\":" << c.actual.checksum
          << ",\"expected_checksum\":" << c.expected_checksum << ",\"epoch\":" << c.actual.epoch
          << ",\"reason\":" << quote(reason_name(c.actual.reason)) << ",\"ok\":" << c.ok
          << ",\"elapsed_ms\":" << c.elapsed_ms << ",\"cancel_request_to_completion_ms\":" << c.cancel_ms << '}';
    }
    f << "]}\n";
    std::cout << "DONE probe ok=" << ok << " report=" << output << '\n';
    return ok ? 0 : 1;
}
