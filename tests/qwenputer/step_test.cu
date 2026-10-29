#include "strata/core/qwenputer.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

using namespace strata::qwenputer;
static void ck(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
template<class T> static std::vector<T> read(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("fixture_io");
    const auto size = static_cast<std::size_t>(file.tellg());
    if (size % sizeof(T) || size > (1ull << 30)) throw std::runtime_error("fixture_size");
    std::vector<T> data(size / sizeof(T));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size)))
        throw std::runtime_error("fixture_io");
    return data;
}
int main(int argc, char** argv) try {
    if (argc < 6) { std::cerr << "artifact binding inputs residual final-state [--initial path] [--dump path] [--max-steps N] [--negative-control]\n"; return 2; }
    const char* initial_path = nullptr;
    const char* dump_path = nullptr;
    std::size_t max_steps = 200000;
    bool negative = false;
    for (int i = 6; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--negative-control") negative = true;
        else if (i + 1 < argc && arg == "--initial") initial_path = argv[++i];
        else if (i + 1 < argc && arg == "--dump") dump_path = argv[++i];
        else if (i + 1 < argc && arg == "--max-steps") max_steps = std::stoull(argv[++i]);
        else throw std::runtime_error("unknown_test_argument");
    }
    const auto binding_data = read<std::uint8_t>(argv[2]);
    if (binding_data.size() != sizeof(Binding)) return 2;
    Binding binding;
    std::memcpy(&binding, binding_data.data(), sizeof(binding));
    Artifact artifact;
    MemoryPlan memory;
    std::string error;
    if (!load_artifact(argv[1], binding, {}, artifact, memory, error)) throw std::runtime_error(error);
    if (negative) {
        for (const auto& f : artifact.ffns) if (unsigned(f.out) == artifact.config[15])
            for (int j = 0; j < f.count; ++j) artifact.ffn_terms[f.start + j].coefficient *= 2;
    }
    const auto inputs = read<Input>(argv[3]);
    const auto expected = read<float>(argv[4]);
    const auto final_state = read<float>(argv[5]);
    const auto dimensions = artifact.dims.size();
    if (inputs.empty() || inputs.size() > 200000 || expected.size() != inputs.size() * dimensions
        || final_state.size() != artifact.config[6]) throw std::runtime_error("fixture_geometry");
    const auto steps = std::min(inputs.size(), max_steps);
    if (steps == 0) throw std::runtime_error("fixture_empty");
    std::vector<float> initial;
    if (initial_path) {
        initial = read<float>(initial_path);
        if (initial.size() != artifact.config[6]) throw std::runtime_error("initial_geometry");
    }
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Executor executor;
    if (!executor.init(artifact, stream, error)) throw std::runtime_error("executor_init:" + error);
    cudaGraph_t graph;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    if (!executor.enqueue_step(stream, error)) throw std::runtime_error("enqueue:" + error);
    ck(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t instance;
    ck(cudaGraphInstantiate(&instance, graph, 0));
    Input* tokens = nullptr;
    float* residuals = nullptr;
    StepStatus* statuses = nullptr;
    ck(cudaMalloc(&tokens, inputs.size() * sizeof(Input)));
    ck(cudaMalloc(&residuals, steps * dimensions * sizeof(float)));
    ck(cudaMalloc(&statuses, steps * sizeof(StepStatus)));
    ck(cudaMemcpyAsync(tokens, inputs.data(), inputs.size() * sizeof(Input), cudaMemcpyHostToDevice, stream));
    if (!executor.enqueue_reset(stream, error)) throw std::runtime_error(error);
    if (initial_path) ck(cudaMemcpyAsync(executor.device_state(), initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    cudaEvent_t start, stop;
    ck(cudaEventCreate(&start)); ck(cudaEventCreate(&stop));
    ck(cudaEventRecord(start, stream));
    for (std::size_t i = 0; i < steps; ++i) {
        ck(cudaMemcpyAsync(executor.device_input(), tokens + i, sizeof(Input), cudaMemcpyDeviceToDevice, stream));
        ck(cudaGraphLaunch(instance, stream));
        ck(cudaMemcpyAsync(residuals + i * dimensions, executor.device_residual(), dimensions * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream));
        ck(cudaMemcpyAsync(statuses + i, executor.device_status(), sizeof(StepStatus), cudaMemcpyDeviceToDevice, stream));
    }
    ck(cudaEventRecord(stop, stream));
    std::vector<float> actual(steps * dimensions), state(final_state.size());
    std::vector<StepStatus> predictions(steps);
    StepStatus status;
    ck(cudaMemcpyAsync(actual.data(), residuals, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    ck(cudaMemcpyAsync(state.data(), executor.device_state(), state.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    ck(cudaMemcpyAsync(&status, executor.device_status(), sizeof(status), cudaMemcpyDeviceToHost, stream));
    ck(cudaMemcpyAsync(predictions.data(), statuses, steps * sizeof(StepStatus), cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream));
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            if (mismatches < 8) std::cerr << "residual step=" << i / dimensions << " dim=" << i % dimensions
                << " expected=" << expected[i] << " actual=" << actual[i] << '\n';
            ++mismatches;
        }
    }
    for (std::size_t i = 0; steps == inputs.size() && i < state.size(); ++i) {
        if (state[i] != final_state[i]) {
            if (mismatches < 8) std::cerr << "state " << i << " expected=" << final_state[i] << " actual=" << state[i] << '\n';
            ++mismatches;
        }
    }
    std::size_t owned = 0;
    std::string text;
    for (std::size_t i = 0; i < steps; ++i) if (predictions[i].owned) {
        ++owned;
        const auto next = predictions[i].next;
        if (i + 1 < inputs.size() && (next.kind != inputs[i + 1].kind || next.id != inputs[i + 1].id)) {
            if (mismatches < 8) std::cerr << "prediction step=" << i << " differs\n";
            ++mismatches;
        }
        if (inputs[i].kind == TokenKind::Trace && next.kind == TokenKind::Text) {
            if (next.id >= artifact.printable_index.size()) ++mismatches;
            else {
                const auto p = artifact.printable_index[next.id];
                if (p >= 0 && p < 256 && text.size() < 4096) text += char(p);
            }
        }
    }
    if (dump_path) {
        std::vector<float> lags(memory.lag_each / sizeof(float));
        ck(cudaMemcpy(lags.data(), executor.device_lags(), memory.lag_each, cudaMemcpyDeviceToHost));
        std::ofstream out(dump_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(lags.data()), std::streamsize(memory.lag_each));
        if (!out) throw std::runtime_error("dump_io");
    }
    float elapsed = 0;
    ck(cudaEventElapsedTime(&elapsed, start, stop));
    std::cout << "steps=" << steps << " residual_values=" << actual.size() << " state_values=" << state.size()
              << " mismatches=" << mismatches << " error=" << status.error
              << " owned_predictions=" << owned << " device_buffers=" << executor.allocated_bytes()
              << " diagnostic_replay_ms=" << elapsed << "\nnative_predicted_text=" << text << '\n';
    ck(cudaEventDestroy(start)); ck(cudaEventDestroy(stop));
    ck(cudaFree(residuals)); ck(cudaFree(tokens)); ck(cudaFree(statuses));
    ck(cudaGraphExecDestroy(instance)); ck(cudaGraphDestroy(graph)); ck(cudaStreamDestroy(stream));
    return mismatches || status.error ? 1 : 0;
} catch (const std::exception& e) {
    std::cerr << "FAIL " << e.what() << '\n'; return 1;
}
