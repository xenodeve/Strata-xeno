#include "strata/core/igc_integer_extended.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strata::igc::integer_extended;
static void check(cudaError_t c) {
    if (c != cudaSuccess) throw std::runtime_error(cudaGetErrorString(c));
}
static void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
static strata::igc::Integer integer(const char* text) {
    strata::igc::Integer value;
    value.negative = *text == '-';
    if (value.negative) ++text;
    const auto n = std::strlen(text);
    require(n > 0 && n <= 62, "literal_digits");
    value.length = static_cast<uint8_t>(n);
    for (size_t i = 0; i < n; ++i) {
        require(text[n - 1 - i] >= '0' && text[n - 1 - i] <= '9', "literal_digit");
        value.digits[i] = static_cast<uint8_t>(text[n - 1 - i] - '0');
    }
    return value;
}
struct Device {
    cudaStream_t stream = nullptr;
    Request* input = nullptr;
    Result* output = nullptr;
    Device() {
        try {
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            check(cudaMalloc(&input, sizeof(Request) * strata::igc::kMaxBatch));
            check(cudaMalloc(&output, sizeof(Result) * strata::igc::kMaxBatch));
        } catch (...) { release(); throw; }
    }
    void release() {
        if (stream) cudaStreamSynchronize(stream);
        if (output) cudaFree(output);
        if (input) cudaFree(input);
        if (stream) cudaStreamDestroy(stream);
        output = nullptr; input = nullptr; stream = nullptr;
    }
    ~Device() { release(); }
    Result run(Request request) {
        check(cudaMemcpyAsync(input, &request, sizeof(request), cudaMemcpyHostToDevice, stream));
        std::string error;
        require(execute(input, output, 1, stream, error), error.c_str());
        Result actual;
        check(cudaMemcpyAsync(&actual, output, sizeof(actual), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        return actual;
    }
};
static void literal(Device& device, Operation op, const char* lhs, const char* rhs,
                    const char* expected, const char* name) {
    Request request;
    request.operation = op; request.invocation = 7; request.position = 65;
    request.lhs = integer(lhs); request.rhs = integer(rhs);
    const auto actual = device.run(request);
    const auto value = integer(expected);
    require(actual.version == kAbi && actual.invocation == 7 && actual.position == 65
        && actual.status == Status::Success && !std::memcmp(&actual.value, &value, sizeof(value)), name);
}
static void failure(Device& device, Operation op, const char* lhs, const char* rhs,
                    Status status, const char* name) {
    Request request;
    request.operation = op; request.invocation = 8; request.position = 66;
    request.lhs = integer(lhs); request.rhs = integer(rhs);
    const auto actual = device.run(request);
    const strata::igc::Integer absent{};
    require(actual.version == kAbi && actual.invocation == 8 && actual.position == 66
        && actual.status == status && !std::memcmp(&actual.value, &absent, sizeof(absent)), name);
}
static void compatibility(Device& device) {
    strata::igc::ExactRequest old;
    old.invocation = 9; old.position = 2ull << 40;
    old.lhs = integer("31"); old.rhs = integer("-24");
    old.operation = strata::igc::Operation::Multiply;
    check(cudaMemcpyAsync(device.input, &old, sizeof(old), cudaMemcpyHostToDevice, device.stream));
    std::string error;
    require(execute(device.input, device.output, 1, device.stream, error), error.c_str());
    Result foreign;
    check(cudaMemcpyAsync(&foreign, device.output, sizeof(foreign), cudaMemcpyDeviceToHost, device.stream));
    check(cudaStreamSynchronize(device.stream));
    const strata::igc::Integer absent{};
    require(foreign.status == Status::Malformed && !std::memcmp(&foreign.value, &absent, sizeof(absent)), "abi1_refused_by_abi2");
    require(strata::igc::execute_exact(reinterpret_cast<strata::igc::ExactRequest*>(device.input),
        reinterpret_cast<strata::igc::ExactResult*>(device.output), 1, device.stream, error), error.c_str());
    strata::igc::ExactResult legacy;
    check(cudaMemcpyAsync(&legacy, device.output, sizeof(legacy), cudaMemcpyDeviceToHost, device.stream));
    check(cudaStreamSynchronize(device.stream));
    const auto product = integer("-744");
    require(legacy.version == 1 && legacy.invocation == 9 && legacy.position == old.position
        && legacy.status == strata::igc::Status::Success && !std::memcmp(&legacy.value, &product, sizeof(product)), "abi1_multiply_unchanged");
    Request extended;
    extended.invocation = 10; extended.position = 21;
    extended.lhs = integer("-7"); extended.rhs = integer("3");
    check(cudaMemcpyAsync(device.input, &extended, sizeof(extended), cudaMemcpyHostToDevice, device.stream));
    require(strata::igc::execute_exact(reinterpret_cast<strata::igc::ExactRequest*>(device.input),
        reinterpret_cast<strata::igc::ExactResult*>(device.output), 1, device.stream, error), error.c_str());
    check(cudaMemcpyAsync(&legacy, device.output, sizeof(legacy), cudaMemcpyDeviceToHost, device.stream));
    check(cudaStreamSynchronize(device.stream));
    require(legacy.status == strata::igc::Status::Malformed && !std::memcmp(&legacy.value, &absent, sizeof(absent)), "abi2_refused_by_abi1");
}
// Native test executable receives only requests. Expected values are computed
// separately by the offline Python standard-library oracle, never by this path.
static void file_requests(Device& device, const char* input_path, const char* output_path,
                          unsigned batch, unsigned repeats) {
    require(batch > 0 && batch <= strata::igc::kMaxBatch && repeats <= 64, "file_bounds");
    std::ifstream input(input_path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(input), "file_open");
    const auto bytes = input.tellg();
    require(bytes > 0 && bytes <= 4096 * sizeof(Request) && bytes % sizeof(Request) == 0, "file_size");
    std::vector<Request> requests(static_cast<size_t>(bytes) / sizeof(Request));
    std::vector<Result> results(requests.size());
    input.seekg(0); input.read(reinterpret_cast<char*>(requests.data()), bytes);
    require(static_cast<bool>(input), "file_read");
    for (size_t offset = 0; offset < requests.size(); offset += batch) {
        const auto count = static_cast<uint32_t>(std::min<size_t>(batch, requests.size() - offset));
        check(cudaMemcpyAsync(device.input, requests.data() + offset, count * sizeof(Request), cudaMemcpyHostToDevice, device.stream));
        check(cudaMemsetAsync(device.output, 0x5a, count * sizeof(Result), device.stream));
        std::string error;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t executable = nullptr;
        try {
            if (repeats) check(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal));
            require(execute(device.input, device.output, count, device.stream, error), error.c_str());
            if (repeats) {
                check(cudaStreamEndCapture(device.stream, &graph));
                check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                for (unsigned i = 0; i < repeats; ++i) check(cudaGraphLaunch(executable, device.stream));
            }
            check(cudaMemcpyAsync(results.data() + offset, device.output, count * sizeof(Result), cudaMemcpyDeviceToHost, device.stream));
            check(cudaStreamSynchronize(device.stream));
        } catch (...) {
            if (executable) cudaGraphExecDestroy(executable);
            if (graph) cudaGraphDestroy(graph);
            throw;
        }
        if (executable) check(cudaGraphExecDestroy(executable));
        if (graph) check(cudaGraphDestroy(graph));
    }
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(results.data()), results.size() * sizeof(Result));
    require(static_cast<bool>(output), "file_write");
    std::cout << "PASS native requests=" << requests.size() << " batch=" << batch << " graph_repeats=" << repeats << '\n';
}
int main(int argc, char** argv) try {
    Device device;
    if (argc == 5) {
        file_requests(device, argv[1], argv[2], std::stoul(argv[3]), std::stoul(argv[4]));
        return 0;
    }
    require(argc == 1, "arguments");
    literal(device, Operation::FloorDivide, "-7", "3", "-3", "negative_floor_division_literal");
    literal(device, Operation::FloorDivide, "7", "-3", "-3", "negative_divisor_floor");
    literal(device, Operation::FloorDivide, "-7", "-3", "2", "positive_floor");
    literal(device, Operation::FloorDivide, "0", "-3", "0", "zero_floor");
    literal(device, Operation::FloorDivide, "-6", "3", "-2", "exact_floor");
    literal(device, Operation::Modulo, "-7", "3", "2", "negative_modulo_literal");
    literal(device, Operation::Modulo, "7", "-3", "-2", "negative_divisor_modulo");
    literal(device, Operation::Modulo, "-7", "-3", "-1", "both_negative_modulo");
    literal(device, Operation::Modulo, "-6", "3", "0", "exact_modulo");
    literal(device, Operation::Gcd, "-48", "18", "6", "signed_gcd_literal");
    literal(device, Operation::Gcd, "0", "0", "0", "zero_gcd");
    literal(device, Operation::Gcd, "0", "-91", "91", "one_zero_gcd");
    literal(device, Operation::Gcd, "10000000000000000000000000000000000000000000000000000000000000", "10", "10", "max_length_gcd");
    literal(device, Operation::Power, "-3", "5", "-243", "signed_power_literal");
    literal(device, Operation::Power, "0", "0", "1", "zero_power_zero");
    literal(device, Operation::Power, "-1", "-99999999999999999999999999999999999999999999999999999999999999", "-1", "negative_unit_power");
    literal(device, Operation::Power, "1", "99999999999999999999999999999999999999999999999999999999999999", "1", "huge_unit_power");
    literal(device, Operation::Power, "10", "61", "10000000000000000000000000000000000000000000000000000000000000", "max_length_power");
    failure(device, Operation::Power, "2", "206", Status::Overflow, "power_overflow");
    failure(device, Operation::Power, "2", "-1", Status::Unsupported, "rational_power_not_integer");
    failure(device, Operation::Power, "0", "-1", Status::DomainError, "zero_negative_power");
    failure(device, Operation::FloorDivide, "1", "0", Status::DomainError, "divide_zero");
    failure(device, Operation::Modulo, "1", "0", Status::DomainError, "modulo_zero");
    compatibility(device);
    std::string error;
    require(!execute(device.input, reinterpret_cast<Result*>(device.input), 1, device.stream, error), "overlap_must_be_rejected");
    require(!execute(device.input, reinterpret_cast<Result*>(reinterpret_cast<char*>(device.input) + 159), 1, device.stream, error), "tail_overlap_must_be_rejected");
    require(!execute(device.input, device.output, 0, device.stream, error), "zero_batch_rejected");
    require(!execute(device.input, device.output, 65, device.stream, error), "oversize_batch_rejected");
    require(!execute(device.input, device.output, 1, nullptr, error), "default_stream_rejected");
    require(!execute(device.input, device.output, 1, cudaStreamLegacy, error), "legacy_stream_rejected");
    require(!execute(device.input, device.output, 1, cudaStreamPerThread, error), "perthread_stream_rejected");
    std::cout << "PASS extended integer floor/modulo/gcd/power literals\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
