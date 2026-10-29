#include "strata/core/igc_exact.hpp"
#include <cuda_runtime.h>

namespace strata::igc {
namespace {
__device__ bool valid(const Integer& value) {
    if (!value.length || value.length > kMaxDigits || value.negative > 1) return false;
    for (int i = 0; i < kMaxDigits; ++i)
        if (value.digits[i] > 9 || (i >= value.length && value.digits[i] != 0)) return false;
    if (value.length > 1 && value.digits[value.length - 1] == 0) return false;
    return !(value.length == 1 && value.digits[0] == 0 && value.negative);
}
__device__ bool zero(const Integer& value) { return value.length == 1 && value.digits[0] == 0; }
__device__ int compare(const Integer& a, const Integer& b) {
    if (a.length != b.length) return a.length > b.length ? 1 : -1;
    for (int i = a.length - 1; i >= 0; --i)
        if (a.digits[i] != b.digits[i]) return a.digits[i] > b.digits[i] ? 1 : -1;
    return 0;
}
struct Work {
    int length = 1, negative = 0;
    int digits[2 * kMaxDigits + 1]{};
};
__device__ void subtract_magnitude(const Integer& a, const Integer& b, Work& result) {
    int borrow = 0;
    result.length = a.length;
    for (int i = 0; i < a.length; ++i) {
        int digit = a.digits[i] - (i < b.length ? b.digits[i] : 0) - borrow;
        borrow = digit < 0;
        if (borrow) digit += 10;
        result.digits[i] = digit;
    }
}
__device__ ExactResult calculate(const ExactRequest& request) {
    ExactResult result{};
    result.invocation = request.invocation;
    result.position = request.position;
    if (request.version != kExactAbi || request.reserved || !request.invocation) return result;
    if (request.domain != kIntegerDomain || (request.operation != Operation::Add &&
            request.operation != Operation::Subtract && request.operation != Operation::Multiply)) {
        result.status = Status::Unsupported;
        return result;
    }
    const Integer& a = request.lhs;
    const Integer& b = request.rhs;
    if (!valid(a) || !valid(b)) return result;
    Work work{};
    if (request.operation == Operation::Multiply) {
        work.length = a.length + b.length;
        work.negative = a.negative != b.negative;
        for (int i = 0; i < a.length; ++i)
            for (int j = 0; j < b.length; ++j) work.digits[i + j] += a.digits[i] * b.digits[j];
        for (int i = 0; i < 2 * kMaxDigits; ++i) {
            work.digits[i + 1] += work.digits[i] / 10;
            work.digits[i] %= 10;
        }
    } else {
        const int b_negative = request.operation == Operation::Subtract && !zero(b) ? !b.negative : b.negative;
        if (a.negative == b_negative) {
            const int length = a.length > b.length ? a.length : b.length;
            int carry = 0;
            work.negative = a.negative;
            for (int i = 0; i < length; ++i) {
                const int digit = (i < a.length ? a.digits[i] : 0) + (i < b.length ? b.digits[i] : 0) + carry;
                work.digits[i] = digit % 10;
                carry = digit / 10;
            }
            work.digits[length] = carry;
            work.length = length + (carry != 0);
        } else {
            const int order = compare(a, b);
            if (order >= 0) {
                subtract_magnitude(a, b, work);
                work.negative = a.negative;
            } else {
                subtract_magnitude(b, a, work);
                work.negative = b_negative;
            }
        }
    }
    while (work.length > 1 && work.digits[work.length - 1] == 0) --work.length;
    if (work.length > kMaxDigits) {
        result.status = Status::Overflow;
        return result;
    }
    result.status = Status::Success;
    result.value.length = static_cast<uint8_t>(work.length);
    result.value.negative = static_cast<uint8_t>(work.digits[0] == 0 && work.length == 1 ? 0 : work.negative);
    for (int i = 0; i < work.length; ++i) result.value.digits[i] = static_cast<uint8_t>(work.digits[i]);
    return result;
}
__global__ void execute_kernel(const ExactRequest* input, ExactResult* output, uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = calculate(input[i]);
}
__global__ void broadcast_kernel(ExactResult* output,uint32_t count) {
    const uint32_t row=blockIdx.x*blockDim.x+threadIdx.x+1;
    if(row<count)output[row]=output[0];
}
}

bool execute_exact(const ExactRequest* input, ExactResult* output, uint32_t count,
                   void* stream, std::string& error) {
    const auto created_stream = static_cast<cudaStream_t>(stream);
    if (!input || !output || !count || count > kMaxBatch || !created_stream ||
            created_stream == cudaStreamLegacy || created_stream == cudaStreamPerThread) {
        error = "exact_launch_arguments";
        return false;
    }
    execute_kernel<<<(count + 63) / 64, 64, 0, created_stream>>>(input, output, count);
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess) { error = cudaGetErrorString(status); return false; }
    error.clear();
    return true;
}
bool broadcast_exact_result(ExactResult* output,uint32_t count,void* stream,std::string& error) {
    const auto created_stream=static_cast<cudaStream_t>(stream);
    if(!output || !count || count>kMaxBatch || !created_stream || created_stream==cudaStreamLegacy || created_stream==cudaStreamPerThread) {
        error="exact_broadcast_arguments";return false;
    }
    if(count>1)broadcast_kernel<<<1,64,0,created_stream>>>(output,count);
    const auto status=cudaPeekAtLastError();
    if(status!=cudaSuccess){error=cudaGetErrorString(status);return false;}
    error.clear();return true;
}
} // namespace strata::igc
