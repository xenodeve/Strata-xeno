// Toolchain diagnostic only: no model, circuit, firmware or arithmetic oracle.
// Run on logical device 0 after explicitly selecting the primary GPU by UUID.
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK(call) do { \
    const auto result_ = (call); \
    if (result_ != cudaSuccess) { \
        std::printf("FAIL %s: %d %s\n", #call, int(result_), cudaGetErrorString(result_)); \
        return 1; \
    } \
} while (0)

__global__ void initialize(unsigned* count, cudaGraphConditionalHandle condition) {
    *count = 0;
    cudaGraphSetConditional(condition, 1);
}

__global__ void work(unsigned* data) {
    if (threadIdx.x == 0) data[2918] = 0;
    __syncthreads();
    for (unsigned i = threadIdx.x; i < 2918; i += blockDim.x) data[i] = 0;
    __syncthreads();
    for (unsigned i = threadIdx.x; i < 550; i += blockDim.x) data[i] = i;
}

__global__ void finish(unsigned* count, cudaGraphConditionalHandle condition) {
    ++*count;
    cudaGraphSetConditional(condition, *count < 6);
}

int main() {
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    unsigned *data, *count;
    CHECK(cudaMalloc(&data, 2919 * sizeof(unsigned)));
    CHECK(cudaMalloc(&count, sizeof(unsigned)));
    for (int i = 0; i < 6; ++i) {
        work<<<1, 256, 0, stream>>>(data);
        CHECK(cudaStreamSynchronize(stream));
    }
    std::puts("regular_barriers_pass");
    std::fflush(stdout);

    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphConditionalHandle condition;
    CHECK(cudaGraphCreate(&graph, 0));
    CHECK(cudaGraphConditionalHandleCreate(&condition, graph, 0, cudaGraphCondAssignDefault));
    void* arguments[]{&count, &condition};
    cudaKernelNodeParams kernel{};
    kernel.func = reinterpret_cast<void*>(initialize);
    kernel.gridDim = dim3(1);
    kernel.blockDim = dim3(1);
    kernel.kernelParams = arguments;
    cudaGraphNode_t initial, loop, body_work, body_finish;
    CHECK(cudaGraphAddKernelNode(&initial, graph, nullptr, 0, &kernel));
    cudaGraphNodeParams node{};
    node.type = cudaGraphNodeTypeConditional;
    node.conditional.handle = condition;
    node.conditional.type = cudaGraphCondTypeWhile;
    node.conditional.size = 1;
    CHECK(cudaGraphAddNode(&loop, graph, &initial, nullptr, 1, &node));
    const auto body = node.conditional.phGraph_out[0];
    kernel.func = reinterpret_cast<void*>(work);
    kernel.blockDim = dim3(256);
    void* work_arguments[]{&data};
    kernel.kernelParams = work_arguments;
    CHECK(cudaGraphAddKernelNode(&body_work, body, nullptr, 0, &kernel));
    kernel.func = reinterpret_cast<void*>(finish);
    kernel.blockDim = dim3(1);
    kernel.kernelParams = arguments;
    CHECK(cudaGraphAddKernelNode(&body_finish, body, &body_work, 1, &kernel));
    CHECK(cudaGraphInstantiate(&instance, graph, 0));
    CHECK(cudaGraphLaunch(instance, stream));
    CHECK(cudaStreamSynchronize(stream));
    unsigned actual;
    CHECK(cudaMemcpy(&actual, count, sizeof(actual), cudaMemcpyDeviceToHost));
    std::printf("conditional_barriers_steps=%u\n", actual);
    CHECK(cudaGraphExecDestroy(instance));
    CHECK(cudaGraphDestroy(graph));
    CHECK(cudaFree(data));
    CHECK(cudaFree(count));
    CHECK(cudaStreamDestroy(stream));
    return actual == 6 ? 0 : 1;
}
