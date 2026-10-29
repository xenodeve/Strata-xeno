#include "strata/core/igc_rational_exact.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>
#include <bit>
#include <fstream>
#include <vector>
#include <iostream>
#include <stdexcept>
using namespace strata::igc::rational_exact;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
static strata::igc::Integer integer(const char* text){
    strata::igc::Integer out;out.negative=*text=='-';if(out.negative)++text;
    const auto n=std::strlen(text);require(n && n<=62,"literal_length");out.length=static_cast<uint8_t>(n);
    for(size_t i=0;i<n;++i){require(text[n-1-i]>='0' && text[n-1-i]<='9',"literal_digit");out.digits[i]=text[n-1-i]-'0';}
    return out;
}
struct Device {
    cudaStream_t stream=nullptr;Request* input=nullptr;Result* output=nullptr;
    Device(){try{check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));check(cudaMalloc(&input,64*sizeof(Request)));check(cudaMalloc(&output,64*sizeof(Result)));}catch(...){release();throw;}}
    void release(){if(stream)cudaStreamSynchronize(stream);if(output)cudaFree(output);if(input)cudaFree(input);if(stream)cudaStreamDestroy(stream);stream=nullptr;input=nullptr;output=nullptr;}
    ~Device(){release();}
    Result run(const Request& request){
        check(cudaMemcpyAsync(input,&request,sizeof(request),cudaMemcpyHostToDevice,stream));std::string error;
        require(execute(input,output,1,stream,error),error.c_str());Result result;
        check(cudaMemcpyAsync(&result,output,sizeof(result),cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));return result;
    }
};
static void literal(Device& device,Operation operation,const char* an,const char* ad,const char* bn,const char* bd,
                    const char* en,const char* ed,const char* name){
    Request request;request.invocation=11;request.position=65;request.operation=operation;
    request.lhs.numerator=integer(an);request.lhs.denominator=integer(ad);
    request.rhs.numerator=integer(bn);request.rhs.denominator=integer(bd);
    const auto actual=device.run(request);Rational expected{integer(en),integer(ed)};
    require(actual.version==kAbi && actual.invocation==11 && actual.position==65 && actual.status==Status::Success
        && !std::memcmp(&actual.value,&expected,sizeof(expected)),name);
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
int main(int argc,char** argv)try{
    static_assert(std::endian::native==std::endian::little);
    Device device;
    if(argc==5){file_requests(device,argv[1],argv[2],std::stoul(argv[3]),std::stoul(argv[4]));return 0;}
    const std::string mode=argc==2?argv[1]:"all";
    require(argc<=2 && (mode=="all" || mode=="subtract" || mode=="multiply" || mode=="divide" || mode=="decimal"),"test_arguments");
    literal(device,Operation::Add,"1","3","1","6","1","2","one_third_plus_one_sixth");
    literal(device,Operation::Add,"2","4","0","17","1","2","unreduced_operand");
    constexpr const char* large="99999999999999999999999999999999999999999999999999999999999999";
    literal(device,Operation::Add,large,large,large,large,"2","1","reduce_125_digit_intermediate");
    if(mode=="all" || mode=="subtract"){
        literal(device,Operation::Subtract,"1","3","1","2","-1","6","signed_subtract");
        literal(device,Operation::Subtract,large,large,large,large,"0","1","huge_cancellation_zero");
    }
    if(mode=="all" || mode=="multiply")literal(device,Operation::Multiply,"-2","3","9","-4","3","2","signed_denominator_multiply");
    if(mode=="all" || mode=="divide")literal(device,Operation::Divide,"2","3","-4","9","-3","2","signed_rational_divide");
    if(mode=="all" || mode=="decimal"){
        Request decimal;decimal.invocation=12;decimal.position=66;decimal.domain=kDecimalDomain;
        decimal.lhs.numerator=integer("1");decimal.lhs.denominator=integer("1");decimal.lhs.scale=1;
        decimal.rhs.numerator=integer("2");decimal.rhs.denominator=integer("1");decimal.rhs.scale=1;
        const auto actual=device.run(decimal);const Rational expected{integer("3"),integer("10")};
        require(actual.status==Status::Success && !std::memcmp(&actual.value,&expected,sizeof(expected)),"point_one_plus_point_two_is_three_tenths");
    }
    auto failure=[&](const Request& request,Status status,const char* name){
        const auto actual=device.run(request);const Rational absent{};
        require(actual.version==kAbi && actual.invocation==request.invocation && actual.position==request.position
            && actual.status==status && !std::memcmp(&actual.value,&absent,sizeof(absent)),name);
    };
    Request bad;bad.invocation=19;bad.position=75;
    bad.lhs.numerator=integer("1");bad.lhs.denominator=integer("3");
    bad.rhs.numerator=integer("1");bad.rhs.denominator=integer("6");
    bad.lhs.denominator=integer("0");failure(bad,Status::DomainError,"zero_left_denominator");
    bad.lhs.denominator=integer("3");bad.rhs.denominator=integer("0");failure(bad,Status::DomainError,"zero_right_denominator");
    bad.rhs.denominator=integer("6");bad.rhs.numerator=integer("0");bad.operation=Operation::Divide;
    failure(bad,Status::DomainError,"divide_by_zero_fraction");
    bad.operation=Operation::Add;bad.lhs.numerator=integer(large);bad.lhs.denominator=integer("1");
    bad.rhs.numerator=integer("1");bad.rhs.denominator=integer("1");failure(bad,Status::Overflow,"numerator_final_overflow");
    bad.operation=Operation::Multiply;bad.lhs.numerator=integer("1");bad.lhs.denominator=integer(large);
    bad.rhs.denominator=integer(large);failure(bad,Status::Overflow,"denominator_final_overflow");
    for(uint32_t version:{1u,2u}){bad.version=version;failure(bad,Status::Malformed,"foreign_abi_refused");}
    std::string error;
    require(!execute(device.input,reinterpret_cast<Result*>(device.input),1,device.stream,error),"overlap_must_be_rejected");
    require(!execute(device.input,reinterpret_cast<Result*>(reinterpret_cast<char*>(device.input)+303),1,device.stream,error),"tail_overlap_refused");
    require(!execute(device.input,device.output,0,device.stream,error),"zero_batch_refused");
    require(!execute(device.input,device.output,65,device.stream,error),"oversize_batch_refused");
    require(!execute(device.input,device.output,1,nullptr,error),"default_stream_refused");
    require(!execute(device.input,device.output,1,cudaStreamLegacy,error),"legacy_stream_refused");
    require(!execute(device.input,device.output,1,cudaStreamPerThread,error),"per_thread_stream_refused");
    std::cout<<"PASS exact rational literals mode="<<mode<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
