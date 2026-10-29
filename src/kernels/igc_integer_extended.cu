#include "strata/core/igc_integer_extended.hpp"
#include <cuda_runtime.h>
namespace strata::igc::integer_extended {
namespace {
constexpr int capacity=kMaxDigits+1;
struct Magnitude {int length=1;int digits[capacity]{};};
__device__ bool valid(const Integer& value){
    if(!value.length || value.length>kMaxDigits || value.negative>1)return false;
    for(int i=0;i<kMaxDigits;++i)if(value.digits[i]>9 || (i>=value.length && value.digits[i]))return false;
    return !(value.length>1 && value.digits[value.length-1]==0)
        && !(value.length==1 && value.digits[0]==0 && value.negative);
}
__device__ Magnitude magnitude(const Integer& value){Magnitude out;out.length=value.length;
    for(int i=0;i<kMaxDigits;++i)out.digits[i]=value.digits[i];return out;}
__device__ void trim(Magnitude& value){for(int i=0;i<capacity && value.length>1 && !value.digits[value.length-1];++i)--value.length;}
__device__ bool zero(const Magnitude& value){return value.length==1 && !value.digits[0];}
__device__ int compare(const Magnitude& a,const Magnitude& b){
    if(a.length!=b.length)return a.length>b.length?1:-1;
    for(int i=a.length-1;i>=0;--i)if(a.digits[i]!=b.digits[i])return a.digits[i]>b.digits[i]?1:-1;
    return 0;
}
__device__ void subtract(Magnitude& a,const Magnitude& b){
    int borrow=0;
    for(int i=0;i<a.length;++i){int digit=a.digits[i]-(i<b.length?b.digits[i]:0)-borrow;
        borrow=digit<0;a.digits[i]=digit+(borrow?10:0);}
    trim(a);
}
__device__ bool increment(Magnitude& value){
    int carry=1;
    for(int i=0;i<capacity && carry;++i){const int next=value.digits[i]+carry;value.digits[i]=next%10;carry=next/10;
        if(i>=value.length)value.length=i+1;}
    return !carry;
}
// Each prefix remainder is below the divisor. Appending a decimal digit needs
// at most nine subtractions; no input can drive an unbounded quotient search.
__device__ bool divide(const Magnitude& a,const Magnitude& b,Magnitude& quotient,Magnitude& remainder){
    quotient={};remainder={};quotient.length=a.length;
    for(int i=a.length-1;i>=0;--i){
        if(remainder.length>=capacity)return false;
        for(int j=capacity-1;j>0;--j)remainder.digits[j]=remainder.digits[j-1];
        remainder.digits[0]=a.digits[i];++remainder.length;trim(remainder);
        int digit=0;
        for(int j=0;j<9 && compare(remainder,b)>=0;++j){subtract(remainder,b);++digit;}
        if(compare(remainder,b)>=0)return false;
        quotient.digits[i]=digit;
    }
    trim(quotient);return true;
}
__device__ bool multiply(const Magnitude& a,const Magnitude& b,Magnitude& output){
    int digits[2*kMaxDigits+1]{};
    for(int i=0;i<a.length;++i)
        for(int j=0;j<b.length;++j)digits[i+j]+=a.digits[i]*b.digits[j];
    for(int i=0;i<2*kMaxDigits;++i){digits[i+1]+=digits[i]/10;digits[i]%=10;}
    int length=2*kMaxDigits;
    for(int i=0;i<2*kMaxDigits && length>1 && !digits[length-1];++i)--length;
    if(length>kMaxDigits)return false;
    output={};output.length=length;
    for(int i=0;i<length;++i)output.digits[i]=digits[i];
    return true;
}
__device__ void publish(Result& result,const Magnitude& value,bool negative){
    if(value.length>kMaxDigits){result.status=Status::Overflow;return;}
    result.status=Status::Success;result.value.length=static_cast<uint8_t>(value.length);
    result.value.negative=static_cast<uint8_t>(negative && !zero(value));
    for(int i=0;i<value.length;++i)result.value.digits[i]=static_cast<uint8_t>(value.digits[i]);
}
__device__ Result calculate(const Request& request){
    Result result;result.invocation=request.invocation;result.position=request.position;
    if(request.version!=kAbi || request.reserved || !request.invocation)return result;
    if(request.domain!=kIntegerDomain || (request.operation!=Operation::FloorDivide && request.operation!=Operation::Modulo && request.operation!=Operation::Gcd && request.operation!=Operation::Power)){result.status=Status::Unsupported;return result;}
    if(!valid(request.lhs) || !valid(request.rhs))return result;
    const auto a=magnitude(request.lhs),b=magnitude(request.rhs);
    if(request.operation==Operation::Power){
        Magnitude accumulator;accumulator.digits[0]=1;
        if(zero(b)){publish(result,accumulator,false);return result;}
        if(a.length==1 && a.digits[0]==1){
            publish(result,accumulator,request.lhs.negative && (b.digits[0]&1));return result;
        }
        if(zero(a)){
            if(request.rhs.negative){result.status=Status::DomainError;return result;}
            publish(result,a,false);return result;
        }
        if(request.rhs.negative){result.status=Status::Unsupported;return result;}
        // All remaining bases have magnitude >=2. 2**206 has 63 digits.
        if(b.length>3){result.status=Status::Overflow;return result;}
        int exponent=0;
        for(int i=b.length-1;i>=0;--i)exponent=10*exponent+b.digits[i];
        if(exponent>=206){result.status=Status::Overflow;return result;}
        auto factor=a;
        for(int bit=0;bit<8 && exponent;++bit){
            Magnitude product;
            if(exponent&1){
                if(!multiply(accumulator,factor,product)){result.status=Status::Overflow;return result;}
                accumulator=product;
            }
            exponent>>=1;
            if(exponent){
                if(!multiply(factor,factor,product)){result.status=Status::Overflow;return result;}
                factor=product;
            }
        }
        publish(result,accumulator,request.lhs.negative && (b.digits[0]&1));return result;
    }
    if(request.operation==Operation::Gcd){
        auto left=a,right=b;
        // <4*kMaxDigits bits. Every two Euclidean steps at least halve the
        // larger magnitude, including an initial swap of unequal operands.
        for(int i=0;i<8*kMaxDigits+2;++i){
            if(zero(right)){publish(result,left,false);return result;}
            Magnitude quotient,remainder;
            if(!divide(left,right,quotient,remainder)){result.status=Status::Fault;return result;}
            left=right;right=remainder;
        }
        result.status=Status::Fault;return result;
    }
    if(zero(b)){result.status=Status::DomainError;return result;}
    Magnitude quotient,remainder;
    if(!divide(a,b,quotient,remainder)){result.status=Status::Fault;return result;}
    const bool negative=request.lhs.negative!=request.rhs.negative;
    if(request.operation==Operation::Modulo){
        if(negative && !zero(remainder)){auto adjusted=b;subtract(adjusted,remainder);remainder=adjusted;}
        publish(result,remainder,request.rhs.negative);return result;
    }
    if(negative && !zero(remainder) && !increment(quotient)){result.status=Status::Overflow;return result;}
    publish(result,quotient,negative);return result;
}
__global__ void kernel(const Request* input,Result* output,uint32_t count){
    const uint32_t row=threadIdx.x;
    if(row<count)output[row]=calculate(input[row]);
}
}
bool execute(const Request* input,Result* output,uint32_t count,void* raw,std::string& error){
    const auto stream=static_cast<cudaStream_t>(raw);
    if(!input || !output || !count || count>kMaxBatch || !stream || stream==cudaStreamLegacy || stream==cudaStreamPerThread){error="integer_extended_launch_arguments";return false;}
    const auto input_address=reinterpret_cast<uintptr_t>(input),output_address=reinterpret_cast<uintptr_t>(output);
    if((input_address<=output_address && output_address-input_address<count*sizeof(Request)) ||
       (output_address<=input_address && input_address-output_address<count*sizeof(Result))){
        error="integer_extended_overlapping_arrays";return false;
    }
    kernel<<<1,64,0,stream>>>(input,output,count);const auto status=cudaPeekAtLastError();
    if(status!=cudaSuccess){error=cudaGetErrorString(status);return false;}error.clear();return true;
}
}
