#include "strata/core/igc_rational_exact.hpp"
#include <cuda_runtime.h>
namespace strata::igc::rational_exact {
namespace {
// Two 62-digit cross-products need124 digits; their sum needs125.
constexpr int capacity=2*kMaxDigits+2;
struct Big { int length=1; int digits[capacity]{}; };
struct Fraction { Big numerator,denominator; bool negative=false; };
__device__ bool valid(const Integer& value) {
    if(!value.length || value.length>kMaxDigits || value.negative>1)return false;
    for(int i=0;i<kMaxDigits;++i)
        if(value.digits[i]>9 || (i>=value.length && value.digits[i]))return false;
    return !(value.length>1 && !value.digits[value.length-1])
        && !(value.length==1 && !value.digits[0] && value.negative);
}
__device__ Big magnitude(const Integer& value) {
    Big out;out.length=value.length;
    for(int i=0;i<kMaxDigits;++i)out.digits[i]=value.digits[i];
    return out;
}
__device__ Big unit(){Big out;out.digits[0]=1;return out;}
__device__ bool zero(const Big& value){return value.length==1 && !value.digits[0];}
__device__ void trim(Big& value) {
    for(int i=0;i<capacity && value.length>1 && !value.digits[value.length-1];++i)--value.length;
}
__device__ int compare(const Big& a,const Big& b) {
    if(a.length!=b.length)return a.length>b.length?1:-1;
    for(int i=a.length-1;i>=0;--i)
        if(a.digits[i]!=b.digits[i])return a.digits[i]>b.digits[i]?1:-1;
    return 0;
}
__device__ void subtract(Big& a,const Big& b) {
    int borrow=0;
    for(int i=0;i<a.length;++i) {
        const int digit=a.digits[i]-(i<b.length?b.digits[i]:0)-borrow;
        borrow=digit<0;a.digits[i]=digit+(borrow?10:0);
    }
    trim(a);
}
__device__ bool add(const Big& a,const Big& b,Big& out) {
    const int length=a.length>b.length?a.length:b.length;
    if(length>=capacity)return false;
    out={};int carry=0;
    for(int i=0;i<length;++i) {
        const int digit=(i<a.length?a.digits[i]:0)+(i<b.length?b.digits[i]:0)+carry;
        out.digits[i]=digit%10;carry=digit/10;
    }
    out.digits[length]=carry;out.length=length+(carry!=0);return true;
}
__device__ bool multiply(const Big& a,const Big& b,Big& out) {
    if(a.length+b.length>capacity)return false;
    out={};out.length=a.length+b.length;
    for(int i=0;i<a.length;++i)
        for(int j=0;j<b.length;++j)out.digits[i+j]+=a.digits[i]*b.digits[j];
    for(int i=0;i<capacity-1;++i){out.digits[i+1]+=out.digits[i]/10;out.digits[i]%=10;}
    if(out.digits[capacity-1]>9)return false;
    trim(out);return true;
}
__device__ bool divide(const Big& a,const Big& b,Big& quotient,Big& remainder) {
    if(zero(b))return false;
    quotient={};remainder={};quotient.length=a.length;
    for(int i=a.length-1;i>=0;--i) {
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
__device__ bool gcd(Big a,Big b,Big& out) {
    // Under4 bits per decimal digit; every two Euclidean steps halve max(a,b).
    for(int i=0;i<8*capacity+2;++i) {
        if(zero(b)){out=a;return true;}
        Big quotient,remainder;
        if(!divide(a,b,quotient,remainder))return false;
        a=b;b=remainder;
    }
    return false;
}
__device__ Status decode(const Operand& input,uint32_t domain,Fraction& out) {
    if(input.reserved || !valid(input.numerator) || !valid(input.denominator))return Status::Malformed;
    out.numerator=magnitude(input.numerator);
    if(domain==kRationalDomain) {
        if(input.scale)return Status::Malformed;
        out.denominator=magnitude(input.denominator);
        out.negative=input.numerator.negative!=input.denominator.negative;
        return zero(out.denominator)?Status::DomainError:Status::Success;
    }
    if(input.denominator.negative || input.denominator.length!=1 || input.denominator.digits[0]!=1)return Status::Malformed;
    if(input.scale>61)return Status::Unsupported;
    out.denominator={};out.denominator.length=static_cast<int>(input.scale)+1;
    out.denominator.digits[input.scale]=1;out.negative=input.numerator.negative;
    return Status::Success;
}
__device__ bool normalize(Fraction& value) {
    if(zero(value.numerator)){value.denominator=unit();value.negative=false;return true;}
    Big common,numerator,denominator,remainder;
    if(!gcd(value.numerator,value.denominator,common)
        || !divide(value.numerator,common,numerator,remainder) || !zero(remainder)
        || !divide(value.denominator,common,denominator,remainder) || !zero(remainder))return false;
    value.numerator=numerator;value.denominator=denominator;return true;
}
__device__ void publish(Result& result,const Fraction& value) {
    if(value.numerator.length>kMaxDigits || value.denominator.length>kMaxDigits){result.status=Status::Overflow;return;}
    result.value.numerator.length=static_cast<uint8_t>(value.numerator.length);
    result.value.numerator.negative=static_cast<uint8_t>(value.negative && !zero(value.numerator));
    result.value.denominator.length=static_cast<uint8_t>(value.denominator.length);
    for(int i=0;i<value.numerator.length;++i)result.value.numerator.digits[i]=static_cast<uint8_t>(value.numerator.digits[i]);
    for(int i=0;i<value.denominator.length;++i)result.value.denominator.digits[i]=static_cast<uint8_t>(value.denominator.digits[i]);
    result.status=Status::Success;
}
__device__ Result calculate(const Request& input) {
    Result result;result.invocation=input.invocation;result.position=input.position;
    if(input.version!=kAbi || input.reserved || !input.invocation)return result;
    const auto operation=static_cast<uint32_t>(input.operation);
    if((input.domain!=kRationalDomain && input.domain!=kDecimalDomain) || operation<1 || operation>4
        || (input.domain==kDecimalDomain && input.operation==Operation::Divide)){result.status=Status::Unsupported;return result;}
    Fraction a,b,value;
    result.status=decode(input.lhs,input.domain,a);if(result.status!=Status::Success)return result;
    result.status=decode(input.rhs,input.domain,b);if(result.status!=Status::Success)return result;
    result.status=Status::Fault;
    if(input.operation==Operation::Multiply) {
        if(!multiply(a.numerator,b.numerator,value.numerator)
            || !multiply(a.denominator,b.denominator,value.denominator))return result;
        value.negative=a.negative!=b.negative;
    } else if(input.operation==Operation::Divide) {
        if(zero(b.numerator)){result.status=Status::DomainError;return result;}
        if(!multiply(a.numerator,b.denominator,value.numerator)
            || !multiply(a.denominator,b.numerator,value.denominator))return result;
        value.negative=a.negative!=b.negative;
    } else {
        Big left,right;
        if(!multiply(a.numerator,b.denominator,left) || !multiply(b.numerator,a.denominator,right)
            || !multiply(a.denominator,b.denominator,value.denominator))return result;
        if(input.operation==Operation::Subtract && !zero(b.numerator))b.negative=!b.negative;
        if(a.negative==b.negative) {
            if(!add(left,right,value.numerator))return result;
            value.negative=a.negative;
        } else if(compare(left,right)>=0) {
            subtract(left,right);value.numerator=left;value.negative=a.negative;
        } else {
            subtract(right,left);value.numerator=right;value.negative=b.negative;
        }
    }
    if(!normalize(value))return result;
    publish(result,value);return result;
}
__global__ void kernel(const Request* input,Result* output,uint32_t count) {
    const auto row=threadIdx.x;
    if(row<count)output[row]=calculate(input[row]);
}
}
bool execute(const Request* input,Result* output,uint32_t count,void* raw,std::string& error) {
    const auto stream=static_cast<cudaStream_t>(raw);
    if(!input || !output || !count || count>kMaxBatch || !stream || stream==cudaStreamLegacy || stream==cudaStreamPerThread) {
        error="rational_launch_arguments";return false;
    }
    const auto a=reinterpret_cast<uintptr_t>(input),b=reinterpret_cast<uintptr_t>(output);
    if((a<=b && b-a<count*sizeof(Request)) || (b<=a && a-b<count*sizeof(Result))) {
        error="rational_overlapping_arrays";return false;
    }
    kernel<<<1,64,0,stream>>>(input,output,count);
    const auto status=cudaPeekAtLastError();
    if(status!=cudaSuccess){error=cudaGetErrorString(status);return false;}
    error.clear();return true;
}
}
