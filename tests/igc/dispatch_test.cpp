#include "strata/core/igc_dispatch.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace strata::igc;
static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void check(cudaError_t v){if(v!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(v));}
struct Buffer {void* p=nullptr;explicit Buffer(std::size_t n){check(cudaMalloc(&p,n));}~Buffer(){if(p)cudaFree(p);}template<class T>T* as(){return static_cast<T*>(p);}};
struct Stream {cudaStream_t s=nullptr;Stream(){check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}~Stream(){if(s){cudaStreamSynchronize(s);cudaStreamDestroy(s);}}};
// Independent test fixture/oracle only, never linked to the production kernel.
static Integer integer(int v){Integer r;int a=std::abs(v);r.negative=v<0;r.length=0;do{r.digits[r.length++]=uint8_t(a%10);a/=10;}while(a);return r;}
static void fields(float* target,const std::array<int,9>& value){const int sizes[9]={4,2,2,2,2,10,10,10,10};int offset=0;
    for(int h=0;h<9;++h){for(int j=0;j<sizes[h];++j)target[offset+j]=j==value[h]?2.f:-1.f;offset+=sizes[h];}}
int main()try{
    std::string error;
    require(assemble_selective_exact({},nullptr,nullptr,0,nullptr,false,error),"hard_off_contract");
    constexpr int count=64;Stream stream;
    std::vector<float> a(count*52),b(count*52),ready(count*2,-1);std::vector<uint32_t> as(count),bs(count),rs(count);
    std::vector<uint64_t> ids(count),anchors(count);std::vector<uint32_t> codes(count);std::vector<ExactRequest> requests(count),expected(count);
    std::vector<ExactResult> results(count);
    const uint32_t expected_codes[16]={0,0,0,1,2,3,3,4,5,5,5,2,2,1,2,5};
    for(int i=0;i<count;++i){
        ids[i]=100+i;anchors[i]=20+i;ready[i*2+1]=2;
        int left=13,right=7,op=1;
        if(i%16==1){left=-92;right=-18;op=2;}if(i%16==2){left=-12;right=64;op=3;}
        if(i/16==1){if(i%16==0)left=0;if(i%16==1)left=0;if(i%16==2)right=-64;}
        if(i/16==2){if(i%16==0){left=-13;right=0;}if(i%16==1){left=7;right=13;}if(i%16==2){left=12;right=-64;}}
        if(i/16==3){if(i%16==0)left=right=0;if(i%16==1){left=92;right=18;}if(i%16==2)left=0;}
        std::array<int,9> f{op,left<0,right<0,std::abs(left)>=10,std::abs(right)>=10,std::abs(left)%10,std::abs(left)/10,std::abs(right)%10,std::abs(right)/10};
        if(i%16==8){f[3]=0;f[5]=3;f[6]=1;}
        if(i%16==9){f[3]=1;f[6]=0;}
        if(i%16==10){f[1]=1;f[3]=0;f[5]=f[6]=0;}
        if(i%16==13)f[0]=0;
        if(i%16==15){f[2]=1;f[4]=0;f[7]=f[8]=0;}
        fields(a.data()+i*52,f);fields(b.data()+i*52,f);
        if(i%16==3){ready[i*2]=2;ready[i*2+1]=-1;as[i]=2;a[i*52]=std::numeric_limits<float>::quiet_NaN();}
        if(i==19)ready[i*2]=ready[i*2+1]=1; // readiness tie selects no-call
        if(i%16==4)as[i]=1;if(i%16==5)ready[i*2]=std::numeric_limits<float>::quiet_NaN();
        if(i%16==6)a[i*52+17]=std::numeric_limits<float>::quiet_NaN();
        if(i%16==7){f[5]=4;fields(b.data()+i*52,f);}
        if(i%16==11)ids[i]=0;if(i%16==12)anchors[i]=262144;if(i%16==14)rs[i]=2;
        if(i%16==13)for(int j=0;j<4;++j)a[i*52+j]=b[i*52+j]=0; // operator tie selects0
        auto& q=expected[i];q.domain=0;q.operation=static_cast<Operation>(0);q.invocation=ids[i];q.position=anchors[i]<262144?anchors[i]:0;
        if(i%16<3){q.domain=1;q.operation=static_cast<Operation>(op);q.lhs=integer(left);q.rhs=integer(right);}
    }
    Buffer da(a.size()*4),db(b.size()*4),dr(ready.size()*4),das(as.size()*4),dbs(bs.size()*4),drs(rs.size()*4),di(ids.size()*8),dp(anchors.size()*8),dq(count*sizeof(ExactRequest)),dc(count*4),result(count*sizeof(ExactResult));
    auto upload=[&](Buffer& dest,const auto& source){check(cudaMemcpyAsync(dest.p,source.data(),source.size()*sizeof(source[0]),cudaMemcpyHostToDevice,stream.s));};
    upload(da,a);upload(db,b);upload(dr,ready);upload(das,as);upload(dbs,bs);upload(drs,rs);upload(di,ids);upload(dp,anchors);
    SelectionInputs in{da.as<float>(),db.as<float>(),dr.as<float>(),das.as<uint32_t>(),dbs.as<uint32_t>(),drs.as<uint32_t>(),di.as<uint64_t>(),dp.as<uint64_t>()};
    require(!assemble_selective_exact(in,dq.as<ExactRequest>(),dc.as<uint32_t>(),65,stream.s,true,error),"bad_count_accepted");
    require(!assemble_selective_exact({},dq.as<ExactRequest>(),dc.as<uint32_t>(),count,stream.s,true,error),"bad_pointers_accepted");
    auto invoke=[&]{if(!assemble_selective_exact(in,dq.as<ExactRequest>(),dc.as<uint32_t>(),count,stream.s,true,error))throw std::runtime_error(error);
        if(!execute_exact(dq.as<ExactRequest>(),result.as<ExactResult>(),count,stream.s,error))throw std::runtime_error(error);};
    for(int graph_mode=0;graph_mode<2;++graph_mode){
        cudaGraph_t graph=nullptr;cudaGraphExec_t graph_exec=nullptr;
        if(graph_mode){check(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeThreadLocal));invoke();check(cudaStreamEndCapture(stream.s,&graph));check(cudaGraphInstantiate(&graph_exec,graph,nullptr,nullptr,0));}
        for(int step=0;step<3;++step){
            if(step==1){auto no=ready;for(int i=0;i<count;++i){no[i*2]=2;no[i*2+1]=-1;}upload(dr,no);check(cudaStreamSynchronize(stream.s));}
            else upload(dr,ready);
            if(graph_exec)check(cudaGraphLaunch(graph_exec,stream.s));else invoke();
            check(cudaMemcpyAsync(requests.data(),dq.p,requests.size()*sizeof(ExactRequest),cudaMemcpyDeviceToHost,stream.s));
            check(cudaMemcpyAsync(results.data(),result.p,results.size()*sizeof(ExactResult),cudaMemcpyDeviceToHost,stream.s));
            check(cudaMemcpyAsync(codes.data(),dc.p,count*4,cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
            for(int i=0;i<count;++i){
                auto want=expected[i];uint32_t code=expected_codes[i%16];
                if(step==1){want.domain=0;want.operation=static_cast<Operation>(0);want.lhs={};want.rhs={};code=(i%16==11||i%16==12||i%16==14)?2:1;}
                require(codes[i]==code,"dispatch_status");require(!std::memcmp(&requests[i],&want,sizeof(want)),"typed_request_bytes");
                require(results[i].invocation==want.invocation && results[i].position==want.position,"result_owner");
                if(code==0){const int values[4][3]={{20,-74,-768},{7,18,768},{-13,-6,-768},{0,74,0}};const auto value=integer(values[i/16][i%16]);require(results[i].status==Status::Success && !std::memcmp(&results[i].value,&value,sizeof(value)),"exact_result");}
                else {Integer empty;require(results[i].status!=Status::Success && !std::memcmp(&results[i].value,&empty,sizeof(empty)),"failure_retained_answer");}
            }
        }
        if(graph_exec)check(cudaGraphExecDestroy(graph_exec));if(graph)check(cudaGraphDestroy(graph));
    }
    auto before=requests;auto before_codes=codes;
    require(assemble_selective_exact({},dq.as<ExactRequest>(),dc.as<uint32_t>(),0,nullptr,false,error),"off_noop_failed");
    check(cudaMemcpyAsync(requests.data(),dq.p,count*sizeof(ExactRequest),cudaMemcpyDeviceToHost,stream.s));check(cudaMemcpyAsync(codes.data(),dc.p,count*4,cudaMemcpyDeviceToHost,stream.s));check(cudaStreamSynchronize(stream.s));
    require(!std::memcmp(before.data(),requests.data(),count*sizeof(ExactRequest)) && before_codes==codes,"off_changed_bytes");
    std::cout<<"{\"cases\":64,\"normal_graph_A_B_A\":true,\"all_typed_and_exact_checks\":true,\"no_learned_quality_claim\":true}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
