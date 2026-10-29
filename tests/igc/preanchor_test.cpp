#include "strata/program/igc_observation.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

using strata::program::igc_observation::Capture;
static void require(bool value, const char* name) {
    if (!value) throw std::runtime_error(name);
}
int main(int argc, char** argv) try {
    if (argc!=2) return 2;
    Capture c;
    std::string error;
    std::vector<int64_t> prefix(100);
    for (int i=0;i<100;++i) prefix[i]=i;
    require(c.begin(prefix,error),"begin");
    int copied=0;
    auto copy=[&](int64_t offset,std::span<float> row) {
        ++copied;
        std::fill(row.begin(),row.end(),float(offset)+0.25f);
        return true;
    };
    require(c.observe(0,10,1,copy,error)&&copied==0,"outside selected tail copied data");
    require(c.observe(20,3,1,copy,error)&&copied==3,"window selection");
    require(c.observe(19,1,2,copy,error)&&copied==4,"unordered chunk");
    require(c.write(argv[1],error),"write sorted sparse fixture");
    require(c.write(std::string(argv[1])+".embedding",error,Capture::Stage::Embedding),"embedding stage fixture");
    require(c.write(std::string(argv[1])+".attention",error,Capture::Stage::FirstAttention),"attention stage fixture");
    require(c.count()==4&&c.first()==19&&c.anchor()==99,"bounded tail geometry");
    require(!c.observe(20,1,1,copy,error),"duplicate position accepted");
    require(!c.write(argv[1],error),"invalidated collector wrote data");
    std::vector<int64_t> small{11,12};
    require(c.begin(small,error)&&c.count()==0,"reset leaked old rows");
    require(!c.observe(0,2,1,copy,error),"anchor was captured as pre-anchor");
    require(!c.write(argv[1],error),"out-of-bounds state published");
    require(c.begin(small,error),"reset after bounds failure");
    auto bad=[](int64_t,std::span<float> row) {
        std::fill(row.begin(),row.end(),0.f);row[0]=std::numeric_limits<float>::quiet_NaN();return true;
    };
    require(!c.observe(0,1,1,bad,error),"nonfinite row accepted");
    require(c.begin(small,error),"reset after nonfinite");
    auto failed=[](int64_t,std::span<float>){return false;};
    require(!c.observe(0,1,1,failed,error)&&!c.write(argv[1],error),"failed owner copy published");
    require(!c.begin(std::vector<int64_t>{},error),"empty prefix accepted");
    require(!c.begin(std::vector<int64_t>{248320},error),"wrong vocabulary accepted");
    require(!c.begin(std::vector<int64_t>(262145,1),error),"oversized prefix accepted");
    std::cout<<"PASS selected F32 rows, cap, missing rows, reset, duplicates, bounds, finite and copy refusal\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr<<error.what()<<'\n';return 1;
}
