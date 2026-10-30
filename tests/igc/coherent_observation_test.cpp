#include "strata/program/igc_coherent_observation.hpp"
#include <iostream>
#include <stdexcept>
using namespace strata::program::igc_coherent;
static void require(bool v,const char* why){if(!v)throw std::runtime_error(why);}
int main()try{
    std::vector<uint8_t> raw{'I','G','C','C','P','L','0','2'};
    auto put=[&](uint32_t v){for(int i=0;i<4;++i)raw.push_back(static_cast<uint8_t>(v>>(i*8)));};
    put(2);put(2);raw.insert(raw.end(),32,0x11);raw.insert(raw.end(),32,0x22);
    put(0);put(0);put(0);put(3);put(1);put(2);put(3);
    put(1);put(0);put(1);put(4);put(1);put(2);put(3);put(7);
    auto seal=[&]{auto d=strata::qwenputer::sha256(raw);raw.insert(raw.end(),d.begin(),d.end());};seal();
    Plan p;std::string e;require(parse_plan(raw,p,e),e.c_str());require(p.records.size()==2,"plan count");
    require(p.records[1].tokens==std::vector<int64_t>({1,2,3,7}),"consumed prefix only");
    raw.back()^=1;require(!parse_plan(raw,p,e),"digest corruption refused");raw.back()^=1;
    raw.resize(raw.size()-32);raw[128]=9;seal();require(!parse_plan(raw,p,e),"foreign output prefix refused");
    std::cout<<"PASS coherent plan digest/order/consumed-prefix contracts\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
