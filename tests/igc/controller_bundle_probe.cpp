#include "strata/core/igc_controller_bundle.hpp"
#include <iostream>
#include <cstring>
int main(int argc,char** argv){
    if(argc!=2 && !(argc==3 && std::string(argv[2])=="--production") && !(argc==4 && std::string(argv[2])=="--transaction"))return 3;
    strata::igc::ControllerBundle bundle;std::string error;
    const bool diagnostic=argc==2 || std::string(argv[2])!="--production";
    if(!strata::igc::load_controller_bundle(argv[1],diagnostic,bundle,error)){std::cerr<<error<<'\n';return 2;}
    if(argc==4){
        const auto old=bundle;
        if(strata::igc::load_controller_bundle(argv[3],true,bundle,error))return 4;
        if(bundle.model!=old.model || bundle.tokenizer!=old.tokenizer || bundle.manifest!=old.manifest)return 5;
        for(std::size_t i=0;i<bundle.networks.size();++i){const auto& a=bundle.networks[i];const auto& b=old.networks[i];
            if(a.kind!=b.kind || a.max_digits!=b.max_digits || a.diagnostic_only!=b.diagnostic_only || a.state!=b.state || a.protocol!=b.protocol || a.schema!=b.schema || a.offsets!=b.offsets || a.payload.size()!=b.payload.size() || std::memcmp(a.payload.data(),b.payload.data(),a.payload.size()*4))return 6;}
        std::cout<<"transaction=preserved ";
    }
    std::cout<<"PASS controller_bundle kinds=";
    for(const auto& a:bundle.networks)std::cout<<a.kind;
    std::cout<<" diagnostic_only=1\n";return 0;
}
