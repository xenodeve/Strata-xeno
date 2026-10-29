#include "strata/core/qwenputer_binding.hpp"
#include <iostream>
#include <string_view>
using namespace strata::qwenputer;
static bool digest(std::string_view text, Digest& result) {
    if(text.size()!=64)return false;
    for(std::size_t i=0;i<32;++i) {
        auto hex=[](char c){return c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:-1;};
        const auto a=hex(text[i*2]),b=hex(text[i*2+1]); if(a<0||b<0)return false;
        result[i]=static_cast<std::uint8_t>(a*16+b);
    }
    return true;
}
int main(int argc,char** argv) {
    if(argc<5)return 2;
    Binding binding;
    if(!digest(argv[2],binding.digests[4])||!digest(argv[3],binding.digests[5]))return 2;
    std::vector<std::filesystem::path> models;
    for(int i=4;i<argc;++i)models.emplace_back(argv[i]);
    VerifiedInputs verified; std::string error;
    const bool result=verified.bind(binding,argv[1],models,error);
    if(!result) {
        if(verified.proof()) {std::cerr<<"partial_proof_published";return 3;}
        std::cout<<"REFUSED "<<error<<'\n';return 1;
    }
    if(!verified.proof()||verified.proof()->model!=binding.digests[4]||verified.proof()->tokenizer!=binding.digests[5])return 3;
    std::cout<<"PASS bound\n";return 0;
}
