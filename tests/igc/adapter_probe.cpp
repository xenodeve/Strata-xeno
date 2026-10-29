#include "strata/core/igc_adapter.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

static strata::igc::adapter::Digest decode(const char* text) {
    const std::string value(text);if(value.size()!=64)throw std::runtime_error("digest length");
    strata::igc::adapter::Digest result{};
    auto digit=[](char c)->int {if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::runtime_error("digest hex");};
    for(std::size_t i=0;i<32;++i)result[i]=static_cast<uint8_t>((digit(value[i*2])<<4)|digit(value[i*2+1]));
    return result;
}
int main(int argc,char** argv) try {
    if(argc!=6 && argc!=7)throw std::runtime_error("artifact model-digest tokenizer-digest file-digest output [--diagnostic]");
    const bool diagnostic=argc==7 && std::string(argv[6])=="--diagnostic";
    if(argc==7 && !diagnostic)throw std::runtime_error("unknown option");
    strata::igc::adapter::Binding binding{decode(argv[2]),decode(argv[3]),decode(argv[4])};
    strata::igc::adapter::Artifact artifact;std::string error;
    if(!strata::igc::adapter::load(argv[1],binding,diagnostic,artifact,error))throw std::runtime_error(error);
    if(std::filesystem::exists(argv[5]))throw std::runtime_error("output already exists");
    std::ofstream output(argv[5],std::ios::binary);
    output.write(reinterpret_cast<const char*>(artifact.payload.data()),static_cast<std::streamsize>(artifact.payload.size()*4));
    output.close();if(!output)throw std::runtime_error("payload write failed");
    std::cout<<"{\"kind\":"<<artifact.kind<<",\"tensors\":"<<artifact.offsets.size()
             <<",\"diagnostic_only\":true,\"payload_floats\":"<<artifact.payload.size()<<"}\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
