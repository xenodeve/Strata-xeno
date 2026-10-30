#include "strata/core/igc_controller_bundle.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
namespace strata::igc {
bool load_controller_bundle(const std::filesystem::path& path,bool diagnostic,ControllerBundle& out,std::string& error)try{
    if(!diagnostic){error="controller_bundle_diagnostic_only";return false;}
    static_assert(std::endian::native==std::endian::little);
    constexpr std::array<const char*,4> names={"input-embedding.bin","input-attention.bin","readiness.bin","output.bin"};
    constexpr std::array<uint32_t,4> kinds={1,1,3,2};
    if(!std::filesystem::is_regular_file(std::filesystem::symlink_status(path)) || std::filesystem::file_size(path)!=204){
        error="controller_bundle_manifest_file";return false;
    }
#ifdef _WIN32
    const auto attributes=GetFileAttributesW(path.c_str());
    if(attributes==INVALID_FILE_ATTRIBUTES || (attributes&FILE_ATTRIBUTE_REPARSE_POINT)){
        error="controller_bundle_manifest_reparse";return false;
    }
#endif
    std::array<uint8_t,204> raw{};std::ifstream file(path,std::ios::binary);
    file.read(reinterpret_cast<char*>(raw.data()),raw.size());uint32_t version=0;std::memcpy(&version,raw.data()+8,4);
    if(!file || file.peek()!=std::char_traits<char>::eof() || std::memcmp(raw.data(),"IGCLAT01",8) || (version!=1 && version!=2)){
        error="controller_bundle_manifest_header";return false;
    }
    for(std::size_t start=12;start<raw.size();start+=32)
        if(std::all_of(raw.begin()+start,raw.begin()+start+32,[](uint8_t v){return v==0;})){
            error="controller_bundle_manifest_zero_digest";return false;
        }
    ControllerBundle candidate;std::copy_n(raw.data()+12,32,candidate.model.begin());std::copy_n(raw.data()+44,32,candidate.tokenizer.begin());
    candidate.manifest=strata::qwenputer::sha256(raw);candidate.single_readout=version==2;
    for(std::size_t i=0;i<names.size();++i){
        adapter::Binding binding;binding.model=candidate.model;binding.tokenizer=candidate.tokenizer;
        std::copy_n(raw.data()+76+i*32,32,binding.file.begin());
        if(!adapter::load(path.parent_path()/names[i],binding,true,candidate.networks[i],error)){
            error="controller_bundle_network_"+std::to_string(i)+":"+error;return false;
        }
        if(candidate.networks[i].kind!=kinds[i]){error="controller_bundle_role_"+std::to_string(i);return false;}
    }
    out=std::move(candidate);error.clear();return true;
}catch(const std::exception& e){error=std::string("controller_bundle_manifest:")+e.what();return false;}
}
