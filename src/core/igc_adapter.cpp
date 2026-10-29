#include "strata/core/igc_adapter.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <new>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace strata::igc::adapter {
namespace {
constexpr std::size_t maximum=8u<<20,header=256,descriptor=40;
template<class T>T scalar(std::span<const uint8_t> bytes,std::size_t offset) {
    T result;std::memcpy(&result,bytes.data()+offset,sizeof(T));return result;
}
Digest digest_at(std::span<const uint8_t> bytes,std::size_t offset) {
    Digest value;std::memcpy(value.data(),bytes.data()+offset,32);return value;
}
bool zeros(std::span<const uint8_t> bytes) {
    return std::all_of(bytes.begin(),bytes.end(),[](uint8_t value){return value==0;});
}
bool fail(std::string& error,const char* value) {error=value;return false;}
}
bool parse(std::span<const uint8_t> bytes,const Binding& binding,bool allow_diagnostic,
           Artifact& output,std::string& error) try {
    static_assert(std::endian::native==std::endian::little);
    output={};error.clear();
    if(bytes.size()<header+32 || bytes.size()>maximum)return fail(error,"igc_adapter_file_bounds");
    if(strata::qwenputer::sha256(bytes)!=binding.file ||
       strata::qwenputer::sha256(bytes.first(bytes.size()-32))!=digest_at(bytes,bytes.size()-32))
        return fail(error,"igc_adapter_digest");
    const auto kind=scalar<uint32_t>(bytes,12),count=scalar<uint32_t>(bytes,20);
    const auto definition=adapter_schema::get(kind);
    if(std::memcmp(bytes.data(),"STIGCM01",8) || scalar<uint32_t>(bytes,8)!=1 ||
       definition.kind==0 || scalar<uint32_t>(bytes,16)!=1 || count==0 || count>128 ||
       count!=definition.tensors.size() || scalar<uint32_t>(bytes,24)!=2560 ||
       scalar<uint32_t>(bytes,28)!=4 || scalar<uint32_t>(bytes,32)!=(kind==1?80u:0u) ||
       scalar<uint32_t>(bytes,36)!=(kind==1?2u:62u) || scalar<uint32_t>(bytes,40)!=1 ||
       scalar<uint32_t>(bytes,44)!=1 || scalar<uint32_t>(bytes,48)!=7 || scalar<uint32_t>(bytes,52)!=0 ||
       !zeros(bytes.subspan(224,32)) || digest_at(bytes,64)!=binding.model ||
       digest_at(bytes,96)!=binding.tokenizer || digest_at(bytes,192)!=definition.digest)
        return fail(error,"igc_adapter_header_or_binding");
    if(!allow_diagnostic)return fail(error,"igc_adapter_diagnostic_only");
    const auto payload_bytes=scalar<uint64_t>(bytes,56);
    const auto end=header+static_cast<std::size_t>(count)*descriptor;
    const auto start=(end+15)&~std::size_t(15);
    if(payload_bytes>maximum || payload_bytes%4 || start>bytes.size()-32 ||
       payload_bytes!=bytes.size()-32-start || !zeros(bytes.subspan(end,start-end)))
        return fail(error,"igc_adapter_payload_bounds");
    uint64_t cursor=0;
    // Validate all file-controlled counts/offsets/values before allocating weights.
    for(uint32_t i=0;i<count;++i) {
        const auto row=header+static_cast<std::size_t>(i)*descriptor;
        const auto& expected=definition.tensors[i];
        const auto offset=scalar<uint64_t>(bytes,row+24),elements=scalar<uint64_t>(bytes,row+32);
        if(scalar<uint32_t>(bytes,row)!=i || scalar<uint32_t>(bytes,row+4)!=expected.rank ||
           offset!=((cursor+15)&~uint64_t(15)) || elements!=expected.count ||
           offset>payload_bytes || elements>(payload_bytes-offset)/4)
            return fail(error,"igc_adapter_tensor_layout");
        for(std::size_t d=0;d<4;++d)
            if(scalar<uint32_t>(bytes,row+8+d*4)!=expected.dimensions[d])
                return fail(error,"igc_adapter_tensor_shape");
        if(!zeros(bytes.subspan(start+static_cast<std::size_t>(cursor),static_cast<std::size_t>(offset-cursor))))
            return fail(error,"igc_adapter_tensor_padding");
        for(uint64_t j=0;j<elements;++j)
            if((scalar<uint32_t>(bytes,start+static_cast<std::size_t>(offset+j*4))&0x7f800000u)==0x7f800000u)
                return fail(error,"igc_adapter_nonfinite");
        cursor=offset+elements*4;
    }
    if(cursor!=payload_bytes)return fail(error,"igc_adapter_unused_payload");
    Artifact candidate;candidate.kind=kind;candidate.max_digits=kind==1?2u:62u;
    candidate.state=digest_at(bytes,128);candidate.protocol=digest_at(bytes,160);candidate.schema=definition.digest;
    candidate.offsets.reserve(count);candidate.payload.resize(static_cast<std::size_t>(payload_bytes/4));
    std::memcpy(candidate.payload.data(),bytes.data()+start,static_cast<std::size_t>(payload_bytes));
    for(uint32_t i=0;i<count;++i)
        candidate.offsets.push_back(scalar<uint64_t>(bytes,header+static_cast<std::size_t>(i)*descriptor+24)/4);
    output=std::move(candidate);return true;
} catch(const std::bad_alloc&) {output={};return fail(error,"igc_adapter_host_allocation");}

bool load(const std::filesystem::path& path,const Binding& binding,bool allow_diagnostic,
          Artifact& output,std::string& error) try {
    output={};error.clear();std::error_code code;
    const auto status=std::filesystem::symlink_status(path,code);
    if(code || !std::filesystem::is_regular_file(status))return fail(error,"igc_adapter_regular_file");
#ifdef _WIN32
    const auto attributes=GetFileAttributesW(path.c_str());
    if(attributes==INVALID_FILE_ATTRIBUTES || (attributes&FILE_ATTRIBUTE_REPARSE_POINT))
        return fail(error,"igc_adapter_reparse_file");
#endif
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()<static_cast<std::streamoff>(header+32) || file.tellg()>static_cast<std::streamoff>(maximum))
        return fail(error,"igc_adapter_file_bounds");
    const auto count=static_cast<std::size_t>(file.tellg());std::vector<uint8_t> bytes(count);
    file.seekg(0);file.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(count));
    if(!file || file.peek()!=std::char_traits<char>::eof())return fail(error,"igc_adapter_file_changed_or_read");
    return parse(bytes,binding,allow_diagnostic,output,error);
} catch(const std::bad_alloc&) {output={};return fail(error,"igc_adapter_host_allocation");}
} // namespace strata::igc::adapter
