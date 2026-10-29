#include "strata/core/qwenputer_binding.hpp"
#include <fstream>
#include <algorithm>
#include <string_view>
#include <limits>
#include <map>
#include <exception>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#endif

namespace strata::qwenputer {
namespace {
std::string hex(const Digest& value) {
    constexpr char digits[]="0123456789abcdef";
    std::string text;
    for(auto b:value){text+=digits[b>>4];text+=digits[b&15];}
    return text;
}
struct ReadInput {
    Digest digest{};
    std::uint64_t bytes=0;
#if defined(_WIN32)
    HANDLE handle=INVALID_HANDLE_VALUE;
    ~ReadInput(){if(handle!=INVALID_HANDLE_VALUE)CloseHandle(handle);}
#else
    std::ifstream file;
#endif
    bool open(const std::filesystem::path& path,std::uint64_t limit,std::string& error) {
#if defined(_WIN32)
        handle=CreateFileW(path.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_FLAG_SEQUENTIAL_SCAN,nullptr);
        LARGE_INTEGER size{};
        if(handle==INVALID_HANDLE_VALUE || GetFileType(handle)!=FILE_TYPE_DISK || !GetFileSizeEx(handle,&size) || size.QuadPart<0) {
            error="binding_file";return false;
        }
        bytes=static_cast<std::uint64_t>(size.QuadPart);
        if(bytes>limit){error="binding_file_size";return false;}
        struct Provider {
            BCRYPT_ALG_HANDLE algorithm=nullptr;
            BCRYPT_HASH_HANDLE hash=nullptr;
            ~Provider(){if(hash)BCryptDestroyHash(hash);if(algorithm)BCryptCloseAlgorithmProvider(algorithm,0);}
        } provider;
        if(BCryptOpenAlgorithmProvider(&provider.algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)!=0
                || BCryptCreateHash(provider.algorithm,&provider.hash,nullptr,0,nullptr,0,0)!=0) {
            error="binding_hash_provider";return false;
        }
        std::vector<std::uint8_t> buffer(4u<<20);
        std::uint64_t left=bytes;
        while(left) {
            const auto count=static_cast<DWORD>(std::min<std::uint64_t>(left,buffer.size())); DWORD got=0;
            if(!ReadFile(handle,buffer.data(),count,&got,nullptr) || !got || got>count){error="binding_file";return false;}
            if(BCryptHashData(provider.hash,buffer.data(),got,0)!=0){error="binding_hash_provider";return false;}
            left-=got;
        }
        if(BCryptFinishHash(provider.hash,digest.data(),static_cast<ULONG>(digest.size()),0)!=0) {
            error="binding_hash_provider";return false;
        }
#else
        std::error_code ec;
        if(!std::filesystem::is_regular_file(path,ec)){error="binding_file";return false;}
        file.open(path,std::ios::binary);
        if(!file){error="binding_file";return false;}
        if(!sha256_stream(file,limit,digest,bytes,error))return false;
#endif
        return true;
    }
    std::string description()const {
        return "{\"bytes\":"+std::to_string(bytes)+",\"sha256\":\""+hex(digest)+"\"}";
    }
};
Digest text_digest(const std::string& text) {
    return sha256({reinterpret_cast<const std::uint8_t*>(text.data()),text.size()});
}
}
struct VerifiedInputs::Impl {
    InputProof proof;
    std::vector<std::unique_ptr<ReadInput>> inputs;
};
VerifiedInputs::VerifiedInputs() = default;
VerifiedInputs::~VerifiedInputs() = default;
bool VerifiedInputs::bind(const Binding& binding,const std::filesystem::path& tokenizer,
                          std::span<const std::filesystem::path> models,std::string& error) try {
    if(impl_){error="inputs_already_bound";return false;}
    if(models.empty()){error="binding_model_sources";return false;}
    auto candidate=std::make_unique<Impl>();
    std::map<std::string,std::string> entries;
    for(const char* name:{"tokenizer.json","vocab.json","merges.txt","token_type.json","chat_template.jinja"}) {
        const auto path=tokenizer/name;
        std::error_code ec;
        if(std::string_view(name)=="chat_template.jinja" && !std::filesystem::exists(path,ec)) {
            if(ec){error="binding_file";return false;}
            continue;
        }
        auto input=std::make_unique<ReadInput>();
        if(!input->open(path,64ull<<20,error))return false;
        entries.emplace(name,input->description());candidate->inputs.push_back(std::move(input));
    }
    std::string canonical="{";
    for(const auto& [name,description]:entries) {
        if(canonical.size()>1)canonical+=',';
        canonical+='"'+name+"\":"+description;
    }
    canonical+='}';candidate->proof.tokenizer=text_digest(canonical);
    if(candidate->proof.tokenizer!=binding.digests[5]){error="tokenizer_binding";return false;}
    canonical="{\"kind\":\"strata-model-files-v1\",\"shards\":[";
    for(const auto& path:models) {
        auto input=std::make_unique<ReadInput>();
        if(!input->open(path,std::numeric_limits<std::uint64_t>::max(),error))return false;
        if(canonical.back()!='[')canonical+=',';
        canonical+=input->description();candidate->proof.model=input->digest;
        candidate->inputs.push_back(std::move(input));
    }
    canonical+="]}";
    if(models.size()>1)candidate->proof.model=text_digest(canonical);
    if(candidate->proof.model!=binding.digests[4]){error="model_binding";return false;}
    impl_=std::move(candidate);error.clear();return true;
} catch(const std::exception& e){error=std::string("binding_input:")+e.what();return false;}
const InputProof* VerifiedInputs::proof() const {
    return impl_?&impl_->proof:nullptr;
}
} // namespace strata::qwenputer
