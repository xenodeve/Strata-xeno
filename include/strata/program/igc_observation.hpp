#pragma once
// Offline observation contract. The caller owns/fences GPU data before Copy.
// Missing cached-prefix rows remain absent, never invented from KV or stale data.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace strata::program::igc_observation {
class Capture {
public:
    enum class Stage { FinalHC, Embedding, FirstAttention };
    static constexpr uint32_t max_rows=80, hc=4, embedding=2560, dimensions=hc*embedding;
    bool begin(std::span<const int64_t> prefix,std::string& error) {
        enabled_=false;bad_=false;rows_.clear();tokens_.clear();
        if (prefix.empty() || prefix.size()>262144) return fail(error,"preanchor_prefix_size");
        for (auto token:prefix) if (token<0 || token>=248320) return fail(error,"preanchor_vocabulary");
        anchor_=static_cast<int64_t>(prefix.size())-1;
        first_=std::max<int64_t>(0,anchor_-max_rows);
        for (int64_t p=first_;p<anchor_;++p) tokens_.push_back(static_cast<uint32_t>(prefix[static_cast<std::size_t>(p)]));
        enabled_=true;error.clear();return true;
    }
    bool active() const { return enabled_&&!bad_; }
    int64_t anchor() const { return anchor_; }
    int64_t first() const { return first_; }
    std::size_t count() const { return rows_.size(); }

    template<class Copy> bool observe(int64_t p0,int64_t n,uint32_t source,Copy&& copy,std::string& error) {
        if (!active()) return fail(error,"preanchor_not_active");
        if (p0<0 || n<=0 || p0>anchor_ || n>anchor_-p0 || (source!=1 && source!=2))
            return fail(error,"preanchor_chunk_bounds");
        const int64_t start=std::max(p0,first_),end=std::min(p0+n,anchor_);
        for (int64_t p=start;p<end;++p) {
            if (std::any_of(rows_.begin(),rows_.end(),[p](const Row& r){return r.position==p;}))
                return fail(error,"preanchor_duplicate");
            Row row;row.position=p;row.source=source;
            row.token=tokens_[static_cast<std::size_t>(p-first_)];row.values.resize(dimensions);
            if (!copy(p-p0,std::span<float>(row.values))) return fail(error,"preanchor_owner_copy");
            if (!std::all_of(row.values.begin(),row.values.end(),[](float v){return std::isfinite(v);}))
                return fail(error,"preanchor_nonfinite");
            rows_.push_back(std::move(row));
        }
        error.clear();return true;
    }
    bool write(const std::string& path,std::string& error,Stage stage=Stage::FinalHC) {
        static_assert(std::endian::native==std::endian::little);
        if (!active() || rows_.size()>max_rows) return fail(error,"preanchor_write_state");
        const char* magic=stage==Stage::FinalHC ? "IGCPREA1" : stage==Stage::Embedding ? "IGCEMB01" :
                          stage==Stage::FirstAttention ? "IGCATN01" : nullptr;
        if (!magic) return fail(error,"preanchor_stage");
        std::ofstream file(path,std::ios::binary|std::ios::trunc);
        if (!file) return fail(error,"preanchor_write_open");
        file.write(magic,8);
        auto put=[&]<class T>(const T& value){file.write(reinterpret_cast<const char*>(&value),sizeof(value));};
        const uint32_t version=1,n=static_cast<uint32_t>(rows_.size());
        const uint64_t anchor=static_cast<uint64_t>(anchor_),prompt=anchor+1;
        put(version);put(hc);put(embedding);put(max_rows);put(n);put(anchor);put(prompt);
        std::sort(rows_.begin(),rows_.end(),[](const Row& a,const Row& b){return a.position<b.position;});
        for (const auto& row:rows_) {
            put(row.position);put(row.token);put(row.source);
            file.write(reinterpret_cast<const char*>(row.values.data()),dimensions*sizeof(float));
        }
        file.close();
        if (!file) return fail(error,"preanchor_write_io");
        error.clear();return true;
    }
private:
    struct Row {int64_t position=0;uint32_t token=0,source=0;std::vector<float> values;};
    bool fail(std::string& error,const char* message) {bad_=true;error=message;return false;}
    bool enabled_=false,bad_=false;
    int64_t first_=0,anchor_=0;
    std::vector<uint32_t> tokens_;
    std::vector<Row> rows_;
};
} // namespace strata::program::igc_observation
