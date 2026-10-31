#pragma once
// Natural-language firing: the sidecar of docs/plans/2026-10-31-mip-sidecar-spec.md on the host. A causal encoder over
// the tokens of an answer; each new token costs one step because the keys and values of earlier tokens are kept. It
// proposes an operation and two pointers into the numbers written so far, never a value.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <immintrin.h>

// The engine's release flags do not vectorise a float sum, which measured 2.1-2.8 ms per token against 0.3-0.5.
#if defined(__GNUC__) || defined(__clang__)
#define STRATA_SIDECAR_SIMD __attribute__((target("avx2,fma")))
#else
#define STRATA_SIDECAR_SIMD
#endif

namespace strata::igc {

STRATA_SIDECAR_SIMD inline float sidecar_dot(const float* a, const float* b, size_t n) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum);
    const __m128 half = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    const __m128 pair = _mm_add_ps(half, _mm_movehl_ps(half, half));
    float total = _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1)));
    for (; i < n; ++i) total += a[i] * b[i];
    return total;
}

STRATA_SIDECAR_SIMD inline void sidecar_add(float scale, const float* x, float* y, size_t n) {
    const __m256 factor = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(factor, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    for (; i < n; ++i) y[i] += scale * x[i];
}

struct SidecarProposal {
    int operation = 0, left = 0, right = 0;
    float fire = 0, confidence = 0;
};

class Sidecar {
public:
    static constexpr int kOperations = 5, kNone = 4;

    bool load(const std::string& path, std::string& error) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { error = "sidecar: cannot open " + path; return false; }
        char magic[8];
        int32_t head[8];
        bool ok = std::fread(magic, 1, 8, f) == 8 && !std::memcmp(magic, "STRSIDE1", 8) && std::fread(head, 4, 8, f) == 8;
        if (ok) {
            width_ = head[0]; layers_ = head[1]; heads_ = head[2]; vocabulary_ = head[3]; table_width_ = head[4];
            common_ = head[5]; max_tokens_ = head[6]; max_entries_ = head[7];
            part_ = width_ / heads_;
            table_.resize(static_cast<size_t>(vocabulary_) * table_width_);
            index_.resize(vocabulary_);
            ok = std::fread(table_.data(), 2, table_.size(), f) == table_.size() && std::fread(index_.data(), 4, index_.size(), f) == index_.size();
        }
        const auto read = [&](std::vector<float>& v, size_t n) {
            v.resize(n);
            ok = ok && std::fread(v.data(), 4, n, f) == n;
        };
        const size_t w = width_;
        if (ok) {
            read(project_weight_, w * table_width_); read(project_bias_, w); read(project_norm_w_, w); read(project_norm_b_, w);
            read(learned_, (common_ + 1) * w); read(position_, max_tokens_ * w);
            layer_.resize(layers_);
            for (Layer& l : layer_) {
                read(l.norm1_w, w); read(l.norm1_b, w); read(l.in_w, 3 * w * w); read(l.in_b, 3 * w); read(l.out_w, w * w); read(l.out_b, w);
                read(l.norm2_w, w); read(l.norm2_b, w); read(l.ff1_w, 4 * w * w); read(l.ff1_b, 4 * w); read(l.ff2_w, 4 * w * w); read(l.ff2_b, w);
            }
            read(norm_w_, w); read(norm_b_, w); read(operation_w_, kOperations * w); read(operation_b_, kOperations);
            read(left_w_, w * w); read(left_b_, w); read(right_w_, w * w); read(right_b_, w); read(entry_w_, w * w); read(entry_b_, w);
            read(recency_, max_entries_ * w);
            std::vector<float> scale;
            read(scale, 1);
            if (ok) scale_ = scale[0];
        }
        std::fclose(f);
        if (!ok) { error = "sidecar: bad file " + path; return false; }
        reset();
        return true;
    }

    bool loaded() const { return width_ > 0; }
    int tokens() const { return count_; }
    int capacity() const { return max_tokens_; }

    void reset() {
        count_ = 0;
        states_.clear();
        keys_.assign(layers_, {});
        values_.assign(layers_, {});
    }

    // One token of the answer. Past capacity the caller rebuilds from the tail, as training saw it.
    void push(int32_t token) {
        const size_t w = width_;
        std::vector<float> x(w, 0.f);
        const int id = token >= 0 && token < vocabulary_ ? token : 0;
        std::vector<float> row(table_width_);
        for (int i = 0; i < table_width_; ++i) row[i] = half(table_[static_cast<size_t>(id) * table_width_ + i]);
        linear(project_weight_.data(), project_bias_.data(), row.data(), x.data(), w, table_width_);
        layer_norm(x.data(), project_norm_w_.data(), project_norm_b_.data());
        const float* learned = &learned_[static_cast<size_t>(index_[id]) * w];
        const float* position = &position_[static_cast<size_t>(count_) * w];
        for (size_t i = 0; i < w; ++i) x[i] += learned[i] + position[i];
        std::vector<float> a(w), projected(3 * w), mixed(w), inner(4 * w), output(w);
        for (int l = 0; l < layers_; ++l) {
            Layer& L = layer_[l];
            a = x;
            layer_norm(a.data(), L.norm1_w.data(), L.norm1_b.data());
            linear(L.in_w.data(), L.in_b.data(), a.data(), projected.data(), 3 * w, w);
            keys_[l].insert(keys_[l].end(), projected.begin() + w, projected.begin() + 2 * w);
            values_[l].insert(values_[l].end(), projected.begin() + 2 * w, projected.end());
            attend(l, projected.data(), mixed.data());
            linear(L.out_w.data(), L.out_b.data(), mixed.data(), output.data(), w, w);
            for (size_t i = 0; i < w; ++i) x[i] += output[i];
            a = x;
            layer_norm(a.data(), L.norm2_w.data(), L.norm2_b.data());
            linear(L.ff1_w.data(), L.ff1_b.data(), a.data(), inner.data(), 4 * w, w);
            for (float& v : inner) v = gelu(v);
            linear(L.ff2_w.data(), L.ff2_b.data(), inner.data(), output.data(), w, 4 * w);
            for (size_t i = 0; i < w; ++i) x[i] += output[i];
        }
        layer_norm(x.data(), norm_w_.data(), norm_b_.data());
        states_.insert(states_.end(), x.begin(), x.end());
        ++count_;
    }

    // Scores at the last token for numbers ending at `ends`, the raw outputs the training code gives.
    void score(const std::vector<int>& ends, float* operation, float* left, float* right) const {
        const size_t w = width_;
        const float* query = &states_[static_cast<size_t>(count_ - 1) * w];
        linear(operation_w_.data(), operation_b_.data(), query, operation, kOperations, w);
        std::vector<float> head_l(w), head_r(w), entry(w);
        linear(left_w_.data(), left_b_.data(), query, head_l.data(), w, w);
        linear(right_w_.data(), right_b_.data(), query, head_r.data(), w, w);
        normalize(head_l.data());
        normalize(head_r.data());
        const int n = static_cast<int>(ends.size());
        for (int i = 0; i < n; ++i) {
            linear(entry_w_.data(), entry_b_.data(), &states_[static_cast<size_t>(ends[i]) * w], entry.data(), w, w);
            const float* rank = &recency_[static_cast<size_t>(std::clamp(n - 1 - i, 0, max_entries_ - 1)) * w];
            for (size_t k = 0; k < w; ++k) entry[k] += rank[k];
            normalize(entry.data());
            left[i] = scale_ * sidecar_dot(entry.data(), head_l.data(), w);
            right[i] = scale_ * sidecar_dot(entry.data(), head_r.data(), w);
        }
    }

    SidecarProposal propose(const std::vector<int>& ends) const {
        const int n = static_cast<int>(ends.size());
        std::vector<float> left(n), right(n);
        float operation[kOperations];
        score(ends, operation, left.data(), right.data());
        softmax(operation, kOperations);
        softmax(left.data(), n);
        softmax(right.data(), n);
        SidecarProposal out;
        out.fire = 1 - operation[kNone];
        out.operation = static_cast<int>(std::max_element(operation, operation + kNone) - operation);
        out.left = static_cast<int>(std::max_element(left.begin(), left.end()) - left.begin());
        out.right = static_cast<int>(std::max_element(right.begin(), right.end()) - right.begin());
        out.confidence = out.fire * left[out.left] * right[out.right];
        return out;
    }

private:
    struct Layer {
        std::vector<float> norm1_w, norm1_b, in_w, in_b, out_w, out_b, norm2_w, norm2_b, ff1_w, ff1_b, ff2_w, ff2_b;
    };

    static float half(uint16_t h) {
        const int sign = h >> 15, exponent = (h >> 10) & 31, fraction = h & 1023;
        float v;
        if (exponent == 0) v = std::ldexp(static_cast<float>(fraction), -24);
        else if (exponent == 31) v = std::numeric_limits<float>::infinity();
        else v = std::ldexp(static_cast<float>(fraction + 1024), exponent - 25);
        return sign ? -v : v;
    }
    static float gelu(float x) { return 0.5f * x * (1 + std::erf(x * 0.70710678118f)); }
    static void softmax(float* v, int n) {
        const float top = *std::max_element(v, v + n);
        float sum = 0;
        for (int i = 0; i < n; ++i) { v[i] = std::exp(v[i] - top); sum += v[i]; }
        for (int i = 0; i < n; ++i) v[i] /= sum;
    }
    void normalize(float* v) const {
        float sum = 0;
        for (int i = 0; i < width_; ++i) sum += v[i] * v[i];
        sum = std::max(std::sqrt(sum), 1e-12f);
        for (int i = 0; i < width_; ++i) v[i] /= sum;
    }
    // y = Wx + b, with W stored row by row as torch does
    static void linear(const float* weight, const float* bias, const float* x, float* y, size_t rows, size_t columns) {
        for (size_t r = 0; r < rows; ++r) y[r] = bias[r] + sidecar_dot(weight + r * columns, x, columns);
    }
    void layer_norm(float* x, const float* weight, const float* bias) const {
        double mean = 0, variance = 0;
        for (int i = 0; i < width_; ++i) mean += x[i];
        mean /= width_;
        for (int i = 0; i < width_; ++i) variance += (x[i] - mean) * (x[i] - mean);
        const float scale = 1.f / std::sqrt(static_cast<float>(variance / width_) + 1e-5f);
        for (int i = 0; i < width_; ++i) x[i] = (x[i] - static_cast<float>(mean)) * scale * weight[i] + bias[i];
    }
    // The new token's query over every key so far, itself included.
    void attend(int layer, const float* query, float* mixed) const {
        const size_t t = count_ + 1, w = width_;
        std::vector<float> weight(t);
        const float root = std::sqrt(static_cast<float>(part_));
        for (int h = 0; h < heads_; ++h) {
            const float* q = query + h * part_;
            for (size_t i = 0; i < t; ++i) {
                weight[i] = sidecar_dot(q, &keys_[layer][i * w + h * part_], part_) / root;
            }
            softmax(weight.data(), static_cast<int>(t));
            float* out = mixed + h * part_;
            std::fill(out, out + part_, 0.f);
            for (size_t i = 0; i < t; ++i) sidecar_add(weight[i], &values_[layer][i * w + h * part_], out, part_);
        }
    }

    int width_ = 0, layers_ = 0, heads_ = 0, part_ = 0, vocabulary_ = 0, table_width_ = 0, common_ = 0, max_tokens_ = 0, max_entries_ = 0;
    int count_ = 0;
    float scale_ = 1;
    std::vector<uint16_t> table_;
    std::vector<int32_t> index_;
    std::vector<float> project_weight_, project_bias_, project_norm_w_, project_norm_b_, learned_, position_;
    std::vector<float> norm_w_, norm_b_, operation_w_, operation_b_, left_w_, left_b_, right_w_, right_b_, entry_w_, entry_b_, recency_;
    std::vector<Layer> layer_;
    std::vector<float> states_;
    std::vector<std::vector<float>> keys_, values_;
};

}  // namespace strata::igc
