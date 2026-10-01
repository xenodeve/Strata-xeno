#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

// Read-only expert slots on the display GPU. open() touches the whole allocation and
// refuses a result below the measured 2560 MiB free-VRAM floor.
class SecondaryArena {
public:
    using FreeReader = bool (*)(int ordinal, uint64_t& lower_free, std::string& err, void* context);
    SecondaryArena() = default;
    ~SecondaryArena();
    SecondaryArena(const SecondaryArena&) = delete;
    SecondaryArena& operator=(const SecondaryArena&) = delete;

    bool open(int ordinal, const std::vector<uint64_t>& ranked_blob_bytes,
              uint64_t max_bytes, std::string& err, FreeReader reader = nullptr,
              void* reader_context = nullptr, uint64_t free_floor_bytes = 2560ull << 20);
    // Returns false and retains the pointer if CUDA could not release it, so the caller can retry.
    bool close(std::string* err = nullptr);
    uint8_t* slot_ptr(uint64_t slot) const;
    /// #11: the bytes a slot holds (its first blob, 256-aligned); a swap's newcomer must fit it.  0 when out of range.
    uint64_t slot_bytes(uint64_t slot) const {
        return slot < slots() ? offsets_[(size_t) slot + 1] - offsets_[(size_t) slot] : 0;
    }
    /// Whether a blob of `bytes` fits `slot`: the fills, and every 4070 swap (its pairs span layers, and a native
    /// pack's blobs differ per layer, so an unchecked newcomer would overwrite the next slot's expert).
    bool fits(uint64_t slot, uint64_t bytes) const { return bytes > 0 && bytes <= slot_bytes(slot); }
    bool fill_slot(uint64_t slot, const uint8_t* blob, uint64_t bytes, std::string& err);
    bool verify_slot(uint64_t slot, const uint8_t* blob, uint64_t bytes, std::string& err);
    /// The display card's free-memory floor, checked once (a pipelined fill that bypasses fill_slot calls it last).
    bool check_free_floor(std::string& err);
    uint64_t slots() const { return offsets_.empty() ? 0 : offsets_.size() - 1; }
    uint64_t bytes() const { return offsets_.empty() ? 0 : offsets_.back(); }
    uint64_t lower_free_after() const { return lower_free_after_; }

private:
    int ordinal_ = -1;
    uint8_t* base_ = nullptr;
    std::vector<uint64_t> offsets_;
    uint64_t lower_free_after_ = 0;
    uint64_t free_floor_bytes_ = 2560ull << 20;
};

} // namespace strata::core
