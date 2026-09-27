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
              void* reader_context = nullptr);
    void close();
    uint8_t* slot_ptr(uint64_t slot) const;
    uint64_t slots() const { return offsets_.empty() ? 0 : offsets_.size() - 1; }
    uint64_t bytes() const { return offsets_.empty() ? 0 : offsets_.back(); }
    uint64_t lower_free_after() const { return lower_free_after_; }

private:
    int ordinal_ = -1;
    uint8_t* base_ = nullptr;
    std::vector<uint64_t> offsets_;
    uint64_t lower_free_after_ = 0;
};

} // namespace strata::core
