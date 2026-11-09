#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <utility>

namespace oos::input
{

// An array the snapshot reads in place from its producer's buffer, which is not an Arrow table (the rotations of the
// displacement systems, which O_o_Reader gives as a plain array of doubles). `owner` keeps the producer's buffer alive
// as long as a copy of the BorrowedArray lives; what it holds is the producer's business (api::ModelInput holds the
// vector the caller handed over).
//
// Thread-safety: immutable; copies may be read and destroyed on any thread.
template <typename T>
struct BorrowedArray
{
    std::span<const T> values;
    std::shared_ptr<const void> owner;

    BorrowedArray() = default;
    BorrowedArray(std::span<const T> borrowed, std::shared_ptr<const void> keepAlive)
        : values(borrowed), owner(std::move(keepAlive))
    {
    }

    [[nodiscard]] std::size_t size() const noexcept { return values.size(); }
    [[nodiscard]] bool empty() const noexcept { return values.empty(); }
    [[nodiscard]] const T* data() const noexcept { return values.data(); }
};

} // namespace oos::input
