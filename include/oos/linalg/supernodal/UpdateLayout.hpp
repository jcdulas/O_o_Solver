#pragma once

#include <cstdint>

namespace oos::linalg
{

// How an update matrix of order n is stored (its lower triangle): square, column-major of leading dimension n
// (blockColumns 0), or by blocks of blockColumns = B columns (the full path streamed: about half the
// values), block k holding the rows [kB, n) of the columns [kB, kB + B) column-major (leading dimension n − kB), the
// blocks one after the other.
struct UpdateLayout
{
    // The block width of the streamed solve's update matrices: wide enough that the upper triangle of a diagonal block
    // (B/2 values a column) is a small share of a large update (B/n), narrow enough not to matter for a small one.
    static constexpr std::int32_t STREAMED_BLOCK_COLUMNS = 256;

    std::int64_t order = 0;
    std::int64_t blockColumns = 0;

    [[nodiscard]] bool isBlocked() const noexcept { return blockColumns > 0 && blockColumns < order; }
    // The values stored: the full blocks, then the last (of width n − (K − 1)·B ≤ B, its rows as many).
    [[nodiscard]] std::int64_t valueCount() const noexcept
    {
        if (!isBlocked())
            return order * order;
        const std::int64_t last = (order - 1) / blockColumns;
        const std::int64_t width = order - last * blockColumns;
        return blockOffset(last) + width * width;
    }
    // The offset o_j such that element (i, j), i >= j, is at o_j + i: o_j ≥ 0 (the earlier blocks hold at least kB
    // values).
    [[nodiscard]] std::int64_t columnOffset(std::int64_t j) const noexcept
    {
        if (!isBlocked())
            return j * order;
        const std::int64_t k = j / blockColumns;
        const std::int64_t first = k * blockColumns;
        return blockOffset(k) + (j - first) * (order - first) - first;
    }
    // The offset of block k's first value.
    [[nodiscard]] std::int64_t blockOffset(std::int64_t k) const noexcept
    {
        // Σ_{m<k} (n − mB)·B.
        return k * blockColumns * order - blockColumns * blockColumns * (k * (k - 1) / 2);
    }
};

} // namespace oos::linalg
