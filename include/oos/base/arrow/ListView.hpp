#pragma once

#include "oos/base/arrow/ListOffsets.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::base
{

// A variable-size list column ("list<int32>": the grid points of each element): per row a span of the child values.
// One of the typed views of ContractReader (base/arrow/ColumnViews.hpp), in place for both offset widths.
//
// Thread-safety: copyable, immutable, safe to read from several threads. Valid while the ContractReader lives.
template <typename T>
class ListView
{
public:
    ListView() = default;
    // `offsets` relative to `values` (length + 1 of them, checked by the reader).
    ListView(ListOffsets offsets, const T* values, std::int64_t length)
        : m_offsets(offsets), m_values(values), m_length(length)
    {
    }

    [[nodiscard]] std::int64_t size() const noexcept { return m_length; }
    [[nodiscard]] std::span<const T> operator[](std::int64_t row) const noexcept
    {
        const std::int64_t begin = m_offsets[row];
        return {m_values + begin, static_cast<std::size_t>(m_offsets[row + 1] - begin)};
    }
    // The offset of row `row` into the child values; offset(size()) is the end of the last row.
    [[nodiscard]] std::int64_t offset(std::int64_t row) const noexcept { return m_offsets[row]; }

private:
    ListOffsets m_offsets;
    const T* m_values = nullptr;
    std::int64_t m_length = 0;
};

} // namespace oos::base
