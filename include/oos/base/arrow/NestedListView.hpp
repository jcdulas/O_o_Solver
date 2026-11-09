#pragma once

#include "oos/base/arrow/ListOffsets.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::base
{

// A column of type "list<fixed_list<T>[n]>": per row a list of items of n values each (O_o_Reader gives the stations
// of a PBEAM this way). One of the typed views of ContractReader (base/arrow/ColumnViews.hpp), in place for both
// offset widths.
//
// Thread-safety: copyable, immutable, safe to read from several threads. Valid while the ContractReader lives.
template <typename T>
class NestedListView
{
public:
    NestedListView() = default;
    // `offsets` count items (length + 1 of them, checked by the reader); `values` is the first value of item 0.
    NestedListView(ListOffsets offsets, const T* values, std::int64_t length, std::int64_t width)
        : m_offsets(offsets), m_values(values), m_length(length), m_width(width)
    {
    }

    [[nodiscard]] std::int64_t size() const noexcept { return m_length; }
    // The number of values of each item (n).
    [[nodiscard]] std::int64_t width() const noexcept { return m_width; }
    // The number of items of a row.
    [[nodiscard]] std::int64_t itemCount(std::int64_t row) const noexcept
    {
        return m_offsets[row + 1] - m_offsets[row];
    }
    // The n values of item `item` of a row.
    [[nodiscard]] std::span<const T> item(std::int64_t row, std::int64_t item) const noexcept
    {
        return {m_values + (m_offsets[row] + item) * m_width, static_cast<std::size_t>(m_width)};
    }

private:
    ListOffsets m_offsets;
    const T* m_values = nullptr;
    std::int64_t m_length = 0;
    std::int64_t m_width = 0;
};

} // namespace oos::base
