#pragma once

#include "oos/base/arrow/ListOffsets.hpp"
#include "oos/base/arrow/Utf8View.hpp"

#include <cstdint>
#include <string_view>

namespace oos::base
{

// The texts of a "list<utf8>" column (the describers of an output request: {"SortByPoint", "PrintListing"}): per row a
// run of the texts of one utf8 child, between two list offsets. One of the typed views of ContractReader
// (base/arrow/ColumnViews.hpp), in place.
//
// Thread-safety: copyable, immutable, safe to read from several threads. Valid while the ContractReader lives.
class TextListView
{
public:
    TextListView() = default;
    // `texts` is the whole child column; `offsets` index it, length + 1 of them.
    TextListView(ListOffsets offsets, Utf8View texts, std::int64_t length)
        : m_offsets(offsets), m_texts(texts), m_length(length)
    {
    }

    [[nodiscard]] std::int64_t size() const noexcept { return m_length; }
    // The number of texts of a row.
    [[nodiscard]] std::int64_t count(std::int64_t row) const noexcept { return m_offsets[row + 1] - m_offsets[row]; }
    // Text `item` (0 <= item < count(row)) of a row.
    [[nodiscard]] std::string_view at(std::int64_t row, std::int64_t item) const noexcept
    {
        return m_texts[m_offsets[row] + item];
    }

private:
    ListOffsets m_offsets;
    Utf8View m_texts;
    std::int64_t m_length = 0;
};

} // namespace oos::base
