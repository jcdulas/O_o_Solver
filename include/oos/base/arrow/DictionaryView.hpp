#pragma once

#include "oos/base/arrow/Utf8View.hpp"

#include <cstdint>
#include <limits>
#include <string_view>

namespace oos::base
{

// A dictionary-encoded text column ("dictionary<utf8>": the card types of a mixed table): per row an index into the
// dictionary of distinct values. One of the typed views of ContractReader (base/arrow/ColumnViews.hpp). The indices
// are read in place whatever their signed width (int8, int16, int32 as O_o_Reader gives them, int64): no copy.
//
// Thread-safety: copyable, immutable, safe to read from several threads. Valid while the ContractReader lives.
class DictionaryView
{
public:
    DictionaryView() = default;
    // `indices` points at the index of row 0, each of indexBytes bytes (1, 2, 4 or 8, signed).
    DictionaryView(const void* indices, int indexBytes, std::int64_t length, Utf8View values)
        : m_indices(indices), m_indexBytes(indexBytes), m_length(length), m_values(values)
    {
    }

    [[nodiscard]] std::int64_t size() const noexcept { return m_length; }
    // The index of a row; -1 for an index outside int32 (it is outside the dictionary anyway).
    [[nodiscard]] std::int32_t index(std::int64_t row) const noexcept
    {
        switch (m_indexBytes)
        {
        case 1:
            return static_cast<const std::int8_t*>(m_indices)[row];
        case 2:
            return static_cast<const std::int16_t*>(m_indices)[row];
        case 8:
        {
            const std::int64_t wide = static_cast<const std::int64_t*>(m_indices)[row];
            return wide >= 0 && wide <= std::numeric_limits<std::int32_t>::max() ? static_cast<std::int32_t>(wide) : -1;
        }
        default:
            return static_cast<const std::int32_t*>(m_indices)[row];
        }
    }
    // The text of a row; "" for a row whose index is outside the dictionary (a null row may hold any index).
    [[nodiscard]] std::string_view operator[](std::int64_t row) const noexcept
    {
        const std::int32_t position = index(row);
        return position >= 0 && position < m_values.size() ? m_values[position] : std::string_view();
    }
    [[nodiscard]] const Utf8View& values() const noexcept { return m_values; }

private:
    const void* m_indices = nullptr;
    int m_indexBytes = static_cast<int>(sizeof(std::int32_t));
    std::int64_t m_length = 0;
    Utf8View m_values;
};

} // namespace oos::base
