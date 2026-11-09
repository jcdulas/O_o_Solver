#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace oos::base
{

// The texts of a utf8 column: per row a slice of one byte buffer, between two int32 offsets. One of the typed views of
// ContractReader (base/arrow/ColumnViews.hpp), in place.
//
// Thread-safety: copyable, immutable, safe to read from several threads. Valid while the ContractReader lives.
class Utf8View
{
public:
    Utf8View() = default;
    // `offsets` points at the offset of row 0; there are length + 1 of them.
    Utf8View(const std::int32_t* offsets, const char* data, std::int64_t length)
        : m_offsets(offsets), m_data(data), m_length(length)
    {
    }

    [[nodiscard]] std::int64_t size() const noexcept { return m_length; }
    [[nodiscard]] std::string_view operator[](std::int64_t row) const noexcept
    {
        return {m_data + m_offsets[row], static_cast<std::size_t>(m_offsets[row + 1] - m_offsets[row])};
    }

private:
    const std::int32_t* m_offsets = nullptr;
    const char* m_data = nullptr;
    std::int64_t m_length = 0;
};

} // namespace oos::base
