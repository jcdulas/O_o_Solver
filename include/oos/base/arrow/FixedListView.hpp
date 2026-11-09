#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace oos::base
{

// A fixed-size list column ("fixed_list<float64>[3]": coordinates): `width` contiguous values per row. One of the
// typed views of ContractReader (base/arrow/ColumnViews.hpp), in place.
//
// Thread-safety: copyable, immutable, safe to read from several threads. Valid while the ContractReader lives.
template <typename T>
class FixedListView
{
public:
    FixedListView() = default;
    FixedListView(const T* values, std::int64_t length, std::int64_t width)
        : m_values(values), m_length(length), m_width(width)
    {
    }

    [[nodiscard]] std::int64_t size() const noexcept { return m_length; }
    [[nodiscard]] std::int64_t width() const noexcept { return m_width; }
    [[nodiscard]] std::span<const T> operator[](std::int64_t row) const noexcept
    {
        return {m_values + row * m_width, static_cast<std::size_t>(m_width)};
    }
    // Every value, row after row: size() · width() of them.
    [[nodiscard]] std::span<const T> all() const noexcept
    {
        return {m_values, static_cast<std::size_t>(m_length * m_width)};
    }

private:
    const T* m_values = nullptr;
    std::int64_t m_length = 0;
    std::int64_t m_width = 0;
};

} // namespace oos::base
