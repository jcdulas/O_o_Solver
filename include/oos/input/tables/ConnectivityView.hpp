#pragma once

#include "oos/base/arrow/ColumnViews.hpp"

#include <cstdint>
#include <span>
#include <utility>

namespace oos::input
{

// The grid points (or their rows) of each element of a batch, read in place whichever of O_o_Reader's two layouts the
// column has: a fixed list of n values per row for a table of one card type ("fixed_list<int32>[4]"), a list for a
// table of mixed types ("list<int32>"). Same lifetime and thread-safety as the views of base/arrow/ColumnViews.hpp.
class ConnectivityView
{
public:
    ConnectivityView() = default;
    explicit ConnectivityView(base::FixedListView<std::int32_t> fixed) : m_fixed(fixed), m_isFixed(true) {}
    explicit ConnectivityView(base::ListView<std::int32_t> variable) : m_variable(std::move(variable)) {}

    [[nodiscard]] std::int64_t size() const noexcept { return m_isFixed ? m_fixed.size() : m_variable.size(); }
    [[nodiscard]] std::span<const std::int32_t> operator[](std::int64_t row) const noexcept
    {
        return m_isFixed ? m_fixed[row] : m_variable[row];
    }
    // The number of values per row when every row has the same (a fixed list), 0 otherwise.
    [[nodiscard]] std::int64_t fixedWidth() const noexcept { return m_isFixed ? m_fixed.width() : 0; }
    // The first value of row 0, where the column's values start in the producer's buffer (for no-copy checks).
    [[nodiscard]] const std::int32_t* data() const noexcept
    {
        if (m_isFixed)
            return m_fixed.all().data();
        return size() > 0 ? m_variable[0].data() : nullptr;
    }

private:
    base::FixedListView<std::int32_t> m_fixed;
    base::ListView<std::int32_t> m_variable;
    bool m_isFixed = false;
};

} // namespace oos::input
