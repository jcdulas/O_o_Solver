#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/NestedListView.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace oos::input
{

// A column of type "list<fixed_list<float64>[n]>", read in place: per row a list of items of n values each. O_o_Reader
// gives the stations of a PBEAM this way (station_section: A, I1, I2, I12, J, NSM of each station; station_stress: C1
// to F2 of each station). The reading and its checks are base::ContractReader::nestedLists, the one validator of list
// offsets; this class keeps the element code's view of a float64 column with no nulls.
//
// Same lifetime and thread-safety as the views of base/arrow/ColumnViews.hpp: valid while the reader lives; immutable.
class NestedListView
{
public:
    NestedListView() = default;
    // Throws ContractError when the reader has no such column, when it has another type, when a row, an item or a
    // value is null, or when the list offsets do not stay within the items.
    NestedListView(const base::ContractReader& reader, const std::string& column)
        : m_view(reader.nestedLists<double>(column))
    {
    }

    [[nodiscard]] std::int64_t size() const noexcept { return m_view.size(); }
    // The number of values of each item (n).
    [[nodiscard]] std::int64_t width() const noexcept { return m_view.width(); }
    // The number of items of a row.
    [[nodiscard]] std::int64_t itemCount(std::int64_t row) const noexcept { return m_view.itemCount(row); }
    // The n values of item `item` of a row.
    [[nodiscard]] std::span<const double> item(std::int64_t row, std::int64_t item) const noexcept
    {
        return m_view.item(row, item);
    }

private:
    base::NestedListView<double> m_view;
};

} // namespace oos::input
