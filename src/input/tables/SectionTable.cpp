#include "oos/input/tables/SectionTable.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/contract/TableContracts.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

namespace oos::input
{

SectionTable::SectionTable(base::ImportedTable sections)
    : m_reader(std::move(sections), contractOf(TableRole::Sections))
{
    static constexpr std::array<const char*, 8> RECOVERY_COLUMNS{"c1", "c2", "d1", "d2", "e1", "e2", "f1", "f2"};
    static_assert(RECOVERY_COLUMNS.size() == static_cast<std::size_t>(RECOVERY_VALUE_COUNT));
    const auto rows = static_cast<std::size_t>(rowCount());
    const auto width = static_cast<std::size_t>(RECOVERY_VALUE_COUNT);
    m_recoveryPoints.resize(rows * width);
    for (std::size_t column = 0; column < RECOVERY_COLUMNS.size(); ++column)
    {
        const std::span<const double> values = m_reader.values<double>(RECOVERY_COLUMNS[column]);
        for (std::size_t row = 0; row < rows; ++row)
            m_recoveryPoints[row * width + column] = values[row];
    }
    if (m_reader.hasColumn("n1") != m_reader.hasColumn("n2"))
        throw base::ContractError(
            "sectionsTable: expected the neutral-axis columns n1 and n2 together, found only one of them; "
            "give "
            "both, or neither when every shear centre is on its neutral axis");
    const std::span<const std::int32_t> ids = propIds();
    const std::span<const double> xxb = positions();
    for (std::size_t row = 1; row < ids.size(); ++row)
    {
        const bool isSorted = ids[row - 1] < ids[row] || (ids[row - 1] == ids[row] && xxb[row - 1] < xxb[row]);
        if (!isSorted)
            throw base::ContractError(base::concatenate(
                {"sectionsTable: expected the rows sorted by propId, then by increasing xxb, found propId ",
                 std::to_string(ids[row]), " at BeamShapeProp.stationRatio ", std::to_string(xxb[row]),
                 " after propId ", std::to_string(ids[row - 1]), " at BeamShapeProp.stationRatio ",
                 std::to_string(xxb[row - 1]), " (row ", std::to_string(row),
                 "); give the stations of each property once, in increasing BeamShapeProp.stationRatio"}));
    }
}

std::optional<SectionRows> SectionTable::findProperty(std::int32_t propId) const
{
    const std::span<const std::int32_t> ids = propIds();
    const auto [first, last] = std::equal_range(ids.begin(), ids.end(), propId);
    if (first == last)
        return std::nullopt;
    return SectionRows{first - ids.begin(), last - ids.begin()};
}

} // namespace oos::input
