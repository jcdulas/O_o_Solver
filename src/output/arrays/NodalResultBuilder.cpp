#include "oos/output/arrays/NodalResultBuilder.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>

namespace oos::output
{

NodalResultTable NodalResultBuilder::build(std::string kind, std::int32_t subcase, std::span<const double> valuesG,
                                           base::MemoryBudget& budget) const
{
    return build(std::move(kind), subcase, valuesG, budget, true, {});
}

NodalResultTable NodalResultBuilder::build(std::string kind, std::int32_t subcase, std::span<const double> valuesG,
                                           base::MemoryBudget& budget, bool isAll,
                                           std::span<const std::int32_t> members) const
{
    if (valuesG.size() != static_cast<std::size_t>(m_dofMap->gSetSize()))
        throw std::invalid_argument(
            base::concatenate({"NodalResultBuilder: expected ", std::to_string(m_dofMap->gSetSize()),
                               " values (the g set), found ", std::to_string(valuesG.size())}));
    const auto grids = static_cast<std::size_t>(m_dofMap->gridCount());
    const auto scalars = static_cast<std::size_t>(m_dofMap->scalarCount());
    // The grid and scalar rows taken, in the order of the DofMap: every row, or those of the members.
    // PERFORMANCE: a SET's members are found by the point index (a lookup each) instead of a search of the SET for
    // every point of the model; sorting the rows keeps the table in the order of the DofMap, so it is the same table.
    // Measured: with the DofMap columns of the same merge, 600 cases of 197 640 DOFs from 11.1 s to 3.7 s (DofMap.hpp).
    base::BudgetedBuffer<std::int32_t> gridRows;
    base::BudgetedBuffer<std::int32_t> scalarRows;
    std::size_t gridCount = 0;
    std::size_t scalarCount = 0;
    if (!isAll)
    {
        gridRows = base::BudgetedBuffer<std::int32_t>(budget, members.size(), "result table grid rows");
        scalarRows = base::BudgetedBuffer<std::int32_t>(budget, members.size(), "result table scalar rows");
        for (const std::int32_t id : members)
        {
            const std::optional<input::PointRef> point = m_dofMap->points().find(id);
            if (!point)
                continue;
            if (point->kind == input::PointKind::Scalar)
                scalarRows[scalarCount++] = point->row;
            else
                gridRows[gridCount++] = point->row;
        }
        const auto sortUnique = [](base::BudgetedBuffer<std::int32_t>& values, std::size_t& count)
        {
            std::int32_t* begin = values.data();
            std::sort(begin, begin + count);
            count = static_cast<std::size_t>(std::unique(begin, begin + count) - begin);
        };
        sortUnique(gridRows, gridCount);
        sortUnique(scalarRows, scalarCount);
    }
    const std::size_t rows = isAll ? grids + scalars : gridCount + scalarCount;
    NodalResultTable table;
    table.kind = std::move(kind);
    table.subcase = subcase;
    table.ids = base::BudgetedBuffer<std::int32_t>(budget, rows, "result table ids");
    table.pointTypes = base::BudgetedBuffer<std::int8_t>(budget, rows, "result table point types");
    table.values =
        base::BudgetedBuffer<double>(budget, rows * NodalResultTable::COMPONENT_COUNT, "result table values");
    std::size_t row = 0;
    const auto addGrid = [&](std::int32_t gridRow)
    {
        table.ids[row] = m_dofMap->gridId(gridRow);
        table.pointTypes[row] = static_cast<std::int8_t>(PointType::Grid);
        for (int c = 0; c < NodalResultTable::COMPONENT_COUNT; ++c)
            table.values[row * NodalResultTable::COMPONENT_COUNT + static_cast<std::size_t>(c)] =
                valuesG[static_cast<std::size_t>(m_dofMap->gridDof(gridRow, c + 1))];
        ++row;
    };
    const auto addScalar = [&](std::int32_t scalarRow)
    {
        table.ids[row] = m_dofMap->scalarId(scalarRow);
        table.pointTypes[row] = static_cast<std::int8_t>(PointType::Scalar);
        table.values[row * NodalResultTable::COMPONENT_COUNT] =
            valuesG[static_cast<std::size_t>(m_dofMap->scalarDof(scalarRow))];
        ++row;
    };
    if (isAll)
    {
        for (std::size_t grid = 0; grid < grids; ++grid)
            addGrid(static_cast<std::int32_t>(grid));
        for (std::size_t s = 0; s < scalars; ++s)
            addScalar(static_cast<std::int32_t>(s));
        return table;
    }
    for (std::size_t k = 0; k < gridCount; ++k)
        addGrid(gridRows[k]);
    for (std::size_t k = 0; k < scalarCount; ++k)
        addScalar(scalarRows[k]);
    return table;
}

} // namespace oos::output
