#include "oos/design/responses/ResultLookup.hpp"

namespace oos::design
{

namespace
{

// Positions 0 and 1 of a station table: ends A and B (ElementResultTable: positions along the element, 0 to 1).
constexpr double END_A_POSITION = 0.0;
constexpr double END_B_POSITION = 1.0;

} // namespace

ResultLookup::ResultLookup(const output::ResultArrays& arrays) : m_arrays(&arrays)
{
    for (std::size_t t = 0; t < arrays.nodalTables.size(); ++t)
        m_nodalTables.emplace(std::make_pair(arrays.nodalTables[t].kind, arrays.nodalTables[t].subcase), t);
    for (std::size_t t = 0; t < arrays.elementTables.size(); ++t)
    {
        const output::ElementResultTable& table = arrays.elementTables[t];
        m_elementTables.emplace(std::make_tuple(table.kind, table.elementType, table.subcase), t);
    }
}

std::optional<double> ResultLookup::nodalValue(std::string_view kind, std::int32_t subcase, std::int32_t id,
                                               std::int32_t component)
{
    const auto found = m_nodalTables.find({std::string(kind), subcase});
    if (found == m_nodalTables.end() || component < 1 || component > output::NodalResultTable::COMPONENT_COUNT)
        return std::nullopt;
    const output::NodalResultTable& table = m_arrays->nodalTables[found->second];
    auto [rows, isNew] = m_nodalRows.try_emplace(found->second);
    if (isNew)
        for (std::size_t r = 0; r < table.rowCount(); ++r)
            if (table.pointTypes[r] == static_cast<std::int8_t>(output::PointType::Grid))
                rows->second.emplace(table.ids[r], r);
    const auto row = rows->second.find(id);
    if (row == rows->second.end())
        return std::nullopt;
    return table
        .values[row->second * output::NodalResultTable::COMPONENT_COUNT + static_cast<std::size_t>(component - 1)];
}

const output::ElementResultTable* ResultLookup::elementTable(std::string_view tableKind, std::string_view elementType,
                                                             std::int32_t subcase) const
{
    const auto found = m_elementTables.find({std::string(tableKind), std::string(elementType), subcase});
    return found == m_elementTables.end() ? nullptr : &m_arrays->elementTables[found->second];
}

std::span<const double> ResultLookup::elementRow(std::string_view tableKind, std::string_view elementType,
                                                 std::int32_t subcase, std::int32_t elementId, ItemPlace place,
                                                 std::int32_t layer)
{
    const auto found = m_elementTables.find({std::string(tableKind), std::string(elementType), subcase});
    if (found == m_elementTables.end())
        return {};
    const output::ElementResultTable& table = m_arrays->elementTables[found->second];
    auto [rows, isNew] = m_elementRows.try_emplace(found->second);
    if (isNew)
        for (std::size_t r = 0; r < table.rowCount(); ++r)
        {
            auto [range, isFirst] = rows->second.try_emplace(table.ids[r], r, r + 1);
            if (!isFirst)
                range->second.second = r + 1;
        }
    const auto range = rows->second.find(elementId);
    if (range == rows->second.end())
        return {};
    const std::size_t width = table.componentCount();
    for (std::size_t r = range->second.first; r < range->second.second; ++r)
    {
        // The centre: grid point 0 (a shell's centre rows, a rod, a spring, a Bar2) of the layer asked; the ends of
        // a station table: positions 0 and 1.
        const bool isPlace =
            place == ItemPlace::Centre
                ? !table.hasPositions && table.nodeIds[r] == 0
                : table.hasPositions &&
                      table.positions[r] == (place == ItemPlace::EndA ? END_A_POSITION : END_B_POSITION);
        if (isPlace && (layer == 0 || table.layers[r] == layer))
            return {table.values.data() + r * width, width};
    }
    return {};
}

std::optional<double> ResultLookup::elementValue(const ItemLocation& location, std::string_view elementType,
                                                 std::int32_t subcase, std::int32_t elementId, std::int32_t layer)
{
    const output::ElementResultTable* table = elementTable(location.tableKind, elementType, subcase);
    if (table == nullptr)
        return std::nullopt;
    const std::optional<std::size_t> column = location.columnIn(table->components);
    const std::span<const double> row =
        elementRow(location.tableKind, elementType, subcase, elementId, location.place, layer);
    if (!column || *column >= row.size())
        return std::nullopt;
    return row[*column];
}

} // namespace oos::design
