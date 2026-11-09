#include "oos/output/envelopes/EnvelopeAccumulator.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>

namespace oos::output
{

namespace
{

constexpr std::array<std::string_view, 4> NODAL_KINDS{"displacements", "singlePointConstrForces",
                                                      "multiPointConstrForces", "appliedLoads"};
constexpr std::array<std::string_view, 4> ELEMENT_KINDS{"stress", "strain", "forces", "strainEnergy"};
constexpr std::string_view MAGNITUDE = "magnitude";

bool isNodalKind(std::string_view kind)
{
    return std::find(NODAL_KINDS.begin(), NODAL_KINDS.end(), kind) != NODAL_KINDS.end();
}

// The value of a row offered for a spec: a component, or the magnitude of the translation of a nodal row.
std::optional<double> quantityOf(const std::vector<std::string>& components, std::span<const double> row,
                                 const std::string& quantity)
{
    for (std::size_t k = 0; k < components.size(); ++k)
        if (components[k] == quantity)
            return row[k];
    return std::nullopt;
}

using RowLabel = std::tuple<std::int32_t, std::int32_t, std::int32_t, double>;

} // namespace

EnvelopeAccumulator::EnvelopeAccumulator(std::vector<EnvelopeSpec> specs) : m_specs(std::move(specs))
{
    for (EnvelopeSpec& spec : m_specs)
    {
        if (spec.kind == "displacement")
            spec.kind = "displacements";
        if (spec.kind == "force")
            spec.kind = "forces";
        const bool isKnown = isNodalKind(spec.kind) ||
                             std::find(ELEMENT_KINDS.begin(), ELEMENT_KINDS.end(), spec.kind) != ELEMENT_KINDS.end();
        if (!isKnown)
            throw std::invalid_argument(base::concatenate(
                {"envelopes: expected a kind among displacements, singlePointConstrForces, multiPointConstrForces, "
                 "appliedLoads, stress, strain, forces, strainEnergy, found '",
                 spec.kind, "'"}));
        if (spec.mode != "max" && spec.mode != "min" && spec.mode != "absMax")
            throw std::invalid_argument(base::concatenate(
                {"envelopes: expected the mode max, min or absMax, found '", spec.mode, "' for ", spec.kind}));
        if (spec.quantity == MAGNITUDE && !isNodalKind(spec.kind))
            throw std::invalid_argument(base::concatenate(
                {"envelopes: 'magnitude' is the length of the translation of a nodal table, found it for '", spec.kind,
                 "'; name a column of the element tables (vonMises, oxx, axial...)"}));
    }
    m_isQuantityFound.assign(m_specs.size(), 0);
}

bool EnvelopeAccumulator::takes(const std::string& kind) const
{
    return std::any_of(m_specs.begin(), m_specs.end(), [&](const EnvelopeSpec& spec) { return spec.kind == kind; });
}

EnvelopeAccumulator::Accumulated& EnvelopeAccumulator::entryFor(std::size_t spec, const std::string& kind,
                                                                const std::string& elementType, bool& isNew)
{
    for (Accumulated& entry : m_entries)
        if (entry.spec == spec && entry.table.elementType == elementType && entry.table.kind == kind)
        {
            isNew = false;
            return entry;
        }
    isNew = true;
    Accumulated entry;
    entry.spec = spec;
    entry.table.kind = kind;
    entry.table.elementType = elementType;
    entry.table.quantity = m_specs[spec].quantity;
    entry.table.mode = m_specs[spec].mode;
    m_entries.push_back(std::move(entry));
    return m_entries.back();
}

void EnvelopeAccumulator::update(Accumulated& entry, std::size_t row, double value, std::int32_t subcase) const
{
    double& kept = entry.table.values[row];
    std::int32_t& keptCase = entry.table.cases[row];
    const std::string& mode = entry.table.mode;
    // NaN (an undefined margin) never replaces a value; the first case wins a tie.
    if (std::isnan(value))
        return;
    bool isBetter = keptCase == 0 && std::isnan(kept);
    if (!isBetter)
    {
        if (mode == "max")
            isBetter = value > kept;
        else if (mode == "min")
            isBetter = value < kept;
        else
            isBetter = std::abs(value) > std::abs(kept);
    }
    if (isBetter)
    {
        kept = value;
        keptCase = subcase;
    }
}

namespace
{

const std::vector<std::string> NODAL_COMPONENTS{"t1", "t2", "t3", "r1", "r2", "r3"};

RowLabel labelOf(const EnvelopeTable& table, std::size_t row)
{
    return RowLabel{table.ids[row], table.nodeIds[row], table.layers[row],
                    table.positions.empty() ? 0.0 : table.positions[row]};
}

} // namespace

bool EnvelopeAccumulator::takesNodal(std::size_t spec, const NodalResultTable& table) const
{
    if (m_specs[spec].kind != table.kind)
        return false;
    return m_specs[spec].quantity == MAGNITUDE || std::find(NODAL_COMPONENTS.begin(), NODAL_COMPONENTS.end(),
                                                            m_specs[spec].quantity) != NODAL_COMPONENTS.end();
}

bool EnvelopeAccumulator::takesElement(std::size_t spec, const ElementResultTable& table) const
{
    return m_specs[spec].kind == table.kind && std::find(table.components.begin(), table.components.end(),
                                                         m_specs[spec].quantity) != table.components.end();
}

void EnvelopeAccumulator::initialiseRows(Accumulated& entry, const NodalResultTable& table)
{
    entry.table.ids.assign(table.ids.data(), table.ids.data() + table.rowCount());
    entry.table.nodeIds.assign(table.rowCount(), 0);
    entry.table.layers.assign(table.rowCount(), 0);
    entry.table.values.assign(table.rowCount(), std::nan(""));
    entry.table.cases.assign(table.rowCount(), 0);
}

void EnvelopeAccumulator::initialiseRows(Accumulated& entry, const ElementResultTable& table)
{
    const std::size_t rows = table.rowCount();
    entry.table.ids.assign(table.ids.data(), table.ids.data() + rows);
    entry.table.nodeIds.assign(table.nodeIds.data(), table.nodeIds.data() + rows);
    entry.table.layers.assign(table.layers.data(), table.layers.data() + rows);
    if (table.hasPositions)
        entry.table.positions.assign(table.positions.data(), table.positions.data() + rows);
    entry.table.values.assign(rows, std::nan(""));
    entry.table.cases.assign(rows, 0);
}

void EnvelopeAccumulator::defineRows(const NodalResultTable& table)
{
    for (std::size_t s = 0; s < m_specs.size(); ++s)
    {
        if (!takesNodal(s, table))
            continue;
        bool isNew = false;
        Accumulated& entry = entryFor(s, table.kind, "", isNew);
        if (isNew)
            initialiseRows(entry, table);
    }
}

void EnvelopeAccumulator::defineRows(const ElementResultTable& table)
{
    for (std::size_t s = 0; s < m_specs.size(); ++s)
    {
        if (!takesElement(s, table))
            continue;
        bool isNew = false;
        Accumulated& entry = entryFor(s, table.kind, table.elementType, isNew);
        if (isNew)
            initialiseRows(entry, table);
    }
}

void EnvelopeAccumulator::offer(const NodalResultTable& table)
{
    for (std::size_t s = 0; s < m_specs.size(); ++s)
    {
        if (!takesNodal(s, table))
            continue;
        const bool isMagnitude = m_specs[s].quantity == MAGNITUDE;
        m_isQuantityFound[s] = 1;
        bool isNew = false;
        Accumulated& entry = entryFor(s, table.kind, "", isNew);
        if (isNew)
            initialiseRows(entry, table);
        // The rows of a nodal table are the points of its request, in the order of the node table: matched by ID.
        const bool isSameRows = entry.table.ids.size() == table.rowCount() &&
                                std::equal(entry.table.ids.begin(), entry.table.ids.end(), table.ids.data());
        if (!isSameRows && entry.sortedRows.size() != entry.table.ids.size())
        {
            entry.sortedRows.resize(entry.table.ids.size());
            std::iota(entry.sortedRows.begin(), entry.sortedRows.end(), std::size_t{0});
            std::sort(entry.sortedRows.begin(), entry.sortedRows.end(),
                      [&](std::size_t a, std::size_t b) { return entry.table.ids[a] < entry.table.ids[b]; });
        }
        const std::vector<std::size_t>& sorted = entry.sortedRows;
        std::size_t next = 0; // the row after the last one matched: a part of a table keeps the table's order
        for (std::size_t row = 0; row < table.rowCount(); ++row)
        {
            std::size_t target = row;
            if (!isSameRows)
            {
                if (next < entry.table.ids.size() && entry.table.ids[next] == table.ids[row])
                    target = next;
                else
                {
                    const auto found = std::lower_bound(sorted.begin(), sorted.end(), table.ids[row],
                                                        [&](std::size_t index, std::int32_t id)
                                                        { return entry.table.ids[index] < id; });
                    if (found == sorted.end() || entry.table.ids[*found] != table.ids[row])
                    {
                        ++m_unmatchedRows;
                        continue;
                    }
                    target = *found;
                }
                next = target + 1;
            }
            const std::span<const double> values(table.values.data() + row * NodalResultTable::COMPONENT_COUNT,
                                                 NodalResultTable::COMPONENT_COUNT);
            const double value = isMagnitude
                                     ? std::sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2])
                                     : *quantityOf(NODAL_COMPONENTS, values, m_specs[s].quantity);
            update(entry, target, value, table.subcase);
        }
    }
}

void EnvelopeAccumulator::offer(const ElementResultTable& table)
{
    for (std::size_t s = 0; s < m_specs.size(); ++s)
    {
        if (!takesElement(s, table))
            continue;
        m_isQuantityFound[s] = 1;
        bool isNew = false;
        Accumulated& entry = entryFor(s, table.kind, table.elementType, isNew);
        const std::size_t rows = table.rowCount();
        if (isNew)
            initialiseRows(entry, table);
        bool isSameRows = entry.table.ids.size() == rows;
        for (std::size_t row = 0; row < rows && isSameRows; ++row)
            isSameRows = entry.table.ids[row] == table.ids[row] && entry.table.nodeIds[row] == table.nodeIds[row] &&
                         entry.table.layers[row] == table.layers[row];
        if (!isSameRows && entry.sortedRows.size() != entry.table.ids.size())
        {
            entry.sortedRows.resize(entry.table.ids.size());
            std::iota(entry.sortedRows.begin(), entry.sortedRows.end(), std::size_t{0});
            std::sort(entry.sortedRows.begin(), entry.sortedRows.end(),
                      [&](std::size_t a, std::size_t b) { return labelOf(entry.table, a) < labelOf(entry.table, b); });
        }
        const std::vector<std::size_t>& sorted = entry.sortedRows;
        const std::size_t width = table.componentCount();
        std::size_t next = 0; // the row after the last one matched: a part of a table keeps the table's order
        for (std::size_t row = 0; row < rows; ++row)
        {
            std::size_t target = row;
            if (!isSameRows)
            {
                const RowLabel label{table.ids[row], table.nodeIds[row], table.layers[row],
                                     table.hasPositions ? table.positions[row] : 0.0};
                if (next < entry.table.ids.size() && labelOf(entry.table, next) == label)
                    target = next;
                else
                {
                    const auto found = std::lower_bound(sorted.begin(), sorted.end(), label,
                                                        [&](std::size_t index, const RowLabel& value)
                                                        { return labelOf(entry.table, index) < value; });
                    if (found == sorted.end() || labelOf(entry.table, *found) != label)
                    {
                        ++m_unmatchedRows;
                        continue;
                    }
                    target = *found;
                }
                next = target + 1;
            }
            const std::span<const double> values(table.values.data() + row * width, width);
            update(entry, target, *quantityOf(table.components, values, m_specs[s].quantity), table.subcase);
        }
    }
}

std::vector<EnvelopeTable> EnvelopeAccumulator::tables() const
{
    for (std::size_t s = 0; s < m_specs.size(); ++s)
        if (m_isQuantityFound[s] == 0)
            throw std::invalid_argument(base::concatenate(
                {"envelopes: no ", m_specs[s].kind, " table of the analysis has the quantity '", m_specs[s].quantity,
                 "'; name one of its columns (or 'magnitude' for a nodal table), and check that the Case Control "
                 "requests the table"}));
    std::vector<EnvelopeTable> result;
    for (std::size_t s = 0; s < m_specs.size(); ++s)
        for (const Accumulated& entry : m_entries)
            if (entry.spec == s)
                result.push_back(entry.table);
    return result;
}

} // namespace oos::output
