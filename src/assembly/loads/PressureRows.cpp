#include "oos/assembly/loads/PressureRows.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace oos::assembly
{

namespace
{

constexpr std::array<const char*, 3> DIRECTION_COLUMNS{"directionX", "directionY", "directionZ"};
constexpr std::array<const char*, 4> CORNER_COLUMNS{"p1", "p2", "p3", "p4"};
// One value of a field stands for the four corners (O_o_Reader: a pressure a row when the four are equal).
constexpr std::int32_t ONE_VALUE_FOR_FOUR = 1;
constexpr std::int32_t FOUR_CORNERS = 4;

[[noreturn]] void refuseFields(const std::string& what)
{
    throw base::ContractError(base::concatenate(
        {"pressureFields (pressureTable(detail=\"fields\")): ", what,
         "; take the table again from O_o_Reader (data contract 55 or later), or its rows (detail=\"rows\")"}));
}

} // namespace

PressureRows::PressureRows(const input::LoadTable& loads, base::MemoryBudget& budget)
    : m_table(&loads.pressures()), m_isFields(loads.isPressureFields())
{
    const base::ContractReader& table = *m_table;
    m_setIds = table.values<std::int32_t>("setId");
    m_index = LoadSetIndex(m_setIds, budget);
    m_cards = table.dictionaryTexts("card");
    m_faceNodes = table.lists<std::int32_t>("faceNodes", base::NullPolicy::Allow);
    m_validFaces = table.validity("faceNodes");
    m_frameIds = table.values<std::int32_t>("frameId", base::NullPolicy::Allow);
    m_validFrameIds = table.validity("frameId");
    for (std::size_t c = 0; c < 3; ++c)
    {
        m_direction[c] = table.values<double>(DIRECTION_COLUMNS[c], base::NullPolicy::Allow);
        m_validDirection[c] = table.validity(DIRECTION_COLUMNS[c]);
    }
    m_scales = table.values<double>("scale", base::NullPolicy::Allow);
    m_validScales = table.validity("scale");
    if (!m_isFields)
    {
        m_rowCount = table.rowCount();
        m_elementIds = table.values<std::int32_t>("elementId", base::NullPolicy::Allow);
        m_validElementIds = table.validity("elementId");
        for (std::size_t c = 0; c < 4; ++c)
        {
            m_corners[c] = table.values<double>(CORNER_COLUMNS[c], base::NullPolicy::Allow);
        }
        return;
    }

    // The fields: their first rows, and the runs of every element list with the first row of each run in its list.
    const std::span<const std::int64_t> counts = table.values<std::int64_t>("elementCount");
    m_elementLists = table.values<std::int32_t>("elementList", base::NullPolicy::Allow);
    m_validLists = table.validity("elementList");
    m_elementIdRuns = table.lists<std::int32_t>("eidRuns");
    m_isUniform = table.booleans("isUniform");
    m_cornerCounts = table.values<std::int32_t>("cornerCount");
    m_values = table.lists<double>("values", base::NullPolicy::Allow);
    const auto fieldCount = static_cast<std::size_t>(table.rowCount());
    m_fieldFirstRow = base::BudgetedBuffer<std::int64_t>(budget, fieldCount + 1, "pressure fields: rows");
    std::int32_t largestList = -1;
    std::size_t runCount = 0;
    for (std::size_t f = 0; f < fieldCount; ++f)
    {
        const auto at = static_cast<std::int64_t>(f);
        if (counts[f] < 0)
            refuseFields(base::concatenate({"field ", std::to_string(f), " has ", std::to_string(counts[f]),
                                            " rows; expected a count of zero or more"}));
        m_fieldFirstRow[f + 1] = m_fieldFirstRow[f] + counts[f];
        const std::int32_t corners = m_cornerCounts[f];
        if (corners != ONE_VALUE_FOR_FOUR && corners != FOUR_CORNERS)
            refuseFields(base::concatenate(
                {"field ", std::to_string(f), " has ", std::to_string(corners), " pressures a row; expected 1 or 4"}));
        const std::int64_t expected = (m_isUniform[at] ? 1 : counts[f]) * corners;
        if (static_cast<std::int64_t>(m_values[at].size()) != expected)
            refuseFields(base::concatenate({"field ", std::to_string(f), " has ", std::to_string(m_values[at].size()),
                                            " values; expected ", std::to_string(expected)}));
        if (m_validLists.isValid(at))
            largestList = std::max(largestList, m_elementLists[f]);
        if (m_elementIdRuns[at].size() % 2 != 0)
            refuseFields(base::concatenate({"field ", std::to_string(f), " has an odd number of run bounds"}));
        runCount += m_elementIdRuns[at].size() / 2;
    }
    m_rowCount = m_fieldFirstRow[fieldCount];
    const auto lists = static_cast<std::size_t>(largestList + 1);
    m_listField = base::BudgetedBuffer<std::int64_t>(budget, lists, "pressure fields: element lists");
    m_listRunStart = base::BudgetedBuffer<std::int64_t>(budget, lists + 1, "pressure fields: element lists");
    m_runRowStart = base::BudgetedBuffer<std::int64_t>(budget, runCount, "pressure fields: element runs");
    std::fill(m_listField.data(), m_listField.data() + lists, std::int64_t{-1});
    std::size_t nextRun = 0;
    for (std::size_t f = 0; f < fieldCount; ++f)
    {
        const auto at = static_cast<std::int64_t>(f);
        const std::span<const std::int32_t> runs = m_elementIdRuns[at];
        if (runs.empty())
            continue;
        if (!m_validLists.isValid(at))
            refuseFields(base::concatenate({"field ", std::to_string(f), " gives element runs but no element list"}));
        const auto list = static_cast<std::size_t>(m_elementLists[f]);
        if (m_listField[list] >= 0)
            refuseFields(base::concatenate({"element list ", std::to_string(list), " is given twice"}));
        m_listField[list] = at;
        m_listRunStart[list] = static_cast<std::int64_t>(nextRun);
        std::int64_t rows = 0;
        for (std::size_t r = 0; r < runs.size(); r += 2)
        {
            if (runs[r + 1] < runs[r])
                refuseFields(
                    base::concatenate({"element list ", std::to_string(list), " has the run (", std::to_string(runs[r]),
                                       ", ", std::to_string(runs[r + 1]), ") whose last element is before its first"}));
            m_runRowStart[nextRun++] = rows;
            rows += static_cast<std::int64_t>(runs[r + 1]) - runs[r] + 1;
        }
    }
    // A list's runs end where the next list's (by ID) start; m_listRunStart is filled for the lists given only.
    for (std::size_t f = 0; f < fieldCount; ++f)
    {
        const auto at = static_cast<std::int64_t>(f);
        if (!m_validLists.isValid(at))
            continue;
        const auto list = static_cast<std::size_t>(m_elementLists[f]);
        if (m_listField[list] < 0)
            refuseFields(base::concatenate({"field ", std::to_string(f), " names element list ", std::to_string(list),
                                            ", which no field gives the runs of"}));
        const std::span<const std::int32_t> runs = runsOfField(at);
        std::int64_t rows = 0;
        for (std::size_t r = 0; r < runs.size(); r += 2)
            rows += static_cast<std::int64_t>(runs[r + 1]) - runs[r] + 1;
        if (rows != counts[f])
            refuseFields(base::concatenate({"field ", std::to_string(f), " has ", std::to_string(counts[f]),
                                            " rows but its element list ", std::to_string(list), " holds ",
                                            std::to_string(rows), " elements"}));
    }
}

std::int64_t PressureRows::rowCountOf(std::int32_t setId) const
{
    if (m_table == nullptr)
        return 0;
    const std::span<const std::int64_t> rows = m_index.rowsOf(setId);
    if (!m_isFields)
        return static_cast<std::int64_t>(rows.size());
    std::int64_t count = 0;
    for (const std::int64_t field : rows)
        count +=
            m_fieldFirstRow[static_cast<std::size_t>(field) + 1] - m_fieldFirstRow[static_cast<std::size_t>(field)];
    return count;
}

PressureRow PressureRows::rowOfTable(std::int64_t row) const
{
    const auto index = static_cast<std::size_t>(row);
    PressureRow out;
    out.setId = m_setIds[index];
    out.card = m_cards[row];
    out.hasElement = m_validElementIds.isValid(row);
    out.elementId = out.hasElement ? m_elementIds[index] : 0;
    out.hasFace = m_validFaces.isValid(row);
    if (out.hasFace)
        out.faceNodes = m_faceNodes[row];
    for (std::size_t c = 0; c < 4; ++c)
        out.corners[c] = m_corners[c][index]; // as the table holds it (a null is the NaN O_o_Reader writes there)
    out.hasFrameId = m_validFrameIds.isValid(row);
    out.frameId = out.hasFrameId ? m_frameIds[index] : 0;
    for (std::size_t c = 0; c < 3; ++c)
        out.direction[c] =
            m_validDirection[c].isValid(row) ? m_direction[c][index] : std::numeric_limits<double>::quiet_NaN();
    out.scale = m_scales[index];
    return out;
}

PressureRow PressureRows::fieldRow(std::int64_t field) const
{
    const auto index = static_cast<std::size_t>(field);
    PressureRow out;
    out.setId = m_setIds[index];
    out.card = m_cards[field];
    out.hasElement = m_validLists.isValid(field);
    out.hasFace = m_validFaces.isValid(field);
    if (out.hasFace)
        out.faceNodes = m_faceNodes[field];
    out.hasFrameId = m_validFrameIds.isValid(field);
    out.frameId = out.hasFrameId ? m_frameIds[index] : 0;
    for (std::size_t c = 0; c < 3; ++c)
        out.direction[c] =
            m_validDirection[c].isValid(field) ? m_direction[c][index] : std::numeric_limits<double>::quiet_NaN();
    out.scale = m_scales[index];
    return out;
}

void PressureRows::setCorners(std::int64_t field, std::int64_t k, PressureRow& row) const
{
    const std::span<const double> values = m_values[field];
    const auto corners = static_cast<std::size_t>(m_cornerCounts[static_cast<std::size_t>(field)]);
    const std::size_t at = m_isUniform[field] ? 0 : static_cast<std::size_t>(k) * corners;
    for (std::size_t c = 0; c < 4; ++c)
        row.corners[c] = values[at + (corners == 1 ? 0 : c)];
}

std::span<const std::int32_t> PressureRows::runsOfField(std::int64_t field) const
{
    const auto list = static_cast<std::size_t>(m_elementLists[static_cast<std::size_t>(field)]);
    return m_elementIdRuns[m_listField[list]];
}

PressureRow PressureRows::rowAt(std::int64_t row) const
{
    if (!m_isFields)
        return rowOfTable(row);
    const std::int64_t* begin = m_fieldFirstRow.data();
    const std::int64_t* end = begin + m_fieldFirstRow.size();
    // The field whose rows hold `row`: the last whose first row is at or before it (a field of no row skipped).
    const std::int64_t* found = std::upper_bound(begin, end, row) - 1;
    const std::int64_t field = found - begin;
    const std::int64_t k = row - *found;
    PressureRow out = fieldRow(field);
    setCorners(field, k, out);
    if (!out.hasElement)
        return out;
    const auto list = static_cast<std::size_t>(m_elementLists[static_cast<std::size_t>(field)]);
    const std::span<const std::int32_t> runs = m_elementIdRuns[m_listField[list]];
    const std::int64_t* starts = m_runRowStart.data() + m_listRunStart[list];
    const std::int64_t* startsEnd = starts + runs.size() / 2;
    const std::int64_t* run = std::upper_bound(starts, startsEnd, k) - 1;
    out.elementId = runs[static_cast<std::size_t>(2 * (run - starts))] + static_cast<std::int32_t>(k - *run);
    return out;
}

} // namespace oos::assembly
