#pragma once

#include "oos/assembly/loads/LoadSetIndex.hpp"
#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/input/tables/LoadTable.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace oos::assembly
{

// One row of pressureTable as its default form gives it (a PointsPressureLoad, ShellPressureLoad or FacePressureLoad on
// one element or face): what LoadVectorBuilder reads of it. A null of the table is a NaN for the reals and a false flag
// for the others.
struct PressureRow
{
    std::int32_t setId = 0;
    std::string_view card;
    bool hasElement = false; // false for a PointsPressureLoad, a pressure on the face of grid points
    std::int32_t elementId = 0;
    bool hasFace = false; // the face of a solid, the grid points of a PointsPressureLoad
    std::span<const std::int32_t> faceNodes;
    std::array<double, 4> corners{}; // p1 to p4, NaN when null
    bool hasFrameId = false;
    std::int32_t frameId = 0;
    std::array<double, 3> direction{}; // NaN when null
    double scale = 0.0;                // NaN when null
};

// The rows of pressureTable in either of O_o_Reader's forms:
// - the rows (the default form), one per card and element, read in place;
// - the fields (detail="fields", input::LoadTable::isPressureFields()), a row per field of consecutive cards sharing
//   all but their element and pressure: its element_count rows are numbered after the previous fields' (the rows of
//   the default form, in order), the k-th row's element the k-th of the field's element list (its (first, last) runs,
//   given on the first field of the list), its pressures values[0..corner_count) for a uniform field, else
//   values[k·corner_count..], one value standing for the four corners. O_o_Reader expands them so to the rows of the
//   default form to the bit, so that every load made from them is the rows' to the bit.
// The fields of a large model hold its 73 M pressure rows in 560 MB instead of 12.8 GB.
//
// Thread-safety: immutable once built; concurrent reads are safe. Ownership: borrows the load table (its snapshot must
// outlive it); owns its index (budgeted: a set and row entry per row of the default form, per field and run of the
// fields).
class PressureRows
{
public:
    PressureRows() = default;
    PressureRows(const input::LoadTable& loads, base::MemoryBudget& budget);

    [[nodiscard]] bool isFields() const noexcept { return m_isFields; }
    // The rows of the default form.
    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_rowCount; }
    // The rows of set `setId` (its own cards).
    [[nodiscard]] std::int64_t rowCountOf(std::int32_t setId) const;
    [[nodiscard]] bool hasRows(std::int32_t setId) const { return rowCountOf(setId) > 0; }

    // Calls visit(row, pressureRow) for each row of set `setId`, in table order (`row` its index in the default form).
    template <typename Visit>
    void forEachRowOf(std::int32_t setId, Visit&& visit) const
    {
        if (m_table == nullptr)
            return;
        if (!m_isFields)
        {
            for (const std::int64_t row : m_index.rowsOf(setId))
                visit(row, rowOfTable(row));
            return;
        }
        for (const std::int64_t field : m_index.rowsOf(setId))
            forEachRowOfField(field, visit);
    }

    // The row `row` of the default form (O(log) in the fields and the runs of its list).
    [[nodiscard]] PressureRow rowAt(std::int64_t row) const;

private:
    [[nodiscard]] PressureRow rowOfTable(std::int64_t row) const;
    // The row of field `field` before its pressures and element: its card, set, face, system, direction, scale.
    [[nodiscard]] PressureRow fieldRow(std::int64_t field) const;
    void setCorners(std::int64_t field, std::int64_t k, PressureRow& row) const;

    template <typename Visit>
    void forEachRowOfField(std::int64_t field, Visit& visit) const
    {
        PressureRow row = fieldRow(field);
        const auto at = static_cast<std::size_t>(field);
        const std::int64_t first = m_fieldFirstRow[at];
        const std::int64_t count = m_fieldFirstRow[at + 1] - first;
        if (!row.hasElement)
        {
            for (std::int64_t k = 0; k < count; ++k)
            {
                setCorners(field, k, row);
                visit(first + k, row);
            }
            return;
        }
        const std::span<const std::int32_t> runs = runsOfField(field);
        std::int64_t k = 0;
        for (std::size_t r = 0; r + 1 < runs.size() && k < count; r += 2)
            for (std::int32_t elementId = runs[r]; elementId <= runs[r + 1] && k < count; ++elementId, ++k)
            {
                row.elementId = elementId;
                setCorners(field, k, row);
                visit(first + k, row);
            }
    }
    // The (first, last) runs of the element list of a field (given on its first field).
    [[nodiscard]] std::span<const std::int32_t> runsOfField(std::int64_t field) const;

    const base::ContractReader* m_table = nullptr;
    bool m_isFields = false;
    std::int64_t m_rowCount = 0;
    LoadSetIndex m_index; // the rows, or the fields, by set
    base::DictionaryView m_cards;
    base::ListView<std::int32_t> m_faceNodes;
    base::ValidityView m_validFaces;
    std::span<const std::int32_t> m_setIds;
    std::span<const std::int32_t> m_frameIds;
    base::ValidityView m_validFrameIds;
    std::array<std::span<const double>, 3> m_direction{};
    std::array<base::ValidityView, 3> m_validDirection{};
    std::span<const double> m_scales;
    base::ValidityView m_validScales;
    // The default form.
    std::span<const std::int32_t> m_elementIds;
    base::ValidityView m_validElementIds;
    std::array<std::span<const double>, 4> m_corners{};
    // The fields.
    base::BudgetedBuffer<std::int64_t> m_fieldFirstRow; // fields + 1
    base::BudgetedBuffer<std::int64_t> m_listField;     // per element list ID: the field that gives its runs, or -1
    base::BudgetedBuffer<std::int64_t> m_runRowStart;   // per run of every list: its first row in the list
    base::BudgetedBuffer<std::int64_t> m_listRunStart;  // per element list ID: its first run in m_runRowStart
    std::span<const std::int32_t> m_elementLists;
    base::ValidityView m_validLists;
    base::ListView<std::int32_t> m_elementIdRuns;
    base::BooleanView m_isUniform;
    std::span<const std::int32_t> m_cornerCounts;
    base::ListView<double> m_values;
};

} // namespace oos::assembly
