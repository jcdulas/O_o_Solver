#pragma once

#include "oos/base/arrow/ImportedTable.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace oos::input
{

// One value given to one cell of a real column: the row of the table, the item of a fixed-list column
// ("fixed_list<float64>[n]", from 0), of a list ("list<float64>", the plies of a PCOMP, the dimensions of a PBARL: the
// position in the list of the row) or of a list of fixed lists ("list<fixed_list<float64>[n]>", the stations of a
// PBEAM: station · n + component) or SCALAR_ITEM for a "float64" column, and the value.
struct CellValue
{
    static constexpr std::int32_t SCALAR_ITEM = -1;

    std::int64_t row = 0;
    std::int32_t item = SCALAR_ITEM;
    double value = 0.0;
};

// The new values of some cells of the real columns of a snapshot's tables, and the snapshot that reads them
// (evaluating a design point and its gradients never edits the model: the design point is applied as an overlay of the
// snapshot). The design layer fills it with the field values of a design point (property, material and element
// columns); applyTo() gives a new ModelSnapshot that reads every column of the base snapshot in place except the
// overlaid ones, which it owns. The element families need no change: they read the overlaid snapshot as any other.
//
// How the sharing works: each table of the new snapshot is a new top-level Arrow struct whose children are the base
// table's column structures (borrowed), except the overlaid columns, replaced by copies holding the new values. The
// release callback of a shared table frees only what it owns and the reference that keeps the base alive, never a
// borrowed child (the Arrow C Data Interface lets a producer's release do what it owns). Cost: the overlaid columns
// are copied (8 bytes per row of each overlaid column; a list of fixed lists copies its items and keeps the base's
// list offsets); everything else is shared.
//
// Thread-safety: none while being filled; applyTo() is const and may run concurrently. Ownership: owns its values.
class ColumnOverlay
{
public:
    // The new values of one column of one table.
    struct ColumnEdits
    {
        std::string table; // a name ModelSnapshot::findTable() accepts: "properties/ShellProp", "materialsTable"
        std::string column;
        std::vector<CellValue> cells; // in the order they were set
    };

    // Records a value. std::invalid_argument when the item is below SCALAR_ITEM. A cell given twice is refused by
    // applyTo() (one writer per cell: a design model refuses a field driven by two relations before it gets here).
    void setValue(const std::string& table, const std::string& column, const CellValue& cell);

    [[nodiscard]] bool isEmpty() const noexcept { return m_columns.empty(); }
    [[nodiscard]] std::size_t cellCount() const noexcept;
    // The overlaid columns, in the order of their first value.
    [[nodiscard]] const std::vector<ColumnEdits>& columns() const noexcept { return m_columns; }

    // The snapshot `base` with the overlaid values. `keepAlive` keeps the base's buffers alive as long as the new
    // snapshot lives (the shared_ptr that owns `base`, or empty when the caller guarantees that `base` outlives the
    // result). The versions are the base's: the design stamps, not the versions, tell overlays apart.
    // std::invalid_argument naming the table, column, row or item when the base has no such table, the column is not
    // "float64", "fixed_list<float64>[n]", "list<float64>" or "list<fixed_list<float64>[n]>", the row is outside the
    // table, the item does not fit the column (or the list or stations of its row), or a cell was given twice;
    // ContractError as SnapshotBuilder::build().
    [[nodiscard]] ModelSnapshot applyTo(const ModelSnapshot& base, const std::shared_ptr<const void>& keepAlive) const;

private:
    std::vector<ColumnEdits> m_columns;
};

// A float64 column O_o_Solver adds to a table of O_o_Reader: its name and one value per row.
struct AddedRealColumn
{
    std::string name;
    std::vector<double> values;
};

// The table `source` (named `name` in messages) with the columns `added` after its own, the source's columns shared as
// ColumnOverlay shares them, the added ones owned by the result (the n1, n2 of the sections table that O_o_Solver's
// snapshot computes from O_o_Reader's shear centres). `keepAlive` keeps the source's buffers alive (the
// shared_ptr that owns it). std::invalid_argument when a name is taken or a column has not one value per row.
[[nodiscard]] base::ImportedTable appendRealColumns(const std::string& name, const base::ImportedTable& source,
                                                    const std::shared_ptr<const void>& keepAlive,
                                                    std::vector<AddedRealColumn> added);

} // namespace oos::input
