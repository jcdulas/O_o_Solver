#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"
#include "oos/input/tables/BorrowedArray.hpp"

#include <cstdint>
#include <optional>
#include <span>

namespace oos::input
{

// The grid points of the model (cardTable(deck, "nodes") of O_o_Reader): ID, position system CP, displacement system
// CD, permanent constraints PS, superelement SEID, and the position in the basic system (xyzGlobal, whatever CP); the
// rotation from each grid point's CD system to the basic system (displacementRotations(deck): per row a 3 x 3 matrix,
// row-major, whose columns are the directions 1, 2, 3 of CD in the basic system, evaluated at the point); and the
// scalar points (SPOINT, EPOINT), which share the ID space of the grid points.
//
// Row r of every column is the grid point of row r of cardTable(deck, "nodes"), which the nodesRows columns of the
// element batches index.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported tables; the rotations are borrowed.
class NodeTable
{
public:
    // Throws ContractError when a table breaks its contract or the rotations are not 9 values per grid point.
    NodeTable(base::ImportedTable nodes, std::optional<base::ImportedTable> scalarPoints,
              BorrowedArray<double> cdRotations);

    [[nodiscard]] std::int64_t rowCount() const noexcept { return m_nodes.rowCount(); }
    [[nodiscard]] std::span<const std::int32_t> ids() const { return m_nodes.values<std::int32_t>("id"); }
    // The system of the position (CP) and the displacement system (CD) of each grid point.
    [[nodiscard]] std::span<const std::int32_t> positionFrameIds() const
    {
        return m_nodes.values<std::int32_t>("positionFrameId");
    }
    [[nodiscard]] std::span<const std::int32_t> displacementFrameIds() const
    {
        return m_nodes.values<std::int32_t>("displacementFrameId");
    }
    // The permanent constraints (PS) as a component code (123456 digits; 0 for none).
    [[nodiscard]] std::span<const std::int32_t> permanentConstraints() const
    {
        return m_nodes.values<std::int32_t>("permanentConstraints");
    }
    // The superelement of each grid point (SEID).
    [[nodiscard]] std::span<const std::int32_t> superelementIds() const
    {
        return m_nodes.values<std::int32_t>("superelementId");
    }
    [[nodiscard]] base::FixedListView<double> positionsInBasic() const
    {
        return m_nodes.fixedLists<double>("xyzGlobal");
    }

    // The CD-to-basic rotations, 9 values per grid point (empty when the snapshot was taken without them).
    [[nodiscard]] bool hasCdRotations() const noexcept { return !m_cdRotations.empty(); }
    [[nodiscard]] std::span<const double> cdRotations() const noexcept { return m_cdRotations.values; }

    [[nodiscard]] std::span<const std::int32_t> scalarPointIds() const;

    [[nodiscard]] const base::ContractReader& reader() const noexcept { return m_nodes; }
    [[nodiscard]] const base::ContractReader* scalarPointReader() const noexcept
    {
        return m_scalarPoints ? &*m_scalarPoints : nullptr;
    }

private:
    base::ContractReader m_nodes;
    std::optional<base::ContractReader> m_scalarPoints;
    BorrowedArray<double> m_cdRotations;
};

} // namespace oos::input
