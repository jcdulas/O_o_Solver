#pragma once

#include "oos/base/arrow/ContractReader.hpp"
#include "oos/base/arrow/ImportedTable.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oos::input
{

// The rows [first, last) of one side of a fastener: its piercing point (empty when the table has none).
struct PiercingRows
{
    std::int64_t first = 0;
    std::int64_t last = 0;

    [[nodiscard]] std::int64_t count() const noexcept { return last - first; }
    [[nodiscard]] bool isEmpty() const noexcept { return last == first; }
};

// The rows of both sides of one fastener.
struct FastenerPiercingRows
{
    PiercingRows sideA;
    PiercingRows sideB;
};

// A side of a fastener whose piercing point O_o_Reader did not find: the field of the card the problem is about (its
// neutral name, nodeA or nodeB when the reader names none) and
// what it could not find (its tolerance, the nearest element), which the element reports as its problem.
struct SideFailure
{
    std::int32_t elementId = 0;
    bool isSideA = true;
    std::string field;
    std::string message;
};

// The piercing points of the CFAST and CWELD elements (O_o_Reader fastenerPiercings, data contract 60), in the layout
// the elements, the fastener constraints and the spider absorption read: for each side A and B of each fastener, one
// row: its piercing point (GA or GB, given on the card or found by the reader on the shells), with the grid points of
// the shell element it pierces (or of the face of grid points of a CWELD GRIDID) and the shape functions of that
// element at the point, which elements/fasteners/PiercingInterpolation turns into the motion of the point. The sides of
// a CWELD ALIGN and the GS end of a point-to-patch CWELD have no row (the element connects those grid points directly);
// a side whose point the reader did not find has a failure instead (findFailure).
//
// Invariant: the rows are sorted by elementId, then side A before side B, each side once; every row has as many shape
// functions as grid points, and at least one grid point. The rows are a copy of the found sides of the reader's
// table (std::vector: about a hundred bytes per side, less than the producer's table, which the snapshot holds anyway).
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns the imported table and the copy.
class FastenerPiercingTable
{
public:
    // Throws ContractError when the table breaks its contract, holds a side other than "A" and "B", a side twice, or a
    // found side whose grid points and shape functions differ in number or are none.
    explicit FastenerPiercingTable(base::ImportedTable piercings);

    [[nodiscard]] std::int64_t rowCount() const noexcept { return static_cast<std::int64_t>(m_fastenerIds.size()); }
    [[nodiscard]] std::span<const std::int32_t> fastenerIds() const noexcept { return m_fastenerIds; }
    // The neutral card name: "Fastener" (CFAST) or "Weld" (CWELD).
    [[nodiscard]] std::string_view card(std::int64_t row) const;
    [[nodiscard]] bool isSideA(std::int64_t row) const { return m_isSideA[static_cast<std::size_t>(row)] != 0; }
    // GA or GB as given on the card; null (isPointGridGiven) when the point is generated.
    [[nodiscard]] std::span<const std::int32_t> pointGrids() const noexcept { return m_pointGrids; }
    [[nodiscard]] base::ValidityView isPointGridGiven() const { return {m_pointGridBits.data(), 0, rowCount()}; }
    // The piercing point of the side, in the basic system.
    [[nodiscard]] base::FixedListView<double> piercingPoints() const
    {
        return {m_piercingPoints.data(), rowCount(), 3};
    }
    // The shell element pierced (0 for the face of grid points of a CWELD GRIDID).
    [[nodiscard]] std::span<const std::int32_t> elementIds() const noexcept { return m_elementIds; }
    // The grid points of the pierced element and its shape functions at the piercing point.
    [[nodiscard]] base::ListView<std::int32_t> grids() const
    {
        return {base::ListOffsets(nullptr, m_offsets.data()), m_grids.data(), rowCount()};
    }
    [[nodiscard]] base::ListView<double> weights() const
    {
        return {base::ListOffsets(nullptr, m_offsets.data()), m_weights.data(), rowCount()};
    }

    // The rows of a fastener, or nothing when the table has none for it.
    [[nodiscard]] std::optional<FastenerPiercingRows> findFastener(std::int32_t elementId) const;
    // Why the reader did not find the piercing point of a side, or null when it did (or the side has no row).
    [[nodiscard]] const SideFailure* findFailure(std::int32_t elementId, bool isSideA) const;

    // O_o_Reader's table as it came (the snapshot's findTable gives it).
    [[nodiscard]] const base::ContractReader& reader() const noexcept { return m_reader; }

private:
    base::ContractReader m_reader;
    std::vector<std::string> m_cardNames;
    std::vector<std::int32_t> m_fastenerIds;
    std::vector<std::uint8_t> m_cardIndices;
    std::vector<std::uint8_t> m_isSideA;
    std::vector<std::int32_t> m_pointGrids;
    std::vector<std::uint8_t> m_pointGridBits; // Arrow validity bitmap of m_pointGrids
    std::vector<double> m_piercingPoints;      // 3 per row
    std::vector<std::int32_t> m_elementIds;
    std::vector<std::int64_t> m_offsets; // rowCount() + 1, into m_grids and m_weights
    std::vector<std::int32_t> m_grids;
    std::vector<double> m_weights;
    std::vector<SideFailure> m_failures; // sorted by elementId, side A first
};

} // namespace oos::input
