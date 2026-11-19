#pragma once

#include "oos/base/arrow/ImportedTable.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/input/tables/ElementBatch.hpp"
#include "oos/input/tables/NodeTable.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

// The footprint of a fastener on the shells it pierces: the square that stands for its cross-section, and the points
// of that square on the shell elements around the piercing point.
//
// A point load on a shell is singular (the displacement in the plane under a point force and the rotation under a
// point moment grow without bound as the elements resolve them), so a fastener tied to the shell at its piercing
// point alone gives a joint that depends on the mesh. The fastener is tied over its cross-section instead: a square
// of the same area, π D² / 4, of side (√π / 2) D, in the tangent plane of the shells at the piercing point (its
// normal the mean of the normals of the elements that hold the point), its first side along the first side of the
// pierced element. Two sets of points of the square are located on the shells:
// - its four corners, the auxiliary points of the fastener, at which the ClassicFormulation reads the motion of the
//   shells (elements/fasteners/PiercingInterpolation);
// - its 3 × 3 Gauss points, over which a formulation that has the rotation of the shell as a field of its own takes
//   the mean of the motion of the shell (elements::FormulationModel::shellPointMotion).
// A point is located on the pierced element or on the shell elements around it that continue its sheet, up to
// FOOTPRINT_RINGS rings of elements away: the mesh may be finer than the fastener.

namespace oos::input
{

// The side of the square of the area of a circle of diameter 1: √π / 2.
inline constexpr double FOOTPRINT_SIDE_PER_DIAMETER = 0.886226925452758;
// How many rings of shell elements around the pierced one the points of a footprint are looked for in (one ring: the
// elements that hold a corner of the pierced one; the next: those that hold a corner of an element of the ring).
inline constexpr int FOOTPRINT_RINGS = 2;
// How many times the square of the corners is halved when one of them falls off the shells (a fastener on a free
// edge, a mesh finer than FOOTPRINT_RINGS rings can cover): after that the fastener has no corners, and is tied to
// the grid points of the pierced element.
inline constexpr int FOOTPRINT_HALVINGS = 3;
// A point this far beyond a side of an element, in its parent coordinates, is on that side: every element that
// holds it so has its share of it.
inline constexpr double FOOTPRINT_ON_SIDE = 1.0e-6;
// A point no element holds is taken on the nearest element when it is within this distance beyond a side of it, in
// its parent coordinates (the tolerance of the reader on the piercing points; the sheets of a curved skin, flat
// between their grid points, leave such gaps between the nearest points of neighbours).
inline constexpr double FOOTPRINT_SIDE_TOLERANCE = 0.02;
// A shell element continues the sheet of the pierced element when its normal is within this angle, in direction, of
// the normal of the pierced element: a web or a rib that meets the sheet along the line the fastener is on does
// not.
inline constexpr double PIERCING_PATCH_ANGLE_DEGREES = 20.0;

// One point of a footprint: the shell element it is on, its weight in its set (the weights of a set sum to 1 when
// the shells hold every point of it), the grid points of the element and the shape functions of its card there.
struct FootprintSample
{
    std::int32_t elementId = 0;
    double weight = 0.0;
    std::span<const std::int32_t> grids;
    std::span<const double> shapeValues;
};

// The samples of one set (the corners, or the Gauss points) of the footprints of the rows of a table, row after row.
//
// Invariant: endRow() was called once per row; a row may have no sample (no footprint). Ownership: owns its arrays
// (std::vector: bounded by the fastener sides of the model, at most a few shells each).
class FootprintSamples
{
public:
    FootprintSamples() = default;
    // `rows` rows without a sample.
    explicit FootprintSamples(std::int64_t rows) : m_rowOffsets(static_cast<std::size_t>(rows) + 1, 0) {}

    // add() appends a sample to the row being filled, endRow() ends that row.
    void add(std::int32_t elementId, double weight, std::span<const std::int32_t> grids,
             std::span<const double> shapeValues);
    void endRow() { m_rowOffsets.push_back(static_cast<std::int64_t>(m_elementIds.size())); }

    [[nodiscard]] std::int64_t rowCount() const noexcept { return static_cast<std::int64_t>(m_rowOffsets.size()) - 1; }
    [[nodiscard]] std::int64_t count(std::int64_t row) const
    {
        return m_rowOffsets[static_cast<std::size_t>(row) + 1] - m_rowOffsets[static_cast<std::size_t>(row)];
    }
    // Sample `k` (0 to count(row) − 1) of a row.
    [[nodiscard]] FootprintSample at(std::int64_t row, std::int64_t k) const;

private:
    std::vector<std::int64_t> m_rowOffsets{0}; // rows + 1, into m_elementIds and m_weights
    std::vector<std::int32_t> m_elementIds;
    std::vector<double> m_weights;
    std::vector<std::int64_t> m_gridOffsets{0}; // samples + 1, into m_grids and m_shapeValues
    std::vector<std::int32_t> m_grids;
    std::vector<double> m_shapeValues;
};

// The cards of the shell elements a fastener pierces.
inline constexpr std::array<std::string_view, 6> PIERCED_SHELL_CARDS{
    "ShellQuad4", "ShellTria3", "ShellQuad8", "ShellTria6", "ShellQuad4Drilling", "ShellTria3Drilling"};

// A shell element of the model: its ID and its grid points in the order of its card (the corners first; 0 for an
// absent mid-side point).
struct ShellRef
{
    std::int32_t id = 0;
    std::span<const std::int32_t> grids;
};

// The shell elements of a model (PIERCED_SHELL_CARDS) by ID and around each grid point, with the positions of the
// grid points: what the patches and the footprints of the piercing points are built from.
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns its indexes (16 bytes per shell and per grid
// point of a shell: built for the time of FastenerPiercingTable::buildPatches); `nodes` and `batches` must outlive it.
class ShellMeshIndex
{
public:
    ShellMeshIndex(const NodeTable& nodes, std::span<const ElementBatch> batches);

    // The position of a grid point in the basic system, or nothing when the model does not have it.
    [[nodiscard]] std::optional<base::Vec3> positionOf(std::int32_t gridId) const;
    // The shell element of this ID, or nothing.
    [[nodiscard]] std::optional<ShellRef> find(std::int32_t elementId) const;
    // Calls visit(const ShellRef&) for every shell element that holds the grid point.
    template <typename Visit>
    void forEachShellAt(std::int32_t gridId, Visit visit) const
    {
        const auto [first, last] = shellsAt(gridId);
        for (std::size_t k = first; k < last; ++k)
            visit(refOf(m_shellsAtGrid[k].batch, m_shellsAtGrid[k].row));
    }
    // The unit normal of the shell of grid points `grids`: along the cross product of two sides of a triangle, of the
    // diagonals of a quadrilateral. Nothing when a corner is not a grid point of the model or the corners are on a
    // line.
    [[nodiscard]] std::optional<base::Vec3> normalOf(std::span<const std::int32_t> grids) const;
    // The corners of a shell of `gridCount` grid points: three for 3 or 6 grid points, else four.
    [[nodiscard]] static constexpr std::size_t cornerCount(std::size_t gridCount) noexcept
    {
        return gridCount == 3 || gridCount == 6 ? 3 : 4;
    }

private:
    struct ShellAtGrid
    {
        std::int32_t gridId = 0;
        std::uint32_t batch = 0;
        std::int64_t row = 0;
    };
    struct ShellOfId
    {
        std::int32_t id = 0;
        std::uint32_t batch = 0;
        std::int64_t row = 0;
    };

    [[nodiscard]] ShellRef refOf(std::uint32_t batch, std::int64_t row) const;
    [[nodiscard]] std::pair<std::size_t, std::size_t> shellsAt(std::int32_t gridId) const;

    std::span<const ElementBatch> m_batches;
    base::FixedListView<double> m_positions;
    std::vector<std::pair<std::int32_t, std::int64_t>> m_nodeRows; // (grid point ID, row), sorted
    std::vector<ShellAtGrid> m_shellsAtGrid;                      // sorted by grid point
    std::vector<ShellOfId> m_shellsById;                          // sorted by ID
};

// Appends one row to `corners` and one to `mean`: the footprint of the fastener of diameter `diameter` that pierces
// the shell element `piercedElementId` at the point where the shape functions of its card are `shapeValues`.
// `corners` gets the four corners of the square, each on one element with the weight ¼, or nothing: when
// `hasCorners` is false (a fastener tied to the pierced element alone by its card), or when a corner falls off the
// shells with the square halved FOOTPRINT_HALVINGS times. `mean` gets the 3 × 3 Gauss points of the square with
// their weights, those of them the shells hold; a point on a side shared by several elements is on each of them,
// with an equal share of its weight. Both rows are empty without a diameter above 0, when the model does not have
// the element or one of its grid points, or when its corners are on a line.
void appendFootprint(const ShellMeshIndex& mesh, std::int32_t piercedElementId, std::span<const double> shapeValues,
                     double diameter, bool hasCorners, FootprintSamples& corners, FootprintSamples& mean);

} // namespace oos::input
