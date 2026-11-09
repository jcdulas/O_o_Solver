#pragma once

#include "oos/assembly/constraints/FaceShapeFunctions.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <span>

namespace oos::assembly
{

// The largest number of grid points of a face: a quadrilateral with its four mid-side points.
inline constexpr std::size_t MAX_GLUE_FACE_POINTS = 8;

// One face of the primary (master) surface of a tie: its grid points in the element's order (the corners, then the
// mid-side points of its edges, 0 for an absent one, as FaceShapeFunctions reads them) and their positions in the basic
// system (zero for an absent mid-side point). Trivially copyable: kept in budgeted arrays.
struct GlueFace
{
    std::array<std::int32_t, MAX_GLUE_FACE_POINTS> gridIds{};
    std::array<base::Vec3, MAX_GLUE_FACE_POINTS> positions{};
    std::int32_t pointCount = 0; // 3, 4, 6 or 8

    [[nodiscard]] std::span<const std::int32_t> grids() const noexcept
    {
        return {gridIds.data(), static_cast<std::size_t>(pointCount)};
    }
    [[nodiscard]] std::size_t cornerCount() const noexcept { return pointCount == 3 || pointCount == 6 ? 3 : 4; }
    // The point of the face at local coordinates (ξ, η): Σ N_i(ξ, η)·x_i. Precondition: a valid face.
    [[nodiscard]] base::Vec3 pointAt(FaceCoordinates at) const;
};

// The two distances of the tie of a secondary grid point (CalculiX *TIE, see GlueSearch).
struct GlueTolerances
{
    double position = 0.0; // the largest orthogonal distance from the point to the face
    double inPlane = 0.0;  // the largest distance, parallel to the face, of the projection outside the face
};

// Where a secondary grid point is tied: the face, the local coordinates of the tied point on it and the values of the
// face's shape functions there, and the distances.
struct GlueProjection
{
    std::int32_t face = -1; // index of the face; −1: no face within the tolerances
    FaceCoordinates coordinates;
    FaceShapeValues shape;
    double normalDistance = 0.0;   // from the point to its foot on the face's surface
    double inPlaneDistance = 0.0;  // from the foot to the face (0 for a foot inside the face)
    // The distance from the point to the nearest point of the nearest face looked at, tied or not (infinity when no
    // face was near enough to be looked at): how far a point left untied is.
    double nearestDistance = std::numeric_limits<double>::infinity();
};

// The mean length of the corner-to-corner edges of the faces: the "typical element size" of the tie's default
// tolerance (GlueSearch). 0 for no face.
[[nodiscard]] double typicalElementSize(std::span<const GlueFace> faces);

// The node-to-surface search of the tie of the open-source solver CalculiX (G. Dhondt, CalculiX CrunchiX User's Manual,
// version 2.21, 2023, section 7.132 *TIE, "tied contact"): a secondary (slave) grid point is tied to a face of the
// primary (master) surface when its orthogonal distance to the face is at most the position tolerance and its
// projection lies within the face or, measured parallel to the face, no farther from it than the default tolerance
// (2.5 % of the typical element size). Of the faces that qualify, the one whose nearest point is nearest the grid point
// is taken (the lowest face index between equal distances: the result does not depend on the order of the cells).
//
// The projection: the foot of the point on the surface of the face, x(ξ, η) = Σ N_i(ξ, η)·x_i, by Gauss-Newton on the
// squared distance (exact in one step on a flat parallelogram; a few steps on a warped or quadratic face), the normal
// distance |x_s − x(ξ, η)|; then the local coordinates brought back onto the face (clampToFace), whose point is the
// tied point, the in-plane distance its distance to the foot. The weights of the tie are the shape functions at the
// tied point: within the face, so that no weight is extrapolated. (CalculiX triangulates the master surface to find
// the face; this search looks at the faces themselves through a grid of cells: the faces near a point are those
// registered in its cell.)
//
// Thread-safety: immutable once built; project() may run from several threads. Ownership: copies the faces' cells
// (budgeted), views the faces, which must outlive it.
class GlueSearch
{
public:
    // `searchRadius`: the largest distance from a point to the nearest point of a face that may qualify (the position
    // tolerance plus the in-plane tolerance). Throws MemoryBudgetError.
    GlueSearch(std::span<const GlueFace> faces, double searchRadius, base::MemoryBudget& budget);

    [[nodiscard]] GlueProjection project(base::Vec3 point, const GlueTolerances& tolerances) const;

private:
    // One face registered in one cell of the grid (cell = (ix·ny + iy)·nz + iz).
    struct CellEntry
    {
        std::int64_t cell = 0;
        std::int32_t face = 0;
    };

    [[nodiscard]] std::array<std::int64_t, 3> cellOf(base::Vec3 point) const;
    [[nodiscard]] std::int64_t keyOf(const std::array<std::int64_t, 3>& index) const noexcept;

    std::span<const GlueFace> m_faces;
    base::BudgetedBuffer<CellEntry> m_entries; // sorted by cell, then face
    base::Vec3 m_origin{};
    double m_cellSize = 1.0;
    std::array<std::int64_t, 3> m_cellCounts{1, 1, 1};
};

} // namespace oos::assembly
