#include "oos/assembly/constraints/GlueSearch.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>
#include <variant>

namespace oos::assembly
{

namespace
{

// The shape functions of a valid face (its point count and corners checked by the caller).
FaceShapeValues shapeOf(const GlueFace& face, FaceCoordinates at)
{
    return std::get<FaceShapeValues>(faceShapeFunctions(face.grids(), at.xi, at.eta));
}

// The local coordinates of the centre of a face: the start of the projection.
FaceCoordinates centreOf(const GlueFace& face)
{
    constexpr double TRIANGLE_CENTRE = 1.0 / 3.0;
    return face.cornerCount() == 3 ? FaceCoordinates{TRIANGLE_CENTRE, TRIANGLE_CENTRE} : FaceCoordinates{0.0, 0.0};
}

// The local coordinates of the foot of `target` on the surface of a face, x(ξ, η) = Σ N_i(ξ, η)·x_i: Gauss-Newton on
// |target − x(ξ, η)|², the tangents by central differences of the map (exact for the bilinear and linear maps, O(h²)
// for the quadratic ones, which changes the iterations only, not the point they converge to). Nothing when the
// iterations leave the finite numbers (a degenerate face).
std::optional<FaceCoordinates> footOnFace(const GlueFace& face, base::Vec3 target)
{
    constexpr int MAX_ITERATIONS = 30;
    constexpr double STEP = 1.0e-4;            // of the local coordinates, which span 2 (or 1 on a triangle)
    constexpr double CONVERGED_STEP = 1.0e-13; // of the local coordinates: round-off
    FaceCoordinates c = centreOf(face);
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration)
    {
        const base::Vec3 r = target - face.pointAt(c);
        const base::Vec3 a = (face.pointAt({c.xi + STEP, c.eta}) - face.pointAt({c.xi - STEP, c.eta})) / (2.0 * STEP);
        const base::Vec3 b = (face.pointAt({c.xi, c.eta + STEP}) - face.pointAt({c.xi, c.eta - STEP})) / (2.0 * STEP);
        const double aa = base::dot(a, a);
        const double ab = base::dot(a, b);
        const double bb = base::dot(b, b);
        const double determinant = aa * bb - ab * ab;
        if (!(determinant > 0.0))
            return std::nullopt;
        const double dXi = (bb * base::dot(a, r) - ab * base::dot(b, r)) / determinant;
        const double dEta = (aa * base::dot(b, r) - ab * base::dot(a, r)) / determinant;
        c.xi += dXi;
        c.eta += dEta;
        if (!std::isfinite(c.xi) || !std::isfinite(c.eta))
            return std::nullopt;
        if (std::abs(dXi) + std::abs(dEta) < CONVERGED_STEP)
            break;
    }
    return c;
}

} // namespace

base::Vec3 GlueFace::pointAt(FaceCoordinates at) const
{
    const FaceShapeValues shape = shapeOf(*this, at);
    base::Vec3 x{};
    for (std::size_t k = 0; k < shape.count; ++k)
        x = x + shape.values[k] * positions[k];
    return x;
}

double typicalElementSize(std::span<const GlueFace> faces)
{
    double total = 0.0;
    std::size_t edgeCount = 0;
    for (const GlueFace& face : faces)
    {
        const std::size_t corners = face.cornerCount();
        for (std::size_t k = 0; k < corners; ++k)
            total += base::norm(face.positions[(k + 1) % corners] - face.positions[k]);
        edgeCount += corners;
    }
    return edgeCount > 0 ? total / static_cast<double>(edgeCount) : 0.0;
}

GlueSearch::GlueSearch(std::span<const GlueFace> faces, double searchRadius, base::MemoryBudget& budget)
    : m_faces(faces)
{
    if (faces.empty())
        return;
    // The box of each face (its grid points; a quadratic face bulges no farther than its mid-side points place it, to
    // the approximation of the search), grown by the search radius. The cells are as large as the largest grown box, so
    // that a box spans at most two cells per axis.
    const auto boxOf = [&](const GlueFace& face)
    {
        base::Vec3 low = face.positions[0];
        base::Vec3 high = face.positions[0];
        for (std::size_t k = 1; k < static_cast<std::size_t>(face.pointCount); ++k)
        {
            if (face.gridIds[k] == 0)
                continue;
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                low[axis] = std::min(low[axis], face.positions[k][axis]);
                high[axis] = std::max(high[axis], face.positions[k][axis]);
            }
        }
        const base::Vec3 grow{searchRadius, searchRadius, searchRadius};
        return std::make_pair(low - grow, high + grow);
    };
    base::Vec3 low = boxOf(faces[0]).first;
    base::Vec3 high = boxOf(faces[0]).second;
    double largest = 0.0;
    for (const GlueFace& face : faces)
    {
        const auto [faceLow, faceHigh] = boxOf(face);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            low[axis] = std::min(low[axis], faceLow[axis]);
            high[axis] = std::max(high[axis], faceHigh[axis]);
            largest = std::max(largest, faceHigh[axis] - faceLow[axis]);
        }
    }
    m_origin = low;
    m_cellSize = largest > 0.0 ? largest : 1.0;
    for (std::size_t axis = 0; axis < 3; ++axis)
        m_cellCounts[axis] = static_cast<std::int64_t>(std::floor((high[axis] - low[axis]) / m_cellSize)) + 1;

    // Each face in every cell its grown box overlaps: counted, then written, then sorted by cell.
    const auto forEachCell = [&](const GlueFace& face, auto visit)
    {
        const auto [faceLow, faceHigh] = boxOf(face);
        const std::array<std::int64_t, 3> first = cellOf(faceLow);
        const std::array<std::int64_t, 3> last = cellOf(faceHigh);
        for (std::int64_t i = first[0]; i <= last[0]; ++i)
            for (std::int64_t j = first[1]; j <= last[1]; ++j)
                for (std::int64_t k = first[2]; k <= last[2]; ++k)
                    visit(keyOf({i, j, k}));
    };
    std::size_t entryCount = 0;
    for (const GlueFace& face : faces)
        forEachCell(face, [&](std::int64_t) { ++entryCount; });
    m_entries = base::BudgetedBuffer<CellEntry>(budget, entryCount, "glue: search cells");
    std::size_t next = 0;
    for (std::size_t f = 0; f < faces.size(); ++f)
        forEachCell(faces[f], [&](std::int64_t cell) { m_entries[next++] = {cell, static_cast<std::int32_t>(f)}; });
    std::sort(m_entries.data(), m_entries.data() + entryCount, [](const CellEntry& a, const CellEntry& b)
              { return a.cell != b.cell ? a.cell < b.cell : a.face < b.face; });
}

std::array<std::int64_t, 3> GlueSearch::cellOf(base::Vec3 point) const
{
    std::array<std::int64_t, 3> index{};
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        // −1 and the cell count stand for "before the first cell" and "after the last" (a point far outside).
        const double at = std::floor((point[axis] - m_origin[axis]) / m_cellSize);
        index[axis] = static_cast<std::int64_t>(std::clamp(at, -1.0, static_cast<double>(m_cellCounts[axis])));
    }
    return index;
}

std::int64_t GlueSearch::keyOf(const std::array<std::int64_t, 3>& index) const noexcept
{
    return (index[0] * m_cellCounts[1] + index[1]) * m_cellCounts[2] + index[2];
}

GlueProjection GlueSearch::project(base::Vec3 point, const GlueTolerances& tolerances) const
{
    GlueProjection best;
    if (m_faces.empty())
        return best;
    const std::array<std::int64_t, 3> index = cellOf(point);
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (index[axis] < 0 || index[axis] >= m_cellCounts[axis])
            return best; // outside every grown box: no face is near
    const std::int64_t cell = keyOf(index);
    const auto first = std::lower_bound(m_entries.data(), m_entries.data() + m_entries.size(), cell,
                                        [](const CellEntry& entry, std::int64_t key) { return entry.cell < key; });
    double bestDistance = std::numeric_limits<double>::infinity();
    for (const CellEntry* entry = first; entry != m_entries.data() + m_entries.size() && entry->cell == cell; ++entry)
    {
        const GlueFace& face = m_faces[static_cast<std::size_t>(entry->face)];
        const std::optional<FaceCoordinates> foot = footOnFace(face, point);
        if (!foot)
            continue;
        const FaceCoordinates onFace = clampToFace(static_cast<std::size_t>(face.pointCount), *foot);
        const base::Vec3 footPoint = face.pointAt(*foot);
        const base::Vec3 tiedPoint = face.pointAt(onFace);
        const double distance = base::norm(point - tiedPoint);
        best.nearestDistance = std::min(best.nearestDistance, distance);
        const double normalDistance = base::norm(point - footPoint);
        const double inPlaneDistance = base::norm(footPoint - tiedPoint);
        if (normalDistance > tolerances.position || inPlaneDistance > tolerances.inPlane || !(distance < bestDistance))
            continue;
        bestDistance = distance;
        best.face = entry->face;
        best.coordinates = onFace;
        best.shape = shapeOf(face, onFace);
        best.normalDistance = normalDistance;
        best.inPlaneDistance = inPlaneDistance;
    }
    return best;
}

} // namespace oos::assembly
