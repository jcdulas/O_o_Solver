#include "oos/elements/mixedHybrid/solid/VolumeGeometry.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/JacobiEigen.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace oos::elements::mixedHybrid
{

namespace
{

// The traces of the space are the face spaces, exactly: the functions of an element reproduce the face functions to
// round-off, and an element whose functions miss them by this much or more is refused (document 35, section 3.2).
constexpr double TRACE_REPRODUCTION_TOLERANCE = 1e-8;

// The face rule of the trace matrix has p + 2 points per direction: on a face the products of two functions of degree
// p are integrated exactly by it, so that the least-squares fit at its points is that of the face space.
constexpr std::size_t TRACE_RULE_EXTRA_POINTS = 2;

// A function of a midside parabola: s (1 − s) is 1/4 at the middle of its edge, so that the coefficient 4 d puts the
// middle of the edge at the offset d.
constexpr double MID_SIDE_COEFFICIENT = 4.0;

// Where the functions of the faces start: nv + ne·ec, then the counts of the faces before each one; and the total.
struct FunctionLayout
{
    std::vector<std::size_t> firstFace;
    std::size_t count = 0;
};

FunctionLayout functionLayout(VolumeShape shape, std::size_t p, std::size_t pf)
{
    FunctionLayout layout;
    layout.count = shapeCornerCount(shape) + shapeEdges(shape).size() * edgeFunctionCount(p);
    for (const ShapeFace& face : shapeFaces(shape))
    {
        layout.firstFace.push_back(layout.count);
        layout.count += faceFunctionCount(pf, face.cornerCount);
    }
    return layout;
}

// The traces of the basis of the space (rows of A) and of the functions of the element (rows of B) at the points of
// the face rules: A C = B for the coefficients C of the functions.
struct TraceRows
{
    Matrix spaceTraces;   // A: points × dim of the space
    Matrix elementTraces; // B: points × functions of the element
};

TraceRows traceRows(VolumeShape shape, std::span<const base::EntityId> order, std::size_t p, std::size_t pf,
                    const FunctionLayout& layout)
{
    const std::span<const ShapeFace> faces = shapeFaces(shape);
    const std::size_t cornerCount = shapeCornerCount(shape);
    const std::size_t perEdge = edgeFunctionCount(p);
    std::vector<std::vector<double>> rowsA;
    std::vector<std::vector<double>> rowsB;
    for (std::size_t lf = 0; lf < faces.size(); ++lf)
    {
        const std::span<const std::size_t> face = faces[lf].cornerSpan();
        const std::size_t n = face.size();
        std::vector<base::EntityId> numbers;
        for (std::size_t k : face)
            numbers.push_back(order[k]);
        for (const SurfaceRulePoint& point : faceRule(n, p + TRACE_RULE_EXTRA_POINTS))
        {
            const SolidPoint r = facePoint(shape, face, point.point);
            rowsA.push_back(volumeSpace(shape, p, r));
            const std::vector<double> values = faceFunctions<double>(numbers, p, pf, point.point);
            std::vector<double> b(layout.count, 0.0);
            // the corners of the face
            for (std::size_t k = 0; k < n; ++k)
                b[face[k]] = values[k];
            // the edges of the face: the local edge of the element and its functions
            for (std::size_t k = 0; k < n; ++k)
            {
                const std::optional<std::size_t> edge = shapeEdgeIndex(shape, face[k], face[(k + 1) % n]);
                OOS_ASSERT(edge.has_value());
                for (std::size_t m = 0; m < perEdge; ++m)
                    b[cornerCount + *edge * perEdge + m] = values[n + k * perEdge + m];
            }
            const std::size_t inside = faceFunctionCount(pf, n);
            for (std::size_t m = 0; m < inside; ++m)
                b[layout.firstFace[lf] + m] = values[n + n * perEdge + m];
            rowsB.push_back(std::move(b));
        }
    }
    TraceRows rows{Matrix(rowsA.size(), volumeSpaceDimension(shape, p)), Matrix(rowsB.size(), layout.count)};
    for (std::size_t i = 0; i < rowsA.size(); ++i)
    {
        rows.spaceTraces.setRow(i, rowsA[i]);
        rows.elementTraces.setRow(i, rowsB[i]);
    }
    return rows;
}

// The level of each function of the element: 1 for the corners, m + 2 for function m of an edge, the degree of its
// traces for a function of a face (3 + m + n on a triangle, 2 + max(m, n) on a quadrilateral).
std::vector<std::size_t> targetLevels(VolumeShape shape, std::size_t p, std::size_t pf, const FunctionLayout& layout)
{
    const std::size_t cornerCount = shapeCornerCount(shape);
    const std::size_t perEdge = edgeFunctionCount(p);
    std::vector<std::size_t> level(layout.count, 1);
    for (std::size_t e = 0; e < shapeEdges(shape).size(); ++e)
        for (std::size_t m = 0; m < perEdge; ++m)
            level[cornerCount + e * perEdge + m] = m + 2;
    const std::span<const ShapeFace> faces = shapeFaces(shape);
    for (std::size_t lf = 0; lf < faces.size(); ++lf)
    {
        std::size_t index = layout.firstFace[lf];
        if (faces[lf].cornerCount == 3)
        {
            for (std::size_t total = 0; total + 3 <= pf; ++total)
                for (std::size_t m = 0; m <= total; ++m)
                    level[index++] = 3 + total;
        }
        else if (pf >= 2)
        {
            for (std::size_t m = 0; m + 1 < pf; ++m)
                for (std::size_t n = 0; n + 1 < pf; ++n)
                    level[index + m * (pf - 1) + n] = 2 + std::max(m, n);
        }
    }
    return level;
}

// The columns `columns` of m.
Matrix selectColumns(const Matrix& m, const std::vector<std::size_t>& columns)
{
    Matrix out(m.rows(), columns.size());
    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < columns.size(); ++j)
            out(i, j) = m(i, columns[j]);
    return out;
}

// The refusal of an element whose functions do not reproduce their traces: the worst residual and, for each function
// that misses, its index, its level and its residual.
std::string traceRefusal(VolumeShape shape, std::size_t p, std::size_t pf, const Matrix& residual,
                         const std::vector<std::size_t>& level)
{
    std::string message = std::string(shapeName(shape)) + " p " + std::to_string(p) + " pf " + std::to_string(pf) +
                          ": the functions of its faces are not traces of its space (" +
                          std::to_string(maxAbs(residual)) + "):";
    for (std::size_t j = 0; j < residual.columns(); ++j)
    {
        double worst = 0.0;
        for (std::size_t i = 0; i < residual.rows(); ++i)
            worst = std::max(worst, std::fabs(residual(i, j)));
        if (worst >= TRACE_REPRODUCTION_TOLERANCE)
            message +=
                " (" + std::to_string(j) + ", level " + std::to_string(level[j]) + ", " + std::to_string(worst) + ")";
    }
    return message;
}

// The element functions kept once built (elementFunctions), per shape, degrees and ranks of the corner numbers: a
// mesh has few orientations of its elements, and each new one costs a minimum-norm inverse (about 25 ms for a
// hexahedron of degree 4 on one thread). At most this many are kept (about 100 kB each for a hexahedron of degree 4);
// the orientations beyond are built at each call.
constexpr std::size_t ELEMENT_FUNCTION_CACHE_ENTRIES = 1024;

using FunctionKey = std::tuple<int, std::size_t, std::size_t, std::vector<std::size_t>>;

// The rank of each corner number among the numbers of the element (the comparisons that orient its edges and faces).
std::vector<std::size_t> ranksOf(std::span<const base::EntityId> order)
{
    std::vector<std::size_t> ranks(order.size(), 0);
    for (std::size_t i = 0; i < order.size(); ++i)
        for (std::size_t j = 0; j < order.size(); ++j)
            if (order[j] < order[i])
                ++ranks[i];
    return ranks;
}

SolidResult<Matrix> buildElementFunctions(VolumeShape shape, std::span<const base::EntityId> order, std::size_t p,
                                          std::size_t pf);

} // namespace

SolidResult<Matrix> elementFunctions(VolumeShape shape, std::span<const base::EntityId> order, std::size_t p,
                                     std::size_t pf)
{
    OOS_ASSERT(order.size() == shapeCornerCount(shape));
    // Shared by every thread (the function is pure: a kept result is the one a new build would give).
    static std::mutex mutex;
    static std::map<FunctionKey, Matrix> kept;
    FunctionKey key{static_cast<int>(shape), p, pf, ranksOf(order)};
    {
        const std::lock_guard<std::mutex> lock(mutex);
        const auto found = kept.find(key);
        if (found != kept.end())
            return found->second;
    }
    SolidResult<Matrix> built = buildElementFunctions(shape, order, p, pf);
    if (built.ok())
    {
        const std::lock_guard<std::mutex> lock(mutex);
        if (kept.size() < ELEMENT_FUNCTION_CACHE_ENTRIES)
            kept.emplace(std::move(key), built.value());
    }
    return built;
}

namespace
{

SolidResult<Matrix> buildElementFunctions(VolumeShape shape, std::span<const base::EntityId> order, std::size_t p,
                                          std::size_t pf)
{
    const FunctionLayout layout = functionLayout(shape, p, pf);
    const TraceRows rows = traceRows(shape, order, p, pf, layout);
    const Matrix& a = rows.spaceTraces;
    const Matrix& b = rows.elementTraces;
    // Each function is taken in the smallest space that has its traces (the space of the level of its traces), of
    // least norm there: the functions of the corners are those of level 1 (a straight element is the map of its
    // corners), and what is left inside a level belongs to the bubbles.
    const std::vector<std::size_t> levelOfBasis = volumeLevels(shape, p);
    const std::vector<std::size_t> level = targetLevels(shape, p, pf, layout);
    Matrix coefficients(a.columns(), layout.count);
    for (std::size_t lev = 1; lev <= std::max<std::size_t>(p, 1); ++lev)
    {
        std::vector<std::size_t> columns;
        for (std::size_t k = 0; k < a.columns(); ++k)
            if (levelOfBasis[k] <= lev)
                columns.push_back(k);
        std::vector<std::size_t> targets;
        for (std::size_t j = 0; j < layout.count; ++j)
            if (level[j] == lev)
                targets.push_back(j);
        if (targets.empty())
            continue;
        const Matrix inverse = minimumNormInverse(selectColumns(a, columns));
        for (std::size_t j : targets)
        {
            const Vector cj = inverse * b.column(j);
            for (std::size_t k = 0; k < columns.size(); ++k)
                coefficients(columns[k], j) = cj[k];
        }
    }
    // the traces of the space are the face spaces: the functions must have their traces
    const Matrix residual = a * coefficients - b;
    if (!(maxAbs(residual) < TRACE_REPRODUCTION_TOLERANCE))
        return SolidResult<Matrix>::refused(traceRefusal(shape, p, pf, residual, level));
    return coefficients;
}

} // namespace

SolidResult<VolumeGeometry> VolumeGeometry::straight(VolumeShape shape, std::vector<SolidPoint> cornerPositions,
                                                     std::span<const base::EntityId> order, std::size_t p,
                                                     std::size_t pf)
{
    OOS_ASSERT(cornerPositions.size() == shapeCornerCount(shape));
    SolidResult<Matrix> coefficients = elementFunctions(shape, order, p, pf);
    if (!coefficients.ok())
        return SolidResult<VolumeGeometry>::refused(coefficients.refusal());
    VolumeGeometry geometry;
    geometry.shape = shape;
    geometry.corners = std::move(cornerPositions);
    geometry.edges.assign(shapeEdges(shape).size(), std::vector<SolidPoint>(edgeFunctionCount(p), SolidPoint{}));
    for (const ShapeFace& face : shapeFaces(shape))
        geometry.faces.emplace_back(faceFunctionCount(pf, face.cornerCount), SolidPoint{});
    geometry.traceDegree = p;
    geometry.faceDegree = pf;
    geometry.functionCoefficients = std::move(coefficients.value());
    return geometry;
}

bool VolumeGeometry::curved() const
{
    const auto anyNonZero = [](const std::vector<std::vector<SolidPoint>>& lists)
    {
        for (const std::vector<SolidPoint>& list : lists)
            for (const SolidPoint& point : list)
                for (double value : point)
                    if (value != 0.0)
                        return true;
        return false;
    };
    return anyNonZero(edges) || anyNonZero(faces);
}

double VolumeGeometry::coordinate(std::size_t j, std::size_t m) const
{
    const std::size_t cornerCount = corners.size();
    if (j < cornerCount)
        return corners[j][m];
    j -= cornerCount;
    const std::size_t perEdge = edgeFunctionCount(traceDegree);
    if (j < edges.size() * perEdge)
        return edges[j / perEdge][j % perEdge][m];
    j -= edges.size() * perEdge;
    for (const std::vector<SolidPoint>& face : faces)
    {
        if (j < face.size())
            return face[j][m];
        j -= face.size();
    }
    OOS_ASSERT(false);
    return 0.0;
}

void VolumeGeometry::setMidSideNode(std::size_t edge, const SolidPoint& midSideNode)
{
    const std::size_t perEdge = edgeFunctionCount(traceDegree);
    if (perEdge == 0)
        return;
    const ShapeEdge ends = shapeEdges(shape)[edge];
    std::vector<SolidPoint>& coefficients = edges[edge];
    coefficients.assign(perEdge, SolidPoint{});
    for (std::size_t m = 0; m < 3; ++m)
    {
        const double middle = 0.5 * (corners[ends[0]][m] + corners[ends[1]][m]);
        coefficients[0][m] = MID_SIDE_COEFFICIENT * (midSideNode[m] - middle);
    }
}

} // namespace oos::elements::mixedHybrid
