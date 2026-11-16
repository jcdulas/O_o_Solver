#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/elements/mixedHybrid/core/JacobiEigen.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/solid/SolidData.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeGeometry.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeShape.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

// The elements and fields shared by the checks of the solid kernel (SolidKernelChecks.cpp, SolidLoadChecks.cpp):
// distorted and straight elements of every shape, the grid numbers and the material of the reference's tests, the
// parameters of rigid motions and of linear displacement fields.
namespace oos::checks::solid
{

namespace mh = oos::elements::mixedHybrid;
using GridId = oos::base::EntityId;

inline constexpr std::array<mh::VolumeShape, 4> SHAPES{mh::VolumeShape::Tetrahedron, mh::VolumeShape::Pyramid,
                                                       mh::VolumeShape::Pentahedron, mh::VolumeShape::Hexahedron};

// The steel of the reference's tests.
inline mh::SolidMaterial steel()
{
    return mh::SolidMaterial::isotropic(2.0e11, 0.3, 7800.0, 1.2e-5);
}

// The grid numbers of the reference's tests, taken in the order of the corners of a shape.
inline std::vector<GridId> numbersOf(mh::VolumeShape shape)
{
    constexpr std::array<GridId, 8> NUMBERS{7, 3, 9, 5, 12, 1, 4, 10};
    return {NUMBERS.begin(), NUMBERS.begin() + static_cast<std::ptrdiff_t>(mh::shapeCornerCount(shape))};
}

// The reference element moved and sheared, the corners moved apart.
// Reference: volume::tests::distorted
inline std::vector<mh::SolidPoint> distorted(mh::VolumeShape shape)
{
    std::vector<mh::SolidPoint> out;
    const std::span<const mh::SolidPoint> vertices = mh::shapeVertices(shape);
    for (std::size_t k = 0; k < vertices.size(); ++k)
    {
        const mh::SolidPoint& r = vertices[k];
        const double f = static_cast<double>(k);
        out.push_back({0.7 * r[0] + 0.06 * std::sin(1.3 * f) + 0.1 * r[1], 0.6 * r[1] + 0.05 * std::cos(2.1 * f),
                       0.5 * r[2] + 0.04 * std::sin(0.7 * f + 0.3) + 0.05 * r[0]});
    }
    return out;
}

// An affine image of the reference element: a straight element.
inline std::vector<mh::SolidPoint> affine(mh::VolumeShape shape)
{
    std::vector<mh::SolidPoint> out;
    for (const mh::SolidPoint& r : mh::shapeVertices(shape))
        out.push_back({0.7 * r[0] + 0.1 * r[1] - 0.05 * r[2] + 0.2, 0.08 * r[0] + 0.6 * r[1] + 0.03 * r[2] - 0.1,
                       0.05 * r[0] - 0.04 * r[1] + 0.5 * r[2] + 0.3});
    return out;
}

// The offset of the mid-side node of edge e from the middle of its chord (the probe of the reference uses the same).
inline mh::SolidPoint midSideOffset(std::size_t e)
{
    const double f = static_cast<double>(e);
    return {0.03 * std::sin(1.7 * f), 0.02 * std::cos(0.9 * f), 0.025 * std::sin(0.4 * f + 1.0)};
}

// A geometry, with a mid-side node on every edge when `midSide`.
inline std::optional<mh::VolumeGeometry> geometryOf(mh::VolumeShape shape, std::vector<mh::SolidPoint> corners,
                                                    std::span<const GridId> order, std::size_t p, std::size_t pf,
                                                    bool midSide)
{
    mh::SolidResult<mh::VolumeGeometry> built = mh::VolumeGeometry::straight(shape, std::move(corners), order, p, pf);
    if (!built.ok())
        return std::nullopt;
    mh::VolumeGeometry geometry = std::move(built.value());
    if (midSide)
    {
        const std::span<const mh::ShapeEdge> edges = mh::shapeEdges(shape);
        for (std::size_t e = 0; e < edges.size(); ++e)
        {
            const mh::SolidPoint& a = geometry.corners[edges[e][0]];
            const mh::SolidPoint& b = geometry.corners[edges[e][1]];
            const mh::SolidPoint offset = midSideOffset(e);
            geometry.setMidSideNode(
                e, {0.5 * (a[0] + b[0]) + offset[0], 0.5 * (a[1] + b[1]) + offset[1], 0.5 * (a[2] + b[2]) + offset[2]});
        }
    }
    return geometry;
}

inline mh::SolidOptions optionsOf(std::size_t p, std::size_t pf, std::size_t d)
{
    mh::SolidOptions options;
    options.traceDegree = p;
    options.faceDegree = pf;
    options.stressDegree = d;
    options.interiorFieldCount = 3;
    return options;
}

// The parameters of a rigid motion: mode 0 … 2 a unit translation (the corner functions are a partition of unity),
// mode 3 … 5 the rotation e_m × X, carried by every function through the coefficients of the geometry.
// Reference: volume::tests::rigid_motions_and_rank
inline mh::Vector rigidParameters(const mh::VolumeGeometry& geometry, std::size_t mode)
{
    const std::size_t functionCount = geometry.functionCount();
    mh::Vector out(3 * functionCount, 0.0);
    for (std::size_t j = 0; j < functionCount; ++j)
    {
        mh::SolidPoint u{};
        if (mode < 3)
        {
            u[mode] = j < geometry.corners.size() ? 1.0 : 0.0;
        }
        else
        {
            mh::SolidPoint axis{};
            axis[mode - 3] = 1.0;
            u = mh::cross3(
                axis, mh::SolidPoint{geometry.coordinate(j, 0), geometry.coordinate(j, 1), geometry.coordinate(j, 2)});
        }
        for (std::size_t c = 0; c < 3; ++c)
            out[3 * j + c] = u[c];
    }
    return out;
}

// The parameters of the displacement u = A X + b (A a 3 × 3 matrix): A times the coefficients of the geometry, and b
// on the corner functions.
inline mh::Vector linearParameters(const mh::VolumeGeometry& geometry, const std::array<std::array<double, 3>, 3>& a,
                                   const mh::SolidPoint& b)
{
    const std::size_t functionCount = geometry.functionCount();
    mh::Vector out(3 * functionCount, 0.0);
    for (std::size_t j = 0; j < functionCount; ++j)
        for (std::size_t c = 0; c < 3; ++c)
        {
            double value = j < geometry.corners.size() ? b[c] : 0.0;
            for (std::size_t m = 0; m < 3; ++m)
                value += a[c][m] * geometry.coordinate(j, m);
            out[3 * j + c] = value;
        }
    return out;
}

inline std::vector<double> sortedEigenvalues(const mh::Matrix& m)
{
    std::vector<double> values = mh::jacobiEigen(m).values;
    std::sort(values.begin(), values.end());
    return values;
}

inline double asymmetry(const mh::Matrix& m)
{
    return mh::maxAbs(m - m.transposed()) / mh::maxAbs(m);
}

} // namespace oos::checks::solid
