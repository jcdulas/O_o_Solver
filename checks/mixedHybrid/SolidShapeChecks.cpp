#include "CheckRunner.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeElement.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeGeometry.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeShape.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace mh = oos::elements::mixedHybrid;
using GridId = oos::base::EntityId;

namespace
{

constexpr std::array<mh::VolumeShape, 4> SHAPES{mh::VolumeShape::Tetrahedron, mh::VolumeShape::Pyramid,
                                                mh::VolumeShape::Pentahedron, mh::VolumeShape::Hexahedron};

// The grid numbers of the reference's tests, taken in the order of the corners of a shape.
std::vector<GridId> numbersOf(mh::VolumeShape shape)
{
    constexpr std::array<GridId, 8> NUMBERS{7, 3, 9, 5, 12, 1, 4, 10};
    return {NUMBERS.begin(), NUMBERS.begin() + static_cast<std::ptrdiff_t>(mh::shapeCornerCount(shape))};
}

// x = M r + b: an affine image of the reference element.
mh::SolidPoint affineImage(const mh::SolidPoint& r)
{
    return {0.7 * r[0] + 0.1 * r[1] - 0.05 * r[2] + 0.2, 0.08 * r[0] + 0.6 * r[1] + 0.03 * r[2] - 0.1,
            0.05 * r[0] - 0.04 * r[1] + 0.5 * r[2] + 0.3};
}

std::vector<mh::SolidPoint> affineCorners(mh::VolumeShape shape)
{
    std::vector<mh::SolidPoint> out;
    for (const mh::SolidPoint& vertex : mh::shapeVertices(shape))
        out.push_back(affineImage(vertex));
    return out;
}

// Points of the coordinates of a face away from the points of the rule that fits the functions.
std::vector<std::array<double, 2>> faceSamples(std::size_t cornerCount)
{
    if (cornerCount == 3)
        return {{0.13, 0.21}, {0.4, 0.35}, {0.05, 0.7}, {0.61, 0.02}};
    return {{0.13, 0.21}, {0.4, 0.85}, {0.95, 0.7}, {0.61, 0.02}};
}

// The values the functions of the element must have at the face point c of face lf: the face functions of its numbers
// put on the functions of the element (the numbering of VolumeGeometry.hpp), zero for every other function.
std::vector<double> expectedTraces(mh::VolumeShape shape, std::span<const GridId> order, std::size_t p, std::size_t pf,
                                   std::size_t lf, const std::array<double, 2>& c)
{
    const std::span<const mh::ShapeFace> faces = mh::shapeFaces(shape);
    const std::span<const std::size_t> face = faces[lf].cornerSpan();
    const std::size_t n = face.size();
    std::vector<GridId> numbers;
    for (std::size_t k : face)
        numbers.push_back(order[k]);
    const std::vector<double> values = mh::faceFunctions<double>(numbers, p, pf, c);
    const std::size_t cornerCount = mh::shapeCornerCount(shape);
    const std::size_t perEdge = mh::edgeFunctionCount(p);
    std::size_t firstFace = cornerCount + mh::shapeEdges(shape).size() * perEdge;
    for (std::size_t f = 0; f < lf; ++f)
        firstFace += mh::faceFunctionCount(pf, faces[f].cornerCount);
    std::size_t total = firstFace;
    for (std::size_t f = lf; f < faces.size(); ++f)
        total += mh::faceFunctionCount(pf, faces[f].cornerCount);
    std::vector<double> out(total, 0.0);
    for (std::size_t k = 0; k < n; ++k)
        out[face[k]] = values[k];
    for (std::size_t k = 0; k < n; ++k)
    {
        const std::size_t edge = *mh::shapeEdgeIndex(shape, face[k], face[(k + 1) % n]);
        for (std::size_t m = 0; m < perEdge; ++m)
            out[cornerCount + edge * perEdge + m] = values[n + k * perEdge + m];
    }
    for (std::size_t m = 0; m < mh::faceFunctionCount(pf, n); ++m)
        out[firstFace + m] = values[n + n * perEdge + m];
    return out;
}

double maxDifference(const std::vector<double>& a, const std::vector<double>& b)
{
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

std::string label(mh::VolumeShape shape, std::size_t p, std::size_t pf)
{
    return std::string(mh::shapeName(shape)) + " p " + std::to_string(p) + " pf " + std::to_string(pf);
}

} // namespace

// The functions of a quadrilateral face are the same whatever corner it is given from and in either direction: the
// corner functions follow their corners, the edge functions their edges, the inside is shared.
// Reference: volume::tests::face_functions_are_intrinsic
CHECK_CASE("solid: face functions are intrinsic")
{
    const std::array<GridId, 4> g{17, 4, 9, 12};
    const std::array<double, 2> c{0.31, 0.67};
    const std::vector<double> base = mh::faceFunctions<double>(g, 4, 4, c);
    constexpr std::array<std::array<double, 2>, 4> CORNER{{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}}};
    for (std::size_t start = 0; start < 4; ++start)
    {
        for (bool reversed : {false, true})
        {
            std::array<std::size_t, 4> idx{};
            std::array<GridId, 4> h{};
            for (std::size_t k = 0; k < 4; ++k)
            {
                idx[k] = reversed ? (start + 4 - k) % 4 : (start + k) % 4;
                h[k] = g[idx[k]];
            }
            // the same point in the new frame: corner idx[0] is the origin, idx[1] along s, idx[3] along t
            const std::array<double, 2> p0 = CORNER[idx[0]];
            const std::array<double, 2> es{CORNER[idx[1]][0] - p0[0], CORNER[idx[1]][1] - p0[1]};
            const std::array<double, 2> et{CORNER[idx[3]][0] - p0[0], CORNER[idx[3]][1] - p0[1]};
            const std::array<double, 2> d{c[0] - p0[0], c[1] - p0[1]};
            const std::array<double, 2> cn{d[0] * es[0] + d[1] * es[1], d[0] * et[0] + d[1] * et[1]};
            const std::vector<double> other = mh::faceFunctions<double>(h, 4, 4, cn);
            const std::string where = "start " + std::to_string(start) + (reversed ? " reversed" : "");
            for (std::size_t k = 0; k < 4; ++k)
                runner.expect(std::fabs(other[k] - base[idx[k]]) < 1e-14, where + ": corner " + std::to_string(k));
            for (std::size_t m = 0; m < 9; ++m)
                runner.expect(std::fabs(other[16 + m] - base[16 + m]) < 1e-14, where + ": inside " + std::to_string(m));
            for (std::size_t k = 0; k < 4; ++k)
            {
                // edge k of the new order is the edge (idx[k], idx[k + 1]) of the base
                const std::size_t a = idx[k];
                const std::size_t b = idx[(k + 1) % 4];
                const std::size_t kb = (a + 1) % 4 == b ? a : b;
                for (std::size_t m = 0; m < 3; ++m)
                    runner.expect(std::fabs(other[4 + 3 * k + m] - base[4 + 3 * kb + m]) < 1e-14,
                                  where + ": edge " + std::to_string(k) + " function " + std::to_string(m));
            }
        }
    }
}

// The same for a triangular face given from any corner in either direction (the point keeps its barycentric
// coordinates, carried by the corners).
CHECK_CASE("solid: triangle face functions are intrinsic")
{
    const std::array<GridId, 3> g{17, 4, 9};
    const std::array<double, 3> l{1.0 - 0.23 - 0.41, 0.23, 0.41};
    constexpr std::size_t P = 5;
    const std::vector<double> base = mh::faceFunctions<double>(g, P, P, std::array<double, 2>{l[1], l[2]});
    const std::size_t perEdge = mh::edgeFunctionCount(P);
    const std::size_t inside = mh::faceFunctionCount(P, 3);
    for (std::size_t start = 0; start < 3; ++start)
    {
        for (bool reversed : {false, true})
        {
            std::array<std::size_t, 3> idx{};
            std::array<GridId, 3> h{};
            for (std::size_t k = 0; k < 3; ++k)
            {
                idx[k] = reversed ? (start + 3 - k) % 3 : (start + k) % 3;
                h[k] = g[idx[k]];
            }
            const std::vector<double> other =
                mh::faceFunctions<double>(h, P, P, std::array<double, 2>{l[idx[1]], l[idx[2]]});
            const std::string where = "start " + std::to_string(start) + (reversed ? " reversed" : "");
            for (std::size_t k = 0; k < 3; ++k)
                runner.expect(std::fabs(other[k] - base[idx[k]]) < 1e-14, where + ": corner " + std::to_string(k));
            for (std::size_t k = 0; k < 3; ++k)
            {
                const std::size_t a = idx[k];
                const std::size_t b = idx[(k + 1) % 3];
                const std::size_t kb = (a + 1) % 3 == b ? a : b;
                for (std::size_t m = 0; m < perEdge; ++m)
                    runner.expect(std::fabs(other[3 + perEdge * k + m] - base[3 + perEdge * kb + m]) < 1e-14,
                                  where + ": edge " + std::to_string(k) + " function " + std::to_string(m));
            }
            for (std::size_t m = 0; m < inside; ++m)
                runner.expect(std::fabs(other[3 + 3 * perEdge + m] - base[3 + 3 * perEdge + m]) < 1e-14,
                              where + ": inside " + std::to_string(m));
        }
    }
}

// The tables of the shapes: corner counts, closed surfaces (every edge on two faces, V − E + F = 2), the edges of the
// faces among the edges of the shape, the mid-side node order of the cards.
CHECK_CASE("solid: shape tables")
{
    for (mh::VolumeShape shape : SHAPES)
    {
        const std::string name = mh::shapeName(shape);
        const std::size_t corners = mh::shapeCornerCount(shape);
        runner.expect(mh::volumeShapeOfCornerCount(corners) == shape, name + ": shape of its corner count");
        const std::span<const mh::ShapeEdge> edges = mh::shapeEdges(shape);
        const std::span<const mh::ShapeFace> faces = mh::shapeFaces(shape);
        runner.expect(corners + faces.size() == edges.size() + 2, name + ": Euler's formula");
        std::vector<int> facesOfEdge(edges.size(), 0);
        for (const mh::ShapeFace& face : faces)
        {
            for (std::size_t k = 0; k < face.cornerCount; ++k)
            {
                const auto edge = mh::shapeEdgeIndex(shape, face.corners[k], face.corners[(k + 1) % face.cornerCount]);
                runner.expect(edge.has_value(), name + ": an edge of a face is an edge of the shape");
                if (edge)
                    ++facesOfEdge[*edge];
            }
        }
        for (std::size_t e = 0; e < edges.size(); ++e)
            runner.expect(facesOfEdge[e] == 2, name + ": edge " + std::to_string(e) + " on two faces");
        runner.expect(mh::volumeLevels(shape, 4).size() == mh::volumeSpaceDimension(shape, 4),
                      name + ": a level per function");
    }
    runner.expect(!mh::volumeShapeOfCornerCount(7).has_value(), "no shape of 7 corners");
    // mid-side node k of a card is on edge k: SolidHexa G9-G12 on the face G1-G4, G13-G16 on the vertical edges
    const std::span<const mh::ShapeEdge> hexa = mh::shapeEdges(mh::VolumeShape::Hexahedron);
    runner.expect(hexa[0] == mh::ShapeEdge{0, 1} && hexa[4] == mh::ShapeEdge{0, 4} && hexa[11] == mh::ShapeEdge{7, 4},
                  "hexahedron edges in the order of the mid-side nodes");
    const std::span<const mh::ShapeEdge> tetra = mh::shapeEdges(mh::VolumeShape::Tetrahedron);
    runner.expect(tetra[2] == mh::ShapeEdge{2, 0} && tetra[3] == mh::ShapeEdge{0, 3},
                  "tetrahedron edges in the order of the mid-side nodes");
    const mh::VolumeElement element = mh::VolumeElement::fromCorners(std::vector<GridId>{1, 2, 3, 4}, 0)
                                          .withMidSideNodes(std::vector<GridId>{5, mh::NO_MID_SIDE_NODE, 7, 8, 9, 10});
    runner.expect(element.shape() == mh::VolumeShape::Tetrahedron, "element shape");
    runner.expect(element.midSideNodeOf(0) == 5 && !element.midSideNodeOf(1).has_value(), "mid-side nodes");
}

// The functions of an element have the face functions as traces at any point of its faces, not only at the points
// they were fitted at, and vanish on the faces that do not have them; every shape, several degrees.
CHECK_CASE("solid: element functions have the face functions as traces")
{
    constexpr std::array<std::pair<std::size_t, std::size_t>, 5> DEGREES{{{1, 1}, {2, 2}, {3, 2}, {4, 3}, {4, 4}}};
    for (mh::VolumeShape shape : SHAPES)
    {
        const std::vector<GridId> order = numbersOf(shape);
        for (const auto& [p, pf] : DEGREES)
        {
            const mh::SolidResult<mh::VolumeGeometry> geometry =
                mh::VolumeGeometry::straight(shape, affineCorners(shape), order, p, pf);
            runner.expect(geometry.ok(), label(shape, p, pf) + ": built (" + geometry.refusal() + ")");
            if (!geometry.ok())
                continue;
            double worst = 0.0;
            const std::span<const mh::ShapeFace> faces = mh::shapeFaces(shape);
            for (std::size_t lf = 0; lf < faces.size(); ++lf)
            {
                for (const std::array<double, 2>& c : faceSamples(faces[lf].cornerCount))
                {
                    const mh::SolidPoint r = mh::facePoint(shape, faces[lf].cornerSpan(), c);
                    const std::vector<double> values = geometry.value().functions(r);
                    worst = std::max(worst, maxDifference(values, expectedTraces(shape, order, p, pf, lf, c)));
                }
            }
            runner.expect(worst < 1e-11, label(shape, p, pf) + ": traces off by " + std::to_string(worst));
        }
    }
}

// A straight element is the map of its corners: an affine image of the reference element is affine (its functions of
// level 1 reproduce the affine fields), and a distorted hexahedron is its trilinear map.
CHECK_CASE("solid: a straight element is the map of its corners")
{
    const std::array<mh::SolidPoint, 3> samples{{{-0.3, -0.2, -0.4}, {-0.1, -0.5, 0.2}, {-0.6, -0.1, -0.15}}};
    for (mh::VolumeShape shape : SHAPES)
    {
        const mh::SolidResult<mh::VolumeGeometry> geometry =
            mh::VolumeGeometry::straight(shape, affineCorners(shape), numbersOf(shape), 4, 4);
        runner.expect(geometry.ok(), std::string(mh::shapeName(shape)) + ": built");
        if (!geometry.ok())
            continue;
        double worst = 0.0;
        for (const mh::SolidPoint& r : samples)
        {
            const mh::SolidPoint x = geometry.value().at(r);
            const mh::SolidPoint expected = affineImage(r);
            for (std::size_t m = 0; m < 3; ++m)
                worst = std::max(worst, std::fabs(x[m] - expected[m]));
        }
        runner.expect(worst < 1e-13,
                      std::string(mh::shapeName(shape)) + ": affine map off by " + std::to_string(worst));
    }
    // a distorted hexahedron: X(r) = Σ_k (1 ± x)(1 ± y)(1 ± z) / 8 X_k
    const std::vector<mh::SolidPoint> corners{{0.0, 0.0, 0.0},  {1.1, 0.1, -0.1}, {1.2, 0.9, 0.05}, {-0.1, 1.0, 0.1},
                                              {0.1, 0.05, 0.8}, {0.9, -0.1, 1.1}, {1.0, 1.1, 0.9},  {0.05, 0.95, 1.2}};
    const mh::SolidResult<mh::VolumeGeometry> hexa = mh::VolumeGeometry::straight(
        mh::VolumeShape::Hexahedron, corners, std::vector<GridId>{21, 2, 40, 5, 13, 8, 30, 11}, 4, 4);
    runner.expect(hexa.ok(), "distorted hexahedron built");
    if (!hexa.ok())
        return;
    const std::span<const mh::SolidPoint> vertices = mh::shapeVertices(mh::VolumeShape::Hexahedron);
    double worst = 0.0;
    for (const mh::SolidPoint& r : samples)
    {
        mh::SolidPoint expected{};
        for (std::size_t k = 0; k < 8; ++k)
        {
            const double w =
                (1.0 + vertices[k][0] * r[0]) * (1.0 + vertices[k][1] * r[1]) * (1.0 + vertices[k][2] * r[2]) / 8.0;
            for (std::size_t m = 0; m < 3; ++m)
                expected[m] += w * corners[k][m];
        }
        const mh::SolidPoint x = hexa.value().at(r);
        for (std::size_t m = 0; m < 3; ++m)
            worst = std::max(worst, std::fabs(x[m] - expected[m]));
    }
    runner.expect(worst < 1e-13, "trilinear map off by " + std::to_string(worst));
}

// A mid-side node makes its edge the parabola through it: the middle of every edge of the element is at its node.
CHECK_CASE("solid: mid-side nodes are on their edges")
{
    for (mh::VolumeShape shape : SHAPES)
    {
        mh::SolidResult<mh::VolumeGeometry> built =
            mh::VolumeGeometry::straight(shape, affineCorners(shape), numbersOf(shape), 2, 2);
        runner.expect(built.ok(), std::string(mh::shapeName(shape)) + ": built");
        if (!built.ok())
            continue;
        mh::VolumeGeometry& geometry = built.value();
        const std::span<const mh::ShapeEdge> edges = mh::shapeEdges(shape);
        const std::span<const mh::SolidPoint> vertices = mh::shapeVertices(shape);
        std::vector<mh::SolidPoint> nodes;
        for (std::size_t e = 0; e < edges.size(); ++e)
        {
            const mh::SolidPoint& a = geometry.corners[edges[e][0]];
            const mh::SolidPoint& b = geometry.corners[edges[e][1]];
            const double f = static_cast<double>(e);
            nodes.push_back({0.5 * (a[0] + b[0]) + 0.03 * std::sin(1.7 * f), 0.5 * (a[1] + b[1]) + 0.02 * std::cos(f),
                             0.5 * (a[2] + b[2]) - 0.025 * std::sin(0.4 * f + 1.0)});
            geometry.setMidSideNode(e, nodes.back());
        }
        runner.expect(geometry.curved(), std::string(mh::shapeName(shape)) + ": curved");
        double worst = 0.0;
        for (std::size_t e = 0; e < edges.size(); ++e)
        {
            const mh::SolidPoint& ra = vertices[edges[e][0]];
            const mh::SolidPoint& rb = vertices[edges[e][1]];
            const mh::SolidPoint middle{0.5 * (ra[0] + rb[0]), 0.5 * (ra[1] + rb[1]), 0.5 * (ra[2] + rb[2])};
            const mh::SolidPoint x = geometry.at(middle);
            for (std::size_t m = 0; m < 3; ++m)
                worst = std::max(worst, std::fabs(x[m] - nodes[e][m]));
        }
        runner.expect(worst < 1e-13,
                      std::string(mh::shapeName(shape)) + ": edge middles off by " + std::to_string(worst));
    }
}
