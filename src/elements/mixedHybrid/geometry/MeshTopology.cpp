#include "oos/elements/mixedHybrid/geometry/MeshTopology.hpp"

#include "SurfaceReconstruction.hpp"
#include "oos/base/errors/InternalError.hpp"

#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <vector>

namespace oos::elements::mixedHybrid
{

namespace
{

// Below this component the edge is at a right angle to a held axis.
constexpr double HELD_AXIS_TOLERANCE = 1e-8;

Vector3 basisVector(int component)
{
    Vector3 e{};
    e[static_cast<std::size_t>(component)] = 1.0;
    return e;
}

double signOf(double v)
{
    return v < 0.0 ? -1.0 : 1.0;
}

// The direction, normal to the edge a − b of unit direction ev, of a translation held at both its grids: the held
// axis when one is held and the edge is across it, the direction across the edge in the plane of the two held axes
// when two are held. None otherwise.
// Reference: src/model.rs::fixed_normal_direction.
std::optional<Vector3> fixedNormalDirection(const MeshReadingInput& input, std::size_t a, std::size_t b,
                                            const Vector3& ev)
{
    std::vector<int> fixed;
    for (int c = 0; c < 3; ++c)
        if (input.heldTranslations.contains({a, c}) && input.heldTranslations.contains({b, c}))
            fixed.push_back(c);
    if (fixed.size() == 1)
    {
        if (std::abs(ev[static_cast<std::size_t>(fixed[0])]) < HELD_AXIS_TOLERANCE)
            return basisVector(fixed[0]);
        return std::nullopt;
    }
    if (fixed.size() == 2)
    {
        const std::size_t c0 = static_cast<std::size_t>(fixed[0]);
        const std::size_t c1 = static_cast<std::size_t>(fixed[1]);
        const std::array<double, 2> se{ev[c0], ev[c1]};
        if (std::sqrt(se[0] * se[0] + se[1] * se[1]) > HELD_AXIS_TOLERANCE)
        {
            Vector3 d{};
            d[c0] = -se[1];
            d[c1] = se[0];
            return unit3(d);
        }
    }
    return std::nullopt;
}

// The common frame (e, b, n) of each edge: e from the lower to the higher grid; n the mean of the normals of its
// elements (each turned to the side of the first), less its part along e; b = n × e. On an external edge whose two
// grids hold a translation across it, the frame turns about e to put b or n (the nearer) along the held direction. On
// the edge of a line element alone, n is the orientation vector of the element across the edge (any direction across
// it for a rod, which has none).
// Reference: src/model.rs::Topology::edge_frames.
std::vector<std::array<Vector3, 3>> edgeFrames(const MeshReadingInput& input, const Topology& topology)
{
    const std::size_t edgeCount = topology.edges.size();
    std::vector<Vector3> sum(edgeCount, Vector3{});
    std::vector<std::size_t> count(edgeCount, 0);
    for (std::size_t ie = 0; ie < input.shells.size(); ++ie)
    {
        const Vector3& n = topology.normals[ie];
        const ShellCorners& shell = input.shells[ie];
        for (std::size_t k = 0; k < shell.count; ++k)
        {
            const std::size_t idx = topology.indexOf(shell.at(k), shell.at(k + 1));
            const double s = count[idx] > 0 && dot3(sum[idx], n) < 0.0 ? -1.0 : 1.0;
            sum[idx] = add3(sum[idx], scl3(n, s));
            ++count[idx];
        }
    }
    // the orientation of the first line element of each edge
    std::map<EdgeKey, std::optional<Vector3>> lineOrientation;
    for (const LineGrids& line : input.lines)
        lineOrientation.try_emplace(EdgeKey::of(line.grids[0], line.grids[1]), line.orientation);
    std::vector<std::array<Vector3, 3>> frames;
    frames.reserve(edgeCount);
    for (std::size_t idx = 0; idx < edgeCount; ++idx)
    {
        const EdgeKey edge = topology.edges[idx];
        const Vector3 ev = unit3(sub3(input.grids[edge.higher], input.grids[edge.lower]));
        if (count[idx] == 0)
        {
            const auto line = lineOrientation.find(edge);
            const std::optional<Vector3> orientation =
                line == lineOrientation.end() ? std::optional<Vector3>{} : line->second;
            const auto acrossEdge = [&](const Vector3& v) { return sub3(v, scl3(ev, dot3(v, ev))); };
            const Vector3 seed = std::abs(ev[0]) < 0.6 ? Vector3{1.0, 0.0, 0.0} : Vector3{0.0, 1.0, 0.0};
            Vector3 nv{};
            if (orientation && norm3(acrossEdge(*orientation)) > 0.0)
                nv = unit3(acrossEdge(*orientation));
            else
                nv = unit3(acrossEdge(seed));
            frames.push_back({ev, cross3(nv, ev), nv});
            continue;
        }
        Vector3 nv = unit3(sub3(sum[idx], scl3(ev, dot3(sum[idx], ev))));
        Vector3 bv = cross3(nv, ev);
        if (count[idx] == 1)
        {
            if (const std::optional<Vector3> d = fixedNormalDirection(input, edge.lower, edge.higher, ev))
            {
                if (std::abs(dot3(*d, bv)) >= std::abs(dot3(*d, nv)))
                {
                    bv = scl3(*d, signOf(dot3(*d, bv)));
                    nv = cross3(ev, bv);
                }
                else
                {
                    nv = scl3(*d, signOf(dot3(*d, nv)));
                    bv = cross3(nv, ev);
                }
            }
        }
        frames.push_back({ev, bv, nv});
    }
    return frames;
}

std::size_t addEdge(Topology& topology, const EdgeKey& key)
{
    const auto [it, added] = topology.edgeIndex.try_emplace(key, topology.edges.size());
    if (added)
    {
        topology.edges.push_back(key);
        topology.edgeElements.emplace_back();
    }
    return it->second;
}

} // namespace

std::set<EdgeKey> rawEdgeSet(const MeshReadingInput& input)
{
    std::set<EdgeKey> raw = input.rawEdges;
    for (const std::size_t ie : input.rawElements)
    {
        const ShellCorners& shell = input.shells.at(ie);
        for (std::size_t k = 0; k < shell.count; ++k)
            raw.insert(EdgeKey::of(shell.at(k), shell.at(k + 1)));
    }
    return raw;
}

std::size_t Topology::indexOf(std::size_t a, std::size_t b) const
{
    const auto it = edgeIndex.find(EdgeKey::of(a, b));
    if (it == edgeIndex.end())
        throw base::InternalError("mesh reading: no edge between the two grids");
    return it->second;
}

Topology Topology::fromMesh(const MeshReadingInput& input)
{
    Topology topology;
    topology.rawEdges = rawEdgeSet(input);
    topology.normals.reserve(input.shells.size());
    for (std::size_t ie = 0; ie < input.shells.size(); ++ie)
    {
        const ShellCorners& shell = input.shells[ie];
        const auto x = [&](std::size_t k) { return input.grids[shell.grids[k]]; };
        // the cross product of two sides of a triangle, of the two diagonals of a quadrilateral
        topology.normals.push_back(shell.count == 3 ? unit3(cross3(sub3(x(1), x(0)), sub3(x(2), x(0))))
                                                    : unit3(cross3(sub3(x(2), x(0)), sub3(x(3), x(1)))));
        for (std::size_t k = 0; k < shell.count; ++k)
        {
            const std::size_t idx = addEdge(topology, EdgeKey::of(shell.at(k), shell.at(k + 1)));
            topology.edgeElements[idx].push_back(ie);
        }
    }
    // the edges of the line elements that no shell has: straight, read from nothing
    for (const LineGrids& line : input.lines)
        static_cast<void>(addEdge(topology, EdgeKey::of(line.grids[0], line.grids[1])));
    topology.frames = edgeFrames(input, topology);
    detail::SurfaceReconstruction reading(input, topology);
    const std::size_t edgeCount = topology.edges.size();
    std::vector<Vector3> bulges(edgeCount, Vector3{});
    for (std::size_t e = 0; e < edgeCount; ++e)
        if (!topology.edgeElements[e].empty())
            bulges[e] = reading.bulge(topology.edges[e].lower, topology.edges[e].higher);
    std::vector<double> weights(edgeCount, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t e = 0; e < edgeCount; ++e)
    {
        const Vector3* na = reading.sectionNormal(topology.edges[e].lower, topology.edges[e].higher);
        const Vector3* nb = reading.sectionNormal(topology.edges[e].higher, topology.edges[e].lower);
        // the cosine of half the angle of the normals at the two grids
        if (na != nullptr && nb != nullptr)
            weights[e] = std::sqrt(std::fmax(0.5 * (1.0 + dot3(*na, *nb)), 0.0));
    }
    topology.bulges = std::move(bulges);
    topology.weights = std::move(weights);
    return topology;
}

ElementAxes elementAxes(const std::array<Vector3, 4>& x)
{
    const Vector3 k = unit3(cross3(sub3(x[2], x[0]), sub3(x[3], x[1])));
    const Vector3 v12 = sub3(x[1], x[0]);
    const Vector3 i = unit3(sub3(v12, scl3(k, dot3(v12, k))));
    const Vector3 j = cross3(k, i);
    // coordinates in the mean plane, from grid 1
    const auto planar = [&](const Vector3& p)
    {
        const Vector3 d = sub3(p, x[0]);
        return std::array<double, 2>{dot3(d, i), dot3(d, j)};
    };
    const std::array<double, 2> u0 = planar(x[0]);
    const std::array<double, 2> u1 = planar(x[1]);
    const std::array<double, 2> u2 = planar(x[2]);
    const std::array<double, 2> u3 = planar(x[3]);
    const double x31 = u2[0] - u0[0];
    const double y31 = u2[1] - u0[1];
    const double x42 = u3[0] - u1[0];
    const double y42 = u3[1] - u1[1];
    const double aa = std::hypot(x31, y31);
    const double bb = std::hypot(x42, y42);
    // the bisector of the two diagonals
    const double exi = x31 / aa - x42 / bb;
    const double exj = y31 / aa - y42 / bb;
    const Vector3 xe = unit3(add3(scl3(i, exi), scl3(j, exj)));
    return {{xe, cross3(k, xe), k}, i};
}

ElementAxes elementAxesTriangle(const std::array<Vector3, 3>& x)
{
    const Vector3 k = unit3(cross3(sub3(x[1], x[0]), sub3(x[2], x[0])));
    const Vector3 i = unit3(sub3(x[1], x[0]));
    return {{i, cross3(k, i), k}, i};
}

} // namespace oos::elements::mixedHybrid
