#include "oos/elements/mixedHybrid/geometry/MeshTopology.hpp"

#include "SurfaceReconstruction.hpp"
#include "oos/base/errors/InternalError.hpp"

#include <algorithm>
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
// The angle (radians) by which the normals of two shells may pass the fold angle and the edge still be smooth: the
// round-off of an angle computed from the coordinates (18 elements round a circle turn by 20 degrees each, the default
// angle, to the last bits).
constexpr double FOLD_ANGLE_ROUND_OFF = 1e-9;

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

// The unit normal of a shell: the cross product of two sides of a triangle, of the two diagonals of a quadrilateral.
Vector3 shellNormal(std::span<const Vector3> grids, const ShellCorners& shell)
{
    const auto x = [&](std::size_t k) { return grids[shell.grids[k]]; };
    return shell.count == 3 ? unit3(cross3(sub3(x(1), x(0)), sub3(x(2), x(0))))
                            : unit3(cross3(sub3(x(2), x(0)), sub3(x(3), x(1))));
}

// The position of grid `grid` among the grids of a shell.
std::size_t cornerIndex(const ShellCorners& shell, std::size_t grid)
{
    for (std::size_t k = 0; k < shell.count; ++k)
        if (shell.grids[k] == grid)
            return k;
    throw base::InternalError("mesh reading: grid not in the element");
}

// The shell runs along its edge from grid a to grid b (b comes after a in its grids).
bool runsFrom(const ShellCorners& shell, std::size_t a, std::size_t b)
{
    return shell.at(cornerIndex(shell, a) + 1) == b;
}

// The unit direction into a shell, normal to its edge a − d: from a to the centroid of its grids, less its part along
// the edge.
// Reference: src/model.rs::Reconstruction::across.
Vector3 directionInto(const MeshReadingInput& input, const ShellCorners& shell, std::size_t a, std::size_t d)
{
    Vector3 centroid{};
    for (std::size_t k = 0; k < shell.count; ++k)
        centroid = add3(centroid, scl3(input.grids[shell.grids[k]], 1.0 / static_cast<double>(shell.count)));
    const Vector3 ed = unit3(sub3(input.grids[d], input.grids[a]));
    const Vector3 v = sub3(centroid, input.grids[a]);
    return unit3(sub3(v, scl3(ed, dot3(v, ed))));
}

// The pairs of shells of an edge that face each other across it (positions in `shells`): the two shells of an edge
// of two; at a junction of more, each shell and the one whose direction into it is the most opposite, when the choice
// is mutual and the two directions are opposite at all.
// Reference: src/model.rs::Reconstruction::facing.
std::vector<std::pair<std::size_t, std::size_t>> facingPairs(const MeshReadingInput& input,
                                                             const std::vector<std::size_t>& shells,
                                                             const EdgeKey& edge)
{
    if (shells.size() < 2)
        return {};
    if (shells.size() == 2)
        return {{0, 1}};
    std::vector<Vector3> directions;
    for (const std::size_t shell : shells)
        directions.push_back(directionInto(input, input.shells[shell], edge.lower, edge.higher));
    std::vector<std::size_t> best(shells.size());
    for (std::size_t i = 0; i < shells.size(); ++i)
    {
        // the first of the most opposite directions
        std::optional<std::size_t> bestOther;
        double bestDot = 0.0;
        for (std::size_t o = 0; o < shells.size(); ++o)
        {
            const double d = dot3(directions[i], directions[o]);
            if (o != i && (!bestOther || d < bestDot))
            {
                bestOther = o;
                bestDot = d;
            }
        }
        best[i] = *bestOther;
    }
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    for (std::size_t i = 0; i < shells.size(); ++i)
        if (best[best[i]] == i && i < best[i] && dot3(directions[i], directions[best[i]]) < 0.0)
            pairs.emplace_back(i, best[i]);
    return pairs;
}

// A declaration names the edge whole, or the side of one of the two shells.
bool namesSide(const std::map<EdgeKey, std::set<std::size_t>>& sides, const EdgeKey& edge, std::size_t first,
               std::size_t second)
{
    const auto found = sides.find(edge);
    if (found == sides.end())
        return false;
    return found->second.contains(MeshReadingInput::WHOLE_EDGE) || found->second.contains(first) ||
           found->second.contains(second);
}

// What decided the pairs of shells of one edge.
struct PairReasons
{
    bool isDeclared = false;     // a sharp or smooth declaration decided a pair
    bool isSharpByAngle = false; // the angle made a pair sharp
    bool isSmoothByAngle = false;
};

// The state of one edge and the smooth partners of its shells (Topology::states, Topology::partners).
std::pair<EdgeReading, std::map<std::size_t, std::size_t>> readEdgeState(const MeshReadingInput& input,
                                                                        const Topology& topology, std::size_t index)
{
    const EdgeKey edge = topology.edges[index];
    const std::vector<std::size_t>& shells = topology.edgeElements[index];
    const bool isRaw = topology.rawEdges.contains(edge);
    EdgeReading reading;
    std::map<std::size_t, std::size_t> partners;
    PairReasons reasons;
    for (const auto& [i, j] : facingPairs(input, shells, edge))
    {
        const ShellCorners& first = input.shells[shells[i]];
        const ShellCorners& second = input.shells[shells[j]];
        const double angle = std::fmax(cornerAngle(input.grids, first, second, edge.lower, edge.higher),
                                       cornerAngle(input.grids, first, second, edge.higher, edge.lower));
        reading.angle = std::fmax(reading.angle, angle);
        bool isSmooth = true;
        if (isRaw)
        {
            // the two shells keep each other: the reading asks the raw edges apart
        }
        else if (namesSide(input.sharpSides, edge, shells[i], shells[j]))
        {
            isSmooth = false;
            reasons.isDeclared = true;
        }
        else if (namesSide(input.smoothSides, edge, shells[i], shells[j]))
        {
            reasons.isDeclared = true;
        }
        else if (input.foldAngle)
        {
            isSmooth = *input.foldAngle > 0.0 && angle <= *input.foldAngle + FOLD_ANGLE_ROUND_OFF;
            (isSmooth ? reasons.isSmoothByAngle : reasons.isSharpByAngle) = true;
        }
        if (isSmooth)
        {
            partners[shells[i]] = shells[j];
            partners[shells[j]] = shells[i];
        }
    }
    if (isRaw)
    {
        reading.state = EdgeState::Raw;
        reading.origin = input.rawEdges.contains(edge) ? EdgeStateOrigin::Declared : EdgeStateOrigin::RawElement;
        return {reading, partners};
    }
    if (shells.size() < 2)
        return {reading, partners};
    const bool isSharp = partners.size() < shells.size();
    reading.state = isSharp ? EdgeState::Sharp : EdgeState::Smooth;
    if (reasons.isDeclared)
        reading.origin = EdgeStateOrigin::Declared;
    else if (reasons.isSharpByAngle)
        reading.origin = EdgeStateOrigin::Angle;
    else if (isSharp)
        reading.origin = EdgeStateOrigin::Junction;
    else
        reading.origin = reasons.isSmoothByAngle ? EdgeStateOrigin::Angle : EdgeStateOrigin::Unmeasured;
    return {reading, partners};
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
        {
            // an edge declared smooth or sharp keeps that state: the declaration of an edge comes before that of an
            // element
            const EdgeKey edge = EdgeKey::of(shell.at(k), shell.at(k + 1));
            if (!input.smoothSides.contains(edge) && !input.sharpSides.contains(edge))
                raw.insert(edge);
        }
    }
    return raw;
}

Vector3 cornerNormal(std::span<const Vector3> grids, const ShellCorners& shell, std::size_t grid,
                     const Vector3& fallback)
{
    const std::size_t k = cornerIndex(shell, grid);
    const Vector3& x = grids[grid];
    const Vector3 normal =
        cross3(sub3(grids[shell.at(k + 1)], x), sub3(grids[shell.at(k + shell.count - 1)], x));
    return norm3(normal) > 0.0 ? unit3(normal) : fallback;
}

double cornerAngle(std::span<const Vector3> grids, const ShellCorners& first, const ShellCorners& second,
                   std::size_t grid, std::size_t other)
{
    const Vector3 n1 = cornerNormal(grids, first, grid, shellNormal(grids, first));
    const Vector3 n2 = cornerNormal(grids, second, grid, shellNormal(grids, second));
    // two shells that run along the edge in the same sense turn opposite ways about it
    const double sense = runsFrom(first, grid, other) == runsFrom(second, grid, other) ? -1.0 : 1.0;
    return std::acos(std::clamp(sense * dot3(n1, n2), -1.0, 1.0));
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
        topology.normals.push_back(shellNormal(input.grids, shell));
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
    const std::size_t edgeCount = topology.edges.size();
    topology.states.reserve(edgeCount);
    topology.partners.reserve(edgeCount);
    for (std::size_t e = 0; e < edgeCount; ++e)
    {
        auto [state, partners] = readEdgeState(input, topology, e);
        topology.states.push_back(state);
        topology.partners.push_back(std::move(partners));
    }
    detail::SurfaceReconstruction reading(input, topology);
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
