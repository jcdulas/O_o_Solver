#include "oos/elements/mixedHybrid/geometry/SolidSkin.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

// The representative of the smooth part of face i, with path halving (R. E. Tarjan, J. van Leeuwen, Worst-case
// analysis of set union algorithms, J. ACM 31 (1984) 245-281).
std::size_t findPart(std::vector<std::size_t>& part, std::size_t i)
{
    while (part[i] != i)
    {
        part[i] = part[part[i]];
        i = part[i];
    }
    return i;
}

// The sum of the readings of the conic of one edge of the skin.
struct ConicSum
{
    Vector3 bulge{};
    double weight = 0.0;
    std::size_t count = 0;
};

} // namespace

std::vector<SkinFace> skinFaces(std::span<const Vector3> grids, std::span<const SolidCell> solids)
{
    // the number of solids that have each face, by its sorted grids
    std::map<std::vector<std::size_t>, std::size_t> owners;
    const auto sortedGrids = [&](const SolidCell& solid, const std::vector<std::size_t>& face)
    {
        std::vector<std::size_t> key;
        for (const std::size_t k : face)
            key.push_back(solid.corners[k]);
        std::sort(key.begin(), key.end());
        return key;
    };
    for (const SolidCell& solid : solids)
        for (const std::vector<std::size_t>& face : solid.faces)
            ++owners[sortedGrids(solid, face)];
    std::vector<SkinFace> out;
    for (std::size_t is = 0; is < solids.size(); ++is)
    {
        const SolidCell& solid = solids[is];
        Vector3 sum{};
        for (const std::size_t g : solid.corners)
            sum = add3(sum, grids[g]);
        const Vector3 centre = scl3(sum, 1.0 / static_cast<double>(solid.corners.size()));
        for (std::size_t lf = 0; lf < solid.faces.size(); ++lf)
        {
            const std::vector<std::size_t>& face = solid.faces[lf];
            if (owners[sortedGrids(solid, face)] != 1)
                continue;
            SkinFace skin;
            skin.count = face.size();
            skin.solid = is;
            skin.localFace = lf;
            for (std::size_t k = 0; k < face.size(); ++k)
                skin.grids[k] = solid.corners[face[k]];
            const Vector3& x0 = grids[skin.grids[0]];
            const Vector3& x1 = grids[skin.grids[1]];
            const Vector3& x2 = grids[skin.grids[2]];
            // turned about the outward normal
            if (dot3(cross3(sub3(x1, x0), sub3(x2, x0)), sub3(x0, centre)) < 0.0)
                std::reverse(skin.grids.begin(), skin.grids.begin() + static_cast<std::ptrdiff_t>(skin.count));
            out.push_back(skin);
        }
    }
    return out;
}

SkinReading readSolidSkin(std::span<const Vector3> grids, std::span<const SkinFace> faces,
                          const std::set<EdgeKey>& sharpEdges)
{
    SkinReading reading;
    if (faces.empty())
        return reading;
    // the smooth parts of the skin: faces joined across the edges that two of them share and that are not sharp
    std::map<EdgeKey, std::vector<std::size_t>> sides;
    for (std::size_t it = 0; it < faces.size(); ++it)
    {
        const SkinFace& f = faces[it];
        for (std::size_t k = 0; k < f.count; ++k)
            sides[EdgeKey::of(f.grids[k], f.grids[(k + 1) % f.count])].push_back(it);
    }
    std::vector<std::size_t> part(faces.size());
    std::iota(part.begin(), part.end(), std::size_t{0});
    // the number of edges at each grid that bound a smooth part (sharp, or on the boundary of the skin)
    std::map<std::size_t, std::size_t> sharpAt;
    for (const auto& [edge, list] : sides)
    {
        if (list.size() == 2 && !sharpEdges.contains(edge))
        {
            const std::size_t p = findPart(part, list[0]);
            const std::size_t q = findPart(part, list[1]);
            part[p] = q;
        }
        else
        {
            ++sharpAt[edge.lower];
            ++sharpAt[edge.higher];
        }
    }
    // the shell mesh of the skin: one grid per grid and part
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> index;
    for (std::size_t it = 0; it < faces.size(); ++it)
    {
        const std::size_t pt = findPart(part, it);
        ShellCorners shell;
        shell.count = faces[it].count;
        for (std::size_t k = 0; k < shell.count; ++k)
        {
            const std::size_t g = faces[it].grids[k];
            const auto [entry, added] = index.try_emplace({g, pt}, reading.modelGrid.size());
            if (added)
            {
                reading.mesh.grids.push_back(grids[g]);
                reading.modelGrid.push_back(g);
            }
            shell.grids[k] = entry->second;
        }
        reading.mesh.shells.push_back(shell);
    }
    std::set<std::size_t> through;
    for (std::size_t i = 0; i < reading.modelGrid.size(); ++i)
    {
        const auto found = sharpAt.find(reading.modelGrid[i]);
        if (found != sharpAt.end() && found->second == 2)
            through.insert(i);
    }
    reading.mesh.throughGrids = std::move(through);
    reading.topology = Topology::fromMesh(reading.mesh);
    // the conic of each edge: the mean of its readings (two on a sharp edge)
    std::map<EdgeKey, ConicSum> sums;
    const Topology& topology = reading.topology;
    for (std::size_t idx = 0; idx < topology.edges.size(); ++idx)
    {
        const EdgeKey edge = topology.edges[idx];
        const EdgeKey key = EdgeKey::of(reading.modelGrid[edge.lower], reading.modelGrid[edge.higher]);
        const Vector3 chord = sub3(grids[key.higher], grids[key.lower]);
        const double w = std::isfinite(topology.weights[idx]) ? topology.weights[idx]
                                                              : controlPolygonWeight(chord, topology.bulges[idx]);
        ConicSum& sum = sums[key];
        sum.bulge = add3(sum.bulge, topology.bulges[idx]);
        sum.weight += w;
        ++sum.count;
    }
    for (const auto& [key, sum] : sums)
    {
        const double n = static_cast<double>(sum.count);
        reading.conics[key] = SkinConic{scl3(sum.bulge, 1.0 / n), sum.weight / n};
    }
    return reading;
}

Patch skinFacePatch(const SkinReading& reading, std::size_t face)
{
    const ShellCorners& shell = reading.mesh.shells[face];
    const Topology& topology = reading.topology;
    std::array<Vector3, 4> bulge{};
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::array<double, 4> weight{nan, nan, nan, nan};
    for (std::size_t k = 0; k < shell.count; ++k)
    {
        const std::size_t idx = topology.indexOf(shell.at(k), shell.at(k + 1));
        bulge[k] = topology.bulges[idx];
        weight[k] = topology.weights[idx];
    }
    const auto x = [&](std::size_t k) { return reading.mesh.grids[shell.grids[k]]; };
    if (shell.count == 3)
    {
        TrianglePatch patch = TrianglePatch::fromEdges({x(0), x(1), x(2)}, {bulge[0], bulge[1], bulge[2]},
                                                       {weight[0], weight[1], weight[2]});
        patch.fitInterior();
        return patch;
    }
    RationalPatch patch = RationalPatch::fromWeightedEdges({x(0), x(1), x(2), x(3)}, bulge, weight);
    patch.fitCentre();
    return patch;
}

} // namespace oos::elements::mixedHybrid
