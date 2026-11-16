#pragma once

#include "oos/elements/mixedHybrid/geometry/ElementSurface.hpp"
#include "oos/elements/mixedHybrid/geometry/MeshTopology.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

// Structured meshes for the checks of the geometry of the mixed hybrid formulation.
// Reference: src/model.rs (mapped_mesh, split_mesh, tet_mesh, tet_mesh_creases) and the fill of tests/tria.rs.

namespace oos::checks::geometry
{

namespace mh = oos::elements::mixedHybrid;

using MapOfSquare = std::function<mh::Vector3(double, double)>;
using MapOfCube = std::function<mh::Vector3(double, double, double)>;

// The grids and quadrilaterals of nx × ny cells on the map f(s, t), s, t in [0, 1]: grid (i, j) has index
// i (ny + 1) + j; the cells are counter-clockwise in (s, t).
struct QuadMesh
{
    std::vector<mh::Vector3> grids;
    std::vector<std::array<std::size_t, 4>> quads;
};

inline QuadMesh mappedMesh(std::size_t nx, std::size_t ny, const MapOfSquare& f)
{
    QuadMesh mesh;
    for (std::size_t i = 0; i <= nx; ++i)
        for (std::size_t j = 0; j <= ny; ++j)
            mesh.grids.push_back(
                f(static_cast<double>(i) / static_cast<double>(nx), static_cast<double>(j) / static_cast<double>(ny)));
    const auto id = [&](std::size_t i, std::size_t j) { return i * (ny + 1) + j; };
    for (std::size_t i = 0; i < nx; ++i)
        for (std::size_t j = 0; j < ny; ++j)
            mesh.quads.push_back({id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)});
    return mesh;
}

// How the cells become elements: quadrilaterals; two triangles per cell cut along G1-G3, along G2-G4, or alternately;
// every other cell cut in two triangles along G1-G3.
enum class CellPattern
{
    Quadrilaterals,
    FirstDiagonal,
    SecondDiagonal,
    AlternateDiagonals,
    Mixed,
};

// The shells on the cells of mappedMesh(.., ny, ..).
inline std::vector<mh::ShellCorners> cellShells(std::size_t ny, const std::vector<std::array<std::size_t, 4>>& quads,
                                                CellPattern pattern)
{
    std::vector<mh::ShellCorners> out;
    for (std::size_t iq = 0; iq < quads.size(); ++iq)
    {
        const auto& q = quads[iq];
        const std::size_t i = iq / ny;
        const std::size_t j = iq % ny;
        switch (pattern)
        {
        case CellPattern::Quadrilaterals:
            out.push_back(mh::ShellCorners::quadrilateral(q[0], q[1], q[2], q[3]));
            break;
        case CellPattern::Mixed:
            if ((i + j) % 2 == 0)
            {
                out.push_back(mh::ShellCorners::triangle(q[0], q[1], q[2]));
                out.push_back(mh::ShellCorners::triangle(q[0], q[2], q[3]));
            }
            else
            {
                out.push_back(mh::ShellCorners::quadrilateral(q[0], q[1], q[2], q[3]));
            }
            break;
        default:
        {
            const bool first = pattern == CellPattern::FirstDiagonal ||
                               (pattern == CellPattern::AlternateDiagonals && (i + j) % 2 == 0);
            if (first)
            {
                out.push_back(mh::ShellCorners::triangle(q[0], q[1], q[2]));
                out.push_back(mh::ShellCorners::triangle(q[0], q[2], q[3]));
            }
            else
            {
                out.push_back(mh::ShellCorners::triangle(q[0], q[1], q[3]));
                out.push_back(mh::ShellCorners::triangle(q[1], q[2], q[3]));
            }
        }
        }
    }
    return out;
}

// The reading input of nx × ny cells on a map.
inline mh::MeshReadingInput cellMesh(std::size_t nx, std::size_t ny, const MapOfSquare& f, CellPattern pattern)
{
    const QuadMesh mesh = mappedMesh(nx, ny, f);
    mh::MeshReadingInput input;
    input.grids = mesh.grids;
    input.shells = cellShells(ny, mesh.quads, pattern);
    return input;
}

// The tetrahedra of nx × ny × nz cells on the map f(s, t, w) of the unit cube, six per cell around the diagonal from
// its first corner to its last; grid (i, j, k) has index (i (ny + 1) + j)(nz + 1) + k.
struct TetMesh
{
    std::vector<mh::Vector3> grids;
    std::vector<std::array<std::size_t, 4>> tets;
};

inline TetMesh tetMesh(std::size_t nx, std::size_t ny, std::size_t nz, const MapOfCube& f)
{
    const auto id = [&](std::size_t i, std::size_t j, std::size_t k) { return (i * (ny + 1) + j) * (nz + 1) + k; };
    TetMesh mesh;
    for (std::size_t i = 0; i <= nx; ++i)
        for (std::size_t j = 0; j <= ny; ++j)
            for (std::size_t k = 0; k <= nz; ++k)
                mesh.grids.push_back(f(static_cast<double>(i) / static_cast<double>(nx),
                                       static_cast<double>(j) / static_cast<double>(ny),
                                       static_cast<double>(k) / static_cast<double>(nz)));
    // the six orders in which the three coordinates can be raised from one corner to the other
    constexpr std::array<std::array<std::size_t, 3>, 6> PATHS{
        {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};
    for (std::size_t i = 0; i < nx; ++i)
        for (std::size_t j = 0; j < ny; ++j)
            for (std::size_t k = 0; k < nz; ++k)
                for (const auto& path : PATHS)
                {
                    std::array<std::size_t, 3> c{i, j, k};
                    std::array<std::size_t, 4> tet{};
                    tet[0] = id(i, j, k);
                    for (std::size_t step = 0; step < 3; ++step)
                    {
                        ++c[path[step]];
                        tet[step + 1] = id(c[0], c[1], c[2]);
                    }
                    mesh.tets.push_back(tet);
                }
    return mesh;
}

// The edges of a mesh of tetMesh that run along the twelve edges of its block.
inline std::vector<std::pair<std::size_t, std::size_t>> tetMeshCreases(std::size_t nx, std::size_t ny, std::size_t nz)
{
    const auto id = [&](std::size_t i, std::size_t j, std::size_t k) { return (i * (ny + 1) + j) * (nz + 1) + k; };
    std::vector<std::pair<std::size_t, std::size_t>> out;
    for (const auto& [j, k] : std::array<std::pair<std::size_t, std::size_t>, 4>{{{0, 0}, {ny, 0}, {0, nz}, {ny, nz}}})
        for (std::size_t i = 0; i < nx; ++i)
            out.emplace_back(id(i, j, k), id(i + 1, j, k));
    for (const auto& [i, k] : std::array<std::pair<std::size_t, std::size_t>, 4>{{{0, 0}, {nx, 0}, {0, nz}, {nx, nz}}})
        for (std::size_t j = 0; j < ny; ++j)
            out.emplace_back(id(i, j, k), id(i, j + 1, k));
    for (const auto& [i, j] : std::array<std::pair<std::size_t, std::size_t>, 4>{{{0, 0}, {nx, 0}, {0, ny}, {nx, ny}}})
        for (std::size_t k = 0; k < nz; ++k)
            out.emplace_back(id(i, j, k), id(i, j, k + 1));
    return out;
}

// The surfaces of every shell of a mesh read.
inline std::vector<mh::ElementSurface> surfacesOf(const mh::MeshReadingInput& input, const mh::Topology& topology)
{
    std::vector<mh::ElementSurface> out;
    for (std::size_t ie = 0; ie < input.shells.size(); ++ie)
        out.push_back(mh::elementSurface(input, topology, ie));
    return out;
}

// The largest kink of a geometry report.
inline double largestKink(const std::vector<mh::EdgeGeometry>& report)
{
    double out = 0.0;
    for (const mh::EdgeGeometry& e : report)
        out = std::max(out, e.kink);
    return out;
}

} // namespace oos::checks::geometry
