#pragma once

#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/model/SolidPart.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <numbers>
#include <optional>
#include <set>
#include <vector>

// The meshes and the small global systems of the checks of SolidPart (SolidPartChecks.cpp): blocks of every shape
// built from a lattice of grids, and a dense assembly of the stiffness and the loads of a SolidPart on the numbering
// three translations per grid, then the entity DOFs. The meshes A to E are those of the reference probe (the values
// of SolidPartChecks.cpp were computed by the reference crate at commit 9f55f0b on the same meshes).
namespace oos::checks::solidpart
{

namespace mh = oos::elements::mixedHybrid;
using Kind = oos::elements::EntityLayout::Kind;

inline constexpr double YOUNG = 2.0e11;
inline constexpr double POISSON = 0.3;
inline constexpr double DENSITY = 7800.0;

inline mh::SolidMaterial steel(double expansion = 0.0)
{
    return mh::SolidMaterial::isotropic(YOUNG, POISSON, DENSITY, expansion);
}

inline mh::SolidOptions optionsOf(std::size_t degree)
{
    mh::SolidOptions options;
    options.traceDegree = degree;
    options.faceDegree = degree;
    options.stressDegree = degree;
    options.interiorFieldCount = 3;
    return options;
}

// A lattice of (ni + 1)(nj + 1)(nk + 1) grids, grid (i, j, k) at index (i (nj + 1) + j)(nk + 1) + k.
struct Lattice
{
    std::size_t ni = 1;
    std::size_t nj = 1;
    std::size_t nk = 1;

    [[nodiscard]] std::size_t id(std::size_t i, std::size_t j, std::size_t k) const
    {
        return (i * (nj + 1) + j) * (nk + 1) + k;
    }
    [[nodiscard]] std::size_t gridCount() const { return (ni + 1) * (nj + 1) * (nk + 1); }
    // The corners of cell (i, j, k) in the order of a hexahedron.
    [[nodiscard]] std::array<std::size_t, 8> cell(std::size_t i, std::size_t j, std::size_t k) const
    {
        return {id(i, j, k),     id(i + 1, j, k),     id(i + 1, j + 1, k),     id(i, j + 1, k),
                id(i, j, k + 1), id(i + 1, j, k + 1), id(i + 1, j + 1, k + 1), id(i, j + 1, k + 1)};
    }
    [[nodiscard]] bool onBoundary(std::size_t i, std::size_t j, std::size_t k) const
    {
        return i == 0 || i == ni || j == 0 || j == nj || k == 0 || k == nk;
    }
    // The edges of the lattice whose two grids lie on two common boundary planes: the edges of the block.
    [[nodiscard]] std::set<mh::EdgeKey> creases() const
    {
        const auto planes = [&](std::size_t i, std::size_t j, std::size_t k)
        {
            std::set<int> out;
            if (i == 0)
                out.insert(0);
            if (i == ni)
                out.insert(1);
            if (j == 0)
                out.insert(2);
            if (j == nj)
                out.insert(3);
            if (k == 0)
                out.insert(4);
            if (k == nk)
                out.insert(5);
            return out;
        };
        std::set<mh::EdgeKey> out;
        for (std::size_t i = 0; i <= ni; ++i)
            for (std::size_t j = 0; j <= nj; ++j)
                for (std::size_t k = 0; k <= nk; ++k)
                {
                    const std::set<int> a = planes(i, j, k);
                    const std::array<std::array<std::size_t, 3>, 3> steps{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
                    for (const auto& s : steps)
                    {
                        if (i + s[0] > ni || j + s[1] > nj || k + s[2] > nk)
                            continue;
                        const std::set<int> b = planes(i + s[0], j + s[1], k + s[2]);
                        std::size_t common = 0;
                        for (const int p : a)
                            common += b.count(p);
                        if (common >= 2)
                            out.insert(mh::EdgeKey::of(id(i, j, k), id(i + s[0], j + s[1], k + s[2])));
                    }
                }
        return out;
    }
};

// A mesh: what SolidPart reads, its lattice, and the grids to clamp and to load in the checks.
struct Mesh
{
    mh::SolidPartInput input;
    Lattice lattice;
};

inline mh::SolidPartElement elementOf(std::int32_t id, std::vector<std::size_t> corners,
                                      std::vector<std::size_t> midSide = {})
{
    mh::SolidPartElement element;
    element.elementId = id;
    element.corners = std::move(corners);
    element.midSideGrids = std::move(midSide);
    element.material = steel();
    return element;
}

inline void addHexahedra(Mesh& mesh)
{
    const Lattice& l = mesh.lattice;
    for (std::size_t i = 0; i < l.ni; ++i)
        for (std::size_t j = 0; j < l.nj; ++j)
            for (std::size_t k = 0; k < l.nk; ++k)
            {
                const std::array<std::size_t, 8> c = l.cell(i, j, k);
                mesh.input.solids.push_back(
                    elementOf(static_cast<std::int32_t>(mesh.input.solids.size() + 1), {c.begin(), c.end()}));
            }
}

// Grids of a lattice at f(i, j, k), with IDs 1, 2, ... in index order.
using GridPlacement = std::function<mh::SolidPoint(std::size_t, std::size_t, std::size_t)>;

inline Mesh latticeMesh(const Lattice& lattice, const GridPlacement& f, std::size_t degree)
{
    Mesh mesh;
    mesh.lattice = lattice;
    mesh.input.options = optionsOf(degree);
    for (std::size_t i = 0; i <= lattice.ni; ++i)
        for (std::size_t j = 0; j <= lattice.nj; ++j)
            for (std::size_t k = 0; k <= lattice.nk; ++k)
            {
                mesh.input.gridPositions.push_back(f(i, j, k));
                mesh.input.gridIds.push_back(static_cast<std::int32_t>(mesh.input.gridIds.size() + 1));
            }
    mesh.input.sharpEdges = lattice.creases();
    return mesh;
}

// Appends a grid at the middle of grids a and b moved by `offset`; returns its index.
inline std::size_t addMidSideGrid(Mesh& mesh, std::size_t a, std::size_t b, const mh::SolidPoint& offset)
{
    const mh::SolidPoint& xa = mesh.input.gridPositions[a];
    const mh::SolidPoint& xb = mesh.input.gridPositions[b];
    mesh.input.gridPositions.push_back(
        {0.5 * (xa[0] + xb[0]) + offset[0], 0.5 * (xa[1] + xb[1]) + offset[1], 0.5 * (xa[2] + xb[2]) + offset[2]});
    mesh.input.gridIds.push_back(static_cast<std::int32_t>(mesh.input.gridIds.size() + 1));
    return mesh.input.gridPositions.size() - 1;
}

// A: a sector of a thick cylinder (radii 1 and 1.3, 30 degrees, height 0.5) in 1 × 2 × 1 hexahedra.
inline Mesh sectorMesh(std::size_t degree)
{
    Mesh mesh = latticeMesh(
        {1, 2, 1},
        [](std::size_t i, std::size_t j, std::size_t k)
        {
            const double r = 1.0 + 0.3 * static_cast<double>(i);
            const double t = (std::numbers::pi / 6.0) * static_cast<double>(j) / 2.0;
            return mh::SolidPoint{r * std::cos(t), r * std::sin(t), 0.5 * static_cast<double>(k)};
        },
        degree);
    addHexahedra(mesh);
    return mesh;
}

// B: a box of 2 × 1 × 1 hexahedra with mid-side grids on some edges only (some given by one of the two elements of
// a shared edge).
inline Mesh partialMidSideMesh(std::size_t degree)
{
    Mesh mesh = latticeMesh(
        {2, 1, 1},
        [](std::size_t i, std::size_t j, std::size_t k)
        {
            return mh::SolidPoint{0.5 * static_cast<double>(i), 0.4 * static_cast<double>(j),
                                  0.3 * static_cast<double>(k)};
        },
        degree);
    const std::array<std::size_t, 8> c0 = mesh.lattice.cell(0, 0, 0);
    const std::array<std::size_t, 8> c1 = mesh.lattice.cell(1, 0, 0);
    std::vector<std::size_t> mid0(12, mh::NO_MID_SIDE_GRID);
    mid0[8] = addMidSideGrid(mesh, c0[4], c0[5], {0.0, 0.0, 0.04});
    mid0[9] = addMidSideGrid(mesh, c0[5], c0[6], {0.0, 0.0, 0.03});
    mid0[0] = addMidSideGrid(mesh, c0[0], c0[1], {0.0, -0.02, 0.0});
    std::vector<std::size_t> mid1(12, mh::NO_MID_SIDE_GRID);
    mid1[8] = addMidSideGrid(mesh, c1[4], c1[5], {0.0, 0.0, 0.05});
    mid1[10] = addMidSideGrid(mesh, c1[6], c1[7], {0.01, 0.0, 0.02});
    mesh.input.solids.push_back(elementOf(1, {c0.begin(), c0.end()}, mid0));
    mesh.input.solids.push_back(elementOf(2, {c1.begin(), c1.end()}, mid1));
    return mesh;
}

// C: a unit cube of five pyramids and two tetrahedra (the sixth pyramid split along a diagonal of its base) about an
// inner apex.
inline Mesh pyramidTetraMesh(std::size_t degree)
{
    Mesh mesh = latticeMesh(
        {1, 1, 1}, [](std::size_t i, std::size_t j, std::size_t k)
        { return mh::SolidPoint{static_cast<double>(i), static_cast<double>(j), static_cast<double>(k)}; }, degree);
    mesh.input.sharpEdges.clear();
    mesh.input.gridPositions.push_back({0.55, 0.45, 0.52});
    mesh.input.gridIds.push_back(9);
    const std::size_t apex = 8;
    const std::array<std::size_t, 8> c = mesh.lattice.cell(0, 0, 0);
    const std::array<std::array<std::size_t, 4>, 6> faces{
        {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}}};
    const std::vector<mh::SolidPoint>& x = mesh.input.gridPositions;
    std::int32_t id = 1;
    for (std::size_t f = 0; f < faces.size(); ++f)
    {
        std::array<std::size_t, 4> base{c[faces[f][0]], c[faces[f][1]], c[faces[f][2]], c[faces[f][3]]};
        // the base turning counterclockwise seen from the apex
        const mh::SolidPoint normal = mh::cross3(mh::sub3(x[base[2]], x[base[0]]), mh::sub3(x[base[3]], x[base[1]]));
        const mh::SolidPoint middle =
            mh::scl3(mh::add3(mh::add3(x[base[0]], x[base[1]]), mh::add3(x[base[2]], x[base[3]])), 0.25);
        if (mh::dot3(normal, mh::sub3(x[apex], middle)) < 0.0)
            std::reverse(base.begin(), base.end());
        if (f == 3)
        {
            mesh.input.solids.push_back(elementOf(id++, {base[0], base[1], base[2], apex}));
            mesh.input.solids.push_back(elementOf(id++, {base[0], base[2], base[3], apex}));
        }
        else
        {
            mesh.input.solids.push_back(elementOf(id++, {base[0], base[1], base[2], base[3], apex}));
        }
    }
    return mesh;
}

// D: a layer of 2 × 2 × 1 cells of two pentahedra each, its middle column moved.
inline Mesh pentaLayerMesh(std::size_t degree)
{
    Mesh mesh = latticeMesh(
        {2, 2, 1},
        [](std::size_t i, std::size_t j, std::size_t k)
        {
            const bool middle = i == 1 && j == 1;
            return mh::SolidPoint{0.5 * static_cast<double>(i) + (middle ? 0.06 : 0.0),
                                  0.4 * static_cast<double>(j) + (middle ? -0.04 : 0.0), 0.3 * static_cast<double>(k)};
        },
        degree);
    mesh.input.sharpEdges.clear();
    std::int32_t id = 1;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
        {
            const std::array<std::size_t, 8> c = mesh.lattice.cell(i, j, 0);
            mesh.input.solids.push_back(elementOf(id++, {c[0], c[1], c[2], c[4], c[5], c[6]}));
            mesh.input.solids.push_back(elementOf(id++, {c[0], c[2], c[3], c[4], c[6], c[7]}));
        }
    return mesh;
}

// E: 2 × 2 × 1 hexahedra under a dome z = 0.4 + 0.3 (0.5 − x² − 0.6 y²).
inline Mesh domeMesh(std::size_t degree)
{
    Mesh mesh = latticeMesh(
        {2, 2, 1},
        [](std::size_t i, std::size_t j, std::size_t k)
        {
            const double x = 0.5 * static_cast<double>(i) - 0.5;
            const double y = 0.5 * static_cast<double>(j) - 0.5;
            const double top = 0.4 + 0.3 * (0.5 - x * x - 0.6 * y * y);
            return mh::SolidPoint{x, y, top * static_cast<double>(k)};
        },
        degree);
    addHexahedra(mesh);
    return mesh;
}

// A block [0, 1] × [0, 0.8] × [0, 0.6] of n × n × n cells, its interior grids moved by 0.2 of a cell.
inline Mesh distortedBlock(std::size_t n, std::size_t degree)
{
    const double h = 1.0 / static_cast<double>(n);
    const Lattice lattice{n, n, n};
    return latticeMesh(
        lattice,
        [&](std::size_t i, std::size_t j, std::size_t k)
        {
            const double fi = static_cast<double>(i);
            const double fj = static_cast<double>(j);
            const double fk = static_cast<double>(k);
            const double d = lattice.onBoundary(i, j, k) ? 0.0 : 0.2 * h;
            return mh::SolidPoint{fi * h + d * std::sin(1.7 * fi + 0.9 * fj - 0.6 * fk),
                                  0.8 * (fj * h + d * std::cos(2.3 * fi - 1.3 * fj + 0.8 * fk)),
                                  0.6 * (fk * h + d * std::sin(0.7 * fi + 1.9 * fj + 1.1 * fk))};
        },
        degree);
}

inline Mesh hexaBlock(std::size_t n, std::size_t degree)
{
    Mesh mesh = distortedBlock(n, degree);
    addHexahedra(mesh);
    return mesh;
}

// The same block, each cell cut into six tetrahedra about its diagonal from corner 0 to corner 6 (the same cut in
// every cell, so that the faces between cells match).
inline Mesh tetraBlock(std::size_t n, std::size_t degree)
{
    Mesh mesh = distortedBlock(n, degree);
    const std::array<std::array<std::size_t, 4>, 6> tetrahedra{
        {{0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6}}};
    std::int32_t id = 1;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < n; ++k)
            {
                const std::array<std::size_t, 8> c = mesh.lattice.cell(i, j, k);
                for (const auto& t : tetrahedra)
                    mesh.input.solids.push_back(elementOf(id++, {c[t[0]], c[t[1]], c[t[2]], c[t[3]]}));
            }
    return mesh;
}

// A block of 2 × 2 × 1 hexahedra, every edge with its mid-side grid, those of the top face raised by 0.08 of their
// length (a curved top), shared by the elements of the edge.
inline Mesh curvedMidSideBlock(std::size_t degree)
{
    Mesh mesh = latticeMesh(
        {2, 2, 1},
        [](std::size_t i, std::size_t j, std::size_t k)
        {
            return mh::SolidPoint{0.5 * static_cast<double>(i), 0.4 * static_cast<double>(j),
                                  0.6 * static_cast<double>(k)};
        },
        degree);
    addHexahedra(mesh);
    std::map<mh::EdgeKey, std::size_t> mids;
    const std::vector<mh::ShapeEdge> edges(mh::shapeEdges(mh::VolumeShape::Hexahedron).begin(),
                                           mh::shapeEdges(mh::VolumeShape::Hexahedron).end());
    for (mh::SolidPartElement& element : mesh.input.solids)
    {
        for (const mh::ShapeEdge& e : edges)
        {
            const std::size_t a = element.corners[e[0]];
            const std::size_t b = element.corners[e[1]];
            const auto found = mids.find(mh::EdgeKey::of(a, b));
            if (found != mids.end())
            {
                element.midSideGrids.push_back(found->second);
                continue;
            }
            const mh::SolidPoint& xa = mesh.input.gridPositions[a];
            const mh::SolidPoint& xb = mesh.input.gridPositions[b];
            const bool top = std::fabs(xa[2] - 0.6) < 1e-12 && std::fabs(xb[2] - 0.6) < 1e-12;
            const double rise = top ? 0.08 * mh::norm3(mh::sub3(xb, xa)) : 0.0;
            const std::size_t g = addMidSideGrid(mesh, a, b, {0.0, 0.0, rise});
            mids.emplace(mh::EdgeKey::of(a, b), g);
            element.midSideGrids.push_back(g);
        }
    }
    return mesh;
}

// ---- a small dense global system: three translations per grid, then the entity DOFs

inline std::size_t globalDof(const mh::SolidPart& part, const mh::SolidLocalUnknown& u)
{
    return u.onGrid ? 3 * u.index + u.component
                    : 3 * part.gridPositions().size() + part.dofOffset(u.index) + u.component;
}

inline std::size_t systemSize(const mh::SolidPart& part, Kind kind)
{
    const std::size_t entityCount = part.entities(kind).size();
    return 3 * part.gridPositions().size() + part.dofOffset(entityCount);
}

struct System
{
    mh::Matrix stiffness;
    mh::Matrix mass;
    mh::Vector load;
    std::vector<std::optional<mh::VolumeKernel>> kernels;
    // The DOFs some solid has (the grids that are no corner, the mid-side grids, have none).
    std::vector<bool> used;
    bool ok = true;
};

using VolumeLoads = std::function<std::vector<mh::SolidLoad>(std::size_t solid)>;
using FaceLoads = std::function<mh::Vector(std::size_t solid, const mh::VolumeKernel& kernel)>;

// The stiffness (and mass) of the solids of `part` and the loads: the volume loads of each solid through its kernel,
// and the face loads on its parameters.
inline System assemble(const mh::SolidPart& part, Kind kind, bool withMass, const VolumeLoads& volumeLoads = {},
                       const FaceLoads& faceLoads = {})
{
    const std::size_t n = systemSize(part, kind);
    System system;
    system.stiffness = mh::Matrix(n, n);
    if (withMass)
        system.mass = mh::Matrix(n, n);
    system.load = mh::Vector(n, 0.0);
    system.used.assign(n, false);
    system.kernels.resize(part.solidCount());
    for (std::size_t is = 0; is < part.solidCount(); ++is)
    {
        if (!part.usable(is))
            continue;
        const std::vector<mh::SolidLoad> loads = volumeLoads ? volumeLoads(is) : std::vector<mh::SolidLoad>{};
        mh::SolidResult<mh::VolumeKernel> kernel = part.buildKernel(is, kind, withMass, loads);
        if (!kernel.ok())
        {
            system.ok = false;
            continue;
        }
        const std::vector<mh::SolidLocalUnknown> local = part.localUnknowns(is, kind);
        std::vector<std::size_t> dofs;
        for (const mh::SolidLocalUnknown& u : local)
        {
            dofs.push_back(globalDof(part, u));
            system.used[dofs.back()] = true;
        }
        const mh::Matrix k = part.localStiffness(is, kernel.value(), kind);
        for (std::size_t i = 0; i < dofs.size(); ++i)
            for (std::size_t j = 0; j < dofs.size(); ++j)
                system.stiffness(dofs[i], dofs[j]) += k(i, j);
        if (withMass)
        {
            const mh::Matrix m = part.localMass(is, kernel.value(), kind);
            for (std::size_t i = 0; i < dofs.size(); ++i)
                for (std::size_t j = 0; j < dofs.size(); ++j)
                    system.mass(dofs[i], dofs[j]) += m(i, j);
        }
        for (const mh::SolidLoadResult& result : kernel.value().loads())
            for (std::size_t i = 0; i < result.force.size(); ++i)
                system.load[dofs[i]] += result.force[i];
        if (faceLoads)
        {
            const mh::Vector f = faceLoads(is, kernel.value());
            for (std::size_t i = 0; i < f.size(); ++i)
                system.load[dofs[i]] += f[i];
        }
        system.kernels[is] = std::move(kernel.value());
    }
    return system;
}

// The solution of K u = f with the DOFs of `held` at their values, the DOFs no solid has at zero; empty when the
// free system is singular.
inline std::optional<mh::Vector> solveHeld(const System& system, const std::map<std::size_t, double>& held)
{
    const std::size_t n = system.load.size();
    std::vector<std::size_t> freeDofs;
    for (std::size_t i = 0; i < n; ++i)
        if (!held.contains(i) && system.used[i])
            freeDofs.push_back(i);
    mh::Vector u(n, 0.0);
    for (const auto& [dof, value] : held)
        u[dof] = value;
    mh::Matrix kff(freeDofs.size(), freeDofs.size());
    mh::Vector rhs(freeDofs.size());
    for (std::size_t a = 0; a < freeDofs.size(); ++a)
    {
        double r = system.load[freeDofs[a]];
        for (const auto& [dof, value] : held)
            r -= system.stiffness(freeDofs[a], dof) * value;
        rhs[a] = r;
        for (std::size_t b = 0; b < freeDofs.size(); ++b)
            kff(a, b) = system.stiffness(freeDofs[a], freeDofs[b]);
    }
    const mh::LuDecomposition lu(kff);
    const std::optional<mh::Vector> x = lu.solve(rhs);
    if (!x)
        return std::nullopt;
    for (std::size_t a = 0; a < freeDofs.size(); ++a)
        u[freeDofs[a]] = (*x)[a];
    return u;
}

// The values of the local unknowns of a solid in a global vector.
inline mh::Vector localValues(const mh::SolidPart& part, std::size_t solid, Kind kind, const mh::Vector& u)
{
    mh::Vector out;
    for (const mh::SolidLocalUnknown& unknown : part.localUnknowns(solid, kind))
        out.push_back(u[globalDof(part, unknown)]);
    return out;
}

// The global vector of the affine field u = b + A x: the grids, and the entities by SolidPart::affineField.
inline mh::Vector affineVector(const mh::SolidPart& part, Kind kind, const std::array<std::array<double, 3>, 3>& a,
                               const mh::SolidPoint& b)
{
    oos::base::Mat3 gradient;
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            gradient(r, c) = a[r][c];
    mh::Vector out(systemSize(part, kind), 0.0);
    const std::span<const mh::SolidPoint> x = part.gridPositions();
    for (std::size_t g = 0; g < x.size(); ++g)
        for (std::size_t c = 0; c < 3; ++c)
            out[3 * g + c] = b[c] + a[c][0] * x[g][0] + a[c][1] * x[g][1] + a[c][2] * x[g][2];
    const std::vector<double> entities = part.affineField(gradient, kind);
    for (std::size_t i = 0; i < entities.size(); ++i)
        out[3 * x.size() + i] = entities[i];
    return out;
}

// The gradient of the rigid rotation about the basic axis `axis`: A y = e_axis × y.
inline std::array<std::array<double, 3>, 3> rotationGradient(std::size_t axis)
{
    std::array<std::array<double, 3>, 3> a{};
    mh::SolidPoint e{};
    e[axis] = 1.0;
    for (std::size_t d = 0; d < 3; ++d)
    {
        mh::SolidPoint ed{};
        ed[d] = 1.0;
        const mh::SolidPoint column = mh::cross3(e, ed);
        for (std::size_t c = 0; c < 3; ++c)
            a[c][d] = column[c];
    }
    return a;
}

// The six rigid motions: three translations, three rotations about the basic axes through the origin.
inline std::vector<mh::Vector> rigidVectors(const mh::SolidPart& part, Kind kind)
{
    std::vector<mh::Vector> out;
    for (std::size_t c = 0; c < 3; ++c)
    {
        mh::SolidPoint b{};
        b[c] = 1.0;
        out.push_back(affineVector(part, kind, {}, b));
    }
    for (std::size_t axis = 0; axis < 3; ++axis)
        out.push_back(affineVector(part, kind, rotationGradient(axis), {}));
    return out;
}

inline double maxAbsOf(std::span<const double> v)
{
    double out = 0.0;
    for (const double x : v)
        out = std::fmax(out, std::fabs(x));
    return out;
}

} // namespace oos::checks::solidpart
