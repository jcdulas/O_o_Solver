#include "oos/elements/mixedHybrid/core/JacobiEigen.hpp"
#include "oos/elements/mixedHybrid/model/SolidPart.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <utility>
#include <vector>

// The supports of the entity DOFs of the solids derived from the supports of the grids (specification, section 5.6;
// src/tdnns/solid.rs::Solids::supports with edge_bc).
//
// Which DOFs: an edge of the skin whose two grids hold the translation component c holds the component c of its
// functions; a face of the skin whose grids all hold c holds the component c of its functions (the reference's rule).
//
// Their values: the reference holds them at zero, which is exact for zero values and for straight edges and flat
// faces only. Here each takes the image of the held grid values by a linear map L, exact for the motions of a small
// basis that the held values see, and L carries its reaction back to the grids (the transpose). An affine field
// u = b + A x gives the entity value A g (g the geometry coefficient of the function, SolidPart::affineField), so L
// estimates the part of A that the value needs from a patch of held grid values:
// - the patch: the grids of the face; for an edge, the grids of the first face of the skin (by key) that has the edge
//   and whose grids all hold c, or else the two grids of the edge;
// - the basis: the 3 translations, the 3 rotations about the centroid x0 of the patch, and the uniform strains along
//   the patch (the 3 in-plane strains of a face, in the plane of its normal; the stretch along an edge), each written
//   u_k(x) = b_k + A_k (x − x0) / h with h the size of the patch so that the columns are of the same scale;
// - V (held components × motions) their values at the held components of the patch, and L = V⁺ᵀ t, t_k = (A_k g)_c / h
//   the value the motion k gives the entity DOF, V⁺ the minimum-norm least-squares inverse (minimumNormInverse).
// L reproduces every motion of the basis that the held values determine: the rigid motions of a face whose grids are
// held, of a symmetry plane (the normal component held on a flat face), of an edge but its twist about its chord
// (which its two grids cannot see). A straight edge or a flat face has g = 0, hence t = 0 and an empty map.
//
// Thread-safety: pure functions of the SolidPart.

namespace oos::elements::mixedHybrid
{

namespace
{

constexpr std::size_t COMPONENTS = 3;

// A[c][d] = ∂u_c/∂x_d.
using Gradient = std::array<std::array<double, 3>, 3>;

// A motion of the basis of a patch: u(x) = b + A (x − x0) / h.
struct PatchMotion
{
    SolidPoint translation{};
    Gradient gradient{};
};

// The map of a patch: its held components, their values, the basis motions and V⁺ (motions × held components).
struct PatchFit
{
    std::vector<std::pair<std::size_t, std::size_t>> held; // (grid, basic component)
    std::vector<double> values;
    std::vector<PatchMotion> motions;
    double size = 1.0;
    Matrix inverse;
};

using HeldValues = std::map<std::size_t, std::array<std::optional<double>, 3>>;

bool holds(const HeldValues& held, std::size_t grid, std::size_t component)
{
    const auto found = held.find(grid);
    return found != held.end() && found->second[component].has_value();
}

// The gradient of the rotation about the basic axis j: A y = e_j × y.
Gradient rotationGradient(std::size_t axis)
{
    Gradient a{};
    SolidPoint e{};
    e[axis] = 1.0;
    for (std::size_t d = 0; d < COMPONENTS; ++d)
    {
        SolidPoint ed{};
        ed[d] = 1.0;
        const SolidPoint column = cross3(e, ed);
        for (std::size_t c = 0; c < COMPONENTS; ++c)
            a[c][d] = column[c];
    }
    return a;
}

// The gradient of the uniform strain (u vᵀ + v uᵀ) / 2.
Gradient strainGradient(const SolidPoint& u, const SolidPoint& v)
{
    Gradient a{};
    for (std::size_t c = 0; c < COMPONENTS; ++c)
        for (std::size_t d = 0; d < COMPONENTS; ++d)
            a[c][d] = 0.5 * (u[c] * v[d] + v[c] * u[d]);
    return a;
}

// The map of the patch of the grids `grids` with the strain motions of gradients `strains`.
PatchFit fitPatch(std::span<const SolidPoint> x, std::vector<std::size_t> grids, const std::vector<Gradient>& strains,
                  const HeldValues& held)
{
    std::sort(grids.begin(), grids.end());
    PatchFit fit;
    SolidPoint centre{};
    for (const std::size_t g : grids)
        centre = add3(centre, x[g]);
    centre = scl3(centre, 1.0 / static_cast<double>(grids.size()));
    double size = 0.0;
    for (const std::size_t g : grids)
        size = std::max(size, norm3(sub3(x[g], centre)));
    fit.size = size > 0.0 ? size : 1.0;
    for (std::size_t c = 0; c < COMPONENTS; ++c)
    {
        PatchMotion translation;
        translation.translation[c] = 1.0;
        fit.motions.push_back(translation);
    }
    for (std::size_t j = 0; j < COMPONENTS; ++j)
        fit.motions.push_back({SolidPoint{}, rotationGradient(j)});
    for (const Gradient& strain : strains)
        fit.motions.push_back({SolidPoint{}, strain});
    for (const std::size_t g : grids)
    {
        const auto found = held.find(g);
        if (found == held.end())
            continue;
        for (std::size_t c = 0; c < COMPONENTS; ++c)
        {
            if (!found->second[c])
                continue;
            fit.held.emplace_back(g, c);
            fit.values.push_back(*found->second[c]);
        }
    }
    Matrix v(fit.held.size(), fit.motions.size());
    for (std::size_t row = 0; row < fit.held.size(); ++row)
    {
        const auto [g, c] = fit.held[row];
        const SolidPoint offset = scl3(sub3(x[g], centre), 1.0 / fit.size);
        for (std::size_t k = 0; k < fit.motions.size(); ++k)
        {
            const PatchMotion& motion = fit.motions[k];
            v(row, k) = motion.translation[c] + motion.gradient[c][0] * offset[0] + motion.gradient[c][1] * offset[1] +
                        motion.gradient[c][2] * offset[2];
        }
    }
    fit.inverse = minimumNormInverse(v);
    return fit;
}

// The uniform strains along a face of grids `cycle` (cyclic order): the in-plane strains of the plane of its normal
// (of two sides on a triangle, of the diagonals on a quadrilateral, which may be warped).
std::vector<Gradient> faceStrains(std::span<const SolidPoint> x, std::span<const std::size_t> cycle)
{
    const SolidPoint& x0 = x[cycle[0]];
    const SolidPoint normal = cycle.size() == 3 ? cross3(sub3(x[cycle[1]], x0), sub3(x[cycle[2]], x0))
                                                : cross3(sub3(x[cycle[2]], x0), sub3(x[cycle[3]], x[cycle[1]]));
    const SolidPoint n = unit3(normal);
    const SolidPoint side = sub3(x[cycle[1]], x0);
    const SolidPoint t1 = unit3(sub3(side, scl3(n, dot3(side, n))));
    const SolidPoint t2 = cross3(n, t1);
    return {strainGradient(t1, t1), strainGradient(t2, t2), strainGradient(t1, t2)};
}

// The stretch along an edge.
std::vector<Gradient> edgeStrains(std::span<const SolidPoint> x, const EdgeKey& edge)
{
    const SolidPoint t = unit3(sub3(x[edge.higher], x[edge.lower]));
    return {strainGradient(t, t)};
}

// The support of component c of the function of geometry coefficient g by the map of `fit`.
SolidDerivedSupport supportOf(const PatchFit& fit, std::size_t entity, std::size_t dof, const SolidPoint& g,
                              std::size_t c)
{
    SolidDerivedSupport out;
    out.entity = entity;
    out.dof = dof;
    // t_k: the value motion k gives the DOF, (A_k g)_c / h
    std::vector<double> target(fit.motions.size());
    for (std::size_t k = 0; k < fit.motions.size(); ++k)
    {
        const Gradient& a = fit.motions[k].gradient;
        target[k] = (a[c][0] * g[0] + a[c][1] * g[1] + a[c][2] * g[2]) / fit.size;
    }
    for (std::size_t row = 0; row < fit.held.size(); ++row)
    {
        double coefficient = 0.0;
        for (std::size_t k = 0; k < fit.motions.size(); ++k)
            coefficient += fit.inverse(k, row) * target[k];
        if (coefficient == 0.0)
            continue;
        out.carriedTo.push_back({fit.held[row].first, fit.held[row].second, coefficient});
        out.value += coefficient * fit.values[row];
    }
    return out;
}

} // namespace

namespace
{

// Two held directions are one when they are parallel within this (1 − |cosine|).
constexpr double PARALLEL_TOLERANCE = 1e-10;

// The directions off the basic axes that every grid of `grids` holds (those of the first grid that each other grid
// holds too, up to the sign), and whether a component of the first grid is held along another direction at another
// grid (a displacement system that turns over the entity, which it does not carry).
struct SharedDirections
{
    std::vector<SolidPoint> directions;
    bool differ = false;
};

SharedDirections sharedDirectionsOf(const std::map<std::size_t, std::vector<SolidSkewAxis>>& skewAt,
                                    std::span<const std::size_t> grids)
{
    SharedDirections shared;
    const auto first = skewAt.find(grids.front());
    if (first == skewAt.end())
        return shared;
    for (const std::size_t grid : grids)
        if (skewAt.count(grid) == 0)
            return shared;
    for (const SolidSkewAxis& axis : first->second)
    {
        bool isShared = true;
        for (const std::size_t grid : grids)
        {
            bool isHeldThere = false;
            for (const SolidSkewAxis& other : skewAt.at(grid))
            {
                if (std::fabs(dot3(axis.direction, other.direction)) >= 1.0 - PARALLEL_TOLERANCE)
                    isHeldThere = true;
                else
                    shared.differ = shared.differ || other.component == axis.component;
            }
            isShared = isShared && isHeldThere;
        }
        if (isShared)
            shared.directions.push_back(axis.direction);
    }
    return shared;
}

// The rows that hold the functions of an entity (`functionCount` of them, three components each) along `direction`.
void appendSkewRows(std::size_t entity, std::size_t functionCount, const SolidPoint& direction,
                    std::vector<SolidSupportRow>& rows)
{
    for (std::size_t m = 0; m < functionCount; ++m)
    {
        SolidSupportRow row;
        row.entity = entity;
        for (std::size_t c = 0; c < 3; ++c)
            if (direction[c] != 0.0)
                row.dofs.emplace_back(3 * m + c, direction[c]);
        rows.push_back(std::move(row));
    }
}

} // namespace

std::vector<SolidDerivedSupport> SolidPart::deriveSupports(std::span<const SolidHeldComponent> held,
                                                           std::span<const SolidSkewAxis> skew,
                                                           SolidSupportRows* rows) const
{
    // the directions off the basic axes: the functions of an edge or a face whose grids all hold one are held along it
    if (rows != nullptr && !skew.empty())
    {
        std::map<std::size_t, std::vector<SolidSkewAxis>> skewAt;
        for (const SolidSkewAxis& axis : skew)
            skewAt[axis.grid].push_back(axis);
        for (std::size_t e = 0; e < m_edges.size(); ++e)
        {
            const SolidPartEdge& edge = m_edges[e];
            if (!m_edgeEntities[e] || !edge.onSkin || edge.follows)
                continue;
            const std::array<std::size_t, 2> grids{edge.grids.lower, edge.grids.higher};
            const SharedDirections shared = sharedDirectionsOf(skewAt, grids);
            for (const SolidPoint& direction : shared.directions)
                appendSkewRows(*m_edgeEntities[e], edge.geometry.size(), direction, rows->rows);
            rows->uncarried += shared.differ ? 1 : 0;
        }
        for (std::size_t f = 0; f < m_faces.size(); ++f)
        {
            const SolidPartFace& face = m_faces[f];
            if (!m_faceEntities[f] || !face.onSkin() || face.follows)
                continue;
            const SharedDirections shared = sharedDirectionsOf(skewAt, face.view());
            for (const SolidPoint& direction : shared.directions)
                appendSkewRows(*m_faceEntities[f], face.geometry.size(), direction, rows->rows);
            rows->uncarried += shared.differ ? 1 : 0;
        }
    }

    const std::span<const SolidPoint> x = m_input.gridPositions;
    HeldValues heldAt;
    for (const SolidHeldComponent& h : held)
        if (h.grid < x.size() && h.component < COMPONENTS)
            heldAt[h.grid][h.component] = h.value;

    // the faces of each edge, by key: those of the skin first (an edge of the skin takes its held values along a face
    // of the skin when one holds them)
    std::vector<std::vector<std::size_t>> facesOfEdge(m_edges.size());
    for (const bool skin : {true, false})
        for (std::size_t f = 0; f < m_faces.size(); ++f)
        {
            const SolidPartFace& face = m_faces[f];
            if (face.onSkin() != skin)
                continue;
            for (std::size_t k = 0; k < face.count; ++k)
                facesOfEdge[m_edgeIndex.at(EdgeKey::of(face.grids[k], face.grids[(k + 1) % face.count]))].push_back(f);
        }
    const auto faceHolds = [&](std::size_t f, std::size_t c)
    {
        const std::span<const std::size_t> grids = m_faces[f].view();
        return std::all_of(grids.begin(), grids.end(), [&](std::size_t g) { return holds(heldAt, g, c); });
    };
    std::map<std::size_t, PatchFit> faceFits;
    std::map<std::size_t, PatchFit> edgeFits;
    const auto faceFit = [&](std::size_t f) -> const PatchFit&
    {
        auto found = faceFits.find(f);
        if (found == faceFits.end())
        {
            const std::span<const std::size_t> grids = m_faces[f].view();
            found = faceFits.emplace(f, fitPatch(x, {grids.begin(), grids.end()}, faceStrains(x, grids), heldAt)).first;
        }
        return found->second;
    };
    const auto edgeFit = [&](std::size_t e) -> const PatchFit&
    {
        auto found = edgeFits.find(e);
        if (found == edgeFits.end())
        {
            const EdgeKey& key = m_edges[e].grids;
            found = edgeFits.emplace(e, fitPatch(x, {key.lower, key.higher}, edgeStrains(x, key), heldAt)).first;
        }
        return found->second;
    };

    std::vector<SolidDerivedSupport> out;
    for (std::size_t e = 0; e < m_edges.size(); ++e)
    {
        const SolidPartEdge& edge = m_edges[e];
        // an edge that follows a shell or a line is held through it (the ties), never directly
        if (!m_edgeEntities[e] || !edge.onSkin || edge.follows)
            continue;
        std::array<const PatchFit*, COMPONENTS> fits{};
        for (std::size_t c = 0; c < COMPONENTS; ++c)
        {
            if (!holds(heldAt, edge.grids.lower, c) || !holds(heldAt, edge.grids.higher, c))
                continue;
            const std::vector<std::size_t>& around = facesOfEdge[e];
            const auto heldFace =
                std::find_if(around.begin(), around.end(), [&](std::size_t f) { return faceHolds(f, c); });
            fits[c] = heldFace != around.end() ? &faceFit(*heldFace) : &edgeFit(e);
        }
        for (std::size_t m = 0; m < edge.geometry.size(); ++m)
            for (std::size_t c = 0; c < COMPONENTS; ++c)
                if (fits[c])
                    out.push_back(supportOf(*fits[c], *m_edgeEntities[e], COMPONENTS * m + c, edge.geometry[m], c));
    }
    for (std::size_t f = 0; f < m_faces.size(); ++f)
    {
        const SolidPartFace& face = m_faces[f];
        if (!m_faceEntities[f] || !face.onSkin() || face.follows)
            continue;
        std::array<bool, COMPONENTS> heldComponent{};
        for (std::size_t c = 0; c < COMPONENTS; ++c)
            heldComponent[c] = faceHolds(f, c);
        if (std::none_of(heldComponent.begin(), heldComponent.end(), [](bool b) { return b; }))
            continue;
        const PatchFit& fit = faceFit(f);
        for (std::size_t m = 0; m < face.geometry.size(); ++m)
            for (std::size_t c = 0; c < COMPONENTS; ++c)
                if (heldComponent[c])
                    out.push_back(supportOf(fit, *m_faceEntities[f], COMPONENTS * m + c, face.geometry[m], c));
    }
    return out;
}

} // namespace oos::elements::mixedHybrid
