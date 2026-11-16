#include "oos/elements/mixedHybrid/model/SolidPart.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/core/MaterialMatrices.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"
#include "oos/elements/mixedHybrid/core/SolidSpaces.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/geometry/SolidSkin.hpp"
#include "oos/elements/mixedHybrid/model/SolidPartProjections.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

constexpr std::size_t COMPONENTS = 3;
constexpr std::size_t STRESS_COMPONENTS = 6;

// The degree of the traces and of the faces at which the skin is read from the mesh: the reference reads it for
// quartic traces and faces only (`quartic` in solid.rs::build); below, the skin is straight but for the mid-side grids.
constexpr std::size_t SKIN_READING_DEGREE = 4;

// A mid-side grid moved along its chord d by δ |d| from the middle folds its edge when |δ| reaches 1/4: the parabola
// x(s) = X_a + s d + 4 s (1 − s) (X_m − (X_a + X_b)/2) has the slope |d| (1 + 4 δ (1 − 2s)) along the chord, which
// vanishes at an end of the edge when |δ| = 1/4 (the quarter point).
constexpr double MID_SIDE_FOLDING_SHIFT = 0.25;

std::vector<std::size_t> sortedCopy(std::span<const std::size_t> grids)
{
    std::vector<std::size_t> out(grids.begin(), grids.end());
    std::sort(out.begin(), out.end());
    return out;
}

bool allDistinct(std::span<const std::size_t> grids)
{
    const std::vector<std::size_t> sorted = sortedCopy(grids);
    return std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
}

// The grids of a face in its canonical cyclic order: from its lowest grid towards the lower of its two neighbours.
// The functions of a face do not depend on the corner nor on the direction it is given in (VolumeShape.hpp): this
// order only makes the projections of its geometry independent of the element that met it first.
std::array<std::size_t, 4> canonicalCycle(const std::vector<std::size_t>& cycle)
{
    const std::size_t n = cycle.size();
    const std::size_t first = static_cast<std::size_t>(std::min_element(cycle.begin(), cycle.end()) - cycle.begin());
    const bool forward = cycle[(first + 1) % n] < cycle[(first + n - 1) % n];
    std::array<std::size_t, 4> out{};
    for (std::size_t k = 0; k < n; ++k)
        out[k] = forward ? cycle[(first + k) % n] : cycle[(first + n - k) % n];
    return out;
}

// The law of a material given in its axes, written in the basic axes: D_basic = Tᵀ D T, T the transformation of the
// engineering strains from the basic axes to the material axes (solidStrainTransformation), so that the strain
// energy is the same in both frames.
SolidMaterial materialInBasicAxes(const SolidMaterial& material, const base::Mat3& axes)
{
    const elements::Mat6 t = solidStrainTransformation(axes);
    SolidMaterial out = material;
    for (std::size_t i = 0; i < STRESS_COMPONENTS; ++i)
        for (std::size_t j = 0; j < STRESS_COMPONENTS; ++j)
        {
            double sum = 0.0;
            for (std::size_t k = 0; k < STRESS_COMPONENTS; ++k)
                for (std::size_t l = 0; l < STRESS_COMPONENTS; ++l)
                    sum += t(k, i) * material.stiffness[k][l] * t(l, j);
            out.stiffness[i][j] = sum;
        }
    return out;
}

// The point of the rational quadratic arc from xa to xb of control point `control` and weight w at the parameter t
// (the conic of an edge of the skin, document 25 of the reference, 3.2):
//   x(t) = (b0 xa + b1 control + b2 xb) / (b0 + b1 + b2),  b0 = (1 − t)², b1 = 2 w t (1 − t), b2 = t².
SolidPoint conicPoint(const SolidPoint& xa, const SolidPoint& control, const SolidPoint& xb, double w, double t)
{
    const double b0 = (1.0 - t) * (1.0 - t);
    const double b1 = 2.0 * w * t * (1.0 - t);
    const double b2 = t * t;
    SolidPoint out{};
    for (std::size_t m = 0; m < COMPONENTS; ++m)
        out[m] = (b0 * xa[m] + b1 * control[m] + b2 * xb[m]) / (b0 + b1 + b2);
    return out;
}

SolidPoint meanOf(std::span<const SolidPoint> positions, std::span<const std::size_t> grids)
{
    SolidPoint sum{};
    for (const std::size_t g : grids)
        sum = add3(sum, positions[g]);
    return scl3(sum, 1.0 / static_cast<double>(grids.size()));
}

} // namespace

// ---- building

SolidPart::SolidPart(SolidPartInput input) : m_input(std::move(input))
{
    const SolidOptions& options = m_input.options;
    m_traceDegree = std::max<std::size_t>(options.traceDegree, 1);
    m_faceDegree = std::max<std::size_t>(std::min(options.faceDegree, options.traceDegree), 1);
    readSolids();
    buildTopology();
    findTies();
    buildGeometry();
    buildEntities();
}

bool SolidPart::takesCurves() const
{
    return m_traceDegree == SKIN_READING_DEGREE && m_faceDegree == SKIN_READING_DEGREE;
}

void SolidPart::addProblem(std::size_t solid, SolidPartProblem::Severity severity, std::string message)
{
    m_problems.push_back({severity, solid, m_input.solids[solid].elementId, std::move(message)});
}

void SolidPart::readSolids()
{
    const std::size_t gridCount = m_input.gridPositions.size();
    const std::size_t count = m_input.solids.size();
    m_usable.assign(count, false);
    m_shapes.assign(count, VolumeShape::Tetrahedron);
    m_basicMaterials.assign(count, SolidMaterial{});
    m_solidEdges.assign(count, {});
    m_solidFaces.assign(count, {});
    using Severity = SolidPartProblem::Severity;
    for (std::size_t is = 0; is < count; ++is)
    {
        const SolidPartElement& element = m_input.solids[is];
        const std::optional<VolumeShape> shape = volumeShapeOfCornerCount(element.corners.size());
        if (!shape)
        {
            addProblem(is, Severity::Error,
                       std::to_string(element.corners.size()) + " corner grid points: a solid has 4, 5, 6 or 8");
            continue;
        }
        const bool inRange =
            std::all_of(element.corners.begin(), element.corners.end(), [&](std::size_t g) { return g < gridCount; });
        if (!inRange || !allDistinct(element.corners))
        {
            addProblem(is, Severity::Error, "its corner grid points are not distinct grid points of the model");
            continue;
        }
        const std::size_t edgeCount = shapeEdges(*shape).size();
        if (!element.midSideGrids.empty() && element.midSideGrids.size() != edgeCount)
        {
            addProblem(is, Severity::Error,
                       std::to_string(element.midSideGrids.size()) + " mid-side grid points for a " +
                           shapeName(*shape) + " of " + std::to_string(edgeCount) + " edges");
            continue;
        }
        const bool midSideInRange = std::all_of(element.midSideGrids.begin(), element.midSideGrids.end(),
                                                [&](std::size_t g) { return g == NO_MID_SIDE_GRID || g < gridCount; });
        if (!midSideInRange)
        {
            addProblem(is, Severity::Error, "a mid-side grid point is not a grid point of the model");
            continue;
        }
        m_usable[is] = true;
        m_shapes[is] = *shape;
        m_basicMaterials[is] = materialInBasicAxes(element.material, element.materialAxes);
        m_byElementId.push_back(is);
    }
    std::stable_sort(m_byElementId.begin(), m_byElementId.end(), [&](std::size_t a, std::size_t b)
                     { return m_input.solids[a].elementId < m_input.solids[b].elementId; });
}

void SolidPart::buildTopology()
{
    // the edges and faces met, by key, with the solids that have them in increasing element ID
    struct FaceMet
    {
        std::vector<std::size_t> cycle;
        std::vector<std::size_t> solids;
    };
    std::map<EdgeKey, std::vector<std::size_t>> edgeSolids;
    std::map<std::vector<std::size_t>, FaceMet> faceMet;
    for (const std::size_t is : m_byElementId)
    {
        const std::vector<std::size_t>& corners = m_input.solids[is].corners;
        for (const ShapeEdge& e : shapeEdges(m_shapes[is]))
            edgeSolids[EdgeKey::of(corners[e[0]], corners[e[1]])].push_back(is);
        for (const ShapeFace& f : shapeFaces(m_shapes[is]))
        {
            std::vector<std::size_t> cycle;
            for (const std::size_t k : f.cornerSpan())
                cycle.push_back(corners[k]);
            FaceMet& met = faceMet[sortedCopy(cycle)];
            if (met.solids.empty())
                met.cycle = cycle;
            met.solids.push_back(is);
        }
    }
    for (auto& [key, solids] : edgeSolids)
    {
        m_edgeIndex.emplace(key, m_edges.size());
        SolidPartEdge edge;
        edge.grids = key;
        edge.solids = std::move(solids);
        m_edges.push_back(std::move(edge));
    }
    for (auto& [key, met] : faceMet)
    {
        const std::size_t index = m_faces.size();
        m_faceIndex.emplace(key, index);
        // any three grids of a face give it
        for (std::size_t a = 0; a < key.size(); ++a)
            for (std::size_t b = a + 1; b < key.size(); ++b)
                for (std::size_t c = b + 1; c < key.size(); ++c)
                    m_faceByThree[{key[a], key[b], key[c]}] = index;
        SolidPartFace face;
        face.count = key.size();
        face.grids = canonicalCycle(met.cycle);
        face.solids = std::move(met.solids);
        m_faces.push_back(std::move(face));
    }
    for (const SolidPartFace& face : m_faces)
    {
        if (!face.onSkin())
            continue;
        for (std::size_t k = 0; k < face.count; ++k)
            m_edges[m_edgeIndex.at(EdgeKey::of(face.grids[k], face.grids[(k + 1) % face.count]))].onSkin = true;
    }
    for (const std::size_t is : m_byElementId)
    {
        const std::vector<std::size_t>& corners = m_input.solids[is].corners;
        for (const ShapeEdge& e : shapeEdges(m_shapes[is]))
            m_solidEdges[is].push_back(m_edgeIndex.at(EdgeKey::of(corners[e[0]], corners[e[1]])));
        for (const ShapeFace& f : shapeFaces(m_shapes[is]))
        {
            std::vector<std::size_t> cycle;
            for (const std::size_t k : f.cornerSpan())
                cycle.push_back(corners[k]);
            m_solidFaces[is].push_back(m_faceIndex.at(sortedCopy(cycle)));
        }
    }
}

void SolidPart::buildGeometry()
{
    const std::size_t perEdge = edgeFunctionCount(m_traceDegree);
    for (SolidPartEdge& edge : m_edges)
        edge.geometry.assign(perEdge, SolidPoint{});
    for (SolidPartFace& face : m_faces)
        face.geometry.assign(faceFunctionCount(m_faceDegree, face.count), SolidPoint{});

    // In the order of the reference (src/tdnns/solid.rs::build): the mid-side grids of every edge, the skin read from
    // the mesh where nothing is followed, the parabolas of the mid-side grids, then what follows a shell or a line
    // takes its geometry from it (followTies), over the rules before it.

    // rule 3 of document 35: the mid-side grids, the first solid by element ID giving the grid of an edge (an edge
    // that follows a shell or a line has its mid-side grid too: the grid follows the edge, midSideTerms)
    const std::span<const SolidPoint> x = m_input.gridPositions;
    using Severity = SolidPartProblem::Severity;
    std::set<std::size_t> cornerGrids;
    for (const std::size_t is : m_byElementId)
        cornerGrids.insert(m_input.solids[is].corners.begin(), m_input.solids[is].corners.end());
    for (const std::size_t is : m_byElementId)
    {
        const SolidPartElement& element = m_input.solids[is];
        for (std::size_t le = 0; le < element.midSideGrids.size(); ++le)
        {
            const std::size_t mid = element.midSideGrids[le];
            if (mid == NO_MID_SIDE_GRID)
                continue;
            SolidPartEdge& edge = m_edges[m_solidEdges[is][le]];
            const auto gridId = [&](std::size_t g)
            { return std::to_string(g < m_input.gridIds.size() ? m_input.gridIds[g] : static_cast<std::int32_t>(g)); };
            const std::string edgeName =
                "the edge between grid points " + gridId(edge.grids.lower) + " and " + gridId(edge.grids.higher);
            if (edge.midSideGrid != NO_MID_SIDE_GRID)
            {
                if (edge.midSideGrid != mid)
                    addProblem(is, Severity::Warning,
                               edgeName + " has the mid-side grid point " + gridId(mid) + " here and " +
                                   gridId(edge.midSideGrid) + " in an element of lower ID: that one is kept");
                continue;
            }
            edge.midSideGrid = mid;
            // a corner carries unknowns of its own, a mid-side grid follows its edge: it cannot be both
            if (cornerGrids.contains(mid))
                addProblem(is, Severity::Error,
                           "the mid-side grid point " + gridId(mid) + " of " + edgeName + " is a corner of a solid");
            const SolidPoint chord = sub3(x[edge.grids.higher], x[edge.grids.lower]);
            const SolidPoint middle = scl3(add3(x[edge.grids.lower], x[edge.grids.higher]), 0.5);
            const double shift = dot3(sub3(x[mid], middle), chord) / dot3(chord, chord);
            if (std::fabs(shift) >= MID_SIDE_FOLDING_SHIFT)
                addProblem(is, Severity::Error,
                           "the mid-side grid point " + gridId(mid) + " of " + edgeName +
                               " is a quarter of the edge or more from its middle: the edge folds");
        }
    }

    // rule 4: the rest of the skin read from the mesh; the interior edges and faces stay straight. The faces are
    // projected before the parabolas are set, as the reference does (read_skin, then the mid-side grids): a face of
    // the skin that has a mid-side edge among straight ones takes the curve of the skin reading as if that edge were
    // straight, and the parabola of the edge adds to it.
    const EdgeProjection projection = EdgeProjection::of(m_traceDegree);
    if (takesCurves())
        readSkin(projection);

    for (SolidPartEdge& edge : m_edges)
    {
        if (edge.midSideGrid == NO_MID_SIDE_GRID || perEdge == 0)
            continue;
        // the parabola through the mid-side grid: 4 (X_m − (X_a + X_b)/2) on s (1 − s), which is 1/4 at s = 1/2
        const SolidPoint middle = scl3(add3(x[edge.grids.lower], x[edge.grids.higher]), 0.5);
        edge.geometry.assign(perEdge, SolidPoint{});
        edge.geometry[0] = scl3(sub3(x[edge.midSideGrid], middle), 4.0);
    }

    // rule 0: what follows a shell or a line (SolidPartTies.cpp)
    followTies(projection);
}

void SolidPart::readSkin(const EdgeProjection& projection)
{
    const std::span<const SolidPoint> x = m_input.gridPositions;
    // the solids as the reading of the skin takes them, in increasing element ID
    std::vector<SolidCell> cells;
    for (const std::size_t is : m_byElementId)
    {
        SolidCell cell;
        cell.corners = m_input.solids[is].corners;
        for (const ShapeFace& f : shapeFaces(m_shapes[is]))
            cell.faces.emplace_back(f.cornerSpan().begin(), f.cornerSpan().end());
        cells.push_back(std::move(cell));
    }
    const std::vector<SkinFace> skin = skinFaces(x, cells);
    if (skin.empty())
        return;
    const SkinReading reading = readSolidSkin(x, skin, m_input.sharpEdges);

    // each edge: its conic (the mean of its readings) projected on the functions of the edge
    for (const auto& [key, conic] : reading.conics)
    {
        const std::optional<std::size_t> e = edgeIndex(key);
        if (!e)
            continue;
        SolidPartEdge& edge = m_edges[*e];
        if (edge.follows || edge.midSideGrid != NO_MID_SIDE_GRID)
            continue;
        const SolidPoint& xa = x[key.lower];
        const SolidPoint& xb = x[key.higher];
        // the control point of the conic: the middle of the chord moved by twice the bulge
        const SolidPoint control = add3(scl3(add3(xa, xb), 0.5), scl3(conic.bulge, 2.0));
        std::vector<SolidPoint> values;
        for (const double t : projection.parameters)
            values.push_back(conicPoint(xa, control, xb, conic.weight, t));
        edge.geometry = projection.project(values, xa, xb);
    }

    // each face: what its patch adds to its grids and to its edges (as read here: an edge with a mid-side grid is
    // still straight), on the functions of the face
    for (std::size_t it = 0; it < skin.size(); ++it)
    {
        const SkinFace& skinFace = skin[it];
        const std::size_t f = *faceIndex(skinFace.view());
        SolidPartFace& face = m_faces[f];
        const std::size_t nf = face.count;
        std::vector<std::size_t> faceEdges;
        bool allMidSide = true;
        for (std::size_t k = 0; k < nf; ++k)
        {
            faceEdges.push_back(m_edgeIndex.at(EdgeKey::of(face.grids[k], face.grids[(k + 1) % nf])));
            allMidSide = allMidSide && m_edges[faceEdges.back()].midSideGrid != NO_MID_SIDE_GRID;
        }
        // a face all of whose edges have a mid-side grid keeps no curve of its own (rule 3)
        if (face.follows || face.geometry.empty() || allMidSide)
            continue;
        const Patch patch = skinFacePatch(reading, it);
        const std::span<const std::size_t> g = face.view();
        const std::span<const std::size_t> t = skinFace.view();
        // pos[i]: where grid i of the face (its canonical order) stands in the order of the skin face
        std::vector<std::size_t> pos(nf);
        for (std::size_t i = 0; i < nf; ++i)
            pos[i] = static_cast<std::size_t>(std::find(t.begin(), t.end(), g[i]) - t.begin());
        std::vector<base::EntityId> order;
        std::vector<SolidPoint> corners;
        std::vector<std::vector<SolidPoint>> edgeCoefficients;
        for (std::size_t i = 0; i < nf; ++i)
        {
            order.push_back(static_cast<base::EntityId>(g[i]));
            corners.push_back(x[g[i]]);
            edgeCoefficients.push_back(m_edges[faceEdges[i]].geometry);
        }
        const FaceProjection fp = FaceProjection::of(order, m_traceDegree, m_faceDegree);
        std::vector<SolidPoint> values;
        for (const std::array<double, 2>& c : fp.points)
        {
            // the same point in the order of the skin face: its weights there, then its coordinates on the patch
            const std::vector<double> w = faceWeights(nf, c);
            std::vector<double> wt(nf);
            for (std::size_t i = 0; i < nf; ++i)
                wt[pos[i]] = w[i];
            if (nf == 3)
            {
                // the rational triangle at (u, v) = (2 L1 − 1, 2 L2 − 1)
                values.push_back(patch.evaluate(2.0 * wt[1] - 1.0, 2.0 * wt[2] - 1.0));
                continue;
            }
            const std::array<double, 2> ct{wt[1] + wt[2], wt[2] + wt[3]};
            const std::vector<double> w4 = faceWeights(4, ct);
            double xi = 0.0;
            double eta = 0.0;
            for (std::size_t k = 0; k < 4; ++k)
            {
                const ReferencePoint corner = referenceCorner(4, k);
                xi += w4[k] * corner[0];
                eta += w4[k] * corner[1];
            }
            values.push_back(patch.evaluate(xi, eta));
        }
        face.geometry = fp.project(values, corners, edgeCoefficients);
    }
}

void SolidPart::buildEntities()
{
    const std::span<const SolidPoint> x = m_input.gridPositions;
    m_edgeEntities.assign(m_edges.size(), std::nullopt);
    m_faceEntities.assign(m_faces.size(), std::nullopt);
    m_interiorEntities.assign(m_input.solids.size(), std::nullopt);
    for (std::size_t e = 0; e < m_edges.size(); ++e)
    {
        const SolidPartEdge& edge = m_edges[e];
        if (edge.geometry.empty())
            continue;
        m_edgeEntities[e] = m_entities.size();
        const std::array<std::size_t, 2> grids{edge.grids.lower, edge.grids.higher};
        m_entities.push_back({SolidPartEntity::Kind::Edge, e, COMPONENTS * edge.geometry.size(), meanOf(x, grids)});
    }
    for (std::size_t f = 0; f < m_faces.size(); ++f)
    {
        const SolidPartFace& face = m_faces[f];
        if (face.geometry.empty())
            continue;
        m_faceEntities[f] = m_entities.size();
        m_entities.push_back(
            {SolidPartEntity::Kind::Face, f, COMPONENTS * face.geometry.size(), meanOf(x, face.view())});
    }
    m_staticEntityCount = m_entities.size();
    // the interior fields of a modal analysis: the kernel has them for a solid of positive density only
    const std::size_t fields = m_input.options.interiorFieldCount;
    for (const std::size_t is : m_byElementId)
    {
        if (fields == 0 || !(m_basicMaterials[is].density > 0.0))
            continue;
        m_interiorEntities[is] = m_entities.size();
        m_entities.push_back({SolidPartEntity::Kind::Interior, is, fields, meanOf(x, m_input.solids[is].corners)});
    }
    m_dofOffsets.assign(m_entities.size() + 1, 0);
    for (std::size_t e = 0; e < m_entities.size(); ++e)
        m_dofOffsets[e + 1] = m_dofOffsets[e] + m_entities[e].dofCount;
}

// ---- lookups

std::optional<std::size_t> SolidPart::edgeIndex(const EdgeKey& key) const
{
    const auto found = m_edgeIndex.find(key);
    if (found == m_edgeIndex.end())
        return std::nullopt;
    return found->second;
}

std::optional<std::size_t> SolidPart::faceIndex(std::span<const std::size_t> grids) const
{
    if (grids.size() < 3)
        return std::nullopt;
    std::vector<std::size_t> key = sortedCopy(grids);
    if (key.size() == 4)
    {
        const auto found = m_faceIndex.find(key);
        return found == m_faceIndex.end() ? std::nullopt : std::optional<std::size_t>(found->second);
    }
    const auto found = m_faceByThree.find({key[0], key[1], key[2]});
    if (found == m_faceByThree.end())
        return std::nullopt;
    return found->second;
}

std::span<const SolidPartEntity> SolidPart::entities(EntityLayout::Kind kind) const
{
    const std::size_t count = kind == EntityLayout::Kind::Modal ? m_entities.size() : m_staticEntityCount;
    return {m_entities.data(), count};
}

std::optional<std::size_t> SolidPart::edgeEntity(std::size_t edge) const
{
    return m_edgeEntities[edge];
}

std::optional<std::size_t> SolidPart::faceEntity(std::size_t face) const
{
    return m_faceEntities[face];
}

std::optional<std::size_t> SolidPart::interiorEntity(std::size_t solid) const
{
    return m_interiorEntities[solid];
}

std::vector<std::pair<std::size_t, std::size_t>> SolidPart::midSideGrids() const
{
    std::vector<std::pair<std::size_t, std::size_t>> out;
    for (std::size_t e = 0; e < m_edges.size(); ++e)
        if (m_edges[e].midSideGrid != NO_MID_SIDE_GRID)
            out.emplace_back(m_edges[e].midSideGrid, e);
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<SolidTerm> SolidPart::midSideTerms(std::size_t edge, std::size_t component) const
{
    // the grid is the point s = 1/2 of the parabola of its edge (VolumeGeometry::setMidSideNode)
    constexpr double MIDDLE = 0.5;
    const SolidPartEdge& e = m_edges[edge];
    std::vector<SolidTerm> out{{{true, e.grids.lower, component}, 1.0 - MIDDLE},
                               {{true, e.grids.higher, component}, MIDDLE}};
    if (!m_edgeEntities[edge])
        return out;
    const std::vector<double> f = edgeFunctions(m_traceDegree, MIDDLE);
    for (std::size_t m = 0; m < f.size(); ++m)
        if (f[m] != 0.0)
            out.push_back({{false, *m_edgeEntities[edge], COMPONENTS * m + component}, f[m]});
    return out;
}

std::vector<base::EntityId> SolidPart::orderOf(std::size_t solid) const
{
    // the grid indices increase with the grid IDs: they orient the edges and faces as the IDs would
    std::vector<base::EntityId> out;
    for (const std::size_t g : m_input.solids[solid].corners)
        out.push_back(static_cast<base::EntityId>(g));
    return out;
}

std::vector<SolidLocalUnknown> SolidPart::localUnknowns(std::size_t solid, EntityLayout::Kind kind) const
{
    OOS_ASSERT(m_usable[solid]);
    std::vector<SolidLocalUnknown> out;
    for (const std::size_t g : m_input.solids[solid].corners)
        for (std::size_t c = 0; c < COMPONENTS; ++c)
            out.push_back({true, g, c});
    const auto addEntity = [&](const std::optional<std::size_t>& entity)
    {
        if (!entity)
            return;
        for (std::size_t i = 0; i < m_entities[*entity].dofCount; ++i)
            out.push_back({false, *entity, i});
    };
    for (const std::size_t e : m_solidEdges[solid])
        addEntity(m_edgeEntities[e]);
    for (const std::size_t f : m_solidFaces[solid])
        addEntity(m_faceEntities[f]);
    if (kind == EntityLayout::Kind::Modal)
        addEntity(m_interiorEntities[solid]);
    return out;
}

// ---- kernels

SolidResult<VolumeGeometry> SolidPart::geometryOf(std::size_t solid) const
{
    OOS_ASSERT(m_usable[solid]);
    std::vector<SolidPoint> positions;
    for (const std::size_t g : m_input.solids[solid].corners)
        positions.push_back(m_input.gridPositions[g]);
    const std::vector<base::EntityId> order = orderOf(solid);
    SolidResult<VolumeGeometry> built =
        VolumeGeometry::straight(m_shapes[solid], std::move(positions), order, m_traceDegree, m_faceDegree);
    if (!built.ok())
        return built;
    VolumeGeometry& geometry = built.value();
    for (std::size_t le = 0; le < m_solidEdges[solid].size(); ++le)
        geometry.edges[le] = m_edges[m_solidEdges[solid][le]].geometry;
    for (std::size_t lf = 0; lf < m_solidFaces[solid].size(); ++lf)
        geometry.faces[lf] = m_faces[m_solidFaces[solid][lf]].geometry;
    return built;
}

SolidResult<VolumeKernel> SolidPart::buildKernel(std::size_t solid, EntityLayout::Kind kind, bool withMass,
                                                 std::span<const SolidLoad> loads) const
{
    SolidResult<VolumeGeometry> geometry = geometryOf(solid);
    if (!geometry.ok())
        return SolidResult<VolumeKernel>::refused(geometry.refusal());
    SolidOptions options = m_input.options;
    const bool interior = kind == EntityLayout::Kind::Modal && m_interiorEntities[solid].has_value();
    if (!interior)
        options.interiorFieldCount = 0;
    const std::vector<base::EntityId> order = orderOf(solid);
    return VolumeKernel::build(geometry.value(), order, m_basicMaterials[solid], options, withMass || interior, loads);
}

std::variant<SolidPart::SharedKernel, std::string> SolidPart::kernelOf(std::size_t solid, EntityLayout::Kind kind,
                                                                       bool withMass) const
{
    const bool interior = kind == EntityLayout::Kind::Modal && m_interiorEntities[solid].has_value();
    const bool mass = withMass || interior;
    KernelCache& cache = *m_kernelCache;
    const auto serves = [&](std::size_t s)
    {
        return s < cache.kernels.size() && cache.kernels[s] && (!mass || cache.hasMass[s]) &&
               (!interior || cache.hasInterior[s]);
    };
    {
        const std::lock_guard<std::mutex> lock(cache.mutex);
        if (serves(solid))
            return cache.kernels[solid];
    }
    SolidResult<VolumeKernel> built = buildKernel(solid, kind, withMass);
    if (!built.ok())
        return built.refusal();
    SharedKernel kernel = std::make_shared<const VolumeKernel>(std::move(built.value()));
    if (m_input.kernelCacheBytes == 0)
        return kernel;
    // The bytes of the kernel: its matrices (the rest is small).
    const auto bytesOf = [](const Matrix& m) { return m.rows() * m.columns() * sizeof(double); };
    std::size_t bytes = bytesOf(kernel->stiffness()) + bytesOf(kernel->stressOperator()) +
                        bytesOf(kernel->bubbleOperator()) + bytesOf(kernel->geometry().functionCoefficients);
    if (kernel->mass())
        bytes += bytesOf(*kernel->mass());
    if (kernel->interiorFields())
        bytes += bytesOf(kernel->interiorFields()->massCoupling);
    for (const SolidLoadResult& load : kernel->loads())
        bytes += (load.force.size() + load.particularStresses.size()) * sizeof(double);
    const std::lock_guard<std::mutex> lock(cache.mutex);
    if (cache.kernels.size() < m_input.solids.size())
    {
        cache.kernels.resize(m_input.solids.size());
        cache.hasMass.resize(m_input.solids.size(), false);
        cache.hasInterior.resize(m_input.solids.size(), false);
        cache.entryBytes.resize(m_input.solids.size(), 0);
    }
    if (serves(solid))
        return kernel; // another thread kept one meanwhile
    // A kernel kept with less gives its place to this one.
    const std::size_t total = cache.bytes - cache.entryBytes[solid] + bytes;
    if (total <= m_input.kernelCacheBytes)
    {
        cache.kernels[solid] = kernel;
        cache.hasMass[solid] = kernel->mass().has_value();
        cache.hasInterior[solid] = kernel->interiorFields().has_value();
        cache.entryBytes[solid] = bytes;
        cache.bytes = total;
    }
    return kernel;
}

std::optional<SolidPartProblem> SolidPart::validate(std::size_t solid) const
{
    const std::variant<SharedKernel, std::string> kernel = kernelOf(solid, EntityLayout::Kind::Static, false);
    if (const std::string* refusal = std::get_if<std::string>(&kernel))
        return SolidPartProblem{SolidPartProblem::Severity::Error, solid, m_input.solids[solid].elementId, *refusal};
    return std::nullopt;
}

Matrix SolidPart::localStiffness(std::size_t solid, const VolumeKernel& kernel, EntityLayout::Kind kind) const
{
    const std::size_t nq = kernel.parameterCount();
    const bool interior = kind == EntityLayout::Kind::Modal && m_interiorEntities[solid].has_value();
    if (!interior)
        return kernel.stiffness();
    if (!kernel.interiorFields())
        throw base::InternalError("the kernel of a solid of a modal analysis was built without its interior fields");
    const Vector& stiffnesses = kernel.interiorFields()->stiffnesses;
    Matrix out(nq + stiffnesses.size(), nq + stiffnesses.size());
    out.setBlock(0, 0, kernel.stiffness());
    for (std::size_t i = 0; i < stiffnesses.size(); ++i)
        out(nq + i, nq + i) = stiffnesses[i];
    return out;
}

Matrix SolidPart::localMass(std::size_t solid, const VolumeKernel& kernel, EntityLayout::Kind kind) const
{
    if (!kernel.mass())
        throw base::InternalError("the kernel of a solid was built without its mass");
    const std::size_t nq = kernel.parameterCount();
    const bool interior = kind == EntityLayout::Kind::Modal && m_interiorEntities[solid].has_value();
    if (!interior)
        return *kernel.mass();
    if (!kernel.interiorFields())
        throw base::InternalError("the kernel of a solid of a modal analysis was built without its interior fields");
    const Matrix& coupling = kernel.interiorFields()->massCoupling;
    const std::size_t fields = coupling.columns();
    Matrix out(nq + fields, nq + fields);
    out.setBlock(0, 0, *kernel.mass());
    for (std::size_t i = 0; i < fields; ++i)
    {
        // the interior fields are orthonormal for the mass, coupled to the parameters by the mass only
        out(nq + i, nq + i) = 1.0;
        for (std::size_t q = 0; q < nq; ++q)
        {
            out(q, nq + i) = coupling(q, i);
            out(nq + i, q) = coupling(q, i);
        }
    }
    return out;
}

// ---- loads

SolidLoad SolidPart::accelerationLoad(std::size_t solid, const SolidPoint& acceleration) const
{
    return SolidLoad::uniformAcceleration(m_basicMaterials[solid].density, acceleration);
}

SolidLoad SolidPart::thermalLoad(std::size_t solid, TemperatureField temperatureRise) const
{
    const double alpha = m_basicMaterials[solid].thermalExpansion;
    return SolidLoad::thermal([alpha, field = std::move(temperatureRise)](const SolidPoint& x)
                              { return alpha * field(x); });
}

SolidLoad SolidPart::rigidInertiaLoad(std::size_t solid, std::size_t mode) const
{
    return SolidLoad::rigidInertia(m_basicMaterials[solid].density, mode);
}

std::optional<std::size_t> SolidPart::localFaceOf(std::size_t solid, std::span<const std::size_t> faceGrids) const
{
    const std::vector<std::size_t>& corners = m_input.solids[solid].corners;
    const std::span<const ShapeFace> faces = shapeFaces(m_shapes[solid]);
    for (std::size_t lf = 0; lf < faces.size(); ++lf)
    {
        const std::span<const std::size_t> own = faces[lf].cornerSpan();
        const bool hasAll = std::all_of(
            faceGrids.begin(), faceGrids.end(), [&](std::size_t g)
            { return std::any_of(own.begin(), own.end(), [&](std::size_t k) { return corners[k] == g; }); });
        if (hasAll)
            return lf;
    }
    return std::nullopt;
}

std::optional<std::pair<std::size_t, std::size_t>> SolidPart::solidFaceOf(std::span<const std::size_t> grids) const
{
    const std::optional<std::size_t> f = faceIndex(grids);
    if (!f)
        return std::nullopt;
    // the solids of a face are in increasing element ID
    const std::size_t solid = m_faces[*f].solids.front();
    const std::optional<std::size_t> local = localFaceOf(solid, m_faces[*f].view());
    if (!local)
        return std::nullopt;
    return std::pair{solid, *local};
}

Vector SolidPart::faceTractionLoad(const VolumeKernel& kernel, std::size_t localFace,
                                   const VolumeKernel::TractionField& traction) const
{
    return kernel.faceTractionLoad(localFace, traction, m_traceDegree + FACE_LOAD_EXTRA_POINTS);
}

Vector SolidPart::faceStressLoad(const VolumeKernel& kernel, std::size_t localFace,
                                 const VolumeKernel::StressField& stress) const
{
    return kernel.faceStressLoad(localFace, stress, m_traceDegree + FACE_LOAD_EXTRA_POINTS);
}

Vector SolidPart::facePressureLoad(const VolumeKernel& kernel, std::size_t localFace,
                                   const VolumeKernel::PressureField& pressure) const
{
    return kernel.facePressureLoad(localFace, pressure, m_traceDegree + FACE_LOAD_EXTRA_POINTS);
}

Vector SolidPart::faceCornerPressureLoad(const VolumeKernel& kernel, std::size_t solid, std::size_t localFace,
                                         std::span<const std::size_t> cornerGrids,
                                         std::span<const double> cornerPressures) const
{
    OOS_ASSERT(cornerGrids.size() == cornerPressures.size());
    const ShapeFace& face = shapeFaces(m_shapes[solid])[localFace];
    const std::vector<std::size_t>& corners = m_input.solids[solid].corners;
    // the pressure at each corner of the face, in the order of the face of the shape
    std::vector<double> atCorner;
    for (const std::size_t k : face.cornerSpan())
    {
        const auto found = std::find(cornerGrids.begin(), cornerGrids.end(), corners[k]);
        OOS_ASSERT(found != cornerGrids.end());
        atCorner.push_back(cornerPressures[static_cast<std::size_t>(found - cornerGrids.begin())]);
    }
    const std::size_t n = m_traceDegree + FACE_LOAD_EXTRA_POINTS;
    // facePointsOf follows the points of faceRule in their order
    const std::vector<SurfaceRulePoint> rule = faceRule(face.cornerCount, n);
    const std::vector<VolumeFacePoint> points = kernel.facePointsOf(localFace, n);
    Vector out(kernel.parameterCount(), 0.0);
    for (std::size_t q = 0; q < points.size(); ++q)
    {
        const VolumeFacePoint& point = points[q];
        const std::vector<double> w = faceWeights(face.cornerCount, rule[q].point);
        double pressure = 0.0;
        for (std::size_t k = 0; k < face.cornerCount; ++k)
            pressure += w[k] * atCorner[k];
        // the traction −p n dA, the area vector a1 × a2 times the weight, outward
        const SolidPoint area = scl3(cross3(point.tangentS, point.tangentT), point.weight * point.outward);
        const SolidPoint traction = scl3(area, -pressure);
        const std::vector<double> values = kernel.functionsAt(point.reference);
        for (std::size_t j = 0; j < values.size(); ++j)
        {
            if (values[j] == 0.0)
                continue;
            for (std::size_t c = 0; c < COMPONENTS; ++c)
                out[COMPONENTS * j + c] += values[j] * traction[c];
        }
    }
    return out;
}

Matrix SolidPart::followerPressureStiffness(const VolumeKernel& kernel, std::size_t localFace,
                                            const VolumeKernel::PressureField& pressure) const
{
    return kernel.followerPressureStiffness(localFace, pressure, m_traceDegree + FACE_LOAD_EXTRA_POINTS);
}

// ---- stresses

Vector SolidPart::stressCoefficients(const VolumeKernel& kernel, std::span<const double> local,
                                     std::span<const double> loadFactors)
{
    OOS_ASSERT(local.size() >= kernel.parameterCount() && loadFactors.size() <= kernel.loads().size());
    Vector beta = kernel.stressOperator() * local.first(kernel.parameterCount());
    for (std::size_t l = 0; l < loadFactors.size(); ++l)
    {
        if (loadFactors[l] == 0.0)
            continue;
        const Vector& particular = kernel.loads()[l].particularStresses;
        for (std::size_t i = 0; i < beta.size(); ++i)
            beta[i] += loadFactors[l] * particular[i];
    }
    return beta;
}

SolidPointStresses SolidPart::pointStresses(const VolumeKernel& kernel, std::span<const double> beta)
{
    const VolumeShape shape = kernel.geometry().shape;
    SolidPointStresses out;
    out.centre = kernel.stressAt(beta, shapeCentre(shape));
    for (const SolidPoint& vertex : shapeVertices(shape))
        out.corners.push_back(kernel.stressAt(beta, vertex));
    return out;
}

VoigtVector SolidPart::toMaterialAxes(std::size_t solid, const VoigtVector& basic) const
{
    // the tensor of the Voigt components (σxx, σyy, σzz, τxy, τyz, τzx)
    const std::array<std::array<double, 3>, 3> s{
        {{basic[0], basic[3], basic[5]}, {basic[3], basic[1], basic[4]}, {basic[5], basic[4], basic[2]}}};
    const base::Mat3& a = m_input.solids[solid].materialAxes;
    std::array<std::array<double, 3>, 3> local{};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
        {
            double sum = 0.0;
            for (std::size_t k = 0; k < 3; ++k)
                for (std::size_t l = 0; l < 3; ++l)
                    sum += a(k, i) * s[k][l] * a(l, j);
            local[i][j] = sum;
        }
    return {local[0][0], local[1][1], local[2][2], local[0][1], local[1][2], local[0][2]};
}

std::vector<std::optional<VoigtVector>> SolidPart::gridStresses(
    std::span<const std::vector<VoigtVector>> cornerStresses) const
{
    const std::size_t gridCount = m_input.gridPositions.size();
    std::vector<VoigtVector> sum(gridCount, VoigtVector{});
    std::vector<std::size_t> count(gridCount, 0);
    for (std::size_t is = 0; is < cornerStresses.size() && is < m_input.solids.size(); ++is)
    {
        const std::vector<std::size_t>& corners = m_input.solids[is].corners;
        if (cornerStresses[is].empty())
            continue;
        OOS_ASSERT(cornerStresses[is].size() == corners.size());
        for (std::size_t k = 0; k < corners.size(); ++k)
        {
            for (std::size_t c = 0; c < STRESS_COMPONENTS; ++c)
                sum[corners[k]][c] += cornerStresses[is][k][c];
            ++count[corners[k]];
        }
    }
    std::vector<std::optional<VoigtVector>> out(gridCount);
    for (std::size_t g = 0; g < gridCount; ++g)
    {
        if (count[g] == 0)
            continue;
        VoigtVector mean{};
        for (std::size_t c = 0; c < STRESS_COMPONENTS; ++c)
            mean[c] = sum[g][c] / static_cast<double>(count[g]);
        out[g] = mean;
    }
    return out;
}

SolidResult<Matrix> SolidPart::geometricStiffness(const VolumeKernel& kernel, std::span<const double> beta)
{
    return kernel.geometric(beta);
}

// ---- affine fields

std::vector<double> SolidPart::affineField(const base::Mat3& gradient, EntityLayout::Kind kind) const
{
    const std::span<const SolidPartEntity> list = entities(kind);
    std::vector<double> out(m_dofOffsets[list.size()], 0.0);
    for (std::size_t e = 0; e < list.size(); ++e)
    {
        const SolidPartEntity& entity = list[e];
        if (entity.kind == SolidPartEntity::Kind::Interior)
            continue;
        const std::vector<SolidPoint>& geometry = entity.kind == SolidPartEntity::Kind::Edge
                                                      ? m_edges[entity.index].geometry
                                                      : m_faces[entity.index].geometry;
        // u = b + A X with X = Σ N_j X_j: function m carries A times its geometry coefficient
        for (std::size_t m = 0; m < geometry.size(); ++m)
            for (std::size_t c = 0; c < COMPONENTS; ++c)
                out[m_dofOffsets[e] + COMPONENTS * m + c] =
                    gradient(c, 0) * geometry[m][0] + gradient(c, 1) * geometry[m][1] + gradient(c, 2) * geometry[m][2];
    }
    return out;
}

} // namespace oos::elements::mixedHybrid
