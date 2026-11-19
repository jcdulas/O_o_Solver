#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/model/MixedHybridModel.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeShape.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

// The points of the elements of the formulation that the elements of other cards read: a fastener or a weld that
// pierces a shell, a glued contact whose primary face is a face of a shell or of a solid
// (docs/mixed-hybrid-formulation-spec.md section 5.10; not in the reference).

namespace oos::elements::mixedHybrid
{

namespace
{

// The terms of a row on one DOF each, summed, by increasing DOF (kind, row, component), without the zero ones.
void mergeTerms(std::vector<FormulationRelationTerm>& terms)
{
    std::map<std::tuple<int, std::int32_t, int>, std::pair<ElementDof, double>> sums;
    for (const FormulationRelationTerm& term : terms)
    {
        auto& [dof, sum] = sums[{static_cast<int>(term.dof.kind), term.dof.row, static_cast<int>(term.dof.component)}];
        dof = term.dof;
        sum += term.coefficient;
    }
    terms.clear();
    for (const auto& [key, sum] : sums)
        if (sum.second != 0.0)
            terms.push_back({sum.first, sum.second});
}

// The reference point of a shell at the point given by the shape functions `shapeValues` of grid points of its card
// (node table rows `gridRows`): Σ N_j r_j / Σ N_j, r_j the reference corner of a corner grid point, the middle of the
// edge of a mid-side grid point. The shape functions of a card reproduce the linear functions of its parameters,
// those of the reference element of the shell: on a flat element of straight sides it is the point itself. Empty
// when a grid point is not one of the shell's.
std::optional<ReferencePoint> shellReferencePoint(const MixedShell& shell, std::span<const std::int32_t> gridRows,
                                                  std::span<const double> shapeValues)
{
    if (gridRows.empty() || gridRows.size() != shapeValues.size())
        return std::nullopt;
    ReferencePoint point{};
    double total = 0.0;
    for (std::size_t j = 0; j < gridRows.size(); ++j)
    {
        if (gridRows[j] < 0)
            continue; // an absent mid-side point of a face
        std::optional<ReferencePoint> at;
        for (std::size_t k = 0; k < shell.cornerCount && !at; ++k)
        {
            if (shell.gridRows[k] == gridRows[j])
                at = referenceCorner(shell.cornerCount, k);
            else if (shell.midRows[k] >= 0 && shell.midRows[k] == gridRows[j])
                at = referenceEdgePointAndDirection(shell.cornerCount, k, 0.5).first;
        }
        if (!at)
            return std::nullopt;
        point[0] += shapeValues[j] * (*at)[0];
        point[1] += shapeValues[j] * (*at)[1];
        total += shapeValues[j];
    }
    if (!(total > 0.0))
        return std::nullopt;
    return ReferencePoint{point[0] / total, point[1] / total};
}

} // namespace

std::optional<FormulationModel::PointMotion> MixedHybridModel::shellPointMotion(
    std::int32_t elementId, const base::Vec3& point, std::span<const ShellSample> samples) const
{
    if (m_shellPart == nullptr || !m_shellPart->usable() || m_shellOfId.find(elementId) == m_shellOfId.end())
        return std::nullopt;

    // The mean over the samples of the motion of the shell carried rigidly to the piercing point P,
    //   u_P = mean(u + θ × (P − X)),   θ_P = mean(θ),
    // X the point of the surface of a sample: exact for a rigid motion whatever the samples.
    PointMotion motion;
    double total = 0.0;
    for (const ShellSample& sample : samples)
    {
        const auto found = m_shellOfId.find(sample.elementId);
        if (found == m_shellOfId.end())
            continue;
        const MixedShell& shell = m_shells[found->second];
        const std::optional<ReferencePoint> at = shellReferencePoint(shell, sample.gridRows, sample.shapeValues);
        if (!at)
            continue;
        const ShellPart::PointRows rows = m_shellPart->pointRows(shell.partIndex, (*at)[0], (*at)[1]);
        for (std::size_t r = 0; r < 6; ++r)
            appendPartTerms(rows.rows[r], sample.weight, motion.rows[r]);
        const std::array<double, 3> d{point[0] - rows.point[0], point[1] - rows.point[1], point[2] - rows.point[2]};
        for (std::size_t c = 0; c < 3; ++c)
        {
            const std::size_t next = (c + 1) % 3;
            const std::size_t last = (c + 2) % 3;
            // (θ × d)_c = θ_next d_last − θ_last d_next
            if (d[last] != 0.0)
                appendPartTerms(rows.rows[3 + next], sample.weight * d[last], motion.rows[c]);
            if (d[next] != 0.0)
                appendPartTerms(rows.rows[3 + last], -sample.weight * d[next], motion.rows[c]);
        }
        total += sample.weight;
    }
    if (!(total > 0.0))
        return std::nullopt;
    for (std::vector<FormulationRelationTerm>& row : motion.rows)
    {
        mergeTerms(row);
        for (FormulationRelationTerm& term : row)
            term.coefficient /= total;
    }
    return motion;
}

std::optional<FormulationModel::PointMotion> MixedHybridModel::facePointMotion(
    std::span<const std::int32_t> faceGridRows, std::span<const double> shapeValues) const
{
    const std::size_t count = faceGridRows.size();
    if ((count != 3 && count != 4 && count != 6 && count != 8) || shapeValues.size() != count)
        return std::nullopt;
    const std::size_t corners = count == 3 || count == 6 ? 3 : 4;
    const std::optional<FormulationFace> face = faceOf(faceGridRows.first(corners));
    if (!face)
        return std::nullopt;
    PointMotion motion;
    if (const MixedShell* shell = shellOf(face->card, face->batchRow))
    {
        const std::optional<ReferencePoint> at = shellReferencePoint(*shell, faceGridRows, shapeValues);
        if (!at)
            return std::nullopt;
        // the displacement and the rotation of the shell there: a point glued off the surface follows the rotation
        // (on an edge of the shell, the rotation about the edge is that of the edge)
        const ShellPart::PointRows rows = m_shellPart->pointRowsOnEdges(shell->partIndex, (*at)[0], (*at)[1]);
        for (std::size_t r = 0; r < 6; ++r)
            appendPartTerms(rows.rows[r], 1.0, motion.rows[r]);
        motion.point = base::Vec3{rows.point[0], rows.point[1], rows.point[2]};
    }
    else if (const MixedSolid* solid = solidOf(face->card, face->batchRow))
    {
        // The point of the reference shape: Σ N_j r_j / Σ N_j over the vertices of the corners and the middles of the
        // edges of the mid-side points.
        const VolumeShape shape = m_solidPart->shapeOf(solid->partIndex);
        const std::span<const SolidPoint> vertices = shapeVertices(shape);
        const std::span<const ShapeEdge> edges = shapeEdges(shape);
        SolidPoint r{};
        double total = 0.0;
        for (std::size_t j = 0; j < count; ++j)
        {
            if (faceGridRows[j] < 0)
                continue;
            std::optional<SolidPoint> at;
            const std::int32_t grid = gridIndexOfRow(faceGridRows[j]);
            for (std::size_t k = 0; k < solid->part.corners.size() && !at; ++k)
                if (grid >= 0 && solid->part.corners[k] == static_cast<std::size_t>(grid))
                    at = vertices[k];
            for (std::size_t e = 0; e < solid->midRows.size() && e < edges.size() && !at; ++e)
                if (solid->midRows[e] == faceGridRows[j])
                    at = scl3(add3(vertices[edges[e][0]], vertices[edges[e][1]]), 0.5);
            if (!at)
                return std::nullopt;
            r = add3(r, scl3(*at, shapeValues[j]));
            total += shapeValues[j];
        }
        if (!(total > 0.0))
            return std::nullopt;
        r = scl3(r, 1.0 / total);
        std::variant<SolidPart::SharedKernel, std::string> kernel =
            m_solidPart->kernelOf(solid->partIndex, EntityLayout::Kind::Static, false);
        if (!std::holds_alternative<SolidPart::SharedKernel>(kernel))
            return std::nullopt; // a refused solid: its family reports it
        const std::vector<double> functions = std::get<SolidPart::SharedKernel>(kernel)->functionsAt(r);
        const std::vector<SolidLocalUnknown> local =
            m_solidPart->localUnknowns(solid->partIndex, EntityLayout::Kind::Static);
        // u_c = Σ_j N_j q_{3j+c} (the interior fields of a modal analysis vanish on the faces)
        for (std::size_t c = 0; c < 3; ++c)
        {
            std::vector<SolidTerm> terms;
            for (std::size_t j = 0; j < functions.size() && 3 * j + c < local.size(); ++j)
                if (functions[j] != 0.0)
                    terms.push_back({local[3 * j + c], functions[j]});
            appendSolidTerms(terms, 1.0, motion.rows[c]);
        }
    }
    else
    {
        return std::nullopt;
    }
    for (std::vector<FormulationRelationTerm>& row : motion.rows)
        mergeTerms(row);
    return motion;
}

void MixedHybridModel::boundaryTraces(std::span<const std::int32_t> gridRows,
                                      std::vector<BoundaryTrace>& traces) const
{
    solidBoundaryTraces(gridRows, traces);
    if (m_shellPart == nullptr || !m_shellPart->usable())
        return;
    const auto isGiven = [&](std::int32_t row) { return std::binary_search(gridRows.begin(), gridRows.end(), row); };
    const auto entityDofOf = [&](std::size_t unknown)
    {
        const ShellPartUnknown described = m_shellPart->describe(unknown);
        return ElementDof::entity(m_shellEntityRow[described.index], static_cast<int>(described.component));
    };
    const std::vector<EdgeKey>& edges = m_shellPart->topology().edges;
    for (std::size_t e = 0; e < edges.size(); ++e)
    {
        const std::vector<std::int32_t> rows{m_gridRows[edges[e].lower], m_gridRows[edges[e].higher]};
        if (!isGiven(rows[0]) || !isGiven(rows[1]))
            continue;
        const ShellPart::EdgeMoments moments = m_shellPart->edgeMoments(e);
        if (moments.points.empty())
            continue; // the edge of a line element that no shell has
        BoundaryTrace trace;
        trace.gridRows = rows;
        for (const SpaceVector& point : moments.points)
            trace.points.push_back(base::Vec3{point[0], point[1], point[2]});
        const std::size_t pointCount = moments.points.size();
        // the displacement unknown (l, c): the moment l of the component c
        for (std::size_t i = 0; i < moments.displacementUnknowns.size(); ++i)
        {
            BoundaryUnknown moment{entityDofOf(moments.displacementUnknowns[i]), false, {}, {}};
            moment.weights.assign(pointCount, base::Vec3{});
            for (std::size_t g = 0; g < pointCount; ++g)
                moment.weights[g][i % 3] = moments.displacementWeights[i / 3][g];
            trace.unknowns.push_back(std::move(moment));
        }
        for (std::size_t l = 0; l < moments.rotationUnknowns.size(); ++l)
        {
            BoundaryUnknown moment{entityDofOf(moments.rotationUnknowns[l]), true, {}, {}};
            for (const SpaceVector& weight : moments.rotationWeights[l])
                moment.weights.push_back(base::Vec3{weight[0], weight[1], weight[2]});
            trace.unknowns.push_back(std::move(moment));
        }
        traces.push_back(std::move(trace));
    }
}

void MixedHybridModel::solidBoundaryTraces(std::span<const std::int32_t> gridRows,
                                           std::vector<BoundaryTrace>& traces) const
{
    if (m_solidPart == nullptr)
        return;
    // the grids of the model among the given grid points, increasing (the grids of the solids are those of the model)
    std::vector<std::size_t> grids;
    for (const std::int32_t row : gridRows)
        if (const std::int32_t grid = gridIndexOfRow(row); grid >= 0)
            grids.push_back(static_cast<std::size_t>(grid));
    std::sort(grids.begin(), grids.end());
    for (const SolidSkinTrace& skin : m_solidPart->skinTraces(grids))
    {
        BoundaryTrace trace;
        for (const std::size_t grid : skin.grids)
            trace.gridRows.push_back(m_gridRows[grid]);
        for (const SolidPoint& point : skin.points)
            trace.points.push_back(base::Vec3{point[0], point[1], point[2]});
        // the function m along the basic axis c: the DOF 3 m + c of the entity
        for (std::size_t m = 0; m < skin.weights.rows(); ++m)
            for (std::size_t c = 0; c < 3; ++c)
            {
                BoundaryUnknown unknown{
                    ElementDof::entity(m_solidEntityRow[skin.entity], static_cast<int>(3 * m + c)), false, {}, {}};
                unknown.weights.assign(skin.points.size(), base::Vec3{});
                for (std::size_t p = 0; p < skin.points.size(); ++p)
                    unknown.weights[p][c] = skin.weights(m, p);
                for (const TieTerm& term : skin.terms[m])
                {
                    TieUnknown along = term.unknown;
                    along.dof += c;
                    appendTieTerms(along, term.coefficient, unknown.terms);
                }
                trace.unknowns.push_back(std::move(unknown));
            }
        traces.push_back(std::move(trace));
    }
}

} // namespace oos::elements::mixedHybrid
