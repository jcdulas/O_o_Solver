#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"
#include "oos/elements/mixedHybrid/core/SolidSpaces.hpp"
#include "oos/elements/mixedHybrid/model/ShellPart.hpp"
#include "oos/elements/mixedHybrid/model/SolidPart.hpp"
#include "oos/elements/mixedHybrid/model/SolidPartProjections.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeShape.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

// The ties of the solids with the shells and the lines alone (spec section 3.4, ties 1 to 4; document 34 of the
// reference, section 8.2; SolidPart.hpp): what each edge and face of the solids follows, the geometry it takes from
// it, and the rows of the ties.
//
// Every tie is a projection that a rigid motion and a uniform strain pass through exactly: an affine field u = b + A x
// gives the shell the displacement b + A X(ξ, η) of its own surface, and the solid edge or face that follows the shell
// has the geometry of that same surface projected the same way, so the row of each function gives A times its
// geometry coefficient, the value SolidPart::affineField gives it.
//
// Thread-safety: findTies and followTies build the part; ties() is a pure function of it.

namespace oos::elements::mixedHybrid
{

namespace
{

constexpr std::size_t COMPONENTS = 3;
constexpr std::size_t QUADRILATERAL = 4;
constexpr std::size_t TRIANGLE = 3;
// The halves of a quadrilateral shell (three of its corners) that may cover a triangle of the solids.
constexpr std::array<std::array<std::size_t, 3>, 4> QUADRILATERAL_HALVES{{{0, 1, 2}, {0, 2, 3}, {0, 1, 3}, {1, 2, 3}}};
// The points of the rule of the projection of the functions of a line alone on those of the solid edge: k + 4, the
// products of two polynomials of degree k integrated exactly (the reference's edge_rule(k + 4)).
constexpr std::size_t ALONE_PROJECTION_EXTRA_POINTS = 4;
// The points of the rule of the rotation unknowns of an edge (the reference's edge_rule(12)): for a rigid rotation ω,
// W² |X_s| ω · t P_l = ω · (P_s W − P W_s) P_l (X = P / W along the edge) is a polynomial of degree at most 2k along a
// conic edge, integrated exactly.
constexpr std::size_t ROTATION_RULE_POINTS = 12;
// The displacement functions of an edge alone: 1 − s and s at its grids come first (AloneEdge::functions).
constexpr std::size_t ALONE_GRID_FUNCTIONS = 2;

using Row = std::map<TieUnknown, double>;

TieUnknown gridUnknown(std::size_t grid, std::size_t component)
{
    return {TieUnknown::Kind::Grid, grid, component};
}

TieUnknown entityUnknown(std::size_t entity, std::size_t dof)
{
    return {TieUnknown::Kind::SolidEntity, entity, dof};
}

// A local unknown of a solid as an unknown of a tie.
TieUnknown unknownOf(const SolidLocalUnknown& local)
{
    return local.onGrid ? gridUnknown(local.index, local.component) : entityUnknown(local.index, local.component);
}

// out += c × row.
void addRow(Row& out, const Row& row, double c)
{
    for (const auto& [unknown, value] : row)
        out[unknown] += c * value;
}

// The local edge of a shell that joins the two grids of `key`.
std::optional<std::size_t> localEdgeOf(const ShellCorners& corners, const EdgeKey& key)
{
    for (std::size_t k = 0; k < corners.count; ++k)
        if (EdgeKey::of(corners.at(k), corners.at(k + 1)) == key)
            return k;
    return std::nullopt;
}

// The corner of a shell at grid `grid` (one of its corners).
std::size_t cornerAt(const ShellCorners& corners, std::size_t grid)
{
    const std::span<const std::size_t> view = corners.view();
    return static_cast<std::size_t>(std::find(view.begin(), view.end(), grid) - view.begin());
}

// A solid around an edge (tie 4): its kernel, its local unknowns, and the reference points of the lower and the
// higher grid of the edge.
struct SolidAroundEdge
{
    SolidPart::SharedKernel kernel;
    std::vector<SolidLocalUnknown> local;
    SolidPoint lower{};
    SolidPoint higher{};
};

// The solids of an edge whose kernel is accepted (a refused one has its problem reported by validate()), with the
// reference points of the two grids of the edge in each.
std::vector<SolidAroundEdge> solidsAround(const SolidPart& part, const SolidPartEdge& edge)
{
    std::vector<SolidAroundEdge> out;
    for (const std::size_t is : edge.solids)
    {
        std::variant<SolidPart::SharedKernel, std::string> kernel =
            part.kernelOf(is, EntityLayout::Kind::Static, false);
        if (!std::holds_alternative<SolidPart::SharedKernel>(kernel))
            continue;
        const std::vector<std::size_t>& corners = part.solid(is).corners;
        const auto localOf = [&](std::size_t grid)
        { return static_cast<std::size_t>(std::find(corners.begin(), corners.end(), grid) - corners.begin()); };
        const std::span<const SolidPoint> vertices = shapeVertices(part.shapeOf(is));
        out.push_back({std::get<SolidPart::SharedKernel>(std::move(kernel)),
                       part.localUnknowns(is, EntityLayout::Kind::Static), vertices[localOf(edge.grids.lower)],
                       vertices[localOf(edge.grids.higher)]});
    }
    return out;
}

// The rotation of the solids about the unit vector t at the parameter s of their edge (from its lower grid): half the
// curl of their displacement, the mean over the solids of the edge, as a row. With u = Σ_j N_j q_j,
// curl (N e_c) = ∇N × e_c, so ω · t = ½ Σ_jc q_jc (t × ∇N_j)_c.
// Reference: src/tdnns/solid.rs::Solids::rotation_about
Row rotationAbout(std::span<const SolidAroundEdge> around, double s, const SolidPoint& t)
{
    Row out;
    const double count = static_cast<double>(around.size());
    for (const SolidAroundEdge& solid : around)
    {
        const SolidPoint r = add3(scl3(solid.lower, 1.0 - s), scl3(solid.higher, s));
        const std::vector<SolidPoint> gradients = solid.kernel->gradients(r);
        for (std::size_t j = 0; j < gradients.size(); ++j)
        {
            const SolidPoint w = cross3(t, gradients[j]);
            for (std::size_t c = 0; c < COMPONENTS; ++c)
                if (w[c] != 0.0)
                    out[unknownOf(solid.local[COMPONENTS * j + c])] += 0.5 * w[c] / count;
        }
    }
    return out;
}

} // namespace

// ---- what the edges and faces follow

// Reference: src/tdnns/solid.rs::build (edge_kind, face_kind)
void SolidPart::findTies()
{
    const ShellPart* shells = m_input.shells != nullptr && m_input.shells->usable() ? m_input.shells : nullptr;
    findEdgeTies(shells);
    if (shells != nullptr)
        for (std::size_t ie = 0; ie < shells->input().shells.size(); ++ie)
            coverFaces(ie);
    for (SolidPartEdge& edge : m_edges)
        edge.follows = edge.tie.kind != SolidEdgeTie::Kind::Own;
    for (SolidPartFace& face : m_faces)
        face.follows = face.tie.coveredByShell;
}

// An edge of shells follows the first shell of the edge (by increasing element ID); an edge of lines alone is
// followed by its lines.
// Reference: src/tdnns/solid.rs::build (edge_kind)
void SolidPart::findEdgeTies(const ShellPart* shells)
{
    std::map<EdgeKey, std::size_t> aloneByKey;
    for (std::size_t i = 0; i < m_input.aloneEdges.size(); ++i)
        aloneByKey.emplace(m_input.aloneEdges[i].grids, i);
    for (SolidPartEdge& edge : m_edges)
    {
        if (shells != nullptr)
        {
            const Topology& topology = shells->topology();
            const auto found = topology.edgeIndex.find(edge.grids);
            if (found != topology.edgeIndex.end() && !topology.edgeElements[found->second].empty())
            {
                const std::size_t shell = topology.edgeElements[found->second].front();
                const std::optional<std::size_t> localEdge =
                    localEdgeOf(shells->input().shells[shell].corners, edge.grids);
                OOS_ASSERT(localEdge.has_value());
                edge.tie.kind = SolidEdgeTie::Kind::Shell;
                edge.tie.shell = shell;
                edge.tie.localEdge = *localEdge;
                continue;
            }
        }
        const auto alone = aloneByKey.find(edge.grids);
        if (alone != aloneByKey.end())
        {
            edge.tie.kind = SolidEdgeTie::Kind::Alone;
            edge.tie.aloneEdge = alone->second;
        }
    }
}

// The faces that shell `shell` covers: the face of its corners, or for a quadrilateral the faces of its four halves
// (the first shell of a face wins).
// Reference: src/tdnns/solid.rs::build (face_kind)
void SolidPart::coverFaces(std::size_t shell)
{
    const ShellCorners& corners = m_input.shells->input().shells[shell].corners;
    // the corner sets: the whole shell, then the halves of a quadrilateral
    std::vector<std::vector<std::size_t>> sets(1);
    for (std::size_t k = 0; k < corners.count; ++k)
        sets.front().push_back(k);
    if (corners.count == QUADRILATERAL)
        for (const std::array<std::size_t, 3>& half : QUADRILATERAL_HALVES)
            sets.emplace_back(half.begin(), half.end());
    for (const std::vector<std::size_t>& set : sets)
    {
        std::vector<std::size_t> grids;
        for (const std::size_t k : set)
            grids.push_back(corners.at(k));
        std::sort(grids.begin(), grids.end());
        const auto found = m_faceIndex.find(grids);
        if (found == m_faceIndex.end() || m_faces[found->second].tie.coveredByShell)
            continue;
        SolidPartFace& face = m_faces[found->second];
        face.tie.coveredByShell = true;
        face.tie.shell = shell;
        for (std::size_t i = 0; i < face.count; ++i)
            face.tie.corners[i] = cornerAt(corners, face.grids[i]);
        if (corners.count == QUADRILATERAL && face.count == TRIANGLE)
            followDiagonal(found->second);
    }
}

// A triangle of the solids under half of a quadrilateral shell has the diagonal of the quadrilateral for an edge (its
// two grids at opposite corners): that edge follows the shell where nothing else carries it.
// Reference: src/tdnns/solid.rs::build (EdgeKind::Diagonal)
void SolidPart::followDiagonal(std::size_t face)
{
    const SolidPartFace& f = m_faces[face];
    for (std::size_t i = 0; i < TRIANGLE; ++i)
    {
        const std::size_t j = (i + 1) % TRIANGLE;
        const std::size_t ca = f.tie.corners[i];
        const std::size_t cb = f.tie.corners[j];
        if ((ca + 2) % QUADRILATERAL != cb)
            continue;
        SolidPartEdge& diagonal = m_edges[m_edgeIndex.at(EdgeKey::of(f.grids[i], f.grids[j]))];
        if (diagonal.tie.kind != SolidEdgeTie::Kind::Own)
            continue;
        const bool fromLower = f.grids[i] < f.grids[j];
        diagonal.tie.kind = SolidEdgeTie::Kind::Diagonal;
        diagonal.tie.shell = f.tie.shell;
        diagonal.tie.lowerCorner = fromLower ? ca : cb;
        diagonal.tie.higherCorner = fromLower ? cb : ca;
    }
}

// Reference: src/tdnns/solid.rs::build (the geometry of what follows the shells and the lines alone)
void SolidPart::followTies(const EdgeProjection& projection)
{
    const std::span<const SolidPoint> x = m_input.gridPositions;
    for (std::size_t e = 0; e < m_edges.size(); ++e)
    {
        SolidPartEdge& edge = m_edges[e];
        // an edge of lines alone is straight, whatever its mid-side grid and the skin around it
        if (edge.tie.kind == SolidEdgeTie::Kind::Alone)
        {
            edge.geometry.assign(edge.geometry.size(), SolidPoint{});
            continue;
        }
        if (!edge.follows || !takesCurves() || edge.geometry.empty())
            continue;
        // the curve of the shell along the edge, projected as the skin is
        const std::vector<SolidPoint> values =
            shellPoints(edge.tie.shell, followedEdgePoints(e, projection.parameters));
        edge.geometry = projection.project(values, x[edge.grids.lower], x[edge.grids.higher]);
    }
    if (!takesCurves())
        return;
    for (std::size_t f = 0; f < m_faces.size(); ++f)
    {
        SolidPartFace& face = m_faces[f];
        if (!face.follows || face.geometry.empty())
            continue;
        // the surface of the shell over the face, less what its grids and its edges (as they now are) carry
        std::vector<base::EntityId> order;
        std::vector<SolidPoint> corners;
        std::vector<std::vector<SolidPoint>> edgeCoefficients;
        for (std::size_t i = 0; i < face.count; ++i)
        {
            order.push_back(static_cast<base::EntityId>(face.grids[i]));
            corners.push_back(x[face.grids[i]]);
            const EdgeKey key = EdgeKey::of(face.grids[i], face.grids[(i + 1) % face.count]);
            edgeCoefficients.push_back(m_edges[m_edgeIndex.at(key)].geometry);
        }
        const FaceProjection fp = FaceProjection::of(order, m_traceDegree, m_faceDegree);
        const std::vector<SolidPoint> values = shellPoints(face.tie.shell, followedFacePoints(f, fp.points));
        face.geometry = fp.project(values, corners, edgeCoefficients);
    }
}

// ---- the points and the rows of the shells

// Reference: src/tdnns/solid.rs::Context::edge_point_of, and the diagonal of build and ties
std::vector<ReferencePoint> SolidPart::followedEdgePoints(std::size_t edge, std::span<const double> parameters) const
{
    const SolidEdgeTie& tie = m_edges[edge].tie;
    std::vector<ReferencePoint> out;
    for (const double s : parameters)
    {
        if (tie.kind == SolidEdgeTie::Kind::Shell)
        {
            out.push_back(m_input.shells->edgeReferencePoint(tie.shell, tie.localEdge, s));
            continue;
        }
        // the diagonal of a quadrilateral: the straight line of the reference square from corner to corner
        const ReferencePoint a = referenceCorner(QUADRILATERAL, tie.lowerCorner);
        const ReferencePoint b = referenceCorner(QUADRILATERAL, tie.higherCorner);
        out.push_back({(1.0 - s) * a[0] + s * b[0], (1.0 - s) * a[1] + s * b[1]});
    }
    return out;
}

// The point of the face at its coordinates c is Σ_i w_i(c) corner i (faceWeights); the same weights of the corners of
// the shell there give its reference point.
// Reference: src/tdnns/solid.rs::Context::face_point_of
std::vector<ReferencePoint> SolidPart::followedFacePoints(std::size_t face,
                                                          std::span<const std::array<double, 2>> points) const
{
    const SolidPartFace& f = m_faces[face];
    const std::size_t cornerCount = m_input.shells->input().shells[f.tie.shell].corners.count;
    std::vector<ReferencePoint> out;
    for (const std::array<double, 2>& c : points)
    {
        const std::vector<double> w = faceWeights(f.count, c);
        ReferencePoint r{};
        for (std::size_t i = 0; i < f.count; ++i)
        {
            const ReferencePoint corner = referenceCorner(cornerCount, f.tie.corners[i]);
            r[0] += w[i] * corner[0];
            r[1] += w[i] * corner[1];
        }
        out.push_back(r);
    }
    return out;
}

// The points of the surface of a shell, X = Σ_i x_i N_i / W.
// Reference: src/tdnns/solid.rs::Context::shell_rows (its point, no function kept)
std::vector<SolidPoint> SolidPart::shellPoints(std::size_t shell, std::span<const ReferencePoint> points) const
{
    const auto none = [](const FunctionKind&) { return false; };
    std::vector<SolidPoint> out;
    for (const ShellDisplacementRows& at : m_input.shells->displacementRowsAt(shell, points, none))
        out.push_back(at.point);
    return out;
}

// Reference: src/tdnns/solid.rs::Context::shell_rows
std::vector<std::array<SolidPart::TieRow, 3>> SolidPart::shellRows(std::size_t shell,
                                                                   std::span<const ReferencePoint> points,
                                                                   const std::optional<std::size_t>& onEdge) const
{
    const ShellPart& shells = *m_input.shells;
    std::function<bool(const FunctionKind&)> keep;
    if (onEdge)
    {
        // on an edge of the shell, the functions of its two corners and of that edge (the others vanish there)
        const std::size_t edge = *onEdge;
        const std::size_t next = (edge + 1) % shells.input().shells[shell].corners.count;
        keep = [edge, next](const FunctionKind& kind)
        {
            if (kind.place == FunctionKind::Place::Vertex)
                return kind.entity == edge || kind.entity == next;
            return kind.place == FunctionKind::Place::Edge && kind.entity == edge;
        };
    }
    // a grid translation of the shells is the translation of the grid; the other unknowns are the shells' own
    const std::size_t firstEntity = shells.firstEntityUnknown();
    constexpr std::size_t GRID_SLOTS = 6;
    std::vector<std::array<TieRow, 3>> out;
    for (const ShellDisplacementRows& at : shells.displacementRowsAt(shell, points, keep))
    {
        std::array<TieRow, 3> rows;
        for (std::size_t c = 0; c < COMPONENTS; ++c)
            for (const ShellPartTerm& term : at.rows[c])
            {
                const TieUnknown unknown = term.unknown < firstEntity
                                               ? gridUnknown(term.unknown / GRID_SLOTS, term.unknown % GRID_SLOTS)
                                               : TieUnknown{TieUnknown::Kind::ShellUnknown, term.unknown, 0};
                rows[c][unknown] += term.coefficient;
            }
        out.push_back(std::move(rows));
    }
    return out;
}

// ---- the rows of the ties

// Reference: src/tdnns/solid.rs::Solids::ties
std::vector<SolidTie> SolidPart::ties(const std::function<bool(const TieUnknown&)>& isHeld) const
{
    std::vector<PendingTie> pending;
    const EdgeProjection projection = EdgeProjection::of(m_traceDegree);
    for (std::size_t e = 0; e < m_edges.size(); ++e)
    {
        if (m_edges[e].tie.kind == SolidEdgeTie::Kind::Alone)
            addAloneEdgeTies(e, pending);
        else
            addEdgeTies(e, projection, pending);
    }
    for (std::size_t f = 0; f < m_faces.size(); ++f)
        addFaceTies(f, pending);
    for (std::size_t e = 0; e < m_edges.size(); ++e)
        addRotationTies(e, pending);

    // a dependent held by a support is left out; the first row of a dependent wins; no zero coefficient
    std::set<TieUnknown> written;
    std::vector<SolidTie> out;
    for (const PendingTie& tie : pending)
    {
        if ((isHeld && isHeld(tie.dependent)) || !written.insert(tie.dependent).second)
            continue;
        SolidTie row{tie.dependent, {}};
        for (const auto& [unknown, coefficient] : tie.row)
            if (coefficient != 0.0)
                row.terms.push_back({unknown, coefficient});
        out.push_back(std::move(row));
    }
    return out;
}

// Tie 1: the functions of an edge that follows a shell are the projection of the trace of the shell less its linear
// part, coefficient m = Σ_q a(m, q) u(s_q) + a0[m] u(lower grid) + a1[m] u(higher grid) (EdgeProjection).
// Reference: src/tdnns/solid.rs::Solids::ties (EdgeKind::Shell, EdgeKind::Diagonal)
void SolidPart::addEdgeTies(std::size_t edge, const EdgeProjection& projection, std::vector<PendingTie>& out) const
{
    const SolidPartEdge& e = m_edges[edge];
    const bool followsShell = e.tie.kind == SolidEdgeTie::Kind::Shell || e.tie.kind == SolidEdgeTie::Kind::Diagonal;
    if (!followsShell || !m_edgeEntities[edge])
        return;
    // on an edge of the shell its corners and that edge only; on the diagonal every function of the shell
    const std::optional<std::size_t> onEdge =
        e.tie.kind == SolidEdgeTie::Kind::Shell ? std::optional<std::size_t>(e.tie.localEdge) : std::nullopt;
    const std::vector<std::array<TieRow, 3>> field =
        shellRows(e.tie.shell, followedEdgePoints(edge, projection.parameters), onEdge);
    for (std::size_t m = 0; m < e.geometry.size(); ++m)
        for (std::size_t c = 0; c < COMPONENTS; ++c)
        {
            TieRow row;
            for (std::size_t q = 0; q < field.size(); ++q)
                addRow(row, field[q][c], projection.weights(m, q));
            row[gridUnknown(e.grids.lower, c)] += projection.lowerWeights[m];
            row[gridUnknown(e.grids.higher, c)] += projection.upperWeights[m];
            out.push_back({entityUnknown(*m_edgeEntities[edge], COMPONENTS * m + c), std::move(row)});
        }
}

// Tie 3: both the line alone and the solid edge are linear between the grids plus functions that vanish there; those
// of the line, f_l = s (1 − s) P_l(2s − 1) along the axes of its frame, follow the L2 projection of those of the solid,
// g_m along the basic axes: q_line(l, axis) = Σ_m C(l, m) axis · q_solid(m), C = G⁻¹ R, G_li = ∫ f_l f_i,
// R_lm = ∫ f_l g_m (the same functions when k and the trace degree are 4).
// Reference: src/tdnns/solid.rs::Solids::ties (EdgeKind::Alone)
void SolidPart::addAloneEdgeTies(std::size_t edge, std::vector<PendingTie>& out) const
{
    const SolidPartEdge& e = m_edges[edge];
    const SolidPartAloneEdge& alone = m_input.aloneEdges[e.tie.aloneEdge];
    const std::size_t k = alone.alone.degree;
    const std::size_t lineFunctions = k - 1;
    const std::size_t solidFunctions = e.geometry.size();
    Matrix gram(lineFunctions, lineFunctions);
    Matrix cross(lineFunctions, solidFunctions);
    for (const LineRulePoint& point : unitIntervalRule(k + ALONE_PROJECTION_EXTRA_POINTS))
    {
        const std::vector<double> line = alone.alone.functions(point.point).values;
        const std::vector<double> solid = edgeFunctions(m_traceDegree, point.point);
        for (std::size_t i = 0; i < lineFunctions; ++i)
        {
            const double fi = line[ALONE_GRID_FUNCTIONS + i];
            for (std::size_t j = 0; j < lineFunctions; ++j)
                gram(i, j) += point.weight * fi * line[ALONE_GRID_FUNCTIONS + j];
            for (std::size_t m = 0; m < solidFunctions; ++m)
                cross(i, m) += point.weight * fi * solid[m];
        }
    }
    const std::optional<Matrix> gramInverse = inverse(gram);
    if (!gramInverse)
        throw base::InternalError("the Gram matrix of the functions of an edge alone is singular");
    const Matrix c = *gramInverse * cross;
    for (std::size_t l = 0; l < lineFunctions; ++l)
        for (std::size_t axis = 0; axis < COMPONENTS; ++axis)
        {
            TieRow row;
            for (std::size_t m = 0; m < solidFunctions; ++m)
                for (std::size_t cc = 0; cc < COMPONENTS; ++cc)
                    row[entityUnknown(*m_edgeEntities[edge], COMPONENTS * m + cc)] +=
                        c(l, m) * alone.alone.frame[axis][cc];
            out.push_back({{TieUnknown::Kind::AloneEdge, alone.edge, COMPONENTS * l + axis}, std::move(row)});
        }
}

// Tie 2: the functions of a face that a shell covers are the projection of the displacement of the shell less what
// the grids and the edges of the face carry, coefficient m = Σ_q b(m, q) (u(c_q) − Σ_i φ_i(c_q) u_i − Σ_kn φ_kn(c_q)
// q_kn) (FaceProjection).
// Reference: src/tdnns/solid.rs::Solids::ties (the faces)
void SolidPart::addFaceTies(std::size_t face, std::vector<PendingTie>& out) const
{
    const SolidPartFace& f = m_faces[face];
    if (!f.follows || !m_faceEntities[face])
        return;
    const std::size_t nf = f.count;
    std::vector<base::EntityId> order;
    std::vector<std::optional<std::size_t>> edgeEntities;
    for (std::size_t i = 0; i < nf; ++i)
    {
        order.push_back(static_cast<base::EntityId>(f.grids[i]));
        edgeEntities.push_back(m_edgeEntities[m_edgeIndex.at(EdgeKey::of(f.grids[i], f.grids[(i + 1) % nf]))]);
    }
    const FaceProjection fp = FaceProjection::of(order, m_traceDegree, m_faceDegree);
    const std::vector<std::array<TieRow, 3>> field =
        shellRows(f.tie.shell, followedFacePoints(face, fp.points), std::nullopt);
    for (std::size_t m = 0; m < fp.interiorCount; ++m)
        for (std::size_t c = 0; c < COMPONENTS; ++c)
        {
            TieRow row;
            for (std::size_t q = 0; q < fp.points.size(); ++q)
            {
                const double b = fp.weights(m, q);
                const std::vector<double>& phi = fp.functionValues[q];
                addRow(row, field[q][c], b);
                for (std::size_t i = 0; i < nf; ++i)
                    row[gridUnknown(f.grids[i], c)] -= b * phi[i];
                for (std::size_t k = 0; k < nf; ++k)
                    for (std::size_t n = 0; n < fp.perEdge; ++n)
                        row[entityUnknown(*edgeEntities[k], COMPONENTS * n + c)] -= b * phi[nf + k * fp.perEdge + n];
            }
            out.push_back({entityUnknown(*m_faceEntities[face], COMPONENTS * m + c), std::move(row)});
        }
}

// Tie 4: the rotation unknowns l of an edge of shells or of lines alone that the solids have, where no shell of the
// edge lies on a face of the solids, follow the rotation of the solids about the edge: the Legendre coefficients
// (2l + 1) ∫ W² |X_s| (ω · t) P_l(2s − 1) ds of the rotation of the edge (ShellEdgeRotationFrame; W = 1 and
// |X_s| = L on a straight edge of lines alone), ω the mean rotation of the solids of the edge (rotationAbout).
// Reference: src/tdnns/solid.rs::Solids::ties (the rotations about the edges)
void SolidPart::addRotationTies(std::size_t edge, std::vector<PendingTie>& out) const
{
    const SolidPartEdge& e = m_edges[edge];
    const std::vector<LineRulePoint> rule = unitIntervalRule(ROTATION_RULE_POINTS);
    // the rotation unknowns of the edge, its tangent and the measure of its rotation at the points of the rule
    std::vector<TieUnknown> dependents;
    std::vector<ShellEdgeRotationFrame> frames;
    if (e.tie.kind == SolidEdgeTie::Kind::Shell)
    {
        const ShellPart& shells = *m_input.shells;
        const std::size_t topologyEdge = shells.topology().edgeIndex.at(e.grids);
        const std::vector<std::size_t>& owners = shells.topology().edgeElements[topologyEdge];
        // a shell lies on a face of the solids when one of its other corners makes a face of the solids with the edge
        const auto lies = [&](std::size_t shell)
        {
            for (const std::size_t g : shells.input().shells[shell].corners.view())
            {
                if (g == e.grids.lower || g == e.grids.higher)
                    continue;
                std::array<std::size_t, 3> three{e.grids.lower, e.grids.higher, g};
                std::sort(three.begin(), three.end());
                if (m_faceByThree.contains(three))
                    return true;
            }
            return false;
        };
        if (std::all_of(owners.begin(), owners.end(), lies))
            return;
        const std::size_t first = shells.entities()[shells.edgeEntity(topologyEdge)].firstUnknown;
        for (std::size_t l = 0; l < shells.degree(); ++l)
            dependents.push_back({TieUnknown::Kind::ShellUnknown, first + shells.edgeRotationDof(l), 0});
        for (const LineRulePoint& point : rule)
            frames.push_back(shells.edgeRotationFrame(e.tie.shell, e.tie.localEdge, point.point));
    }
    else if (e.tie.kind == SolidEdgeTie::Kind::Alone)
    {
        const SolidPartAloneEdge& alone = m_input.aloneEdges[e.tie.aloneEdge];
        const std::size_t k = alone.alone.degree;
        for (std::size_t l = 0; l < k; ++l)
            dependents.push_back({TieUnknown::Kind::AloneEdge, alone.edge, COMPONENTS * (k - 1) + l});
        const SolidPoint chord = sub3(alone.alone.upperPoint, alone.alone.lowerPoint);
        frames.assign(rule.size(), ShellEdgeRotationFrame{unit3(chord), norm3(chord)});
    }
    else
    {
        return;
    }

    const std::vector<SolidAroundEdge> around = solidsAround(*this, e);
    if (around.empty())
        return;
    std::vector<TieRow> rotations;
    for (std::size_t q = 0; q < rule.size(); ++q)
        rotations.push_back(rotationAbout(around, rule[q].point, frames[q].tangent));
    for (std::size_t l = 0; l < dependents.size(); ++l)
    {
        TieRow row;
        for (std::size_t q = 0; q < rule.size(); ++q)
            addRow(row, rotations[q],
                   static_cast<double>(2 * l + 1) * rule[q].weight * legendreValue(l, 2.0 * rule[q].point - 1.0) *
                       frames[q].measure);
        out.push_back({dependents[l], std::move(row)});
    }
}

// ---- the functions of the skin from a field known at points

std::vector<SolidSkinTrace> SolidPart::skinTraces(std::span<const std::size_t> grids) const
{
    std::vector<SolidSkinTrace> out;
    const auto isGiven = [&](std::size_t grid) { return std::binary_search(grids.begin(), grids.end(), grid); };
    const std::span<const SolidPoint> x = m_input.gridPositions;
    const EdgeProjection projection = EdgeProjection::of(m_traceDegree);
    // An edge: the curve X(s) = (1 − s) X_a + s X_b + Σ_n g_n b_n(s), and the projection of the field less its
    // linear part (tie 1): coefficient m = Σ_p a(m, p) u(s_p) + a0[m] u(a) + a1[m] u(b).
    for (std::size_t e = 0; e < m_edges.size(); ++e)
    {
        const SolidPartEdge& edge = m_edges[e];
        if (!m_edgeEntities[e] || !edge.onSkin || edge.follows || edge.geometry.empty() ||
            !isGiven(edge.grids.lower) || !isGiven(edge.grids.higher))
            continue;
        SolidSkinTrace trace;
        trace.grids = {edge.grids.lower, edge.grids.higher};
        trace.entity = *m_edgeEntities[e];
        for (const double s : projection.parameters)
        {
            const std::vector<double> functions = edgeFunctions(m_traceDegree, s);
            SolidPoint point = add3(scl3(x[edge.grids.lower], 1.0 - s), scl3(x[edge.grids.higher], s));
            for (std::size_t n = 0; n < edge.geometry.size(); ++n)
                point = add3(point, scl3(edge.geometry[n], functions[n]));
            trace.points.push_back(point);
        }
        trace.weights = projection.weights;
        for (std::size_t m = 0; m < edge.geometry.size(); ++m)
            trace.terms.push_back({{gridUnknown(edge.grids.lower, 0), projection.lowerWeights[m]},
                                   {gridUnknown(edge.grids.higher, 0), projection.upperWeights[m]}});
        out.push_back(std::move(trace));
    }
    // A face: the surface X(c) = Σ_i φ_i(c) X_i + Σ_kn φ_kn(c) g_kn + Σ_m φ_m(c) g_m (its grids, its edges, its own
    // functions), and the projection of the field less what its grids and its edges carry (tie 2):
    // coefficient m = Σ_p b(m, p) (u(c_p) − Σ_i φ_i(c_p) u_i − Σ_kn φ_kn(c_p) q_kn).
    for (std::size_t f = 0; f < m_faces.size(); ++f)
    {
        const SolidPartFace& face = m_faces[f];
        const std::span<const std::size_t> cycle = face.view();
        if (!m_faceEntities[f] || !face.onSkin() || face.follows || face.geometry.empty() ||
            !std::all_of(cycle.begin(), cycle.end(), isGiven))
            continue;
        const std::size_t nf = face.count;
        std::vector<base::EntityId> order;
        std::vector<std::size_t> edges;
        bool hasEdgeFunctions = true;
        for (std::size_t i = 0; i < nf; ++i)
        {
            order.push_back(static_cast<base::EntityId>(face.grids[i]));
            edges.push_back(m_edgeIndex.at(EdgeKey::of(face.grids[i], face.grids[(i + 1) % nf])));
            hasEdgeFunctions = hasEdgeFunctions && m_edgeEntities[edges.back()].has_value();
        }
        if (!hasEdgeFunctions)
            continue;
        const FaceProjection fp = FaceProjection::of(order, m_traceDegree, m_faceDegree);
        if (fp.interiorCount != face.geometry.size())
            throw base::InternalError("the functions of a face of the skin are not those of its projection");
        SolidSkinTrace trace;
        trace.grids.assign(cycle.begin(), cycle.end());
        trace.entity = *m_faceEntities[f];
        const std::size_t firstInterior = nf + nf * fp.perEdge;
        for (std::size_t p = 0; p < fp.points.size(); ++p)
        {
            const std::vector<double>& phi = fp.functionValues[p];
            SolidPoint point{};
            for (std::size_t i = 0; i < nf; ++i)
                point = add3(point, scl3(x[face.grids[i]], phi[i]));
            for (std::size_t k = 0; k < nf; ++k)
                for (std::size_t n = 0; n < fp.perEdge && n < m_edges[edges[k]].geometry.size(); ++n)
                    point = add3(point, scl3(m_edges[edges[k]].geometry[n], phi[nf + k * fp.perEdge + n]));
            for (std::size_t m = 0; m < fp.interiorCount; ++m)
                point = add3(point, scl3(face.geometry[m], phi[firstInterior + m]));
            trace.points.push_back(point);
        }
        trace.weights = Matrix(fp.interiorCount, fp.points.size());
        for (std::size_t m = 0; m < fp.interiorCount; ++m)
        {
            std::vector<double> onGrids(nf, 0.0);
            std::vector<double> onEdges(nf * fp.perEdge, 0.0);
            for (std::size_t p = 0; p < fp.points.size(); ++p)
            {
                const double b = fp.weights(m, p);
                const std::vector<double>& phi = fp.functionValues[p];
                trace.weights(m, p) = b;
                for (std::size_t i = 0; i < nf; ++i)
                    onGrids[i] -= b * phi[i];
                for (std::size_t j = 0; j < nf * fp.perEdge; ++j)
                    onEdges[j] -= b * phi[nf + j];
            }
            std::vector<TieTerm> terms;
            for (std::size_t i = 0; i < nf; ++i)
                terms.push_back({gridUnknown(face.grids[i], 0), onGrids[i]});
            for (std::size_t k = 0; k < nf; ++k)
                for (std::size_t n = 0; n < fp.perEdge; ++n)
                    terms.push_back(
                        {entityUnknown(*m_edgeEntities[edges[k]], COMPONENTS * n), onEdges[k * fp.perEdge + n]});
            trace.terms.push_back(std::move(terms));
        }
        out.push_back(std::move(trace));
    }
    return out;
}

} // namespace oos::elements::mixedHybrid
