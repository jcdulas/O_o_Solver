#include "oos/elements/mixedHybrid/model/ShellPart.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"
#include "oos/elements/mixedHybrid/shell/ShellPointLaw.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace oos::elements::mixedHybrid
{

namespace
{

// Two mid-side grids of one edge given by two shells are the same point when they are closer than this share of the
// chord of the edge.
constexpr double SAME_MIDSIDE_TOLERANCE = 1e-9;

// The surface of an element as a function of its reference coordinates.
SurfaceAt surfaceOf(const Patch& patch)
{
    return [&patch](double xi, double eta) { return patch.evaluate(xi, eta); };
}

// The number of the functions of a basis whose place is the interior.
std::size_t interiorCount(const std::vector<FunctionKind>& kinds)
{
    return static_cast<std::size_t>(
        std::count_if(kinds.begin(), kinds.end(), [](const FunctionKind& kind) { return kind.isInterior(); }));
}

} // namespace

ShellPart::ShellPart(ShellPartInput input) : m_input(std::move(input))
{
    m_pointsPerDirection =
        m_input.pointsPerDirection > 0 ? m_input.pointsPerDirection : shellPointsPerDirection(m_input.degree);
    if (m_input.degree < 2)
    {
        addProblem(ShellPartProblem::Subject::Model, 0, "the shells need a displacement of degree 2 at least");
        return;
    }
    if (!checkInput())
        return;
    // The reading of the mesh: the shells alone (the edges of the lines alone are the integrator's; they come after
    // those of the shells in the reference's topology and change nothing of these).
    for (const ShellPartGrid& grid : m_input.grids)
        m_reading.grids.push_back(grid.position);
    for (const ShellPartShell& shell : m_input.shells)
        m_reading.shells.push_back(shell.corners);
    m_reading.rawEdges = m_input.rawEdges;
    m_reading.rawElements = m_input.rawElements;
    m_reading.heldTranslations = m_input.heldTranslations;
    m_topology = Topology::fromMesh(m_reading);
    applyMidsideGrids();
    buildShells();
    placeLines();
    m_usable = true;
    if (m_input.checkElements)
        checkElements();
}

void ShellPart::addProblem(ShellPartProblem::Subject subject, std::size_t index, std::string message)
{
    m_problems.push_back({subject, index, std::move(message)});
}

std::string ShellPart::gridName(std::size_t grid) const
{
    return std::to_string(m_input.grids[grid].id);
}

bool ShellPart::checkInput()
{
    const std::size_t ng = m_input.grids.size();
    bool readable = true;
    for (std::size_t ie = 0; ie < m_input.shells.size(); ++ie)
    {
        const ShellPartShell& shell = m_input.shells[ie];
        const std::size_t nv = shell.corners.count;
        if (nv != 3 && nv != 4)
        {
            addProblem(ShellPartProblem::Subject::Shell, ie,
                       "expected 3 or 4 corner grids, found " + std::to_string(nv));
            readable = false;
            continue;
        }
        bool indices = true;
        for (std::size_t k = 0; k < nv; ++k)
            indices = indices && shell.corners.grids[k] < ng;
        for (std::size_t k = 0; k < nv; ++k)
            indices = indices && (shell.midside[k] < 0 || static_cast<std::size_t>(shell.midside[k]) < ng);
        if (!indices)
        {
            addProblem(ShellPartProblem::Subject::Shell, ie, "a grid of the element is not a grid of the shells");
            readable = false;
            continue;
        }
        // distinct corners that span a plane: the normal of the element, by the cross product of two sides of a
        // triangle or of the two diagonals of a quadrilateral, is defined
        bool distinct = true;
        for (std::size_t a = 0; a < nv; ++a)
            for (std::size_t b = a + 1; b < nv; ++b)
                distinct = distinct && shell.corners.grids[a] != shell.corners.grids[b] &&
                           norm3(sub3(m_input.grids[shell.corners.grids[a]].position,
                                      m_input.grids[shell.corners.grids[b]].position)) > 0.0;
        const auto x = [&](std::size_t k) { return m_input.grids[shell.corners.grids[k]].position; };
        const SpaceVector normal =
            nv == 3 ? cross3(sub3(x(1), x(0)), sub3(x(2), x(0))) : cross3(sub3(x(2), x(0)), sub3(x(3), x(1)));
        if (!distinct || !(norm3(normal) > 0.0))
        {
            addProblem(ShellPartProblem::Subject::Shell, ie,
                       "degenerate element: its corner grids are not distinct points of a surface");
            readable = false;
        }
    }
    for (const std::size_t ie : m_input.rawElements)
        if (ie >= m_input.shells.size())
        {
            addProblem(ShellPartProblem::Subject::Model, 0, "a raw element is not a shell");
            readable = false;
        }
    for (const EdgeKey& edge : m_input.rawEdges)
        if (edge.higher >= ng)
        {
            addProblem(ShellPartProblem::Subject::Model, 0, "a raw edge is not between grids of the shells");
            readable = false;
        }
    for (const auto& [grid, component] : m_input.heldTranslations)
        if (grid >= ng || component < 0 || component > 2)
        {
            addProblem(ShellPartProblem::Subject::Model, 0, "a held translation is not one of a grid of the shells");
            readable = false;
        }
    return readable;
}

void ShellPart::applyMidsideGrids()
{
    std::vector<bool> fixed(m_topology.edges.size(), false);
    for (std::size_t ie = 0; ie < m_input.shells.size(); ++ie)
    {
        const ShellPartShell& shell = m_input.shells[ie];
        for (std::size_t k = 0; k < shell.corners.count; ++k)
        {
            if (shell.midside[k] < 0)
                continue;
            const std::size_t a = shell.corners.at(k);
            const std::size_t b = shell.corners.at(k + 1);
            const std::size_t e = m_topology.indexOf(a, b);
            const SpaceVector& xa = m_input.grids[a].position;
            const SpaceVector& xb = m_input.grids[b].position;
            const SpaceVector& xm = m_input.grids[static_cast<std::size_t>(shell.midside[k])].position;
            // the parabola through X_a, X_m, X_b: the conic of weight 1 whose control point is 2 X_m − (X_a + X_b)/2,
            // whose bulge (half the offset of the control point from the middle of the chord) is X_m − (X_a + X_b)/2
            const SpaceVector bulge = sub3(xm, scl3(add3(xa, xb), 0.5));
            if (!fixed[e])
            {
                m_topology.bulges[e] = bulge;
                m_topology.weights[e] = 1.0;
                fixed[e] = true;
            }
            else if (norm3(sub3(bulge, m_topology.bulges[e])) > SAME_MIDSIDE_TOLERANCE * norm3(sub3(xb, xa)))
            {
                addProblem(ShellPartProblem::Subject::Shell, ie,
                           "its mid-side grid of the edge between grids " + gridName(a) + " and " + gridName(b) +
                               " is not the one an earlier element gives that edge (the earlier one is kept)");
            }
        }
    }
}

const ShellReferenceElement& ShellPart::referenceElement(std::size_t cornerCount) const
{
    return shellReferenceElement(cornerCount, m_input.degree, m_pointsPerDirection);
}

const InteriorProjection& ShellPart::projectionOf(std::size_t cornerCount) const
{
    const std::optional<InteriorProjection>& projection =
        cornerCount == 3 ? m_triangleProjection : m_quadrilateralProjection;
    if (!projection)
        throw base::InternalError("shell part: no interior projection for this corner count");
    return *projection;
}

std::size_t ShellPart::localEdgeOf(std::size_t shell, std::size_t edge) const
{
    const ShellCorners& corners = m_input.shells[shell].corners;
    for (std::size_t k = 0; k < corners.count; ++k)
        if (EdgeKey::of(corners.at(k), corners.at(k + 1)) == m_topology.edges[edge])
            return k;
    throw base::InternalError("shell part: the element does not have the edge");
}

void ShellPart::buildShells()
{
    const std::size_t ng = m_input.grids.size();
    const std::size_t ne = m_input.shells.size();
    const std::size_t nedge = m_topology.edges.size();
    const std::size_t k = m_input.degree;
    const std::size_t p = k - 1;
    for (const ShellPartShell& shell : m_input.shells)
    {
        std::optional<InteriorProjection>& projection =
            shell.corners.count == 3 ? m_triangleProjection : m_quadrilateralProjection;
        if (!projection)
            projection = InteriorProjection::build(referenceElement(shell.corners.count));
    }

    // ---- numbering: six per grid, then per edge the moments of the displacement trace (3 (k − 1)), the shear along
    // the edge (k) and the rotation about it (k), then per shell its interior unknowns (displacement and shear)
    const std::size_t perEdge = edgeDofCount();
    const auto edgeBase = [&](std::size_t e) { return 6 * ng + perEdge * e; };
    std::vector<std::size_t> elementBase(ne);
    std::size_t next = 6 * ng + perEdge * nedge;
    for (std::size_t ie = 0; ie < ne; ++ie)
    {
        const ShellReferenceElement& re = referenceElement(m_input.shells[ie].corners.count);
        elementBase[ie] = next;
        next += 3 * interiorCount(re.h1Kinds) + interiorCount(re.hcurlKinds);
    }
    m_unknownCount = next;

    // ---- geometry: the surfaces, the edge moments of the shared curves (from the first shell of each edge)
    std::vector<ElementSurface> surfaces;
    std::vector<Polynomial2> weights;
    surfaces.reserve(ne);
    weights.reserve(ne);
    for (std::size_t ie = 0; ie < ne; ++ie)
    {
        surfaces.push_back(elementSurface(m_reading, m_topology, ie));
        weights.push_back(weightPolynomial(surfaces.back().patch));
    }
    std::vector<std::vector<SpaceVector>> edgeGeometry(nedge);
    for (std::size_t e = 0; e < nedge; ++e)
    {
        const std::size_t ie = m_topology.edgeElements[e].front();
        const ShellCorners& corners = m_input.shells[ie].corners;
        const std::size_t kk = localEdgeOf(ie, e);
        const bool forward = corners.at(kk) < corners.at(kk + 1);
        edgeGeometry[e] = edgeMapMoments(corners.count, k, kk, forward, surfaceOf(surfaces[ie].patch), weights[ie]);
    }

    // ---- the shells
    m_shells.resize(ne);
    for (std::size_t ie = 0; ie < ne; ++ie)
    {
        const ShellPartShell& shell = m_input.shells[ie];
        const std::size_t nv = shell.corners.count;
        const ShellReferenceElement& re = referenceElement(nv);
        ShellElementData& el = m_shells[ie];
        el.cornerCount = nv;
        el.nodes.assign(shell.corners.grids.begin(), shell.corners.grids.begin() + static_cast<std::ptrdiff_t>(nv));
        for (std::size_t kk = 0; kk < nv; ++kk)
        {
            el.edges.push_back(m_topology.indexOf(el.nodes[kk], el.nodes[(kk + 1) % nv]));
            el.edgeSigns.push_back(el.nodes[kk] < el.nodes[(kk + 1) % nv] ? 1.0 : -1.0);
        }
        // the isoparametric map: the corners, the edge moments of the shared curves, the projection of the interior
        std::vector<SpaceVector> corners;
        std::vector<std::vector<SpaceVector>> canonicalMoments;
        for (std::size_t kk = 0; kk < nv; ++kk)
        {
            corners.push_back(m_input.grids[el.nodes[kk]].position);
            canonicalMoments.push_back(edgeGeometry[el.edges[kk]]);
        }
        el.weight = weights[ie];
        el.mapCoefficients = mapCoefficients(re, projectionOf(nv), corners, canonicalMoments, el.edgeSigns,
                                             surfaceOf(surfaces[ie].patch), weights[ie]);
        // the section: the material direction given, or the angle θ from the side G1-G2 about the element normal
        const ElementSurface& surface = surfaces[ie];
        SpaceVector direction{};
        if (shell.materialDirection)
        {
            direction = *shell.materialDirection;
        }
        else
        {
            const SpaceVector zAxis = cross3(surface.axes[0], surface.axes[1]);
            const SpaceVector across = cross3(zAxis, surface.side);
            direction = add3(scl3(surface.side, std::cos(shell.theta)), scl3(across, std::sin(shell.theta)));
        }
        const double referenceThickness = shell.section.referenceThickness;
        const std::array<double, 4> gridThicknesses = shell.gridThicknesses.value_or(
            std::array<double, 4>{referenceThickness, referenceThickness, referenceThickness, referenceThickness});
        el.section = ShellSection(shell.section, cornerThicknesses(nv, gridThicknesses), direction);
        el.offset = shell.offset.value_or(shell.section.offset);
        el.axes = {surface.axes[0], surface.axes[1]};
        el.kirchhoff = shell.kirchhoff;
        el.mapReferenceConstants = m_input.mapReferenceConstants;

        // ---- the part unknowns of the local unknowns, and their signs (local = sign × part)
        const std::size_t nx = shellUnknownCount(re);
        el.dofs.assign(nx, 0);
        el.signs.assign(nx, 1.0);
        std::size_t m = 0;
        for (std::size_t i = 0; i < re.h1Count; ++i)
        {
            const FunctionKind& kind = re.h1Kinds[i];
            for (std::size_t c = 0; c < 3; ++c)
            {
                const std::size_t li = 3 * i + c;
                if (kind.place == FunctionKind::Place::Vertex)
                {
                    el.dofs[li] = 6 * el.nodes[kind.entity] + c;
                }
                else if (kind.place == FunctionKind::Place::Edge)
                {
                    el.dofs[li] = edgeBase(el.edges[kind.entity]) + 3 * kind.moment + c;
                    // the moment of odd degree changes sign with the direction of the edge
                    if (el.edgeSigns[kind.entity] < 0.0 && kind.moment % 2 == 1)
                        el.signs[li] = -1.0;
                }
                else
                {
                    el.dofs[li] = elementBase[ie] + 3 * m + c;
                }
            }
            if (kind.isInterior())
                ++m;
        }
        const std::size_t n1i = m;
        std::size_t mc = 0;
        for (std::size_t i = 0; i < re.hcurlCount; ++i)
        {
            const FunctionKind& kind = re.hcurlKinds[i];
            const std::size_t li = 3 * re.h1Count + i;
            if (kind.place == FunctionKind::Place::Edge)
            {
                const std::size_t e = el.edges[kind.entity];
                el.dofs[li] = edgeBase(e) + edgeShearDof(kind.moment);
                // γ_T · t_T + γ_R · t_R = 0 (document 34 of the reference, 2.4): +1 for the first shell of the edge,
                // −1 for the others, times the change of sign of the odd moments with the direction of the edge
                const double joint = m_topology.edgeElements[e].front() == ie ? 1.0 : -1.0;
                const double reversed = el.edgeSigns[kind.entity] < 0.0 && kind.moment % 2 == 1 ? -1.0 : 1.0;
                el.signs[li] = joint * reversed;
            }
            else
            {
                el.dofs[li] = elementBase[ie] + 3 * n1i + mc;
                ++mc;
            }
        }
        for (std::size_t kk = 0; kk < nv; ++kk)
            for (std::size_t l = 0; l <= p; ++l)
                el.dofs[3 * re.h1Count + re.hcurlCount + kk * (p + 1) + l] =
                    edgeBase(el.edges[kk]) + edgeRotationDof(l);
    }
    buildEntities(surfaces);
}

void ShellPart::buildEntities(std::span<const ElementSurface> surfaces)
{
    const std::size_t ng = m_input.grids.size();
    const std::size_t perEdge = edgeDofCount();
    for (std::size_t e = 0; e < m_topology.edges.size(); ++e)
    {
        const std::size_t ie = m_topology.edgeElements[e].front();
        const std::size_t kk = localEdgeOf(ie, e);
        const ReferencePoint middle = referenceEdgePointAndDirection(m_input.shells[ie].corners.count, kk, 0.5).first;
        ShellPartEntity entity;
        entity.kind = ShellPartEntity::Kind::Edge;
        entity.grids = {m_topology.edges[e].lower, m_topology.edges[e].higher};
        entity.edge = e;
        entity.shell = ie;
        entity.dofCount = perEdge;
        entity.firstUnknown = 6 * ng + perEdge * e;
        entity.position = surfaces[ie].patch.evaluate(middle[0], middle[1]);
        m_entities.push_back(entity);
    }
    std::size_t next = 6 * ng + perEdge * m_topology.edges.size();
    for (std::size_t ie = 0; ie < m_input.shells.size(); ++ie)
    {
        const std::size_t nv = m_input.shells[ie].corners.count;
        const ShellReferenceElement& re = referenceElement(nv);
        const ReferencePoint centre = referenceCentre(nv);
        ShellPartEntity entity;
        entity.kind = ShellPartEntity::Kind::Interior;
        entity.shell = ie;
        entity.dofCount = 3 * interiorCount(re.h1Kinds) + interiorCount(re.hcurlKinds);
        entity.firstUnknown = next;
        entity.position = surfaces[ie].patch.evaluate(centre[0], centre[1]);
        next += entity.dofCount;
        m_entities.push_back(entity);
    }
}

void ShellPart::checkElements()
{
    for (std::size_t ie = 0; ie < m_input.shells.size(); ++ie)
        if (std::optional<std::string> refusal = elementRefusal(ie))
            addProblem(ShellPartProblem::Subject::Shell, ie, std::move(*refusal));
    for (std::size_t il = 0; il < m_input.lines.size(); ++il)
    {
        if (!m_lineEdges[il])
            continue;
        std::variant<LineData, std::string> data = lineData(il);
        if (const std::string* message = std::get_if<std::string>(&data))
            addProblem(ShellPartProblem::Subject::Line, il, *message);
    }
}

ShellElementData ShellPart::element(std::size_t shell) const
{
    ShellElementData data = m_shells[shell];
    ElementJets jets = elementJets(referenceElement(data.cornerCount), data.weight);
    data.volumeJets = std::move(jets.volume);
    data.edgeJets = std::move(jets.edge);
    data.edgeWeights = std::move(jets.edgeWeight);
    return data;
}

std::variant<ShellElementData, std::string> ShellPart::build(std::size_t shell, bool withMatrices, bool withMass) const
{
    const ShellPartShell& input = m_input.shells[shell];
    const std::size_t nv = input.corners.count;
    // a singular section: no bending stiffness, or a thickness that is not positive at a corner
    if (!inverse33(input.section.d))
        return std::string("singular section: it has no bending stiffness (its matrix D is singular)");
    const std::array<double, 4>& thickness = m_shells[shell].section.thickness();
    const std::array<std::size_t, 4> cornerSlot =
        nv == 3 ? std::array<std::size_t, 4>{0, 1, 3, 3} : std::array<std::size_t, 4>{0, 1, 2, 3};
    for (std::size_t k = 0; k < nv; ++k)
        if (!(thickness[cornerSlot[k]] > 0.0))
            return std::string("singular section: its thickness at a corner is not positive");
    ShellElementData data = element(shell);
    const ShellReferenceElement& re = referenceElement(nv);
    const ShellElementDefinition def = data.definition();
    // an element that turns too much: the frame of its constant states is carried from the normal of its corners by
    // the minimal rotation, which needs the normal to stay within 90° of it
    const SpaceVector zAxis = unit3(cross3(data.axes[0], data.axes[1]));
    for (std::size_t q = 0; q < re.points.size(); ++q)
        if (!(dot3(zAxis, volumeGeometry(def, re.h1Count, q).nu) > 0.0))
            return std::string(
                "the element turns too much: its normal turns by 90 degrees or more from the normal "
                "of its corners");
    if (!withMatrices)
        return data;
    std::variant<ShellElementMatrices, std::string> matrices = buildShellElement(re, def, withMass);
    if (const std::string* message = std::get_if<std::string>(&matrices))
        return "degenerate element: " + *message;
    data.matrices = std::get<ShellElementMatrices>(std::move(matrices));
    return data;
}

std::variant<ShellElementData, std::string> ShellPart::elementWithMatrices(std::size_t shell) const
{
    return elementWithMatrices(shell, true);
}

std::variant<ShellElementData, std::string> ShellPart::elementWithMatrices(std::size_t shell, bool withMass) const
{
    const bool mass = withMass && m_input.withMass;
    {
        const std::lock_guard<std::mutex> lock(m_cache->mutex);
        if (shell < m_cache->elements.size() && m_cache->elements[shell] &&
            (!mass || m_cache->elements[shell]->matrices.mass))
            return *m_cache->elements[shell];
    }
    std::variant<ShellElementData, std::string> built = build(shell, true, mass);
    const ShellElementData* data = std::get_if<ShellElementData>(&built);
    if (data == nullptr || m_input.matrixCacheBytes == 0)
        return built;
    // The bytes of the element: its matrices and its tables (the rest is small).
    const auto bytesOf = [](const Matrix& m) { return m.rows() * m.columns() * sizeof(double); };
    const ShellElementMatrices& matrices = data->matrices;
    std::size_t bytes = bytesOf(matrices.stiffness) + bytesOf(matrices.pairing) + bytesOf(matrices.complianceInverse) +
                        bytesOf(matrices.reggeInterpolation) + bytesOf(matrices.reggeFunctionalInverse) +
                        (matrices.mass ? bytesOf(*matrices.mass) : 0) + data->volumeJets.size() * sizeof(ReferenceJet);
    for (const std::vector<EdgeJet>& jets : data->edgeJets)
        bytes += jets.size() * sizeof(EdgeJet);
    const std::lock_guard<std::mutex> lock(m_cache->mutex);
    if (m_cache->elements.size() < m_shells.size())
    {
        m_cache->elements.resize(m_shells.size());
        m_cache->entryBytes.resize(m_shells.size(), 0);
    }
    std::shared_ptr<const ShellElementData>& kept = m_cache->elements[shell];
    // Another thread kept it meanwhile (with its mass when the mass is asked).
    if (kept && (!mass || kept->matrices.mass))
        return built;
    // An element kept without its mass gives its place to the one with it.
    const std::size_t total = m_cache->bytes - m_cache->entryBytes[shell] + bytes;
    if (total <= m_input.matrixCacheBytes)
    {
        kept = std::make_shared<const ShellElementData>(*data);
        m_cache->entryBytes[shell] = bytes;
        m_cache->bytes = total;
    }
    return built;
}

std::optional<std::string> ShellPart::elementRefusal(std::size_t shell) const
{
    std::variant<ShellElementData, std::string> built = elementWithMatrices(shell, false);
    if (std::string* message = std::get_if<std::string>(&built))
        return std::move(*message);
    return std::nullopt;
}

std::span<const std::size_t> ShellPart::localUnknowns(std::size_t shell) const
{
    return m_shells[shell].dofs;
}

std::span<const double> ShellPart::localSigns(std::size_t shell) const
{
    return m_shells[shell].signs;
}

std::vector<std::pair<std::size_t, std::size_t>> ShellPart::shellsAt(std::size_t grid) const
{
    std::vector<std::pair<std::size_t, std::size_t>> at;
    for (std::size_t ie = 0; ie < m_shells.size(); ++ie)
    {
        const std::vector<std::size_t>& nodes = m_shells[ie].nodes;
        const auto found = std::find(nodes.begin(), nodes.end(), grid);
        if (found != nodes.end())
            at.emplace_back(ie, static_cast<std::size_t>(found - nodes.begin()));
    }
    return at;
}

ShellPartUnknown ShellPart::describe(std::size_t unknown) const
{
    if (unknown < firstEntityUnknown())
        return {ShellPartUnknown::Kind::GridComponent, unknown / 6, unknown % 6};
    if (unknown >= m_unknownCount)
        throw base::InternalError("shell part: unknown out of range");
    // the last entity that starts at or before the unknown
    const auto after = std::upper_bound(m_entities.begin(), m_entities.end(), unknown,
                                        [](std::size_t u, const ShellPartEntity& e) { return u < e.firstUnknown; });
    const auto entity = static_cast<std::size_t>(after - m_entities.begin()) - 1;
    return {ShellPartUnknown::Kind::EntityDof, entity, unknown - m_entities[entity].firstUnknown};
}

std::vector<EdgeGeometry> ShellPart::geometryReport() const
{
    if (m_input.shells.empty())
        return {};
    // the points of the edge rule (the same on the triangle and on the quadrilateral)
    const std::vector<double>& tau = referenceElement(m_input.shells.front().corners.count).edgeParameters;
    // the isoparametric surface of the elements at the points of their edges; the report asks for the points of one
    // element after the other, so the tables of the last element are kept
    struct Cache
    {
        std::size_t shell = 0;
        std::optional<ShellElementData> data;
    } cache;
    const EdgeSurfaceEvaluator point = [&](std::size_t shell, std::size_t localEdge, double t)
    {
        if (!cache.data || cache.shell != shell)
        {
            cache.data = element(shell);
            cache.shell = shell;
        }
        const ShellReferenceElement& re = referenceElement(cache.data->cornerCount);
        const ShellElementDefinition def = cache.data->definition();
        // the tabulated point of the edge rule (the reference reads them), else the map at the parameter
        const auto at = std::find(re.edgeParameters.begin(), re.edgeParameters.end(), t);
        if (at != re.edgeParameters.end())
        {
            const auto g = static_cast<std::size_t>(at - re.edgeParameters.begin());
            const PointGeometry geometry = edgeGeometry(def, re.h1Count, localEdge, g);
            return EdgeSurfacePoint{geometry.x, geometry.nu};
        }
        const ReferencePoint uv = referenceEdgePointAndDirection(re.cornerCount, localEdge, t).first;
        const PointGeometry geometry = geometryAt(re, def, uv[0], uv[1]);
        return EdgeSurfacePoint{geometry.x, geometry.nu};
    };
    return mixedHybrid::geometryReport(m_reading, m_topology, tau, point);
}

} // namespace oos::elements::mixedHybrid
