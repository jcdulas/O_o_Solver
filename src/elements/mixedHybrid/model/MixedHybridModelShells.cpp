// The shells of the model of the mixed hybrid formulation (MixedHybridModel.hpp): their reading from the six shell
// cards, the ShellPart that numbers their edges and interiors, and what the formulation tells the solver about them
// (the rotations of their grid points, the supports read along their edges, the affine fields on their unknowns).

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementFormulation.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/model/MixedHybridModel.hpp"
#include "oos/elements/mixedHybrid/model/ShellSectionAdapter.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"
#include "oos/elements/shells/ShellConnectivity.hpp"
#include "oos/elements/shells/ShellSection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <string_view>

namespace oos::elements::mixedHybrid
{

namespace
{

// The shell cards of the formulation, with their corner count and whether their connectivity lists mid-side nodes.
struct ShellCard
{
    std::string_view card;
    std::size_t corners = 4;
    bool hasMidSideNodes = false;
};
constexpr std::array<ShellCard, 6> SHELL_CARDS{{{"ShellQuad4", 4, false},
                                                {"ShellTria3", 3, false},
                                                {"ShellQuad4Drilling", 4, false},
                                                {"ShellTria3Drilling", 3, false},
                                                {"ShellQuad8", 4, true},
                                                {"ShellTria6", 3, true}}};

// The material frame ID of a shell given its THETA (O_o_Reader: −1 when field 7 holds THETA).
constexpr std::int32_t NO_MATERIAL_FRAME = -1;

// The relative threshold of the rows of the mean corner rotation (the reference's tie 6: terms below 1e-14 of the
// largest dropped).
constexpr double ROTATION_ROW_THRESHOLD = 1.0e-14;
// The bytes of shell elements with their matrices the ShellPart keeps once built (ShellPartInput::matrixCacheBytes):
// about 250 shells of degree 4 with their mass, so that the check, the stiffness, the mass, the loads and the recovery
// of a model of that size build each element once (a few milliseconds each).
constexpr std::size_t SHELL_MATRIX_CACHE_BYTES = std::size_t{128} << 20;

// The geometry report (spec section 10.5): the edges along which the normals of two shells differ by more than this
// angle (degrees) are named, at most GEOMETRY_REPORT_EDGES of them, the largest angles first. Below it, a kink is the
// round-off of the reading of a smooth surface.
constexpr double GEOMETRY_REPORT_ANGLE = 1.0;
constexpr std::size_t GEOMETRY_REPORT_EDGES = 10;
// The rounding of an edge (the angle between the normal of the surface read along it and the plane of the corners of
// a shell of the edge) above which the report names it: a fold that the declarations do not name is rounded to about
// half its angle; a smooth surface meshed finely enough turns by a few degrees over an element.
constexpr double ROUNDING_REPORT_ANGLE = 15.0;
// The points of each edge where the rounding is read: τ = i / (ROUNDING_POINTS − 1).
constexpr std::size_t ROUNDING_POINTS = 5;

base::Diagnostic shellProblem(std::string_view card, std::int32_t elementId, std::string field, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = problem_code::UNSUPPORTED_FIELD;
    diagnostic.card = std::string(card);
    diagnostic.id = elementId;
    diagnostic.field = std::move(field);
    diagnostic.message = std::move(message);
    return diagnostic;
}

// The grid rows of the corners and mid-side nodes of the shell of batch row `row` (−1 for an absent mid-side node).
Resolution<std::pair<std::array<std::int32_t, 4>, std::array<std::int32_t, 4>>> shellNodes(
    const ElementBatchView& view, const ShellConnectivity& connectivity, std::int64_t row, const ShellCard& card)
{
    using Nodes = std::pair<std::array<std::int32_t, 4>, std::array<std::int32_t, 4>>;
    using Result = Resolution<Nodes>;
    Nodes nodes{{}, {-1, -1, -1, -1}};
    if (!card.hasMidSideNodes)
    {
        if (card.corners == 4)
        {
            Resolution<elements::ShellCorners<4>> corners = resolveShellCorners<4>(view, connectivity, row);
            if (!corners)
                return Result::failure(std::move(corners.problem));
            nodes.first = corners.value->rows;
        }
        else
        {
            Resolution<elements::ShellCorners<3>> corners = resolveShellCorners<3>(view, connectivity, row);
            if (!corners)
                return Result::failure(std::move(corners.problem));
            nodes.first = {corners.value->rows[0], corners.value->rows[1], corners.value->rows[2], -1};
        }
        return Result::success(nodes);
    }
    // A shell with mid-side nodes: its corners required, each mid-side node optional (an ID 0: absent).
    const std::span<const std::int32_t> ids = connectivity.ids[row];
    const std::span<const std::int32_t> rows = connectivity.rows[row];
    for (std::size_t i = 0; i < 2 * card.corners && i < ids.size(); ++i)
    {
        const bool isCorner = i < card.corners;
        if (!isCorner && ids[i] == 0)
            continue;
        const std::optional<std::int32_t> gridRow = i < rows.size() && rows[i] >= 0
                                                        ? std::optional<std::int32_t>(rows[i])
                                                        : view.index().points().findGrid(ids[i]);
        if (!gridRow)
            return Result::failure(elementProblem(
                view, row, problem_code::MISSING_POINT, std::string("node").append(std::to_string(i + 1)),
                base::concatenate({"expected the ID of a grid point, found ", std::to_string(ids[i]),
                                   ", which the model does not define; define it or correct the card"})));
        (isCorner ? nodes.first[i] : nodes.second[i - card.corners]) = checkedGridRow(view, *gridRow, ids[i]);
    }
    for (std::size_t i = card.corners; i < 4; ++i)
        nodes.first[i] = -1;
    return Result::success(nodes);
}

} // namespace

void MixedHybridModel::readShells(const SnapshotIndex& index)
{
    const input::ModelSnapshot& snapshot = index.snapshot();
    for (const ShellCard& card : SHELL_CARDS)
    {
        const input::ElementBatch* batch = snapshot.findElementBatch(card.card);
        if (batch == nullptr || batch->rowCount() == 0)
            continue;
        const ElementBatchView view(*batch, index);
        const ShellConnectivity connectivity(view);
        const ShellSectionColumns sections(view);
        const std::span<const double> theta = view.columns().values<double>("materialAngle");
        const std::span<const std::int32_t> matFrameId = view.columns().values<std::int32_t>("matFrameId");
        const std::span<const double> offset = view.columns().values<double>("offset", base::NullPolicy::Allow);
        const base::ValidityView isOffsetGiven = view.columns().validity("offset");
        const std::span<const std::int32_t> thicknessFlag = view.columns().values<std::int32_t>("thicknessFlag");
        const base::FixedListView<double> cornerThickness = view.columns().fixedLists<double>("cornerThickness");
        std::vector<std::int32_t>& shellOfRow =
            m_shellOfRow
                .emplace(std::string(card.card),
                         std::vector<std::int32_t>(static_cast<std::size_t>(batch->rowCount()), -1))
                .first->second;
        for (std::int64_t row = 0; row < view.rowCount(); ++row)
        {
            const auto r = static_cast<std::size_t>(row);
            const std::int32_t id = view.elementId(row);
            auto nodes = shellNodes(view, connectivity, row, card);
            if (!nodes)
            {
                addElementProblem(card.card, row, std::move(nodes.problem));
                continue;
            }
            Resolution<elements::ShellSection> classic = resolveShellSection(view, sections, row);
            if (!classic)
            {
                addElementProblem(card.card, row, std::move(classic.problem));
                continue;
            }
            std::variant<Section, std::string> section = mixedSectionOf(*classic.value);
            if (const std::string* message = std::get_if<std::string>(&section))
            {
                addElementProblem(card.card, row, shellProblem(card.card, id, "propId", *message));
                continue;
            }
            MixedShell shell;
            shell.card = card.card;
            shell.batchRow = row;
            shell.elementId = id;
            shell.cornerCount = card.corners;
            shell.gridRows = nodes.value->first;
            shell.midRows = nodes.value->second;
            shell.classicSection = *classic.value;
            shell.part.elementId = id;
            shell.part.section = std::get<Section>(std::move(section));
            shell.part.kirchhoff = !classic.value->hasShearFlexibility;
            // The corner thicknesses (a laminate keeps its own).
            if (!classic.value->isLaminate())
            {
                Resolution<std::array<double, 4>> thickness4;
                std::array<double, 4> corners{};
                if (card.corners == 4)
                {
                    Resolution<std::array<double, 4>> t =
                        resolveCornerThicknesses<4>(view, row, *classic.value, thicknessFlag[r], cornerThickness[row]);
                    if (!t)
                    {
                        addElementProblem(card.card, row, std::move(t.problem));
                        continue;
                    }
                    corners = *t.value;
                }
                else
                {
                    Resolution<std::array<double, 3>> t =
                        resolveCornerThicknesses<3>(view, row, *classic.value, thicknessFlag[r], cornerThickness[row]);
                    if (!t)
                    {
                        addElementProblem(card.card, row, std::move(t.problem));
                        continue;
                    }
                    corners = {(*t.value)[0], (*t.value)[1], (*t.value)[2], 0.0};
                }
                shell.part.gridThicknesses = corners;
            }
            // The material axis: THETA from G1→G2 (the ShellPart applies it as the reference does), or the x axis of
            // the material system MCID, which the element projects on its tangent plane.
            if (matFrameId[r] == NO_MATERIAL_FRAME)
                shell.part.theta = theta[r] * std::numbers::pi / 180.0;
            else
            {
                Resolution<input::CoordinateFrame> system =
                    resolveRectangularFrame(view, row, matFrameId[r], "matFrameId");
                if (!system)
                {
                    addElementProblem(card.card, row, std::move(system.problem));
                    continue;
                }
                const base::Mat3& axes = system.value->axes;
                shell.part.materialDirection = SpaceVector{axes(0, 0), axes(1, 0), axes(2, 0)};
            }
            if (isOffsetGiven.isValid(row))
                shell.part.offset = offset[r];
            shellOfRow[r] = static_cast<std::int32_t>(m_shells.size());
            m_shells.push_back(std::move(shell));
        }
    }
    // The ShellPart takes the shells by increasing element ID: that order is also their index there.
    std::vector<std::size_t> order(m_shells.size());
    for (std::size_t i = 0; i < order.size(); ++i)
        order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return m_shells[a].elementId < m_shells[b].elementId; });
    std::vector<MixedShell> sorted;
    sorted.reserve(m_shells.size());
    for (const std::size_t i : order)
        sorted.push_back(std::move(m_shells[i]));
    m_shells = std::move(sorted);
    for (auto& [card, rows] : m_shellOfRow)
        std::fill(rows.begin(), rows.end(), -1);
    for (std::size_t i = 0; i < m_shells.size(); ++i)
    {
        m_shells[i].partIndex = i;
        m_shellOfRow.at(std::string(m_shells[i].card))[static_cast<std::size_t>(m_shells[i].batchRow)] =
            static_cast<std::int32_t>(i);
    }
}

void MixedHybridModel::buildShellPart(const SnapshotIndex& index)
{
    if (m_shells.empty())
        return;
    const input::ModelSnapshot& snapshot = index.snapshot();
    const input::NodeTable& nodes = snapshot.nodes();
    const base::FixedListView<double> positions = nodes.positionsInBasic();
    const std::span<const std::int32_t> ids = nodes.ids();

    ShellPartInput input;
    input.withMass = true;
    input.matrixCacheBytes = SHELL_MATRIX_CACHE_BYTES;
    for (const std::int32_t row : m_gridRows)
    {
        const std::span<const double> xyz = positions[row];
        input.grids.push_back({{xyz[0], xyz[1], xyz[2]}, ids[static_cast<std::size_t>(row)]});
    }
    for (MixedShell& shell : m_shells)
    {
        std::array<std::size_t, 4> corners{};
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
            corners[k] = static_cast<std::size_t>(m_gridIndexOfRow.at(shell.gridRows[k]));
        shell.part.corners = shell.cornerCount == 4
                                 ? ShellCorners::quadrilateral(corners[0], corners[1], corners[2], corners[3])
                                 : ShellCorners::triangle(corners[0], corners[1], corners[2]);
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
            shell.part.midside[k] = shell.midRows[k] >= 0 ? m_gridIndexOfRow.at(shell.midRows[k]) : -1;
        input.shells.push_back(shell.part);
    }
    // The lines on the edges of the shells, by increasing element ID.
    std::vector<std::size_t> shellLines;
    for (std::size_t l = 0; l < m_lines.size(); ++l)
        if (m_lines[l].shellLine == 0) // marked by buildGridsAndEdges
            shellLines.push_back(l);
    std::sort(shellLines.begin(), shellLines.end(),
              [&](std::size_t a, std::size_t b) { return m_lines[a].elementId < m_lines[b].elementId; });
    for (std::size_t i = 0; i < shellLines.size(); ++i)
    {
        MixedLine& line = m_lines[shellLines[i]];
        line.shellLine = static_cast<std::int32_t>(i);
        input.lines.push_back({line.elementId, line.element, line.section});
    }
    for (const std::int32_t grid : m_lineGrids)
        input.lineGrids.insert(static_cast<std::size_t>(grid));
    // The declarations of the program, by the model's grid and shell indices.
    if (const std::optional<input::FormulationGeometry>& geometry = snapshot.formulationGeometry())
    {
        const auto gridIndexOfId = [&](std::int32_t id) -> std::int32_t
        {
            const std::optional<std::int32_t> row = index.points().findGrid(id);
            return row ? gridIndexOfRow(*row) : -1;
        };
        for (const input::GridEdge& edge : geometry->rawEdges)
        {
            const std::int32_t a = gridIndexOfId(edge[0]);
            const std::int32_t b = gridIndexOfId(edge[1]);
            if (a < 0 || b < 0)
            {
                base::Diagnostic diagnostic;
                diagnostic.code = "geometry-declaration-unknown";
                diagnostic.message = base::concatenate(
                    {"expected the raw edges of the geometry declarations between grid points of the shells or lines "
                     "of ",
                     MIXED_HYBRID_FORMULATION, ", found the edge between grid points ", std::to_string(edge[0]),
                     " and ", std::to_string(edge[1]), "; correct the declaration"});
                m_problems.push_back(std::move(diagnostic));
                continue;
            }
            input.rawEdges.insert(EdgeKey::of(static_cast<std::size_t>(a), static_cast<std::size_t>(b)));
        }
        for (const std::int32_t elementId : geometry->rawElements)
        {
            const auto found = std::find_if(m_shells.begin(), m_shells.end(),
                                            [&](const MixedShell& shell) { return shell.elementId == elementId; });
            if (found == m_shells.end())
            {
                base::Diagnostic diagnostic;
                diagnostic.code = "geometry-declaration-unknown";
                diagnostic.message =
                    base::concatenate({"expected the raw elements of the geometry declarations to be shells of ",
                                       MIXED_HYBRID_FORMULATION, ", found element ", std::to_string(elementId),
                                       "; correct the declaration"});
                m_problems.push_back(std::move(diagnostic));
                continue;
            }
            input.rawElements.push_back(found->partIndex);
        }
    }

    m_shellPart = std::make_unique<ShellPart>(std::move(input));
    for (const ShellPartProblem& problem : m_shellPart->problems())
    {
        switch (problem.subject)
        {
        case ShellPartProblem::Subject::Shell:
        {
            const MixedShell& shell = m_shells[problem.index];
            addElementProblem(shell.card, shell.batchRow,
                              shellProblem(shell.card, shell.elementId, "", problem.message));
            break;
        }
        case ShellPartProblem::Subject::Line:
        {
            for (const MixedLine& line : m_lines)
                if (line.shellLine == static_cast<std::int32_t>(problem.index))
                    addElementProblem(line.card, line.batchRow,
                                      shellProblem(line.card, line.elementId, "", problem.message));
            break;
        }
        case ShellPartProblem::Subject::Model:
        {
            base::Diagnostic diagnostic;
            diagnostic.code = "invalid-geometry";
            diagnostic.message = problem.message;
            m_problems.push_back(std::move(diagnostic));
            break;
        }
        }
    }
    if (!m_shellPart->usable())
        return;

    // Its entities: the edges of the shells, then their interiors, in both layouts.
    for (const ShellPartEntity& entity : m_shellPart->entities())
    {
        EntityName name;
        const base::Vec3 position{entity.position[0], entity.position[1], entity.position[2]};
        if (entity.kind == ShellPartEntity::Kind::Edge)
        {
            name.kind = EntityKind::Edge;
            name.gridIds = {ids[static_cast<std::size_t>(m_gridRows[entity.grids[0]])],
                            ids[static_cast<std::size_t>(m_gridRows[entity.grids[1]])], 0, 0};
        }
        else
        {
            name.kind = EntityKind::Interior;
            name.elementId = m_shells[entity.shell].elementId;
        }
        m_shellEntityRow.push_back(m_staticLayout.add(static_cast<int>(entity.dofCount), position, name));
        m_modalLayout.add(static_cast<int>(entity.dofCount), position, name);
    }

    reportGeometry();

    // The mid-side grids that only the edge of their shell carries: their translations follow the edge.
    std::set<std::int32_t> cornerGrids;
    for (const MixedShell& shell : m_shells)
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
            cornerGrids.insert(m_gridIndexOfRow.at(shell.gridRows[k]));
    for (const MixedShell& shell : m_shells)
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
        {
            if (shell.midRows[k] < 0)
                continue;
            const std::int32_t grid = m_gridIndexOfRow.at(shell.midRows[k]);
            if (cornerGrids.count(grid) != 0 || m_lineGrids.count(grid) != 0 || m_midSideGrids.count(grid) != 0)
                continue;
            const ReferencePoint middle = referenceEdgePointAndDirection(shell.cornerCount, k, 0.5).first;
            m_midSideGrids.emplace(grid, MidSideGrid{shell.partIndex, middle[0], middle[1]});
        }
}

std::map<EdgeKey, double> MixedHybridModel::edgeRoundings() const
{
    // The largest angle, along each edge, between the normal of the surface read and the normal of the plane of the
    // corners of each shell of the edge (the z axis of its element axes).
    std::map<EdgeKey, double> roundings;
    for (const MixedShell& shell : m_shells)
    {
        const ShellElementData element = m_shellPart->element(shell.partIndex);
        const ShellReferenceElement& reference = m_shellPart->referenceElement(shell.cornerCount);
        const ShellElementDefinition definition = element.definition();
        const SpaceVector z = unit3(cross3(element.axes[0], element.axes[1]));
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
        {
            const EdgeKey key =
                EdgeKey::of(static_cast<std::size_t>(m_gridIndexOfRow.at(shell.gridRows[k])),
                            static_cast<std::size_t>(m_gridIndexOfRow.at(shell.gridRows[(k + 1) % shell.cornerCount])));
            double& largest = roundings[key];
            for (std::size_t i = 0; i < ROUNDING_POINTS; ++i)
            {
                const double tau = static_cast<double>(i) / static_cast<double>(ROUNDING_POINTS - 1);
                const ReferencePoint point = referenceEdgePointAndDirection(shell.cornerCount, k, tau).first;
                const SpaceVector& normal = geometryAt(reference, definition, point[0], point[1]).nu;
                largest = std::max(largest, std::acos(std::clamp(dot3(normal, z), -1.0, 1.0)));
            }
        }
    }
    return roundings;
}

void MixedHybridModel::reportGeometry()
{
    // What the reading of the mesh made of the surface of the shells (spec section 10.5): the edges declared raw, the
    // edges where it left the normals of two shells apart (a kink of a raw edge is wanted; elsewhere it shows a fold
    // the declarations do not name, or a mesh too coarse for its curvature), and those where it turned the surface far
    // from the plane of the corners of a shell (a fold that the declarations do not name is rounded).
    std::vector<EdgeGeometry> edges = m_shellPart->geometryReport();
    const std::map<EdgeKey, double> roundings = edgeRoundings();
    std::size_t rawCount = 0;
    for (const EdgeGeometry& edge : edges)
        rawCount += edge.raw ? 1 : 0;
    const auto degrees = [](double radians) { return radians * 180.0 / std::numbers::pi; };
    const auto edgeName = [&](const EdgeKey& key)
    {
        return base::concatenate({"the edge between grid points ", std::to_string(m_gridIds[key.lower]), " and ",
                                  std::to_string(m_gridIds[key.higher])});
    };
    const auto roundingOf = [&](const EdgeGeometry& edge)
    {
        const auto found = roundings.find(edge.grids);
        return found == roundings.end() ? 0.0 : found->second;
    };
    std::sort(edges.begin(), edges.end(), [](const EdgeGeometry& a, const EdgeGeometry& b) { return a.kink > b.kink; });
    const auto mostRounded =
        std::max_element(edges.begin(), edges.end(),
                         [&](const EdgeGeometry& a, const EdgeGeometry& b) { return roundingOf(a) < roundingOf(b); });
    base::Diagnostic summary;
    summary.severity = base::Severity::Information;
    summary.code = "geometry-report";
    summary.message = base::concatenate(
        {"the shells of ", MIXED_HYBRID_FORMULATION, " have ", std::to_string(edges.size()), " edges, ",
         std::to_string(rawCount), " declared raw; ",
         edges.empty() || edges.front().kink == 0.0
             ? std::string("the normals of the shells agree along every edge")
             : base::concatenate({"the largest angle between the normals of two shells along an edge is ",
                                  std::to_string(degrees(edges.front().kink)), " degrees, along ",
                                  edgeName(edges.front().grids)}),
         edges.empty()
             ? std::string()
             : base::concatenate({"; the surface read turns from the plane of the corners of a shell by up to ",
                                  std::to_string(degrees(roundingOf(*mostRounded))), " degrees, along ",
                                  edgeName(mostRounded->grids)})});
    m_problems.push_back(std::move(summary));
    for (std::size_t i = 0; i < edges.size() && i < GEOMETRY_REPORT_EDGES; ++i)
    {
        const EdgeGeometry& edge = edges[i];
        if (degrees(edge.kink) <= GEOMETRY_REPORT_ANGLE)
            break;
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Information;
        diagnostic.code = "geometry-kink";
        diagnostic.card = "Node";
        diagnostic.id = m_gridIds[edge.grids.lower];
        diagnostic.message = base::concatenate(
            {"the normals of the ", std::to_string(edge.elements), " shells along ", edgeName(edge.grids),
             " differ by up to ", std::to_string(degrees(edge.kink)), " degrees",
             edge.raw ? " (declared raw)"
                      : " (not declared raw: a fold the declarations do not name, or a mesh too coarse for the "
                        "curvature there)"});
        m_problems.push_back(std::move(diagnostic));
    }
    std::sort(edges.begin(), edges.end(),
              [&](const EdgeGeometry& a, const EdgeGeometry& b) { return roundingOf(a) > roundingOf(b); });
    for (std::size_t i = 0; i < edges.size() && i < GEOMETRY_REPORT_EDGES; ++i)
    {
        const EdgeGeometry& edge = edges[i];
        if (degrees(roundingOf(edge)) <= ROUNDING_REPORT_ANGLE)
            break;
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Information;
        diagnostic.code = "geometry-rounding";
        diagnostic.card = "Node";
        diagnostic.id = m_gridIds[edge.grids.lower];
        diagnostic.message = base::concatenate(
            {"the surface read along ", edgeName(edge.grids), " turns by up to ",
             std::to_string(degrees(roundingOf(edge))),
             " degrees from the plane of the corners of its shells: a fold the declarations do not name is rounded "
             "(declare it raw), or the mesh is too coarse for the curvature there"});
        m_problems.push_back(std::move(diagnostic));
    }
}

const MixedShell* MixedHybridModel::shellOf(std::string_view card, std::int64_t batchRow) const
{
    const auto found = m_shellOfRow.find(card);
    if (found == m_shellOfRow.end() || batchRow < 0 || static_cast<std::size_t>(batchRow) >= found->second.size() ||
        m_shellPart == nullptr || !m_shellPart->usable())
        return nullptr;
    const std::int32_t shell = found->second[static_cast<std::size_t>(batchRow)];
    return shell < 0 ? nullptr : &m_shells[static_cast<std::size_t>(shell)];
}

std::int32_t MixedHybridModel::gridIndexOfRow(std::int32_t row) const
{
    const auto found = m_gridIndexOfRow.find(row);
    return found == m_gridIndexOfRow.end() ? -1 : found->second;
}

MixedHybridModel::LocalDofs MixedHybridModel::partDofs(std::span<const std::size_t> unknowns,
                                                       std::span<const double> signs) const
{
    LocalDofs local;
    for (std::size_t i = 0; i < unknowns.size(); ++i)
    {
        const ShellPartUnknown unknown = m_shellPart->describe(unknowns[i]);
        if (unknown.kind == ShellPartUnknown::Kind::GridComponent)
        {
            const std::int32_t row = m_gridRows[unknown.index];
            local.dofs.push_back(ElementDof::grid(row, static_cast<int>(unknown.component) + 1));
            local.basicGridRows.push_back(row);
        }
        else
        {
            local.dofs.push_back(
                ElementDof::entity(m_shellEntityRow[unknown.index], static_cast<int>(unknown.component)));
            local.basicGridRows.push_back(-1);
        }
        local.signs.push_back(i < signs.size() ? signs[i] : 1.0);
    }
    return local;
}

void MixedHybridModel::appendPartTerms(std::span<const ShellPartTerm> partTerms, double factor,
                                       std::vector<FormulationRelationTerm>& terms) const
{
    for (const ShellPartTerm& term : partTerms)
    {
        const ShellPartUnknown unknown = m_shellPart->describe(term.unknown);
        const double coefficient = factor * term.coefficient;
        if (unknown.kind == ShellPartUnknown::Kind::EntityDof)
        {
            terms.push_back({ElementDof::entity(m_shellEntityRow[unknown.index], static_cast<int>(unknown.component)),
                             coefficient});
            continue;
        }
        // A grid component in the basic system: u_basic(b) = Σ_s R(b, s) u_CD(s).
        const base::Mat3& frame = m_gridFrames[unknown.index];
        const std::size_t b = unknown.component % 3;
        const int first = unknown.component < 3 ? 1 : 4;
        for (std::size_t s = 0; s < 3; ++s)
            if (frame(b, s) != 0.0)
                terms.push_back({ElementDof::grid(m_gridRows[unknown.index], first + static_cast<int>(s)),
                                 coefficient * frame(b, s)});
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// What the formulation tells the solver

void MixedHybridModel::ties(EntityLayout::Kind, std::span<const HeldGridComponent> held,
                            std::vector<ElementDof>& dependents, std::vector<std::size_t>& offsets,
                            std::vector<FormulationRelationTerm>& terms, base::DiagnosticSink& sink) const
{
    // Ties 1 to 4: the solids with the shells and the lines alone.
    solidTies(held, dependents, offsets, terms, sink);
    if (m_shellPart == nullptr || !m_shellPart->usable())
        return;
    // Tie 6: a line alone that ends at a grid point of shells where no line lies on a shell edge turns with the shells
    // there: its rotation Λ is the mean rotation of the shells at their corners (spec section 3.4). The components a
    // support holds stay held.
    std::set<std::int32_t> onShellEdges;
    std::set<std::int32_t> alone;
    for (const MixedLine& line : m_lines)
        for (const std::int32_t row : line.gridRows)
            (line.shellLine >= 0 ? onShellEdges : alone).insert(m_gridIndexOfRow.at(row));
    for (const std::int32_t grid : alone)
    {
        if (onShellEdges.count(grid) != 0 || m_shellPart->shellsAt(static_cast<std::size_t>(grid)).empty())
            continue;
        const std::array<std::vector<ShellPartTerm>, 3> rows =
            m_shellPart->meanRotationRows(static_cast<std::size_t>(grid), ROTATION_ROW_THRESHOLD);
        const base::Mat3& frame = m_gridFrames[static_cast<std::size_t>(grid)];
        const std::int32_t row = m_gridRows[static_cast<std::size_t>(grid)];
        for (int s = 0; s < 3; ++s)
        {
            const bool isHeld = std::any_of(held.begin(), held.end(), [&](const HeldGridComponent& h)
                                            { return h.gridRow == row && h.component == 4 + s; });
            if (isHeld)
                continue;
            // θ_CD(s) = Σ_b R(b, s) θ_basic(b).
            dependents.push_back(ElementDof::grid(row, 4 + s));
            for (std::size_t b = 0; b < 3; ++b)
                appendPartTerms(rows[b], frame(b, static_cast<std::size_t>(s)), terms);
            offsets.push_back(terms.size());
        }
    }
}

void MixedHybridModel::shellDeterminedComponents(std::vector<HeldGridComponent>& components) const
{
    if (m_shellPart == nullptr || !m_shellPart->usable())
        return;
    // The rotations of the corner grid points of the shells that no line element has (section 5.4).
    std::set<std::int32_t> corners;
    for (const MixedShell& shell : m_shells)
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
            corners.insert(m_gridIndexOfRow.at(shell.gridRows[k]));
    for (const std::int32_t grid : corners)
    {
        if (m_lineGrids.count(grid) != 0)
            continue;
        for (int component = 4; component <= 6; ++component)
            components.push_back({m_gridRows[static_cast<std::size_t>(grid)], component, 0.0});
    }
    // The translations of the mid-side grid points that only their edge carries.
    for (const auto& [grid, middle] : m_midSideGrids)
        for (int component = 1; component <= 3; ++component)
            components.push_back({m_gridRows[static_cast<std::size_t>(grid)], component, 0.0});
}

void MixedHybridModel::shellDeterminedRow(std::int32_t grid, int component,
                                          std::vector<FormulationRelationTerm>& terms) const
{
    const base::Mat3& frame = m_gridFrames[static_cast<std::size_t>(grid)];
    const auto s = static_cast<std::size_t>((component - 1) % 3);
    if (component >= 4)
    {
        const std::array<std::vector<ShellPartTerm>, 3> rows =
            m_shellPart->meanRotationRows(static_cast<std::size_t>(grid), ROTATION_ROW_THRESHOLD);
        for (std::size_t b = 0; b < 3; ++b)
            appendPartTerms(rows[b], frame(b, s), terms);
        return;
    }
    const MidSideGrid& middle = m_midSideGrids.at(grid);
    const ShellDisplacementRows rows = m_shellPart->displacementRows(middle.shell, middle.xi, middle.eta);
    for (std::size_t b = 0; b < 3; ++b)
        appendPartTerms(rows.rows[b], frame(b, s), terms);
}

void MixedHybridModel::fillShellComponents(const NodalDisplacements& displacements,
                                           const std::function<void(std::int32_t, int, double)>& write) const
{
    if (m_shellPart == nullptr || !m_shellPart->usable())
        return;
    // The value of a ShellPart unknown in the solution (its grid components carried to the basic system).
    const auto valueOf = [&](std::size_t unknownNumber)
    {
        const ShellPartUnknown unknown = m_shellPart->describe(unknownNumber);
        if (unknown.kind == ShellPartUnknown::Kind::EntityDof)
            return displacements.valueAt(
                ElementDof::entity(m_shellEntityRow[unknown.index], static_cast<int>(unknown.component)));
        const base::Mat3& frame = m_gridFrames[unknown.index];
        const std::size_t b = unknown.component % 3;
        const int first = unknown.component < 3 ? 1 : 4;
        double value = 0.0;
        for (std::size_t c = 0; c < 3; ++c)
            value += frame(b, c) *
                     displacements.valueAt(ElementDof::grid(m_gridRows[unknown.index], first + static_cast<int>(c)));
        return value;
    };
    // The mean rotation of the shells at each corner grid without lines: θ = Σ_e θ_e(corner) / n, each shell's
    // rotation at its corner a linear map of its unknowns (rotationRowsOf), the shells built once each.
    std::map<std::int32_t, std::pair<std::array<double, 3>, int>> sums;
    std::set<std::int32_t> wanted;
    for (const MixedShell& shell : m_shells)
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
        {
            const std::int32_t grid = m_gridIndexOfRow.at(shell.gridRows[k]);
            if (m_lineGrids.count(grid) == 0)
                wanted.insert(grid);
        }
    for (const MixedShell& shell : m_shells)
    {
        bool needed = false;
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
            needed = needed || wanted.count(m_gridIndexOfRow.at(shell.gridRows[k])) != 0;
        if (!needed)
            continue;
        const ShellElementData element = m_shellPart->element(shell.partIndex);
        const ShellReferenceElement& reference = m_shellPart->referenceElement(shell.cornerCount);
        const std::span<const std::size_t> unknowns = m_shellPart->localUnknowns(shell.partIndex);
        const std::span<const double> signs = m_shellPart->localSigns(shell.partIndex);
        Vector local(unknowns.size());
        for (std::size_t i = 0; i < unknowns.size(); ++i)
            local[i] = signs[i] * valueOf(unknowns[i]);
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
        {
            const std::int32_t grid = m_gridIndexOfRow.at(shell.gridRows[k]);
            if (wanted.count(grid) == 0)
                continue;
            const ReferencePoint corner = referenceCorner(shell.cornerCount, k);
            const Matrix rows = rotationRowsOf(reference, element.definition(), corner[0], corner[1]);
            auto& [sum, count] = sums[grid];
            for (std::size_t b = 0; b < 3; ++b)
            {
                double value = 0.0;
                for (std::size_t i = 0; i < rows.columns() && i < local.size(); ++i)
                    value += rows(b, i) * local[i];
                sum[b] += value;
            }
            ++count;
        }
    }
    for (const auto& [grid, total] : sums)
    {
        const base::Mat3& frame = m_gridFrames[static_cast<std::size_t>(grid)];
        const std::int32_t row = m_gridRows[static_cast<std::size_t>(grid)];
        for (std::size_t s = 0; s < 3; ++s)
        {
            // θ_CD(s) = Σ_b R(b, s) θ_basic(b).
            double value = 0.0;
            for (std::size_t b = 0; b < 3; ++b)
                value += frame(b, s) * total.first[b] / total.second;
            write(row, 4 + static_cast<int>(s), value);
        }
    }
    // The mid-side grids: the point of their edge.
    for (const auto& [grid, middle] : m_midSideGrids)
    {
        const ShellDisplacementRows rows = m_shellPart->displacementRows(middle.shell, middle.xi, middle.eta);
        const base::Mat3& frame = m_gridFrames[static_cast<std::size_t>(grid)];
        std::array<double, 3> basic{};
        for (std::size_t b = 0; b < 3; ++b)
            for (const ShellPartTerm& term : rows.rows[b])
                basic[b] += term.coefficient * valueOf(term.unknown);
        for (std::size_t s = 0; s < 3; ++s)
            write(m_gridRows[static_cast<std::size_t>(grid)], 1 + static_cast<int>(s),
                  frame(0, s) * basic[0] + frame(1, s) * basic[1] + frame(2, s) * basic[2]);
    }
}

std::map<std::int32_t, MixedHybridModel::BasicSupport> MixedHybridModel::basicSupportsOf(
    std::span<const HeldGridComponent> held, base::DiagnosticSink& sink) const
{
    std::map<std::int32_t, BasicSupport> perGrid;
    std::map<std::int32_t, BasicSupport> cdOfGrid;
    for (const HeldGridComponent& component : held)
    {
        const std::int32_t grid = gridIndexOfRow(component.gridRow);
        if (grid < 0 || component.component < 1 || component.component > 6)
            continue;
        BasicSupport& cd = cdOfGrid[grid];
        cd.isHeld[static_cast<std::size_t>(component.component - 1)] = true;
        cd.values[static_cast<std::size_t>(component.component - 1)] = component.value;
    }
    // A basic axis lies in the span of the held CD axes when its part out of them is below this (a rotation matrix of
    // round-off).
    constexpr double ALIGNED = 1.0e-12;
    std::vector<std::int32_t> skewed;
    for (const auto& [grid, cd] : cdOfGrid)
    {
        const base::Mat3& frame = m_gridFrames[static_cast<std::size_t>(grid)];
        BasicSupport basic;
        bool isSkewed = false;
        for (std::size_t group = 0; group < 2; ++group)
        {
            // The basic value of the held CD components (the others counted 0): v_basic = R v_CD.
            for (std::size_t b = 0; b < 3; ++b)
            {
                double value = 0.0;
                double outside = 0.0; // the part of the basic axis b out of the held span
                for (std::size_t s = 0; s < 3; ++s)
                {
                    if (cd.isHeld[3 * group + s])
                        value += frame(b, s) * cd.values[3 * group + s];
                    else
                        outside += frame(b, s) * frame(b, s);
                }
                basic.isHeld[3 * group + b] = outside <= ALIGNED;
                basic.values[3 * group + b] = value;
            }
            for (std::size_t s = 0; s < 3; ++s)
            {
                if (!cd.isHeld[3 * group + s])
                    continue;
                // A held CD axis that is not a held basic axis.
                bool isCovered = false;
                for (std::size_t b = 0; b < 3; ++b)
                    isCovered = isCovered ||
                                (basic.isHeld[3 * group + b] && std::fabs(std::fabs(frame(b, s)) - 1.0) <= ALIGNED);
                isSkewed = isSkewed || !isCovered;
            }
        }
        perGrid[grid] = basic;
        if (isSkewed)
            skewed.push_back(grid);
    }
    if (!skewed.empty())
    {
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Warning;
        diagnostic.code = "support-not-carried-along-edge";
        diagnostic.card = "Node";
        diagnostic.id = m_gridIds[static_cast<std::size_t>(skewed.front())];
        diagnostic.message = base::concatenate(
            {"expected the supports of the grid points of shells and solids of ", MIXED_HYBRID_FORMULATION,
             " along the axes of the basic system or of a common displacement system, found ",
             std::to_string(skewed.size()),
             " grid points holding a skew direction (the first is named): the edges and faces hold only the basic "
             "directions their grid points hold"});
        sink.report(std::move(diagnostic));
    }
    return perGrid;
}

void MixedHybridModel::shellSupports(const std::map<std::int32_t, BasicSupport>& perGrid,
                                     std::vector<DerivedSupport>& supports, base::DiagnosticSink& sink) const
{
    if (m_shellPart == nullptr || !m_shellPart->usable())
        return;
    std::vector<ShellGridSupport> grids;
    for (const auto& [grid, support] : perGrid)
        grids.push_back({static_cast<std::size_t>(grid), support.isHeld, support.values});
    const ShellSupports derived = m_shellPart->supportsOf(grids);
    for (const ShellDerivedSupport& support : derived.held)
    {
        const ShellPartUnknown unknown = m_shellPart->describe(support.unknown);
        if (unknown.kind != ShellPartUnknown::Kind::EntityDof)
            continue; // the grid components themselves are the cards' supports
        DerivedSupport out;
        out.dof = ElementDof::entity(m_shellEntityRow[unknown.index], static_cast<int>(unknown.component));
        out.value = support.value;
        // The reaction carried to grid components in the basic system, then to the CD systems: f_CD = Rᵀ f_basic.
        for (const ShellGridTerm& term : support.carriedTo)
        {
            const base::Mat3& frame = m_gridFrames[term.grid];
            const std::size_t b = term.component % 3;
            const int first = term.component < 3 ? 1 : 4;
            for (std::size_t s = 0; s < 3; ++s)
                if (frame(b, s) != 0.0)
                    out.carriedTo.push_back({ElementDof::grid(m_gridRows[term.grid], first + static_cast<int>(s)),
                                             term.coefficient * frame(b, s)});
        }
        supports.push_back(std::move(out));
    }
    for (const std::string& note : derived.notes)
    {
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Warning;
        diagnostic.code = "support-note";
        diagnostic.message = note;
        sink.report(std::move(diagnostic));
    }
}

} // namespace oos::elements::mixedHybrid
