// The shells of the model of the mixed hybrid formulation (MixedHybridModel.hpp): their reading from the six shell
// cards, the ShellPart that numbers their edges and interiors, and what the formulation tells the solver about them
// (the rotations of their grid points, the supports read along their edges, the affine fields on their unknowns).

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementFormulation.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/mixedHybrid/core/JacobiEigen.hpp"
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
// A direction of the rotation of a grid point is not stiffened by its lines when their stiffness along it is at most
// this fraction of their largest there (the threshold of the singularity processor of the grid points).
constexpr double UNSTIFFENED_ROTATION_RATIO = 1.0e-8;
// The bytes of shell elements with their matrices the ShellPart keeps once built (ShellPartInput::matrixCacheBytes):
// about 300 shells of degree 4 with their mass, so that the check, the stiffness, the mass, the loads and the recovery
// of a model of that size build each element once (a few milliseconds each).
constexpr std::size_t SHELL_MATRIX_CACHE_BYTES = std::size_t{128} << 20;
// The bytes the ShellPart may hold in all, the elements beyond the first budget kept with the operators of the stress
// recovery alone (ShellPartInput::recoveryCacheBytes): about 5 000 shells of degree 4. The recovery of a model of
// that size reads each element instead of building it a second time, which took as long as the stiffnesses; what is
// kept, about 0.2 MB a shell, is a fifth of what the factorisation of such a model takes for each of its shells.
constexpr std::size_t SHELL_RECOVERY_CACHE_BYTES = std::size_t{1} << 30;

// The geometry report (spec section 10.5): the edges along which the normals of two shells differ by more than this
// angle (degrees) are named, at most GEOMETRY_REPORT_EDGES of each kind (see reportGeometry), the largest angles
// first. Below it, a kink is the round-off of the reading of a smooth surface.
constexpr double GEOMETRY_REPORT_ANGLE = 1.0;
constexpr std::size_t GEOMETRY_REPORT_EDGES = 10;
// The rounding of an edge (the angle between the normal of the surface read along it and the plane of the corners of
// a shell of the edge) above which the report names it: a fold that the declarations do not name is rounded to about
// half its angle; a smooth surface meshed finely enough turns by a few degrees over an element.
constexpr double ROUNDING_REPORT_ANGLE = 15.0;
// A rounding along an edge that the reading takes as smooth is a warning above this angle (a fold of twice it that no
// declaration names); below it the surface may as well be a smooth one meshed coarsely, which the formulation reads as
// meant: an information.
constexpr double ROUNDING_WARNING_ANGLE = 30.0;
// The turn of an edge of the outline in the surface (the angle between its curve and its chord at a grid point,
// EdgeGeometry::outlineTurn) above which the report names it. A corner of the outline that no raw edge declares is
// read as a curve: its two edges leave their chords by the turn of the corner between them, so one at least by half
// of it, and a corner that turns by 60 degrees or more is always named. A smooth outline leaves its chords by half
// the angle of an element: a circle of seven elements or more (the rim of a hole) is not named.
constexpr double OUTLINE_ROUNDING_REPORT_ANGLE = 30.0;
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
            // A membrane alone has no transverse shear: offset, it bends about the grid points as a Kirchhoff plate;
            // without offset it has no moments and no shear unknown of its own (ShellPart::unstiffenedUnknowns holds
            // the shear that no other shell has).
            if (shell.part.section.parts == SectionParts::MembraneOnly)
                shell.part.kirchhoff = shell.part.offset.value_or(0.0) != 0.0;
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
        m_shellOfId.emplace(m_shells[i].elementId, i);
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
    input.recoveryCacheBytes = SHELL_RECOVERY_CACHE_BYTES;
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
    if (const std::optional<input::FormulationGeometry>& geometry = m_declarations)
    {
        const auto gridIndexOfId = [&](std::int32_t id) { return this->gridIndexOfId(index, id); };
        // The edges of the shells and of the lines: a raw edge is one of them.
        std::set<EdgeKey> modelEdges;
        const auto gridOfRow = [&](std::int32_t row) { return static_cast<std::size_t>(m_gridIndexOfRow.at(row)); };
        for (const MixedShell& shell : m_shells)
            for (std::size_t k = 0; k < shell.cornerCount; ++k)
                modelEdges.insert(EdgeKey::of(gridOfRow(shell.gridRows[k]),
                                              gridOfRow(shell.gridRows[(k + 1) % shell.cornerCount])));
        for (const MixedLine& line : m_lines)
            modelEdges.insert(EdgeKey::of(gridOfRow(line.gridRows[0]), gridOfRow(line.gridRows[1])));
        for (const input::GridEdge& edge : geometry->rawEdges)
        {
            const std::int32_t a = gridIndexOfId(edge[0]);
            const std::int32_t b = gridIndexOfId(edge[1]);
            const bool isEdge =
                a >= 0 && b >= 0 &&
                modelEdges.count(EdgeKey::of(static_cast<std::size_t>(a), static_cast<std::size_t>(b))) != 0;
            if (a >= 0 && b >= 0 && !isEdge)
            {
                base::Diagnostic diagnostic;
                diagnostic.code = "geometry-declaration-unknown";
                diagnostic.message = base::concatenate(
                    {"expected the raw edges of the geometry declarations to be edges of the shells or lines of ",
                     MIXED_HYBRID_FORMULATION, " (two grid points that one of them joins), found grid points ",
                     std::to_string(edge[0]), " and ", std::to_string(edge[1]),
                     ", which no shell or line joins; correct the declaration"});
                m_problems.push_back(std::move(diagnostic));
                continue;
            }
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
    declareShellGeometry(index, input);

    m_shellPart = std::make_unique<ShellPart>(std::move(input));
    for (const ShellPartProblem& problem : m_shellPart->problems())
    {
        switch (problem.subject)
        {
        case ShellPartProblem::Subject::Shell:
        {
            const MixedShell& shell = m_shells[problem.index];
            base::Diagnostic diagnostic = shellProblem(shell.card, shell.elementId, "", problem.message);
            if (problem.isGeometry)
                diagnostic.code = problem_code::DEGENERATE_GEOMETRY;
            addElementProblem(shell.card, shell.batchRow, std::move(diagnostic));
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

void MixedHybridModel::findUnstiffenedLineRotations()
{
    if (m_shellPart == nullptr || !m_shellPart->usable())
        return;
    const ShellPart& part = *m_shellPart;
    // The grid points of shells whose lines all lie on shell edges (a line alone that ends at one brings a kernel of
    // its own: those grid points are left as they are).
    std::set<std::int32_t> onShellEdges;
    std::set<std::int32_t> alone;
    for (const MixedLine& line : m_lines)
        for (const std::int32_t row : line.gridRows)
            (line.shellLine >= 0 ? onShellEdges : alone).insert(m_gridIndexOfRow.at(row));
    // The stiffness of the lines on the three rotation components (basic) of each such grid point.
    std::map<std::int32_t, std::array<std::array<double, 3>, 3>> blocks;
    for (const std::int32_t grid : onShellEdges)
        if (alone.count(grid) == 0 && !part.shellsAt(static_cast<std::size_t>(grid)).empty())
            blocks.emplace(grid, std::array<std::array<double, 3>, 3>{});
    const auto rotationOf = [&](std::size_t unknown) -> std::optional<std::pair<std::int32_t, std::size_t>>
    {
        const ShellPartUnknown described = part.describe(unknown);
        if (described.kind != ShellPartUnknown::Kind::GridComponent || described.component < 3)
            return std::nullopt;
        return std::make_pair(static_cast<std::int32_t>(described.index), described.component - 3);
    };
    for (std::size_t line = 0; line < part.lineCount() && !blocks.empty(); ++line)
    {
        std::variant<LineData, std::string> built = part.lineData(line);
        const LineData* data = std::get_if<LineData>(&built);
        if (data == nullptr)
            continue; // a refused line: the problems of the part say so
        for (std::size_t i = 0; i < data->dofs.size(); ++i)
        {
            const auto rowRotation = rotationOf(data->dofs[i]);
            if (!rowRotation || blocks.count(rowRotation->first) == 0)
                continue;
            for (std::size_t j = 0; j < data->dofs.size(); ++j)
            {
                const auto columnRotation = rotationOf(data->dofs[j]);
                if (columnRotation && columnRotation->first == rowRotation->first)
                    blocks[rowRotation->first][rowRotation->second][columnRotation->second] +=
                        data->signs[i] * data->signs[j] * data->stiffness(i, j);
            }
        }
    }
    for (const auto& [grid, block] : blocks)
    {
        Matrix symmetric(3, 3);
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                symmetric(i, j) = 0.5 * (block[i][j] + block[j][i]);
        const SymmetricEigen eigen = jacobiEigen(symmetric);
        const double largest = std::max({eigen.values[0], eigen.values[1], eigen.values[2]});
        std::vector<base::Vec3> directions;
        for (std::size_t d = 0; d < 3; ++d)
            if (!(eigen.values[d] > UNSTIFFENED_ROTATION_RATIO * largest))
                directions.push_back({eigen.vectors(0, d), eigen.vectors(1, d), eigen.vectors(2, d)});
        if (!directions.empty())
            m_unstiffenedLineRotations.emplace(grid, std::move(directions));
    }
}

void MixedHybridModel::holdUnstiffenedShellDofs()
{
    if (m_shellPart == nullptr || !m_shellPart->usable())
        return;
    const ShellPart& part = *m_shellPart;
    // The entities of the shells whose displacement a solid follows (ties 1 and 2): the solids stiffen them.
    std::vector<bool> followed(part.entities().size(), false);
    const auto followEdge = [&](const EdgeKey& key)
    {
        const auto found = part.topology().edgeIndex.find(key);
        if (found != part.topology().edgeIndex.end())
            followed[part.edgeEntity(found->second)] = true;
    };
    const auto followShell = [&](std::size_t shell)
    {
        followed[part.interiorEntity(shell)] = true;
        const std::span<const std::size_t> corners = part.input().shells[shell].corners.view();
        for (std::size_t k = 0; k < corners.size(); ++k)
            followEdge(EdgeKey::of(corners[k], corners[(k + 1) % corners.size()]));
    };
    if (m_solidPart != nullptr)
    {
        for (const SolidPartEdge& edge : m_solidPart->edges())
        {
            if (edge.tie.kind == SolidEdgeTie::Kind::Shell)
                followEdge(edge.grids);
            else if (edge.tie.kind == SolidEdgeTie::Kind::Diagonal)
                followShell(edge.tie.shell);
        }
        for (const SolidPartFace& face : m_solidPart->faces())
            if (face.tie.coveredByShell)
                followShell(face.tie.shell);
    }
    const ShellUnstiffened unstiffened = part.unstiffenedUnknowns([&](std::size_t entity) { return followed[entity]; });
    for (const std::size_t unknown : unstiffened.unknowns)
    {
        const ShellPartUnknown described = part.describe(unknown);
        m_unstiffenedEntityDofs.push_back(
            ElementDof::entity(m_shellEntityRow[described.index], static_cast<int>(described.component)));
    }
    if (unstiffened.displacementDirections == 0)
        return;
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Information;
    diagnostic.code = "unstiffened-directions-held";
    diagnostic.message = base::concatenate(
        {std::to_string(unstiffened.displacementDirections),
         " displacement directions of the edges and interiors of membrane-only and bending-only shells that no element "
         "stiffens (the normal of a flat membrane, the plane of a flat bending-only plate) are held at zero"});
    if (unstiffened.skewedDirections > 0)
        diagnostic.message += base::concatenate(
            {"; ", std::to_string(unstiffened.skewedDirections),
             " of them do not lie along the basic axes: they are held through the components closest to them, and the "
             "displacement of those functions keeps a part along them that carries no stress (the first is on ",
             m_staticLayout.describe(m_shellEntityRow[unstiffened.firstSkewedEntity]), ")"});
    m_problems.push_back(std::move(diagnostic));
}

// ---- the geometry report (reportGeometry) ----------------------------------------------------------------------

namespace
{

// The measure of an edge below which every edge is taken by edgesAbove (the measures are angles, never negative).
constexpr double EVERY_EDGE = -1.0;
// How the declarations are given to the program, as the messages name them.
constexpr const char* RAW_EDGES_DECLARATION = "formulationGeometry.rawEdges";
constexpr const char* RAW_ELEMENTS_DECLARATION = "formulationGeometry.rawElements";
constexpr const char* SHARP_EDGES_DECLARATION = "formulationGeometry.sharpEdges";
constexpr const char* SMOOTH_EDGES_DECLARATION = "formulationGeometry.smoothEdges";
constexpr const char* LINE_ENDS_DECLARATION = "formulationGeometry.lineEnds";
// An edge that the fold angle alone made sharp is named when its angle is under this many times the fold angle: a
// curved surface meshed coarsely gives such angles as a shallow fold does. Above, it is a fold and is only counted.
constexpr double FOLD_WARNING_FACTOR = 2.0;

double inDegrees(double radians)
{
    return radians * 180.0 / std::numbers::pi;
}

// An edge that the geometry report names, with the angle (degrees) it is named for.
struct NamedEdge
{
    EdgeGeometry edge;
    double degrees = 0.0;
};

// The edges that `selects` keeps whose angle `measure` (radians) is above `threshold` (degrees), the largest angles
// first; among equal angles the lower grids first, so that the order does not depend on that of the elements.
template <typename Measure, typename Selects>
std::vector<NamedEdge> edgesAbove(const std::vector<EdgeGeometry>& edges, double threshold, const Measure& measure,
                                  const Selects& selects)
{
    std::vector<NamedEdge> out;
    for (const EdgeGeometry& edge : edges)
        if (selects(edge) && inDegrees(measure(edge)) > threshold)
            out.push_back({edge, inDegrees(measure(edge))});
    std::sort(out.begin(), out.end(),
              [](const NamedEdge& a, const NamedEdge& b)
              { return a.degrees != b.degrees ? a.degrees > b.degrees : a.edge.grids < b.edge.grids; });
    return out;
}

// The names of a state and of its origin in the table of the edges read (FormulationModel::surfaceEdges).
std::string_view stateName(const EdgeGeometry& edge)
{
    return edge.raw ? "Raw" : edge.sharp ? "Sharp" : "Smooth";
}

std::string_view originName(EdgeStateOrigin origin)
{
    switch (origin)
    {
    case EdgeStateOrigin::Declared:
        return "Declared";
    case EdgeStateOrigin::RawElement:
        return "RawElement";
    case EdgeStateOrigin::Angle:
    case EdgeStateOrigin::Unmeasured:
        return "Angle";
    case EdgeStateOrigin::Junction:
        return "Junction";
    case EdgeStateOrigin::Outline:
        break;
    }
    return "Outline";
}

// The reading goes on smoothly across an edge of one or two shells that is neither raw nor sharp. Across a raw or a
// sharp edge it does not, and at a junction of three shells or more the shells that do not face each other end there:
// a kink is expected along them.
bool isReadSmooth(const EdgeGeometry& edge)
{
    return !edge.raw && !edge.sharp && edge.elements <= 2;
}

// The fold angle alone made the edge sharp (no declaration names it).
bool isSharpByAngle(const EdgeGeometry& edge)
{
    return edge.sharp && edge.origin == EdgeStateOrigin::Angle;
}

// Why a kink or a rounding along an edge that is not read smooth is expected.
std::string expectedBecause(const EdgeGeometry& edge)
{
    if (edge.raw)
        return " (declared raw)";
    if (edge.elements > 2)
        return base::concatenate({" (a junction of ", std::to_string(edge.elements), " shells)"});
    return isSharpByAngle(edge) ? " (sharp by the fold angle of the model)" : " (declared sharp)";
}

// The message of a kink: the normals of the shells of an edge differ along it. Where the reading meant the surface
// to be smooth, the formulation still takes the kink as a fold (spec section 3.6): the results can be far off with
// kinks of a few degrees, so the message says what to do.
std::string kinkMessage(const NamedEdge& named, const std::string& edgeName)
{
    const std::string found =
        base::concatenate({"the normals of the ", std::to_string(named.edge.elements), " shells along ", edgeName,
                           " differ by up to ", std::to_string(named.degrees), " degrees"});
    if (!isReadSmooth(named.edge))
        return found + expectedBecause(named.edge);
    return base::concatenate(
        {found, " and the edge is read smooth: the formulation takes this kink as a fold of the shells. If the "
                "shells fold there, declare the edge sharp (",
         SHARP_EDGES_DECLARATION, "), or raw when the fold is straight (", RAW_EDGES_DECLARATION,
         "); if the surface is smooth there, the mesh is too coarse or too distorted for its curvature to be read: "
         "refine it, give the shells mid-side nodes, or declare the shells raw (",
         RAW_ELEMENTS_DECLARATION, ": facets that fold along their edges)"});
}

// The message of a rounding out of the plane of the shells: the surface read along an edge turns far from the plane
// of the corners of its shells.
std::string roundingMessage(const NamedEdge& named, const std::string& edgeName)
{
    const std::string found =
        base::concatenate({"the surface read along ", edgeName, " turns by up to ", std::to_string(named.degrees),
                           " degrees from the plane of the corners of its shells"});
    if (!isReadSmooth(named.edge))
        return found + expectedBecause(named.edge);
    return base::concatenate(
        {found, ": a fold shallower than the fold angle of the model is rounded there, or the mesh is coarse for "
                "the curvature. If the shells fold, declare sharp the edges along the fold (",
         SHARP_EDGES_DECLARATION, "), or raw when the fold is straight (", RAW_EDGES_DECLARATION, ", or the shells: ",
         RAW_ELEMENTS_DECLARATION, "); if the surface is smooth, it is read curved as meant"});
}

// The message of a rounding in the surface: an edge of the outline of the shells leaves its chord (the reading takes
// the outline round a corner that no raw edge declares, and the area of the shells changes).
std::string outlineMessage(const NamedEdge& named, const std::string& edgeName)
{
    return base::concatenate(
        {"the outline of the shells along ", edgeName,
         " is read as a curve that leaves the chord of the edge by up to ", std::to_string(named.degrees),
         " degrees in the surface: where the outline turns at a grid point, the reading takes it round the corner "
         "unless the declarations end it there, and the area of the shells changes. If the outline has a corner at "
         "one of the two grid points, declare raw the outline edges that meet there (",
         RAW_EDGES_DECLARATION, "), or the end of their line at the corner (", LINE_ENDS_DECLARATION,
         "); if the outline is a smooth curve, it is read as meant, and a finer mesh along it or mid-side nodes remove "
         "this warning"});
}

// The message of an edge that the fold angle alone made sharp while its angle is near the fold angle: a shallow fold
// or a curved surface meshed coarsely, which only the model can tell apart.
std::string foldByAngleMessage(const NamedEdge& named, const std::string& edgeName)
{
    return base::concatenate(
        {"the normals of the shells along ", edgeName, " differ by ", std::to_string(named.degrees),
         " degrees at a grid point, more than the fold angle of the model (ShellNormalAngle), and nothing declares "
         "the edge: it is read sharp, a fold of the shells. If the surface is smooth there, the mesh is coarse for "
         "its curvature and the surface is computed as facets: declare the edge smooth (",
         SMOOTH_EDGES_DECLARATION, ") or refine the mesh; if the shells fold there, declare the edge sharp (",
         SHARP_EDGES_DECLARATION, ") to remove this warning"});
}

// The name of an edge in the messages, by the IDs of its grid points.
std::string edgeNameOf(const std::vector<std::int32_t>& gridIds, const EdgeKey& key)
{
    return base::concatenate({"the edge between grid points ", std::to_string(gridIds[key.lower]), " and ",
                              std::to_string(gridIds[key.higher])});
}

// The diagnostics that name the first GEOMETRY_REPORT_EDGES edges of `named`, each at the lower grid point of its
// edge, with the text of `message(edge, name of the edge)`.
template <typename Message>
void nameEdges(const std::vector<NamedEdge>& named, base::Severity severity, const char* code,
               const std::vector<std::int32_t>& gridIds, const Message& message, std::vector<base::Diagnostic>& out)
{
    for (std::size_t i = 0; i < named.size() && i < GEOMETRY_REPORT_EDGES; ++i)
    {
        base::Diagnostic diagnostic;
        diagnostic.severity = severity;
        diagnostic.code = code;
        diagnostic.card = "Node";
        diagnostic.id = gridIds[named[i].edge.grids.lower];
        diagnostic.message = message(named[i], edgeNameOf(gridIds, named[i].edge.grids));
        out.push_back(std::move(diagnostic));
    }
}

// The edges at grid `grid` that the reading made curved (not those a mid-side grid point gives), by increasing grids:
// the edges to declare raw where the reading closed a corner. `edges` is the report, in the order of the topology.
std::vector<EdgeKey> edgesReadCurvedAt(const Topology& topology, const std::vector<EdgeGeometry>& edges,
                                       std::size_t grid)
{
    std::vector<EdgeKey> out;
    for (std::size_t e = 0; e < topology.edges.size(); ++e)
    {
        const EdgeKey key = topology.edges[e];
        if ((key.lower == grid || key.higher == grid) && !edges[e].given && norm3(topology.bulges[e]) > 0.0)
            out.push_back(key);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// The refusal of a shell whose two edge curves leave no angle at one of its corners (degenerateCorners): what was
// found and what to declare.
std::string closedCornerMessage(const std::vector<std::int32_t>& gridIds, const ShellCorners& shell,
                                std::size_t corner, const std::vector<EdgeKey>& readCurved)
{
    const std::string grid = std::to_string(gridIds[shell.at(corner)]);
    std::string message = base::concatenate(
        {"the edges of the shell from grid point ", grid, " to grid points ",
         std::to_string(gridIds[shell.at(corner + 1)]), " and ",
         std::to_string(gridIds[shell.at(corner + shell.count - 1)]),
         " are tangent to each other at grid point ", grid,
         " as the mesh is read, or cross there: the shell has no angle left at that corner, where nothing can be "
         "computed. "});
    if (readCurved.empty())
        return message + "Check the grid points and the mid-side grid points of the shell";
    message += base::concatenate({"The reading takes a mesh line round a corner that no raw edge declares: declare "
                                  "raw (",
                                  RAW_EDGES_DECLARATION, ") the edges read curved at grid point ", grid, ": "});
    for (std::size_t i = 0; i < readCurved.size(); ++i)
        message += (i == 0 ? "" : ", ") + edgeNameOf(gridIds, readCurved[i]);
    return message;
}

// The refusals of the shells of `part` that have a closed corner, as (shell of the part, message), in the order of
// the shells. `edges` is the report of the part.
std::vector<std::pair<std::size_t, std::string>> closedCornerRefusals(const ShellPart& part,
                                                                      const std::vector<EdgeGeometry>& edges,
                                                                      const std::vector<std::int32_t>& gridIds)
{
    std::vector<std::pair<std::size_t, std::string>> out;
    const MeshReadingInput& reading = part.meshReading();
    for (const DegenerateCorner& corner : degenerateCorners(reading, part.topology()))
    {
        const ShellCorners& shell = reading.shells[corner.element];
        out.emplace_back(corner.element,
                         closedCornerMessage(gridIds, shell, corner.corner,
                                             edgesReadCurvedAt(part.topology(), edges, shell.at(corner.corner))));
    }
    return out;
}

// The first sentences of the summary of the report: the counts and the largest of each measure, each list holding
// every edge, the largest measure first.
std::string summaryMessage(const std::vector<std::int32_t>& gridIds, std::size_t rawCount, std::size_t sharpCount,
                           std::size_t sharpByAngleCount, const std::vector<NamedEdge>& byKink,
                           const std::vector<NamedEdge>& byRounding, const std::vector<NamedEdge>& byOutlineTurn)
{
    const auto largest = [&](const std::vector<NamedEdge>& sorted)
    {
        return base::concatenate({std::to_string(sorted.front().degrees), " degrees, along ",
                                  edgeNameOf(gridIds, sorted.front().edge.grids)});
    };
    std::string message =
        base::concatenate({"the shells of ", MIXED_HYBRID_FORMULATION, " have ", std::to_string(byKink.size()),
                           " edges, ", std::to_string(rawCount), " declared raw, ", std::to_string(sharpCount),
                           " sharp (", std::to_string(sharpByAngleCount), " by the fold angle of the model alone); "});
    if (byKink.empty() || byKink.front().degrees == 0.0)
        message += "the normals of the shells agree along every edge";
    else
        message += "the largest angle between the normals of two shells along an edge is " + largest(byKink);
    if (!byRounding.empty())
        message += "; the surface read turns from the plane of the corners of a shell by up to " + largest(byRounding);
    if (!byOutlineTurn.empty() && byOutlineTurn.front().degrees > 0.0)
        message += "; the outline read leaves the chord of an edge by up to " + largest(byOutlineTurn);
    return message;
}

} // namespace

std::map<EdgeKey, double> MixedHybridModel::edgeRoundings() const
{
    // The largest angle, along each edge, between the normal of the surface read and the normal of the plane of the
    // corners of each shell of the edge (the z axis of its element axes).
    std::map<EdgeKey, double> roundings;
    for (const MixedShell& shell : m_shells)
    {
        const ShellElementData& element = m_shellPart->elementAtPoints(shell.partIndex);
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
    // What the reading of the mesh made of the surface of the shells (spec section 10.5):
    // - the edges declared raw, and the sharp ones; among these, the edges that the fold angle alone made sharp with
    //   an angle near it (a shallow fold or a curved surface meshed coarsely, which the model alone can tell apart: a
    //   warning);
    // - the edges where it left the normals of two shells apart. A kink of a raw edge or of a junction is expected
    //   (information). Elsewhere it is a fold the declarations do not name, or a mesh too coarse or too distorted for
    //   its curvature, and the formulation takes it as a fold: a warning;
    // - the edges where it turned the surface far from the plane of the corners of a shell (a fold that the
    //   declarations do not name is rounded: a warning);
    // - the edges of the outline that it bent in the surface (a corner of the outline that no raw edge declares is
    //   rounded, which the normals do not show on a flat plate: a warning);
    // - the corners of the shells that it closed (the shell is refused: an error of the element).
    const std::vector<EdgeGeometry> edges = m_shellPart->geometryReport();
    const std::map<EdgeKey, double> roundings = edgeRoundings();
    const auto kinkOf = [](const EdgeGeometry& edge) { return edge.kink; };
    const auto roundingOf = [&](const EdgeGeometry& edge)
    {
        const auto found = roundings.find(edge.grids);
        return found == roundings.end() ? 0.0 : found->second;
    };
    // the turn that the reading gave the outline: not that of a curve the model gives by a mid-side grid point
    const auto outlineTurnOf = [](const EdgeGeometry& edge) { return edge.given ? 0.0 : edge.outlineTurn; };
    const auto every = [](const EdgeGeometry&) { return true; };
    const auto expected = [](const EdgeGeometry& edge) { return !isReadSmooth(edge); };

    // The corners where the two edge curves of a shell leave no angle: nothing can be computed there (the normal of
    // the surface is 0 / 0, a NaN in the results), so the shell is refused. A shell already refused keeps its first
    // refusal, and this one after it.
    for (const auto& [index, message] : closedCornerRefusals(*m_shellPart, edges, m_gridIds))
    {
        const MixedShell& shell = m_shells[index];
        base::Diagnostic refusal;
        refusal.code = problem_code::DEGENERATE_GEOMETRY;
        refusal.card = std::string(shell.card);
        refusal.id = shell.elementId;
        refusal.message = message;
        const auto [found, added] =
            m_elementProblems.try_emplace(std::make_pair(std::string(shell.card), shell.batchRow), refusal);
        if (!added)
            found->second.message += "; " + message;
    }

    // The table of the edges read, by increasing grid point IDs (the grids of the model are in that order).
    std::vector<SurfaceEdgeReading> table;
    for (const EdgeGeometry& edge : edges)
        table.push_back({m_gridIds[edge.grids.lower], m_gridIds[edge.grids.higher], "Shells", stateName(edge),
                         originName(edge.origin), inDegrees(edge.angle), inDegrees(edge.kink),
                         inDegrees(roundingOf(edge))});
    std::sort(table.begin(), table.end(), [](const SurfaceEdgeReading& a, const SurfaceEdgeReading& b)
              { return a.grid1 != b.grid1 ? a.grid1 < b.grid1 : a.grid2 < b.grid2; });
    m_surfaceEdges.insert(m_surfaceEdges.end(), table.begin(), table.end());

    // The edges that the fold angle alone made sharp with an angle near it: a shallow fold or a coarse curved mesh.
    const double foldDegrees = inDegrees(m_foldAngle);
    const auto angleOf = [](const EdgeGeometry& edge) { return edge.angle; };
    const auto nearFoldAngle = [&](const EdgeGeometry& edge)
    { return isSharpByAngle(edge) && inDegrees(edge.angle) < FOLD_WARNING_FACTOR * foldDegrees; };
    std::vector<NamedEdge> foldsByAngle = edgesAbove(edges, EVERY_EDGE, angleOf, nearFoldAngle);
    // the nearest to the fold angle first: the most doubtful
    std::reverse(foldsByAngle.begin(), foldsByAngle.end());

    // The edges to name, the largest angles first: those the reading meant smooth (warnings), then the others.
    const std::vector<NamedEdge> kinks = edgesAbove(edges, GEOMETRY_REPORT_ANGLE, kinkOf, isReadSmooth);
    const std::vector<NamedEdge> expectedKinks = edgesAbove(edges, GEOMETRY_REPORT_ANGLE, kinkOf, expected);
    const std::vector<NamedEdge> rounded = edgesAbove(edges, ROUNDING_REPORT_ANGLE, roundingOf, isReadSmooth);
    const std::vector<NamedEdge> expectedRounded = edgesAbove(edges, ROUNDING_REPORT_ANGLE, roundingOf, expected);
    const std::vector<NamedEdge> outline = edgesAbove(edges, OUTLINE_ROUNDING_REPORT_ANGLE, outlineTurnOf, every);

    // The summary: the counts and the largest of each measure.
    const auto rawCount = std::count_if(edges.begin(), edges.end(), [](const EdgeGeometry& edge) { return edge.raw; });
    const auto sharpCount =
        std::count_if(edges.begin(), edges.end(), [](const EdgeGeometry& edge) { return edge.sharp; });
    const auto sharpByAngleCount = std::count_if(edges.begin(), edges.end(), isSharpByAngle);
    base::Diagnostic summary;
    summary.severity = base::Severity::Information;
    summary.code = "geometry-report";
    summary.message = summaryMessage(m_gridIds, static_cast<std::size_t>(rawCount),
                                     static_cast<std::size_t>(sharpCount), static_cast<std::size_t>(sharpByAngleCount),
                                     edgesAbove(edges, EVERY_EDGE, kinkOf, every),
                                     edgesAbove(edges, EVERY_EDGE, roundingOf, every),
                                     edgesAbove(edges, EVERY_EDGE, outlineTurnOf, every));
    if (!kinks.empty() || !rounded.empty() || !outline.empty())
        summary.message += base::concatenate(
            {"; the warnings name the largest of ", std::to_string(kinks.size()), " kinks and ",
             std::to_string(rounded.size()), " roundings along edges that the reading takes as smooth, and of ",
             std::to_string(outline.size()), " rounded edges of the outline"});
    if (!foldsByAngle.empty())
        summary.message += base::concatenate(
            {"; ", std::to_string(foldsByAngle.size()), " edges are sharp by the fold angle alone with an angle under ",
             std::to_string(FOLD_WARNING_FACTOR * foldDegrees), " degrees (a shallow fold, or a mesh coarse for its "
                                                                 "curvature)"});
    m_problems.push_back(std::move(summary));

    nameEdges(foldsByAngle, base::Severity::Warning, "geometry-fold-by-angle", m_gridIds, foldByAngleMessage,
              m_problems);
    nameEdges(kinks, base::Severity::Warning, "geometry-kink", m_gridIds, kinkMessage, m_problems);
    nameEdges(expectedKinks, base::Severity::Information, "geometry-kink", m_gridIds, kinkMessage, m_problems);
    std::vector<NamedEdge> folds;       // read smooth and rounded as an undeclared fold would be
    std::vector<NamedEdge> coarseCurves; // read smooth, rounded as a coarse smooth surface is
    for (const NamedEdge& edge : rounded)
        (edge.degrees > ROUNDING_WARNING_ANGLE ? folds : coarseCurves).push_back(edge);
    nameEdges(folds, base::Severity::Warning, "geometry-rounding", m_gridIds, roundingMessage, m_problems);
    nameEdges(coarseCurves, base::Severity::Information, "geometry-rounding", m_gridIds, roundingMessage, m_problems);
    nameEdges(expectedRounded, base::Severity::Information, "geometry-rounding", m_gridIds, roundingMessage,
              m_problems);
    nameEdges(outline, base::Severity::Warning, "geometry-rounding", m_gridIds, outlineMessage, m_problems);
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

namespace
{

// The `count` components (rows of v) that hold the `count` directions (columns of v) best: the subset of {0, 1, 2}
// with the largest |det V_C| (the first on a tie), in increasing order in the first `count` entries.
std::array<std::size_t, 3> componentsHolding(const std::array<std::array<double, 3>, 3>& v, std::size_t count)
{
    if (count >= 3)
        return {0, 1, 2};
    std::array<std::size_t, 3> best{0, 1, 2};
    double bestMinor = -1.0;
    if (count == 1)
    {
        for (std::size_t c = 0; c < 3; ++c)
            if (std::fabs(v[c][0]) > bestMinor)
            {
                bestMinor = std::fabs(v[c][0]);
                best = {c, 0, 0};
            }
        return best;
    }
    for (std::size_t a = 0; a < 3; ++a)
        for (std::size_t b = a + 1; b < 3; ++b)
        {
            const double minor = std::fabs(v[a][0] * v[b][1] - v[a][1] * v[b][0]);
            if (minor > bestMinor)
            {
                bestMinor = minor;
                best = {a, b, 0};
            }
        }
    return best;
}

// The inverse of A(i, j) = V(C_j, i), i and j below `count` (Gauss-Jordan with partial pivoting).
std::array<std::array<double, 3>, 3> inverseOfLeading(const std::array<std::array<double, 3>, 3>& v,
                                                      const std::array<std::size_t, 3>& chosen, std::size_t count)
{
    std::array<std::array<double, 6>, 3> work{};
    for (std::size_t i = 0; i < count; ++i)
    {
        for (std::size_t j = 0; j < count; ++j)
            work[i][j] = v[chosen[j]][i];
        work[i][3 + i] = 1.0;
    }
    for (std::size_t k = 0; k < count; ++k)
    {
        std::size_t pivot = k;
        for (std::size_t i = k + 1; i < count; ++i)
            if (std::fabs(work[i][k]) > std::fabs(work[pivot][k]))
                pivot = i;
        std::swap(work[k], work[pivot]);
        const double diagonal = work[k][k];
        for (double& value : work[k])
            value /= diagonal;
        for (std::size_t i = 0; i < count; ++i)
        {
            if (i == k)
                continue;
            const double factor = work[i][k];
            for (std::size_t j = 0; j < 6; ++j)
                work[i][j] -= factor * work[k][j];
        }
    }
    std::array<std::array<double, 3>, 3> inverse{};
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j < count; ++j)
            inverse[i][j] = work[i][3 + j];
    return inverse;
}

} // namespace

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
    // The rotation of a grid point of shells in the directions its lines on shell edges do not stiffen (a rod, a bar
    // that bends in one plane; tie 5 of the reference holds it at zero there): it follows the shells, Λ·v = θ·v with θ
    // their mean rotation, so that a moment, a spring or a rigid element at the grid point acts on the shells. With
    // the s directions V (CD components), the s components C that hold them best (the largest |det V_C|) are made
    // dependent: Λ_C = V_C⁻ᵀ·(Vᵀ·θ − V_Fᵀ·Λ_F), F the other components.
    for (const auto& [grid, directions] : m_unstiffenedLineRotations)
    {
        const base::Mat3& frame = m_gridFrames[static_cast<std::size_t>(grid)];
        const std::int32_t row = m_gridRows[static_cast<std::size_t>(grid)];
        const std::size_t count = directions.size();
        // V(c, i): component c (CD) of direction i, v_CD = Rᵀ·v_basic.
        std::array<std::array<double, 3>, 3> v{};
        for (std::size_t i = 0; i < count; ++i)
            for (std::size_t c = 0; c < 3; ++c)
                v[c][i] =
                    frame(0, c) * directions[i][0] + frame(1, c) * directions[i][1] + frame(2, c) * directions[i][2];
        const std::array<std::size_t, 3> chosen = componentsHolding(v, count);
        // A(i, j) = V(C_j, i), inverted by Gauss-Jordan (at most 3 × 3, |det| the largest of the choices).
        std::array<std::array<double, 3>, 3> inverse = inverseOfLeading(v, chosen, count);
        const std::array<std::vector<ShellPartTerm>, 3> rows =
            m_shellPart->meanRotationRows(static_cast<std::size_t>(grid), ROTATION_ROW_THRESHOLD);
        for (std::size_t j = 0; j < count; ++j)
        {
            const int component = 4 + static_cast<int>(chosen[j]);
            const bool isHeld = std::any_of(held.begin(), held.end(), [&](const HeldGridComponent& h)
                                            { return h.gridRow == row && h.component == component; });
            if (isHeld)
                continue;
            dependents.push_back(ElementDof::grid(row, component));
            for (std::size_t c = 0; c < 3; ++c)
            {
                // The coefficient of θ_c (and, with the other sign, of a free Λ_c): Σ_i A⁻¹(j, i)·V(c, i).
                double coefficient = 0.0;
                for (std::size_t i = 0; i < count; ++i)
                    coefficient += inverse[j][i] * v[c][i];
                if (coefficient == 0.0)
                    continue;
                // θ_CD(c) = Σ_b R(b, c) θ_basic(b).
                for (std::size_t b = 0; b < 3; ++b)
                    appendPartTerms(rows[b], coefficient * frame(b, c), terms);
                const bool isChosen = std::find(chosen.begin(), chosen.begin() + static_cast<std::ptrdiff_t>(count),
                                                c) != chosen.begin() + static_cast<std::ptrdiff_t>(count);
                if (!isChosen)
                    terms.push_back({ElementDof::grid(row, 4 + static_cast<int>(c)), -coefficient});
            }
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

std::optional<base::Vec3> MixedHybridModel::commonShellNormal(std::int32_t gridRow) const
{
    const std::int32_t grid = gridIndexOfRow(gridRow);
    if (grid < 0 || m_shellPart == nullptr || !m_shellPart->usable())
        return std::nullopt;
    const std::optional<SpaceVector> normal = m_shellPart->commonNormalAt(static_cast<std::size_t>(grid));
    if (!normal)
        return std::nullopt;
    // n_CD(s) = Σ_b R(b, s) n_basic(b).
    const base::Mat3& frame = m_gridFrames[static_cast<std::size_t>(grid)];
    base::Vec3 inCd{};
    for (std::size_t s = 0; s < 3; ++s)
        inCd[s] = frame(0, s) * (*normal)[0] + frame(1, s) * (*normal)[1] + frame(2, s) * (*normal)[2];
    return inCd;
}

void MixedHybridModel::shellEdges(std::vector<std::array<std::int32_t, 2>>& edges) const
{
    if (m_shellPart == nullptr || !m_shellPart->usable())
        return;
    for (const EdgeKey& edge : m_shellPart->topology().edges)
        edges.push_back({m_gridRows[edge.lower], m_gridRows[edge.higher]});
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
    // rotation at its corner a linear map of its unknowns (rotationRowsOf), the shells built once each; over the shells
    // that carry moments where one does (ShellPart::rotationShellsAt).
    std::map<std::int32_t, std::pair<std::array<double, 3>, int>> sums;
    std::set<std::int32_t> wanted;
    std::set<std::int32_t> withMoments;
    for (const MixedShell& shell : m_shells)
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
        {
            const std::int32_t grid = m_gridIndexOfRow.at(shell.gridRows[k]);
            if (m_lineGrids.count(grid) == 0)
                wanted.insert(grid);
            if (m_shellPart->carriesMoments(shell.partIndex))
                withMoments.insert(grid);
        }
    for (const MixedShell& shell : m_shells)
    {
        bool needed = false;
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
            needed = needed || wanted.count(m_gridIndexOfRow.at(shell.gridRows[k])) != 0;
        if (!needed)
            continue;
        const ShellElementData& element = m_shellPart->elementAtPoints(shell.partIndex);
        const ShellReferenceElement& reference = m_shellPart->referenceElement(shell.cornerCount);
        const std::span<const std::size_t> unknowns = m_shellPart->localUnknowns(shell.partIndex);
        const std::span<const double> signs = m_shellPart->localSigns(shell.partIndex);
        Vector local(unknowns.size());
        for (std::size_t i = 0; i < unknowns.size(); ++i)
            local[i] = signs[i] * valueOf(unknowns[i]);
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
        {
            const std::int32_t grid = m_gridIndexOfRow.at(shell.gridRows[k]);
            if (wanted.count(grid) == 0 ||
                (withMoments.count(grid) != 0 && !m_shellPart->carriesMoments(shell.partIndex)))
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
    std::span<const HeldGridComponent> held) const
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
        cd.followsRelation = cd.followsRelation || component.followsRelation;
    }
    // A basic axis lies in the span of the held CD axes when its part out of them is below this (a rotation matrix of
    // round-off).
    constexpr double ALIGNED = 1.0e-12;
    for (const auto& [grid, cd] : cdOfGrid)
    {
        const base::Mat3& frame = m_gridFrames[static_cast<std::size_t>(grid)];
        BasicSupport basic;
        basic.isHeldInCd = cd.isHeld;
        basic.followsRelation = cd.followsRelation;
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
                // A held CD axis that is no held basic axis: a direction of its own.
                bool isCovered = false;
                for (std::size_t b = 0; b < 3; ++b)
                    isCovered = isCovered ||
                                (basic.isHeld[3 * group + b] && std::fabs(std::fabs(frame(b, s)) - 1.0) <= ALIGNED);
                if (!isCovered)
                    basic.skew.push_back({SpaceVector{frame(0, s), frame(1, s), frame(2, s)}, 3 * group + s});
            }
        }
        perGrid[grid] = std::move(basic);
    }
    return perGrid;
}

void MixedHybridModel::shellSupports(const std::map<std::int32_t, BasicSupport>& perGrid,
                                     std::vector<DerivedSupport>* supports,
                                     std::vector<std::vector<FormulationRelationTerm>>* rows,
                                     base::DiagnosticSink& sink) const
{
    if (m_shellPart == nullptr || !m_shellPart->usable())
        return;
    std::vector<ShellGridSupport> grids;
    for (const auto& [grid, support] : perGrid)
        grids.push_back({static_cast<std::size_t>(grid), support.isHeld, support.values, support.skew,
                         support.followsRelation});
    const ShellSupports derived = m_shellPart->supportsOf(grids);
    for (const ShellDerivedSupport& support : derived.held)
    {
        if (supports == nullptr)
            break;
        const ShellPartUnknown unknown = m_shellPart->describe(support.unknown);
        if (unknown.kind != ShellPartUnknown::Kind::EntityDof)
            continue; // the grid components themselves are the cards' supports
        DerivedSupport out;
        out.dof = ElementDof::entity(m_shellEntityRow[unknown.index], static_cast<int>(unknown.component));
        out.value = support.value;
        // The reaction carried to grid components in the basic system, then to the CD systems: f_CD = Rᵀ f_basic, on
        // the components the grid point holds (the others have no support to report it).
        for (const ShellGridTerm& term : support.carriedTo)
        {
            const base::Mat3& frame = m_gridFrames[term.grid];
            const std::array<bool, 6>& isHeldInCd = perGrid.at(static_cast<std::int32_t>(term.grid)).isHeldInCd;
            const std::size_t b = term.component % 3;
            const std::size_t first = term.component < 3 ? 0 : 3;
            for (std::size_t s = 0; s < 3; ++s)
                if (frame(b, s) != 0.0 && isHeldInCd[first + s])
                    out.carriedTo.push_back(
                        {ElementDof::grid(m_gridRows[term.grid], static_cast<int>(first + s) + 1),
                         term.coefficient * frame(b, s)});
        }
        supports->push_back(std::move(out));
    }
    for (const ShellSupportRow& row : derived.rows)
    {
        if (rows == nullptr)
            break;
        std::vector<FormulationRelationTerm> terms;
        appendPartTerms(row.unknowns, 1.0, terms);
        for (const ShellAxisTerm& term : row.grids)
            terms.push_back({ElementDof::grid(m_gridRows[term.grid], static_cast<int>(term.component) + 1),
                             term.coefficient});
        rows->push_back(std::move(terms));
    }
    if (supports == nullptr)
        return;
    for (const std::string& note : derived.notes)
    {
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Warning;
        diagnostic.code = "support-note";
        diagnostic.message = note;
        sink.report(std::move(diagnostic));
    }
    if (derived.uncarriedEdges > 0)
    {
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Warning;
        diagnostic.code = "support-not-carried-along-edge";
        diagnostic.message = base::concatenate(
            {std::to_string(derived.uncarriedEdges), " edges of shells of ", MIXED_HYBRID_FORMULATION,
             " have their two grid points held along directions that differ (a displacement system that turns along "
             "the edge: a cylindrical system along a circle): the grid points hold their directions, the edge between "
             "them is free along them; hold both directions of the plane they turn in, or refine the mesh there"});
        sink.report(std::move(diagnostic));
    }
}

} // namespace oos::elements::mixedHybrid
