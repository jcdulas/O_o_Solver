// The solids of the model of the mixed hybrid formulation (MixedHybridModel.hpp): their reading from the four solid
// cards, the SolidPart that numbers their edges, faces and interior fields, and what the formulation tells the solver
// about them (their grid points have no rotation, their mid-side grid points follow their edges, the supports read
// along their skin, the affine fields on their unknowns).

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementFormulation.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/core/SolidShapeFunctions.hpp"
#include "oos/elements/mixedHybrid/model/MixedHybridModel.hpp"
#include "oos/elements/solids/SolidConnectivity.hpp"
#include "oos/elements/solids/SolidSection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <string_view>

namespace oos::elements::mixedHybrid
{

namespace
{

// The solid cards of the formulation: their shape in the formulation, their corner and mid-side node counts, and the
// shape of the card's other families without mid-side nodes (whose section rules apply to every element of the card).
struct SolidCard
{
    std::string_view card;
    VolumeShape shape = VolumeShape::Tetrahedron;
    std::size_t corners = 4;
    std::size_t midSides = 6;
    SolidShape linearShape = SolidShape::Tetra4;
};
constexpr std::array<SolidCard, 4> SOLID_CARDS{{{"SolidTetra", VolumeShape::Tetrahedron, 4, 6, SolidShape::Tetra4},
                                                {"SolidPyramid", VolumeShape::Pyramid, 5, 8, SolidShape::Pyram5},
                                                {"SolidPenta", VolumeShape::Pentahedron, 6, 9, SolidShape::Penta6},
                                                {"SolidHexa", VolumeShape::Hexahedron, 8, 12, SolidShape::Hexa8}}};

// The bytes of solid kernels the SolidPart keeps once built (SolidPartInput::kernelCacheBytes): about 80 hexahedra of
// degree 4, or many more tetrahedra, so that the check, the stiffness, the mass and the recovery of a model of that
// size build each kernel once.
constexpr std::size_t SOLID_KERNEL_CACHE_BYTES = std::size_t{256} << 20;

// The bytes of solid kernels the SolidPart keeps in all, those beyond SOLID_KERNEL_CACHE_BYTES without their stiffness
// (SolidPartInput::reducedCacheBytes): about 390 hexahedra of degree 4. The loads, the mass and the recovery of a
// model of that size read its kernels instead of building each again (40 ms for a hexahedron); a larger model builds
// those beyond for its stiffness, then for each of those uses.
constexpr std::size_t SOLID_REDUCED_CACHE_BYTES = std::size_t{1} << 30;

// Two expansion coefficients are the same when they differ by less than this share of the larger.
constexpr double SAME_EXPANSION = 1.0e-12;

base::Diagnostic solidProblem(std::string_view card, std::int32_t elementId, std::string code, std::string field,
                              std::string message, base::Severity severity = base::Severity::Error)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = std::move(code);
    diagnostic.card = std::string(card);
    diagnostic.id = elementId;
    diagnostic.field = std::move(field);
    diagnostic.message = std::move(message);
    return diagnostic;
}

// Whether the expansion of a section is one coefficient along the three directions, without shear (the kernel's).
bool isIsotropicExpansion(const Voigt6& expansion)
{
    const double scale = std::max({std::fabs(expansion[0]), std::fabs(expansion[1]), std::fabs(expansion[2])});
    const double tolerance = SAME_EXPANSION * scale;
    return std::fabs(expansion[1] - expansion[0]) <= tolerance && std::fabs(expansion[2] - expansion[0]) <= tolerance &&
           expansion[3] == 0.0 && expansion[4] == 0.0 && expansion[5] == 0.0;
}

// The iterations of the search of the parent point of a straight element nearest to a position, and the step of the
// parent coordinates below which it stops (Newton converges quadratically on an element that the formulation accepts).
constexpr int PARENT_POINT_ITERATIONS = 30;
constexpr double PARENT_POINT_STEP = 1.0e-13;

// The shape of the card's other families without mid-side nodes, from the corner count.
SolidShape linearShapeOf(const MixedSolid& solid)
{
    switch (solid.cornerRows.size())
    {
    case 4:
        return SolidShape::Tetra4;
    case 5:
        return SolidShape::Pyram5;
    case 6:
        return SolidShape::Penta6;
    default:
        return SolidShape::Hexa8;
    }
}

// The functions of the corners of the straight element at the parent point nearest to the position x: Newton's method
// on X(p) = Σ N_i(p) X_i from the centre, J Δp = x − X(p), J = Σ X_i ∂N_i/∂p.
SolidShapeValues cornerFunctionsAt(SolidShape shape, std::span<const base::Vec3> corners, const base::Vec3& x)
{
    ParentPoint point = solidCentre(shape);
    for (int iteration = 0; iteration < PARENT_POINT_ITERATIONS; ++iteration)
    {
        const SolidShapeValues values = solidShapeAt(shape, point);
        base::Vec3 residual = x;
        base::Mat3 jacobian{};
        for (std::size_t i = 0; i < corners.size(); ++i)
            for (std::size_t c = 0; c < 3; ++c)
            {
                residual[c] -= values.n[i] * corners[i][c];
                jacobian(c, 0) += values.dXi[i] * corners[i][c];
                jacobian(c, 1) += values.dEta[i] * corners[i][c];
                jacobian(c, 2) += values.dZeta[i] * corners[i][c];
            }
        if (base::determinant(jacobian) == 0.0)
            break;
        const base::Vec3 step = base::inverse(jacobian) * residual;
        point.xi += step[0];
        point.eta += step[1];
        point.zeta += step[2];
        if (std::fabs(step[0]) + std::fabs(step[1]) + std::fabs(step[2]) < PARENT_POINT_STEP)
            break;
    }
    return solidShapeAt(shape, point);
}

} // namespace

void MixedHybridModel::readSolids(const SnapshotIndex& index)
{
    const input::ModelSnapshot& snapshot = index.snapshot();
    for (const SolidCard& card : SOLID_CARDS)
    {
        const input::ElementBatch* batch = snapshot.findElementBatch(card.card);
        if (batch == nullptr || batch->rowCount() == 0)
            continue;
        const ElementBatchView view(*batch, index);
        const SolidConnectivity connectivity(view);
        const SolidSectionColumns sections(view);
        std::vector<std::int32_t>& solidOfRow =
            m_solidOfRow
                .emplace(std::string(card.card),
                         std::vector<std::int32_t>(static_cast<std::size_t>(batch->rowCount()), -1))
                .first->second;
        for (std::int64_t row = 0; row < view.rowCount(); ++row)
        {
            const std::int32_t id = view.elementId(row);
            // The corners required, each mid-side node optional (an ID 0: absent; the formulation takes a partial set).
            const std::span<const std::int32_t> ids = connectivity.ids[row];
            const std::span<const std::int32_t> rows = connectivity.rows[row];
            MixedSolid solid;
            bool isRead = true;
            bool hasMidSide = false;
            for (std::size_t i = 0; i < card.corners + card.midSides && isRead; ++i)
            {
                const bool isCorner = i < card.corners;
                const std::int32_t nodeId = i < ids.size() ? ids[i] : 0;
                if (!isCorner && nodeId == 0)
                {
                    solid.midRows.push_back(-1);
                    continue;
                }
                const std::optional<std::int32_t> gridRow = i < rows.size() && rows[i] >= 0
                                                                ? std::optional<std::int32_t>(rows[i])
                                                                : view.index().points().findGrid(nodeId);
                if (!gridRow)
                {
                    addElementProblem(
                        card.card, row,
                        elementProblem(
                            view, row, problem_code::MISSING_POINT, std::string("node").append(std::to_string(i + 1)),
                            base::concatenate({"expected the ID of a grid point, found ", std::to_string(nodeId),
                                               ", which the model does not define; define it "
                                               "or correct the card"})));
                    isRead = false;
                    break;
                }
                const std::int32_t checked = checkedGridRow(view, *gridRow, nodeId);
                if (isCorner)
                    solid.cornerRows.push_back(checked);
                else
                {
                    solid.midRows.push_back(checked);
                    hasMidSide = true;
                }
            }
            if (!isRead)
                continue;
            if (!hasMidSide)
                solid.midRows.clear();
            Resolution<elements::SolidSection> section = resolveSolidSection(view, sections, row, card.linearShape);
            if (!section)
            {
                addElementProblem(card.card, row, std::move(section.problem));
                continue;
            }
            solid.card = card.card;
            solid.batchRow = row;
            solid.elementId = id;
            solid.classicSection = *section.value;
            solid.isotropicExpansion = isIsotropicExpansion(section.value->expansion);
            // The law in the material system CORDM (the SolidPart turns it to the basic axes), its density, and its
            // expansion (the first coefficient: the kernel's is the same along every direction; another one is refused
            // under a temperature load).
            solid.part.elementId = id;
            for (std::size_t i = 0; i < 6; ++i)
                for (std::size_t j = 0; j < 6; ++j)
                    solid.part.material.stiffness[i][j] = section.value->elasticity(i, j);
            solid.part.material.density = section.value->density;
            solid.part.material.thermalExpansion = section.value->expansion[0];
            solid.part.materialAxes = section.value->axes;
            solidOfRow[static_cast<std::size_t>(row)] = static_cast<std::int32_t>(m_solids.size());
            m_solids.push_back(std::move(solid));
        }
    }
    // The SolidPart takes the solids by increasing element ID: that order is also their index there.
    std::sort(m_solids.begin(), m_solids.end(),
              [](const MixedSolid& a, const MixedSolid& b) { return a.elementId < b.elementId; });
    for (auto& [card, rows] : m_solidOfRow)
        std::fill(rows.begin(), rows.end(), -1);
    for (std::size_t i = 0; i < m_solids.size(); ++i)
    {
        m_solids[i].partIndex = i;
        m_solidOfRow.at(std::string(m_solids[i].card))[static_cast<std::size_t>(m_solids[i].batchRow)] =
            static_cast<std::int32_t>(i);
    }
}

namespace
{

// The edges of the skin that the report names at most, and the factor of the fold angle under which an edge that the
// angle alone made sharp is named (as for the shells: MixedHybridModelShells.cpp).
constexpr std::size_t SKIN_REPORT_EDGES = 10;
constexpr double SKIN_FOLD_WARNING_FACTOR = 2.0;

double degreesOf(double radians)
{
    return radians * 180.0 / std::numbers::pi;
}

} // namespace

void MixedHybridModel::reportSkinGeometry()
{
    // The skin is read as the surface of the shells is (spec section 4.4): its edges go to the table of the edges
    // read, and the edges that the fold angle alone made sharp with an angle near it are named, the nearest first. A
    // curved skin meshed coarsely gives such angles as a shallow fold does, and only the model can tell them apart.
    struct NearFold
    {
        double degrees = 0.0;
        EdgeKey edge;
    };
    std::vector<NearFold> nearFolds;
    std::size_t sharpCount = 0;
    std::size_t sharpByAngleCount = 0;
    const double foldDegrees = degreesOf(m_foldAngle);
    const std::map<EdgeKey, EdgeReading>& states = m_solidPart->skinEdgeStates();
    for (const auto& [edge, reading] : states)
    {
        const bool isSharp = reading.state == EdgeState::Sharp;
        const bool isByAngle = reading.origin == EdgeStateOrigin::Angle;
        m_surfaceEdges.push_back({m_gridIds[edge.lower], m_gridIds[edge.higher], "SolidSkin",
                                  isSharp ? "Sharp" : "Smooth", isByAngle ? "Angle" : "Declared",
                                  degreesOf(reading.angle), 0.0, 0.0});
        sharpCount += isSharp ? 1 : 0;
        sharpByAngleCount += isSharp && isByAngle ? 1 : 0;
        if (isSharp && isByAngle && degreesOf(reading.angle) < SKIN_FOLD_WARNING_FACTOR * foldDegrees)
            nearFolds.push_back({degreesOf(reading.angle), edge});
    }
    base::Diagnostic summary;
    summary.severity = base::Severity::Information;
    summary.code = "geometry-report";
    summary.message = base::concatenate(
        {"the skin of the solids of ", MIXED_HYBRID_FORMULATION, " has ", std::to_string(states.size()),
         " edges between two faces, ", std::to_string(sharpCount), " sharp (", std::to_string(sharpByAngleCount),
         " by the fold angle of the model alone)"});
    if (!nearFolds.empty())
        summary.message += base::concatenate(
            {"; ", std::to_string(nearFolds.size()), " edges are sharp by the fold angle alone with an angle under ",
             std::to_string(SKIN_FOLD_WARNING_FACTOR * foldDegrees), " degrees (a shallow fold, or a mesh coarse for "
                                                                      "its curvature)"});
    m_problems.push_back(std::move(summary));
    std::sort(nearFolds.begin(), nearFolds.end(), [](const NearFold& a, const NearFold& b)
              { return a.degrees != b.degrees ? a.degrees < b.degrees : a.edge < b.edge; });
    for (std::size_t i = 0; i < nearFolds.size() && i < SKIN_REPORT_EDGES; ++i)
    {
        const EdgeKey& edge = nearFolds[i].edge;
        base::Diagnostic diagnostic;
        diagnostic.severity = base::Severity::Warning;
        diagnostic.code = "geometry-fold-by-angle";
        diagnostic.card = "Node";
        diagnostic.id = m_gridIds[edge.lower];
        diagnostic.message = base::concatenate(
            {"the normals of the faces of the skin of the solids along the edge between grid points ",
             std::to_string(m_gridIds[edge.lower]), " and ", std::to_string(m_gridIds[edge.higher]), " differ by ",
             std::to_string(nearFolds[i].degrees),
             " degrees at a grid point, more than the fold angle of the model (ShellNormalAngle), and nothing "
             "declares the edge: it is read sharp, a fold of the skin. If the skin is smooth there, the mesh is "
             "coarse for its curvature and the skin is computed as facets: declare the edge smooth "
             "(formulationGeometry.smoothEdges) or refine the mesh; if the skin folds there, declare the edge sharp "
             "(formulationGeometry.sharpEdges) to remove this warning"});
        m_problems.push_back(std::move(diagnostic));
    }
}

void MixedHybridModel::buildSolidPart(const SnapshotIndex& index)
{
    if (m_solids.empty())
        return;
    const input::ModelSnapshot& snapshot = index.snapshot();
    const input::NodeTable& nodes = snapshot.nodes();
    const base::FixedListView<double> positions = nodes.positionsInBasic();

    SolidPartInput input;
    input.kernelCacheBytes = SOLID_KERNEL_CACHE_BYTES;
    input.reducedCacheBytes = SOLID_REDUCED_CACHE_BYTES;
    // The traces, the faces and the stresses of the solids at the degree of the model, as those of the shells and of
    // the lines they share edges with.
    input.options.traceDegree = m_degree;
    input.options.faceDegree = m_degree;
    input.options.stressDegree = m_degree;
    // What the solids tie to (spec section 3.4): the shells, and the edges of lines alone.
    if (m_shellPart != nullptr && m_shellPart->usable())
        input.shells = m_shellPart.get();
    for (std::size_t e = 0; e < m_edges.size(); ++e)
        if (m_edges[e].alone)
            input.aloneEdges.push_back({EdgeKey::of(static_cast<std::size_t>(m_edges[e].grids[0]),
                                                    static_cast<std::size_t>(m_edges[e].grids[1])),
                                        e, *m_edges[e].alone});
    for (std::size_t g = 0; g < m_gridRows.size(); ++g)
    {
        const std::span<const double> xyz = positions[m_gridRows[g]];
        input.gridPositions.push_back({xyz[0], xyz[1], xyz[2]});
        input.gridIds.push_back(m_gridIds[g]);
    }
    for (MixedSolid& solid : m_solids)
    {
        solid.part.corners.clear();
        for (const std::int32_t row : solid.cornerRows)
            solid.part.corners.push_back(static_cast<std::size_t>(m_gridIndexOfRow.at(row)));
        solid.part.midSideGrids.clear();
        for (const std::int32_t row : solid.midRows)
            solid.part.midSideGrids.push_back(row >= 0 ? static_cast<std::size_t>(m_gridIndexOfRow.at(row))
                                                       : NO_MID_SIDE_GRID);
        input.solids.push_back(solid.part);
    }
    declareSkinGeometry(index, input);
    m_solidPart = std::make_unique<SolidPart>(std::move(input));
    markSkinDeclarationsUsed(index);
    reportSkinGeometry();
    for (const SolidPartProblem& problem : m_solidPart->problems())
    {
        const MixedSolid& solid = m_solids[problem.solid];
        addElementProblem(
            solid.card, solid.batchRow,
            solidProblem(solid.card, solid.elementId,
                         problem.severity == SolidPartProblem::Severity::Error ? "invalid-geometry" : "geometry-note",
                         "", problem.message,
                         problem.severity == SolidPartProblem::Severity::Error ? base::Severity::Error
                                                                               : base::Severity::Warning));
    }
    // The kernels refuse what the reading accepts (a folded element...): one kernel each, which the part keeps. A skin
    // read as a smooth surface across an edge where the solid folds (an edge of a block that the declarations do not
    // say sharp) bends the elements along it: the message says where to look.
    for (std::size_t s = 0; s < m_solids.size(); ++s)
        if (m_solidPart->usable(s))
            if (std::optional<SolidPartProblem> problem = m_solidPart->validate(s))
            {
                std::string message = problem->message;
                if (message.find("folds") != std::string::npos)
                    message +=
                        "; the skin of the solids is read as a smooth surface across every edge that the "
                        "geometry declarations and the fold angle of the model leave smooth: declare sharp the "
                        "edges where their surface folds (sharpEdges)";
                addElementProblem(
                    m_solids[s].card, m_solids[s].batchRow,
                    solidProblem(m_solids[s].card, m_solids[s].elementId, "invalid-geometry", "", std::move(message)));
            }

    // Its entities of a static analysis (edges and faces), in both layouts; the interior fields come last
    // (addModalInteriors).
    const std::span<const std::int32_t> ids = nodes.ids();
    const auto idOfGrid = [&](std::size_t grid) { return ids[static_cast<std::size_t>(m_gridRows[grid])]; };
    for (const SolidPartEntity& entity : m_solidPart->entities(EntityLayout::Kind::Static))
    {
        EntityName name;
        if (entity.kind == SolidPartEntity::Kind::Edge)
        {
            const EdgeKey& key = m_solidPart->edges()[entity.index].grids;
            name.kind = EntityKind::Edge;
            name.gridIds = {idOfGrid(key.lower), idOfGrid(key.higher), 0, 0};
        }
        else
        {
            const SolidPartFace& face = m_solidPart->faces()[entity.index];
            name.kind = EntityKind::Face;
            for (std::size_t k = 0; k < face.count; ++k)
                name.gridIds[k] = idOfGrid(face.grids[k]);
        }
        const base::Vec3 position{entity.position[0], entity.position[1], entity.position[2]};
        m_solidEntityRow.push_back(m_staticLayout.add(static_cast<int>(entity.dofCount), position, name));
        m_modalLayout.add(static_cast<int>(entity.dofCount), position, name);
    }

    // The grids that only solids have (no rotation), and the mid-side grids that fix an edge (their translations follow
    // it). A mid-side grid that another element has is refused: it would be two things at once.
    std::set<std::int32_t> otherGrids(m_lineGrids.begin(), m_lineGrids.end());
    for (const MixedShell& shell : m_shells)
    {
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
            otherGrids.insert(m_gridIndexOfRow.at(shell.gridRows[k]));
        for (const std::int32_t mid : shell.midRows)
            if (mid >= 0)
                otherGrids.insert(m_gridIndexOfRow.at(mid));
    }
    for (const MixedSolid& solid : m_solids)
    {
        for (const std::int32_t row : solid.cornerRows)
            if (const std::int32_t grid = m_gridIndexOfRow.at(row); otherGrids.count(grid) == 0)
                m_solidOnlyGrids.insert(grid);
        for (const std::int32_t row : solid.midRows)
            if (row >= 0)
                if (const std::int32_t grid = m_gridIndexOfRow.at(row); otherGrids.count(grid) == 0)
                    m_solidOnlyGrids.insert(grid);
    }
    for (const auto& [grid, edge] : m_solidPart->midSideGrids())
    {
        const auto g = static_cast<std::int32_t>(grid);
        if (otherGrids.count(g) != 0)
        {
            base::Diagnostic diagnostic;
            diagnostic.code = "unsupported-field";
            diagnostic.card = "Node";
            diagnostic.id = m_gridIds[grid];
            diagnostic.message = base::concatenate(
                {"expected the mid-side grid points of the solids of ", MIXED_HYBRID_FORMULATION,
                 " to belong to the solids alone, found grid point ", std::to_string(m_gridIds[grid]),
                 ", which another element has (a mid-side grid point follows its edge); give that element its own "
                 "grid point or leave the mid-side node out"});
            m_problems.push_back(std::move(diagnostic));
            continue;
        }
        m_solidMidSideGrids.emplace(g, edge);
    }
}

void MixedHybridModel::addModalInteriors(const SnapshotIndex& index)
{
    static_cast<void>(index);
    if (m_solidPart == nullptr)
        return;
    const std::span<const SolidPartEntity> modal = m_solidPart->entities(EntityLayout::Kind::Modal);
    for (std::size_t e = m_solidEntityRow.size(); e < modal.size(); ++e)
    {
        const SolidPartEntity& entity = modal[e];
        EntityName name;
        name.kind = EntityKind::Interior;
        name.elementId = m_solids[entity.index].elementId;
        const base::Vec3 position{entity.position[0], entity.position[1], entity.position[2]};
        m_solidEntityRow.push_back(m_modalLayout.add(static_cast<int>(entity.dofCount), position, name));
    }
}

const MixedSolid* MixedHybridModel::solidOf(std::string_view card, std::int64_t batchRow) const
{
    const auto found = m_solidOfRow.find(card);
    if (found == m_solidOfRow.end() || batchRow < 0 || static_cast<std::size_t>(batchRow) >= found->second.size() ||
        m_solidPart == nullptr)
        return nullptr;
    const std::int32_t solid = found->second[static_cast<std::size_t>(batchRow)];
    if (solid < 0 || !m_solidPart->usable(static_cast<std::size_t>(solid)))
        return nullptr;
    return &m_solids[static_cast<std::size_t>(solid)];
}

MixedHybridModel::LocalDofs MixedHybridModel::solidDofs(const MixedSolid& solid, EntityLayout::Kind kind) const
{
    LocalDofs local;
    for (const SolidLocalUnknown& unknown : m_solidPart->localUnknowns(solid.partIndex, kind))
    {
        local.signs.push_back(1.0);
        if (unknown.onGrid)
        {
            const std::int32_t row = m_gridRows[unknown.index];
            local.dofs.push_back(ElementDof::grid(row, static_cast<int>(unknown.component) + 1));
            local.basicGridRows.push_back(row);
        }
        else
        {
            local.dofs.push_back(
                ElementDof::entity(m_solidEntityRow[unknown.index], static_cast<int>(unknown.component)));
            local.basicGridRows.push_back(-1);
        }
    }
    return local;
}

void MixedHybridModel::appendSolidTerms(std::span<const SolidTerm> solidTerms, double factor,
                                        std::vector<FormulationRelationTerm>& terms) const
{
    for (const SolidTerm& term : solidTerms)
    {
        const double coefficient = factor * term.coefficient;
        if (!term.unknown.onGrid)
        {
            terms.push_back(
                {ElementDof::entity(m_solidEntityRow[term.unknown.index], static_cast<int>(term.unknown.component)),
                 coefficient});
            continue;
        }
        // A grid translation in the basic system: u_basic(b) = Σ_s R(b, s) u_CD(s).
        const base::Mat3& frame = m_gridFrames[term.unknown.index];
        const std::size_t b = term.unknown.component;
        for (std::size_t s = 0; s < 3; ++s)
            if (frame(b, s) != 0.0)
                terms.push_back({ElementDof::grid(m_gridRows[term.unknown.index], 1 + static_cast<int>(s)),
                                 coefficient * frame(b, s)});
    }
}

void MixedHybridModel::solidDeterminedComponents(std::vector<HeldGridComponent>& components) const
{
    if (m_solidPart == nullptr)
        return;
    // The rotations of the grid points that only solids have: the solids have none (zero).
    for (const std::int32_t grid : m_solidOnlyGrids)
        for (int component = 4; component <= 6; ++component)
            components.push_back({m_gridRows[static_cast<std::size_t>(grid)], component, 0.0});
    // The translations of the mid-side grid points: the point of their edge.
    for (const auto& [grid, edge] : m_solidMidSideGrids)
        for (int component = 1; component <= 3; ++component)
            components.push_back({m_gridRows[static_cast<std::size_t>(grid)], component, 0.0});
}

void MixedHybridModel::solidDeterminedRow(std::int32_t grid, int component,
                                          std::vector<FormulationRelationTerm>& terms) const
{
    if (component >= 4)
        return; // no rotation: the row is zero
    const std::size_t edge = m_solidMidSideGrids.at(grid);
    const base::Mat3& frame = m_gridFrames[static_cast<std::size_t>(grid)];
    const auto s = static_cast<std::size_t>(component - 1);
    // u_CD(s) = Σ_b R(b, s) u_basic(b).
    for (std::size_t b = 0; b < 3; ++b)
        if (frame(b, s) != 0.0)
            appendSolidTerms(m_solidPart->midSideTerms(edge, b), frame(b, s), terms);
}

void MixedHybridModel::fillSolidComponents(const NodalDisplacements& displacements,
                                           const std::function<void(std::int32_t, int, double)>& write) const
{
    if (m_solidPart == nullptr)
        return;
    for (const std::int32_t grid : m_solidOnlyGrids)
        for (int component = 4; component <= 6; ++component)
            write(m_gridRows[static_cast<std::size_t>(grid)], component, 0.0);
    std::vector<FormulationRelationTerm> terms; // bounded by one row
    for (const auto& [grid, edge] : m_solidMidSideGrids)
        for (int component = 1; component <= 3; ++component)
        {
            terms.clear();
            solidDeterminedRow(grid, component, terms);
            double value = 0.0;
            for (const FormulationRelationTerm& term : terms)
                value += term.coefficient * displacements.valueAt(term.dof);
            write(m_gridRows[static_cast<std::size_t>(grid)], component, value);
        }
}

void MixedHybridModel::solidSupports(const std::map<std::int32_t, BasicSupport>& perGrid,
                                     std::vector<DerivedSupport>* supports,
                                     std::vector<std::vector<FormulationRelationTerm>>* rows,
                                     base::DiagnosticSink& sink) const
{
    if (m_solidPart == nullptr)
        return;
    std::vector<SolidHeldComponent> held;
    std::vector<SolidSkewAxis> skew;
    for (const auto& [grid, support] : perGrid)
    {
        for (std::size_t c = 0; c < 3; ++c)
            if (support.isHeld[c])
                held.push_back({static_cast<std::size_t>(grid), c, support.values[c]});
        for (const ShellSkewAxis& axis : support.skew)
            if (axis.component < 3)
                skew.push_back({static_cast<std::size_t>(grid), axis.direction, axis.component});
    }
    SolidSupportRows skewRows;
    const std::vector<SolidDerivedSupport> derived = m_solidPart->deriveSupports(held, skew, &skewRows);
    for (const SolidDerivedSupport& support : derived)
    {
        if (supports == nullptr)
            break;
        DerivedSupport out;
        out.dof = ElementDof::entity(m_solidEntityRow[support.entity], static_cast<int>(support.dof));
        out.value = support.value;
        // The reaction carried to the grid translations in the basic system, then to the CD systems: f_CD = Rᵀ f_basic.
        for (const SolidSupportTerm& term : support.carriedTo)
        {
            const base::Mat3& frame = m_gridFrames[term.grid];
            for (std::size_t s = 0; s < 3; ++s)
                if (frame(term.component, s) != 0.0)
                    out.carriedTo.push_back({ElementDof::grid(m_gridRows[term.grid], 1 + static_cast<int>(s)),
                                             term.coefficient * frame(term.component, s)});
        }
        supports->push_back(std::move(out));
    }
    for (const SolidSupportRow& row : skewRows.rows)
    {
        if (rows == nullptr)
            break;
        std::vector<FormulationRelationTerm> terms;
        for (const auto& [dof, coefficient] : row.dofs)
            terms.push_back({ElementDof::entity(m_solidEntityRow[row.entity], static_cast<int>(dof)), coefficient});
        rows->push_back(std::move(terms));
    }
    if (supports == nullptr || skewRows.uncarried == 0)
        return;
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Warning;
    diagnostic.code = "support-not-carried-along-edge";
    diagnostic.message = base::concatenate(
        {std::to_string(skewRows.uncarried), " edges and faces of solids of ", MIXED_HYBRID_FORMULATION,
         " have their grid points held along directions that differ (a displacement system that turns over the face: "
         "a cylindrical system on a cylinder): the grid points hold their directions, the edge or the face between "
         "them is free along them; refine the mesh there"});
    sink.report(std::move(diagnostic));
}

std::vector<double> MixedHybridModel::solidCornerChanges(const MixedSolid& solid,
                                                         const ElementTemperatures* temperatures) const
{
    std::vector<double> changes(solid.cornerRows.size(), 0.0);
    if (temperatures == nullptr || !temperatures->hasTemperatureLoad)
        return changes;
    const double reference = solid.classicSection.referenceTemperature;
    for (std::size_t a = 0; a < solid.cornerRows.size(); ++a)
    {
        const std::int32_t node = solid.cornerRows[a];
        const std::optional<double> load = temperatures->load.atGridRow(node);
        if (!load)
            continue;
        const double initial =
            temperatures->initial.isEmpty() ? reference : temperatures->initial.atGridRow(node).value_or(reference);
        changes[a] = *load - initial;
    }
    return changes;
}

bool MixedHybridModel::takesThermalStrain(const MixedSolid& solid, std::span<const double> changes)
{
    if (solid.part.material.thermalExpansion == 0.0)
        return false;
    return std::any_of(changes.begin(), changes.end(), [](double change) { return change != 0.0; });
}

VolumeKernel::PressureField MixedHybridModel::solidFacePressure(const MixedSolid& solid,
                                                                std::span<const std::size_t> faceGrids,
                                                                std::span<const double> cornerPressures) const
{
    std::vector<base::Vec3> corners;
    const std::span<const SolidPoint> grids = m_solidPart->gridPositions();
    // The pressure of each corner of the solid, and whether it is a corner of the face.
    std::vector<double> pressures;
    std::vector<bool> isOnFace;
    for (const std::size_t grid : solid.part.corners)
    {
        corners.push_back({grids[grid][0], grids[grid][1], grids[grid][2]});
        const auto found = std::find(faceGrids.begin(), faceGrids.end(), grid);
        isOnFace.push_back(found != faceGrids.end());
        pressures.push_back(found != faceGrids.end()
                                ? cornerPressures[static_cast<std::size_t>(found - faceGrids.begin())]
                                : 0.0);
    }
    const SolidShape shape = linearShapeOf(solid);
    return [shape, corners, pressures, isOnFace](const SolidPoint& position)
    {
        const SolidShapeValues functions = cornerFunctionsAt(shape, corners, {position[0], position[1], position[2]});
        // On the face the functions of its corners sum to 1; a point the skin reads off the straight face keeps their
        // shares.
        double pressure = 0.0;
        double total = 0.0;
        for (std::size_t i = 0; i < pressures.size(); ++i)
        {
            if (!isOnFace[i])
                continue;
            pressure += functions.n[i] * pressures[i];
            total += functions.n[i];
        }
        return total != 0.0 ? pressure / total : 0.0;
    };
}

SolidLoad MixedHybridModel::solidThermalLoad(const MixedSolid& solid, std::span<const double> changes) const
{
    std::vector<base::Vec3> corners;
    const std::span<const SolidPoint> grids = m_solidPart->gridPositions();
    for (const std::size_t grid : solid.part.corners)
        corners.push_back({grids[grid][0], grids[grid][1], grids[grid][2]});
    const SolidShape shape = linearShapeOf(solid);
    const std::vector<double> values(changes.begin(), changes.end());
    return m_solidPart->thermalLoad(solid.partIndex,
                                    [shape, corners, values](const SolidPoint& position)
                                    {
                                        const SolidShapeValues functions = cornerFunctionsAt(
                                            shape, corners, {position[0], position[1], position[2]});
                                        double change = 0.0;
                                        for (std::size_t i = 0; i < values.size(); ++i)
                                            change += functions.n[i] * values[i];
                                        return change;
                                    });
}

void MixedHybridModel::addBodyForceStresses(const MixedSolid& solid, const BodyAcceleration* acceleration,
                                            const VolumeKernel& kernel, Vector& beta)
{
    const double density = solid.part.material.density;
    if (acceleration == nullptr || density == 0.0 || acceleration->isZero())
        return;
    // The body force ρ a(x), affine in the position (basic axes): its constants, then its gradient by columns
    // (VolumeKernel::affineBodyStresses).
    std::array<double, VolumeKernel::AFFINE_BODY_FIELDS> coefficients{};
    for (std::size_t c = 0; c < 3; ++c)
    {
        coefficients[c] = density * acceleration->constant[c];
        for (std::size_t m = 0; m < 3; ++m)
            coefficients[3 + 3 * m + c] = density * acceleration->gradient(c, m);
    }
    const Vector particular = kernel.affineBodyStresses(coefficients);
    for (std::size_t i = 0; i < beta.size(); ++i)
        beta[i] += particular[i];
}

VolumeKernel MixedHybridModel::solidThermalKernel(const MixedSolid& solid, EntityLayout::Kind kind,
                                                  std::span<const double> changes) const
{
    const SolidLoad load = solidThermalLoad(solid, changes);
    SolidResult<VolumeKernel> kernel =
        m_solidPart->buildKernel(solid.partIndex, kind, false, std::span<const SolidLoad>(&load, 1));
    if (!kernel.ok())
        throw std::logic_error(base::concatenate({"MixedHybridModel: solid ", std::to_string(solid.elementId), ": ",
                                                  kernel.refusal(), " (its check accepted it: a bug of O_o_Solver)"}));
    return std::move(kernel.value());
}

MixedHybridModel::SolidCaseStresses MixedHybridModel::solidStresses(const MixedSolid& solid, EntityLayout::Kind kind,
                                                                    const NodalDisplacements& displacements) const
{
    // The local unknowns: the grid translations in the basic system (u_basic = R u_CD), the entity DOFs.
    const LocalDofs local = solidDofs(solid, kind);
    Vector x(local.dofs.size(), 0.0);
    for (std::size_t i = 0; i < local.dofs.size(); ++i)
    {
        const ElementDof& dof = local.dofs[i];
        if (local.basicGridRows[i] < 0)
        {
            x[i] = displacements.valueAt(dof);
            continue;
        }
        const base::Mat3& frame = m_gridFrames[static_cast<std::size_t>(gridIndexOfRow(dof.row))];
        const auto b = static_cast<std::size_t>(dof.component - 1);
        for (std::size_t s = 0; s < 3; ++s)
            x[i] += frame(b, s) * displacements.valueAt(ElementDof::grid(dof.row, 1 + static_cast<int>(s)));
    }
    SolidCaseStresses out;
    std::vector<double> changes = solidCornerChanges(solid, displacements.temperatures);
    // The loads whose particular stresses the stresses take (spec section 10.3): the thermal strain of the case, with
    // which the kernel is built (the kernel the part keeps has no load), and the body force of its accelerations,
    // which every kernel has.
    if (takesThermalStrain(solid, changes))
    {
        out.kernel = std::make_shared<const VolumeKernel>(solidThermalKernel(solid, kind, changes));
        const std::array<double, 1> factors{1.0};
        out.beta = SolidPart::stressCoefficients(*out.kernel, x, factors);
        out.cornerChanges = std::move(changes);
    }
    else
    {
        // The stresses and the geometric stiffness read no stiffness: the kernel as the part keeps it.
        std::variant<SolidPart::SharedKernel, std::string> kernel =
            m_solidPart->kernelOf(solid.partIndex, kind, SolidPart::SolidKernelRequest{});
        // The check of the model builds the kernels the part keeps whole; it checks the geometry of the others, and
        // their matrices may refuse them here.
        if (const std::string* message = std::get_if<std::string>(&kernel))
            throw base::SolverError(base::concatenate(
                {"solid element ", std::to_string(solid.elementId), " of ", MIXED_HYBRID_FORMULATION, ": ", *message,
                 "; its geometry or its material makes its matrices singular: check its grid points and its "
                 "material"}));
        out.kernel = std::get<SolidPart::SharedKernel>(std::move(kernel));
        out.beta = SolidPart::stressCoefficients(*out.kernel, x, {});
    }
    addBodyForceStresses(solid, displacements.bodyAcceleration, *out.kernel, out.beta);
    return out;
}

void MixedHybridModel::gridPointStresses(EntityLayout::Kind kind, const NodalDisplacements& displacements,
                                         const SolidSelection& takes, std::vector<GridPointStress>& stresses) const
{
    if (m_solidPart == nullptr)
        return;
    // The corner stresses of the solids taken (an empty list leaves a solid out of the means).
    std::vector<std::vector<VoigtVector>> corners(m_solids.size());
    bool any = false;
    for (const MixedSolid& solid : m_solids)
    {
        if (!m_solidPart->usable(solid.partIndex) || (takes && !takes(solid.card, solid.elementId)))
            continue;
        const SolidCaseStresses state = solidStresses(solid, kind, displacements);
        corners[solid.partIndex] = SolidPart::pointStresses(*state.kernel, state.beta).corners;
        any = true;
    }
    if (!any)
        return;
    const std::vector<std::optional<VoigtVector>> means = m_solidPart->gridStresses(corners);
    for (std::size_t grid = 0; grid < means.size(); ++grid)
        if (means[grid])
            stresses.push_back({m_gridRows[grid], *means[grid]});
}

void MixedHybridModel::appendTieTerms(const TieUnknown& unknown, double factor,
                                      std::vector<FormulationRelationTerm>& terms) const
{
    switch (unknown.kind)
    {
    case TieUnknown::Kind::Grid:
    {
        // A grid translation in the basic system: u_basic(b) = Σ_s R(b, s) u_CD(s).
        const base::Mat3& frame = m_gridFrames[unknown.index];
        for (std::size_t s = 0; s < 3; ++s)
            if (frame(unknown.dof, s) != 0.0)
                terms.push_back({ElementDof::grid(m_gridRows[unknown.index], 1 + static_cast<int>(s)),
                                 factor * frame(unknown.dof, s)});
        return;
    }
    case TieUnknown::Kind::SolidEntity:
        terms.push_back({ElementDof::entity(m_solidEntityRow[unknown.index], static_cast<int>(unknown.dof)), factor});
        return;
    case TieUnknown::Kind::ShellUnknown:
    {
        const ShellPartUnknown part = m_shellPart->describe(unknown.index);
        terms.push_back({ElementDof::entity(m_shellEntityRow[part.index], static_cast<int>(part.component)), factor});
        return;
    }
    case TieUnknown::Kind::AloneEdge:
        terms.push_back({ElementDof::entity(m_edges[unknown.index].entity, static_cast<int>(unknown.dof)), factor});
        return;
    }
}

void MixedHybridModel::solidTies(std::span<const HeldGridComponent> held, std::vector<ElementDof>& dependents,
                                 std::vector<std::size_t>& offsets, std::vector<FormulationRelationTerm>& terms,
                                 base::DiagnosticSink& sink) const
{
    if (m_solidPart == nullptr)
        return;
    // The entity DOFs this constraint set holds (a tie is not written on a held unknown, as the reference's): those no
    // element stiffens, and those the supports of the shells hold (the rotation of an edge whose grid points hold their
    // rotations). The supports of the solids never hold a DOF that follows.
    std::vector<DerivedSupport> supports;
    for (const ElementDof& dof : m_unstiffenedEntityDofs)
        supports.push_back({dof, 0.0, {}});
    base::DiagnosticSink quiet; // the notes of the supports are reported by deriveSupports
    shellSupports(basicSupportsOf(held), &supports, nullptr, quiet);
    static_cast<void>(sink);
    std::set<std::pair<std::int32_t, int>> heldEntityDofs;
    for (const DerivedSupport& support : supports)
        heldEntityDofs.insert({support.dof.row, support.dof.component});
    std::vector<FormulationRelationTerm> dependent; // one DOF
    const auto isHeld = [&](const TieUnknown& unknown)
    {
        dependent.clear();
        appendTieTerms(unknown, 1.0, dependent);
        return dependent.size() == 1 &&
               heldEntityDofs.count({dependent.front().dof.row, dependent.front().dof.component}) != 0;
    };
    for (const SolidTie& tie : m_solidPart->ties(isHeld))
    {
        dependent.clear();
        appendTieTerms(tie.dependent, 1.0, dependent);
        dependents.push_back(dependent.front().dof);
        for (const TieTerm& term : tie.terms)
            appendTieTerms(term.unknown, term.coefficient, terms);
        offsets.push_back(terms.size());
    }
}

void MixedHybridModel::solidAffineField(EntityLayout::Kind kind, const base::Mat3& gradient,
                                        std::span<double> entityValues) const
{
    if (m_solidPart == nullptr)
        return;
    const EntityLayout& entities = layout(kind);
    const std::vector<double> values = m_solidPart->affineField(gradient, kind);
    const std::span<const SolidPartEntity> partEntities = m_solidPart->entities(kind);
    for (std::size_t e = 0; e < partEntities.size(); ++e)
    {
        const auto offset = static_cast<std::size_t>(entities.dofOffset(m_solidEntityRow[e]));
        for (std::size_t d = 0; d < partEntities[e].dofCount; ++d)
            entityValues[offset + d] = values[m_solidPart->dofOffset(e) + d];
    }
}

} // namespace oos::elements::mixedHybrid
