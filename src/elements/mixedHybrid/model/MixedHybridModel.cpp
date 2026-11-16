#include "oos/elements/mixedHybrid/model/MixedHybridModel.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/bars_beams/bar2/Bar2Element.hpp"
#include "oos/elements/bars_beams/beam2/Beam2Element.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementFormulation.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/rods/rod2/Rod2Element.hpp"
#include "oos/elements/rods/rod2InlineSection/Rod2InlineSectionElement.hpp"
#include "oos/elements/rods/tube2/Tube2Element.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string_view>

namespace oos::elements::mixedHybrid
{

namespace
{

// The cards whose geometry the formulation reads from the mesh: the shells (their surface) and the solids (their
// skin). The line elements alone are straight and need no declaration.
constexpr std::array<std::string_view, 10> CARDS_READ_FROM_THE_MESH = {
    "ShellQuad4", "ShellTria3", "ShellQuad4Drilling", "ShellTria3Drilling", "ShellQuad8",
    "ShellTria6", "SolidTetra", "SolidPyramid",       "SolidPenta",         "SolidHexa"};

// The DOFs of an edge of lines alone: the moments of the displacement trace, 3 (k − 1), and the twist, k (the line
// kernel's edge layout without the k shear slots a shell edge has).
constexpr int ALONE_EDGE_DOFS = 3 * (static_cast<int>(MIXED_HYBRID_DEGREE) - 1) + static_cast<int>(MIXED_HYBRID_DEGREE);
// The first twist DOF of an edge of lines alone.
constexpr int ALONE_EDGE_FIRST_TWIST = 3 * (static_cast<int>(MIXED_HYBRID_DEGREE) - 1);

// An entity DOF whose stiffness diagonal is at most this share of the largest diagonal of its lines carries nothing
// (the functions across a rod, the twist of a rod without torsion constant): held at 0, as the reference holds the
// unknowns that no element stiffens.
constexpr double UNSTIFFENED_DIAGONAL_RATIO = 1.0e-14;

Vec3Of<double> toArray(const base::Vec3& v)
{
    return {v[0], v[1], v[2]};
}

base::Diagnostic elementDiagnostic(std::string_view card, std::int32_t elementId, std::string code, std::string field,
                                   std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.code = std::move(code);
    diagnostic.card = std::string(card);
    diagnostic.id = elementId;
    diagnostic.field = std::move(field);
    diagnostic.message = std::move(message);
    return diagnostic;
}

// The line kernel's section of a rod.
LineSection rodSection(const RodData& rod)
{
    LineSection section = LineSection::rod(rod.material.youngModulus, rod.material.shearModulus, rod.section.area,
                                           rod.section.torsionalConstant, rod.material.density);
    section.nonStructuralMass = rod.section.nonStructuralMass;
    section.thermalExpansion = rod.material.expansion;
    return section;
}

// The line kernel's section of a bar or beam (its first and last stations: the kernel's law is linear between the
// ends), or the problem.
std::variant<LineSection, std::string> beamSection(const BeamData& beam)
{
    const BeamSection& stations = beam.section;
    if (stations.count > 2)
        return std::string("expected a section linear between the ends in MixedHybridFormulation, found ") +
               std::to_string(stations.count) + " stations";
    const BeamStation& a = stations.stations[0];
    const BeamStation& b = stations.stations[stations.count - 1];
    const double e = beam.material.youngModulus;
    const double g = beam.material.shearModulus;
    // A plane without shear flexibility (factor <= 0) takes the rigid shear of the kernel: k G A = factor E A.
    std::optional<std::array<double, 2>> shearFactors;
    if (stations.shearFactor1 > 0.0 || stations.shearFactor2 > 0.0)
    {
        const double rigid = g > 0.0 ? RIGID_SHEAR_STIFFNESS_FACTOR * e / g : 0.0;
        shearFactors = std::array<double, 2>{stations.shearFactor1 > 0.0 ? stations.shearFactor1 : rigid,
                                             stations.shearFactor2 > 0.0 ? stations.shearFactor2 : rigid};
    }
    const std::array<std::array<double, 5>, 2> ends{
        {{a.area, a.inertia1, a.inertia2, a.productOfInertia, a.torsionalConstant},
         {b.area, b.inertia1, b.inertia2, b.productOfInertia, b.torsionalConstant}}};
    LineSection section =
        LineSection::beam(e, g, beam.material.density, ends, shearFactors, {{a.neutralAxis, b.neutralAxis}});
    section.nonStructuralMass = 0.5 * (a.nonStructuralMass + b.nonStructuralMass);
    section.thermalExpansion = beam.material.expansion;
    return section;
}

// The lines of the batch of a rod card.
template <class Family>
void readRods(const ElementBatchView& view, std::vector<MixedLine>& lines, std::vector<std::int32_t>& lineOfRow,
              std::vector<std::pair<std::int64_t, base::Diagnostic>>& problems)
{
    const typename Family::Columns columns(view);
    for (std::int64_t row = 0; row < view.rowCount(); ++row)
    {
        Resolution<RodData> resolved = Family::resolve(view, columns, row);
        if (!resolved)
        {
            problems.emplace_back(row, std::move(resolved.problem));
            continue;
        }
        const RodData& rod = *resolved.value;
        MixedLine line;
        line.card = Family().card();
        line.batchRow = row;
        line.elementId = view.elementId(row);
        line.gridRows = {rod.geometry.rowA, rod.geometry.rowB};
        line.section = rodSection(rod);
        line.classic = rod;
        lineOfRow[static_cast<std::size_t>(row)] = static_cast<std::int32_t>(lines.size());
        lines.push_back(std::move(line));
    }
}

// The lines of the batch of a bar or beam card.
template <class Family>
void readBeams(const ElementBatchView& view, std::vector<MixedLine>& lines, std::vector<std::int32_t>& lineOfRow,
               std::vector<std::pair<std::int64_t, base::Diagnostic>>& problems)
{
    const typename Family::Columns columns(view);
    const std::string_view card = Family().card();
    for (std::int64_t row = 0; row < view.rowCount(); ++row)
    {
        Resolution<BeamData> resolved = Family::resolve(view, columns, row);
        if (!resolved)
        {
            problems.emplace_back(row, std::move(resolved.problem));
            continue;
        }
        const BeamData& beam = *resolved.value;
        const std::int32_t id = view.elementId(row);
        std::variant<LineSection, std::string> section = beamSection(beam);
        if (const std::string* message = std::get_if<std::string>(&section))
        {
            problems.emplace_back(row,
                                  elementDiagnostic(card, id, "unsupported-field", "stations",
                                                    base::concatenate({*message, " on element ", std::to_string(id),
                                                                       "; give its section at its ends only"})));
            continue;
        }
        MixedLine line;
        line.card = card;
        line.batchRow = row;
        line.elementId = id;
        line.gridRows = {beam.geometry.rowA, beam.geometry.rowB};
        line.section = std::get<LineSection>(section);
        // The orientation: the element's y axis (the part of the orientation vector across its axis); the offsets of
        // the axis from the grid points, in basic.
        const base::Mat3& axes = beam.geometry.axes;
        line.element = line.element.oriented({axes(0, 1), axes(1, 1), axes(2, 1)})
                           .withOffsets(toArray(beam.geometry.offsetA), toArray(beam.geometry.offsetB));
        // The pin flags: the components of the end DOFs [u, θ] in the element system that the line does not connect
        // to its grid point (the card's other family checks that they leave no mechanism).
        for (std::size_t end = 0; end < 2; ++end)
            for (std::size_t component = 0; component < 6; ++component)
                if (beam.geometry.isReleased[6 * end + component])
                {
                    const std::size_t axis = component % 3;
                    line.releases.push_back(
                        {end, component >= 3, {axes(0, axis), axes(1, axis), axes(2, axis)}, -1, 0});
                }
        line.classic = beam;
        lineOfRow[static_cast<std::size_t>(row)] = static_cast<std::int32_t>(lines.size());
        lines.push_back(std::move(line));
    }
}

} // namespace

const MixedHybridModel& MixedHybridModel::of(const SnapshotIndex& index)
{
    return index.computedOnce<MixedHybridModel>([&] { return MixedHybridModel(index); });
}

MixedHybridModel::MixedHybridModel(const SnapshotIndex& index)
{
    checkDeclarations(index);
    readLines(index);
    readShells(index);
    readSolids(index);
    buildGridsAndEdges(index);
    buildShellPart(index);
    buildAloneEdges(index);
    buildSolidPart(index);
    buildEntities(index);
    addModalInteriors(index);
}

void MixedHybridModel::checkDeclarations(const SnapshotIndex& index)
{
    const input::ModelSnapshot& snapshot = index.snapshot();
    if (snapshot.formulationGeometry())
        return;
    for (const std::string_view card : CARDS_READ_FROM_THE_MESH)
    {
        const input::ElementBatch* batch = snapshot.findElementBatch(card);
        if (batch == nullptr || batch->rowCount() == 0)
            continue;
        base::Diagnostic diagnostic;
        diagnostic.code = "geometry-declarations-missing";
        diagnostic.card = std::string(card);
        diagnostic.message = base::concatenate(
            {"expected the geometry declarations of a model whose shells or solids are in ", MIXED_HYBRID_FORMULATION,
             " (its raw edges, raw elements and sharp edges: where the surface is not smooth), found none and ",
             std::to_string(batch->rowCount()), " ", card,
             " elements; give them with the formulationGeometry option (an empty declaration says that every edge "
             "is smooth)"});
        m_problems.push_back(std::move(diagnostic));
        return;
    }
}

void MixedHybridModel::addElementProblem(std::string_view card, std::int64_t batchRow, base::Diagnostic problem)
{
    m_elementProblems.emplace(std::make_pair(std::string(card), batchRow), std::move(problem));
}

void MixedHybridModel::readLines(const SnapshotIndex& index)
{
    const input::ModelSnapshot& snapshot = index.snapshot();
    const auto read = [&](std::string_view card, auto reader)
    {
        const input::ElementBatch* batch = snapshot.findElementBatch(card);
        if (batch == nullptr || batch->rowCount() == 0)
            return;
        const ElementBatchView view(*batch, index);
        std::vector<std::int32_t>& lineOfRow =
            m_lineOfRow
                .emplace(std::string(card), std::vector<std::int32_t>(static_cast<std::size_t>(batch->rowCount()), -1))
                .first->second;
        std::vector<std::pair<std::int64_t, base::Diagnostic>> problems;
        reader(view, m_lines, lineOfRow, problems);
        for (auto& [row, problem] : problems)
            addElementProblem(card, row, std::move(problem));
    };
    read("Rod2", readRods<Rod2Element>);
    read("Rod2InlineSection", readRods<Rod2InlineSectionElement>);
    read("Tube2", readRods<Tube2Element>);
    read("Bar2", readBeams<Bar2Element>);
    read("Beam2", readBeams<Beam2Element>);
}

void MixedHybridModel::buildGridsAndEdges(const SnapshotIndex& index)
{
    const std::span<const std::int32_t> ids = index.snapshot().nodes().ids();
    const auto idOf = [&](std::int32_t row) { return ids[static_cast<std::size_t>(row)]; };

    // The grids of the model, by increasing ID: an order that does not depend on the node table's.
    std::vector<std::int32_t> rows;
    for (const MixedLine& line : m_lines)
        rows.insert(rows.end(), line.gridRows.begin(), line.gridRows.end());
    for (const MixedShell& shell : m_shells)
    {
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
            rows.push_back(shell.gridRows[k]);
        for (const std::int32_t mid : shell.midRows)
            if (mid >= 0)
                rows.push_back(mid);
    }
    for (const MixedSolid& solid : m_solids)
    {
        rows.insert(rows.end(), solid.cornerRows.begin(), solid.cornerRows.end());
        for (const std::int32_t mid : solid.midRows)
            if (mid >= 0)
                rows.push_back(mid);
    }
    std::sort(rows.begin(), rows.end(), [&](std::int32_t a, std::int32_t b) { return idOf(a) < idOf(b); });
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    m_gridRows = rows;
    const base::FixedListView<double> positions = index.snapshot().nodes().positionsInBasic();
    for (std::size_t i = 0; i < m_gridRows.size(); ++i)
    {
        m_gridIndexOfRow.emplace(m_gridRows[i], static_cast<std::int32_t>(i));
        m_gridIds.push_back(idOf(m_gridRows[i]));
        const std::span<const double> xyz = positions[m_gridRows[i]];
        m_gridPositions.push_back({xyz[0], xyz[1], xyz[2]});
    }
    // The CD-to-basic rotation of each grid (the identity for CD 0 or without the snapshot's rotations).
    const input::NodeTable& nodes = index.snapshot().nodes();
    for (const std::int32_t row : m_gridRows)
    {
        base::Mat3 frame = base::Mat3::identity();
        if (nodes.hasCdRotations())
        {
            const std::span<const double> values = nodes.cdRotations().subspan(9 * static_cast<std::size_t>(row), 9);
            for (std::size_t k = 0; k < 9; ++k)
                frame.values[k] = values[k];
        }
        m_gridFrames.push_back(frame);
    }
    // The edges of the shells: a line on one of them takes its kinematics from the shells (the ShellPart).
    std::set<std::pair<std::int32_t, std::int32_t>> shellEdges;
    for (const MixedShell& shell : m_shells)
        for (std::size_t k = 0; k < shell.cornerCount; ++k)
        {
            const std::int32_t a = m_gridIndexOfRow.at(shell.gridRows[k]);
            const std::int32_t b = m_gridIndexOfRow.at(shell.gridRows[(k + 1) % shell.cornerCount]);
            shellEdges.insert({std::min(a, b), std::max(a, b)});
        }

    // The edges of the lines, by their grid indices lower first (the order of the grid IDs).
    std::map<std::pair<std::int32_t, std::int32_t>, std::int32_t> edgeOf;
    for (std::size_t l = 0; l < m_lines.size(); ++l)
    {
        MixedLine& line = m_lines[l];
        const std::int32_t a = m_gridIndexOfRow.at(line.gridRows[0]);
        const std::int32_t b = m_gridIndexOfRow.at(line.gridRows[1]);
        line.element.grids = {static_cast<std::size_t>(a), static_cast<std::size_t>(b)};
        const std::pair<std::int32_t, std::int32_t> key{std::min(a, b), std::max(a, b)};
        edgeOf.emplace(key, 0);
        line.forward = a < b;
        m_lineGrids.insert(a);
        m_lineGrids.insert(b);
        // Marked for buildShellPart, which numbers the lines on the edges of shells from 0.
        line.shellLine = shellEdges.count(key) != 0 ? 0 : -1;
    }
    // Numbered in the order of the keys: deterministic whatever the order of the rows.
    for (auto& [key, edgeIndex] : edgeOf)
    {
        edgeIndex = static_cast<std::int32_t>(m_edges.size());
        MixedEdge edge;
        edge.grids = {key.first, key.second};
        m_edges.push_back(std::move(edge));
    }
    for (std::size_t l = 0; l < m_lines.size(); ++l)
    {
        MixedLine& line = m_lines[l];
        const auto a = static_cast<std::int32_t>(line.element.grids[0]);
        const auto b = static_cast<std::int32_t>(line.element.grids[1]);
        line.edge = edgeOf.at({std::min(a, b), std::max(a, b)});
        m_edges[static_cast<std::size_t>(line.edge)].lines.push_back(static_cast<std::int32_t>(l));
    }
    // The lines of an edge by increasing element ID: the first gives the frame of an edge alone.
    for (MixedEdge& edge : m_edges)
        std::sort(edge.lines.begin(), edge.lines.end(),
                  [&](std::int32_t x, std::int32_t y)
                  {
                      return m_lines[static_cast<std::size_t>(x)].elementId <
                             m_lines[static_cast<std::size_t>(y)].elementId;
                  });
}

void MixedHybridModel::buildAloneEdges(const SnapshotIndex& index)
{
    static_cast<void>(index);
    for (MixedEdge& edge : m_edges)
    {
        if (edge.lines.empty() || m_lines[static_cast<std::size_t>(edge.lines.front())].shellLine >= 0)
            continue;
        // An edge that no shell has: the frame of its first line.
        const MixedLine& first = m_lines[static_cast<std::size_t>(edge.lines.front())];
        const base::Vec3& lower = m_gridPositions[static_cast<std::size_t>(edge.grids[0])];
        const base::Vec3& upper = m_gridPositions[static_cast<std::size_t>(edge.grids[1])];
        std::variant<AloneEdge, std::string> alone =
            aloneEdgeOfLine(first.element, first.section,
                            {static_cast<std::size_t>(edge.grids[0]), static_cast<std::size_t>(edge.grids[1])},
                            {lower[0], lower[1], lower[2]}, {upper[0], upper[1], upper[2]}, MIXED_HYBRID_DEGREE, 0);
        if (const std::string* message = std::get_if<std::string>(&alone))
        {
            for (const std::int32_t l : edge.lines)
            {
                const MixedLine& line = m_lines[static_cast<std::size_t>(l)];
                addElementProblem(
                    line.card, line.batchRow,
                    elementDiagnostic(line.card, line.elementId, "invalid-geometry", "",
                                      base::concatenate({"element ", std::to_string(line.elementId), ": ", *message})));
            }
            continue;
        }
        edge.alone = std::get<AloneEdge>(std::move(alone));
    }
}

void MixedHybridModel::buildEntities(const SnapshotIndex& index)
{
    const input::NodeTable& nodes = index.snapshot().nodes();
    const base::FixedListView<double> positions = nodes.positionsInBasic();
    const std::span<const std::int32_t> ids = nodes.ids();
    for (std::size_t e = 0; e < m_edges.size(); ++e)
    {
        MixedEdge& edge = m_edges[e];
        if (!edge.alone)
            continue;
        const Vec3Of<double>& lower = edge.alone->lowerPoint;
        const Vec3Of<double>& upper = edge.alone->upperPoint;
        const EntityName name{
            EntityKind::Edge,
            {ids[static_cast<std::size_t>(m_gridRows[static_cast<std::size_t>(edge.grids[0])])],
             ids[static_cast<std::size_t>(m_gridRows[static_cast<std::size_t>(edge.grids[1])])], 0, 0},
            0};
        const base::Vec3 middle{0.5 * (lower[0] + upper[0]), 0.5 * (lower[1] + upper[1]), 0.5 * (lower[2] + upper[2])};
        edge.entity = m_staticLayout.add(ALONE_EDGE_DOFS, middle, name);
        m_modalLayout.add(ALONE_EDGE_DOFS, middle, name);

        // The DOFs of the edge that its lines do not stiffen (the functions across a rod...): held at 0.
        std::array<double, ALONE_EDGE_DOFS> diagonal{};
        double largest = 0.0;
        for (const std::int32_t l : edge.lines)
        {
            const MixedLine& line = m_lines[static_cast<std::size_t>(l)];
            std::variant<LineData, std::string> data = aloneLineData(*edge.alone, e, line.element, line.section, {0, 3},
                                                                     linePointCount(MIXED_HYBRID_DEGREE), false);
            if (const std::string* message = std::get_if<std::string>(&data))
            {
                addElementProblem(
                    line.card, line.batchRow,
                    elementDiagnostic(line.card, line.elementId, "invalid-geometry", "",
                                      base::concatenate({"element ", std::to_string(line.elementId), ": ", *message})));
                continue;
            }
            const Matrix& stiffness = std::get<LineData>(data).stiffness;
            for (std::size_t i = 0; i < stiffness.rows(); ++i)
                largest = std::max(largest, std::fabs(stiffness(i, i)));
            const std::vector<LineUnknown> unknowns = aloneLineUnknowns(MIXED_HYBRID_DEGREE);
            for (std::size_t i = 0; i < unknowns.size(); ++i)
            {
                const LineUnknown& unknown = unknowns[i];
                if (unknown.kind == LineUnknownKind::EdgeFunction)
                    diagonal[3 * unknown.index + unknown.component] += std::fabs(stiffness(i, i));
                else if (unknown.kind == LineUnknownKind::EdgeTwist)
                    diagonal[static_cast<std::size_t>(ALONE_EDGE_FIRST_TWIST) + unknown.index] +=
                        std::fabs(stiffness(i, i));
            }
        }
        for (int d = 0; d < ALONE_EDGE_DOFS; ++d)
            if (diagonal[static_cast<std::size_t>(d)] <= UNSTIFFENED_DIAGONAL_RATIO * largest)
                m_unstiffenedEntityDofs.push_back(ElementDof::entity(edge.entity, d));
    }

    // The ends of the lines with released components (pin flags): one entity per end, by increasing element ID.
    std::vector<std::size_t> released;
    for (std::size_t l = 0; l < m_lines.size(); ++l)
    {
        if (m_lines[l].releases.empty())
            continue;
        if (m_lines[l].shellLine >= 0)
        {
            addElementProblem(m_lines[l].card, m_lines[l].batchRow,
                              elementDiagnostic(m_lines[l].card, m_lines[l].elementId, "unsupported-field", "pinFlagsA",
                                                base::concatenate({"expected no pin flags on a line element on an edge "
                                                                   "of shells in ",
                                                                   MIXED_HYBRID_FORMULATION, ", found some on element ",
                                                                   std::to_string(m_lines[l].elementId),
                                                                   ": such a line has no displacement of its own; "
                                                                   "remove them"})));
            continue;
        }
        released.push_back(l);
    }
    std::sort(released.begin(), released.end(),
              [&](std::size_t a, std::size_t b) { return m_lines[a].elementId < m_lines[b].elementId; });
    for (const std::size_t l : released)
    {
        MixedLine& line = m_lines[l];
        for (std::size_t end = 0; end < 2; ++end)
        {
            int count = 0;
            for (const LineRelease& release : line.releases)
                count += release.end == end ? 1 : 0;
            if (count == 0)
                continue;
            const std::int32_t gridRow = line.gridRows[end];
            const std::span<const double> xyz = positions[gridRow];
            const EntityName name{
                EntityKind::LineEnd, {ids[static_cast<std::size_t>(gridRow)], 0, 0, 0}, line.elementId};
            const std::int32_t entity = m_staticLayout.add(count, {xyz[0], xyz[1], xyz[2]}, name);
            m_modalLayout.add(count, {xyz[0], xyz[1], xyz[2]}, name);
            int index = 0;
            for (LineRelease& release : line.releases)
                if (release.end == end)
                {
                    release.entity = entity;
                    release.index = index++;
                }
        }
    }
}

const EntityLayout& MixedHybridModel::layout(EntityLayout::Kind kind) const
{
    return kind == EntityLayout::Kind::Modal ? m_modalLayout : m_staticLayout;
}

void MixedHybridModel::reportProblems(base::DiagnosticSink& sink) const
{
    for (const base::Diagnostic& problem : m_problems)
        sink.report(problem);
}

std::vector<base::Diagnostic> MixedHybridModel::elementProblems(std::string_view card) const
{
    std::vector<base::Diagnostic> problems;
    for (const auto& [key, problem] : m_elementProblems)
        if (key.first == card)
            problems.push_back(problem);
    return problems;
}

const MixedLine* MixedHybridModel::lineOf(std::string_view card, std::int64_t batchRow) const
{
    const auto found = m_lineOfRow.find(card);
    if (found == m_lineOfRow.end() || batchRow < 0 || static_cast<std::size_t>(batchRow) >= found->second.size())
        return nullptr;
    const std::int32_t line = found->second[static_cast<std::size_t>(batchRow)];
    if (line < 0)
        return nullptr;
    const MixedLine& data = m_lines[static_cast<std::size_t>(line)];
    if (data.shellLine >= 0)
        return m_shellPart != nullptr && m_shellPart->usable() ? &data : nullptr;
    // A line whose edge could not be read has no data either.
    return m_edges[static_cast<std::size_t>(data.edge)].alone ? &data : nullptr;
}

MixedHybridModel::LocalDofs MixedHybridModel::aloneLineDofs(const MixedLine& line) const
{
    const MixedEdge& edge = m_edges[static_cast<std::size_t>(line.edge)];
    LocalDofs local;
    for (const LineUnknown& unknown : aloneLineUnknowns(MIXED_HYBRID_DEGREE))
    {
        local.signs.push_back(1.0);
        const auto component = static_cast<int>(unknown.component);
        switch (unknown.kind)
        {
        case LineUnknownKind::GridTranslation:
        {
            const std::int32_t row = m_gridRows[static_cast<std::size_t>(edge.grids[unknown.end])];
            local.dofs.push_back(ElementDof::grid(row, component + 1));
            local.basicGridRows.push_back(row);
            break;
        }
        case LineUnknownKind::EdgeFunction:
            local.dofs.push_back(ElementDof::entity(edge.entity, 3 * static_cast<int>(unknown.index) + component));
            local.basicGridRows.push_back(-1);
            break;
        case LineUnknownKind::EdgeTwist:
            local.dofs.push_back(
                ElementDof::entity(edge.entity, ALONE_EDGE_FIRST_TWIST + static_cast<int>(unknown.index)));
            local.basicGridRows.push_back(-1);
            break;
        case LineUnknownKind::GridRotation:
        {
            const std::int32_t row = line.gridRows[unknown.end];
            local.dofs.push_back(ElementDof::grid(row, component + 4));
            local.basicGridRows.push_back(row);
            break;
        }
        }
    }
    return local;
}

void MixedHybridModel::determinedGridComponents(EntityLayout::Kind, std::vector<HeldGridComponent>& components) const
{
    shellDeterminedComponents(components);
    solidDeterminedComponents(components);
}

void MixedHybridModel::determinedRow(EntityLayout::Kind, std::int32_t gridRow, int component,
                                     std::vector<FormulationRelationTerm>& terms) const
{
    const std::int32_t grid = gridIndexOfRow(gridRow);
    if (m_solidOnlyGrids.count(grid) != 0 || m_solidMidSideGrids.count(grid) != 0)
        solidDeterminedRow(grid, component, terms);
    else
        shellDeterminedRow(grid, component, terms);
}

void MixedHybridModel::fillDeterminedComponents(EntityLayout::Kind, const NodalDisplacements& displacements,
                                                const std::function<void(std::int32_t, int, double)>& write) const
{
    fillShellComponents(displacements, write);
    fillSolidComponents(displacements, write);
}

void MixedHybridModel::deriveSupports(EntityLayout::Kind, std::span<const HeldGridComponent> held,
                                      std::vector<DerivedSupport>& supports, base::DiagnosticSink& sink) const
{
    // The DOFs no element stiffens, held at 0 in every constraint set.
    for (const ElementDof& dof : m_unstiffenedEntityDofs)
        supports.push_back({dof, 0.0, {}});
    if ((m_shellPart == nullptr || !m_shellPart->usable()) && m_solidPart == nullptr)
        return;
    const std::map<std::int32_t, BasicSupport> basic = basicSupportsOf(held, sink);
    shellSupports(basic, supports, sink);
    solidSupports(basic, supports);
}

void MixedHybridModel::affineField(EntityLayout::Kind kind, const base::Mat3& gradient, const base::Vec3& translation,
                                   std::span<double> entityValues) const
{
    solidAffineField(kind, gradient, entityValues);
    const EntityLayout& entities = layout(kind);
    // The rotation of the field: the axial vector of the skew part of its gradient, ω = ½ (A32 − A23, A13 − A31,
    // A21 − A12).
    const base::Vec3 omega{0.5 * (gradient(2, 1) - gradient(1, 2)), 0.5 * (gradient(0, 2) - gradient(2, 0)),
                           0.5 * (gradient(1, 0) - gradient(0, 1))};
    for (const MixedEdge& edge : m_edges)
    {
        if (!edge.alone)
            continue;
        // An edge of lines alone is straight: an affine field is linear along it, so its functions (which vanish at
        // the grids and hold the part beyond the linear one) take nothing; its twist coefficient 0 is the rotation
        // about the edge times its chord, ω · (X_upper − X_lower) (aloneLineRigidMotion).
        const Vec3Of<double>& lower = edge.alone->lowerPoint;
        const Vec3Of<double>& upper = edge.alone->upperPoint;
        const double twist =
            omega[0] * (upper[0] - lower[0]) + omega[1] * (upper[1] - lower[1]) + omega[2] * (upper[2] - lower[2]);
        entityValues[static_cast<std::size_t>(entities.dofOffset(edge.entity)) + ALONE_EDGE_FIRST_TWIST] = twist;
    }
    // The shells: their edges and interiors take the field of their own unknowns (the edge rotations from ω).
    if (m_shellPart != nullptr && m_shellPart->usable())
    {
        const std::size_t first = m_shellPart->firstEntityUnknown();
        std::vector<double> partValues(m_shellPart->unknownCount() - first, 0.0);
        const std::array<SpaceVector, 3> rows{{{gradient(0, 0), gradient(0, 1), gradient(0, 2)},
                                               {gradient(1, 0), gradient(1, 1), gradient(1, 2)},
                                               {gradient(2, 0), gradient(2, 1), gradient(2, 2)}}};
        m_shellPart->affineField(rows, {translation[0], translation[1], translation[2]}, partValues);
        const std::vector<ShellPartEntity>& partEntities = m_shellPart->entities();
        for (std::size_t e = 0; e < partEntities.size(); ++e)
        {
            const auto offset = static_cast<std::size_t>(entities.dofOffset(m_shellEntityRow[e]));
            for (std::size_t d = 0; d < partEntities[e].dofCount; ++d)
                entityValues[offset + d] = partValues[partEntities[e].firstUnknown - first + d];
        }
    }
}

std::optional<FormulationModel::FormulationFace> MixedHybridModel::faceOf(std::span<const std::int32_t> gridRows) const
{
    if (gridRows.size() != 3 && gridRows.size() != 4)
        return std::nullopt;
    std::vector<std::size_t> grids;
    for (const std::int32_t row : gridRows)
    {
        const std::int32_t grid = gridIndexOfRow(row);
        if (grid < 0)
            return std::nullopt;
        grids.push_back(static_cast<std::size_t>(grid));
    }
    // The vector area of the face in the order given (the right-hand rule: ½ (x2 − x0) × (x3 − x1) for four points)
    // and its centre.
    const auto at = [&](std::size_t k) { return m_gridPositions[grids[k % grids.size()]]; };
    const base::Vec3 area = grids.size() == 3 ? 0.5 * base::cross(at(1) - at(0), at(2) - at(0))
                                              : 0.5 * base::cross(at(2) - at(0), at(3) - at(1));
    base::Vec3 centre{};
    for (std::size_t k = 0; k < grids.size(); ++k)
        centre = centre + at(k) / static_cast<double>(grids.size());
    std::vector<std::size_t> sorted = grids;
    std::sort(sorted.begin(), sorted.end());

    // A shell whose corners are those grid points: the pressure along its normal (the right-hand rule of its corners),
    // of the sign of the two normals.
    if (m_shellPart != nullptr && m_shellPart->usable())
        for (const MixedShell& shell : m_shells)
        {
            if (shell.cornerCount != grids.size())
                continue;
            std::vector<std::size_t> corners;
            for (std::size_t k = 0; k < shell.cornerCount; ++k)
                corners.push_back(static_cast<std::size_t>(m_gridIndexOfRow.at(shell.gridRows[k])));
            std::vector<std::size_t> sortedCorners = corners;
            std::sort(sortedCorners.begin(), sortedCorners.end());
            if (sortedCorners != sorted)
                continue;
            const auto corner = [&](std::size_t k) { return m_gridPositions[corners[k % corners.size()]]; };
            const base::Vec3 normal = corners.size() == 3 ? base::cross(corner(1) - corner(0), corner(2) - corner(0))
                                                          : base::cross(corner(2) - corner(0), corner(3) - corner(1));
            return FormulationFace{shell.card, shell.batchRow, base::dot(normal, area) >= 0.0 ? 1.0 : -1.0};
        }
    // A face of a solid: the pressure of the solid acts inward, the force P × area: p_in = −P (n · n_out), the outward
    // normal pointing from the centre of the solid to that of the face.
    if (m_solidPart != nullptr)
        if (const std::optional<std::pair<std::size_t, std::size_t>> found = m_solidPart->solidFaceOf(grids))
        {
            const MixedSolid& solid = m_solids[found->first];
            base::Vec3 solidCentre{};
            for (const std::size_t grid : solid.part.corners)
                solidCentre = solidCentre + m_gridPositions[grid] / static_cast<double>(solid.part.corners.size());
            const double outward = base::dot(area, centre - solidCentre);
            return FormulationFace{solid.card, solid.batchRow, outward >= 0.0 ? -1.0 : 1.0};
        }
    return std::nullopt;
}

} // namespace oos::elements::mixedHybrid

namespace oos::elements
{

const FormulationModel* formulationModelOf(const SnapshotIndex& index, const ElementRegistry& registry)
{
    if (!registry.isMixedHybrid())
        return nullptr;
    return &mixedHybrid::MixedHybridModel::of(index);
}

} // namespace oos::elements
