#include "oos/output/sections/SectionLoads.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/threading/ParallelFor.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementResultBlock.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingGeometry.hpp"
#include "oos/elements/shells/shellTria3Drilling/ShellTria3DrillingGeometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace oos::output
{

namespace
{

// The sections of one task of the rows (displacementRows): a few dozen members each.
constexpr std::size_t SECTIONS_PER_PIECE = 8;

// Sums the entries of a row per DOF, in their order (a stable sort): the contributions of the members in member order.
void mergeRow(std::vector<std::pair<base::DofIndex, double>>& row)
{
    std::stable_sort(row.begin(), row.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::size_t next = 0;
    for (std::size_t i = 0; i < row.size(); ++i)
    {
        if (next > 0 && row[next - 1].first == row[i].first)
            row[next - 1].second += row[i].second;
        else
            row[next++] = row[i];
    }
    row.resize(next);
}

constexpr std::array<std::string_view, 8> SHELL_COMPONENTS{"nx", "ny", "nxy", "mx", "my", "mxy", "qx", "qy"};
constexpr std::array<std::string_view, 8> BAR_COMPONENTS{
    "bendingMomentA1", "bendingMomentA2", "bendingMomentB1", "bendingMomentB2", "shear1", "shear2", "axial", "torque"};
constexpr std::array<std::string_view, 14> BEAM_COMPONENTS{
    "bendingMoment1A", "bendingMoment2A", "shear1A", "shear2A", "axialForceA", "totalTorqueA", "warpingTorqueA",
    "bendingMoment1B", "bendingMoment2B", "shear1B", "shear2B", "axialForceB", "totalTorqueB", "warpingTorqueB"};
// The CBEAM force columns of one station (Beam2Element::outputColumns), read at end A and at end B.
constexpr std::array<std::string_view, 7> BEAM_STATION_COLUMNS{
    "bendingMoment1", "bendingMoment2", "shear1", "shear2", "axialForce", "totalTorque", "warpingTorque"};
constexpr std::array<std::string_view, 2> ROD_COMPONENTS{"axial", "torsion"};

// The DOFs of the largest element a section takes (a CQUAD8: eight grid points of six DOFs), the bound of the entries
// of the rows before they are merged.
constexpr std::uint64_t MEMBER_DOF_BOUND = 8 * 6;
using Transform = std::array<double, MAX_SECTION_COMPONENTS * MAX_SECTION_COMPONENTS>;

// The number of corners of a shell card (CQUAD8 and CTRIA6 list the corners first): its frame is that of the corners.
std::size_t cornerCountOf(std::string_view card) noexcept
{
    if (card == "ShellQuad4" || card == "ShellQuad4Drilling" || card == "ShellQuad8")
        return 4;
    if (card == "ShellTria3" || card == "ShellTria3Drilling" || card == "ShellTria6")
        return 3;
    return 0;
}

// The element frame and the area of a shell, from the positions of its corners in the element set it is evaluated on.
struct ShellPlane
{
    base::Mat3 axes = base::Mat3::identity(); // columns x, y, n in basic
    double area = 0.0;
    bool isValid = false;
};

ShellPlane shellPlaneOf(const elements::ElementBatchView& view, std::int64_t row)
{
    const std::size_t corners = cornerCountOf(view.card());
    const input::ElementBatch& batch = view.batch();
    if (corners == 0 || !batch.reader().hasColumn("nodesRows") || !batch.reader().validity("nodesRows").isValid(row))
        return {};
    const std::span<const std::int32_t> rows = batch.nodeRows()[row];
    if (rows.size() < corners)
        return {};
    if (corners == 4)
    {
        std::array<base::Vec3, 4> positions;
        for (std::size_t i = 0; i < 4; ++i)
            positions[i] = view.positionInBasic(rows[i]);
        const elements::ShellQuad4DrillingGeometryResult quad = elements::buildShellQuad4DrillingGeometry(positions);
        if (quad.problem != elements::QuadGeometryProblem::None)
            return {};
        return {quad.geometry.axes, quad.geometry.area(), true};
    }
    std::array<base::Vec3, 3> positions;
    for (std::size_t i = 0; i < 3; ++i)
        positions[i] = view.positionInBasic(rows[i]);
    const elements::ShellTria3DrillingGeometryResult tria = elements::buildShellTria3DrillingGeometry(positions);
    if (tria.isDegenerate)
        return {};
    return {tria.geometry.axes, tria.geometry.area, true};
}

// The cosine and sine of the angle from the x axis of the element to the projection x_s of `axis` on its plane; nothing
// when the projection is too short (SECTION_AXIS_MIN_PROJECTION).
std::optional<std::pair<double, double>> sectionAngle(const base::Mat3& axes, base::Vec3 axis)
{
    const base::Vec3 x{axes(0, 0), axes(1, 0), axes(2, 0)};
    const base::Vec3 y{axes(0, 1), axes(1, 1), axes(2, 1)};
    const base::Vec3 n{axes(0, 2), axes(1, 2), axes(2, 2)};
    const base::Vec3 projected = axis - base::dot(axis, n) * n;
    const double length = base::norm(projected);
    if (!(length > SECTION_AXIS_MIN_PROJECTION * base::norm(axis)))
        return std::nullopt;
    const base::Vec3 sectionX = projected / length;
    return std::make_pair(base::dot(sectionX, x), base::dot(sectionX, y));
}

// The rotation T of the shell components (nx, ny, nxy, mx, my, mxy, qx, qy) into the section frame (SectionLoads.hpp),
// scaled by `scale`, row-major in a Transform of MAX_SECTION_COMPONENTS columns: x_s = c·x_e + s·y_e and
// y_s = side·(−s·x_e + c·y_e), side −1 for an element whose normal opposes the section's (its y_s turned so that it
// points as the section's: n_s × x_s).
Transform shellTransform(double c, double s, double side, double scale)
{
    // The projections of x_s and y_s on x_e and y_e.
    const double a = c;
    const double b = s;
    const double p = -side * s;
    const double q = side * c;
    Transform t{};
    const auto at = [&](std::size_t i, std::size_t j) -> double& { return t[i * MAX_SECTION_COMPONENTS + j]; };
    for (const std::size_t first : {std::size_t{0}, std::size_t{3}})
    {
        at(first, first) = scale * a * a;
        at(first, first + 1) = scale * b * b;
        at(first, first + 2) = scale * 2.0 * a * b;
        at(first + 1, first) = scale * p * p;
        at(first + 1, first + 1) = scale * q * q;
        at(first + 1, first + 2) = scale * 2.0 * p * q;
        at(first + 2, first) = scale * a * p;
        at(first + 2, first + 1) = scale * b * q;
        at(first + 2, first + 2) = scale * (a * q + b * p);
    }
    at(6, 6) = scale * a;
    at(6, 7) = scale * b;
    at(7, 6) = scale * p;
    at(7, 7) = scale * q;
    return t;
}

Transform scaledIdentity(std::size_t count, double scale)
{
    Transform t{};
    for (std::size_t i = 0; i < count; ++i)
        t[i * MAX_SECTION_COMPONENTS + i] = scale;
    return t;
}

base::Diagnostic sectionProblem(const SectionDefinition& definition, std::string code, std::string message,
                                std::string card = "", std::int64_t id = 0)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Error;
    diagnostic.code = std::move(code);
    diagnostic.card = card.empty() ? std::string("SECTION") : std::move(card);
    diagnostic.id = id;
    diagnostic.message = base::concatenate(
        {"section ", definition.name.empty() ? std::to_string(definition.id) : definition.name, ": ", message});
    return diagnostic;
}

// The coordinate system `frameId` of the snapshot (0: basic), or nothing when the model does not have it.
std::optional<input::CoordinateFrame> frameOf(const input::ModelSnapshot& snapshot, std::int32_t frameId)
{
    if (frameId == 0)
        return input::CoordinateFrame{};
    if (!snapshot.hasTable(input::TableRole::CoordinateSystems))
        return std::nullopt;
    const input::CoordinateSystemTable& systems = snapshot.coordinateSystems();
    const std::span<const std::int32_t> ids = systems.ids();
    for (std::size_t row = 0; row < ids.size(); ++row)
        if (ids[row] == frameId)
            return systems.frameAt(static_cast<std::int64_t>(row));
    return std::nullopt;
}

} // namespace

std::optional<SectionFamily> sectionFamilyOf(std::string_view card) noexcept
{
    if (cornerCountOf(card) != 0)
        return SectionFamily::Shell;
    if (card == "Bar2")
        return SectionFamily::Bar;
    if (card == "Beam2")
        return SectionFamily::Beam;
    if (card == "Rod2" || card == "Rod2InlineSection" || card == "Tube2")
        return SectionFamily::Rod;
    return std::nullopt;
}

std::string_view nameOf(SectionFamily family) noexcept
{
    switch (family)
    {
    case SectionFamily::Shell:
        return "shell";
    case SectionFamily::Bar:
        return "bar";
    case SectionFamily::Beam:
        return "beam";
    case SectionFamily::Rod:
        return "rod";
    }
    return "shell";
}

std::span<const std::string_view> sectionComponents(SectionFamily family) noexcept
{
    switch (family)
    {
    case SectionFamily::Shell:
        return SHELL_COMPONENTS;
    case SectionFamily::Bar:
        return BAR_COMPONENTS;
    case SectionFamily::Beam:
        return BEAM_COMPONENTS;
    case SectionFamily::Rod:
        return ROD_COMPONENTS;
    }
    return SHELL_COMPONENTS;
}

SectionLoads::SectionLoads(std::vector<SectionDefinition> definitions, const input::ModelSnapshot& snapshot,
                           const assembly::ElementSet& set, const assembly::DofMap& dofMap, base::DiagnosticSink& sink)
    : m_definitions(std::move(definitions)), m_set(&set), m_dofMap(&dofMap), m_columnsOfBatch(set.batches().size())
{
    resolve(snapshot, sink);
}

void SectionLoads::resolve(const input::ModelSnapshot& snapshot, base::DiagnosticSink& sink)
{
    // The elements named by the definitions, found by one pass over the batches (no index of every element).
    std::vector<std::int32_t> wanted;
    for (const SectionDefinition& definition : m_definitions)
        for (const SectionMember& member : definition.members)
            wanted.push_back(member.elementId);
    std::sort(wanted.begin(), wanted.end());
    wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
    std::vector<std::pair<std::size_t, std::int64_t>> where(wanted.size(), {0, -1}); // batch, row
    const std::span<const assembly::AssemblyBatch> batches = m_set->batches();
    for (std::size_t b = 0; b < batches.size(); ++b)
    {
        const elements::ElementBatchView& view = batches[b].view;
        for (std::int64_t row = 0; row < view.rowCount(); ++row)
        {
            const auto found = std::lower_bound(wanted.begin(), wanted.end(), view.elementId(row));
            if (found == wanted.end() || *found != view.elementId(row))
                continue;
            const auto index = static_cast<std::size_t>(found - wanted.begin());
            if (where[index].second < 0)
                where[index] = {b, row};
        }
    }

    for (std::size_t d = 0; d < m_definitions.size(); ++d)
    {
        const SectionDefinition& definition = m_definitions[d];
        Section section;
        section.definition = d;
        bool isValid = true;
        const auto refuse = [&](base::Diagnostic diagnostic)
        {
            sink.report(std::move(diagnostic));
            isValid = false;
        };
        if (definition.members.empty())
            refuse(sectionProblem(definition, "section-empty",
                                  "expected at least one element, found none; give the elements of the section"));
        std::optional<SectionFamily> family;
        std::string familyCard;
        std::vector<std::int32_t> seen;
        double totalWeight = 0.0;
        for (const SectionMember& member : definition.members)
        {
            const std::size_t index = static_cast<std::size_t>(
                std::lower_bound(wanted.begin(), wanted.end(), member.elementId) - wanted.begin());
            if (where[index].second < 0)
            {
                refuse(sectionProblem(definition, "section-unknown-element",
                                      base::concatenate({"names element ", std::to_string(member.elementId),
                                                         ", which the model does not have; correct the elementId"}),
                                      "", member.elementId));
                continue;
            }
            if (std::find(seen.begin(), seen.end(), member.elementId) != seen.end())
            {
                refuse(sectionProblem(definition, "section-duplicate-element",
                                      base::concatenate({"names element ", std::to_string(member.elementId),
                                                         " twice; give it once, with the sum of its weights"}),
                                      "", member.elementId));
                continue;
            }
            seen.push_back(member.elementId);
            const auto [batch, row] = where[index];
            // The family from the neutral card type; the messages name the card by its deck keyword.
            const std::optional<SectionFamily> memberFamily = sectionFamilyOf(batches[batch].view.card());
            const std::string card = std::string(batches[batch].view.card());
            if (!memberFamily)
            {
                refuse(sectionProblem(
                    definition, "section-element-type",
                    base::concatenate(
                        {"names ", card, " ", std::to_string(member.elementId),
                         ", an element type no section takes; a section holds shells "
                         "(ShellQuad4, ShellQuad4Drilling, ShellQuad8, ShellTria3, ShellTria3Drilling, ShellTria6), Bar2, "
                         "Beam2 or rods (Rod2, Rod2InlineSection, Tube2)"}),
                    card, member.elementId));
                continue;
            }
            if (family && *family != *memberFamily)
            {
                refuse(sectionProblem(
                    definition, "section-mixed-families",
                    base::concatenate({"mixes element families: ", familyCard, " (", std::string(nameOf(*family)),
                                       ") and ", card, " ", std::to_string(member.elementId), " (",
                                       std::string(nameOf(*memberFamily)),
                                       "); a section holds elements of one family: split it in two sections"}),
                    card, member.elementId));
                continue;
            }
            if (!family)
            {
                family = memberFamily;
                familyCard = card;
            }
            if (member.weight && !std::isfinite(*member.weight))
            {
                refuse(sectionProblem(definition, "section-weight",
                                      base::concatenate({"gives element ", std::to_string(member.elementId),
                                                         " a weight that is not finite; give a finite weight or "
                                                         "none (the default of its family)"}),
                                      card, member.elementId));
                continue;
            }
            double weight = member.weight.value_or(1.0);
            if (*memberFamily == SectionFamily::Shell && !member.weight)
            {
                const ShellPlane plane = shellPlaneOf(batches[batch].view, row);
                weight = plane.isValid ? plane.area : 0.0;
            }
            totalWeight += weight;
            section.members.push_back({batch, row, member.elementId, member.weight});
        }
        if (isValid && family)
        {
            section.family = *family;
            if (totalWeight == 0.0)
                refuse(sectionProblem(definition, "section-zero-weight",
                                      "has a total weight of zero (its elements' weights sum to 0); give weights whose "
                                      "sum is not zero"));
            if (section.family == SectionFamily::Shell)
            {
                if (definition.axis.has_value() == definition.frameId.has_value())
                    refuse(
                        sectionProblem(definition, "section-frame",
                                       base::concatenate({"expected its frame by an axis in the basic system or by a "
                                                          "coordinate system frameId, exactly one, found ",
                                                          definition.axis ? "both" : "neither",
                                                          "; the shell forces are rotated into the section frame"})));
                else if (definition.frameId)
                {
                    const std::optional<input::CoordinateFrame> frame = frameOf(snapshot, *definition.frameId);
                    if (!frame)
                        refuse(sectionProblem(
                            definition, "section-frame",
                            base::concatenate({"names the coordinate system ", std::to_string(*definition.frameId),
                                               ", which the model does not have; add its coordinate system card "
                                               "or correct the frameId"}),
                            "", *definition.frameId));
                    else if (frame->kind != input::CoordinateFrame::Kind::Rectangular)
                        refuse(
                            sectionProblem(definition, "section-frame",
                                           base::concatenate({"expected a rectangular coordinate system (RectFrame1 or "
                                                              "RectFrame2), found the cylindrical or spherical system ",
                                                              std::to_string(*definition.frameId),
                                                              "; give an axis or a rectangular frameId"}),
                                           "", *definition.frameId));
                    else
                        section.axis = {frame->axes(0, 0), frame->axes(1, 0), frame->axes(2, 0)};
                }
                else
                {
                    section.axis = *definition.axis;
                    if (!(base::norm(section.axis) > 0.0) || !std::isfinite(base::norm(section.axis)))
                        refuse(sectionProblem(definition, "section-frame",
                                              "expected a finite axis of non-zero length; give the direction of the x "
                                              "axis of the section in the basic system"));
                }
                if (isValid)
                    for (const Member& member : section.members)
                    {
                        const elements::ElementBatchView& view = batches[member.batch].view;
                        const ShellPlane plane = shellPlaneOf(view, member.row);
                        if (!plane.isValid)
                            refuse(sectionProblem(definition, "section-geometry",
                                                  base::concatenate({"element ", std::to_string(member.elementId),
                                                                     " has no plane (degenerate corners); correct its "
                                                                     "grid points"}),
                                                  std::string(view.card()), member.elementId));
                        else if (!sectionAngle(plane.axes, section.axis))
                            refuse(sectionProblem(
                                definition, "section-frame",
                                base::concatenate({"its axis is normal to element ", std::to_string(member.elementId),
                                                   " (no projection on its plane); give an axis "
                                                   "that lies in the plane of the section"}),
                                std::string(view.card()), member.elementId));
                    }
            }
        }
        if (!isValid || !family)
            continue;
        // The force columns of the batches of the members, read once (ColumnMap).
        for (const Member& member : section.members)
        {
            ColumnMap& map = m_columnsOfBatch[member.batch];
            if (map.isKnown)
                continue;
            const std::vector<std::string> columns =
                batches[member.batch].family->outputColumns(elements::ElementOutput::Forces);
            const std::span<const std::string_view> components = sectionComponents(section.family);
            for (std::size_t k = 0; k < components.size(); ++k)
            {
                const std::string_view wantedColumn = section.family == SectionFamily::Beam
                                                          ? BEAM_STATION_COLUMNS[k % BEAM_STATION_COLUMNS.size()]
                                                          : components[k];
                const auto found = std::find(columns.begin(), columns.end(), wantedColumn);
                if (found == columns.end())
                    throw std::logic_error(
                        base::concatenate({"SectionLoads: the force columns of ", batches[member.batch].view.card(),
                                           " lack '", std::string(wantedColumn), "'"}));
                map.columns[k] = static_cast<std::size_t>(found - columns.begin());
            }
            map.isKnown = true;
        }
        section.firstRow = m_rowCount;
        m_rowCount += sectionComponents(section.family).size();
        if (section.family == SectionFamily::Bar || section.family == SectionFamily::Beam)
            for (const Member& member : section.members)
                m_lineMembers.emplace_back(member.batch, member.row, m_sections.size());
        m_sections.push_back(std::move(section));
    }
    std::sort(m_lineMembers.begin(), m_lineMembers.end());
}

std::optional<std::size_t> SectionLoads::lineSectionOf(std::size_t batch, std::int64_t row) const
{
    const auto found = std::lower_bound(
        m_lineMembers.begin(), m_lineMembers.end(), std::make_tuple(batch, row, std::size_t{0}),
        [](const auto& a, const auto& b)
        { return std::tie(std::get<0>(a), std::get<1>(a)) < std::tie(std::get<0>(b), std::get<1>(b)); });
    if (found == m_lineMembers.end() || std::get<0>(*found) != batch || std::get<1>(*found) != row)
        return std::nullopt;
    return std::get<2>(*found);
}

void SectionLoads::memberForces(const assembly::ElementSet& set, const Member& member, SectionFamily family,
                                const elements::NodalDisplacements& nodal, std::span<double> forces) const
{
    const assembly::AssemblyBatch& batch = set.batches()[member.batch];
    elements::ElementResultBlock block;
    batch.family->recover(batch.view, {member.row, member.row + 1}, elements::ElementOutput::Forces, nodal, block);
    memberForcesOf(block, member, family, forces);
}

void SectionLoads::memberForcesOf(const elements::ElementResultBlock& block, const Member& member, SectionFamily family,
                                  std::span<double> forces) const
{
    if (block.size() == 0)
    {
        const elements::ElementBatchView& view = m_set->batches()[member.batch].view;
        throw std::logic_error(base::concatenate(
            {"SectionLoads: ", view.card(), " ", std::to_string(member.elementId), " gave no force row"}));
    }
    forcesOfBlock(block, member, family, forces);
}

void SectionLoads::forcesOfBlock(const elements::ElementResultBlock& block, const Member& member, SectionFamily family,
                                 std::span<double> forces) const
{
    const ColumnMap& map = m_columnsOfBatch[member.batch];
    const std::size_t count = sectionComponents(family).size();
    // The row of a shell's centre (grid point 0, the first row its recovery gives), of a bar, a rod; a beam's first
    // (end A) and last (end B) stations.
    std::size_t first = 0;
    if (family == SectionFamily::Shell)
        for (std::size_t i = 0; i < block.size(); ++i)
            if (block.place(i).nodeId == 0)
            {
                first = i;
                break;
            }
    for (std::size_t k = 0; k < count; ++k)
    {
        const bool isEndB = family == SectionFamily::Beam && k >= BEAM_STATION_COLUMNS.size();
        forces[k] = block.values(isEndB ? block.size() - 1 : first)[map.columns[k]];
    }
}

std::vector<Transform> SectionLoads::memberCoefficients(std::size_t s, const assembly::ElementSet& set) const
{
    const Section& section = m_sections[s];
    const std::size_t count = sectionComponents(section.family).size();
    std::vector<double> weights; // bounded by the members
    std::vector<std::pair<double, double>> angles;
    std::vector<base::Vec3> normals;
    base::Vec3 sectionNormal{0.0, 0.0, 0.0}; // Σ area·n_e
    double total = 0.0;
    for (const Member& member : section.members)
    {
        double weight = member.weight.value_or(1.0);
        std::pair<double, double> angle{1.0, 0.0};
        if (section.family == SectionFamily::Shell)
        {
            const ShellPlane plane = shellPlaneOf(set.batches()[member.batch].view, member.row);
            if (!plane.isValid)
                throw std::invalid_argument(
                    base::concatenate({"SectionLoads: element ", std::to_string(member.elementId),
                                       " has no plane in the element set evaluated; give "
                                       "sections of shell elements"}));
            if (!member.weight)
                weight = plane.area;
            const std::optional<std::pair<double, double>> found = sectionAngle(plane.axes, section.axis);
            if (!found)
                throw std::invalid_argument(base::concatenate(
                    {"SectionLoads: the axis of the section is normal to element ", std::to_string(member.elementId),
                     "; give the section an axis in the plane of its elements"}));
            angle = *found;
            const base::Vec3 normal{plane.axes(0, 2), plane.axes(1, 2), plane.axes(2, 2)};
            normals.push_back(normal);
            sectionNormal = sectionNormal + plane.area * normal;
        }
        weights.push_back(weight);
        angles.push_back(angle);
        total += weight;
    }
    // The normal of the section: the area-weighted mean of its shells' (the first's when they cancel); an element whose
    // normal opposes it has its y_s turned (shellTransform), as O_o_Reader's readBlock turns it for element groups.
    if (!normals.empty() && !(base::norm(sectionNormal) > 0.0))
        sectionNormal = normals.front();
    const SectionDefinition& definition = m_definitions[section.definition];
    const double divisor = definition.reduce == SectionReduce::Average ? total : 1.0;
    std::vector<Transform> transforms;
    transforms.reserve(section.members.size());
    for (std::size_t m = 0; m < section.members.size(); ++m)
    {
        const double scale = weights[m] / divisor;
        transforms.push_back(section.family == SectionFamily::Shell
                                 ? shellTransform(angles[m].first, angles[m].second,
                                                  base::dot(normals[m], sectionNormal) < 0.0 ? -1.0 : 1.0, scale)
                                 : scaledIdentity(count, scale));
    }
    return transforms;
}

void SectionLoads::evaluateSection(std::size_t s, const assembly::ElementSet& set,
                                   std::span<const double> displacementsG,
                                   const elements::ElementTemperatures* temperatures, std::span<double> out) const
{
    const Section& section = m_sections[s];
    const std::size_t count = sectionComponents(section.family).size();
    const auto gSize = static_cast<std::size_t>(m_dofMap->gSetSize());
    if (displacementsG.size() != gSize || out.size() != count)
        throw std::invalid_argument(
            base::concatenate({"SectionLoads::evaluateSection: expected ", std::to_string(gSize), " displacements and ",
                               std::to_string(count), " values, found ", std::to_string(displacementsG.size()), " and ",
                               std::to_string(out.size())}));
    const auto gridValues = static_cast<std::size_t>(6) * static_cast<std::size_t>(m_dofMap->gridCount());
    const elements::NodalDisplacements nodal{displacementsG.first(gridValues), displacementsG.subspan(gridValues),
                                             temperatures};
    const std::vector<Transform> transforms = memberCoefficients(s, set);
    std::fill(out.begin(), out.end(), 0.0);
    std::array<double, MAX_SECTION_COMPONENTS> forces{};
    for (std::size_t m = 0; m < section.members.size(); ++m)
    {
        memberForces(set, section.members[m], section.family, nodal, forces);
        for (std::size_t k = 0; k < count; ++k)
        {
            double sum = 0.0;
            for (std::size_t l = 0; l < count; ++l)
                sum += transforms[m][k * MAX_SECTION_COMPONENTS + l] * forces[l];
            out[k] += sum;
        }
    }
}

bool SectionLoads::sectionFieldRates(std::size_t s, const assembly::ElementSet& set,
                                     std::span<const double> displacementsG,
                                     const elements::ElementTemperatures* temperatures,
                                     const elements::DesignedField& field,
                                     const std::function<bool(std::size_t, std::int64_t)>& isReached,
                                     std::span<double> out) const
{
    const Section& section = m_sections[s];
    const std::size_t count = sectionComponents(section.family).size();
    const auto gSize = static_cast<std::size_t>(m_dofMap->gSetSize());
    if (displacementsG.size() != gSize || out.size() != count)
        throw std::invalid_argument(
            base::concatenate({"SectionLoads::sectionFieldRates: expected ", std::to_string(gSize),
                               " displacements and ", std::to_string(count), " values, found ",
                               std::to_string(displacementsG.size()), " and ", std::to_string(out.size())}));
    const auto gridValues = static_cast<std::size_t>(6) * static_cast<std::size_t>(m_dofMap->gridCount());
    const elements::NodalDisplacements nodal{displacementsG.first(gridValues), displacementsG.subspan(gridValues),
                                             temperatures};
    std::fill(out.begin(), out.end(), 0.0);
    std::vector<Transform> transforms; // made at the first member reached
    std::array<double, MAX_SECTION_COMPONENTS> rates{};
    for (std::size_t m = 0; m < section.members.size(); ++m)
    {
        const Member& member = section.members[m];
        if (!isReached(member.batch, member.row))
            continue;
        const assembly::AssemblyBatch& batch = set.batches()[member.batch];
        elements::ElementResultBlock block;
        if (!batch.family->recoveryFieldRates(batch.view, member.row, field, elements::ElementOutput::Forces, nodal,
                                              block) ||
            block.size() == 0)
            return false;
        forcesOfBlock(block, member, section.family, rates);
        const auto read = rates.begin() + static_cast<std::ptrdiff_t>(count);
        if (std::any_of(rates.begin(), read, [](double rate) { return std::isnan(rate); }))
            return false;
        if (transforms.empty())
            transforms = memberCoefficients(s, set);
        for (std::size_t k = 0; k < count; ++k)
        {
            double sum = 0.0;
            for (std::size_t l = 0; l < count; ++l)
                sum += transforms[m][k * MAX_SECTION_COMPONENTS + l] * rates[l];
            out[k] += sum;
        }
    }
    return true;
}

void SectionLoads::evaluate(const assembly::ElementSet& set, std::span<const double> displacementsG,
                            const elements::ElementTemperatures* temperatures, std::span<double> out) const
{
    if (out.size() != m_rowCount)
        throw std::invalid_argument(base::concatenate({"SectionLoads::evaluate: expected ", std::to_string(m_rowCount),
                                                       " values, found ", std::to_string(out.size())}));
    for (std::size_t s = 0; s < m_sections.size(); ++s)
        evaluateSection(s, set, displacementsG, temperatures, out.subspan(m_sections[s].firstRow, componentCount(s)));
}

const SparseRows& SectionLoads::displacementRows(base::MemoryBudget& budget, base::ThreadPool* pool)
{
    if (m_rowsBuilt)
        return m_rows;
    const auto gSize = static_cast<std::size_t>(m_dofMap->gSetSize());
    const auto gridValues = static_cast<std::size_t>(6) * static_cast<std::size_t>(m_dofMap->gridCount());
    // The entries before they are merged per DOF: at most components × the DOFs of the members, reserved first.
    std::uint64_t bound = 0;
    for (const Section& section : m_sections)
        bound += static_cast<std::uint64_t>(sectionComponents(section.family).size()) * section.members.size() *
                 MEMBER_DOF_BOUND;
    const base::MemoryReservation charge =
        budget.reserve(bound * sizeof(std::pair<base::DofIndex, double>), "section rows before merging");
    std::vector<std::vector<std::pair<base::DofIndex, double>>> rows(m_rowCount);
    // The sections of a piece share its unit displacements (one at a time non-zero); each writes its own rows only.
    base::parallelFor(
        pool, m_sections.size(), SECTIONS_PER_PIECE,
        [&](std::size_t begin, std::size_t end)
        {
            base::BudgetedBuffer<double> unit(budget, gSize, "section unit displacements");
            std::fill(unit.data(), unit.data() + gSize, 0.0);
            const elements::NodalDisplacements nodal{unit.span().first(gridValues), unit.span().subspan(gridValues),
                                                     nullptr};
            elements::ElementMatrixBlock stiffness;
            std::vector<elements::ElementDof> unitDofs;               // bounded by an element
            std::vector<elements::ElementResultBlock> unitRecoveries; // one per DOF of an element
            std::array<double, MAX_SECTION_COMPONENTS> forces{};
            for (std::size_t s = begin; s < end; ++s)
            {
                const Section& section = m_sections[s];
                const std::size_t count = sectionComponents(section.family).size();
                const std::vector<Transform> transforms = memberCoefficients(s, *m_set);
                for (std::size_t m = 0; m < section.members.size(); ++m)
                {
                    const Member& member = section.members[m];
                    const assembly::AssemblyBatch& batch = m_set->batches()[member.batch];
                    // PERFORMANCE: the family's rows at every unit displacement from one formulation of the
                    // element (the same bits as a recovery per DOF, which resolved and formulated it again for
                    // each DOF, after a stiffness for its DOFs).
                    // Measured: a design point of a stiffened barrel panel (3 118 sections of CQUAD4,
                    // CBAR and CBEAM), its connectivity phase 1.2 → 0.15 s on one thread.
                    const bool isUnitByFamily = batch.family->recoverAtUnitDisplacements(
                        batch.view, member.row, elements::ElementOutput::Forces, unitDofs, unitRecoveries);
                    if (!isUnitByFamily)
                    {
                        stiffness.clear();
                        batch.family->computeStiffness(batch.view, {member.row, member.row + 1}, stiffness);
                        if (stiffness.empty())
                            continue;
                        const std::span<const elements::ElementDof> stiffnessDofs = stiffness.dofs(0);
                        unitDofs.assign(stiffnessDofs.begin(), stiffnessDofs.end());
                    }
                    // One column of R_e per DOF of the element: its forces under a unit displacement.
                    for (std::size_t d = 0; d < unitDofs.size(); ++d)
                    {
                        const auto g = static_cast<std::size_t>(m_dofMap->dofOf(unitDofs[d]));
                        if (isUnitByFamily)
                            memberForcesOf(unitRecoveries[d], member, section.family, forces);
                        else
                        {
                            unit[g] = 1.0;
                            memberForces(*m_set, member, section.family, nodal, forces);
                            unit[g] = 0.0;
                        }
                        for (std::size_t k = 0; k < count; ++k)
                        {
                            double coefficient = 0.0;
                            for (std::size_t l = 0; l < count; ++l)
                                coefficient += transforms[m][k * MAX_SECTION_COMPONENTS + l] * forces[l];
                            if (coefficient != 0.0)
                                rows[section.firstRow + k].emplace_back(static_cast<base::DofIndex>(g), coefficient);
                        }
                    }
                }
                // Merged per DOF, the members' contributions summed in member order (a stable sort).
                for (std::size_t k = 0; k < count; ++k)
                    mergeRow(rows[section.firstRow + k]);
            }
        });
    m_rows = SparseRows(rows, budget);
    m_rowsBuilt = true;
    return m_rows;
}

} // namespace oos::output
