#include "oos/output/sections/ZoneContour.hpp"

#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementMatrixBlock.hpp"
#include "oos/elements/core/ElementVectorBlock.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace oos::output
{

namespace
{

using RowEntries = std::vector<std::pair<base::DofIndex, double>>;

// The cards whose stiffness is given component by component (a spring, a CBUSH, a gap): the components of an enclosed
// one kept outside are those it has a stiffness for, exactly; the other elements act through every component of a
// kind (translations, rotations) in which their stiffness has a row (so a CBUSH counts only its components with a
// stiffness).
constexpr std::array<std::string_view, 7> COMPONENTWISE_CARDS{
    "SpringDofProperty", "SpringDofInline", "SpringScalarProperty", "SpringScalarInline", "Bushing", "Bushing1D", "Gap2"};

base::Diagnostic zoneProblem(const ZoneDefinition& definition, base::Severity severity, std::string code,
                             std::string message, std::int64_t id = 0)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = std::move(code);
    diagnostic.card = "ZONE";
    diagnostic.id = id;
    diagnostic.message = base::concatenate(
        {"zone ", definition.name.empty() ? std::to_string(definition.id) : definition.name, ": ", message});
    return diagnostic;
}

// The coordinate system `frameId` of the snapshot (0: basic), or nothing.
std::optional<input::CoordinateFrame> frameOf(const input::ModelSnapshot& snapshot, std::int32_t frameId)
{
    if (frameId == 0 || frameId == DISPLACEMENT_SYSTEMS)
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

// The grid points an element couples, with the components its matrix has a non-zero row at (the stiffness, or the
// mass of a mass element), as ElementConnectivity takes the nodes from the element matrices.
struct ElementUse
{
    std::map<std::int32_t, ComponentMask> grids; // grid row → components with a non-zero row
    bool hasStiffness = false;
};

ElementUse elementUse(const assembly::AssemblyBatch& batch, std::int64_t row)
{
    ElementUse use;
    elements::ElementMatrixBlock block;
    const elements::ElementRange range{row, row + 1};
    if (batch.family->facets().has(elements::Facet::Stiffness))
        batch.family->computeStiffness(batch.view, range, block);
    use.hasStiffness = !block.empty();
    if (block.empty() && batch.family->facets().has(elements::Facet::Mass))
        batch.family->computeMass(batch.view, range, elements::MassOptions{}, block);
    for (std::size_t i = 0; i < block.size(); ++i)
    {
        const std::span<const elements::ElementDof> dofs = block.dofs(i);
        const std::span<const double> matrix = block.matrix(i);
        const std::size_t n = dofs.size();
        for (std::size_t r = 0; r < n; ++r)
        {
            if (dofs[r].kind != input::PointKind::Grid)
                continue;
            ComponentMask& mask = use.grids[dofs[r].row];
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                sum += std::abs(matrix[r * n + j]);
            if (sum > 0.0)
                mask = static_cast<ComponentMask>(mask | (1U << (dofs[r].component - 1)));
        }
    }
    return use;
}

// The components through which an element kept outside acts on a grid point (COMPONENTWISE_CARDS).
ComponentMask outsideComponents(std::string_view card, ComponentMask rows)
{
    if (std::find(COMPONENTWISE_CARDS.begin(), COMPONENTWISE_CARDS.end(), card) != COMPONENTWISE_CARDS.end())
        return rows;
    ComponentMask mask = 0;
    if ((rows & TRANSLATIONS) != 0)
        mask = static_cast<ComponentMask>(mask | TRANSLATIONS);
    if ((rows & ROTATIONS) != 0)
        mask = static_cast<ComponentMask>(mask | ROTATIONS);
    return mask;
}

// The components of the zone's system that the contour data of a point give: its own (CD) when its displacement
// system has the zone's axes, else every component of each kind it has (a component of CD is then no component of
// the zone's system).
ComponentMask zoneComponents(ComponentMask components, const base::Mat3& rotation)
{
    double offIdentity = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            offIdentity = std::max(offIdentity, std::abs(rotation(i, j) - (i == j ? 1.0 : 0.0)));
    if (offIdentity <= SAME_FRAME_TOLERANCE)
        return components;
    return outsideComponents("", components);
}

// W of the class comment, row r (0 to 5) and column k (0 to 5) of a point: the resultant about the zone's origin in
// its system of the six components of a force in the point's displacement system.
double resultantWeight(const ZoneContour::ZonePoint& point, std::size_t r, std::size_t k, const base::Mat3& toBasic,
                       const base::Mat3& axes)
{
    if (r < 3)
        return k < 3 ? point.rotation(r, k) : 0.0;
    if (k >= 3)
        return point.rotation(r - 3, k - 3);
    // Aᵀ·(arm × (R_cd·e_k)).
    const base::Vec3 direction{toBasic(0, k), toBasic(1, k), toBasic(2, k)};
    const base::Vec3 moment = base::cross(point.arm, direction);
    return axes(0, r - 3) * moment[0] + axes(1, r - 3) * moment[1] + axes(2, r - 3) * moment[2];
}

// Sorts the entries of a row by DOF and sums those of one DOF, in the order they were added (deterministic).
void mergeRow(RowEntries& row)
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
    row.erase(std::remove_if(row.begin(), row.end(), [](const auto& entry) { return entry.second == 0.0; }), row.end());
}

// target += scale · source (entries appended, merged later).
void addScaled(RowEntries& target, double scale, const RowEntries& source)
{
    if (scale == 0.0)
        return;
    for (const auto& [dof, value] : source)
        target.emplace_back(dof, scale * value);
}

} // namespace

ComponentMask componentMaskOf(std::string_view digits)
{
    ComponentMask mask = 0;
    for (const char digit : digits)
    {
        if (digit < '1' || digit > '6')
            throw std::invalid_argument(
                base::concatenate({"expected component digits 1 to 6, found '", std::string(digits), "'"}));
        mask = static_cast<ComponentMask>(mask | (1U << (digit - '1')));
    }
    return mask;
}

std::string componentDigitsOf(ComponentMask mask)
{
    std::string digits;
    for (int c = 1; c <= 6; ++c)
        if (hasComponent(mask, c))
            digits.push_back(static_cast<char>('0' + c));
    return digits;
}

ZoneContour::ZoneContour(std::vector<ZoneDefinition> definitions, std::span<const ZoneSubcaseConstraint> subcases,
                         const input::ModelSnapshot& snapshot, const assembly::ElementSet& set,
                         const assembly::DofMap& dofMap, base::MemoryBudget& budget, base::DiagnosticSink& sink)
    : m_definitions(std::move(definitions)),
      m_set(&set),
      m_dofMap(&dofMap),
      m_zeros(budget, static_cast<std::size_t>(dofMap.gSetSize()), "zone contour zero displacements")
{
    std::fill(m_zeros.data(), m_zeros.data() + m_zeros.size(), 0.0);
    m_zones.resize(m_definitions.size());
    m_rows.resize(m_definitions.size());
    for (std::size_t d = 0; d < m_definitions.size(); ++d)
        resolve(d, subcases, snapshot, budget, sink);
}

std::optional<std::size_t> ZoneContour::findZone(std::int32_t id) const noexcept
{
    for (std::size_t z = 0; z < m_definitions.size(); ++z)
        if (m_definitions[z].id == id)
            return z;
    return std::nullopt;
}

void ZoneContour::resolve(std::size_t d, std::span<const ZoneSubcaseConstraint> subcases,
                          const input::ModelSnapshot& snapshot, base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    const ZoneDefinition& definition = m_definitions[d];
    // The card types are neutral names; the messages show their deck keywords.

    Zone& zone = m_zones[d];
    zone.definition = d;
    ContourReport& report = zone.report;
    report.zoneId = definition.id;
    report.name = definition.name;
    report.unit = definition.unit;
    bool isValid = true;
    const auto refuse = [&](base::Diagnostic diagnostic)
    {
        sink.report(diagnostic);
        report.diagnostics.push_back(std::move(diagnostic));
        isValid = false;
    };
    const auto warn = [&](base::Diagnostic diagnostic)
    {
        sink.report(diagnostic);
        report.diagnostics.push_back(std::move(diagnostic));
    };
    const input::PointIndex& points = m_dofMap->points();

    // The rigid elements of the model by ID: their grid rows (rigid_elements_table).
    const input::RigidTable* rigid =
        snapshot.hasTable(input::TableRole::RigidElements) ? &snapshot.rigidElements() : nullptr;
    std::map<std::int32_t, std::pair<std::string, std::set<std::int32_t>>> rigidElements;
    if (rigid != nullptr)
    {
        const std::span<const std::int32_t> ids = rigid->ids();
        const std::span<const std::int32_t> nodeIds = rigid->nodeIds();
        const base::DictionaryView types = rigid->types();
        for (std::size_t row = 0; row < ids.size(); ++row)
        {
            auto& [card, rows] = rigidElements[ids[row]];
            card = std::string(types[static_cast<std::int64_t>(row)]);
            if (const std::optional<std::int32_t> gridRow = points.findGrid(nodeIds[row]))
                rows.insert(*gridRow);
        }
    }

    // The members: the listed elements, and the enclosed ones the policy takes (rigid elements and 0D elements on one
    // point always, the others with enclosed="include").
    std::vector<std::int32_t> memberIds(definition.elements.begin(), definition.elements.end());
    std::vector<const EnclosedElement*> keptOutside;
    for (const EnclosedElement& enclosed : definition.enclosed)
    {
        const bool isRigid = rigidElements.count(enclosed.elementId) != 0;
        std::string decision;
        if (isRigid)
            decision = "held";
        else if (enclosed.isPointElement)
            decision = "added";
        else if (definition.enclosedPolicy == EnclosedPolicy::Include)
            decision = "included";
        else
            decision = definition.enclosedPolicy == EnclosedPolicy::Warn ? "kept outside" : "excluded";
        if (decision == "kept outside" || decision == "excluded")
            keptOutside.push_back(&enclosed);
        else
            memberIds.push_back(enclosed.elementId);
        report.enclosed.push_back({enclosed.elementId, enclosed.card, decision});
    }
    std::sort(memberIds.begin(), memberIds.end());
    memberIds.erase(std::unique(memberIds.begin(), memberIds.end()), memberIds.end());
    report.elements = memberIds;
    if (memberIds.empty())
        refuse(zoneProblem(definition, base::Severity::Error, "zone-empty",
                           "expected the elements of the zone, found none; give its element IDs or its unit"));

    // The elements (members and kept outside) in the element set, found by one pass over the batches.
    std::vector<std::int32_t> wanted = memberIds;
    for (const EnclosedElement* enclosed : keptOutside)
        wanted.push_back(enclosed->elementId);
    std::sort(wanted.begin(), wanted.end());
    wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());
    std::vector<std::pair<std::size_t, std::int64_t>> where(wanted.size(), {0, -1});
    const std::span<const assembly::AssemblyBatch> batches = m_set->batches();
    for (std::size_t b = 0; b < batches.size(); ++b)
    {
        const elements::ElementBatchView& view = batches[b].view;
        for (std::int64_t row = 0; row < view.rowCount(); ++row)
        {
            const auto found = std::lower_bound(wanted.begin(), wanted.end(), view.elementId(row));
            if (found != wanted.end() && *found == view.elementId(row))
                where[static_cast<std::size_t>(found - wanted.begin())] = {b, row};
        }
    }
    const auto locate = [&](std::int32_t elementId)
    {
        return where[static_cast<std::size_t>(std::lower_bound(wanted.begin(), wanted.end(), elementId) -
                                              wanted.begin())];
    };

    // The grid points of the members, and which members are fasteners and rigid elements.
    std::set<std::int32_t> zoneRows;
    std::set<std::int32_t> memberRigid;
    std::set<std::int32_t> memberFasteners;
    for (const std::int32_t elementId : memberIds)
    {
        if (const auto found = rigidElements.find(elementId); found != rigidElements.end())
        {
            memberRigid.insert(elementId);
            zoneRows.insert(found->second.second.begin(), found->second.second.end());
            continue;
        }
        const auto [batch, row] = locate(elementId);
        if (row < 0)
        {
            refuse(zoneProblem(definition, base::Severity::Error, "zone-unknown-element",
                               base::concatenate({"names element ", std::to_string(elementId),
                                                  ", which the model does not have (neither an element nor a rigid "
                                                  "element); correct its elements"}),
                               elementId));
            continue;
        }
        const ElementUse use = elementUse(batches[batch], row);
        for (const auto& [gridRow, mask] : use.grids)
            zoneRows.insert(gridRow);
        if (use.hasStiffness)
            zone.members.push_back({batch, row, elementId});
        const std::string_view card = batches[batch].view.card();
        if (card == "Fastener" || card == "Weld")
            memberFasteners.insert(elementId);
    }
    std::sort(zone.members.begin(), zone.members.end(), [](const Member& a, const Member& b)
              { return std::make_pair(a.batch, a.row) < std::make_pair(b.batch, b.row); });

    // The contour: O_o_Reader's boundary, the enclosed elements kept outside, the partial constraints.
    std::map<std::int32_t, ContourPointReport> contour; // by grid row
    for (const ZoneBoundaryPoint& point : definition.boundary)
    {
        const std::optional<std::int32_t> gridRow = points.findGrid(point.grid);
        if (!gridRow)
        {
            refuse(zoneProblem(definition, base::Severity::Error, "zone-unknown-grid",
                               base::concatenate({"its boundary names grid point ", std::to_string(point.grid),
                                                  ", which the model does not have; take the zone again from the "
                                                  "model"}),
                               point.grid));
            continue;
        }
        ContourPointReport& entry = contour[*gridRow];
        entry.grid = point.grid;
        entry.components = static_cast<ComponentMask>(entry.components | point.components);
        entry.isPiercedPoint = entry.isPiercedPoint || point.isPiercedPoint;
        entry.causes.insert(entry.causes.end(), point.causes.begin(), point.causes.end());
        zoneRows.insert(*gridRow);
    }
    std::vector<std::string> warnedElements;
    for (const EnclosedElement* enclosed : keptOutside)
    {
        const auto [batch, row] = locate(enclosed->elementId);
        if (row < 0)
            continue; // a rigid element is never kept outside; an element the set lacks has nothing to give
        const ElementUse use = elementUse(batches[batch], row);
        for (const auto& [gridRow, mask] : use.grids)
        {
            if (zoneRows.count(gridRow) == 0)
                continue;
            const ComponentMask components = outsideComponents(enclosed->card, mask);
            if (components == 0)
                continue;
            ContourPointReport& entry = contour[gridRow];
            entry.grid = m_dofMap->gridId(gridRow);
            entry.components = static_cast<ComponentMask>(entry.components | components);
            entry.causes.push_back({"enclosed", enclosed->card, enclosed->elementId, components});
        }
        if (definition.enclosedPolicy == EnclosedPolicy::Warn)
            warnedElements.push_back(std::string(enclosed->card) + " " + std::to_string(enclosed->elementId));
    }
    if (!warnedElements.empty())
    {
        std::string list;
        for (const std::string& element : warnedElements)
            list.append(list.empty() ? "" : ", ").append(element);
        warn(zoneProblem(
            definition, base::Severity::Warning, "zone-enclosed-outside",
            base::concatenate({"the elements ", list,
                               " are not in its list but have all their grid points in it (a stringer forgotten in "
                               "the list?): they are kept outside, and their grid points are contour points; add "
                               "them to the list, or give enclosed='include' (or 'exclude' to keep them outside "
                               "without this warning)"})));
    }

    // The constraints of every subcase on the grid points of the zone: those of some subcases only make contour points,
    // those of every subcase are held by the zone.
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    std::map<std::int32_t, ComponentMask> someSubcases;   // union over the subcases, by grid row
    std::map<std::int32_t, ComponentMask> everySubcase;   // intersection
    std::map<std::int32_t, std::int32_t> firstConstraint; // grid row → an SPC set that constrains it
    {
        std::map<std::pair<std::optional<std::int32_t>, std::optional<std::int32_t>>,
                 std::map<std::int32_t, ComponentMask>>
            bySets; // the constrained components per (SPC set, load set), each set made once
        bool isFirst = true;
        for (const ZoneSubcaseConstraint& subcase : subcases)
        {
            const auto key = std::make_pair(subcase.spcSet, subcase.loadSet);
            auto [found, isNew] = bySets.try_emplace(key);
            if (isNew)
            {
                base::DiagnosticSink ignored; // the analysis reports the problems of its constraints
                const assembly::SpcSet spc(constraints, snapshot.findLoads(), *m_dofMap, subcase.spcSet,
                                           subcase.loadSet, budget, ignored);
                for (const std::span<const base::DofIndex> dofs : {spc.sbDofs(), spc.sgDofs()})
                    for (const base::DofIndex dof : dofs)
                    {
                        const assembly::DofLabel label = m_dofMap->labelOf(dof);
                        if (label.kind != input::PointKind::Grid || zoneRows.count(label.row) == 0)
                            continue;
                        ComponentMask& mask = found->second[label.row];
                        mask = static_cast<ComponentMask>(mask | (1U << (label.component - 1)));
                    }
            }
            const std::map<std::int32_t, ComponentMask>& constrained = found->second;
            for (const auto& [gridRow, mask] : constrained)
            {
                ComponentMask& some = someSubcases[gridRow];
                if ((mask & ~some) != 0 && subcase.spcSet && firstConstraint.count(gridRow) == 0)
                    firstConstraint[gridRow] = *subcase.spcSet;
                some = static_cast<ComponentMask>(some | mask);
            }
            if (isFirst)
                everySubcase = constrained;
            else
                for (auto& [gridRow, mask] : everySubcase)
                {
                    const auto other = constrained.find(gridRow);
                    mask = static_cast<ComponentMask>(other == constrained.end() ? 0 : (mask & other->second));
                }
            isFirst = false;
        }
    }
    for (const auto& [gridRow, some] : someSubcases)
    {
        const auto every = everySubcase.find(gridRow);
        const ComponentMask held = every == everySubcase.end() ? 0 : every->second;
        const auto partial = static_cast<ComponentMask>(some & ~held);
        if (partial != 0)
        {
            ContourPointReport& entry = contour[gridRow];
            entry.grid = m_dofMap->gridId(gridRow);
            entry.components = static_cast<ComponentMask>(entry.components | partial);
            const auto spcSet = firstConstraint.find(gridRow);
            entry.causes.push_back(
                {"constraint", "SinglePointConstr", spcSet == firstConstraint.end() ? 0 : spcSet->second, partial});
        }
    }
    for (const auto& [gridRow, held] : everySubcase)
        if (held != 0 && contour.count(gridRow) == 0)
            report.constraints.push_back({m_dofMap->gridId(gridRow), held});

    // The constraint rows that act on the contour points: those of the zone (its rigid elements, its fasteners, the
    // MPC equations whose grid points are all in it) and those of the outside.
    std::map<std::int32_t, std::pair<std::vector<std::string>, std::vector<std::string>>> rowsAt; // zone, outside
    for (const auto& [elementId, entry] : rigidElements)
    {
        const bool isMember = memberRigid.count(elementId) != 0;
        for (const std::int32_t gridRow : entry.second)
            if (contour.count(gridRow) != 0)
                (isMember ? rowsAt[gridRow].first : rowsAt[gridRow].second)
                    .push_back(std::string(entry.first) + " " + std::to_string(elementId));
    }
    if (constraints != nullptr && constraints->mpcRowCount() > 0)
    {
        const std::span<const std::int32_t> sets = constraints->mpcSetIds();
        const std::span<const std::int32_t> equations = constraints->mpcEquations();
        const std::span<const std::int32_t> nodeIds = constraints->mpcNodeIds();
        std::map<std::pair<std::int32_t, std::int32_t>, std::vector<std::int32_t>> equationRows;
        for (std::size_t row = 0; row < sets.size(); ++row)
            if (const std::optional<std::int32_t> gridRow = points.findGrid(nodeIds[row]))
                equationRows[{sets[row], equations[row]}].push_back(*gridRow);
        for (const auto& [equation, gridRows] : equationRows)
        {
            const bool isInside =
                std::all_of(gridRows.begin(), gridRows.end(), [&](std::int32_t r) { return zoneRows.count(r) != 0; });
            for (const std::int32_t gridRow : gridRows)
                if (contour.count(gridRow) != 0)
                    (isInside ? rowsAt[gridRow].first : rowsAt[gridRow].second)
                        .push_back(std::string("MultiPointConstr") + " " + std::to_string(equation.first));
        }
    }
    if (snapshot.hasTable(input::TableRole::FastenerPiercings))
    {
        const input::FastenerPiercingTable& piercings = snapshot.fastenerPiercings();
        const base::ListView<std::int32_t> grids = piercings.grids();
        const std::span<const std::int32_t> ends = piercings.pointGrids();
        const base::ValidityView hasEnd = piercings.isPointGridGiven();
        for (std::int64_t row = 0; row < piercings.rowCount(); ++row)
        {
            const std::int32_t elementId = piercings.fastenerIds()[static_cast<std::size_t>(row)];
            const bool isMember = memberFasteners.count(elementId) != 0;
            std::vector<std::int32_t> ids(grids[row].begin(), grids[row].end());
            if (hasEnd.isValid(row))
                ids.push_back(ends[static_cast<std::size_t>(row)]);
            for (const std::int32_t id : ids)
                if (const std::optional<std::int32_t> gridRow = points.findGrid(id);
                    gridRow && contour.count(*gridRow) != 0)
                {
                    std::vector<std::string>& list = isMember ? rowsAt[*gridRow].first : rowsAt[*gridRow].second;
                    const std::string name = std::string(piercings.card(row)) + " " + std::to_string(elementId);
                    if (std::find(list.begin(), list.end(), name) == list.end())
                        list.push_back(name);
                }
        }
    }

    // The system of the data.
    const std::optional<input::CoordinateFrame> frame = frameOf(snapshot, definition.frameId);
    if (!frame)
        refuse(
            zoneProblem(definition, base::Severity::Error, "zone-frame",
                        base::concatenate({"its CID names the coordinate system ", std::to_string(definition.frameId),
                                           ", which the model does not have; add its coordinate system card or "
                                           "correct the ID"}),
                        definition.frameId));
    else if (frame->kind != input::CoordinateFrame::Kind::Rectangular)
        refuse(zoneProblem(
            definition, base::Severity::Error, "zone-frame",
            base::concatenate({"expected a rectangular CID (RectFrame1 or RectFrame2), found the cylindrical or "
                               "spherical system ",
                               std::to_string(definition.frameId), "; give the contour data in a rectangular system"}),
            definition.frameId));
    else
    {
        zone.axes = frame->axes;
        zone.origin = frame->origin;
    }
    const bool usesDisplacementSystems = definition.frameId == DISPLACEMENT_SYSTEMS;
    const auto pointOf = [&](std::int32_t gridRow)
    {
        ZonePoint point;
        point.gridRow = gridRow;
        point.grid = m_dofMap->gridId(gridRow);
        const base::Mat3 toBasic = m_dofMap->cdToBasic(gridRow);
        point.rotation = usesDisplacementSystems ? base::Mat3::identity() : zone.axes.transposed() * toBasic;
        point.arm = m_dofMap->positionInBasic(gridRow) - zone.origin;
        return point;
    };
    for (auto& [gridRow, entry] : contour)
    {
        ZonePoint point = pointOf(gridRow);
        point.components = entry.components;
        const auto acting = rowsAt.find(gridRow);
        if (acting != rowsAt.end() && !acting->second.first.empty())
        {
            if (!acting->second.second.empty())
                refuse(zoneProblem(
                    definition, base::Severity::Error, "zone-shared-constraint",
                    base::concatenate(
                        {"contour grid point ", std::to_string(entry.grid), " is a point of ",
                         acting->second.first.front(), " of the zone and of ", acting->second.second.front(),
                         " outside it: the MultiPointConstr forces there do not tell the force of one from the "
                         "other's; put both in the zone, or both outside"}),
                    entry.grid));
            point.takesConstraintForces = true;
        }
        report.contour.push_back(entry);
        const ComponentMask components = zoneComponents(entry.components, point.rotation);
        for (int c = 1; c <= 6; ++c)
            if (hasComponent(components, c))
                zone.contourDofs.push_back({point.grid, gridRow, c});
        zone.contour.push_back(point);
    }
    for (const std::int32_t gridRow : zoneRows)
    {
        if (contour.count(gridRow) != 0)
            continue;
        const ZonePoint point = pointOf(gridRow);
        report.interior.push_back(point.grid);
        for (int c = 1; c <= 6; ++c)
            zone.interiorDofs.push_back({point.grid, gridRow, c});
        zone.interior.push_back(point);
    }
    std::sort(report.interior.begin(), report.interior.end());
    zone.isValid = isValid;
}

std::vector<double> ZoneContour::elementForces(std::size_t z, const assembly::ElementSet& set,
                                               std::span<const double> displacementsG,
                                               const elements::ElementTemperatures* temperatures) const
{
    const auto gSize = static_cast<std::size_t>(m_dofMap->gSetSize());
    if (displacementsG.size() != gSize)
        throw std::invalid_argument(
            base::concatenate({"ZoneContour::elementForces: expected ", std::to_string(gSize), " displacements, found ",
                               std::to_string(displacementsG.size())}));
    const Zone& zone = m_zones[z];
    std::map<std::int32_t, std::size_t> contourIndex;
    for (std::size_t c = 0; c < zone.contour.size(); ++c)
        contourIndex[zone.contour[c].gridRow] = c;
    std::vector<double> forces(6 * zone.contour.size(), 0.0);
    elements::ElementMatrixBlock stiffness;
    elements::ElementVectorBlock thermal;
    std::vector<double> force; // bounded by the DOFs of one element
    for (const Member& member : zone.members)
    {
        const assembly::AssemblyBatch& batch = set.batches()[member.batch];
        const elements::ElementRange range{member.row, member.row + 1};
        stiffness.clear();
        batch.family->computeStiffness(batch.view, range, stiffness);
        if (stiffness.empty())
            continue;
        const std::span<const elements::ElementDof> dofs = stiffness.dofs(0);
        const std::span<const double> k = stiffness.matrix(0);
        const std::size_t n = dofs.size();
        bool touchesContour = false;
        for (const elements::ElementDof& dof : dofs)
            touchesContour = touchesContour || (dof.kind == input::PointKind::Grid && contourIndex.count(dof.row) != 0);
        if (!touchesContour)
            continue;
        // f = K_e·u_e − f_e,T on the DOFs of the element (StiffnessRecovery's grid point forces, opposite sign).
        force.assign(n, 0.0);
        for (std::size_t r = 0; r < n; ++r)
        {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                sum += k[r * n + j] * displacementsG[static_cast<std::size_t>(m_dofMap->dofOf(dofs[j]))];
            force[r] = sum;
        }
        if (temperatures != nullptr && !batch.family->isThermallyInert())
        {
            thermal.clear();
            batch.family->computeThermalLoads(batch.view, range, *temperatures, thermal);
            for (std::size_t t = 0; t < thermal.size(); ++t)
            {
                const std::span<const elements::ElementDof> thermalDofs = thermal.dofs(t);
                const std::span<const double> values = thermal.values(t);
                for (std::size_t i = 0; i < thermalDofs.size(); ++i)
                    for (std::size_t r = 0; r < n; ++r)
                        if (dofs[r] == thermalDofs[i])
                        {
                            force[r] -= values[i];
                            break;
                        }
            }
        }
        for (std::size_t r = 0; r < n; ++r)
        {
            if (dofs[r].kind != input::PointKind::Grid)
                continue;
            const auto found = contourIndex.find(dofs[r].row);
            if (found != contourIndex.end())
                forces[6 * found->second + static_cast<std::size_t>(dofs[r].component - 1)] += force[r];
        }
    }
    return forces;
}

std::vector<double> ZoneContour::toContourDofs(std::size_t z, std::span<const double> perPoint) const
{
    const Zone& zone = m_zones[z];
    std::map<std::int32_t, std::size_t> contourIndex;
    for (std::size_t c = 0; c < zone.contour.size(); ++c)
        contourIndex[zone.contour[c].gridRow] = c;
    std::vector<double> values;
    values.reserve(zone.contourDofs.size());
    for (const ZoneDof& dof : zone.contourDofs)
    {
        const std::size_t p = contourIndex.at(dof.gridRow);
        const ZonePoint& point = zone.contour[p];
        const auto c = static_cast<std::size_t>(dof.component - 1);
        const std::size_t kind = c < 3 ? 0 : 3;
        double value = 0.0;
        for (std::size_t k = 0; k < 3; ++k)
            value += point.rotation(c - kind, k) * perPoint[6 * p + kind + k];
        values.push_back(value);
    }
    return values;
}

std::array<double, 6> ZoneContour::contourResultant(std::size_t z, std::span<const double> perPoint) const
{
    const Zone& zone = m_zones[z];
    std::array<double, 6> resultant{};
    for (std::size_t p = 0; p < zone.contour.size(); ++p)
    {
        const ZonePoint& point = zone.contour[p];
        const base::Mat3 toBasic = m_dofMap->cdToBasic(point.gridRow);
        for (std::size_t r = 0; r < 6; ++r)
            for (std::size_t k = 0; k < 6; ++k)
                resultant[r] += resultantWeight(point, r, k, toBasic, zone.axes) * perPoint[6 * p + k];
    }
    return resultant;
}

ZoneContour::ThermalPart ZoneContour::thermalPart(std::size_t z,
                                                  const elements::ElementTemperatures& temperatures) const
{
    const std::vector<double> perPoint = elementForces(z, *m_set, m_zeros.span(), &temperatures);
    ThermalPart part;
    part.forces = toContourDofs(z, perPoint);
    const std::array<double, 6> resultant = contourResultant(z, perPoint);
    std::copy(resultant.begin(), resultant.end(), part.equilibrium.begin());
    return part;
}

const ZoneContour::ZoneRows& ZoneContour::rows(std::size_t z, base::MemoryBudget& budget)
{
    if (!m_rows[z])
        m_rows[z] = buildRows(m_zones[z], budget);
    return *m_rows[z];
}

std::unique_ptr<ZoneContour::ZoneRows> ZoneContour::buildRows(const Zone& zone, base::MemoryBudget& budget) const
{
    auto rows = std::make_unique<ZoneRows>();
    if (!zone.isValid)
        return rows;
    std::map<std::int32_t, std::size_t> contourIndex;
    for (std::size_t c = 0; c < zone.contour.size(); ++c)
        contourIndex[zone.contour[c].gridRow] = c;

    // The element part of the forces per contour point and component of CD: Σ_{e∈zone} K_e(row, :), merged in element
    // order. The entries before merging are at most the DOFs of every member per row, reserved first.
    elements::ElementMatrixBlock stiffness;
    std::uint64_t entries = 0;
    for (const Member& member : zone.members)
    {
        stiffness.clear();
        const assembly::AssemblyBatch& batch = m_set->batches()[member.batch];
        batch.family->computeStiffness(batch.view, {member.row, member.row + 1}, stiffness);
        if (!stiffness.empty())
            entries += static_cast<std::uint64_t>(stiffness.dofs(0).size()) * stiffness.dofs(0).size();
    }
    const base::MemoryReservation charge =
        budget.reserve(entries * 2 * sizeof(std::pair<base::DofIndex, double>), "zone contour rows before merging");
    std::vector<RowEntries> raw(6 * zone.contour.size());
    for (const Member& member : zone.members)
    {
        stiffness.clear();
        const assembly::AssemblyBatch& batch = m_set->batches()[member.batch];
        batch.family->computeStiffness(batch.view, {member.row, member.row + 1}, stiffness);
        if (stiffness.empty())
            continue;
        const std::span<const elements::ElementDof> dofs = stiffness.dofs(0);
        const std::span<const double> k = stiffness.matrix(0);
        const std::size_t n = dofs.size();
        for (std::size_t r = 0; r < n; ++r)
        {
            if (dofs[r].kind != input::PointKind::Grid)
                continue;
            const auto found = contourIndex.find(dofs[r].row);
            if (found == contourIndex.end())
                continue;
            RowEntries& row = raw[6 * found->second + static_cast<std::size_t>(dofs[r].component - 1)];
            for (std::size_t j = 0; j < n; ++j)
                if (k[r * n + j] != 0.0)
                    row.emplace_back(m_dofMap->dofOf(dofs[j]), k[r * n + j]);
        }
    }
    for (RowEntries& row : raw)
        mergeRow(row);
    const auto unit = [&](std::int32_t gridRow, std::size_t k)
    { return RowEntries{{m_dofMap->gridDof(gridRow, static_cast<int>(k) + 1), 1.0}}; };

    std::vector<RowEntries> displacements;
    std::vector<RowEntries> forces;
    std::vector<RowEntries> forceConstraints;
    for (const ZoneDof& dof : zone.contourDofs)
    {
        const std::size_t p = contourIndex.at(dof.gridRow);
        const ZonePoint& point = zone.contour[p];
        const auto c = static_cast<std::size_t>(dof.component - 1);
        const std::size_t kind = c < 3 ? 0 : 3;
        RowEntries displacement;
        RowEntries force;
        RowEntries constraint;
        for (std::size_t k = 0; k < 3; ++k)
        {
            const double t = point.rotation(c - kind, k);
            addScaled(displacement, t, unit(point.gridRow, kind + k));
            addScaled(force, t, raw[6 * p + kind + k]);
            if (point.takesConstraintForces)
                addScaled(constraint, -t, unit(point.gridRow, kind + k));
        }
        mergeRow(displacement);
        mergeRow(force);
        mergeRow(constraint);
        displacements.push_back(std::move(displacement));
        forces.push_back(std::move(force));
        forceConstraints.push_back(std::move(constraint));
    }
    std::vector<RowEntries> loads;
    for (const ZoneDof& dof : zone.interiorDofs)
    {
        const auto found = std::find_if(zone.interior.begin(), zone.interior.end(),
                                        [&](const ZonePoint& point) { return point.gridRow == dof.gridRow; });
        const auto c = static_cast<std::size_t>(dof.component - 1);
        const std::size_t kind = c < 3 ? 0 : 3;
        RowEntries load;
        for (std::size_t k = 0; k < 3; ++k)
            addScaled(load, found->rotation(c - kind, k), unit(dof.gridRow, kind + k));
        mergeRow(load);
        loads.push_back(std::move(load));
    }

    // The equilibrium: the contour resultant (rows 0 to 5) over u and the MPC forces, the interior one (rows 6 to 11)
    // over the applied loads and the SPC forces.
    std::vector<RowEntries> equilibriumU(EQUILIBRIUM_ROWS);
    std::vector<RowEntries> equilibriumQ(EQUILIBRIUM_ROWS);
    std::vector<RowEntries> equilibriumP(EQUILIBRIUM_ROWS);
    std::vector<RowEntries> equilibriumR(EQUILIBRIUM_ROWS);
    for (std::size_t p = 0; p < zone.contour.size(); ++p)
    {
        const ZonePoint& point = zone.contour[p];
        const base::Mat3 toBasic = m_dofMap->cdToBasic(point.gridRow);
        for (std::size_t r = 0; r < 6; ++r)
            for (std::size_t k = 0; k < 6; ++k)
            {
                const double w = resultantWeight(point, r, k, toBasic, zone.axes);
                addScaled(equilibriumU[r], w, raw[6 * p + k]);
                if (point.takesConstraintForces)
                    addScaled(equilibriumQ[r], -w, unit(point.gridRow, k));
            }
    }
    for (const ZonePoint& point : zone.interior)
    {
        const base::Mat3 toBasic = m_dofMap->cdToBasic(point.gridRow);
        for (std::size_t r = 0; r < 6; ++r)
            for (std::size_t k = 0; k < 6; ++k)
            {
                const double w = resultantWeight(point, r, k, toBasic, zone.axes);
                addScaled(equilibriumP[6 + r], w, unit(point.gridRow, k));
                addScaled(equilibriumR[6 + r], w, unit(point.gridRow, k));
            }
    }
    for (std::vector<RowEntries>* set : {&equilibriumU, &equilibriumQ, &equilibriumP, &equilibriumR})
        for (RowEntries& row : *set)
            mergeRow(row);

    rows->displacements = SparseRows(displacements, budget);
    rows->forces = SparseRows(forces, budget);
    rows->forceConstraints = SparseRows(forceConstraints, budget);
    rows->loads = SparseRows(loads, budget);
    rows->equilibriumDisplacements = SparseRows(equilibriumU, budget);
    rows->equilibriumConstraints = SparseRows(equilibriumQ, budget);
    rows->equilibriumLoads = SparseRows(equilibriumP, budget);
    rows->equilibriumReactions = SparseRows(equilibriumR, budget);
    return rows;
}

} // namespace oos::output
