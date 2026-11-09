#include "oos/solutions/sol101/ExternalUnitPath.hpp"

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/relief/InertiaRelief.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/SolverError.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/output/sections/ZoneContour.hpp"
#include "oos/reduction/external/ExternalLoadSource.hpp"
#include "oos/solutions/sol101/StaticLoadCases.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::solutions
{

namespace
{

base::Diagnostic unitProblem(const reduction::ExternalUnit& unit, std::int32_t zoneId, base::Severity severity,
                             std::string code, std::string message)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = severity;
    diagnostic.code = std::move(code);
    diagnostic.card = "ZONE";
    diagnostic.id = zoneId;
    diagnostic.message = base::concatenate({"external unit of ", unit.displayName(), ": ", message});
    return diagnostic;
}

// The first entries of a list of IDs as text ("12, 13, 14 and 2 more").
std::string idList(const std::vector<std::int32_t>& ids)
{
    constexpr std::size_t NAMED = 5;
    std::string text;
    for (std::size_t k = 0; k < ids.size() && k < NAMED; ++k)
        text.append(k == 0 ? "" : ", ").append(std::to_string(ids[k]));
    if (ids.size() > NAMED)
        text.append(" and ").append(std::to_string(ids.size() - NAMED)).append(" more");
    return text;
}

} // namespace

ExternalUnitPath::ExternalUnitPath(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink)
    : m_dofMap(inputs.dofMap), m_subcases(inputs.subcases.begin(), inputs.subcases.end())
{
    if (inputs.snapshot == nullptr || inputs.elements == nullptr || inputs.dofMap == nullptr)
        throw std::invalid_argument("ExternalUnitPath: expected the snapshot, the element set and the DOF map");
    for (const ExternalUnitRequest& request : inputs.requests)
        if (!request.unit)
            throw std::invalid_argument(
                base::concatenate({"ExternalUnitPath: the external unit of zone ", request.zone.name,
                                   " has no matrices; give its stiffness (and its loads) to add_external_unit"}));
    resolve(inputs, budget, sink);
}

void ExternalUnitPath::resolve(const Inputs& inputs, base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    const input::ModelSnapshot& snapshot = *inputs.snapshot;
    const assembly::DofMap& dofMap = *inputs.dofMap;

    // The zones, their contour DOFs in the displacement systems of the grid points (the DOF labels of O_o_Solver: grid
    // point and component, in the displacement systems of the grid points).
    std::vector<output::ZoneDefinition> definitions; // bounded by the plugins
    for (const ExternalUnitRequest& request : inputs.requests)
    {
        output::ZoneDefinition definition = request.zone;
        definition.frameId = output::DISPLACEMENT_SYSTEMS;
        definitions.push_back(std::move(definition));
    }
    std::vector<output::ZoneSubcaseConstraint> constraints; // bounded by the subcases
    std::vector<std::int32_t> thermalSubcases;
    for (const StaticSubcase& subcase : m_subcases)
        if (!subcase.isCombination)
        {
            constraints.push_back({subcase.label, subcase.spc, subcase.load});
            if (subcase.hasInitialStrains())
                thermalSubcases.push_back(subcase.label);
        }
    output::ZoneContour contour(definitions, constraints, snapshot, *inputs.elements, dofMap, budget, sink);

    // The things whose constraint rows would stay in the analysis while their stiffness is the plugin's.
    std::set<std::int32_t> rigidIds;
    if (snapshot.hasTable(input::TableRole::RigidElements))
        for (const std::int32_t elementId : snapshot.rigidElements().ids())
            rigidIds.insert(elementId);
    std::map<std::pair<std::int32_t, std::int32_t>, std::vector<std::int32_t>> mpcGrids; // equation → its grid IDs
    if (snapshot.hasTable(input::TableRole::Spc) && snapshot.constraints().mpcRowCount() > 0)
    {
        const input::ConstraintTable& table = snapshot.constraints();
        const std::span<const std::int32_t> sets = table.mpcSetIds();
        const std::span<const std::int32_t> equations = table.mpcEquations();
        const std::span<const std::int32_t> nodeIds = table.mpcNodeIds();
        for (std::size_t row = 0; row < sets.size(); ++row)
            mpcGrids[{sets[row], equations[row]}].push_back(nodeIds[row]);
    }
    std::set<base::DofIndex> supportDofs;
    for (const StaticSubcase& subcase : m_subcases)
        if (!subcase.isCombination)
        {
            base::DiagnosticSink ignored; // the analysis reports the problems of its supports
            for (const base::DofIndex dof : assembly::supportDofsOf(snapshot, dofMap, subcase.tableId, ignored))
                supportDofs.insert(dof);
        }

    std::map<base::EntityId, std::int32_t> zoneOfElement; // replaced element → the zone that holds it
    std::map<std::int32_t, std::int32_t> zoneOfInterior;  // interior grid row → zone
    for (std::size_t z = 0; z < inputs.requests.size(); ++z)
    {
        const ExternalUnitRequest& request = inputs.requests[z];
        const reduction::ExternalUnit& unit = *request.unit;
        const output::ZoneContour::Zone& zone = contour.zones()[z];
        const std::int32_t zoneId = request.zone.id;
        const auto refuse = [&](std::string code, std::string message)
        { sink.report(unitProblem(unit, zoneId, base::Severity::Error, std::move(code), std::move(message))); };
        if (!zone.isValid)
            continue; // ZoneContour named the problem
        if (!thermalSubcases.empty())
            refuse("external-unit-thermal",
                   base::concatenate({"the subcases ", idList(thermalSubcases),
                                      " have a thermal or deformation load (TemperatureSelection(LOAD), "
                                      "DeformationLoadSelection), whose initial "
                                      "strains in the zone are not given to the plugin in this version; run those "
                                      "subcases without the external unit"}));
        std::vector<std::int32_t> rigidMembers;
        std::vector<std::int32_t> fastenerMembers;
        for (const std::int32_t elementId : zone.report.elements)
            if (rigidIds.count(elementId) != 0)
                rigidMembers.push_back(elementId);
        for (const output::ZoneContour::Member& member : zone.members)
        {
            const std::string_view card = inputs.elements->batches()[member.batch].view.card();
            if (card == "Fastener" || card == "Weld")
                fastenerMembers.push_back(member.elementId);
        }
        if (!rigidMembers.empty())
            refuse("external-unit-rigid",
                   base::concatenate({"the rigid elements ", idList(rigidMembers),
                                      " are inside the zone: their constraint rows would stay in the analysis while "
                                      "the plugin's K* holds their stiffness; leave them outside the zone (or model "
                                      "them in the plugin and remove them from the deck)"}));
        if (!fastenerMembers.empty())
            refuse("external-unit-fastener",
                   base::concatenate({"the fasteners ", idList(fastenerMembers),
                                      " are inside the zone: the constraints of their piercing points would stay in "
                                      "the analysis; leave them outside the zone"}));
        std::set<std::int32_t> zoneGrids; // the grid IDs of the zone
        for (const output::ZoneContour::ZonePoint& point : zone.contour)
            zoneGrids.insert(point.grid);
        for (const output::ZoneContour::ZonePoint& point : zone.interior)
            zoneGrids.insert(point.grid);
        std::set<std::int32_t> insideSets; // the MPC sets of the equations inside
        for (const auto& [equation, grids] : mpcGrids)
            if (std::all_of(grids.begin(), grids.end(), [&](std::int32_t g) { return zoneGrids.count(g) != 0; }))
                insideSets.insert(equation.first);
        if (!insideSets.empty())
            refuse("external-unit-mpc",
                   base::concatenate({"MultiPointConstr equations of the sets ",
                                      idList(std::vector<std::int32_t>(insideSets.begin(), insideSets.end())),
                                      " have all their grid points in the zone: their rows would stay in the analysis; "
                                      "move them into the plugin"}));

        reduction::ExternalZone external;
        external.id = zoneId;
        external.name = request.zone.name;
        external.replacedElements.assign(zone.report.elements.begin(), zone.report.elements.end());
        std::vector<output::ZoneContour::ZonePoint> contourPoints = zone.contour;
        std::sort(contourPoints.begin(), contourPoints.end(),
                  [](const auto& a, const auto& b) { return a.grid < b.grid; });
        for (const output::ZoneContour::ZonePoint& point : contourPoints)
            for (int c = 1; c <= 6; ++c)
                if (output::hasComponent(point.components, c))
                {
                    external.contourLabels.push_back({reduction::UnitPointKind::Grid, point.grid,
                                                      reduction::FastenerSide::None, static_cast<std::uint8_t>(c)});
                    external.contourDofs.push_back(dofMap.gridDof(point.gridRow, c));
                }
        std::vector<output::ZoneContour::ZonePoint> interiorPoints = zone.interior;
        std::sort(interiorPoints.begin(), interiorPoints.end(),
                  [](const auto& a, const auto& b) { return a.grid < b.grid; });
        std::vector<std::int32_t> supportedInside;
        for (const output::ZoneContour::ZonePoint& point : interiorPoints)
        {
            for (int c = 1; c <= 6; ++c)
            {
                external.interiorLabels.push_back({reduction::UnitPointKind::Grid, point.grid,
                                                   reduction::FastenerSide::None, static_cast<std::uint8_t>(c)});
                external.interiorDofs.push_back(dofMap.gridDof(point.gridRow, c));
                if (supportDofs.count(dofMap.gridDof(point.gridRow, c)) != 0 &&
                    (supportedInside.empty() || supportedInside.back() != point.grid))
                    supportedInside.push_back(point.grid);
            }
            const auto [found, isNew] = zoneOfInterior.emplace(point.gridRow, zoneId);
            if (!isNew)
                refuse("external-unit-overlap",
                       base::concatenate({"its interior grid point ", std::to_string(point.grid),
                                          " is inside the zone of another external unit (zone ",
                                          std::to_string(found->second), "); give each unit its own zone"}));
        }
        if (!supportedInside.empty())
            refuse("external-unit-free-body-support",
                   base::concatenate({"the FreeBodySupport points ", idList(supportedInside),
                                      " are interior points of the zone; put the supports outside the zone"}));
        external.holdsConstraints = !zone.report.constraints.empty();
        for (const base::EntityId elementId : external.replacedElements)
        {
            const auto [found, isNew] = zoneOfElement.emplace(elementId, zoneId);
            if (!isNew)
                refuse("external-unit-overlap",
                       base::concatenate({"element ", std::to_string(elementId),
                                          " is in the zone of another external unit "
                                          "(zone ",
                                          std::to_string(found->second), "); give each unit its own zone"}));
        }

        const reduction::ExternalUnitChecks checks(unit, external, dofMap, budget, sink);
        Unit placed;
        placed.zoneId = zoneId;
        placed.report = zone.report;
        placed.checks = checks.report();
        if (checks.isAccepted())
            placed.placement = std::make_shared<reduction::ExternalUnitAssembly>(request.unit, external, budget);
        m_units.push_back(std::move(placed));
        m_replaced.insert(m_replaced.end(), external.replacedElements.begin(), external.replacedElements.end());
        m_heldDofs.insert(m_heldDofs.end(), external.interiorDofs.begin(), external.interiorDofs.end());
        for (const output::ZoneContour::ZonePoint& point : interiorPoints)
            m_interiorNodes.push_back(point.gridRow);
    }
    // A contour point of one unit inside another.
    for (const Unit& unit : m_units)
        if (unit.placement)
            for (const base::DofIndex dof : unit.placement->zone().contourDofs)
                if (const auto found = zoneOfInterior.find(dofMap.labelOf(dof).row); found != zoneOfInterior.end())
                {
                    sink.report(unitProblem(
                        unit.placement->unit(), unit.placement->zone().id, base::Severity::Error,
                        "external-unit-overlap",
                        base::concatenate({"its contour grid point ", std::to_string(dofMap.pointIdOf(dof)),
                                           " is an interior point of the zone ", std::to_string(found->second),
                                           " of another external unit; give the units zones that "
                                           "do not overlap"})));
                    break;
                }
    std::sort(m_replaced.begin(), m_replaced.end());
    m_replaced.erase(std::unique(m_replaced.begin(), m_replaced.end()), m_replaced.end());
    std::sort(m_heldDofs.begin(), m_heldDofs.end());
    m_heldDofs.erase(std::unique(m_heldDofs.begin(), m_heldDofs.end()), m_heldDofs.end());
    std::sort(m_interiorNodes.begin(), m_interiorNodes.end());
    m_interiorNodes.erase(std::unique(m_interiorNodes.begin(), m_interiorNodes.end()), m_interiorNodes.end());
    for (const Unit& unit : m_units)
    {
        if (!unit.placement)
            continue;
        const std::vector<std::int32_t> nodes = unit.placement->contourNodes(dofMap);
        m_groupNodes.insert(m_groupNodes.end(), nodes.begin(), nodes.end());
        m_groupOffsets.push_back(static_cast<std::int64_t>(m_groupNodes.size()));
    }
    for (Unit& unit : m_units)
        if (unit.placement)
        {
            const std::size_t n = unit.placement->contourSize();
            unit.caseLoads = linalg::DenseMatrix(budget, n, m_subcases.size(), "external unit case loads");
            unit.caseDisplacements =
                linalg::DenseMatrix(budget, n, m_subcases.size(), "external unit case displacements");
            unit.caseForces = linalg::DenseMatrix(budget, n, m_subcases.size(), "external unit case forces");
        }
}

void ExternalUnitPath::addStiffness(linalg::BlockSparseMatrix& kgg) const
{
    for (const Unit& unit : m_units)
        if (unit.placement)
            unit.placement->addStiffnessTo(kgg, *m_dofMap);
}

void ExternalUnitPath::addMass(linalg::BlockSparseMatrix& mgg) const
{
    for (const Unit& unit : m_units)
        if (unit.placement)
            unit.placement->addMassTo(mgg, *m_dofMap);
}

void ExternalUnitPath::checkMass(bool isInertial, bool isGridPointWeight, const std::string& reason,
                                 base::DiagnosticSink& sink) const
{
    for (const Unit& unit : m_units)
    {
        if (!unit.placement || unit.placement->unit().hasMass())
            continue;
        const reduction::ExternalUnit& external = unit.placement->unit();
        if (isInertial)
            sink.report(unitProblem(external, unit.placement->zone().id, base::Severity::Error, "external-unit-mass",
                                    base::concatenate({"the analysis uses the mass (", reason,
                                                       ") and the unit gives no reduced mass M*; give mass= (the mass "
                                                       "of the local model condensed on the contour DOFs)"})));
        else if (isGridPointWeight)
            sink.report(unitProblem(
                external, unit.placement->zone().id, base::Severity::Warning, "external-unit-mass",
                "PARAM WeightReferenceNode: the unit gives no reduced mass M*, so the mass of its zone is not in "
                "the grid point weight"));
    }
}

void ExternalUnitPath::condenseLoads(std::int32_t constraintSet, std::span<const std::size_t> cases,
                                     const StaticLoadCases& loadCases, base::MemoryBudget& budget,
                                     base::DiagnosticSink& sink)
{
    const std::size_t k = loadCases.basisSize();
    const auto gSize = static_cast<std::size_t>(m_dofMap->gSetSize());
    std::vector<linalg::DenseMatrix> interior; // per unit, interior × k
    for (const Unit& unit : m_units)
        interior.emplace_back(budget, unit.placement ? unit.placement->interiorSize() : 0, k,
                              "external unit interior loads");
    {
        base::BudgetedBuffer<double> loads(budget, gSize, "external unit basis loads");
        base::BudgetedBuffer<double> enforced(budget, gSize, "external unit basis enforced displacements");
        std::vector<double> values; // bounded by the largest zone
        for (std::size_t b = 0; b < k; ++b)
        {
            loadCases.basisVector(b, loads.span(), enforced.span(), budget, sink);
            for (std::size_t u = 0; u < m_units.size(); ++u)
            {
                if (!m_units[u].placement)
                    continue;
                values.assign(m_units[u].placement->interiorSize(), 0.0);
                m_units[u].placement->gatherInterior(loads.span(), values);
                for (std::size_t i = 0; i < values.size(); ++i)
                    interior[u](i, b) = values[i];
            }
        }
    }
    condenseLoads(constraintSet, cases, std::move(interior), loadCases.coefficients(), budget, sink);
}

void ExternalUnitPath::condenseLoads(std::int32_t constraintSet, std::span<const std::size_t> cases,
                                     std::vector<linalg::DenseMatrix> interior, const linalg::DenseMatrix& alpha,
                                     base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    if (interior.size() != m_units.size())
        throw std::invalid_argument(
            base::concatenate({"ExternalUnitPath::condenseLoads: expected the interior loads of ",
                               std::to_string(m_units.size()), " units, found ", std::to_string(interior.size())}));
    const std::size_t k = alpha.rows();
    std::vector<base::EntityId> labels; // the solved subcases of the set
    for (const std::size_t s : cases)
        labels.push_back(m_subcases[s].label);
    for (std::size_t u = 0; u < m_units.size(); ++u)
    {
        Unit& unit = m_units[u];
        if (!unit.placement)
            continue;
        const reduction::ExternalUnitAssembly& placement = *unit.placement;
        const std::size_t n = placement.contourSize();
        unit.currentLoads = linalg::DenseMatrix(budget, n, k, "external unit condensed loads");
        bool isLoaded = false;
        for (std::size_t b = 0; b < k && !isLoaded; ++b)
            for (std::size_t i = 0; i < placement.interiorSize() && !isLoaded; ++i)
                isLoaded = interior[u](i, b) != 0.0;
        if (isLoaded)
        {
            const reduction::ExternalLoadSource* source = placement.unit().loads();
            if (source == nullptr)
            {
                sink.report(unitProblem(
                    placement.unit(), placement.zone().id, base::Severity::Error, "external-unit-loads",
                    base::concatenate({"loads apply inside the zone (subcases ", idList(labels),
                                       ") and the unit gives no condensed loads; give loads= (the condensed loads "
                                       "of the zone's loads per basis vector)"})));
                continue;
            }
            linalg::DenseMatrix condensed(budget, placement.unit().size(), k, "external unit f*");
            source->condensedLoads(placement.unit(), interior[u].view(), condensed.view());
            const std::span<const std::size_t> rows = placement.unitRowOfContour();
            for (std::size_t b = 0; b < k; ++b)
                for (std::size_t a = 0; a < n; ++a)
                {
                    const double value = condensed(rows[a], b);
                    if (!std::isfinite(value))
                        throw base::SolverError(base::concatenate(
                            {"external unit of ", placement.zone().displayName(), ": its condensed loads hold ",
                             std::to_string(value), " (basis vector ", std::to_string(b),
                             "); expected finite values from its load source: check the condensed "
                             "loads its plugin returns"}));
                    unit.currentLoads(a, b) = value;
                }
        }
        for (std::size_t j = 0; j < cases.size(); ++j)
            for (std::size_t a = 0; a < n; ++a)
            {
                double sum = 0.0;
                for (std::size_t b = 0; b < k; ++b)
                    sum += unit.currentLoads(a, b) * alpha(b, j);
                unit.caseLoads(a, cases[j]) = sum;
            }
        ExternalUnitBasis basis;
        basis.constraintSet = constraintSet;
        basis.cases = labels;
        basis.interiorLoads = std::move(interior[u]);
        basis.condensedLoads = linalg::DenseMatrix(budget, n, k, "external unit basis f*");
        for (std::size_t b = 0; b < k; ++b)
            for (std::size_t a = 0; a < n; ++a)
                basis.condensedLoads(a, b) = unit.currentLoads(a, b);
        basis.coefficients = linalg::DenseMatrix(budget, alpha.rows(), alpha.columns(), "external unit basis alpha");
        for (std::size_t c = 0; c < alpha.columns(); ++c)
            for (std::size_t b = 0; b < alpha.rows(); ++b)
                basis.coefficients(b, c) = alpha(b, c);
        unit.bases.push_back(std::move(basis));
    }
}

void ExternalUnitPath::applyToBasisVector(std::size_t b, std::span<double> loadsG) const
{
    for (const Unit& unit : m_units)
    {
        if (!unit.placement)
            continue;
        for (const base::DofIndex dof : unit.placement->zone().interiorDofs)
            loadsG[static_cast<std::size_t>(dof)] = 0.0;
        const std::vector<base::DofIndex>& contour = unit.placement->zone().contourDofs;
        for (std::size_t a = 0; a < contour.size(); ++a)
            loadsG[static_cast<std::size_t>(contour[a])] += unit.currentLoads(a, b);
    }
}

void ExternalUnitPath::applyToExactCase(std::span<double> loadsG, base::MemoryBudget& budget) const
{
    for (const Unit& unit : m_units)
    {
        if (!unit.placement)
            continue;
        const reduction::ExternalUnitAssembly& placement = *unit.placement;
        linalg::DenseMatrix interior(budget, placement.interiorSize(), 1, "external unit exact interior loads");
        bool isLoaded = false;
        for (std::size_t i = 0; i < placement.interiorSize(); ++i)
        {
            interior(i, 0) = loadsG[static_cast<std::size_t>(placement.zone().interiorDofs[i])];
            isLoaded = isLoaded || interior(i, 0) != 0.0;
            loadsG[static_cast<std::size_t>(placement.zone().interiorDofs[i])] = 0.0;
        }
        if (!isLoaded || placement.unit().loads() == nullptr)
            continue; // a loaded zone without a source was refused with its basis
        linalg::DenseMatrix condensed(budget, placement.unit().size(), 1, "external unit exact f*");
        placement.unit().loads()->condensedLoads(placement.unit(), interior.view(), condensed.view());
        const std::span<const std::size_t> rows = placement.unitRowOfContour();
        for (std::size_t a = 0; a < placement.contourSize(); ++a)
            loadsG[static_cast<std::size_t>(placement.zone().contourDofs[a])] += condensed(rows[a], 0);
    }
}

void ExternalUnitPath::combineCase(std::size_t subcase, std::span<const std::size_t> terms,
                                   std::span<const double> coefficients)
{
    for (Unit& unit : m_units)
    {
        if (!unit.placement)
            continue;
        for (std::size_t a = 0; a < unit.placement->contourSize(); ++a)
        {
            double sum = 0.0;
            for (std::size_t t = 0; t < terms.size(); ++t)
                sum += coefficients[t] * unit.caseLoads(a, terms[t]);
            unit.caseLoads(a, subcase) = sum;
        }
    }
}

std::vector<output::StiffnessRecovery::UnitRow> ExternalUnitPath::recordCase(std::size_t subcase,
                                                                             std::span<const double> displacements,
                                                                             std::array<double, 6>& reactions,
                                                                             base::MemoryBudget& /*budget*/)
{
    std::vector<output::StiffnessRecovery::UnitRow> rows; // bounded by the contours
    for (Unit& unit : m_units)
    {
        if (!unit.placement)
            continue;
        const reduction::ExternalUnitAssembly& placement = *unit.placement;
        const std::size_t n = placement.contourSize();
        std::vector<double> contourDisplacements(n); // bounded by the contour
        placement.gatherContour(displacements, contourDisplacements);
        std::vector<double> stiffnessForces(n);
        reduction::ExternalUnitAssembly::multiply(placement.stiffness(), contourDisplacements, stiffnessForces);
        for (std::size_t a = 0; a < n; ++a)
        {
            unit.caseDisplacements(a, subcase) = contourDisplacements[a];
            unit.caseForces(a, subcase) = stiffnessForces[a] - unit.caseLoads(a, subcase);
        }
        unit.recordedCases.push_back(subcase);
        // The GPFORCE row of each contour point: −K*·u_a (the applied loads there hold f*), in its CD.
        std::map<std::int32_t, std::array<double, 6>> ofNode; // bounded by the contour points
        for (std::size_t a = 0; a < n; ++a)
        {
            const linalg::NodeDof nodeDof = m_dofMap->nodeDofOf(placement.zone().contourDofs[a]);
            ofNode[nodeDof.node][static_cast<std::size_t>(nodeDof.component)] -= stiffnessForces[a];
        }
        for (const auto& [node, row] : ofNode)
            rows.push_back({node, placement.zone().id, row});
        const std::array<double, 6> resultant = placement.resultantOf(stiffnessForces, *m_dofMap);
        for (std::size_t c = 0; c < resultant.size(); ++c)
            reactions[c] -= resultant[c];
    }
    std::stable_sort(rows.begin(), rows.end(),
                     [](const output::StiffnessRecovery::UnitRow& a, const output::StiffnessRecovery::UnitRow& b)
                     { return a.node < b.node; });
    return rows;
}

output::OutputPlan ExternalUnitPath::restrictPlan(const output::OutputPlan& plan,
                                                  const assembly::ElementSet& elements) const
{
    output::OutputPlan restricted = plan;
    std::vector<std::int32_t> knownPoints; // bounded by the model's points
    for (std::int32_t node = 0; node < m_dofMap->nodeCount(); ++node)
    {
        if (std::binary_search(m_interiorNodes.begin(), m_interiorNodes.end(), node))
            continue;
        knownPoints.push_back(node < m_dofMap->gridCount() ? m_dofMap->gridId(node)
                                                           : m_dofMap->scalarId(node - m_dofMap->gridCount()));
    }
    std::sort(knownPoints.begin(), knownPoints.end());
    std::vector<std::int32_t> knownElements; // bounded by the model's elements
    for (const assembly::AssemblyBatch& batch : elements.batches())
        for (std::int64_t row = 0; row < batch.view.rowCount(); ++row)
            if (!std::binary_search(m_replaced.begin(), m_replaced.end(), batch.view.elementId(row)))
                knownElements.push_back(batch.view.elementId(row));
    std::sort(knownElements.begin(), knownElements.end());
    const auto restrict = [&](output::OutputKind kind, const std::vector<std::int32_t>& allowed)
    {
        output::OutputRequestPlan request = plan.of(kind);
        if (!request.isRequested)
            return;
        std::vector<std::int32_t> members;
        if (request.isAll)
            members = allowed;
        else
            std::set_intersection(request.members.begin(), request.members.end(), allowed.begin(), allowed.end(),
                                  std::back_inserter(members));
        request.isAll = false;
        request.members = std::move(members);
        restricted.setRequest(kind, std::move(request));
    };
    for (const output::OutputKind kind :
         {output::OutputKind::Displacement, output::OutputKind::AppliedLoads, output::OutputKind::SpcForces,
          output::OutputKind::MpcForces, output::OutputKind::GridPointForces})
        restrict(kind, knownPoints);
    for (const output::OutputKind kind :
         {output::OutputKind::Stress, output::OutputKind::Strain, output::OutputKind::Force})
        restrict(kind, knownElements);
    // The strain energy percentages need the energy of every element, the plugin's included.
    if (plan.of(output::OutputKind::StrainEnergy).isRequested)
        restricted.setRequest(output::OutputKind::StrainEnergy, output::OutputRequestPlan{});
    return restricted;
}

std::vector<ExternalUnitCheckOutcome> ExternalUnitPath::outcomes() const
{
    std::vector<ExternalUnitCheckOutcome> outcomes; // bounded by the plugins
    for (const Unit& unit : m_units)
        outcomes.push_back({unit.zoneId, unit.report, unit.checks});
    return outcomes;
}

std::vector<ExternalUnitCheckOutcome> checkExternalUnits(const input::ModelSnapshot& snapshot,
                                                         std::span<const ExternalUnitRequest> requests,
                                                         base::MemoryBudget& budget, base::DiagnosticSink& sink)
{
    const elements::ElementRegistry registry = elements::ElementRegistry::forModel(snapshot);
    const elements::SnapshotIndex index(snapshot, budget);
    const assembly::ElementSet elements(index, registry);
    const assembly::DofMap dofMap(snapshot.nodes(), index.points());
    const std::vector<StaticSubcase> subcases = staticSubcasesOf(snapshot, {}, sink);
    ExternalUnitPath::Inputs inputs;
    inputs.snapshot = &snapshot;
    inputs.elements = &elements;
    inputs.dofMap = &dofMap;
    inputs.requests = requests;
    inputs.subcases = subcases;
    const ExternalUnitPath path(inputs, budget, sink);
    return path.outcomes();
}

std::vector<ExternalUnitRecord> ExternalUnitPath::takeRecords(base::MemoryBudget& budget)
{
    std::vector<ExternalUnitRecord> records; // bounded by the plugins
    for (Unit& unit : m_units)
    {
        if (!unit.placement)
            continue;
        ExternalUnitRecord record;
        record.placement = unit.placement;
        record.report = unit.report;
        record.checks = unit.checks;
        record.designStamp = unit.placement->unit().designStamp();
        const std::size_t n = unit.placement->contourSize();
        const std::size_t cases = unit.recordedCases.size();
        record.displacements = linalg::DenseMatrix(budget, n, cases, "external unit contour displacements");
        record.condensedLoads = linalg::DenseMatrix(budget, n, cases, "external unit condensed loads");
        record.forces = linalg::DenseMatrix(budget, n, cases, "external unit contour forces");
        for (std::size_t c = 0; c < cases; ++c)
        {
            const std::size_t s = unit.recordedCases[c];
            record.cases.push_back(m_subcases[s].label);
            for (std::size_t a = 0; a < n; ++a)
            {
                record.displacements(a, c) = unit.caseDisplacements(a, s);
                record.condensedLoads(a, c) = unit.caseLoads(a, s);
                record.forces(a, c) = unit.caseForces(a, s);
            }
        }
        record.bases = std::move(unit.bases);
        unit.recordedCases.clear();
        records.push_back(std::move(record));
    }
    return records;
}

} // namespace oos::solutions
