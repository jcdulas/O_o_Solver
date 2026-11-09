#pragma once

#include "oos/assembly/constraints/GlueConstraints.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/input/snapshot/ModelSnapshot.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/reduction/absorption/AbsorptionEligibility.hpp"
#include "oos/reduction/absorption/AbsorptionSafeguard.hpp"
#include "oos/reduction/absorption/SharedConnectionNodes.hpp"
#include "oos/reduction/absorption/SpiderCatalog.hpp"
#include "oos/reduction/plan/CondensationPlan.hpp"
#include "oos/reduction/plan/PlanGraph.hpp"
#include "oos/reduction/units/UnitSet.hpp"
#include "oos/solutions/common/GlueUnitCheck.hpp"

#include <memory>
#include <vector>

namespace oos::solutions
{

class AnalysisModel;

// The condensation plan of a static analysis from a model snapshot: what the reduction module needs that only a
// solution knows, then the plan.
//
// What it gathers (the steps of a static solution before its first assembly, src/solutions/README.md):
// - the analysis model (support check, element check, DOF map, element connectivity: AnalysisModel);
// - the subcases (staticSubcasesOf; the SUBCOMs constrain nothing of their own) and, per subcase, its SPC set (SpcSet);
//   a DOF constrained in every subcase is removed from the units and from the interface problem, a DOF constrained in
//   some subcases only puts its point on the boundary of its unit;
// - the MPC equations, by the SID of their MPC card (an MultiPointConstrUnion is resolved to the cards it holds, from
// the source_sid
//   column of multiPointConstrTable when O_o_Reader gives it, MpcEquations::sourceSetIdsOf): the cards every subcase
//   selects go into the dependent DOFs of the plan graph with the rigid elements and the fasteners (Gm), keyed by their
//   card's SID as connectionClusters names them; a card some subcases select and others do not keeps its equations in
//   the connection system (reduction::AbsorptionOptions::varyingMpcSets);
// - the glued contact: the glue of the contact table every subcase selects goes into Gm too, one relation per glued
//   point (reduction::glueRelationKey; subcases selecting different tables are refused, "glued-contact-varies"); plan()
//   hands the points whose equations join units to the plan as interface relations (glueUnitCheck), which
//   reduction::BoundaryPromotion puts on the boundaries;
// - the node graph of the stiffness pattern (the element groups and the couplings of the m-set elimination,
//   assembly::MsetCouplings, as ConstrainedSystem builds it), the free DOFs of each point and the positions;
// - the FreeBodySupport points (when the snapshot has the dof_sets table): boundary points;
// - the spiders: the catalog of O_o_Reader's spiders and fastener sides, and the measurement of each relation on
//   this model: the conditioning of its change of basis (AbsorptionSafeguard, from the rows of Gm of an
//   RigidInterpolation or the interpolation of the piercing point of a fastener side) and whether a leg DOF of the
//   relation is constrained in
// every subcase; the
//   connections at each grid point (connectionSharing, when the snapshot has it); the points of the contacts between
//   bodies (contactNodePairs: secondary points and primary face points), on which a leg is touched by a connection (a
//   glue equation is a connection like an MPC).
// Then plan(): the connection system (from the snapshot's connection tables), the cluster absorption, the spider
// absorption (request.absorbsSpiders; none without the spiders table but the fastener sides), the boundaries, the plan
// graph with the references of the absorbed RigidInterpolation freed, and the CondensationPlan with its estimates.
// Nothing is assembled: the estimates come from the symbolic analysis.
//
// Thread-safety: immutable once built; plan() may run from several threads on different units. Ownership: owns the
// analysis model, the graph and the arrays (budgeted); borrows the snapshot, which must outlive it. Not movable.
class CondensationPlanner
{
public:
    // Throws as AnalysisModel (UnsupportedCardError, SolverError) and as the constraint builders (SolverError listing
    // the problems), MemoryBudgetError.
    CondensationPlanner(const input::ModelSnapshot& snapshot, base::MemoryBudget& budget, base::ThreadPool* pool,
                        base::DiagnosticSink& sink);
    CondensationPlanner(const CondensationPlanner&) = delete;
    CondensationPlanner& operator=(const CondensationPlanner&) = delete;
    CondensationPlanner(CondensationPlanner&&) = delete;
    CondensationPlanner& operator=(CondensationPlanner&&) = delete;
    ~CondensationPlanner();

    // The plan of `units` (made from this snapshot). absorbsClusters: false keeps every cluster in the connection
    // system (the boundaries are then O_o_Reader's interface points). Throws base::ContractError when the snapshot has
    // no connection tables, std::invalid_argument for a request that names unknown units or points.
    [[nodiscard]] reduction::CondensationPlan plan(const reduction::UnitSet& units,
                                                   const reduction::PlanRequest& request,
                                                   bool absorbsClusters = true) const;

    [[nodiscard]] const reduction::PlanGraph& planGraph() const noexcept { return *m_planGraph; }
    [[nodiscard]] const AnalysisModel& analysisModel() const noexcept { return *m_model; }
    // The points with a DOF constrained in some subcases only, sorted.
    [[nodiscard]] const std::vector<base::EntityId>& partiallyConstrainedNodes() const noexcept
    {
        return m_partiallyConstrainedNodes;
    }
    // The SIDs of the MPC cards some subcases select and others do not, sorted.
    [[nodiscard]] const std::vector<base::EntityId>& varyingMpcSets() const noexcept { return m_varyingMpcSets; }
    [[nodiscard]] const std::vector<base::EntityId>& supportNodes() const noexcept { return m_supportNodes; }
    // The spiders of the model and their measurements (one per spider of the catalog).
    [[nodiscard]] const reduction::SpiderCatalog& spiderCatalog() const noexcept { return *m_spiders; }
    [[nodiscard]] std::span<const reduction::SpiderMeasurement> spiderMeasurements() const noexcept
    {
        return m_spiderMeasurements.span();
    }
    // The glue equations of the contact table every subcase selects (no row without glued contact).
    [[nodiscard]] const assembly::GlueConstraints& glue() const noexcept { return *m_glue; }
    // The glue equations against `units`: the glued points eliminated inside one unit, those whose equations join
    // units (eliminated at the interface: plan() hands them to the plan), with a report line per contact table.
    [[nodiscard]] GlueUnitCheck glueUnitCheck(const reduction::UnitSet& units) const;
    // The connections at each grid point; null when the snapshot has no connectionSharing table.
    [[nodiscard]] const reduction::SharedConnectionNodes* sharedConnectionNodes() const noexcept
    {
        return m_sharing.get();
    }

private:
    const input::ModelSnapshot* m_snapshot;
    base::MemoryBudget* m_budget;
    base::ThreadPool* m_pool;
    std::unique_ptr<AnalysisModel> m_model;
    std::unique_ptr<assembly::GlueConstraints> m_glue;
    std::unique_ptr<linalg::NodeGraph> m_graph;
    base::BudgetedBuffer<base::EntityId> m_nodeIds;
    base::BudgetedBuffer<std::int32_t> m_unconstrainedDofCounts;
    base::BudgetedBuffer<std::int32_t> m_freeDofCounts;
    base::BudgetedBuffer<base::Vec3> m_positions;
    std::unique_ptr<reduction::PlanGraph> m_planGraph;
    std::vector<base::EntityId> m_partiallyConstrainedNodes; // bounded by the constrained points
    std::vector<base::EntityId> m_varyingMpcSets;            // bounded by the subcases
    // Without the source_sid column of multiPointConstrTable: the card sets each selected MPC set holds (review B-1),
    // sorted.
    std::vector<reduction::MpcSetAlias> m_mpcAliases;
    std::vector<base::EntityId> m_supportNodes; // bounded by the FreeBodySupport cards
    std::unique_ptr<reduction::SpiderCatalog> m_spiders;
    base::BudgetedBuffer<reduction::SpiderMeasurement> m_spiderMeasurements; // parallel to the catalog
    std::unique_ptr<reduction::SharedConnectionNodes> m_sharing;
    base::BudgetedBuffer<reduction::GluedPoint> m_gluedPoints; // sorted by point; views the snapshot's contact table
};

} // namespace oos::solutions
