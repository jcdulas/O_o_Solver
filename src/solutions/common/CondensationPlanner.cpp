#include "oos/solutions/common/CondensationPlanner.hpp"

#include "oos/assembly/constraints/FastenerConstraints.hpp"
#include "oos/assembly/constraints/MpcEquations.hpp"
#include "oos/assembly/constraints/MsetCouplings.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/reduction/absorption/FastenerPiercing.hpp"
#include "oos/reduction/absorption/SpiderAbsorption.hpp"
#include "oos/reduction/absorption/SpiderWeights.hpp"
#include "oos/reduction/units/ClusterAbsorption.hpp"
#include "oos/reduction/units/ConnectionSystem.hpp"
#include "oos/reduction/units/UnitBoundaries.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/CommonConstraints.hpp"
#include "oos/solutions/common/GlueUnitCheck.hpp"
#include "oos/solutions/common/RunChecks.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace oos::solutions
{

namespace
{

// The measurement of one spider's relation on the model: the conditioning of its change of basis and whether a leg DOF
// of the relation is constrained in every subcase (such a leg keeps its spider on its legs). Not measured (hasRelation
// false) for a card that is not a spider or whose relation the model does not have.
reduction::SpiderMeasurement measureSpider(const reduction::Spider& spider, const reduction::SpiderWeights& weights,
                                           const input::FastenerPiercingTable* piercings,
                                           std::span<const std::int32_t> constrainedCount, std::int32_t subcaseCount,
                                           base::MemoryBudget& budget)
{
    if (!spider.isSpider)
        return {};
    reduction::SpiderMeasurement measurement;
    std::vector<base::DofIndex> legDofs; // bounded by one spider
    switch (spider.kind)
    {
    case reduction::SpiderKind::RigidBodyStar:
    {
        // Absorbed by its own elimination: no change of basis; its legs are its dependent DOFs, GN keeps its six.
        constexpr std::int32_t RIGID_BODY_STAR_REFERENCE_DOFS = 6;
        legDofs = weights.dependentDofs(spider.key.elementId, assembly::ConstraintSource::RigidBodyStar);
        if (legDofs.empty())
            return {};
        measurement = reduction::AbsorptionSafeguard::rigidBodyStarMeasurement(RIGID_BODY_STAR_REFERENCE_DOFS);
        break;
    }
    case reduction::SpiderKind::RigidInterpolation:
    {
        const std::optional<reduction::SpiderRelation> relation =
            weights.rigidInterpolationRelation(spider.key.elementId);
        if (!relation)
            return {};
        measurement = reduction::AbsorptionSafeguard::measure(relation->weights, budget);
        legDofs = relation->legDofs;
        break;
    }
    case reduction::SpiderKind::FastenerSide:
    {
        if (piercings == nullptr)
            return {};
        std::variant<reduction::SpiderRelation, std::string> relation =
            reduction::FastenerPiercing::relation(*piercings, weights.dofMap(), spider.key.elementId, spider.key.side);
        const auto* found = std::get_if<reduction::SpiderRelation>(&relation);
        if (found == nullptr)
            return {};
        measurement = reduction::AbsorptionSafeguard::measure(found->weights, budget);
        legDofs = found->legDofs;
        break;
    }
    }
    for (const base::DofIndex dof : legDofs)
        measurement.hasLegConstrainedEverywhere =
            measurement.hasLegConstrainedEverywhere || constrainedCount[static_cast<std::size_t>(dof)] == subcaseCount;
    return measurement;
}

// The MPC card sets each selected set holds, when multiPointConstrTable does not name the card of each row (its
// source_sid column): every set of the model is in the table (an MultiPointConstrUnion expanded under its own ID), and
// a set c is held by a selected set S when every equation of c is an equation of S, term for term (point, component and
// the bits of the coefficient). O_o_Reader's connection list names an equation by its card's set; the constraints of
// the analysis by S: the alias lets the plan key the equations as the analysis does.
std::vector<reduction::MpcSetAlias> mpcSetAliases(const input::ConstraintTable* constraints,
                                                  std::span<const StaticSubcase> subcases)
{
    std::vector<reduction::MpcSetAlias> aliases;
    if (constraints == nullptr || assembly::MpcEquations::hasSourceSetIds(constraints))
        return aliases;
    std::set<base::EntityId> selected;
    for (const StaticSubcase& subcase : subcases)
        if (subcase.mpc)
            selected.insert(*subcase.mpc);
    if (selected.empty())
        return aliases;
    using Term = std::tuple<std::int32_t, std::int32_t, std::uint64_t>;
    const std::span<const std::int32_t> setIds = constraints->mpcSetIds();
    const std::span<const std::int32_t> equations = constraints->mpcEquations();
    const std::span<const std::int32_t> nodeIds = constraints->mpcNodeIds();
    const std::span<const std::int32_t> components = constraints->mpcComponents();
    const std::span<const double> coefficients = constraints->mpcCoefficients();
    // The equations of each set, as their terms in order.
    std::map<base::EntityId, std::set<std::vector<Term>>> equationsOf;
    for (std::size_t row = 0; row < setIds.size();)
    {
        std::size_t end = row;
        std::vector<Term> terms; // bounded by the terms of one equation
        while (end < setIds.size() && setIds[end] == setIds[row] && equations[end] == equations[row])
        {
            terms.emplace_back(nodeIds[end], components[end], std::bit_cast<std::uint64_t>(coefficients[end]));
            ++end;
        }
        equationsOf[setIds[row]].insert(std::move(terms));
        row = end;
    }
    for (const base::EntityId set : selected)
    {
        const auto held = equationsOf.find(set);
        if (held == equationsOf.end())
            continue;
        for (const auto& [card, cardEquations] : equationsOf)
            if (card != set &&
                std::includes(held->second.begin(), held->second.end(), cardEquations.begin(), cardEquations.end()))
                aliases.push_back({card, set});
    }
    std::sort(aliases.begin(), aliases.end());
    return aliases;
}

} // namespace

CondensationPlanner::CondensationPlanner(const input::ModelSnapshot& snapshot, base::MemoryBudget& budget,
                                         base::ThreadPool* pool, base::DiagnosticSink& sink)
    : m_snapshot(&snapshot), m_budget(&budget), m_pool(pool)
{
    m_model = std::make_unique<AnalysisModel>(snapshot, AnalysisKind::Statics, budget, pool, nullptr, sink);
    const assembly::DofMap& dofMap = m_model->dofMap();
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;

    // The constraints of every subcase (the SUBCOMs combine solved subcases and constrain nothing of their own).
    std::vector<StaticSubcase> subcases;
    if (snapshot.hasTable(input::TableRole::Subcases))
        subcases = staticSubcasesOf(snapshot, {}, sink);
    std::erase_if(subcases, [](const StaticSubcase& subcase) { return subcase.isCombination; });
    // What every subcase shares (CommonConstraints): the subcases constraining each DOF, the MPC cards every subcase
    // selects and those that vary, one contact table for every subcase (TODO: the glue rows of subcases selecting
    // different tables in the connection system, as the varying MPC sets).
    CommonConstraints common = CommonConstraints::of(snapshot, dofMap, subcases, budget, sink);
    const std::optional<base::EntityId> commonContact = common.commonContact;
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    const base::BudgetedBuffer<std::int32_t>& constrainedCount = common.constrainedCount;
    const std::int32_t subcaseCount = common.subcaseCount;
    const std::vector<base::EntityId>& commonCards = common.commonMpcCards;
    m_varyingMpcSets = common.varyingMpcSets;
    m_mpcAliases = mpcSetAliases(constraints, subcases);
    stopOnErrors(sink, "the condensation plan");

    // The dependent DOFs every subcase shares: rigid elements, fasteners, the MPC cards every subcase selects (the
    // equations of any selected set that come from those cards: every subcase's set holds them).
    const std::optional<base::EntityId> commonMpcSet = common.commonMpcSet();
    const assembly::MpcEquations mpc(constraints, dofMap, commonMpcSet, commonCards, budget, sink);
    // The MPC card of each dependent DOF of an MPC equation, the key of its relation in the plan graph.
    std::map<base::DofIndex, base::EntityId> mpcCardOfDependent;
    for (std::size_t i = 0; i < mpc.size(); ++i)
        mpcCardOfDependent.emplace(mpc.dependentDof(i), mpc.sourceSetIdOf(i));
    const assembly::FastenerConstraints fasteners(
        snapshot.hasTable(input::TableRole::FastenerPiercings) ? &snapshot.fastenerPiercings() : nullptr, dofMap,
        budget, sink);
    // The glue of the common contact table; a component held in every subcase keeps its SPC (GlueConstraints), one
    // held in some subcases only is tied here (its point is a boundary point of its unit already).
    const input::RigidTable* rigidElements =
        snapshot.hasTable(input::TableRole::RigidElements) ? &snapshot.rigidElements() : nullptr;
    std::size_t heldEverywhereCount = 0;
    for (std::size_t dof = 0; dof < gSize; ++dof)
        heldEverywhereCount += constrainedCount[dof] == subcaseCount ? 1 : 0;
    base::BudgetedBuffer<base::DofIndex> heldEverywhere(budget, heldEverywhereCount,
                                                        "plan: DOFs held in every subcase");
    std::size_t nextHeld = 0;
    for (std::size_t dof = 0; dof < gSize; ++dof)
        if (constrainedCount[dof] == subcaseCount)
            heldEverywhere[nextHeld++] = static_cast<base::DofIndex>(dof);
    const assembly::GlueContext glueContext{rigidElements, &mpc, &fasteners, heldEverywhere.span(), {}};
    m_glue = std::make_unique<assembly::GlueConstraints>(
        snapshot, commonContact, subcases.empty() ? 1 : subcases.front().label, dofMap, glueContext, budget, sink);
    const assembly::RigidElimination gm(rigidElements, &mpc, dofMap, budget, sink, &fasteners, m_glue.get());
    stopOnErrors(sink, "the condensation plan");
    const assembly::MsetCouplings couplings(m_model->connectivity().groups(), gm, dofMap, budget);
    const std::vector<linalg::NodeGroupList> groupLists{m_model->connectivity().groups(), couplings.groups()};
    m_graph = std::make_unique<linalg::NodeGraph>(dofMap.nodeCount(), groupLists, budget, pool);

    // Per point: its ID, its unconstrained and free DOFs and its position.
    const auto nodeCount = static_cast<std::size_t>(dofMap.nodeCount());
    m_nodeIds = base::BudgetedBuffer<base::EntityId>(budget, nodeCount, "plan: point IDs");
    m_unconstrainedDofCounts = base::BudgetedBuffer<std::int32_t>(budget, nodeCount, "plan: unconstrained DOFs");
    m_freeDofCounts = base::BudgetedBuffer<std::int32_t>(budget, nodeCount, "plan: free DOFs");
    m_positions = base::BudgetedBuffer<base::Vec3>(budget, nodeCount, "plan: point positions");
    base::BudgetedBuffer<std::uint8_t> isDependent(budget, gSize, "plan: dependent DOFs");
    for (const base::DofIndex dof : gm.mDofs())
        isDependent[static_cast<std::size_t>(dof)] = 1;
    for (std::size_t node = 0; node < nodeCount; ++node)
    {
        const bool isGrid = node < static_cast<std::size_t>(dofMap.gridCount());
        const auto row = static_cast<std::int32_t>(isGrid ? node : node - static_cast<std::size_t>(dofMap.gridCount()));
        m_nodeIds[node] = isGrid ? dofMap.gridId(row) : dofMap.scalarId(row);
        const int dofCount = isGrid ? 6 : 1;
        std::int32_t unconstrained = 0;
        std::int32_t free = 0;
        bool isPartial = false;
        for (int component = 0; component < dofCount; ++component)
        {
            const auto dof =
                static_cast<std::size_t>(isGrid ? dofMap.gridDof(row, component + 1) : dofMap.scalarDof(row));
            const std::int32_t count = constrainedCount[dof];
            if (count == subcaseCount)
                continue;
            ++unconstrained;
            isPartial = isPartial || count > 0;
            free += isDependent[dof] == 0 ? 1 : 0;
        }
        m_unconstrainedDofCounts[node] = unconstrained;
        m_freeDofCounts[node] = free;
        if (isPartial)
            m_partiallyConstrainedNodes.push_back(m_nodeIds[node]);
        // A scalar point takes the position of a grid point it is coupled to (FsetOrdering does the same).
        std::int32_t located = isGrid ? row : -1;
        if (!isGrid)
            for (const std::int32_t neighbour : m_graph->neighbours(static_cast<std::int32_t>(node)))
                if (neighbour < dofMap.gridCount())
                {
                    located = neighbour;
                    break;
                }
        m_positions[node] = located >= 0 ? dofMap.positionInBasic(located) : base::Vec3{};
    }
    std::sort(m_partiallyConstrainedNodes.begin(), m_partiallyConstrainedNodes.end());

    if (snapshot.hasTable(input::TableRole::Supports))
    {
        // dofSetsTable may list the other DOF set cards too (ASET, OmittedDofSet, ...): keep the FreeBodySupport and
        // SUPORT1 points.
        const base::ContractReader& supports = snapshot.supports();
        const std::span<const std::int32_t> nodeIds = supports.values<std::int32_t>("nodeId");
        const base::DictionaryView cards = supports.dictionaryTexts("card");
        for (std::int64_t row = 0; row < supports.rowCount(); ++row)
        {
            if (cards[row] == "FreeBodySupport" || cards[row] == "FreeBodySupportSet")
                m_supportNodes.push_back(nodeIds[static_cast<std::size_t>(row)]);
        }
        std::sort(m_supportNodes.begin(), m_supportNodes.end());
        m_supportNodes.erase(std::unique(m_supportNodes.begin(), m_supportNodes.end()), m_supportNodes.end());
    }

    // The dependent DOFs by point and relation, with the points they depend on (the terms of their rows of Gm).
    std::map<std::pair<std::int32_t, reduction::RelationKey>, reduction::DependentPoint> dependents;
    for (std::size_t i = 0; i < gm.rowCount(); ++i)
    {
        const auto dof = static_cast<std::size_t>(gm.mDofs()[i]);
        if (constrainedCount[dof] == subcaseCount)
            continue;
        const assembly::ConstraintSource source = gm.sourceOf(i);
        const std::int32_t node = dofMap.nodeDofOf(gm.mDofs()[i]).node;
        // A glued point is a relation of its own: inside a unit or at the interface point by point. An MPC
        // equation is keyed by the SID of its MPC card, as the connection system names it.
        reduction::RelationKey relation{source == assembly::ConstraintSource::Mpc, gm.sourceIdOf(i)};
        if (isGlueSource(source))
            relation = reduction::glueRelationKey(dofMap.gridId(node));
        else if (relation.isMpc)
            relation.id = mpcCardOfDependent.at(gm.mDofs()[i]);
        reduction::DependentPoint& entry = dependents[{node, relation}];
        entry.node = node;
        entry.relation = relation;
        ++entry.dofCount;
        for (const assembly::ConstraintTerm& term : gm.row(i))
            entry.independentNodes.push_back(dofMap.nodeDofOf(term.dof).node);
    }
    std::vector<reduction::DependentPoint> dependentList;
    dependentList.reserve(dependents.size());
    for (auto& [key, entry] : dependents)
    {
        std::sort(entry.independentNodes.begin(), entry.independentNodes.end());
        entry.independentNodes.erase(std::unique(entry.independentNodes.begin(), entry.independentNodes.end()),
                                     entry.independentNodes.end());
        dependentList.push_back(std::move(entry));
    }
    m_planGraph =
        std::make_unique<reduction::PlanGraph>(*m_graph, m_nodeIds.span(), m_unconstrainedDofCounts.span(),
                                               m_freeDofCounts.span(), m_positions.span(), std::move(dependentList));

    // The spiders and the measurement of each relation on this model.
    const input::FastenerPiercingTable* piercings =
        snapshot.hasTable(input::TableRole::FastenerPiercings) ? &snapshot.fastenerPiercings() : nullptr;
    m_spiders = std::make_unique<reduction::SpiderCatalog>(
        snapshot.hasTable(input::TableRole::Spiders) ? &snapshot.spiders() : nullptr, piercings);
    const reduction::SpiderWeights weights(gm, dofMap, budget);
    const std::span<const reduction::Spider> spiders = m_spiders->spiders();
    m_spiderMeasurements =
        base::BudgetedBuffer<reduction::SpiderMeasurement>(budget, spiders.size(), "plan: spider measurements");
    for (std::size_t k = 0; k < spiders.size(); ++k)
        m_spiderMeasurements[k] =
            measureSpider(spiders[k], weights, piercings, constrainedCount.span(), subcaseCount, budget);
    if (snapshot.hasTable(input::TableRole::ConnectionSharing))
        m_sharing = std::make_unique<reduction::SharedConnectionNodes>(snapshot.connectionSharing());

    // The points of the contacts between bodies: the secondary point of every node pair and the points of the primary
    // face it meets (given only when the contact table has its face column; 0 for a mid-side point the face leaves
    // out). Every contact counts, glued or not, in every subcase: a spider with a leg there stays on its legs.
    if (snapshot.hasTable(input::TableRole::ContactNodePairs))
    {
        const input::ContactTable& contact = snapshot.contact();
        std::size_t count = 0;
        for (std::int64_t row = 0; row < contact.nodePairCount(); ++row)
        {
            const input::ContactNodePair pair = contact.nodePair(row);
            count += 1 + static_cast<std::size_t>(std::count_if(pair.face.begin(), pair.face.end(),
                                                                [](std::int32_t nodeId) { return nodeId != 0; }));
        }
        m_gluedPoints = base::BudgetedBuffer<reduction::GluedPoint>(budget, count, "plan: points of contact");
        std::size_t filled = 0;
        for (std::int64_t row = 0; row < contact.nodePairCount(); ++row)
        {
            const input::ContactNodePair pair = contact.nodePair(row);
            m_gluedPoints[filled++] = reduction::GluedPoint{pair.nodeId, pair.table};
            for (const std::int32_t nodeId : pair.face)
                if (nodeId != 0)
                    m_gluedPoints[filled++] = reduction::GluedPoint{nodeId, pair.table};
        }
        std::sort(m_gluedPoints.data(), m_gluedPoints.data() + filled,
                  [](const reduction::GluedPoint& a, const reduction::GluedPoint& b)
                  { return a.nodeId != b.nodeId ? a.nodeId < b.nodeId : a.table < b.table; });
    }
}

CondensationPlanner::~CondensationPlanner() = default;

GlueUnitCheck CondensationPlanner::glueUnitCheck(const reduction::UnitSet& units) const
{
    const assembly::DofMap& dofMap = m_model->dofMap();
    const base::BudgetedBuffer<std::int64_t> unitOf = unitOfGridRows(units, *m_snapshot, dofMap, *m_budget);
    return checkGlueAgainstUnits(*m_glue, dofMap, unitOf.span());
}

reduction::CondensationPlan CondensationPlanner::plan(const reduction::UnitSet& units,
                                                      const reduction::PlanRequest& request, bool absorbsClusters) const
{
    const reduction::ConnectionSystem system(units, m_snapshot->connectionNodes(), m_snapshot->connectionClusters(),
                                             m_mpcAliases);
    const reduction::ClusterAbsorption absorption(system,
                                                  reduction::AbsorptionOptions{absorbsClusters, m_varyingMpcSets});
    std::vector<base::EntityId> retained = request.retainedNodes;
    // The glued points whose equations join units: eliminated at the interface with the connection system's relations,
    // their points on the boundaries (reduction::BoundaryPromotion, through the plan's interface relations).
    const GlueUnitCheck glue = glueUnitCheck(units);
    std::sort(retained.begin(), retained.end());
    retained.erase(std::unique(retained.begin(), retained.end()), retained.end());
    // The spline nodes of SOL 144 join the exclusions with the aeroelastic planner (step 7).
    const reduction::AbsorptionExclusions exclusions{
        retained, {}, m_supportNodes, m_partiallyConstrainedNodes, m_gluedPoints.span()};
    const reduction::SpiderAbsorption spiders(absorption, *m_spiders, m_spiderMeasurements.span(), exclusions,
                                              m_sharing.get(), request.absorbsSpiders, request.maxSpiderCondition);
    const reduction::UnitBoundaries boundaries(
        absorption, reduction::BoundaryInputs{retained, m_partiallyConstrainedNodes, m_supportNodes}, &spiders);
    const std::span<const reduction::FreedReference> freed = spiders.freedReferences();
    if (freed.empty())
        return reduction::CondensationPlan(absorption, boundaries, *m_planGraph, request, *m_budget, m_pool, &spiders,
                                           glue.interfaceRelations);

    // The plan graph of this plan: the REFC DOFs of each absorbed RigidInterpolation are no longer dependent but
    // reference DOFs of the change of basis, free DOFs of the unit and of the interface problem; the rows of the given
    // GA (GB) of an absorbed fastener side leave the relations, the pierced element's grid point standing for the side
    // carrying its DOFs.
    const std::span<const std::int32_t> freeCounts = m_freeDofCounts.span();
    base::BudgetedBuffer<std::int32_t> freedCounts(*m_budget, freeCounts.size(), "plan: free DOFs, spiders absorbed");
    std::copy(freeCounts.begin(), freeCounts.end(), freedCounts.data());
    std::vector<reduction::DependentPoint> dependents; // bounded by the dependent points, as the planner's own
    for (const reduction::DependentPoint& entry : m_planGraph->dependents())
    {
        const base::EntityId nodeId = m_nodeIds[static_cast<std::size_t>(entry.node)];
        const std::pair<base::EntityId, base::EntityId> key{nodeId, entry.relation.id};
        const auto found = std::lower_bound(
            freed.begin(), freed.end(), key,
            [](const reduction::FreedReference& reference, const std::pair<base::EntityId, base::EntityId>& wanted)
            { return std::make_pair(reference.nodeId, reference.elementId) < wanted; });
        const bool isFreed = !entry.relation.isMpc && found != freed.end() && found->nodeId == nodeId &&
                             found->elementId == entry.relation.id;
        if (isFreed)
        {
            if (found->isFreeAtReference)
                freedCounts[static_cast<std::size_t>(entry.node)] += entry.dofCount;
            continue;
        }
        dependents.push_back(entry);
    }
    const reduction::PlanGraph graph(*m_graph, m_nodeIds.span(), m_unconstrainedDofCounts.span(), freedCounts.span(),
                                     m_positions.span(), std::move(dependents));
    return reduction::CondensationPlan(absorption, boundaries, graph, request, *m_budget, m_pool, &spiders,
                                       glue.interfaceRelations);
}

} // namespace oos::solutions
