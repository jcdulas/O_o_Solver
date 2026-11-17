#include "oos/solutions/common/CommonConstraints.hpp"

#include "oos/assembly/constraints/FastenerConstraints.hpp"
#include "oos/assembly/constraints/FormulationConstraints.hpp"
#include "oos/assembly/constraints/GlueConstraints.hpp"
#include "oos/assembly/constraints/MpcEquations.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/constraints/SpcSet.hpp"

#include <algorithm>
#include <bit>
#include <map>
#include <set>
#include <tuple>
#include <utility>

namespace oos::solutions
{

namespace
{

// A relation of the formulation as a comparable value: its dependent DOF and its terms, the coefficients by their
// bits (a relation shared by every subcase is the same to the bit).
using RelationTerms = std::vector<std::pair<base::DofIndex, std::uint64_t>>;

// The element formulation's constraints of one subcase, as the full path builds them (solutions::ConstrainedSystem):
// the SPC set with the derived supports, the model's relations (rigid elements, the subcase's MPC set, fasteners,
// glue), then FormulationConstraints. Its relations and excluded DOFs are added to `relations` and `excluded`.
void addFormulationConstraints(const input::ModelSnapshot& snapshot, const assembly::DofMap& dofMap,
                               const StaticSubcase* subcase, const assembly::SpcSet& spc,
                               std::span<const std::uint8_t> stiffenedComponents, base::MemoryBudget& budget,
                               std::map<base::DofIndex, RelationTerms>& relations,
                               std::vector<base::DofIndex>& excluded)
{
    const elements::FormulationModel& formulation = *dofMap.formulation();
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    const input::RigidTable* rigidElements =
        snapshot.hasTable(input::TableRole::RigidElements) ? &snapshot.rigidElements() : nullptr;
    const input::FastenerPiercingTable* piercings =
        snapshot.hasTable(input::TableRole::FastenerPiercings) ? &snapshot.fastenerPiercings() : nullptr;
    // The analysis reports these problems with its constraint sets.
    base::DiagnosticSink quiet;
    const std::optional<base::EntityId> mpcSet = subcase != nullptr ? subcase->mpc : std::nullopt;
    const assembly::MpcEquations mpc(constraints, dofMap, mpcSet, budget, quiet);
    const assembly::FastenerConstraints fasteners(piercings, dofMap, budget, quiet, &formulation);
    const assembly::GlueContext glueContext{rigidElements, &mpc, &fasteners, spc.sbDofs(), spc.sgDofs(), &formulation};
    const assembly::GlueConstraints glue(snapshot, subcase != nullptr ? subcase->contact : std::nullopt,
                                         subcase != nullptr ? subcase->label : 1, dofMap, glueContext, budget, quiet);
    const assembly::RigidElimination userRelations(rigidElements, &mpc, dofMap, budget, quiet, &fasteners, &glue);
    assembly::UsedElsewhere usedElsewhere;
    usedElsewhere.flags.assign(static_cast<std::size_t>(dofMap.gSetSize()), 0);
    usedElsewhere.markConcentratedLoads(snapshot, dofMap);
    if (!stiffenedComponents.empty())
        usedElsewhere.markStiffened(stiffenedComponents, dofMap);
    usedElsewhere.markIndependentTerms(userRelations);
    const assembly::FormulationConstraints own(formulation, dofMap.layoutKind(), dofMap, spc, usedElsewhere,
                                               userRelations, quiet);
    const assembly::FormulationRelations& rows = own.relations();
    for (std::size_t i = 0; i < rows.rowCount(); ++i)
    {
        RelationTerms& terms = relations[rows.dependentDof(i)];
        for (const assembly::ConstraintTerm& term : rows.row(i))
            terms.emplace_back(term.dof, std::bit_cast<std::uint64_t>(term.coefficient));
    }
    excluded.assign(own.excludedDofs().begin(), own.excludedDofs().end());
}

// The DOF of an element that names a g-set DOF.
elements::ElementDof elementDofOf(const assembly::DofLabel& label)
{
    return {label.kind, label.component, label.row};
}

// The nodes of a relation: its dependent DOF's and its terms'.
void addNodesOf(const assembly::DofMap& dofMap, base::DofIndex dependent, const RelationTerms& terms,
                std::set<std::int32_t>& nodes)
{
    nodes.insert(dofMap.nodeDofOf(dependent).node);
    for (const auto& [dof, bits] : terms)
        nodes.insert(dofMap.nodeDofOf(dof).node);
}

} // namespace

CommonConstraints CommonConstraints::of(const input::ModelSnapshot& snapshot, const assembly::DofMap& dofMap,
                                        std::span<const StaticSubcase> subcases, base::MemoryBudget& budget,
                                        base::DiagnosticSink& sink, std::span<const std::uint8_t> stiffenedComponents)
{
    const elements::FormulationModel* formulation = dofMap.formulation();
    // The formulation's constraints of the subcases of distinct constraint selections (SPC set, load set for the
    // enforced motions, MPC set, contact table), each counted once per subcase that selects it.
    using Selection = std::tuple<std::optional<base::EntityId>, std::optional<base::EntityId>,
                                 std::optional<base::EntityId>, std::optional<base::EntityId>>;
    struct FormulationPart
    {
        std::map<base::DofIndex, RelationTerms> relations;
        std::vector<base::DofIndex> excluded;
        std::int32_t subcaseCount = 0;
    };
    std::map<Selection, FormulationPart> formulationParts;
    // The notes of the derived supports are the constraint sets' own, reported by the analysis.
    base::DiagnosticSink derivedSink;
    const auto addFormulation = [&](const StaticSubcase* subcase, const assembly::SpcSet& spc)
    {
        const Selection selection =
            subcase != nullptr ? Selection{subcase->spc, subcase->load, subcase->mpc, subcase->contact} : Selection{};
        FormulationPart& part = formulationParts[selection];
        if (part.subcaseCount++ == 0)
            addFormulationConstraints(snapshot, dofMap, subcase, spc, stiffenedComponents, budget, part.relations,
                                      part.excluded);
    };
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    CommonConstraints common;
    // Glued contact: one Gm for the condensation, so one contact table for every subcase.
    std::set<std::optional<base::EntityId>> contactTables;
    for (const StaticSubcase& subcase : subcases)
        contactTables.insert(subcase.contact);
    if (contactTables.size() == 1)
        common.commonContact = *contactTables.begin();
    if (contactTables.size() > 1)
    {
        base::Diagnostic diagnostic;
        diagnostic.code = "glued-contact-varies";
        diagnostic.card = "ContactSelection";
        diagnostic.message =
            "the subcases select different contact tables (ContactSelection); the condensed path of this "
            "version of "
            "O_o_Solver takes one glued contact for every subcase: select the same ContactSelection in every "
            "subcase, or run "
            "the full path";
        sink.report(std::move(diagnostic));
    }
    const auto gSize = static_cast<std::size_t>(dofMap.gSetSize());
    common.constrainedCount = base::BudgetedBuffer<std::int32_t>(budget, gSize, "subcases constraining each DOF");
    // The MPC cards each subcase selects: its set resolved to the SIDs of the MPC cards it holds.
    std::optional<std::set<base::EntityId>> commonCards;
    std::set<base::EntityId> selectedCards;
    for (const StaticSubcase& subcase : subcases)
    {
        const assembly::SpcSet spc = assembly::withFormulationSupports(
            assembly::SpcSet(constraints, snapshot.findLoads(), dofMap, subcase.spc, subcase.load, budget, sink),
            dofMap, formulation, dofMap.layoutKind(), budget, derivedSink);
        if (formulation != nullptr)
            addFormulation(&subcase, spc);
        // sb and sg are sorted; a DOF in both (an SPC on a GRID PS component) counts once.
        for (const base::DofIndex dof : spc.sbDofs())
            ++common.constrainedCount[static_cast<std::size_t>(dof)];
        for (const base::DofIndex dof : spc.sgDofs())
            if (!std::binary_search(spc.sbDofs().begin(), spc.sbDofs().end(), dof))
                ++common.constrainedCount[static_cast<std::size_t>(dof)];
        std::vector<base::EntityId> cards; // bounded by the MPC cards of one set
        if (subcase.mpc)
        {
            // Without the source_sid column of multiPointConstrTable (to be added by O_o_Reader) an
            // MultiPointConstrUnion stands for its own cards, keyed by its ID, which connectionClusters does not use:
            // CondensationPlanner then matches the cards' equations to the set's (reduction::MpcSetAlias), so that the
            // plan keys them as the analysis does.
            cards = assembly::MpcEquations::sourceSetIdsOf(constraints, *subcase.mpc);
            common.anyMpcSet = subcase.mpc;
        }
        selectedCards.insert(cards.begin(), cards.end());
        if (!commonCards)
            commonCards.emplace(cards.begin(), cards.end());
        else
            std::erase_if(*commonCards, [&cards](base::EntityId setId)
                          { return !std::binary_search(cards.begin(), cards.end(), setId); });
    }
    if (subcases.empty())
    {
        // No subcase table: the permanent constraints of the grid points alone.
        const assembly::SpcSet spc = assembly::withFormulationSupports(
            assembly::SpcSet(constraints, snapshot.findLoads(), dofMap, std::nullopt, std::nullopt, budget, sink),
            dofMap, formulation, dofMap.layoutKind(), budget, derivedSink);
        for (const base::DofIndex dof : spc.sbDofs())
            ++common.constrainedCount[static_cast<std::size_t>(dof)];
        for (const base::DofIndex dof : spc.sgDofs())
            if (!std::binary_search(spc.sbDofs().begin(), spc.sbDofs().end(), dof))
                ++common.constrainedCount[static_cast<std::size_t>(dof)];
        if (formulation != nullptr)
            addFormulation(nullptr, spc);
    }
    common.subcaseCount = static_cast<std::int32_t>(std::max<std::size_t>(subcases.size(), 1));
    if (formulation != nullptr)
    {
        // The excluded DOFs count as held in the subcases that exclude them; the relations and excluded DOFs every
        // subcase has are common, the nodes of the others vary.
        std::map<base::DofIndex, std::int32_t> excludedCount;
        std::map<base::DofIndex, std::pair<const RelationTerms*, std::int32_t>> relationCount;
        std::set<std::int32_t> varying;
        for (const auto& [selection, part] : formulationParts)
        {
            for (const base::DofIndex dof : part.excluded)
            {
                common.constrainedCount[static_cast<std::size_t>(dof)] += part.subcaseCount;
                excludedCount[dof] += part.subcaseCount;
            }
            for (const auto& [dependent, terms] : part.relations)
            {
                auto [entry, isNew] = relationCount.try_emplace(dependent, &terms, 0);
                if (!isNew && *entry->second.first != terms)
                {
                    // The same dependent DOF with other terms: both relations vary.
                    addNodesOf(dofMap, dependent, *entry->second.first, varying);
                    addNodesOf(dofMap, dependent, terms, varying);
                    entry->second.second = -common.subcaseCount; // never common
                }
                entry->second.second += part.subcaseCount;
            }
        }
        for (const auto& [dof, count] : excludedCount)
        {
            if (count == common.subcaseCount)
                common.formulationExcluded.push_back(dof);
            else
                varying.insert(dofMap.nodeDofOf(dof).node);
        }
        auto relations = std::make_shared<assembly::FormulationRelations>(dofMap);
        std::vector<assembly::FormulationTerm> row; // bounded by one relation
        for (const auto& [dependent, entry] : relationCount)
        {
            if (entry.second != common.subcaseCount)
            {
                addNodesOf(dofMap, dependent, *entry.first, varying);
                continue;
            }
            row.clear();
            for (const auto& [dof, bits] : *entry.first)
                row.push_back({elementDofOf(dofMap.labelOf(dof)), std::bit_cast<double>(bits)});
            relations->add(elementDofOf(dofMap.labelOf(dependent)), row);
        }
        if (!relations->empty())
            common.formulationRelations = std::move(relations);
        common.formulationVaryingNodes.assign(varying.begin(), varying.end());
    }
    if (commonCards)
        common.commonMpcCards.assign(commonCards->begin(), commonCards->end());
    for (const base::EntityId setId : selectedCards)
        if (!std::binary_search(common.commonMpcCards.begin(), common.commonMpcCards.end(), setId))
            common.varyingMpcSets.push_back(setId);
    return common;
}

} // namespace oos::solutions
