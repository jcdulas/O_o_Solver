#include "oos/solutions/common/CommonConstraints.hpp"

#include "oos/assembly/constraints/MpcEquations.hpp"
#include "oos/assembly/constraints/SpcSet.hpp"

#include <algorithm>
#include <set>
#include <utility>

namespace oos::solutions
{

CommonConstraints CommonConstraints::of(const input::ModelSnapshot& snapshot, const assembly::DofMap& dofMap,
                                        std::span<const StaticSubcase> subcases, base::MemoryBudget& budget,
                                        base::DiagnosticSink& sink)
{
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
        const assembly::SpcSet spc(constraints, snapshot.findLoads(), dofMap, subcase.spc, subcase.load, budget, sink);
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
        const assembly::SpcSet spc(constraints, snapshot.findLoads(), dofMap, std::nullopt, std::nullopt, budget, sink);
        for (const base::DofIndex dof : spc.sgDofs())
            ++common.constrainedCount[static_cast<std::size_t>(dof)];
    }
    common.subcaseCount = static_cast<std::int32_t>(std::max<std::size_t>(subcases.size(), 1));
    if (commonCards)
        common.commonMpcCards.assign(commonCards->begin(), commonCards->end());
    for (const base::EntityId setId : selectedCards)
        if (!std::binary_search(common.commonMpcCards.begin(), common.commonMpcCards.end(), setId))
            common.varyingMpcSets.push_back(setId);
    return common;
}

} // namespace oos::solutions
