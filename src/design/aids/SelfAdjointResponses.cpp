#include "oos/design/aids/SelfAdjointResponses.hpp"

#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/diagnostics/DiagnosticSink.hpp"
#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/output/arrays/NodalResultTable.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace oos::design
{

namespace
{

// c = Pᵀu = 2·W_ext: the external work the static solution reports is ½·uᵀ·P on the l set
// (solutions/sol101/StaticSolution).
constexpr double WORK_TO_COMPLIANCE = 2.0;

} // namespace

double SelfAdjointResponses::compliance(std::span<const output::SubcaseCheck> checks, std::int32_t subcase)
{
    const auto check = std::find_if(checks.begin(), checks.end(), [subcase](const output::SubcaseCheck& candidate)
                                    { return candidate.subcase == subcase; });
    if (check == checks.end())
        throw base::OptimisationError(base::concatenate(
            {"the compliance of subcase ", std::to_string(subcase),
             ": expected the external work of the subcase in the checks of the analysis, found none; evaluate the "
             "design point again"}));
    if (check->isCombination)
        throw base::OptimisationError(base::concatenate(
            {"the compliance of SubcaseCombination ", std::to_string(subcase),
             ": a combination of subcases does no work of its own (none is reported); constrain the compliance of the "
             "subcases it combines"}));
    return WORK_TO_COMPLIANCE * check->externalWork;
}

std::vector<GridForce> SelfAdjointResponses::appliedForces(const output::ResultArrays& arrays, std::int32_t subcase)
{
    const auto table = std::find_if(arrays.nodalTables.begin(), arrays.nodalTables.end(),
                                    [subcase](const output::NodalResultTable& candidate)
                                    { return candidate.kind == "appliedLoads" && candidate.subcase == subcase; });
    if (table == arrays.nodalTables.end())
        throw base::OptimisationError(
            base::concatenate({"the mean displacement of subcase ", std::to_string(subcase),
                               ": expected the applied loads (AppliedLoadOutput) of the subcase in the "
                               "results of the analysis, found none; the "
                               "design problem asks for them, so evaluate the design point again"}));
    std::vector<GridForce> forces;
    for (std::size_t row = 0; row < table->rowCount(); ++row)
    {
        if (table->pointTypes[row] != static_cast<std::int8_t>(output::PointType::Grid))
            continue;
        const double* values = table->values.data() + row * output::NodalResultTable::COMPONENT_COUNT;
        if (values[0] == 0.0 && values[1] == 0.0 && values[2] == 0.0)
            continue;
        forces.push_back({table->ids[row], {values[0], values[1], values[2]}});
    }
    return forces;
}

double SelfAdjointResponses::forceMagnitude(std::span<const GridForce> forces) noexcept
{
    double total = 0.0;
    for (const GridForce& entry : forces)
        total += std::sqrt(entry.force[0] * entry.force[0] + entry.force[1] * entry.force[1] +
                           entry.force[2] * entry.force[2]);
    return total;
}

std::set<std::int32_t> SelfAdjointResponses::subcasesWithEnforcedDisplacements(const elements::SnapshotIndex& index,
                                                                               std::span<const std::int32_t> subcases,
                                                                               base::MemoryBudget& budget)
{
    const input::ModelSnapshot& snapshot = index.snapshot();
    base::DiagnosticSink sink; // the analysis reports the problems of the constraints
    const std::vector<base::EntityId> asked(subcases.begin(), subcases.end());
    const std::vector<solutions::StaticSubcase> resolved = solutions::staticSubcasesOf(snapshot, asked, sink);
    const assembly::DofMap dofMap(snapshot.nodes(), index.points());
    const input::ConstraintTable* constraints =
        snapshot.hasTable(input::TableRole::Spc) ? &snapshot.constraints() : nullptr;
    std::set<std::int32_t> enforced;
    for (const solutions::StaticSubcase& subcase : resolved)
    {
        if (subcase.isCombination)
            continue;
        const assembly::SpcSet spc(constraints, snapshot.findLoads(), dofMap, subcase.spc, subcase.load, budget, sink);
        if (!spc.enforcedDisplacements().empty())
            enforced.insert(subcase.label);
    }
    return enforced;
}

} // namespace oos::design
