#include "oos/design/optimiser/Convergence.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/OptimisationError.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace oos::design
{

namespace
{

// |b − a| / |a|, or |b − a| when a = 0 (no relative scale).
double relativeChange(double before, double after)
{
    const double change = std::abs(after - before);
    return before != 0.0 ? change / std::abs(before) : change;
}

double largestRelativeChange(const std::vector<double>& before, const std::vector<double>& after, std::string_view what,
                             std::size_t cycle)
{
    if (before.size() != after.size())
        throw base::OptimisationError(
            base::concatenate({"convergence: expected the same number of ", what, " in every cycle, found ",
                               std::to_string(before.size()), " in cycle ", std::to_string(cycle - 1), " and ",
                               std::to_string(after.size()), " in cycle ", std::to_string(cycle),
                               "; give every design cycle the same variables and constraints"}));
    double largest = 0.0;
    for (std::size_t i = 0; i < before.size(); ++i)
        largest = std::max(largest, relativeChange(before[i], after[i]));
    return largest;
}

} // namespace

ConvergenceCriteria ConvergenceCriteria::fromParameters(const OptimisationParameters& parameters)
{
    ConvergenceCriteria criteria;
    criteria.relativeObjectiveChange = parameters.value(DesignOptimParams::RelativeObjectiveConvergence);
    if (parameters.isSet(DesignOptimParams::AbsoluteObjectiveConvergence))
        criteria.absoluteObjectiveChange = parameters.value(DesignOptimParams::AbsoluteObjectiveConvergence);
    criteria.feasibilityTolerance = parameters.value(DesignOptimParams::MaxConstraintViolation);
    criteria.maximumCycles = parameters.integer(DesignOptimParams::MaxDesignCycles);
    return criteria;
}

double ConvergenceCriteria::absoluteChange(double initialObjective) const noexcept
{
    return absoluteObjectiveChange ? *absoluteObjectiveChange : optimiser_defaults::absoluteChangeOf(initialObjective);
}

std::string_view convergenceKindName(ConvergenceKind kind) noexcept
{
    switch (kind)
    {
    case ConvergenceKind::NotConverged:
        return "not_converged";
    case ConvergenceKind::Feasible:
        return "feasible";
    case ConvergenceKind::Infeasible:
        return "infeasible";
    case ConvergenceKind::DesignCycleLimit:
        return "design_cycle_limit";
    }
    return "";
}

ConvergenceCheck Convergence::check(const DesignCycle& previous, const DesignCycle& current, std::size_t cycle,
                                    double initialObjective, const ConvergenceCriteria& criteria)
{
    ConvergenceCheck check;
    check.cycle = cycle;
    check.absoluteObjectiveChange = std::abs(current.objective - previous.objective);
    check.relativeObjectiveChange = relativeChange(previous.objective, current.objective);
    check.maxConstraint = current.maxConstraint;
    check.maxRelativeVariableChange =
        largestRelativeChange(previous.variables, current.variables, "design variables", cycle);
    check.maxRelativePropertyChange =
        largestRelativeChange(previous.properties, current.properties, "designed properties", cycle);
    check.isObjectiveSteady = check.absoluteObjectiveChange <= criteria.absoluteChange(initialObjective) ||
                              check.relativeObjectiveChange <= criteria.relativeObjectiveChange;
    check.isFeasible = current.maxConstraint <= criteria.feasibilityTolerance;
    return check;
}

ConvergenceDecision Convergence::review(std::span<const DesignCycle> history, const ConvergenceCriteria& criteria)
{
    ConvergenceDecision decision;
    if (history.empty())
        return decision;
    const double initialObjective = history.front().objective;
    std::size_t steadyFeasible = 0;
    std::size_t steadyInfeasible = 0;
    for (std::size_t cycle = 1; cycle < history.size(); ++cycle)
    {
        ConvergenceCheck check =
            Convergence::check(history[cycle - 1], history[cycle], cycle, initialObjective, criteria);
        // A cycle that is not steady, or whose feasibility differs, starts the counts again (CONMIN resets its
        // counts of the objective test at an infeasible design).
        steadyFeasible = check.isObjectiveSteady && check.isFeasible ? steadyFeasible + 1 : 0;
        steadyInfeasible = check.isObjectiveSteady && !check.isFeasible ? steadyInfeasible + 1 : 0;
        check.steadyFeasibleCycles = steadyFeasible;
        check.steadyInfeasibleCycles = steadyInfeasible;
        decision.checks.push_back(check);
        if (decision.kind != ConvergenceKind::NotConverged)
            continue;
        if (steadyFeasible >= criteria.consecutiveCycles)
            decision.kind = ConvergenceKind::Feasible;
        else if (steadyInfeasible >= criteria.infeasibleConsecutiveCycles)
            decision.kind = ConvergenceKind::Infeasible;
        else if (static_cast<std::int64_t>(cycle) >= criteria.maximumCycles)
            decision.kind = ConvergenceKind::DesignCycleLimit;
        if (decision.kind != ConvergenceKind::NotConverged)
            decision.cycle = cycle;
    }
    return decision;
}

} // namespace oos::design
