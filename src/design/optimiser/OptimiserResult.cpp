#include "oos/design/optimiser/OptimiserResult.hpp"

namespace oos::design
{

std::string_view optimiserTerminationName(OptimiserTermination termination) noexcept
{
    switch (termination)
    {
    case OptimiserTermination::ObjectiveConverged:
        return "objective_converged";
    case OptimiserTermination::KuhnTucker:
        return "kuhn_tucker";
    case OptimiserTermination::NoProgress:
        return "no_progress";
    case OptimiserTermination::IterationLimit:
        return "iteration_limit";
    case OptimiserTermination::Infeasible:
        return "infeasible";
    }
    return "";
}

} // namespace oos::design
