#include "oos/design/optimiser/Optimiser.hpp"

#include "oos/design/optimiser/FeasibleDirections.hpp"
#include "oos/design/optimiser/SequentialLinear.hpp"
#include "oos/design/optimiser/SequentialQuadratic.hpp"

namespace oos::design
{

OptimiserMethod Optimiser::resolve(OptimiserMethod method) noexcept
{
    return method == OptimiserMethod::Automatic ? OptimiserMethod::FeasibleDirections : method;
}

OptimiserResult Optimiser::run(const OptimisationProblem& problem, std::span<const double> start,
                               const OptimiserSettings& settings, base::MemoryBudget& budget)
{
    switch (resolve(settings.method))
    {
    case OptimiserMethod::SequentialLinear:
        return SequentialLinear::run(problem, start, settings, budget);
    case OptimiserMethod::SequentialQuadratic:
        return SequentialQuadratic::run(problem, start, settings, budget);
    default:
        return FeasibleDirections::run(problem, start, settings, budget);
    }
}

} // namespace oos::design
