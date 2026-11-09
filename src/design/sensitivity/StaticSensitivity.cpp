#include "oos/design/sensitivity/StaticSensitivity.hpp"

#include "SensitivityComputation.hpp"

namespace oos::design
{
DesignGradients StaticSensitivity::compute(const StaticSensitivityInputs& inputs, const GradientOptions& options,
                                           base::MemoryBudget& budget)
{
    detail::SensitivityComputation computation(inputs, options, budget);
    return computation.run();
}

GradientStreamReport StaticSensitivity::stream(const StaticSensitivityInputs& inputs,
                                               const GradientStreamOptions& options, GradientSink& sink,
                                               base::MemoryBudget& budget)
{
    // Every response, every item (no mask): the rows of G. The method is the stream's (its orientation).
    GradientOptions linearisation;
    linearisation.method = options.method;
    detail::SensitivityComputation computation(inputs, linearisation, budget);
    return computation.stream(options, sink);
}

} // namespace oos::design
