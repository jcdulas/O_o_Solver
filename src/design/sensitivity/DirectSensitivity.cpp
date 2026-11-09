#include "oos/design/sensitivity/DirectSensitivity.hpp"

#include "oos/design/sensitivity/SolutionReads.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <algorithm>
#include <vector>

namespace oos::design
{

std::int64_t DirectSensitivity::solve(const SubcaseSolve& context, std::span<const ItemKey> items,
                                      LinearisedItems& linearised)
{
    const SensitivitySystem& system = *context.system;
    const PseudoLoads& loads = *context.loads;
    std::vector<std::size_t> variables; // bounded by the design variables
    for (std::size_t j = 0; j < loads.variableCount(); ++j)
        if (loads.hasLoad(j, context.subcase))
            variables.push_back(j);
    if (variables.empty() || items.empty())
        return 0;
    const auto gSize = static_cast<std::size_t>(system.dofMap().gSetSize());
    const auto equations = static_cast<std::size_t>(system.equationCount());
    base::BudgetedBuffer<double> vectorG(*context.budget, gSize, "direct sensitivity pseudo-load");
    base::BudgetedBuffer<double> column(*context.budget, equations, "direct sensitivity equations");
    // The condensed path recovers the units the items read only.
    const base::BudgetedBuffer<std::uint8_t> reads =
        SolutionReads::ofWeights(system, linearised, items, *context.budget);
    std::int64_t solves = 0;
    for (std::size_t first = 0; first < variables.size(); first += SENSITIVITY_BLOCK_COLUMNS)
    {
        const std::size_t count = std::min(SENSITIVITY_BLOCK_COLUMNS, variables.size() - first);
        linalg::DenseMatrix rightHandSides(*context.budget, equations, count, "direct sensitivity right-hand sides");
        for (std::size_t b = 0; b < count; ++b)
        {
            loads.assemble(variables[first + b], context.subcase, vectorG.span());
            system.toEquations(vectorG.span(), column.span());
            for (std::size_t e = 0; e < equations; ++e)
                rightHandSides(e, b) = column[e];
        }
        if (equations > 0)
            system.solve(rightHandSides.view(), reads.span());
        solves += static_cast<std::int64_t>(count);
        for (std::size_t b = 0; b < count; ++b)
        {
            for (std::size_t e = 0; e < equations; ++e)
                column[e] = rightHandSides(e, b);
            system.fromEquations(column.span(), vectorG.span());
            const std::size_t j = variables[first + b];
            for (const ItemKey& key : items)
            {
                const ItemLinearisation& item = linearised.at(key);
                if (item.readsDisplacements())
                    linearised.gradientOf(key)[j] += dotSparse(item.weights, vectorG.span());
            }
        }
    }
    return solves;
}

} // namespace oos::design
