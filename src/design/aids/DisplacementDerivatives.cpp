#include "oos/design/aids/DisplacementDerivatives.hpp"

#include "../sensitivity/SensitivityComputation.hpp"

#include "oos/base/errors/OutputVolumeError.hpp"
#include "oos/linalg/dense/DenseMatrix.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace oos::design
{

namespace
{

constexpr std::uint64_t DOUBLE_BYTES = sizeof(double);

[[nodiscard]] std::uint64_t multiplySaturated(std::uint64_t a, std::uint64_t b) noexcept
{
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
        return std::numeric_limits<std::uint64_t>::max();
    return a * b;
}

// The direct solves of the variables, block by block (StaticSensitivity's pseudo-loads, then K⁻¹ on the factor of
// each constraint set).
class DisplacementComputation : public detail::SensitivityComputation
{
public:
    DisplacementComputation(const StaticSensitivityInputs& inputs, const GradientOptions& options,
                            base::MemoryBudget& budget)
        : detail::SensitivityComputation(inputs, options, budget)
    {
        m_preparesEverySubcase = true;
    }

    DisplacementDerivativeReport run(std::span<const std::size_t> variables, std::size_t block,
                                     DisplacementDerivativeSink& sink)
    {
        prepareSubcases();
        differenceFields();
        DisplacementDerivativeReport report;
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            const SensitivitySystem& system = systemOf(m_subcases[s]);
            const auto equations = static_cast<std::size_t>(system.equationCount());
            for (std::size_t first = 0; first < variables.size(); first += block)
            {
                const std::size_t count = std::min(block, variables.size() - first);
                base::BudgetedBuffer<double> values(m_budget, count * gSize(), "displacement derivatives block");
                linalg::DenseMatrix rightHandSides(m_budget, equations, count, "displacement derivatives solves");
                std::vector<double> loads(gSize());
                std::vector<double> column(equations);
                for (std::size_t b = 0; b < count; ++b)
                {
                    m_pseudoLoads->assemble(variables[first + b], s, loads);
                    system.toEquations(loads, column);
                    for (std::size_t e = 0; e < equations; ++e)
                        rightHandSides(e, b) = column[e];
                }
                if (equations > 0)
                    system.solve(rightHandSides.view());
                report.solves += static_cast<std::int64_t>(count);
                for (std::size_t b = 0; b < count; ++b)
                {
                    for (std::size_t e = 0; e < equations; ++e)
                        column[e] = rightHandSides(e, b);
                    system.fromEquations(column, values.span().subspan(b * gSize(), gSize()));
                }
                sink.consume({m_subcases[s].subcase.label, first, count, gSize(), values.span()});
                ++report.blocks;
            }
        }
        for (const std::unique_ptr<SensitivitySystem>& system : m_systems)
            report.factorisations += system->factorisationCount();
        return report;
    }
};

} // namespace

std::vector<std::pair<std::int32_t, std::int32_t>> DisplacementDerivatives::labels(const DesignProblem& problem)
{
    base::MemoryBudget budget(base::MemoryBudget::UNLIMITED);
    const elements::SnapshotIndex index(problem.snapshot(), budget);
    const assembly::DofMap dofMap(problem.snapshot().nodes(), index.points());
    std::vector<std::pair<std::int32_t, std::int32_t>> labels;
    for (base::DofIndex dof = 0; dof < dofMap.gSetSize(); ++dof)
    {
        const assembly::DofLabel label = dofMap.labelOf(dof);
        labels.emplace_back(dofMap.pointIdOf(dof), label.kind == input::PointKind::Scalar ? 0 : label.component);
    }
    return labels;
}

DisplacementDerivativeSize DisplacementDerivatives::size(const DesignProblem& problem,
                                                         const DesignEvaluation& evaluation, std::size_t variableCount,
                                                         std::size_t block)
{
    (void)problem;
    DisplacementDerivativeSize size;
    size.subcaseCount = evaluation.analysis.subcases.size();
    size.variableCount = variableCount;
    size.gSize = static_cast<std::size_t>(evaluation.analysis.results.gSetSize);
    size.bytes = multiplySaturated(
        multiplySaturated(static_cast<std::uint64_t>(size.subcaseCount), static_cast<std::uint64_t>(variableCount)),
        multiplySaturated(static_cast<std::uint64_t>(size.gSize), DOUBLE_BYTES));
    // A block holds its g-set vectors and the right-hand sides of its solves (at most g rows each), and two working
    // vectors.
    const std::uint64_t columns = std::min<std::uint64_t>(block, variableCount);
    size.blockPeakBytes = multiplySaturated(multiplySaturated(2, columns) + 2,
                                            multiplySaturated(static_cast<std::uint64_t>(size.gSize), DOUBLE_BYTES));
    return size;
}

void DisplacementDerivatives::requireDestination(const DisplacementDerivativeSize& size, std::string_view destination,
                                                 std::uint64_t capacityBytes)
{
    if (size.bytes <= capacityBytes)
        return;
    std::vector<std::string> remedies{
        "give sink= a callable, which receives the derivatives block by block and holds none of them",
        "ask fewer variables (variables=)", "ask fewer subcases (DesignProblem(subcases=...))"};
    throw base::OutputVolumeError(
        base::concatenate({"the displacement derivatives (", std::to_string(size.variableCount), " variables × ",
                           std::to_string(size.gSize), " DOFs × ", std::to_string(size.subcaseCount), " subcases, ",
                           base::formatBytes(size.bytes), ") do not fit their destination, ", destination,
                           ", which can take ", base::formatBytes(capacityBytes),
                           "; nothing was computed. Remedies: ", remedies[0], "; ", remedies[1], "; ", remedies[2]}),
        size.bytes, remedies);
}

DisplacementDerivativeReport DisplacementDerivatives::stream(const DesignProblem& problem,
                                                             const DesignEvaluation& evaluation,
                                                             std::span<const std::size_t> variables, std::size_t block,
                                                             DisplacementDerivativeSink& sink,
                                                             base::MemoryBudget& budget,
                                                             const solutions::StaticOptions& analysisOptions)
{
    if (block == 0)
        throw std::invalid_argument(
            "displacement_derivatives: expected a block of at least one variable, found 0; give block= a positive "
            "number (256 by default)");
    std::vector<std::size_t> columns(variables.begin(), variables.end());
    if (columns.empty())
        for (std::size_t j = 0; j < problem.model().independentCount(); ++j)
            columns.push_back(j);
    for (const std::size_t j : columns)
        if (j >= problem.model().independentCount())
            throw std::invalid_argument(
                base::concatenate({"displacement_derivatives: expected variables below ",
                                   std::to_string(problem.model().independentCount()),
                                   " (the independent design variables), found ", std::to_string(j)}));
    const ResponseSubcases subcases = problem.responseSubcases(evaluation);
    const CallableRegistry noCallables;
    const StaticSensitivityInputs inputs =
        problem.sensitivityInputs(evaluation, subcases, noCallables, analysisOptions, EQUATION_BATCH_ITEMS);
    const GradientOptions options;
    DisplacementComputation computation(inputs, options, budget);
    DisplacementDerivativeReport report = computation.run(columns, block, sink);
    report.size = size(problem, evaluation, columns.size(), block);
    return report;
}

} // namespace oos::design
