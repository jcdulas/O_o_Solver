#include "oos/design/aids/TrimHessian.hpp"

#include "RecoveredFieldCurvature.hpp"
#include "SizingMatrixRates.hpp"
#include "oos/base/errors/OptimisationError.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <stdexcept>

namespace oos::design
{
namespace
{

using CoefficientKey = std::pair<std::size_t, bool>; // variable, unrestrained

struct CaseWeights
{
    std::vector<double> variables;
    std::vector<detail::WeightedRecoveredField> fields;
    std::map<CoefficientKey, std::array<double, 6>> coefficients;
};

std::optional<std::map<std::size_t, CaseWeights>> trimWeights(
    const DesignProblem& problem, const DesignEvaluation& evaluation, const ProductWeights& weights)
{
    const auto& subcases = evaluation.analysis.trim->subcases;
    std::map<std::size_t, CaseWeights> result;
    const auto add = [&](std::int32_t id, std::size_t item, double weight) -> bool
    {
        const auto response = std::find_if(evaluation.responses.begin(), evaluation.responses.end(),
                                           [&](const auto& candidate) { return candidate.id == id; });
        if (response == evaluation.responses.end() || item >= response->itemCount())
            throw std::invalid_argument("trim Hessian weights refer to an unknown response item");
        if (weight == 0.0)
            return true;
        const auto* definition = problem.responses().findDesignResp1(id);
        if (!definition || definition->definition().mode)
            return false;
        const auto& label = response->items[item];
        const auto subcase = std::find_if(subcases.begin(), subcases.end(),
                                          [&](const auto& candidate) { return candidate.subcase == label.subcase; });
        if (subcase == subcases.end())
            return false;
        auto& selected = result[static_cast<std::size_t>(subcase - subcases.begin())];
        const auto type = definition->definition().type;
        if (type == ResponseType::TrimVariable || type == ResponseType::StabilityDerivative)
        {
            const auto variable = std::find_if(subcase->variables.begin(), subcase->variables.end(),
                [&](const auto& candidate) { return candidate.id == label.entity; });
            if (variable == subcase->variables.end())
                throw base::OptimisationError("trim Hessian variable is not in the selected trim case");
            const auto index = static_cast<std::size_t>(variable - subcase->variables.begin());
            if (type == ResponseType::TrimVariable)
            {
                if (selected.variables.empty())
                    selected.variables.resize(subcase->variables.size(), 0.0);
                selected.variables[index] += weight;
            }
            else
            {
                if (label.component < 1 || label.component > 6)
                    throw base::OptimisationError("trim Hessian stability component must be from 1 to 6");
                auto& coefficients = selected.coefficients[{index, label.layer == 0}];
                coefficients[static_cast<std::size_t>(label.component - 1)] += weight;
            }
            return true;
        }
        if (type != ResponseType::Displacement && type != ResponseType::Stress && type != ResponseType::Strain &&
            type != ResponseType::Force)
            return false;
        selected.fields.push_back({definition, label, weight, response->values[item]});
        return true;
    };
    for (const auto& [id, row] : weights.weights)
    {
        if (weights.sparseWeights.count(id))
            throw std::invalid_argument("a response cannot have both dense and sparse Hessian weights");
        const auto response = std::find_if(evaluation.responses.begin(), evaluation.responses.end(),
                                           [&](const auto& candidate) { return candidate.id == id; });
        if (response == evaluation.responses.end() || row.size() != response->itemCount())
            throw std::invalid_argument("trim Hessian weights have the wrong response item count");
        for (std::size_t item = 0; item < row.size(); ++item)
            if (!add(id, item, row[item]))
                return std::nullopt;
    }
    for (const auto& [id, row] : weights.sparseWeights)
    {
        if (row.rowStart.size() != 2 || row.rowStart[0] != 0 || row.rowStart[1] != row.items.size() ||
            row.values.size() != row.items.size())
            throw std::invalid_argument("trim Hessian sparse weights require one valid row");
        for (std::size_t entry = 0; entry < row.items.size(); ++entry)
            if (!add(id, row.items[entry], row.values[entry]))
                return std::nullopt;
    }
    return result;
}

double dot(std::span<const double> a, std::span<const double> b)
{
    double value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        value += a[i] * b[i];
    return value;
}

class TrimSecondOrder
{
public:
    TrimSecondOrder(const DesignProblem& problem, const DesignEvaluation& evaluation, base::MemoryBudget& budget,
                    base::ThreadPool* pool)
        : m_problem(problem), m_evaluation(evaluation), m_budget(budget), m_pool(pool),
          m_trim(*evaluation.analysis.trim->linearisation)
    {
    }

    std::optional<ProductReport> run(const std::map<std::size_t, CaseWeights>& weights,
                                     std::span<const double> directions, std::size_t count, std::vector<double>& values)
    {
        const auto start = std::chrono::steady_clock::now();
        base::DiagnosticSink sink;
        solutions::AnalysisModel model(*m_evaluation.analysis.snapshot, solutions::AnalysisKind::StaticAeroelastic,
                                        m_budget, m_pool, nullptr, sink);
        values.assign(m_problem.model().independentCount() * count, 0.0);
        for (const auto& [subcase, selected] : weights)
        {
            const auto label = m_evaluation.analysis.trim->subcases[subcase].subcase;
            const auto options = assembly::MassAssemblyOptions::fromParams(model.params(), label, sink);
            detail::SizingMatrixRates matrices(m_problem, m_evaluation, model, m_budget);
            if (!matrices.prepare(options))
                return std::nullopt;
            for (const auto& target : matrices.targets())
                if (target.use.changesDistributedLoads)
                    return std::nullopt;
            detail::RecoveredFieldCurvature recovery(model, m_evaluation.analysis.snapshot, m_budget);
            const auto base = m_trim.trimmedState(subcase);
            if (!recovery.prepare(selected.fields, matrices.targets(), base.displacements))
                return std::nullopt;
            if (!selected.variables.empty() || !selected.fields.empty())
                addState(subcase, std::nullopt, selected.variables, selected.fields.empty() ? nullptr : &recovery,
                           matrices, directions, count, values);
            for (const auto& [key, coefficientWeights] : selected.coefficients)
                addState(subcase, key, coefficientWeights, nullptr, matrices, directions, count, values);
        }
        m_report.usesTrimTangents = true;
        m_report.hessianBlocksReason = "analytic trim Hessian from differentiated coupled state equations";
        m_report.productSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return m_report;
    }

private:
    struct Tangent
    {
        std::vector<double> outputs;
        std::vector<double> displacements;
        std::vector<double> motions;
    };

    Tangent solve(std::size_t subcase, std::optional<CoefficientKey> coefficient,
                   std::span<const double> stiffness, std::span<const double> mass, bool withDisplacements)
    {
        // Recovery-only parameters and zero directions have no implicit state tangent.
        const auto isZero = [](double value) { return value == 0.0; };
        if (std::all_of(stiffness.begin(), stiffness.end(), isZero) &&
            std::all_of(mass.begin(), mass.end(), isZero))
        {
            const auto outputs = coefficient ? 6 : m_evaluation.analysis.trim->subcases[subcase].variables.size();
            const auto size = withDisplacements ? m_trim.gSetSize() : 0;
            return {std::vector<double>(outputs, 0.0), std::vector<double>(size, 0.0), std::vector<double>(size, 0.0)};
        }
        m_report.solves += withDisplacements ? 2 : 1;
        if (coefficient)
        {
            auto part = m_trim.coefficientStateDerivative(subcase, coefficient->first, coefficient->second,
                                                          stiffness, mass, withDisplacements, m_budget);
            return {{part.coefficients.begin(), part.coefficients.end()}, std::move(part.displacements),
                     std::move(part.motions)};
        }
        auto part = m_trim.stateDerivative(subcase, stiffness, mass, withDisplacements, m_budget);
        auto motions = withDisplacements ? m_trim.supportMotion(subcase, part.supportAccelerations)
                                         : std::vector<double>{};
        return {std::move(part.variables), std::move(part.displacements), std::move(motions)};
    }

    // For K u = aero - M w, differentiating twice gives the same coupled operator with loads
    // -(K_pv u + K_p u_v + K_v u_p), -(M_pv (w-a) + M_p w_v + M_v w_p).
    // The applied inertial acceleration a and the aerodynamic/support geometry are fixed. First tangents are
    // streamed by field: memory stays independent of the number of design variables, and every solve reuses the run.
    void addState(std::size_t subcase, std::optional<CoefficientKey> coefficient,
                   std::span<const double> outputWeights, const detail::RecoveredFieldCurvature* recovery,
                   const detail::SizingMatrixRates& matrices, std::span<const double> directions,
                   std::size_t count, std::vector<double>& values)
    {
        const auto base = coefficient ? m_trim.coefficientState(subcase, coefficient->first, coefficient->second,
                                                                 m_budget)
                                      : m_trim.trimmedState(subcase);
        if (coefficient)
            ++m_report.solves;
        const auto g = m_trim.gSetSize();
        const auto& quantities = m_evaluation.analysis.point.values().fields;
        auto applied = coefficient ? std::vector<double>{} : m_trim.appliedAcceleration(subcase, m_budget);
        base::BudgetedBuffer<double> motion(m_budget, g, "trim Hessian effective inertial motion");
        for (std::size_t i = 0; i < g; ++i)
            motion[i] = base.motions[i] - (applied.empty() ? 0.0 : applied[i]);
        base::BudgetedBuffer<double> stiffness(m_budget, g, "trim Hessian stiffness load");
        base::BudgetedBuffer<double> mass(m_budget, g, "trim Hessian mass load");
        // The largest live state is the base, the direction, one field and their mixed tangent.
        auto stateReservation = m_budget.reserve(4 * (3 * g + outputWeights.size()) * sizeof(double),
                                                  "trim Hessian streamed states");
        const auto clear = [&]
        {
            std::fill(stiffness.span().begin(), stiffness.span().end(), 0.0);
            std::fill(mass.span().begin(), mass.span().end(), 0.0);
        };
        for (std::size_t direction = 0; direction < count; ++direction)
        {
            base::BudgetedBuffer<double> moves(m_budget, quantities.size(), "trim Hessian field direction");
            for (std::size_t q = 0; q < quantities.size(); ++q)
            {
                moves[q] = 0.0;
                for (const auto& [column, slope] : quantities[q].gradient)
                    moves[q] += slope * directions[column * count + direction];
            }
            clear();
            matrices.addDirection(moves.span(), base.displacements, motion.span(), stiffness.span(), mass.span());
            const auto along = solve(subcase, coefficient, stiffness.span(), mass.span(), true);
            base::BudgetedBuffer<double> weightRate(m_budget, g, "trim recovered weight derivative");
            base::BudgetedBuffer<double> explicitRate(m_budget, quantities.size(), "trim recovered field curvature");
            std::fill(explicitRate.span().begin(), explicitRate.span().end(), 0.0);
            if (recovery)
            {
                recovery->weightRate(along.displacements, moves.span(), weightRate.span());
                recovery->addExplicitRate(along.displacements, moves.span(), explicitRate.span());
            }
            for (std::size_t q = 0; q < quantities.size(); ++q)
            {
                if (quantities[q].gradient.empty())
                    continue;
                clear();
                matrices.addFirst(q, base.displacements, motion.span(), stiffness.span(), mass.span());
                const auto first = solve(subcase, coefficient, stiffness.span(), mass.span(), true);
                clear();
                matrices.addMixed(q, moves.span(), base.displacements, motion.span(), stiffness.span(), mass.span());
                matrices.addFirst(q, along.displacements, along.motions, stiffness.span(), mass.span());
                matrices.addDirection(moves.span(), first.displacements, first.motions, stiffness.span(), mass.span());
                const auto mixed = solve(subcase, coefficient, stiffness.span(), mass.span(), recovery != nullptr);
                double curvature = dot(outputWeights, mixed.outputs);
                if (recovery)
                    curvature += dot(recovery->weights(), mixed.displacements) +
                                 dot(weightRate.span(), first.displacements) + explicitRate[q];
                for (const auto& [column, slope] : quantities[q].gradient)
                    values[column * count + direction] += slope * curvature;
            }
        }
    }

    const DesignProblem& m_problem;
    const DesignEvaluation& m_evaluation;
    base::MemoryBudget& m_budget;
    base::ThreadPool* m_pool;
    solutions::TrimLinearisation& m_trim;
    ProductReport m_report;
};

} // namespace

std::optional<ProductReport> trimHessianVector(
    const DesignProblem& problem, const DesignEvaluation& evaluation, const ProductWeights& weights,
    std::span<const double> directions, std::size_t count, std::vector<double>& values,
    base::MemoryBudget& budget, base::ThreadPool* pool)
{
    if (weights.rows != 1 || count == 0 || directions.size() != problem.model().independentCount() * count)
        throw std::invalid_argument("trim Hessian requires one weight row and one value per variable and direction");
    if (!evaluation.analysis.trim || !evaluation.analysis.trim->linearisation ||
        evaluation.analysis.trim->path != solutions::AnalysisPath::Full ||
        !evaluation.analysis.point.values().shapes.empty())
        return std::nullopt;
    const auto selected = trimWeights(problem, evaluation, weights);
    if (!selected)
        return std::nullopt;
    return TrimSecondOrder(problem, evaluation, budget, pool).run(*selected, directions, count, values);
}

} // namespace oos::design
