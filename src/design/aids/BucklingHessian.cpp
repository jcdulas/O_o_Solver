#include "oos/design/aids/BucklingHessian.hpp"

#include "../sensitivity/ModalSystem.hpp"
#include "DifferentialMatrixRates.hpp"
#include "SizingMatrixRates.hpp"
#include "oos/assembly/assemble/DifferentialStiffnessAssembler.hpp"
#include "oos/assembly/loads/LoadVectorBuilder.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/linalg/eigen/BorderedEigenSystem.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>

namespace oos::design
{
namespace
{

using RootKey = std::pair<std::size_t, std::size_t>; // solve, tracked mode

std::optional<std::map<RootKey, double>> bucklingWeights(
    const DesignProblem& problem, const DesignEvaluation& evaluation, const ProductWeights& weights)
{
    std::map<RootKey, double> result;
    const auto add = [&](std::int32_t id, std::size_t item, double weight) -> bool
    {
        const auto response = std::find_if(evaluation.responses.begin(), evaluation.responses.end(),
                                           [&](const auto& candidate) { return candidate.id == id; });
        if (response == evaluation.responses.end() || item >= response->itemCount())
            throw std::invalid_argument("buckling Hessian weights refer to an unknown response item");
        if (weight == 0.0)
            return true;
        const auto* definition = problem.responses().findDesignResp1(id);
        if (!definition || definition->definition().type != ResponseType::BucklingFactor)
            return false;
        const auto& label = response->items[item];
        const auto& solves = evaluation.analysis.buckling->solves;
        for (std::size_t s = 0; s < solves.size(); ++s)
            if (solves[s].subcase == label.subcase && label.layer > 0)
            {
                const auto mode = static_cast<std::size_t>(label.layer - 1);
                if (mode >= solves[s].eigenvalues.rows.size())
                    throw base::OptimisationError("buckling Hessian mode is outside the extracted roots");
                result[{s, mode}] += weight;
                return true;
            }
        return false;
    };
    for (const auto& [id, row] : weights.weights)
    {
        if (weights.sparseWeights.count(id))
            throw std::invalid_argument("a response cannot have both dense and sparse Hessian weights");
        const auto response = std::find_if(evaluation.responses.begin(), evaluation.responses.end(),
                                           [&](const auto& candidate) { return candidate.id == id; });
        if (response == evaluation.responses.end() || row.size() != response->itemCount())
            throw std::invalid_argument("buckling Hessian weights have the wrong response item count");
        for (std::size_t item = 0; item < row.size(); ++item)
            if (!add(id, item, row[item]))
                return std::nullopt;
    }
    for (const auto& [id, row] : weights.sparseWeights)
    {
        if (row.rowStart.size() != 2 || row.rowStart[0] != 0 || row.rowStart[1] != row.items.size() ||
            row.values.size() != row.items.size())
            throw std::invalid_argument("buckling Hessian sparse weights require one valid row");
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

class BucklingSecondOrder
{
public:
    BucklingSecondOrder(const DesignProblem& problem, const DesignEvaluation& evaluation,
                        base::MemoryBudget& budget, linalg::KernelKind kernel, base::ThreadPool* pool)
        : m_problem(problem), m_evaluation(evaluation), m_budget(budget), m_pool(pool),
          m_kernels(kernel, linalg::DenseKernelOptions{pool})
    {
    }

    std::optional<ProductReport> run(const std::map<RootKey, double>& weights, std::span<const double> directions,
                                     std::size_t count, std::vector<double>& values)
    {
        const auto start = std::chrono::steady_clock::now();
        base::DiagnosticSink sink;
        solutions::AnalysisModel model(*m_evaluation.analysis.snapshot, solutions::AnalysisKind::Buckling,
                                        m_budget, m_pool, nullptr, sink);
        detail::SizingMatrixRates matrices(m_problem, m_evaluation, model, m_budget);
        // The current closed preload path keeps applied loads fixed and does not differentiate M.
        if (!matrices.prepare({}, false))
            return std::nullopt;
        const bool changesDistributed = std::any_of(matrices.targets().begin(), matrices.targets().end(),
            [](const auto& target) { return target.use.changesDistributedLoads; });
        detail::DifferentialMatrixRates differential(model, matrices.targets());
        assembly::LoadVectorBuilder loads(model.elementSet(), model.dofMap(), m_budget);
        values.assign(m_problem.model().independentCount() * count, 0.0);
        for (std::size_t s = 0; s < m_evaluation.analysis.buckling->solves.size(); ++s)
        {
            if (std::none_of(weights.begin(), weights.end(), [&](const auto& item) { return item.first.first == s; }))
                continue;
            const auto& solve = m_evaluation.analysis.buckling->solves[s];
            const solutions::FactorisedSet* preloadSystem = nullptr;
            for (const auto& system : m_evaluation.analysis.results.systems)
                if (system->solves(solve.preloadSubcase) && !system->relief())
                    preloadSystem = system.get();
            std::span<const double> preload;
            for (const auto& kept : m_evaluation.analysis.results.keptDisplacements)
                if (kept.subcase == solve.preloadSubcase)
                    preload = kept.values.span();
            if (!preloadSystem || preload.empty() || !differential.prepare(preload))
                return std::nullopt;
            const auto cases = solutions::staticSubcasesOf(*m_evaluation.analysis.snapshot,
                                                           {solve.preloadSubcase}, sink);
            for (const auto& selected : cases)
            {
                if (selected.isCombination || selected.hasInitialStrains())
                    return std::nullopt;
                if (!selected.load)
                    continue;
                const bool movesDistributed = changesDistributed &&
                    loads.hasFamilyLoads(*selected.load, assembly::LoadFamily::Distributed);
                if (loads.needsMass(*selected.load) || movesDistributed)
                    return std::nullopt;
            }
            const detail::ModalSystem pencil(model, solve.subcase, m_budget, m_pool);
            linalg::BlockSparseMatrix dgg(pencil.constraints().pattern(), m_budget, "buckling Hessian Kdgg");
            assembly::DifferentialStiffnessAssembler(model.elementSet(), model.dofMap()).assemble(
                preload, nullptr, dgg, m_pool);
            const auto dnn = pencil.constraints().reduce(dgg, m_budget, "buckling Hessian Kdnn");
            const auto d = linalg::SymmetricCsc::fromBlocks(dnn, pencil.selection(), m_budget, m_pool);
            for (const auto& [key, weight] : weights)
                if (key.first == s)
                    addRoot(solve, key.second, weight, *preloadSystem, preload, pencil, d, matrices, differential,
                             directions, count, values);
        }
        m_report.usesBucklingTangents = true;
        m_report.hessianBlocksReason = "analytic buckling Hessian including first and mixed preload tangents";
        m_report.productSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return m_report;
    }

private:
    std::vector<double> solvePreload(const solutions::FactorisedSet& system, std::span<const double> load)
    {
        if (std::all_of(load.begin(), load.end(), [](double value) { return value == 0.0; }))
            return std::vector<double>(load.size(), 0.0);
        std::vector<double> result(load.begin(), load.end());
        system.gm().reduceToN(result);
        const auto n = static_cast<std::size_t>(system.equationCount());
        linalg::DenseMatrix right(m_budget, n, 1, "buckling preload tangent");
        for (std::size_t i = 0; i < n; ++i)
            right(i, 0) = result[static_cast<std::size_t>(system.dofOfEquation(static_cast<std::int32_t>(i)))];
        system.solve(right.view(), m_kernels);
        ++m_report.solves;
        std::fill(result.begin(), result.end(), 0.0);
        for (std::size_t i = 0; i < n; ++i)
            result[static_cast<std::size_t>(system.dofOfEquation(static_cast<std::int32_t>(i)))] = right(i, 0);
        system.gm().expandToG(result);
        return result;
    }

    static void clear(base::BudgetedBuffer<double>& vector)
    {
        std::fill(vector.span().begin(), vector.span().end(), 0.0);
    }

    // Invert the eigenvalue: Kd phi = mu K phi, mu = -1/lambda. K is a positive normalisation even for negative
    // buckling factors. The simple symmetric eigenpair formula then applies with K as its metric matrix.
    void addRoot(const solutions::BucklingSolve& solve, std::size_t mode, double weight,
                  const solutions::FactorisedSet& preloadSystem, std::span<const double> preload,
                  const detail::ModalSystem& pencil, const linalg::SymmetricCsc& differentialMatrix,
                  const detail::SizingMatrixRates& matrices, const detail::DifferentialMatrixRates& differential,
                  std::span<const double> directions, std::size_t count, std::vector<double>& values)
    {
        const double lambda = solve.eigenvalues.rows[mode].eigenvalue;
        if (lambda == 0.0)
            throw base::OptimisationError("buckling Hessians require a nonzero simple load factor");
        constexpr double REPEATED_ROOT_TOLERANCE = 1e-6;
        for (std::size_t other = 0; other < solve.eigenvalues.rows.size(); ++other)
        {
            const double neighbour = solve.eigenvalues.rows[other].eigenvalue;
            if (other != mode && std::abs(neighbour - lambda) <= REPEATED_ROOT_TOLERANCE *
                std::max({std::abs(lambda), std::abs(neighbour), 1e-12}))
                throw base::OptimisationError("a buckling Hessian is not unique at a repeated load factor");
        }
        const double mu = -1.0 / lambda;
        const auto& selection = pencil.selection();
        const auto n = static_cast<std::size_t>(selection.keptDofCount());
        const auto g = preload.size();
        const auto& quantities = m_evaluation.analysis.point.values().fields;
        base::BudgetedBuffer<double> shape(m_budget, g, "buckling Hessian normalised shape");
        std::copy_n(solve.eigenvectors.data() + mode * solve.eigenvectors.leadingDimension(), g, shape.data());
        base::BudgetedBuffer<double> freeShape(m_budget, n, "buckling Hessian free shape");
        base::BudgetedBuffer<double> metricShape(m_budget, n, "buckling Hessian metric shape");
        for (std::size_t i = 0; i < n; ++i)
            freeShape[i] = shape[static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(i)))];
        pencil.stiffness().multiply(freeShape.span(), metricShape.span());
        base::BudgetedBuffer<double> residual(m_budget, n, "buckling Hessian eigenpair residual");
        differentialMatrix.multiply(freeShape.span(), residual.span());
        double error = 0.0, residualScale = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            error = std::max(error, std::abs(residual[i] - mu * metricShape[i]));
            residualScale = std::max(residualScale, std::abs(residual[i]) + std::abs(mu * metricShape[i]));
        }
        if (error > 1e-6 * std::max(residualScale, 1e-20))
            throw base::OptimisationError("buckling Hessian requires a converged full-model buckling shape");
        const double norm = dot(freeShape.span(), metricShape.span());
        if (!(norm > 0.0))
            throw base::OptimisationError("buckling Hessian requires a mode with positive generalised stiffness");
        for (double& value : shape.span())
            value /= std::sqrt(norm);
        for (std::size_t i = 0; i < n; ++i)
        {
            freeShape[i] /= std::sqrt(norm);
            metricShape[i] /= std::sqrt(norm);
        }
        const linalg::BorderedEigenSystem bordered(differentialMatrix, pencil.stiffness(), mu, freeShape.span(),
                                                    m_kernels, m_budget);
        ++m_report.factorisations;
        base::BudgetedBuffer<double> stiffness(m_budget, g, "buckling Hessian stiffness action");
        base::BudgetedBuffer<double> work(m_budget, g, "buckling Hessian geometric action");
        base::BudgetedBuffer<double> unused(m_budget, g, "buckling Hessian unused mass action");
        base::BudgetedBuffer<double> gradient(m_budget, quantities.size(), "inverse buckling root gradient");
        base::BudgetedBuffer<double> metricRates(m_budget, quantities.size(), "buckling metric rates");
        auto stateReservation = m_budget.reserve(3 * g * sizeof(double), "buckling streamed preload tangents");
        // mu_p = phi^T (Kd_p - mu K_p) phi, including the first preload tangent.
        for (std::size_t q = 0; q < quantities.size(); ++q)
        {
            clear(stiffness);
            matrices.addFirst(q, preload, preload, stiffness.span(), unused.span());
            const auto preloadRate = solvePreload(preloadSystem, stiffness.span());
            clear(stiffness);
            clear(work);
            matrices.addFirst(q, shape.span(), shape.span(), stiffness.span(), unused.span(), 1.0);
            differential.addField(q, preload, shape.span(), work.span());
            differential.addPreload(preloadRate, shape.span(), work.span());
            metricRates[q] = dot(shape.span(), stiffness.span());
            gradient[q] = dot(shape.span(), work.span()) - mu * metricRates[q];
        }
        for (std::size_t direction = 0; direction < count; ++direction)
        {
            base::BudgetedBuffer<double> moves(m_budget, quantities.size(), "buckling field direction");
            for (std::size_t q = 0; q < quantities.size(); ++q)
            {
                moves[q] = 0.0;
                for (const auto& [column, slope] : quantities[q].gradient)
                    moves[q] += slope * directions[column * count + direction];
            }
            const double muRate = dot(gradient.span(), moves.span());
            const double metricRate = dot(metricRates.span(), moves.span());
            clear(stiffness);
            matrices.addDirection(moves.span(), preload, preload, stiffness.span(), unused.span());
            const auto preloadAlong = solvePreload(preloadSystem, stiffness.span());
            clear(stiffness);
            clear(work);
            matrices.addDirection(moves.span(), shape.span(), shape.span(), stiffness.span(), unused.span(), 1.0);
            differential.addDirection(moves.span(), preload, shape.span(), work.span());
            differential.addPreload(preloadAlong, shape.span(), work.span());
            for (std::size_t i = 0; i < g; ++i)
                work[i] = -work[i] + mu * stiffness[i];
            pencil.constraints().gm().reduceToN(work.span());
            base::BudgetedBuffer<double> right(m_budget, n, "buckling shape tangent load");
            base::BudgetedBuffer<double> freeTangent(m_budget, n, "buckling free shape tangent");
            for (std::size_t i = 0; i < n; ++i)
                right[i] = work[static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(i)))] +
                           muRate * metricShape[i];
            (void)bordered.solve(right.span(), freeTangent.span(), -0.5 * metricRate);
            ++m_report.solves;
            base::BudgetedBuffer<double> tangent(m_budget, g, "buckling expanded shape tangent");
            clear(tangent);
            for (std::size_t i = 0; i < n; ++i)
            {
                const auto dof = static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(i)));
                tangent[dof] = freeTangent[i];
            }
            pencil.constraints().gm().expandToG(tangent.span());
            for (std::size_t q = 0; q < quantities.size(); ++q)
            {
                if (quantities[q].gradient.empty())
                    continue;
                clear(stiffness);
                matrices.addFirst(q, preload, preload, stiffness.span(), unused.span());
                const auto preloadRate = solvePreload(preloadSystem, stiffness.span());
                clear(stiffness);
                matrices.addMixed(q, moves.span(), preload, preload, stiffness.span(), unused.span());
                matrices.addFirst(q, preloadAlong, preloadAlong, stiffness.span(), unused.span());
                matrices.addDirection(moves.span(), preloadRate, preloadRate, stiffness.span(), unused.span());
                const auto preloadMixed = solvePreload(preloadSystem, stiffness.span());
                clear(stiffness);
                clear(work);
                matrices.addFirst(q, shape.span(), shape.span(), stiffness.span(), unused.span(), 1.0);
                differential.addField(q, preload, shape.span(), work.span());
                differential.addPreload(preloadRate, shape.span(), work.span());
                const double tangentPart = 2.0 * (dot(tangent.span(), work.span()) -
                                                  mu * dot(tangent.span(), stiffness.span()));
                clear(stiffness);
                clear(work);
                matrices.addMixed(q, moves.span(), shape.span(), shape.span(), stiffness.span(), unused.span(), 1.0);
                differential.addMixed(q, moves.span(), preload, shape.span(), work.span());
                differential.addField(q, preloadAlong, shape.span(), work.span());
                differential.addDirection(moves.span(), preloadRate, shape.span(), work.span());
                differential.addPreload(preloadMixed, shape.span(), work.span());
                const double muMixed = tangentPart + dot(shape.span(), work.span()) -
                                       mu * dot(shape.span(), stiffness.span()) - muRate * metricRates[q];
                const double curvature = weight * (muMixed / (mu * mu) -
                                                    2.0 * gradient[q] * muRate / (mu * mu * mu));
                for (const auto& [column, slope] : quantities[q].gradient)
                    values[column * count + direction] += slope * curvature;
            }
        }
    }

    const DesignProblem& m_problem;
    const DesignEvaluation& m_evaluation;
    base::MemoryBudget& m_budget;
    base::ThreadPool* m_pool;
    linalg::DenseKernels m_kernels;
    ProductReport m_report;
};

} // namespace

std::optional<ProductReport> bucklingHessianVector(
    const DesignProblem& problem, const DesignEvaluation& evaluation, const ProductWeights& weights,
    std::span<const double> directions, std::size_t count, std::vector<double>& values,
    base::MemoryBudget& budget, linalg::KernelKind kernel, base::ThreadPool* pool)
{
    if (weights.rows != 1 || count == 0 || directions.size() != problem.model().independentCount() * count)
        throw std::invalid_argument(
            "buckling Hessian requires one weight row and one value per variable and direction");
    if (!evaluation.analysis.buckling || evaluation.analysis.buckling->path != solutions::AnalysisPath::Full ||
        !evaluation.analysis.point.values().shapes.empty())
        return std::nullopt;
    if (!evaluation.trackingWarnings.empty())
        throw std::invalid_argument("buckling Hessian requires unambiguous mode tracking");
    const auto selected = bucklingWeights(problem, evaluation, weights);
    if (!selected)
        return std::nullopt;
    return BucklingSecondOrder(problem, evaluation, budget, kernel, pool).run(*selected, directions, count, values);
}

} // namespace oos::design
