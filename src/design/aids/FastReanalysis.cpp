#include "oos/design/aids/FastReanalysis.hpp"

#include "../sensitivity/SensitivityComputation.hpp"
#include "oos/assembly/assemble/StiffnessAssembler.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/output/arrays/ElementResultBuilder.hpp"
#include "oos/output/arrays/NodalResultBuilder.hpp"
#include "oos/output/arrays/OutputPlan.hpp"
#include "oos/output/arrays/StiffnessRecovery.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace oos::design
{

namespace
{

// A new basis vector whose norm falls below this share of its norm before the orthogonalisation adds nothing the basis
// does not span (the series has stagnated).
constexpr double STAGNATION_RATIO = 1.0e-12;
// The working vectors of a subcase besides the basis and its products, taken from the memory budget before they are
// made: on the l set f', the candidate, the solution, the residual, a product and a solve's right-hand side; on the g
// set the applied and the whole loads, an expanded vector and its product, the displacements, the element, MPC and SPC
// forces.
constexpr std::uint64_t WORKING_VECTORS_ON_L = 6;
constexpr std::uint64_t WORKING_VECTORS_ON_G = 8;

// A value for a message: three significant digits.
std::string formatValue(double value)
{
    std::array<char, 32> text{};
    std::snprintf(text.data(), text.size(), "%.3g", value);
    return text.data();
}

double dot(std::span<const double> a, std::span<const double> b) noexcept
{
    double sum = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k)
        sum += a[k] * b[k];
    return sum;
}

// The Cholesky solution of the small reduced system A·y = b (A symmetric, n × n row-major); false when A is not
// positive definite.
bool solveReduced(std::vector<double> a, std::vector<double> b, std::size_t n, std::vector<double>& y)
{
    for (std::size_t j = 0; j < n; ++j)
    {
        double diagonal = a[j * n + j];
        for (std::size_t k = 0; k < j; ++k)
            diagonal -= a[j * n + k] * a[j * n + k];
        if (!(diagonal > 0.0))
            return false;
        a[j * n + j] = std::sqrt(diagonal);
        for (std::size_t i = j + 1; i < n; ++i)
        {
            double value = a[i * n + j];
            for (std::size_t k = 0; k < j; ++k)
                value -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = value / a[j * n + j];
        }
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t k = 0; k < i; ++k)
            b[i] -= a[i * n + k] * b[k];
        b[i] /= a[i * n + i];
    }
    for (std::size_t i = n; i-- > 0;)
    {
        for (std::size_t k = i + 1; k < n; ++k)
            b[i] -= a[k * n + i] * b[k];
        b[i] /= a[i * n + i];
    }
    y = std::move(b);
    return true;
}

// The approximate displacements of a subcase (the g set) and the work of the solve on the l set: u_lᵀ·f_l and
// u_lᵀ·ρ_l (the external work and EPSILON the static solution reports).
struct Approximation
{
    std::vector<double> displacements;
    double work = 0.0;
    double residualWork = 0.0;
};

// The vectors of a SUBCOM's term at the trial point: its approximated displacements, its applied loads,
// every load (thermal included) and its line loads, combined with the SUBCOM's coefficients.
struct CaseVectors
{
    std::vector<double> displacements;
    std::vector<double> applied;
    std::vector<double> every;
    std::vector<std::vector<elements::ElementLineLoad>> lineLoads;
};

class ReanalysisComputation : public detail::SensitivityComputation
{
public:
    ReanalysisComputation(const StaticSensitivityInputs& inputs, const GradientOptions& options,
                          const DesignProblem& problem, base::MemoryBudget& budget)
        : detail::SensitivityComputation(inputs, options, budget), m_problem(problem)
    {
        m_preparesEverySubcase = true;
    }

    FastReanalysisResult run(std::span<const double> x, std::span<const SuppliedField> supplied,
                             std::span<const SuppliedField> initialSupplied, const CallableRegistry& callables,
                             const FastReanalysisOptions& options, std::size_t blockItems)
    {
        FastReanalysisResult result;
        // The combined approximations work on the factor of the whole model, which the condensed path does not make.
        if (m_in.results->path == solutions::AnalysisPath::Condensed)
        {
            result.fallback =
                "full analysis: the kept analysis is on the condensed path, whose factors (the units' "
                "interiors and the interface) the combined approximations of this version do not use";
            return result;
        }
        // The eigenvalue analyses have no combined approximation here: statics only.
        if (m_in.modes != nullptr || m_in.buckling != nullptr)
        {
            result.fallback =
                "full analysis: the design point has MODES or BUCK subcases, whose eigenvalues the "
                "combined approximations (statics only) do not give";
            return result;
        }
        // The stiffness of an external unit at the new point is the plugin's, which the series of element
        // differences does not know.
        if (hasExternalUnits())
        {
            result.fallback =
                "full analysis: the model has an external unit, whose K* at the new point only its plugin "
                "gives";
            return result;
        }
        try
        {
            prepareSubcases();
        }
        catch (const base::OptimisationError& error)
        {
            result.fallback = std::string("full analysis: ").append(error.what());
            return result;
        }
        for (const detail::SubcaseState& state : m_subcases)
        {
            // The relieved loads Q'·P' of the trial point (its mass moves Q and the accelerations) are not formed
            // here: the series of the kept factor would solve the unrelieved loads.
            if (state.relief != nullptr)
            {
                result.fallback = base::concatenate(
                    {"full analysis: subcase ", std::to_string(state.subcase.label),
                     " is under inertia relief (PARAM InertiaRelief = -1), whose relieved loads at the trial point the "
                     "combined approximations of this version do not form"});
                return result;
            }
            if (state.hasEnforcedDisplacements)
            {
                result.fallback = base::concatenate({"full analysis: subcase ", std::to_string(state.subcase.label),
                                                     " enforces displacements, which the combined approximations of "
                                                     "this version do not take"});
                return result;
            }
        }
        DesignPoint point = m_problem.designPoint(x, supplied);
        for (const base::Diagnostic& distortion : m_problem.distortions(point, m_budget))
            if (distortion.severity == base::Severity::Error)
            {
                result.fallback = std::string("full analysis: the trial point distorts an element (")
                                      .append(distortion.message)
                                      .append(")");
                return result;
            }
        const std::shared_ptr<const input::ModelSnapshot> snapshot = m_problem.overlaySnapshot(point);
        const elements::SnapshotIndex index(*snapshot, m_budget);
        const assembly::ElementSet set(index, m_registry);
        const input::ParamTable* params = snapshot->hasTable(input::TableRole::Params) ? &snapshot->params() : nullptr;
        const assembly::LoadVectorBuilder loads(set, m_dofMap, m_budget);
        // The trial stiffness of each constraint set, assembled on the kept pattern (no factorisation).
        std::vector<std::unique_ptr<linalg::BlockSparseMatrix>> stiffness(m_systems.size());
        for (std::size_t g = 0; g < m_systems.size(); ++g)
        {
            stiffness[g] = std::make_unique<linalg::BlockSparseMatrix>(m_systems[g]->pattern(), m_budget,
                                                                       "fast reanalysis stiffness");
            assembly::StiffnessAssembler(set, m_dofMap).assemble(*stiffness[g], m_in.pool, nullptr);
        }
        const output::NodalResultBuilder nodalBuilder(m_dofMap);
        const output::ElementResultBuilder elementBuilder(set, m_dofMap);
        const output::StiffnessRecovery stiffnessRecovery(set, m_dofMap);
        const output::OutputPlan plan = planOf();
        solutions::StaticResults results;
        results.gSetSize = static_cast<std::int64_t>(gSize());
        results.kernel = "fast reanalysis";
        // The vectors of the SUBCOMs' terms, kept for their combinations.
        std::vector<CaseVectors> termVectors(m_subcases.size());
        std::vector<std::uint8_t> isTerm(m_subcases.size(), 0);
        for (const detail::SubcaseState& state : m_subcases)
            for (const auto& [term, coefficient] : state.terms)
                isTerm[term] = 1;
        for (std::size_t s = 0; s < m_subcases.size(); ++s)
        {
            const detail::SubcaseState& state = m_subcases[s];
            if (state.isCombination())
                continue; // combined from its terms below
            const SensitivitySystem& system = systemOf(state);
            const linalg::BlockSparseMatrix& kgg = *stiffness[state.group];
            // The vectors of the subcase grow with the model: estimated and taken from the budget first.
            const auto equations = static_cast<std::uint64_t>(system.equationCount());
            const base::MemoryReservation working = m_budget.reserve(
                sizeof(double) *
                    ((2 * static_cast<std::uint64_t>(options.maxBasis) + WORKING_VECTORS_ON_L) * equations +
                     WORKING_VECTORS_ON_G * static_cast<std::uint64_t>(gSize())),
                "fast reanalysis vectors");
            // P' (every load) and its applied part (no thermal load, as OLOAD).
            base::DiagnosticSink sink;
            std::vector<double> applied(gSize(), 0.0);
            if (state.subcase.load)
            {
                std::unique_ptr<linalg::BlockSparseMatrix> mass;
                if (loads.needsMass(*state.subcase.load))
                {
                    mass =
                        std::make_unique<linalg::BlockSparseMatrix>(system.pattern(), m_budget, "fast reanalysis mass");
                    const assembly::NonStructuralMassShares shares(set);
                    assembly::MassAssembler(
                        set, m_dofMap, shares,
                        assembly::MassAssemblyOptions::fromParams(params, state.subcase.label, sink))
                        .assemble(*mass, m_in.pool);
                }
                loads.addLoads(*state.subcase.load, mass.get(), applied, m_budget, sink);
            }
            std::vector<double> every = applied;
            if (state.temperatures)
                loads.addThermalLoads(state.temperatures->temperatures(), every);
            if (sink.hasErrors())
            {
                result.fallback = std::string("full analysis: the loads of the trial point: ")
                                      .append(base::formatDiagnostic(sink.diagnostics().front()));
                return result;
            }
            Approximation approximation;
            FastReanalysisSubcase reached{state.subcase.label, 0, 0.0};
            if (!combinedApproximations(system, kgg, every, options, approximation, reached, result))
            {
                result.subcases.push_back(reached);
                result.residual = std::max(result.residual, reached.residual);
                if (result.fallback.empty())
                    result.fallback =
                        base::concatenate({"full analysis: in subcase ", std::to_string(state.subcase.label),
                                           ", the residual of the combined approximations is ",
                                           formatValue(reached.residual), " after ", std::to_string(reached.basis),
                                           " basis vectors, above the tolerance ", formatValue(options.tolerance)});
                return result;
            }
            result.subcases.push_back(reached);
            result.residual = std::max(result.residual, reached.residual);
            std::vector<std::vector<elements::ElementLineLoad>> lineLoads;
            if (state.subcase.load)
                lineLoads = loads.lineLoadsOf(*state.subcase.load);
            recover(state, system, kgg, nodalBuilder, elementBuilder, stiffnessRecovery, plan, applied, every,
                    approximation, lineLoads, results);
            if (isTerm[s] != 0)
                termVectors[s] = {approximation.displacements, std::move(applied), std::move(every),
                                  std::move(lineLoads)};
        }
        // A SUBCOM: the combination of its terms' approximated vectors, recovered as the static solution recovers it
        // (its own temperatures, its terms' line loads scaled).
        for (const detail::SubcaseState& state : m_subcases)
        {
            if (!state.isCombination())
                continue;
            Approximation combined;
            combined.displacements.assign(gSize(), 0.0);
            std::vector<double> applied(gSize(), 0.0);
            std::vector<double> every(gSize(), 0.0);
            std::vector<std::vector<elements::ElementLineLoad>> lineLoads(set.batches().size());
            for (const auto& [term, coefficient] : state.terms)
            {
                const CaseVectors& vectors = termVectors[term];
                for (std::size_t dof = 0; dof < gSize(); ++dof)
                {
                    combined.displacements[dof] += coefficient * vectors.displacements[dof];
                    applied[dof] += coefficient * vectors.applied[dof];
                    every[dof] += coefficient * vectors.every[dof];
                }
                for (std::size_t b = 0; b < vectors.lineLoads.size() && b < lineLoads.size(); ++b)
                    for (elements::ElementLineLoad load : vectors.lineLoads[b])
                    {
                        load.p1 *= coefficient;
                        load.p2 *= coefficient;
                        lineLoads[b].push_back(load);
                    }
            }
            for (std::vector<elements::ElementLineLoad>& batch : lineLoads)
                std::stable_sort(batch.begin(), batch.end(),
                                 [](const elements::ElementLineLoad& left, const elements::ElementLineLoad& right)
                                 { return left.elementRow < right.elementRow; });
            recover(state, systemOf(state), *stiffness[state.group], nodalBuilder, elementBuilder, stiffnessRecovery,
                    plan, applied, every, combined, lineLoads, results);
        }
        result.isWithinTolerance = true;
        std::vector<std::int32_t> analysed(m_in.subcases.begin(), m_in.subcases.end());
        StaticDesignEvaluation analysis{std::move(point), snapshot, std::move(results), std::move(analysed)};
        result.evaluation = std::make_unique<DesignEvaluation>(
            m_problem.evaluateResponses(std::move(analysis), callables, m_budget, initialSupplied, blockItems));
        return result;
    }

private:
    // The outputs of the trial point: those the responses read, and the displacements of every point (the linearity
    // indicator).
    [[nodiscard]] output::OutputPlan planOf() const
    {
        output::OutputPlan plan;
        const std::array<output::OutputRequestPlan, output::OUTPUT_KIND_COUNT> required =
            m_in.responses->requiredOutputs();
        for (std::size_t k = 0; k < output::OUTPUT_KIND_COUNT; ++k)
            if (required[k].isRequested)
                plan.setRequest(static_cast<output::OutputKind>(k), required[k]);
        output::OutputRequestPlan displacements;
        displacements.isRequested = true;
        displacements.isAll = true;
        plan.setRequest(output::OutputKind::Displacement, displacements);
        return plan;
    }

    // K'·v on the l set: the expansion T·v, Kgg'·(T·v), the reduction Tᵀ (SensitivitySystem).
    void applyStiffness(const SensitivitySystem& system, const linalg::BlockSparseMatrix& kgg,
                        std::span<const double> vectorL, std::span<double> productL) const
    {
        std::vector<double> expanded(gSize());
        std::vector<double> product(gSize());
        system.fromEquations(vectorL, expanded);
        kgg.multiply(expanded, product);
        system.toEquations(product, productL);
    }

    // K⁻¹·v on the l set (one solve on the kept factor).
    void solveCurrent(const SensitivitySystem& system, std::span<double> vectorL, FastReanalysisResult& result) const
    {
        const std::size_t equations = vectorL.size();
        linalg::DenseMatrix rightHandSide(m_budget, equations, 1, "fast reanalysis right-hand side");
        for (std::size_t e = 0; e < equations; ++e)
            rightHandSide(e, 0) = vectorL[e];
        if (equations > 0)
            system.solve(rightHandSide.view());
        for (std::size_t e = 0; e < equations; ++e)
            vectorL[e] = rightHandSide(e, 0);
        ++result.solves;
    }

    // The Ritz solution on the basis of the combined approximations (FastReanalysis.hpp), into `approximation`; false
    // when η stays above the tolerance.
    bool combinedApproximations(const SensitivitySystem& system, const linalg::BlockSparseMatrix& kgg,
                                std::span<const double> loadsG, const FastReanalysisOptions& options,
                                Approximation& approximation, FastReanalysisSubcase& reached,
                                FastReanalysisResult& result) const
    {
        const auto equations = static_cast<std::size_t>(system.equationCount());
        std::vector<double> f(equations);
        system.toEquations(loadsG, f);
        if (dot(f, f) == 0.0)
        {
            // No load on the l set: u' = 0 exactly.
            approximation.displacements.assign(gSize(), 0.0);
            reached.residual = 0.0;
            return true;
        }
        std::vector<std::vector<double>> basis;          // orthonormal, on the l set
        std::vector<std::vector<double>> stiffnessBasis; // K'·r of each
        std::vector<double> candidate = f;               // K⁻¹·f' first, then the preconditioned residuals
        solveCurrent(system, candidate, result);
        std::vector<double> solution(equations, 0.0);
        reached.residual = std::numeric_limits<double>::infinity();
        while (basis.size() < options.maxBasis)
        {
            const double before = std::sqrt(dot(candidate, candidate));
            if (before == 0.0)
                break;
            for (int pass = 0; pass < 2; ++pass)
                for (const std::vector<double>& vector : basis)
                {
                    const double projection = dot(vector, candidate);
                    for (std::size_t e = 0; e < equations; ++e)
                        candidate[e] -= projection * vector[e];
                }
            const double after = std::sqrt(dot(candidate, candidate));
            if (after <= STAGNATION_RATIO * before)
                break;
            for (double& value : candidate)
                value /= after;
            std::vector<double> product(equations);
            applyStiffness(system, kgg, candidate, product);
            basis.push_back(std::move(candidate));
            stiffnessBasis.push_back(std::move(product));
            // (Rᵀ·K'·R)·y = Rᵀ·f'.
            const std::size_t n = basis.size();
            std::vector<double> reduced(n * n);
            std::vector<double> rightHandSide(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                rightHandSide[i] = dot(basis[i], f);
                for (std::size_t j = 0; j < n; ++j)
                    reduced[i * n + j] = 0.5 * (dot(basis[i], stiffnessBasis[j]) + dot(basis[j], stiffnessBasis[i]));
            }
            std::vector<double> y;
            if (!solveReduced(reduced, rightHandSide, n, y))
            {
                result.fallback =
                    "full analysis: the reduced system of the combined approximations is not positive "
                    "definite (the trial stiffness has lost its definiteness)";
                reached.basis = n;
                return false;
            }
            std::fill(solution.begin(), solution.end(), 0.0);
            std::vector<double> residual = f;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t e = 0; e < equations; ++e)
                {
                    solution[e] += y[i] * basis[i][e];
                    residual[e] -= y[i] * stiffnessBasis[i][e];
                }
            // η = (ρᵀ·K⁻¹·ρ / f'ᵀ·u')^½; K⁻¹·ρ is the next basis vector.
            candidate = residual;
            solveCurrent(system, candidate, result);
            const double work = dot(f, solution);
            reached.basis = n;
            reached.residual = work > 0.0 ? std::sqrt(std::abs(dot(residual, candidate)) / work) : 0.0;
            if (reached.residual <= options.tolerance)
            {
                approximation.displacements.assign(gSize(), 0.0);
                system.fromEquations(solution, approximation.displacements);
                approximation.work = work;
                approximation.residualWork = dot(solution, residual);
                return true;
            }
        }
        return false;
    }

    // The tables, SPC forces and checks of a subcase from its displacements, as the static solution recovers a case.
    void recover(const detail::SubcaseState& state, const SensitivitySystem& system,
                 const linalg::BlockSparseMatrix& kgg, const output::NodalResultBuilder& nodalBuilder,
                 const output::ElementResultBuilder& elementBuilder, const output::StiffnessRecovery& stiffnessRecovery,
                 const output::OutputPlan& plan, std::span<const double> applied, std::span<const double> every,
                 const Approximation& approximation,
                 const std::vector<std::vector<elements::ElementLineLoad>>& lineLoads,
                 solutions::StaticResults& results) const
    {
        const std::span<const double> displacements = approximation.displacements;
        const std::int32_t label = state.subcase.label;
        const auto addNodal = [&](output::OutputKind kind, std::span<const double> values)
        {
            const output::OutputRequestPlan& request = plan.of(kind);
            if (!request.isRequested)
                return;
            results.arrays.nodalTables.push_back(nodalBuilder.build(std::string(output::tableKindOf(kind)), label,
                                                                    values, m_budget, request.isAll, request.members));
        };
        // SPC forces: Kgg'·u' − P' − the MPC forces on the constrained DOFs (StaticSolution step 5).
        std::vector<double> elementForces(gSize());
        kgg.multiply(displacements, elementForces);
        std::vector<double> mpcForces(gSize(), 0.0);
        system.gm().computeMpcForces(elementForces, every, mpcForces);
        std::vector<double> spcForces(gSize(), 0.0);
        for (std::size_t dof = 0; dof < gSize(); ++dof)
            if (system.isConstrained(static_cast<base::DofIndex>(dof)))
                spcForces[dof] = elementForces[dof] - every[dof] - mpcForces[dof];
        addNodal(output::OutputKind::Displacement, displacements);
        addNodal(output::OutputKind::AppliedLoads, applied);
        addNodal(output::OutputKind::SpcForces, spcForces);
        addNodal(output::OutputKind::MpcForces, mpcForces);
        const elements::ElementTemperatures* temperatures =
            state.temperatures ? &state.temperatures->temperatures() : nullptr;
        const output::ElementOutputSelection selection = output::elementSelectionOf(plan);
        elementBuilder.build(label, displacements, lineLoads, m_budget, results.arrays.elementTables, m_in.pool,
                             nullptr, temperatures, &selection);
        if (plan.of(output::OutputKind::StrainEnergy).isRequested)
            stiffnessRecovery.strainEnergy(label, displacements, plan.of(output::OutputKind::StrainEnergy), m_budget,
                                           results.arrays.elementTables);
        // The external work ½·u_lᵀ·f_l and EPSILON = u_lᵀ·ρ_l / u_lᵀ·f_l on the l set, as the static solution.
        output::SubcaseCheck check;
        check.subcase = label;
        check.isCombination = state.isCombination(); // no EPSILON nor work, as the static solution's SUBCOM
        if (!check.isCombination)
        {
            check.externalWork = 0.5 * approximation.work;
            check.epsilon = approximation.work != 0.0 ? approximation.residualWork / approximation.work : 0.0;
        }
        results.checks.push_back(std::move(check));
    }

    const DesignProblem& m_problem;
};

} // namespace

FastReanalysisResult FastReanalysis::evaluate(const DesignProblem& problem, const DesignEvaluation& current,
                                              std::span<const double> x, std::span<const SuppliedField> supplied,
                                              std::span<const SuppliedField> initialSupplied,
                                              const CallableRegistry& callables, base::MemoryBudget& budget,
                                              const solutions::StaticOptions& analysisOptions,
                                              const FastReanalysisOptions& options, std::size_t blockItems)
{
    if (!(options.tolerance > 0.0) || options.maxBasis == 0)
        throw std::invalid_argument(base::concatenate(
            {"fast reanalysis: expected a positive tolerance and a basis of at least one vector, found ",
             std::to_string(options.tolerance), " and ", std::to_string(options.maxBasis)}));
    const ResponseSubcases subcases = problem.responseSubcases(current);
    const StaticSensitivityInputs inputs =
        problem.sensitivityInputs(current, subcases, callables, analysisOptions, blockItems);
    const GradientOptions gradientOptions;
    ReanalysisComputation computation(inputs, gradientOptions, problem, budget);
    return computation.run(x, supplied, initialSupplied, callables, options, blockItems);
}

} // namespace oos::design
