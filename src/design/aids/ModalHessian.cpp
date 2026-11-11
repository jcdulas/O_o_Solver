#include "oos/design/aids/ModalHessian.hpp"

#include "../sensitivity/ModalSystem.hpp"
#include "RecoveredFieldCurvature.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/design/sensitivity/AffectedElements.hpp"
#include "oos/linalg/eigen/BorderedEigenSystem.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace oos::design
{
namespace
{

constexpr double REPEATED_ROOT_TOLERANCE = 1e-6;

struct RootWeight
{
    std::size_t solve;
    std::size_t mode;
    double first = 0.0;  // weight times dr/dlambda
    double second = 0.0; // weight times d2r/dlambda2
    std::vector<detail::WeightedRecoveredField> fields;
};

// Translate weighted response items to roots. Frequency contributions keep their scalar chain-rule curvature.
std::optional<std::vector<RootWeight>> rootsOf(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                              const ProductWeights& weights)
{
    std::map<std::pair<std::size_t, std::size_t>, RootWeight> roots;
    const auto add = [&](std::int32_t id, std::size_t item, double weight) -> bool
    {
        const auto response = std::find_if(evaluation.responses.begin(), evaluation.responses.end(),
                                           [&](const auto& candidate) { return candidate.id == id; });
        if (response == evaluation.responses.end() || item >= response->itemCount())
            throw std::invalid_argument("modal Hessian weights refer to an unknown response item");
        if (weight == 0.0)
            return true;
        const auto* definition = problem.responses().findDesignResp1(id);
        if (!definition)
            return false;
        const auto& selected = definition->definition();
        if (!selected.mode && selected.type != ResponseType::Eigenvalue && selected.type != ResponseType::Frequency)
            return false;
        const auto& label = response->items[item];
        for (const auto& subcase : evaluation.analysis.modes->subcases)
            if (subcase.subcase == label.subcase && (label.mode > 0 || label.layer > 0))
            {
                const auto mode = static_cast<std::size_t>((selected.mode ? label.mode : label.layer) - 1);
                const auto& solve = evaluation.analysis.modes->solves[subcase.solve];
                if (mode >= solve.eigenvalues.size())
                    throw base::OptimisationError("modal Hessian mode is outside the extracted roots");
                auto& root = roots[{subcase.solve, mode}];
                root.solve = subcase.solve;
                root.mode = mode;
                if (definition->definition().mode)
                    root.fields.push_back({definition, label, weight, response->values[item]});
                else if (definition->definition().type == ResponseType::Eigenvalue)
                    root.first += weight;
                else
                {
                    const double lambda = solve.eigenvalues[mode];
                    if (!(lambda > 0.0))
                        throw base::OptimisationError("frequency Hessians require a positive eigenvalue");
                    const double slope = 1.0 / (4.0 * std::numbers::pi * std::sqrt(lambda));
                    root.first += weight * slope;
                    root.second -= weight * slope / (2.0 * lambda);
                }
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
            throw std::invalid_argument("modal Hessian weights have the wrong response item count");
        for (std::size_t item = 0; item < row.size(); ++item)
            if (!add(id, item, row[item]))
                return std::nullopt;
    }
    for (const auto& [id, row] : weights.sparseWeights)
    {
        if (row.rowStart.size() != 2 || row.rowStart[0] != 0 || row.rowStart[1] != row.items.size() ||
            row.values.size() != row.items.size())
            throw std::invalid_argument("modal Hessian sparse weights require one valid row");
        for (std::size_t entry = 0; entry < row.items.size(); ++entry)
            if (!add(id, row.items[entry], row.values[entry]))
                return std::nullopt;
    }
    std::vector<RootWeight> result;
    for (const auto& [key, root] : roots)
        result.push_back(root);
    return result;
}

class ModalSecondOrder
{
public:
    ModalSecondOrder(const DesignProblem& problem, const DesignEvaluation& evaluation,
                     base::MemoryBudget& budget, linalg::KernelKind kernel, base::ThreadPool* pool)
        : m_problem(problem), m_evaluation(evaluation), m_budget(budget), m_pool(pool),
          m_kernels(kernel, linalg::DenseKernelOptions{pool})
    {
    }

    std::optional<ProductReport> run(const std::vector<RootWeight>& roots, std::span<const double> directions,
                                      std::size_t count, std::vector<double>& values)
    {
        if (!hasPlainRelations())
            return std::nullopt;
        base::DiagnosticSink sink;
        m_model = std::make_unique<solutions::AnalysisModel>(*m_evaluation.analysis.snapshot,
            solutions::AnalysisKind::Modes, m_budget, m_pool, nullptr, sink);
        values.assign(m_problem.model().independentCount() * count, 0.0);
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t solve = 0; solve < m_evaluation.analysis.modes->solves.size(); ++solve)
        {
            if (std::none_of(roots.begin(), roots.end(), [&](const auto& root) { return root.solve == solve; }))
                continue;
            const auto& solution = m_evaluation.analysis.modes->solves[solve];
            const detail::ModalSystem system(*m_model, solution.subcases.front(), m_budget, m_pool);
            if (!prepareFields(system.massOptions().formulation))
                return std::nullopt;
            for (const RootWeight& root : roots)
                if (root.solve == solve && !addRoot(root, system, directions, count, values))
                    return std::nullopt;
        }
        m_report.usesModalTangents = true;
        m_report.hessianBlocksReason = "analytic vibration Hessian from bordered eigenvector tangent solves";
        m_report.productSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return m_report;
    }

private:
    struct FieldElement
    {
        ElementRef element;
        std::size_t quantity;
        elements::DesignedField field;
        elements::FieldUse use;
        elements::ElementMatrixBlock stiffness;
        elements::ElementMatrixBlock mass;
        base::MemoryReservation reservation;
    };

    struct FieldState
    {
        std::unique_ptr<detail::RecoveredFieldCurvature> recovery;
        base::BudgetedBuffer<double> actualShape;
        base::BudgetedBuffer<double> adjoint;
        base::BudgetedBuffer<double> freeAdjoint;
        std::optional<std::size_t> pivot;
        double scale = 1.0;
        double multiplier = 0.0;
        std::map<std::pair<std::size_t, std::size_t>, double> matrixPairs;
    };

    bool hasPlainRelations() const
    {
        const auto& quantities = m_evaluation.analysis.point.values().fields;
        for (std::size_t q = 0; q < quantities.size(); ++q)
        {
            if (quantities[q].gradient.empty())
                continue;
            const auto& target = m_problem.targets()[q];
            const auto& relation = m_problem.model().relations()[q];
            if (relation.form != RelationForm::Linear || !isDirectFieldTarget(relation, target) ||
                m_problem.targets().hasDerivedMaterial(q) ||
                m_problem.targets().derivedProperties().groupOf(q))
                return false;
        }
        return true;
    }

    bool prepareFields(elements::MassFormulation formulation)
    {
        m_entries.clear();
        m_byElement.clear();
        const AffectedElements affected(m_problem.model(), m_model->elementSet());
        const auto& quantities = m_evaluation.analysis.point.values().fields;
        for (std::size_t q = 0; q < quantities.size(); ++q)
        {
            if (quantities[q].gradient.empty())
                continue;
            const auto& relation = m_problem.model().relations()[q];
            for (const auto& element : affected.of(q))
            {
                const auto& batch = m_model->elementSet().batches()[element.batch];
                FieldElement entry;
                entry.element = element;
                entry.quantity = q;
                entry.field = {relation.targetCard, relation.field, relation.targetId, relation.fieldItem};
                entry.use = batch.family->designFieldUse(entry.field);
                if (entry.use.changesStiffness && batch.family->facets().has(elements::Facet::Stiffness) &&
                    !batch.family->stiffnessFieldDerivative(batch.view, element.row, entry.field, entry.stiffness))
                    return false;
                elements::MassOptions options;
                options.formulation = formulation;
                if (entry.use.changesMass && batch.family->facets().has(elements::Facet::Mass) &&
                    !batch.family->massFieldDerivative(batch.view, element.row, entry.field, options, entry.mass))
                    return false;
                std::uint64_t bytes = 0;
                for (const auto* matrix : {&entry.stiffness, &entry.mass})
                    for (std::size_t k = 0; k < matrix->size(); ++k)
                        bytes += matrix->dofs(k).size() * sizeof(elements::ElementDof) +
                                 matrix->matrix(k).size() * sizeof(double);
                entry.reservation = m_budget.reserve(bytes, "modal Hessian element matrix rates");
                m_byElement[element].push_back(m_entries.size());
                m_entries.push_back(std::move(entry));
            }
        }
        return true;
    }

    double product(const elements::ElementMatrixBlock& matrices, std::span<const double> left,
                    std::span<const double> right) const
    {
        double value = 0.0;
        for (std::size_t k = 0; k < matrices.size(); ++k)
        {
            const auto dofs = matrices.dofs(k);
            const auto matrix = matrices.matrix(k);
            for (std::size_t i = 0; i < dofs.size(); ++i)
                for (std::size_t j = 0; j < dofs.size(); ++j)
                    value += left[static_cast<std::size_t>(m_model->dofMap().dofOf(dofs[i]))] *
                             matrix[i * dofs.size() + j] *
                             right[static_cast<std::size_t>(m_model->dofMap().dofOf(dofs[j]))];
        }
        return value;
    }

    void addAction(const elements::ElementMatrixBlock& matrices, double factor, std::span<const double> shape,
                    std::span<double> into) const
    {
        if (factor == 0.0)
            return;
        for (std::size_t k = 0; k < matrices.size(); ++k)
        {
            const auto dofs = matrices.dofs(k);
            const auto matrix = matrices.matrix(k);
            for (std::size_t i = 0; i < dofs.size(); ++i)
                for (std::size_t j = 0; j < dofs.size(); ++j)
                    into[static_cast<std::size_t>(m_model->dofMap().dofOf(dofs[i]))] += factor *
                        matrix[i * dofs.size() + j] * shape[static_cast<std::size_t>(m_model->dofMap().dofOf(dofs[j]))];
        }
    }

    // Explicit matrix curvature at fixed mode: phi^T (K_pq - lambda M_pq) phi, only where two fields share an element.
    bool mixedRates(double root, double weightToMass, elements::MassFormulation formulation,
                     std::span<const double> shape, std::map<std::pair<std::size_t, std::size_t>, double>& rates,
                     std::span<const double> adjoint, double multiplier,
                     std::map<std::pair<std::size_t, std::size_t>, double>& fieldRates) const
    {
        for (const auto& [element, indices] : m_byElement)
        {
            const auto& batch = m_model->elementSet().batches()[element.batch];
            for (std::size_t a = 0; a < indices.size(); ++a)
                for (std::size_t b = a; b < indices.size(); ++b)
                {
                    const auto& first = m_entries[indices[a]];
                    const auto& second = m_entries[indices[b]];
                    double value = 0.0;
                    double fieldValue = 0.0;
                    if (first.use.changesStiffness && second.use.changesStiffness &&
                        batch.family->facets().has(elements::Facet::Stiffness))
                    {
                        elements::ElementMatrixBlock matrix;
                        if (!batch.family->stiffnessFieldMixedDerivative(batch.view, element.row,
                                                                         first.field, second.field, matrix))
                            return false;
                        value += product(matrix, shape, shape);
                        if (!adjoint.empty())
                            fieldValue -= product(matrix, adjoint, shape);
                    }
                    if (first.use.changesMass && second.use.changesMass &&
                        batch.family->facets().has(elements::Facet::Mass))
                    {
                        elements::MassOptions options;
                        options.formulation = formulation;
                        elements::ElementMatrixBlock matrix;
                        if (!batch.family->massFieldMixedDerivative(batch.view, element.row, first.field,
                                                                    second.field, options, matrix))
                            return false;
                        const double mass = weightToMass * product(matrix, shape, shape);
                        value -= root * mass;
                        if (!adjoint.empty())
                            fieldValue += root * weightToMass * product(matrix, adjoint, shape) -
                                          0.5 * multiplier * mass;
                    }
                    if (value != 0.0)
                    {
                        rates[{first.quantity, second.quantity}] += value;
                        if (a != b)
                            rates[{second.quantity, first.quantity}] += value;
                    }
                    if (fieldValue != 0.0)
                    {
                        fieldRates[{first.quantity, second.quantity}] += fieldValue;
                        if (a != b)
                            fieldRates[{second.quantity, first.quantity}] += fieldValue;
                    }
                }
        }
        return true;
    }

    bool prepareFieldState(const RootWeight& weight, const detail::ModalSystem& system,
                            std::span<const double> shape, double originalMass,
                            const linalg::BorderedEigenSystem& bordered, FieldState& state)
    {
        if (weight.fields.empty())
            return true;
        const auto& solve = m_evaluation.analysis.modes->solves[weight.solve];
        const auto& selection = system.selection();
        const auto n = static_cast<std::size_t>(selection.keptDofCount());
        const std::size_t g = shape.size();
        state.scale = solve.method.normalization == solutions::EigenNormalization::Mass ? 1.0 : std::sqrt(originalMass);
        state.actualShape = base::BudgetedBuffer<double>(m_budget, g, "modal field output normalisation");
        for (std::size_t i = 0; i < g; ++i)
            state.actualShape[i] = state.scale * shape[i];
        state.pivot = detail::modalNormalisationPivot(solve.method, state.actualShape.span(), selection,
                                                      m_model->dofMap());
        std::vector<detail::RecoveredFieldTarget> targets;
        targets.reserve(m_entries.size());
        for (const auto& entry : m_entries)
            targets.push_back({entry.element, entry.quantity, entry.field, entry.use});
        state.recovery = std::make_unique<detail::RecoveredFieldCurvature>(*m_model, m_evaluation.analysis.snapshot,
                                                                      m_budget);
        if (!state.recovery->prepare(weight.fields, targets, state.actualShape.span()))
            return false;
        base::BudgetedBuffer<double> weights(m_budget, g, "modal field normalised Hessian weights");
        const auto actual = state.recovery->weights();
        double projection = 0.0;
        for (std::size_t i = 0; i < g; ++i)
        {
            weights[i] = state.scale * actual[i];
            projection += actual[i] * state.actualShape[i];
        }
        if (state.pivot)
            weights[*state.pivot] -= state.scale * projection;
        system.constraints().gm().reduceToN(weights.span());
        base::BudgetedBuffer<double> right(m_budget, n, "modal field Hessian adjoint load");
        state.freeAdjoint = base::BudgetedBuffer<double>(m_budget, n, "modal field Hessian free adjoint");
        for (std::size_t i = 0; i < n; ++i)
            right[i] = weights[static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(i)))];
        state.multiplier = bordered.solve(right.span(), state.freeAdjoint.span());
        ++m_report.solves;
        state.adjoint = base::BudgetedBuffer<double>(m_budget, g, "modal field Hessian adjoint");
        std::fill(state.adjoint.span().begin(), state.adjoint.span().end(), 0.0);
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto dof = static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(i)));
            state.adjoint[dof] = state.freeAdjoint[i];
        }
        system.constraints().gm().expandToG(state.adjoint.span());
        return true;
    }

    // Differentiate the bordered adjoint B [z;t] = [R_phi;0]. Its right-hand side is
    // [R_phi' - A' z - b' t; -b'^T z], where A = K-lambda M and b = M phi.
    // This includes both output normalisation and the local recovery's displacement/field chain rule.
    void addFieldDirection(const FieldState& state, const detail::ModalSystem& system,
                            std::span<const double> shape, std::span<const double> tangent,
                            std::span<const double> freeTangent, std::span<const double> moves,
                            double root, double rootRate, const linalg::BorderedEigenSystem& bordered,
                            std::span<double> curvature)
    {
        if (!state.recovery)
            return;
        const auto& selection = system.selection();
        const auto n = static_cast<std::size_t>(selection.keptDofCount());
        const std::size_t g = shape.size();
        const double weightToMass = system.massOptions().weightToMass;
        const double scaleRate = state.pivot ? -state.scale * state.scale * tangent[*state.pivot] : 0.0;
        base::BudgetedBuffer<double> actualTangent(m_budget, g, "modal field normalised tangent");
        for (std::size_t i = 0; i < g; ++i)
            actualTangent[i] = state.scale * tangent[i] + scaleRate * shape[i];
        base::BudgetedBuffer<double> weightRate(m_budget, g, "modal field adjoint weight derivative");
        state.recovery->weightRate(actualTangent.span(), moves, weightRate.span());
        const auto weights = state.recovery->weights();
        double projection = 0.0, projectionRate = 0.0;
        for (std::size_t i = 0; i < g; ++i)
        {
            projection += weights[i] * state.actualShape[i];
            projectionRate += weightRate[i] * state.actualShape[i] + weights[i] * actualTangent[i];
            weightRate[i] = state.scale * weightRate[i] + scaleRate * weights[i];
        }
        if (state.pivot)
            weightRate[*state.pivot] -= state.scale * projectionRate + scaleRate * projection;
        base::BudgetedBuffer<double> pencilRate(m_budget, g, "modal adjoint pencil derivative");
        base::BudgetedBuffer<double> massRate(m_budget, g, "modal normalisation border derivative");
        std::fill(pencilRate.span().begin(), pencilRate.span().end(), 0.0);
        std::fill(massRate.span().begin(), massRate.span().end(), 0.0);
        for (const auto& entry : m_entries)
        {
            addAction(entry.stiffness, moves[entry.quantity], state.adjoint.span(), pencilRate.span());
            addAction(entry.mass, -root * weightToMass * moves[entry.quantity], state.adjoint.span(),
                       pencilRate.span());
            addAction(entry.mass, weightToMass * moves[entry.quantity], shape, massRate.span());
        }
        system.constraints().gm().reduceToN(weightRate.span());
        system.constraints().gm().reduceToN(pencilRate.span());
        system.constraints().gm().reduceToN(massRate.span());
        base::BudgetedBuffer<double> massAdjoint(m_budget, n, "modal mass times adjoint");
        base::BudgetedBuffer<double> borderRate(m_budget, n, "modal free border derivative");
        system.mass().multiply(state.freeAdjoint.span(), massAdjoint.span());
        system.mass().multiply(freeTangent, borderRate.span());
        base::BudgetedBuffer<double> right(m_budget, n, "modal adjoint derivative load");
        base::BudgetedBuffer<double> freeRate(m_budget, n, "modal free adjoint derivative");
        double constraintRate = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto dof = static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(i)));
            borderRate[i] += massRate[dof];
            right[i] = weightRate[dof] - pencilRate[dof] + rootRate * massAdjoint[i] - state.multiplier * borderRate[i];
            constraintRate -= borderRate[i] * state.freeAdjoint[i];
        }
        const double multiplierRate = bordered.solve(right.span(), freeRate.span(), constraintRate);
        ++m_report.solves;
        base::BudgetedBuffer<double> adjointRate(m_budget, g, "modal field adjoint derivative");
        std::fill(adjointRate.span().begin(), adjointRate.span().end(), 0.0);
        for (std::size_t i = 0; i < n; ++i)
            adjointRate[static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(i)))] = freeRate[i];
        system.constraints().gm().expandToG(adjointRate.span());
        for (const auto& entry : m_entries)
        {
            curvature[entry.quantity] += -product(entry.stiffness, adjointRate.span(), shape) +
                root * weightToMass * product(entry.mass, adjointRate.span(), shape) -
                product(entry.stiffness, state.adjoint.span(), tangent) +
                root * weightToMass * product(entry.mass, state.adjoint.span(), tangent) +
                rootRate * weightToMass * product(entry.mass, state.adjoint.span(), shape) -
                0.5 * multiplierRate * weightToMass * product(entry.mass, shape, shape) -
                state.multiplier * weightToMass * product(entry.mass, tangent, shape);
        }
        for (const auto& [pair, value] : state.matrixPairs)
            curvature[pair.first] += value * moves[pair.second];
        state.recovery->addExplicitRate(actualTangent.span(), moves, curvature);
    }

    bool addRoot(const RootWeight& weight, const detail::ModalSystem& system, std::span<const double> directions,
                  std::size_t count, std::vector<double>& values)
    {
        const auto& solve = m_evaluation.analysis.modes->solves[weight.solve];
        const double root = solve.eigenvalues[weight.mode];
        for (std::size_t other = 0; other < solve.eigenvalues.size(); ++other)
            if (other != weight.mode && std::abs(solve.eigenvalues[other] - root) <= REPEATED_ROOT_TOLERANCE *
                std::max({std::abs(root), std::abs(solve.eigenvalues[other]), 1e-12}))
                throw base::OptimisationError("an eigenvalue Hessian is not unique at a repeated root");
        const auto& selection = system.selection();
        const auto n = static_cast<std::size_t>(selection.keptDofCount());
        const auto g = static_cast<std::size_t>(m_model->dofMap().gSetSize());
        const auto& quantities = m_evaluation.analysis.point.values().fields;
        base::BudgetedBuffer<double> shape(m_budget, g, "modal Hessian normalised shape");
        std::copy_n(solve.eigenvectors.data() + weight.mode * solve.eigenvectors.leadingDimension(), g, shape.data());
        base::BudgetedBuffer<double> freeShape(m_budget, n, "modal Hessian free shape");
        base::BudgetedBuffer<double> massShape(m_budget, n, "modal Hessian mass product");
        for (std::size_t i = 0; i < n; ++i)
            freeShape[i] = shape[static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(i)))];
        system.mass().multiply(freeShape.span(), massShape.span());
        double norm = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            norm += freeShape[i] * massShape[i];
        if (!(norm > 0.0))
            throw base::OptimisationError("modal Hessian requires a positive generalised mass");
        const double scale = 1.0 / std::sqrt(norm);
        for (double& value : shape.span())
            value *= scale;
        for (std::size_t i = 0; i < n; ++i)
        {
            freeShape[i] *= scale;
            massShape[i] *= scale;
        }
        const auto massOptions = system.massOptions();
        const double weightToMass = massOptions.weightToMass;
        base::BudgetedBuffer<double> gradient(m_budget, quantities.size(), "modal eigenvalue field gradient");
        base::BudgetedBuffer<double> massRates(m_budget, quantities.size(), "modal mass field gradient");
        std::fill(gradient.span().begin(), gradient.span().end(), 0.0);
        std::fill(massRates.span().begin(), massRates.span().end(), 0.0);
        for (const auto& entry : m_entries)
        {
            const double mass = weightToMass * product(entry.mass, shape.span(), shape.span());
            massRates[entry.quantity] += mass;
            gradient[entry.quantity] += product(entry.stiffness, shape.span(), shape.span()) - root * mass;
        }
        std::map<std::pair<std::size_t, std::size_t>, double> mixed;
        const linalg::BorderedEigenSystem bordered(system.stiffness(), system.mass(), root, freeShape.span(),
                                                    m_kernels, m_budget);
        ++m_report.factorisations;
        FieldState fields;
        if (!prepareFieldState(weight, system, shape.span(), norm, bordered, fields) ||
            !mixedRates(root, weightToMass, massOptions.formulation, shape.span(), mixed, fields.adjoint.span(),
                         fields.multiplier, fields.matrixPairs))
            return false;
        for (std::size_t direction = 0; direction < count; ++direction)
        {
            base::BudgetedBuffer<double> moves(m_budget, quantities.size(), "modal Hessian field direction");
            double rootRate = 0.0, massRate = 0.0;
            for (std::size_t q = 0; q < quantities.size(); ++q)
            {
                moves[q] = 0.0;
                for (const auto& [column, derivative] : quantities[q].gradient)
                    moves[q] += derivative * directions[column * count + direction];
                rootRate += gradient[q] * moves[q];
                massRate += massRates[q] * moves[q];
            }
            base::BudgetedBuffer<double> tangent(m_budget, g, "modal eigenvector tangent");
            std::fill(tangent.span().begin(), tangent.span().end(), 0.0);
            for (const auto& entry : m_entries)
            {
                addAction(entry.stiffness, -moves[entry.quantity], shape.span(), tangent.span());
                addAction(entry.mass, root * weightToMass * moves[entry.quantity], shape.span(), tangent.span());
            }
            system.constraints().gm().reduceToN(tangent.span());
            base::BudgetedBuffer<double> right(m_budget, n, "modal Hessian tangent load");
            base::BudgetedBuffer<double> freeTangent(m_budget, n, "modal Hessian free tangent");
            for (std::size_t i = 0; i < n; ++i)
                right[i] = tangent[static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(i)))] +
                           rootRate * massShape[i];
            (void)bordered.solve(right.span(), freeTangent.span(), -0.5 * massRate);
            ++m_report.solves;
            std::fill(tangent.span().begin(), tangent.span().end(), 0.0);
            for (std::size_t i = 0; i < n; ++i)
            {
                const auto dof = static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(i)));
                tangent[dof] = freeTangent[i];
            }
            system.constraints().gm().expandToG(tangent.span());
            base::BudgetedBuffer<double> curvature(m_budget, quantities.size(), "modal Hessian field curvature");
            for (std::size_t q = 0; q < quantities.size(); ++q)
                curvature[q] = -rootRate * massRates[q];
            for (const auto& entry : m_entries)
                curvature[entry.quantity] += 2.0 * (product(entry.stiffness, tangent.span(), shape.span()) -
                    root * weightToMass * product(entry.mass, tangent.span(), shape.span()));
            for (const auto& [pair, value] : mixed)
                curvature[pair.first] += value * moves[pair.second];
            base::BudgetedBuffer<double> fieldCurvature(m_budget, quantities.size(), "modal field Hessian curvature");
            std::fill(fieldCurvature.span().begin(), fieldCurvature.span().end(), 0.0);
            addFieldDirection(fields, system, shape.span(), tangent.span(), freeTangent.span(), moves.span(),
                               root, rootRate, bordered, fieldCurvature.span());
            for (std::size_t q = 0; q < quantities.size(); ++q)
                for (const auto& [column, derivative] : quantities[q].gradient)
                    values[column * count + direction] += derivative *
                        (weight.first * curvature[q] + weight.second * rootRate * gradient[q] + fieldCurvature[q]);
        }
        return true;
    }

    const DesignProblem& m_problem;
    const DesignEvaluation& m_evaluation;
    base::MemoryBudget& m_budget;
    base::ThreadPool* m_pool;
    linalg::DenseKernels m_kernels;
    std::unique_ptr<solutions::AnalysisModel> m_model;
    std::vector<FieldElement> m_entries;
    std::map<ElementRef, std::vector<std::size_t>> m_byElement;
    ProductReport m_report;
};

} // namespace

std::optional<ProductReport> modalHessianVector(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                               const ProductWeights& weights, std::span<const double> directions,
                                               std::size_t count, std::vector<double>& values,
                                               base::MemoryBudget& budget, linalg::KernelKind kernel,
                                               base::ThreadPool* pool)
{
    if (weights.rows != 1 || count == 0 || directions.size() != problem.model().independentCount() * count)
        throw std::invalid_argument("modal Hessian requires one weight row and one value per variable and direction");
    if (!evaluation.analysis.modes || !evaluation.analysis.point.values().shapes.empty() ||
        (evaluation.analysis.modes->path != solutions::AnalysisPath::Full &&
         evaluation.analysis.modes->path != solutions::AnalysisPath::OutOfCore))
        return std::nullopt;
    if (!evaluation.trackingWarnings.empty())
        throw std::invalid_argument("modal Hessian requires unambiguous mode tracking");
    const auto roots = rootsOf(problem, evaluation, weights);
    if (!roots)
        return std::nullopt;
    return ModalSecondOrder(problem, evaluation, budget, kernel, pool).run(*roots, directions, count, values);
}

} // namespace oos::design
