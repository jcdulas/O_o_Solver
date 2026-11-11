#include "ModalItems.hpp"
#include "ModalSystem.hpp"

#include "oos/assembly/assemble/MassAssembler.hpp"
#include "oos/base/errors/OptimisationError.hpp"
#include "oos/linalg/eigen/BorderedEigenSystem.hpp"
#include "oos/solutions/common/AnalysisModel.hpp"
#include "oos/solutions/common/ConstrainedSystem.hpp"
#include "oos/solutions/common/FsetOrdering.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <utility>

namespace oos::design::detail
{
namespace
{

constexpr double REPEATED_ROOT_TOLERANCE = 1e-6;
constexpr double VALUE_TOLERANCE = 1e-7;
constexpr double RESIDUAL_TOLERANCE = 1e-6;

void checkValue(double actual, double expected)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > VALUE_TOLERANCE * std::max(1.0, std::abs(expected)))
        throw base::OptimisationError("modal field derivatives do not reproduce the evaluated response");
}

} // namespace

ModalItems::ModalItems(Inputs inputs)
    : m_in(std::move(inputs)), m_kernels(m_in.kernel, linalg::DenseKernelOptions{m_in.pool})
{
}

std::span<const double> ModalItems::shapeOf(const Mode& mode) const
{
    const auto& shapes = m_in.results->solves[mode.solve].eigenvectors;
    return {shapes.data() + mode.column * shapes.leadingDimension(), shapes.rows()};
}

elements::NodalDisplacements ModalItems::nodal(std::span<const double> vector) const
{
    const auto grid = static_cast<std::size_t>(m_in.dofMap->gridCount()) * 6;
    return {vector.first(grid), vector.subspan(grid), nullptr};
}

void ModalItems::add(const ItemKey& key, const DesignResp1Response& response, const ResponseItem& label, double value)
{
    if (label.mode < 1)
        throw base::OptimisationError("modal field response has no selected vibration mode");
    std::size_t solve = m_in.results->solves.size();
    for (const auto& subcase : m_in.results->subcases)
        if (subcase.subcase == label.subcase)
            solve = subcase.solve;
    if (solve >= m_in.results->solves.size())
        throw base::OptimisationError("modal field response has no vibration solve for its subcase");
    const auto column = static_cast<std::size_t>(label.mode - 1);
    const auto& roots = m_in.results->solves[solve].eigenvalues;
    if (column >= roots.size())
        throw base::OptimisationError("modal field response names a mode outside the extracted roots");
    for (std::size_t other = 0; other < roots.size(); ++other)
        if (other != column && std::abs(roots[other] - roots[column]) <=
            REPEATED_ROOT_TOLERANCE * std::max({std::abs(roots[other]), std::abs(roots[column]), 1e-12}))
            throw base::OptimisationError(
                "an individual modal field derivative is not unique at a repeated eigenvalue");
    auto found = std::find_if(m_modes.begin(), m_modes.end(), [&](const Mode& mode)
                              { return mode.solve == solve && mode.column == column; });
    if (found == m_modes.end())
    {
        m_modes.push_back({solve, column, {}, 1.0});
        found = m_modes.end() - 1;
    }
    Item item;
    item.key = key;
    item.label = label;
    item.mode = static_cast<std::size_t>(found - m_modes.begin());
    item.value = value;
    if (response.definition().type != ResponseType::Displacement)
    {
        const auto target = std::find_if(response.elementTargets().begin(), response.elementTargets().end(),
                                         [&](const auto& candidate) { return candidate.elementId == label.entity; });
        if (target == response.elementTargets().end())
            throw base::OptimisationError("modal element response has no resolved element target");
        const auto location = std::find_if(target->items.begin(), target->items.end(),
                                           [&](const auto& entry) { return entry.first == label.component; });
        for (std::size_t batch = 0; batch < m_in.set->batches().size(); ++batch)
            if (m_in.set->batches()[batch].view.card() == target->card && location != target->items.end())
            {
                item.element = ElementRef{batch, target->row};
                const std::vector<ElementItemSpec> specs{
                    {response.definition().type, label.component, location->second, label.layer}};
                item.function = std::make_unique<ElementItemFunction>(*m_in.set->batches()[batch].family,
                                                                      target->row, specs);
                break;
            }
        if (!item.function)
            throw base::OptimisationError("modal element response has no supported recovery family");
    }
    m_items.push_back(std::move(item));
}

void ModalItems::prepare()
{
    if (isEmpty())
        return;
    if (m_in.results->path == solutions::AnalysisPath::Condensed ||
        m_in.results->path == solutions::AnalysisPath::Substructuring)
        throw base::OptimisationError("modal field derivatives require the full vibration analysis path");
    base::DiagnosticSink sink;
    auto& budget = *m_in.budget;
    const auto g = static_cast<std::size_t>(m_in.dofMap->gSetSize());
    solutions::AnalysisModel model(*m_in.snapshot, solutions::AnalysisKind::Modes, budget, m_in.pool, nullptr, sink);
    for (std::size_t solveIndex = 0; solveIndex < m_in.results->solves.size(); ++solveIndex)
    {
        if (std::none_of(m_modes.begin(), m_modes.end(), [&](const Mode& mode) { return mode.solve == solveIndex; }))
            continue;
        const auto& solve = m_in.results->solves[solveIndex];
        const ModalSystem pencil(model, solve.subcases.front(), budget, m_in.pool);
        const auto& system = pencil.constraints();
        const auto& options = pencil.massOptions();
        const auto& selection = pencil.selection();
        const auto n = static_cast<std::size_t>(selection.keptDofCount());
        const auto& stiffness = pencil.stiffness();
        const auto& mass = pencil.mass();
        for (std::size_t modeIndex = 0; modeIndex < m_modes.size(); ++modeIndex)
        {
            Mode& mode = m_modes[modeIndex];
            if (mode.solve != solveIndex)
                continue;
            mode.massOptions.formulation = options.formulation;
            mode.weightToMass = options.weightToMass;
            const auto shape = shapeOf(mode);
            const double root = solve.eigenvalues[mode.column];
            base::BudgetedBuffer<double> freeShape(budget, n, "modal derivative free shape");
            base::BudgetedBuffer<double> work(budget, n, "modal derivative pencil product");
            base::BudgetedBuffer<double> massShape(budget, n, "modal derivative mass product");
            for (std::size_t i = 0; i < n; ++i)
                freeShape[i] = shape[static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(i)))];
            stiffness.multiply(freeShape.span(), work.span());
            mass.multiply(freeShape.span(), massShape.span());
            double residual = 0.0, scale = 0.0;
            for (std::size_t i = 0; i < n; ++i)
            {
                residual = std::max(residual, std::abs(work[i] - root * massShape[i]));
                scale = std::max(scale, std::abs(work[i]) + std::abs(root * massShape[i]));
            }
            if (residual > RESIDUAL_TOLERANCE * std::max(scale, 1e-20))
                throw base::OptimisationError("modal field derivatives require a converged full-model eigenvector");
            const auto pivot = modalNormalisationPivot(solve.method, shape, selection, model.dofMap());
            const linalg::BorderedEigenSystem bordered(stiffness, mass, root, freeShape.span(), m_kernels, budget);
            for (Item& item : m_items)
            {
                if (item.mode != modeIndex)
                    continue;
                base::BudgetedBuffer<double> weights(budget, g, "modal field displacement weights");
                std::fill(weights.span().begin(), weights.span().end(), 0.0);
                if (!item.function)
                {
                    const auto dof = model.dofMap().findDof(item.label.entity, item.label.component);
                    if (!dof)
                        throw base::OptimisationError("modal displacement response names an unknown DOF");
                    weights[static_cast<std::size_t>(*dof)] = 1.0;
                    checkValue(shape[static_cast<std::size_t>(*dof)], item.value);
                }
                else
                {
                    const auto& view = m_in.set->batches()[item.element->batch].view;
                    base::BudgetedBuffer<double> scratch(budget, g, "modal field recovery scratch");
                    const auto slopes = item.function->slopes(view, m_in.snapshot, nodal(shape), [&]
                        {
                            std::copy(shape.begin(), shape.end(), scratch.data());
                            const auto grid = static_cast<std::size_t>(model.dofMap().gridCount()) * 6;
                            return ScratchDisplacements{scratch.span().first(grid), scratch.span().subspan(grid)};
                        }, budget);
                    if (!slopes)
                        throw base::OptimisationError("modal field recovery does not give displacement derivatives");
                    checkValue(slopes->values.front(), item.value);
                    for (std::size_t d = 0; d < slopes->dofs.size(); ++d)
                        weights[static_cast<std::size_t>(model.dofMap().dofOf(slopes->dofs[d]))] += slopes->weights[d];
                }
                if (pivot)
                {
                    double projection = 0.0;
                    for (std::size_t i = 0; i < g; ++i)
                        projection += weights[i] * shape[i];
                    weights[*pivot] -= projection / shape[*pivot];
                }
                system.gm().reduceToN(weights.span());
                base::BudgetedBuffer<double> right(budget, n, "modal field free weights");
                base::BudgetedBuffer<double> adjoint(budget, n, "modal field free adjoint");
                for (std::size_t i = 0; i < n; ++i)
                    right[i] = weights[static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(i)))];
                item.multiplier = bordered.solve(right.span(), adjoint.span());
                item.adjoint = base::BudgetedBuffer<double>(budget, g, "modal field adjoint");
                std::fill(item.adjoint.span().begin(), item.adjoint.span().end(), 0.0);
                for (std::size_t i = 0; i < n; ++i)
                {
                    const auto dof = static_cast<std::size_t>(selection.dofOfEquation(static_cast<std::int32_t>(i)));
                    item.adjoint[dof] = adjoint[i];
                }
                system.gm().expandToG(item.adjoint.span());
            }
        }
    }
}

double ModalItems::product(const ElementMatrixDerivative& matrix, std::span<const double> left,
                           std::span<const double> right) const
{
    double sum = 0.0;
    const std::size_t n = matrix.dofs.size();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            sum += left[static_cast<std::size_t>(m_in.dofMap->dofOf(matrix.dofs[i]))] * matrix.values[i * n + j] *
                   right[static_cast<std::size_t>(m_in.dofMap->dofOf(matrix.dofs[j]))];
    return sum;
}

double ModalItems::explicitRate(Item& item, const ElementRef& element, const elements::DesignedField* field,
                                const LazyFieldPerturbation& perturbation) const
{
    if (!item.element || *item.element != element)
        return 0.0;
    const auto shape = shapeOf(m_modes[item.mode]);
    const auto& view = m_in.set->batches()[element.batch].view;
    if (field != nullptr)
        if (const auto rates = item.function->fieldRates(view, nodal(shape), *field))
            return rates->front();
    const auto& differences = perturbation.get();
    std::array<double, FIELD_SIDE_COUNT> values{};
    for (std::size_t side = 0; side < values.size(); ++side)
    {
        const auto which = static_cast<FieldSide>(side);
        const auto& moved = differences.set(which).batches()[element.batch].view;
        values[side] = item.function->values(moved, differences.snapshot(which), nodal(shape), *m_in.budget).front();
    }
    return differences.derivative(values[0], values[1], values[2], values[3]);
}

void ModalItems::addElement(std::size_t quantity, const ElementRef& element, const elements::FieldUse& use,
                            const ElementMatrixDerivative& stiffness, const MassOf& massOf,
                            const elements::DesignedField* field, const LazyFieldPerturbation& perturbation)
{
    std::map<elements::MassFormulation, ElementMatrixDerivative> masses;
    for (Item& item : m_items)
    {
        const Mode& mode = m_modes[item.mode];
        const auto shape = shapeOf(mode);
        double rate = -product(stiffness, item.adjoint.span(), shape);
        if (use.changesMass)
        {
            auto [found, isNew] = masses.try_emplace(mode.massOptions.formulation);
            if (isNew)
                found->second = massOf(mode.massOptions);
            const auto& mass = found->second;
            const double root = m_in.results->solves[mode.solve].eigenvalues[mode.column];
            rate += mode.weightToMass * (root * product(mass, item.adjoint.span(), shape) -
                                         0.5 * item.multiplier * product(mass, shape, shape));
        }
        if (use.changesRecovery)
            rate += explicitRate(item, element, field, perturbation);
        if (rate != 0.0)
            item.byQuantity[quantity] += rate;
    }
}

void ModalItems::addExplicitParts(const std::vector<FieldValue>& quantities, LinearisedItems& linearised) const
{
    for (const Item& item : m_items)
        for (const auto& [quantity, rate] : item.byQuantity)
            for (const auto& [column, slope] : quantities[quantity].gradient)
                linearised.at(item.key).explicitPart.emplace_back(column, rate * slope);
}

} // namespace oos::design::detail
