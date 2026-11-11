#include "SizingMatrixRates.hpp"

namespace oos::design::detail
{

SizingMatrixRates::SizingMatrixRates(const DesignProblem& problem, const DesignEvaluation& evaluation,
                                     const solutions::AnalysisModel& model, base::MemoryBudget& budget)
    : m_problem(problem), m_evaluation(evaluation), m_model(model), m_budget(budget)
{
}

bool SizingMatrixRates::hasPlainRelations() const
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

void SizingMatrixRates::reserve(Matrices& matrices)
{
    std::uint64_t bytes = 0;
    for (const auto* matrix : {&matrices.stiffness, &matrices.mass})
        for (std::size_t k = 0; k < matrix->size(); ++k)
            bytes += matrix->dofs(k).size() * sizeof(elements::ElementDof) + matrix->matrix(k).size() * sizeof(double);
    matrices.reservation = m_budget.reserve(bytes, "sizing Hessian element matrix rates");
}

bool SizingMatrixRates::prepare(const assembly::MassAssemblyOptions& options, bool needsMass)
{
    if (!hasPlainRelations())
        return false;
    m_first.clear();
    m_mixed.clear();
    m_targets.clear();
    m_byQuantity.clear();
    m_mixedByQuantity.clear();
    m_weightToMass = options.weightToMass;
    const AffectedElements affected(m_problem.model(), m_model.elementSet());
    const auto& quantities = m_evaluation.analysis.point.values().fields;
    std::map<ElementRef, std::vector<std::size_t>> byElement;
    elements::MassOptions massOptions;
    massOptions.formulation = options.formulation;
    for (std::size_t q = 0; q < quantities.size(); ++q)
    {
        if (quantities[q].gradient.empty())
            continue;
        const auto& relation = m_problem.model().relations()[q];
        for (const auto& element : affected.of(q))
        {
            const auto& batch = m_model.elementSet().batches()[element.batch];
            First entry;
            entry.target = {element, q,
                            {relation.targetCard, relation.field, relation.targetId, relation.fieldItem}, {}};
            entry.target.use = batch.family->designFieldUse(entry.target.field);
            if (entry.target.use.changesStiffness && batch.family->facets().has(elements::Facet::Stiffness) &&
                !batch.family->stiffnessFieldDerivative(batch.view, element.row, entry.target.field,
                                                       entry.matrices.stiffness))
                return false;
            if (needsMass && entry.target.use.changesMass && batch.family->facets().has(elements::Facet::Mass) &&
                !batch.family->massFieldDerivative(batch.view, element.row, entry.target.field, massOptions,
                                                  entry.matrices.mass))
                return false;
            reserve(entry.matrices);
            byElement[element].push_back(m_first.size());
            m_byQuantity[q].push_back(m_first.size());
            m_targets.push_back(entry.target);
            m_first.push_back(std::move(entry));
        }
    }
    for (const auto& [element, entries] : byElement)
    {
        const auto& batch = m_model.elementSet().batches()[element.batch];
        for (std::size_t a = 0; a < entries.size(); ++a)
            for (std::size_t b = a; b < entries.size(); ++b)
            {
                const auto& first = m_first[entries[a]].target;
                const auto& second = m_first[entries[b]].target;
                Mixed entry;
                entry.first = first.quantity;
                entry.second = second.quantity;
                if (first.use.changesStiffness && second.use.changesStiffness &&
                    batch.family->facets().has(elements::Facet::Stiffness) &&
                    !batch.family->stiffnessFieldMixedDerivative(batch.view, element.row, first.field, second.field,
                                                                entry.matrices.stiffness))
                    return false;
                if (needsMass && first.use.changesMass && second.use.changesMass &&
                    batch.family->facets().has(elements::Facet::Mass) &&
                    !batch.family->massFieldMixedDerivative(batch.view, element.row, first.field, second.field,
                                                           massOptions, entry.matrices.mass))
                    return false;
                reserve(entry.matrices);
                m_mixedByQuantity[entry.first].push_back(m_mixed.size());
                if (entry.second != entry.first)
                    m_mixedByQuantity[entry.second].push_back(m_mixed.size());
                m_mixed.push_back(std::move(entry));
            }
    }
    return true;
}

void SizingMatrixRates::action(const elements::ElementMatrixBlock& matrix, double factor,
                              std::span<const double> vector, std::span<double> into) const
{
    if (factor == 0.0)
        return;
    for (std::size_t k = 0; k < matrix.size(); ++k)
    {
        const auto dofs = matrix.dofs(k);
        const auto values = matrix.matrix(k);
        for (std::size_t i = 0; i < dofs.size(); ++i)
            for (std::size_t j = 0; j < dofs.size(); ++j)
                into[static_cast<std::size_t>(m_model.dofMap().dofOf(dofs[i]))] += factor *
                    values[i * dofs.size() + j] * vector[static_cast<std::size_t>(m_model.dofMap().dofOf(dofs[j]))];
    }
}

void SizingMatrixRates::add(const Matrices& matrices, double factor, std::span<const double> u,
                           std::span<const double> w, std::span<double> stiffness, std::span<double> mass) const
{
    action(matrices.stiffness, factor, u, stiffness);
    action(matrices.mass, factor * m_weightToMass, w, mass);
}

void SizingMatrixRates::addFirst(std::size_t quantity, std::span<const double> u, std::span<const double> w,
                                std::span<double> stiffness, std::span<double> mass, double factor) const
{
    if (const auto found = m_byQuantity.find(quantity); found != m_byQuantity.end())
        for (const auto index : found->second)
            add(m_first[index].matrices, factor, u, w, stiffness, mass);
}

void SizingMatrixRates::addDirection(std::span<const double> moves, std::span<const double> u,
                                    std::span<const double> w, std::span<double> stiffness,
                                    std::span<double> mass, double factor) const
{
    for (const auto& entry : m_first)
        add(entry.matrices, factor * moves[entry.target.quantity], u, w, stiffness, mass);
}

void SizingMatrixRates::addMixed(std::size_t quantity, std::span<const double> moves, std::span<const double> u,
                                std::span<const double> w, std::span<double> stiffness,
                                std::span<double> mass, double factor) const
{
    const auto found = m_mixedByQuantity.find(quantity);
    if (found == m_mixedByQuantity.end())
        return;
    for (const auto index : found->second)
    {
        const auto& entry = m_mixed[index];
        if (entry.first == quantity)
            add(entry.matrices, factor * moves[entry.second], u, w, stiffness, mass);
        else if (entry.second == quantity)
            add(entry.matrices, factor * moves[entry.first], u, w, stiffness, mass);
    }
}

} // namespace oos::design::detail
