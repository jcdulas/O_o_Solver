#include "DifferentialMatrixRates.hpp"

#include "oos/base/errors/InternalError.hpp"

namespace oos::design::detail
{

DifferentialMatrixRates::DifferentialMatrixRates(const solutions::AnalysisModel& model,
                                                 std::span<const RecoveredFieldTarget> targets,
                                                 const elements::ElementTemperatures* temperatures)
    : m_temperatures(temperatures), m_model(model), m_targets(targets)
{
    for (std::size_t i = 0; i < targets.size(); ++i)
    {
        m_byElement[targets[i].element].push_back(i);
        m_byQuantity[targets[i].quantity].push_back(i);
    }
}

elements::NodalDisplacements DifferentialMatrixRates::nodal(std::span<const double> vector,
                                                            bool includesInitialStrain) const
{
    const assembly::DofMap& dofMap = m_model.dofMap();
    auto state =
        elements::NodalDisplacements::ofGSet(vector, static_cast<std::size_t>(dofMap.gridCount()),
                                             static_cast<std::size_t>(dofMap.scalarCount()), &dofMap.entities());
    if (includesInitialStrain)
        state.temperatures = m_temperatures;
    return state;
}

bool DifferentialMatrixRates::isActive(const RecoveredFieldTarget& target) const
{
    const auto& family = *m_model.elementSet().batches()[target.element.batch].family;
    return !family.isDifferentiallyInert() && family.facets().has(elements::Facet::Differential) &&
           (target.use.changesStiffness || target.use.changesLoads);
}

bool DifferentialMatrixRates::prepare(std::span<const double> preload)
{
    for (const auto& target : m_targets)
    {
        const auto& element = m_model.elementSet().batches()[target.element.batch];
        if (m_temperatures && !element.family->isThermallyInert())
        {
            elements::ElementLoadContext context;
            context.temperatures = m_temperatures;
            const auto append = [&](const elements::ElementVectorBlock& block, NodalLoad& result)
            {
                for (std::size_t k = 0; k < block.size(); ++k)
                    for (std::size_t i = 0; i < block.dofs(k).size(); ++i)
                        result.emplace_back(m_model.dofMap().dofOf(block.dofs(k)[i]), block.values(k)[i]);
            };
            if (target.use.changesLoads)
            {
                elements::ElementVectorBlock first;
                if (!element.family->loadFieldDerivative(element.view, target.element.row, target.field, context,
                                                         first))
                    return false;
                append(first, m_thermalLoads[target.quantity]);
            }
            for (const auto index : m_byElement.at(target.element))
            {
                const auto& other = m_targets[index];
                if (!target.use.changesLoads && !other.use.changesLoads)
                    continue;
                elements::ElementVectorBlock mixed;
                if (!element.family->loadFieldMixedDerivative(element.view, target.element.row, target.field,
                                                              other.field, context, mixed))
                    return false;
                append(mixed, m_mixedThermalLoads[target.quantity][other.quantity]);
            }
        }
        if (!isActive(target))
            continue;
        const auto& batch = m_model.elementSet().batches()[target.element.batch];
        elements::ElementMatrixBlock first;
        if (!batch.family->differentialStiffnessFieldDerivative(batch.view, target.element.row, target.field,
                                                                nodal(preload, true), first))
            return false;
        for (const auto index : m_byElement.at(target.element))
            if (const auto& other = m_targets[index]; isActive(other))
            {
                elements::ElementMatrixBlock mixed;
                if (!batch.family->differentialStiffnessFieldMixedDerivative(
                        batch.view, target.element.row, target.field, other.field, nodal(preload, true), mixed))
                    return false;
            }
    }
    return true;
}

void DifferentialMatrixRates::addThermalLoad(std::size_t quantity, std::span<double> into, double scale) const
{
    if (const auto found = m_thermalLoads.find(quantity); found != m_thermalLoads.end())
        for (const auto& [dof, value] : found->second)
            into[static_cast<std::size_t>(dof)] += scale * value;
}

void DifferentialMatrixRates::addMixedThermalLoad(std::size_t quantity, std::span<const double> moves,
                                                  std::span<double> into) const
{
    if (const auto found = m_mixedThermalLoads.find(quantity); found != m_mixedThermalLoads.end())
        for (const auto& [other, load] : found->second)
            for (const auto& [dof, value] : load)
                into[static_cast<std::size_t>(dof)] += moves[other] * value;
}

void DifferentialMatrixRates::action(const elements::ElementMatrixBlock& matrices, double factor,
                                     std::span<const double> vector, std::span<double> into) const
{
    for (std::size_t k = 0; k < matrices.size(); ++k)
    {
        const auto dofs = matrices.dofs(k);
        const auto matrix = matrices.matrix(k);
        for (std::size_t i = 0; i < dofs.size(); ++i)
            for (std::size_t j = 0; j < dofs.size(); ++j)
                into[static_cast<std::size_t>(m_model.dofMap().dofOf(dofs[i]))] +=
                    factor * matrix[i * dofs.size() + j] *
                    vector[static_cast<std::size_t>(m_model.dofMap().dofOf(dofs[j]))];
    }
}

void DifferentialMatrixRates::addPreload(std::span<const double> preload, std::span<const double> vector,
                                         std::span<double> into) const
{
    for (const auto& batch : m_model.elementSet().batches())
        if (batch.family->facets().has(elements::Facet::Differential) && !batch.family->isDifferentiallyInert())
            for (std::int64_t row = 0; row < batch.view.rowCount(); ++row)
            {
                elements::ElementMatrixBlock matrix;
                batch.family->computeDifferentialStiffness(batch.view, {row, row + 1}, nodal(preload, false), matrix);
                action(matrix, 1.0, vector, into);
            }
}

void DifferentialMatrixRates::addField(std::size_t quantity, std::span<const double> preload,
                                       std::span<const double> vector, std::span<double> into, double factor,
                                       bool includesInitialStrain) const
{
    if (factor == 0.0)
        return;
    const auto found = m_byQuantity.find(quantity);
    if (found == m_byQuantity.end())
        return;
    for (const auto index : found->second)
        if (const auto& target = m_targets[index]; isActive(target))
        {
            const auto& batch = m_model.elementSet().batches()[target.element.batch];
            elements::ElementMatrixBlock matrix;
            if (!batch.family->differentialStiffnessFieldDerivative(batch.view, target.element.row, target.field,
                                                                    nodal(preload, includesInitialStrain), matrix))
                throw base::InternalError("a checked differential matrix field rate became unavailable");
            action(matrix, factor, vector, into);
        }
}

void DifferentialMatrixRates::addDirection(std::span<const double> moves, std::span<const double> preload,
                                           std::span<const double> vector, std::span<double> into,
                                           bool includesInitialStrain) const
{
    for (std::size_t q = 0; q < moves.size(); ++q)
        addField(q, preload, vector, into, moves[q], includesInitialStrain);
}

void DifferentialMatrixRates::addMixed(std::size_t quantity, std::span<const double> moves,
                                       std::span<const double> preload, std::span<const double> vector,
                                       std::span<double> into) const
{
    const auto found = m_byQuantity.find(quantity);
    if (found == m_byQuantity.end())
        return;
    for (const auto index : found->second)
    {
        const auto& target = m_targets[index];
        if (!isActive(target))
            continue;
        const auto& batch = m_model.elementSet().batches()[target.element.batch];
        for (const auto otherIndex : m_byElement.at(target.element))
            if (const auto& other = m_targets[otherIndex]; isActive(other) && moves[other.quantity] != 0.0)
            {
                elements::ElementMatrixBlock matrix;
                if (!batch.family->differentialStiffnessFieldMixedDerivative(
                        batch.view, target.element.row, target.field, other.field, nodal(preload, true), matrix))
                    throw base::InternalError("a checked differential matrix mixed field rate became unavailable");
                action(matrix, moves[other.quantity], vector, into);
            }
    }
}

} // namespace oos::design::detail
