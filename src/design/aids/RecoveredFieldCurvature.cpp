#include "RecoveredFieldCurvature.hpp"

#include "oos/base/errors/OptimisationError.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace oos::design::detail
{

RecoveredFieldCurvature::RecoveredFieldCurvature(const solutions::AnalysisModel& model,
                                         std::shared_ptr<const input::ModelSnapshot> snapshot,
                                         base::MemoryBudget& budget)
    : m_model(model), m_snapshot(std::move(snapshot)), m_budget(budget),
      m_weights(budget, static_cast<std::size_t>(model.dofMap().gSetSize()), "recovered field curvature weights")
{
    std::fill(m_weights.span().begin(), m_weights.span().end(), 0.0);
}

bool RecoveredFieldCurvature::prepare(std::span<const WeightedRecoveredField> fields,
                                  std::span<const RecoveredFieldTarget> targets, std::span<const double> shape)
{
    for (const auto& field : fields)
        if (!add(field, targets, shape))
            return false;
    std::uint64_t bytes = 0;
    for (const auto& block : m_displacements)
        bytes += block.dofs.capacity() * sizeof(elements::ElementDof) + block.values.capacity() * sizeof(double);
    for (const auto& block : m_mixed)
        bytes += block.dofs.capacity() * sizeof(elements::ElementDof) + block.values.capacity() * sizeof(double);
    m_reservation = m_budget.reserve(bytes, "recovered field local curvature blocks");
    return true;
}

bool RecoveredFieldCurvature::add(const WeightedRecoveredField& field, std::span<const RecoveredFieldTarget> targets,
                             std::span<const double> shape)
{
    if (field.response->definition().type == ResponseType::Displacement)
    {
        const auto dof = m_model.dofMap().findDof(field.label.entity, field.label.component);
        if (!dof)
            return false;
        m_weights[static_cast<std::size_t>(*dof)] += field.weight;
        return true;
    }
    const auto& elements = field.response->elementTargets();
    const auto target = std::find_if(elements.begin(), elements.end(), [&](const auto& candidate)
                                     { return candidate.elementId == field.label.entity; });
    if (target == elements.end())
        return false;
    const auto location = std::find_if(target->items.begin(), target->items.end(), [&](const auto& candidate)
                                       { return candidate.first == field.label.component; });
    if (location == target->items.end())
        return false;
    for (std::size_t batch = 0; batch < m_model.elementSet().batches().size(); ++batch)
        if (m_model.elementSet().batches()[batch].view.card() == target->card)
            return addRecovery(field, {batch, target->row},
                {field.response->definition().type, field.label.component, location->second, field.label.layer},
                targets, shape);
    return false;
}

bool RecoveredFieldCurvature::addRecovery(const WeightedRecoveredField& field, const ElementRef& element,
                                      const ElementItemSpec& spec, std::span<const RecoveredFieldTarget> targets,
                                      std::span<const double> shape)
{
    const auto& batch = m_model.elementSet().batches()[element.batch];
    const ElementItemFunction function(*batch.family, element.row, {spec});
    const auto grid = static_cast<std::size_t>(m_model.dofMap().gridCount()) * 6;
    const elements::NodalDisplacements displacements{shape.first(grid), shape.subspan(grid), nullptr};
    base::BudgetedBuffer<double> copy(m_budget, shape.size(), "recovered field curvature recovery scratch");
    const auto scratch = [&]
    {
        std::copy(shape.begin(), shape.end(), copy.data());
        return ScratchDisplacements{copy.span().first(grid), copy.span().subspan(grid)};
    };
    const auto slopes = function.slopes(batch.view, m_snapshot, displacements, scratch, m_budget);
    if (!slopes)
        return false;
    if (std::abs(slopes->values.front() - field.value) > 1e-7 * std::max(1.0, std::abs(field.value)))
        throw base::OptimisationError("recovered field curvature does not reproduce the evaluated response");
    for (std::size_t d = 0; d < slopes->dofs.size(); ++d)
        m_weights[static_cast<std::size_t>(m_model.dofMap().dofOf(slopes->dofs[d]))] +=
            field.weight * slopes->weights[d];
    const std::array<double, 1> weights{field.weight};
    const auto base = function.secondOrder(batch.view, displacements, weights, {}, scratch);
    if (!base)
        return false;
    if (!base->displacementBlock.empty())
        m_displacements.push_back({base->dofs, base->displacementBlock});
    std::vector<const RecoveredFieldTarget*> reached;
    for (const auto& target : targets)
        if (target.element == element && target.use.changesRecovery)
        {
            const std::span<const elements::DesignedField> designed(&target.field, 1);
            const auto part = function.secondOrder(batch.view, displacements, weights, designed, scratch);
            if (!part)
                return false;
            m_mixed.push_back({target.quantity, part->dofs, part->mixedBlock});
            m_fieldPairs[{target.quantity, target.quantity}] += part->fieldBlock.front();
            reached.push_back(&target);
        }
    for (std::size_t a = 0; a < reached.size(); ++a)
        for (std::size_t b = a + 1; b < reached.size(); ++b)
        {
            const auto& first = *reached[a];
            const auto& second = *reached[b];
            const auto value = function.mixedFieldRate(batch.view, displacements, weights, first.field, second.field);
            if (!value)
                return false;
            m_fieldPairs[{first.quantity, second.quantity}] += *value;
            m_fieldPairs[{second.quantity, first.quantity}] += *value;
        }
    return true;
}

void RecoveredFieldCurvature::weightRate(std::span<const double> shapeRate, std::span<const double> fieldMoves,
                                     std::span<double> into) const
{
    std::fill(into.begin(), into.end(), 0.0);
    for (const auto& block : m_displacements)
        for (std::size_t i = 0; i < block.dofs.size(); ++i)
            for (std::size_t j = 0; j < block.dofs.size(); ++j)
                into[static_cast<std::size_t>(m_model.dofMap().dofOf(block.dofs[i]))] +=
                    block.values[i * block.dofs.size() + j] *
                    shapeRate[static_cast<std::size_t>(m_model.dofMap().dofOf(block.dofs[j]))];
    for (const auto& block : m_mixed)
        for (std::size_t i = 0; i < block.dofs.size(); ++i)
            into[static_cast<std::size_t>(m_model.dofMap().dofOf(block.dofs[i]))] +=
                block.values[i] * fieldMoves[block.quantity];
}

void RecoveredFieldCurvature::addExplicitRate(std::span<const double> shapeRate, std::span<const double> fieldMoves,
                                          std::span<double> into) const
{
    for (const auto& block : m_mixed)
        for (std::size_t i = 0; i < block.dofs.size(); ++i)
            into[block.quantity] +=
                block.values[i] * shapeRate[static_cast<std::size_t>(m_model.dofMap().dofOf(block.dofs[i]))];
    for (const auto& [pair, value] : m_fieldPairs)
        into[pair.first] += value * fieldMoves[pair.second];
}

} // namespace oos::design::detail
