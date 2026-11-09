#include "oos/solutions/sol101/KeptUnitModel.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/base/errors/InternalError.hpp"

#include <string>
#include <utility>

namespace oos::solutions
{

KeptUnitModel::KeptUnitModel(const reduction::UnitModel& model, std::shared_ptr<const elements::SnapshotIndex> index,
                             std::shared_ptr<const assembly::DofMap> dofMap,
                             std::shared_ptr<const reduction::UnitElementIndex> elementIndex,
                             std::shared_ptr<const assembly::RigidElimination> unitRelations,
                             std::shared_ptr<const assembly::NonStructuralMassShares> shares,
                             base::MemoryBudget& budget)
    : m_index(std::move(index)),
      m_dofMap(std::move(dofMap)),
      m_registry(elements::ElementRegistry::forModel(m_index->snapshot())),
      m_elementIndex(std::move(elementIndex)),
      m_unitRelations(std::move(unitRelations)),
      m_shares(std::move(shares)),
      m_isHeldEverywhere(budget, model.isConstrainedEverywhere.size(), "kept unit model held DOFs"),
      m_model(model)
{
    if (!m_index || !m_dofMap || !m_elementIndex || !m_unitRelations || model.elements == nullptr)
        throw base::InternalError(
            "KeptUnitModel: expected the index, the DOF map, the element index, the units' relations and the "
            "analysis's element set (an internal error: report it)");
    m_elements = std::make_unique<assembly::ElementSet>(*m_index, m_registry);
    if (m_elements->batches().size() != model.elements->batches().size())
        throw base::InternalError(
            base::concatenate({"KeptUnitModel: the element set made again for the units' equations has ",
                               std::to_string(m_elements->batches().size()), " batches where the analysis's had ",
                               std::to_string(model.elements->batches().size()), " (an internal error: report it)"}));
    for (std::size_t dof = 0; dof < m_isHeldEverywhere.size(); ++dof)
        m_isHeldEverywhere[dof] = model.isConstrainedEverywhere[dof];
    m_model.elements = m_elements.get();
    m_model.dofMap = m_dofMap.get();
    m_model.elementIndex = m_elementIndex.get();
    m_model.unitRelations = m_unitRelations.get();
    m_model.isConstrainedEverywhere = m_isHeldEverywhere.span();
    m_model.nonStructuralMass = m_shares.get();
}

} // namespace oos::solutions
