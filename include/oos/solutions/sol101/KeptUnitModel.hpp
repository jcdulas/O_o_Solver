#pragma once

#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/assemble/NonStructuralMassShares.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/elements/core/ElementRegistry.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/reduction/condense/UnitElementIndex.hpp"
#include "oos/reduction/condense/UnitSystem.hpp"

#include <cstdint>
#include <memory>

namespace oos::solutions
{

// What builds the equations of a condensed unit again after its analysis (the load basis handed over in a memory mode
// that releases the units' equations, whose blocks recover the units they read long after the analysis is gone): the
// model the units are cut from (reduction::UnitModel) with everything it points to, kept beyond the analysis. The
// element set is made again on the run's index with the standard element registry (the analysis's own is gone with its
// model): the same batches in the same order, so the same element matrices and the same equations to the bit (checked:
// the batch count; a unit's numbering and its K* are checked again where it is built and factorised again). The index
// of the elements, the units' relations and the non-structural mass shares are the analysis's own (they borrow
// nothing), the DOFs held in every subcase copied.
//
// Thread-safety: immutable once built; concurrent builds are safe. Ownership: owns its element set (which borrows the
// registry, so the object is neither copied nor moved: held by shared_ptr) and the copy of the held flags (budgeted in
// the analysis's budget, which must outlive it); shares the rest.
class KeptUnitModel
{
public:
    // `model`: the analysis's unit model, whose options are copied and whose element set gives the batch count to
    // check. Throws base::InternalError when the element set made again has another batch count, MemoryBudgetError.
    KeptUnitModel(const reduction::UnitModel& model, std::shared_ptr<const elements::SnapshotIndex> index,
                  std::shared_ptr<const assembly::DofMap> dofMap,
                  std::shared_ptr<const reduction::UnitElementIndex> elementIndex,
                  std::shared_ptr<const assembly::RigidElimination> unitRelations,
                  std::shared_ptr<const assembly::NonStructuralMassShares> shares, base::MemoryBudget& budget);
    KeptUnitModel(const KeptUnitModel&) = delete;
    KeptUnitModel& operator=(const KeptUnitModel&) = delete;
    KeptUnitModel(KeptUnitModel&&) = delete;
    KeptUnitModel& operator=(KeptUnitModel&&) = delete;
    ~KeptUnitModel() = default;

    [[nodiscard]] const reduction::UnitModel& model() const noexcept { return m_model; }

private:
    std::shared_ptr<const elements::SnapshotIndex> m_index;
    std::shared_ptr<const assembly::DofMap> m_dofMap;
    elements::ElementRegistry m_registry;
    std::unique_ptr<assembly::ElementSet> m_elements; // borrows m_registry and the index
    std::shared_ptr<const reduction::UnitElementIndex> m_elementIndex;
    std::shared_ptr<const assembly::RigidElimination> m_unitRelations;
    std::shared_ptr<const assembly::NonStructuralMassShares> m_shares;
    base::BudgetedBuffer<std::uint8_t> m_isHeldEverywhere;
    reduction::UnitModel m_model;
};

} // namespace oos::solutions
