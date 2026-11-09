#pragma once

#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/reduction/absorption/SpiderChangeOfBasis.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace oos::reduction
{

// The relation of one spider on the g set: its weighting matrix and the DOFs of its rows and columns.
struct SpiderRelation
{
    SpiderWeightMatrix weights;
    // The g-set DOFs of the rows of W: REFC of a RigidInterpolation; empty for a fastener side, whose rows are the
    // motion of its piercing point P in the basic system, translations then rotations (FastenerPiercing).
    std::vector<base::DofIndex> referenceDofs;
    // The g-set DOFs of the columns of W, increasing: the leg DOFs the relation uses.
    std::vector<base::DofIndex> legDofs;
};

// The rows of Gm by the element they come from (rigid elements and fastener internal constraints), so that the
// weighting matrix of each spider is found without scanning Gm once per spider.
//
// Thread-safety: immutable once built. Ownership: owns the index (budgeted: one entry per row of Gm); borrows Gm and
// the DOF map, which must outlive it.
class SpiderWeights
{
public:
    SpiderWeights(const assembly::RigidElimination& gm, const assembly::DofMap& dofMap, base::MemoryBudget& budget);

    // The weighting matrix of a RigidInterpolation from its rows of Gm (the REFC DOFs of REFGRID as combinations of
    // the leg DOFs: RigidInterpolation, u_ref = (Sᵀ·W·S)⁻¹·Sᵀ·W·u_legs, assembly::RigidElementEquations), the
    // characteristic length the mean distance of the legs to REFGRID. Nothing when Gm has no row of it (the element was
    // refused, or its REFC DOFs are constrained in every subcase and the planner left them out). The rows are taken as
    // Gm resolves them: for an eligible spider no leg is dependent elsewhere, so their terms are on the leg DOFs alone.
    [[nodiscard]] std::optional<SpiderRelation> rigidInterpolationRelation(base::EntityId elementId) const;

    // The weighting matrix of the side of a Fastener or Weld (`source` Fastener or Weld) whose GA or GB
    // is the grid point `pointNid` of the model, from that point's rows of Gm (assembly::FastenerConstraints: u_GA =
    // A·M·u_grids, the interpolation of the piercing point carried rigidly to GA): the reference DOFs are GA's six, the
    // legs the DOFs of the pierced element's grid points the rows use (FastenerPiercing documents it), the
    // characteristic length the mean distance of those grid points to GA. Nothing when Gm has no row of that point for
    // the fastener.
    [[nodiscard]] std::optional<SpiderRelation> fastenerPointRelation(base::EntityId elementId,
                                                                      assembly::ConstraintSource source,
                                                                      base::EntityId pointNodeId) const;

    // The DOFs an RigidBodyStar makes dependent (its legs' CM components), increasing; empty when Gm has none.
    [[nodiscard]] std::vector<base::DofIndex> dependentDofs(base::EntityId elementId,
                                                            assembly::ConstraintSource source) const;

    [[nodiscard]] const assembly::DofMap& dofMap() const noexcept { return *m_dofMap; }

private:
    // The rows of Gm of an element, increasing.
    [[nodiscard]] std::vector<std::size_t> rowsOf(base::EntityId elementId, assembly::ConstraintSource source) const;
    // The relation of some rows of Gm: their dependent DOFs the reference, their terms the legs.
    [[nodiscard]] SpiderRelation relationOfRows(const std::vector<std::size_t>& rows) const;

    const assembly::RigidElimination* m_gm;
    const assembly::DofMap* m_dofMap;
    // (source, element ID, row), sorted: rows of rigid elements and fastener constraints (not the MPC equations).
    struct RowKey
    {
        std::uint8_t source = 0;
        base::EntityId elementId = 0;
        std::int64_t row = 0;
    };
    base::BudgetedBuffer<RowKey> m_index;
    std::size_t m_indexSize = 0;
};

// The kind of a g-set DOF: components 4 to 6 of a grid point are rotations, the others (translations, scalar points)
// lengths.
[[nodiscard]] SpiderDofKind dofKindOf(const assembly::DofMap& dofMap, base::DofIndex dof) noexcept;

} // namespace oos::reduction
