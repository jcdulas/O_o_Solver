#pragma once

#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/output/arrays/StiffnessRecovery.hpp"
#include "oos/solutions/sol101/KeptLoadBasis.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace oos::solutions
{

// The fixed rows of the units' grid point forces of a load basis (KeptFixedRows): what a constraint set keeps, chosen
// before its vectors are made (plan), each unit's applied loads taken as they are made (take), and the rows made at a
// recovery from the unit's element forces at the points (rowsOf), as statics makes appliedLoad, singlePointConstrForce
// and multiPointConstrForce (RigidElimination::computeMpcForces, its rows in its order):
//     q_m = (K·u − P)_m,   q_n = −Gmᵀ·q_m,   q_s = (K·u − P − q)_s,
// K·u the sum of the elements' forces at the point (StiffnessRecovery::pointForces gives −K_e·u_e). For a reader of
// O_o_Reader's data contract 56 (BasisFixedRowRequest::isMpcSplit) the multiPointConstrForce rows are split by relation
// as statics splits them (RigidElimination::forEachRelationForce): the kept relations then include the rows chained to
// them, and each kept row's relation and own terms are kept.
//
// Thread-safety: stateless; take() writes the column of its unit only (vectors made in parallel). Ownership: none.
class BasisFixedRows
{
public:
    // What the set keeps for `request` (null when nothing is requested): with isAll every DOF and every row of `gm`;
    // otherwise the points of the request, the rows of `gm` whose m-set DOF or a term reaches one of them or whose
    // points straddle a plane of the request, every point of those rows, and the DOFs of these points. `isConstrained`:
    // one flag per g-set DOF (the s and r sets). Throws MemoryBudgetError.
    [[nodiscard]] static std::shared_ptr<KeptFixedRows> plan(const BasisFixedRowRequest& request,
                                                             const assembly::DofMap& dofMap,
                                                             const assembly::RigidElimination& gm,
                                                             std::span<const std::uint8_t> isConstrained,
                                                             std::size_t units, base::MemoryBudget& budget);
    // The bytes plan() keeps for `request` at most (the memory plan of the load basis, before the relations are known):
    // with isAll the loads of every DOF, otherwise those of the points asked (a few relations' points beside).
    [[nodiscard]] static std::uint64_t estimateBytes(const BasisFixedRowRequest& request, std::uint64_t gSize,
                                                     std::uint64_t units) noexcept;
    // Keeps the applied loads `loadsG` (a g-set vector) of unit `unit`.
    static void take(KeptFixedRows& rows, std::size_t unit, std::span<const double> loadsG) noexcept;

    // The fixed rows of unit `unit` at the nodes of `rowNodes` (increasing) that keep them (every one with isAll), from
    // column `column` of `forces` (the element forces at rowNodes and at the kept relations' dependent points) and the
    // kept loads, plus `addedLoadsG` when given (a g-set vector: the relief loads of the interiors recovered on the
    // condensed path). `work` holds the per-DOF sums between calls (sized here, in `budget`). Throws
    // MemoryBudgetError.
    struct Work
    {
        base::BudgetedBuffer<double> stiffness; // K·u per kept DOF
        base::BudgetedBuffer<double> loads;     // P per kept DOF
        base::BudgetedBuffer<double> mpc;       // q per kept DOF
    };
    [[nodiscard]] static std::vector<output::StiffnessRecovery::PointFixedRows> rowsOf(
        const KeptFixedRows& rows, std::size_t unit, std::span<const std::int32_t> rowNodes,
        const output::StiffnessRecovery::PointForces& forces, std::size_t column, std::span<const double> addedLoadsG,
        const assembly::DofMap& dofMap, base::MemoryBudget& budget, Work& work);

    // The nodes whose element forces rowsOf() needs beside rowNodes: the kept relations' dependent points.
    [[nodiscard]] static std::span<const std::int32_t> forceNodes(const KeptFixedRows& rows) noexcept
    {
        return rows.mNodes.span();
    }
};

} // namespace oos::solutions
