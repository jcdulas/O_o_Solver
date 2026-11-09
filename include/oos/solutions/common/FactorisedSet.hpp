#pragma once

#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/dofs/DofSets.hpp"
#include "oos/base/ids/EntityId.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/elements/core/SnapshotIndex.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/kernels/KernelKind.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/SparsePattern.hpp"
#include "oos/solutions/common/FsetSolver.hpp"
#include "oos/solutions/common/KeptRelief.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

namespace oos::solutions
{

// The parts a FactorisedSet keeps, as the static solution hands them over.
struct FactorisedSetParts
{
    std::shared_ptr<const elements::SnapshotIndex> index; // the points the DOF map refers to
    std::shared_ptr<const assembly::DofMap> dofMap;
    std::shared_ptr<const assembly::RigidElimination> gm;
    std::shared_ptr<const assembly::DofSets> solveSets;
    std::shared_ptr<const linalg::SparsePattern> pattern; // Kgg's
    std::shared_ptr<const linalg::BlockSparseMatrix> kgg;
    std::unique_ptr<const FsetSolver> solver; // the l-set factor of the run
    linalg::KernelKind kernel = linalg::KernelKind::Reference;
    std::vector<base::EntityId> subcases; // the subcases solved on it, in solve order
    // The inertia relief of the set (PARAM INREL = −1), null without one.
    std::shared_ptr<const KeptRelief> relief;
};

// One constraint set of a static run kept with its factor after the run (StaticOptions::keepsSystems,
// StaticResults::systems): the DOF map, Gm, the solved sets, Kgg and the l-set factor K_ll, with the subcases solved on
// it. The static sensitivities of a design point solve on it instead of factorising the same matrix again
// (design/sensitivity/SensitivitySystem: the direct method solves K⁻¹·B, the adjoint method solves K⁻¹·Cᵀ).
//
// The solves take the caller's kernels: the kernels and the pool of the run that factorised may be gone; the same
// kernel kind gives the same bits whatever the pool (FsetSolver).
//
// Memory: every part stays reserved in the run's memory budget until the last owner drops the set (the budget must
// outlive it), so a kept set counts against the later work of the budget.
//
// Thread-safety: immutable; solve() may run concurrently. Ownership: shares its parts; borrows, through the DOF map
// and the index, the snapshot of the run, which must outlive it.
class FactorisedSet
{
public:
    // Throws std::invalid_argument when a part is missing.
    explicit FactorisedSet(FactorisedSetParts parts);

    [[nodiscard]] const elements::SnapshotIndex& index() const noexcept { return *m_parts.index; }
    [[nodiscard]] const assembly::DofMap& dofMap() const noexcept { return *m_parts.dofMap; }
    [[nodiscard]] const assembly::RigidElimination& gm() const noexcept { return *m_parts.gm; }
    [[nodiscard]] const assembly::DofSets& solveSets() const noexcept { return *m_parts.solveSets; }
    [[nodiscard]] const linalg::SparsePattern& pattern() const noexcept { return *m_parts.pattern; }
    [[nodiscard]] const linalg::BlockSparseMatrix& kgg() const noexcept { return *m_parts.kgg; }
    [[nodiscard]] linalg::KernelKind kernel() const noexcept { return m_parts.kernel; }
    [[nodiscard]] const std::vector<base::EntityId>& subcases() const noexcept { return m_parts.subcases; }
    // The inertia relief of the set (null without one).
    [[nodiscard]] const KeptRelief* relief() const noexcept { return m_parts.relief.get(); }
    [[nodiscard]] bool solves(base::EntityId subcase) const
    {
        return std::find(m_parts.subcases.begin(), m_parts.subcases.end(), subcase) != m_parts.subcases.end();
    }

    [[nodiscard]] std::int32_t equationCount() const noexcept { return m_parts.solver->equationCount(); }
    [[nodiscard]] base::DofIndex dofOfEquation(std::int32_t equation) const noexcept
    {
        return m_parts.solver->dofOfEquation(equation);
    }
    // B ← K_ll⁻¹·B (equationCount() rows, the factor's order) with `kernels`.
    void solve(linalg::DenseMatrixView rightHandSides, const linalg::DenseKernels& kernels,
               const base::CancellationToken* cancellation = nullptr) const
    {
        m_parts.solver->solve(rightHandSides, kernels, cancellation);
    }

    // The pivot diagnostics of the factorisation (the largest MAXRATIO ratio and its equation), and the grid point and
    // component of an equation: what the numerical diagnostics of a design point report (design/aids).
    [[nodiscard]] const linalg::PivotDiagnostics& pivotDiagnostics() const noexcept
    {
        return m_parts.solver->pivotDiagnostics();
    }
    [[nodiscard]] base::GridComponent nameOfEquation(std::int32_t equation) const
    {
        return m_parts.solver->nameOf(equation);
    }

private:
    FactorisedSetParts m_parts;
};

} // namespace oos::solutions
