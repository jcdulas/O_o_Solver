#pragma once

#include "oos/assembly/assemble/ElementConnectivity.hpp"
#include "oos/assembly/assemble/ElementSet.hpp"
#include "oos/assembly/constraints/FastenerConstraints.hpp"
#include "oos/assembly/constraints/GlueConstraints.hpp"
#include "oos/assembly/constraints/MpcEquations.hpp"
#include "oos/assembly/constraints/MsetCouplings.hpp"
#include "oos/assembly/constraints/RigidElimination.hpp"
#include "oos/assembly/constraints/SpcSet.hpp"
#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/dofs/DofSets.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/design/sensitivity/BoundaryReducedSensitivity.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/SparsePattern.hpp"
#include "oos/solutions/common/FactorisedSet.hpp"
#include "oos/solutions/common/FsetSolver.hpp"
#include "oos/solutions/sol101/CondensedFactorisedSet.hpp"
#include "oos/solutions/sol101/StaticSubcases.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace oos::design
{

// The static system of one constraint set of a design point, factorised for the sensitivity solves (the
// direct method solves K⁻¹·B, the adjoint method solves K⁻¹·Cᵀ): the same reduction as the static solution
// (solutions/sol101/StaticSolution steps 3 and 4: the m set of the rigid elements, fasteners and MPC equations
// eliminated by Gm, the single-point constraints and the AUTOSPC of the grid point singularity processor, the r set of
// FreeBodySupport held as constraints) and the l-set factor K_ll.
//
// The design variables never drive Gm, the constrained DOFs nor the enforced displacements (the relations design
// property, material and element fields, never rigid elements, MPC coefficients or SPC values), so with
// u_g = T·u_l + u_e (T: the l set to the g set through Gm, zero on the s and r sets; u_e the enforced displacements):
//   K_ll·du_l = (Tᵀ·b_g)_l,  b_g = dP_g − dK_gg·u_g   (the pseudo-load: reduceToN then the l rows),
//   du_g = T·du_l                                      (expand: zero on the s and r sets, Gm on the m set).
// toEquations() applies Tᵀ (reduceToN, then the l rows in the order of the factor), fromEquations() applies T.
//
// Inertia relief (PARAM INREL = -1) on rigid support motions only (refused otherwise by the constructor
// with an OptimisationError naming the subcase). The SUBCOM combinations are no subcase of a system: the sensitivities
// solve a SUBCOM on the system of its terms, and the constructor refuses one (std::invalid_argument).
//
// Two ways to the factor: on the constraint set the analysis of the design point kept (solutions::FactorisedSet,
// StaticOptions::keepsSystems: no factorisation, factorisationCount() = 0; the same matrix, so the same bits as a
// factor made again), or built here as the static solution builds it (factorisationCount() = 1). Every solve is a
// block of right-hand sides on it.
//
// A third way, the condensed path: on the condensed constraint set the analysis kept
// (solutions::CondensedFactorisedSet), the solves go through the boundaries (BoundaryReducedSensitivity: the units
// loaded condensed, the interface solved, the units read recovered). There the "equations" are the g set itself:
// toEquations() and fromEquations() copy, and solve() gives T·K⁻¹·Tᵀ·v directly; `isReadDof` of solve() names the DOFs
// the caller reads of the solution (the units not read are not recovered). Kgg and its pattern, which the condensed
// path never assembles, are assembled on the design point's element set at the first call of kgg() or pattern() (the
// responses that read them: SPC forces, contour forces with constraint rows, ESE percentages, compliances, inertial
// load derivatives).
//
// Thread-safety: used on one thread (solve() counts its columns; kgg() and pattern() made on first use). Ownership:
// owns everything it built, or shares the kept set; borrows the DOF map (and the snapshot its node table belongs to),
// the element set of the condensed path and the budget, which must outlive it.
class SensitivitySystem
{
public:
    // `subcases`: the subcases of one constraint set (the same MPC set, the same constrained DOFs, the same
    // FreeBodySupport), as StaticSubcases gives them on the snapshot of `elementSet`, no SUBCOM. Throws
    // OptimisationError (inertia relief), SolverError (problems of the model, as the analysis), SingularityError,
    // MemoryBudgetError.
    SensitivitySystem(const assembly::ElementSet& elementSet, const assembly::DofMap& dofMap,
                      std::vector<solutions::StaticSubcase> subcases, linalg::KernelKind kernel,
                      base::MemoryBudget& budget, base::ThreadPool* pool = nullptr);
    // On the constraint set the analysis kept: `subcases` among those of `kept` (std::invalid_argument otherwise),
    // the solves with the kernels of `kernel` on `pool`. Throws OptimisationError as the other constructor.
    SensitivitySystem(std::shared_ptr<const solutions::FactorisedSet> kept,
                      std::vector<solutions::StaticSubcase> subcases, linalg::KernelKind kernel,
                      base::MemoryBudget& budget, base::ThreadPool* pool = nullptr);
    // On the condensed constraint set the analysis kept, `elementSet` the design point's (for Kgg on
    // demand). Throws as the other constructors.
    SensitivitySystem(std::shared_ptr<const solutions::CondensedFactorisedSet> kept,
                      const assembly::ElementSet& elementSet, std::vector<solutions::StaticSubcase> subcases,
                      linalg::KernelKind kernel, base::MemoryBudget& budget, base::ThreadPool* pool = nullptr);
    SensitivitySystem(const SensitivitySystem&) = delete;
    SensitivitySystem& operator=(const SensitivitySystem&) = delete;
    SensitivitySystem(SensitivitySystem&&) = delete;
    SensitivitySystem& operator=(SensitivitySystem&&) = delete;
    ~SensitivitySystem();

    [[nodiscard]] const std::vector<solutions::StaticSubcase>& subcases() const noexcept { return m_subcases; }
    [[nodiscard]] const assembly::DofMap& dofMap() const noexcept { return *m_dofMap; }
    [[nodiscard]] const assembly::RigidElimination& gm() const noexcept { return *m_gmView; }
    // The DOF sets of the factor; precondition: not on the condensed path (isCondensed()), which has no set of the
    // whole model (isConstrained() answers there).
    [[nodiscard]] const assembly::DofSets& sets() const noexcept { return *m_setsView; }
    // Kgg of the design point (every element), for the products K·w of the SPC force responses; on the condensed path
    // assembled at the first call (with the external units' K*). Throws MemoryBudgetError there.
    [[nodiscard]] const linalg::BlockSparseMatrix& kgg() const;
    [[nodiscard]] const linalg::SparsePattern& pattern() const;
    [[nodiscard]] std::int32_t equationCount() const noexcept
    {
        if (m_condensed)
            return m_dofMap->gSetSize();
        return m_kept ? m_kept->equationCount() : m_solver->equationCount();
    }
    // Whether a g-set DOF is held (s set, AUTOSPC, r set of FreeBodySupport): its SPC force is a result, its
    // displacement fixed.
    [[nodiscard]] bool isConstrained(base::DofIndex dof) const noexcept;

    // Tᵀ·v: `vectorG` (a load or adjoint weight on the g set, not modified) reduced to the n set (P_n + Gmᵀ·P_m) and
    // its l rows written to `equations` (equationCount() values, the factor's order). A copy on the condensed path.
    void toEquations(std::span<const double> vectorG, std::span<double> equations) const;
    // The same, `vectorG` reduced in place (its values are lost): no copy of the g set per column.
    void toEquationsInPlace(std::span<double> vectorG, std::span<double> equations) const;
    // T·w: `vectorG` set to the g-set expansion of `equations`: the l rows, zero on the s and r sets, Gm on the m set.
    // A copy on the condensed path.
    void fromEquations(std::span<const double> equations, std::span<double> vectorG) const;

    // B ← K_ll⁻¹·B (equationCount() rows), counted in solveCount(). On the condensed path B ← T·K⁻¹·Tᵀ·B on the g set,
    // the interiors of the units no flag of `isReadDof` (one per g-set DOF) reads left at zero (empty: every unit
    // read); the full path ignores `isReadDof`.
    void solve(linalg::DenseMatrixView rightHandSides, std::span<const std::uint8_t> isReadDof = {}) const;
    [[nodiscard]] std::int64_t solveCount() const noexcept { return m_solveCount; }
    // The factorisations this system made: 0 on a kept set, 1 otherwise.
    [[nodiscard]] std::int32_t factorisationCount() const noexcept { return m_kept || m_condensed ? 0 : 1; }

    // The condensed path: its solves and their statistics, null on the full path.
    [[nodiscard]] bool isCondensed() const noexcept { return m_condensed != nullptr; }
    // The inertia relief of the constraint set (PARAM INREL = −1), null without one.
    [[nodiscard]] const solutions::KeptRelief* relief() const noexcept
    {
        return m_kept ? m_kept->relief() : m_condensedRelief.get();
    }
    // The accelerations ü_r of a subcase under that relief (nothing without one, or for a subcase the set did not
    // solve).
    [[nodiscard]] std::optional<std::array<double, assembly::MAX_SUPPORT_DOFS>> reliefAccelerations(
        base::EntityId subcase) const
    {
        const solutions::KeptRelief* kept = relief();
        if (kept == nullptr)
            return std::nullopt;
        const std::vector<base::EntityId>& solved = m_kept ? m_kept->subcases() : m_condensedSet->subcases();
        for (std::size_t k = 0; k < solved.size() && k < kept->accelerations.size(); ++k)
            if (solved[k] == subcase)
                return kept->accelerations[k];
        return std::nullopt;
    }
    [[nodiscard]] const BoundaryReducedSensitivity* condensed() const noexcept { return m_condensed.get(); }
    // The g-set DOF of an equation of the factor (the l rows toEquations() writes); precondition: the full path.
    [[nodiscard]] base::DofIndex dofOfEquation(std::int32_t equation) const noexcept
    {
        return m_kept ? m_kept->dofOfEquation(equation) : m_solver->dofOfEquation(equation);
    }

private:
    // The condensed path: Kgg and its pattern on the design point's element set, made once.
    void assembleStiffness() const;

    const assembly::DofMap* m_dofMap = nullptr;
    base::MemoryBudget* m_budget;
    std::vector<solutions::StaticSubcase> m_subcases;
    std::unique_ptr<linalg::DenseKernels> m_kernels;
    std::shared_ptr<const solutions::FactorisedSet> m_kept; // null when the system made its own factor
    // The condensed path: the solves through the boundaries, the kept set, the element set of Kgg.
    std::unique_ptr<BoundaryReducedSensitivity> m_condensed;
    std::shared_ptr<const solutions::CondensedFactorisedSet> m_condensedSet;
    std::shared_ptr<const solutions::KeptRelief> m_condensedRelief; // the condensed set's relief, on the g set
    const assembly::ElementSet* m_elementSet = nullptr;
    base::ThreadPool* m_pool = nullptr;
    mutable std::unique_ptr<assembly::ElementConnectivity> m_lazyConnectivity;
    mutable std::unique_ptr<assembly::MsetCouplings> m_lazyCouplings;
    mutable std::unique_ptr<linalg::NodeGraph> m_lazyGraph;
    mutable std::unique_ptr<linalg::SparsePattern> m_lazyPattern;
    mutable std::unique_ptr<linalg::BlockSparseMatrix> m_lazyKgg;
    // The parts in use, those of the kept set or those below (not owned).
    const assembly::RigidElimination* m_gmView = nullptr;
    const assembly::DofSets* m_setsView = nullptr;
    const linalg::BlockSparseMatrix* m_kggView = nullptr;
    const linalg::SparsePattern* m_patternView = nullptr;
    std::unique_ptr<assembly::MpcEquations> m_mpc;
    std::unique_ptr<assembly::FastenerConstraints> m_fasteners;
    std::unique_ptr<assembly::GlueConstraints> m_glue;
    std::unique_ptr<assembly::RigidElimination> m_gm;
    std::unique_ptr<assembly::SpcSet> m_spc;
    std::unique_ptr<assembly::DofSets> m_solveSets;
    std::unique_ptr<assembly::ElementConnectivity> m_connectivity;
    std::unique_ptr<assembly::MsetCouplings> m_couplings;
    std::unique_ptr<linalg::NodeGraph> m_graph;
    std::unique_ptr<linalg::SparsePattern> m_pattern;
    std::unique_ptr<linalg::BlockSparseMatrix> m_kgg;
    std::unique_ptr<linalg::BlockSparseMatrix> m_knn;
    std::unique_ptr<solutions::FsetSolver> m_solver;
    mutable std::int64_t m_solveCount = 0;
};

} // namespace oos::design
