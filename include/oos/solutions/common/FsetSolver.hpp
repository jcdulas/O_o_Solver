#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/dofs/DofSets.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ProgressCounter.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/dense/DenseKernels.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/ordering/ConstrainedOrdering.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/sparse/BlockSparseMatrix.hpp"
#include "oos/linalg/sparse/DofSelection.hpp"
#include "oos/linalg/sparse/SymmetricCsc.hpp"
#include "oos/linalg/supernodal/PivotDiagnostics.hpp"
#include "oos/linalg/supernodal/SupernodalFactor.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"
#include "oos/solutions/common/FsetOrdering.hpp"
#include "oos/solutions/common/StructureCache.hpp"

#include <cstdint>
#include <memory>

namespace oos::solutions
{

// How the f-set stiffness is ordered and factorised.
struct FsetSolverOptions
{
    linalg::PivotPolicy pivotPolicy; // PARAM MAXRATIO, BAILOUT
    linalg::OrderingMethod ordering = linalg::OrderingMethod::Automatic;
    const base::CancellationToken* cancellation = nullptr; // not owned, may be null
    base::ProgressCounter* progress = nullptr;             // advanced by the columns eliminated; may be null
    // The DOFs factorised: the l set (the f set without the r set of FreeBodySupport; the o set is refused), which is
    // the f set when there is no FreeBodySupport.
    assembly::DofSet keptSet = assembly::DofSet::L;
    // The structure kept between the analyses of the model: the ordering, the symbolic factor and the plan of the
    // solves taken from it when an earlier analysis built them from the same graph, sets and positions, else built and
    // kept there; their bytes reserved in the solver's budget while it lives. Null: built for this solver alone. Not
    // owned.
    StructureCache* structures = nullptr;
};

// What a factorisation took from the structure cache (FsetSolverOptions::structures): whether its ordering and its
// plan of the solves were built by an earlier analysis, and whether its panels are the memory of an earlier factor.
struct FsetReuse
{
    bool isOrderingReused = false;
    bool isPlanReused = false;
    bool isStorageReused = false;
};

// The factorisation of the f-set stiffness Kff and the solves with it, on the full model: the free DOFs are ordered and
// numbered by FsetOrdering, Knn converted to the SymmetricCsc of those DOFs, and factorised by SupernodalFactor in
// positive definite mode with the MAXRATIO test.
//
// A mechanism or a failed ratio test is a SingularityError naming the grid point (or scalar point) ID and component of
// each DOF (the EquationNamer maps an equation to its g-set DOF and point). An f set without DOFs (every DOF
// constrained) needs no factor: solve() then does nothing.
//
// With a structure cache, the factorisation takes the panels' storage a freed factor of the same shape left in the
// cache's slot, and the destructor leaves its own there for the next analysis: a design loop's points factorise in
// memory already mapped.
//
// Thread-safety: solve() is const and may run concurrently. Ownership: owns the selection, the symbolic and numeric
// factors (budgeted); borrows the DOF map, which must outlive it, and the kernels, which must outlive the construction
// and every solve() that does not name its own; shares the kept ordering of the cache. The budget outlives it. Not
// movable (the namer refers to it).
class FsetSolver
{
public:
    // Throws SingularityError, MemoryBudgetError (before the numeric work when the factor does not fit),
    // OperationCancelledError.
    FsetSolver(const linalg::BlockSparseMatrix& knn, const linalg::NodeGraph& graph, const assembly::DofMap& dofMap,
               const assembly::DofSets& sets, const linalg::DenseKernels& kernels, const FsetSolverOptions& options,
               base::MemoryBudget& budget, base::ThreadPool* pool = nullptr);
    FsetSolver(const FsetSolver&) = delete;
    FsetSolver& operator=(const FsetSolver&) = delete;
    FsetSolver(FsetSolver&&) = delete;
    FsetSolver& operator=(FsetSolver&&) = delete;
    ~FsetSolver();

    [[nodiscard]] std::int32_t equationCount() const noexcept { return m_ordering->equationCount(); }
    // The g-set DOF of an equation (the node-major DOFs of DofMap are its g-set numbering).
    [[nodiscard]] base::DofIndex dofOfEquation(std::int32_t equation) const noexcept
    {
        return static_cast<base::DofIndex>(m_ordering->selection().dofOfEquation(equation));
    }
    // The equation of a g-set DOF, or linalg::DofSelection::ELIMINATED.
    [[nodiscard]] std::int32_t equationOfDof(base::DofIndex dof) const noexcept
    {
        return m_ordering->selection().equationOfDof(dof);
    }

    // B ← Kff⁻¹·B, B of equationCount() rows in the equation order. Throws OperationCancelledError between blocks.
    void solve(linalg::DenseMatrixView rightHandSides, const base::CancellationToken* cancellation = nullptr) const;
    // The same with other kernels than those of the construction (the same bits for the same kernel kind, whatever
    // their pool): the solves made after the run that factorised, whose kernels and pool may be gone (FactorisedSet).
    void solve(linalg::DenseMatrixView rightHandSides, const linalg::DenseKernels& kernels,
               const base::CancellationToken* cancellation = nullptr) const;

    // The ratio test of the factorisation (its largest ratio, and the issues kept when BAILOUT is -1).
    [[nodiscard]] const linalg::PivotDiagnostics& pivotDiagnostics() const noexcept
    {
        return m_factor.pivotDiagnostics();
    }
    [[nodiscard]] const linalg::SymbolicFactor& symbolic() const noexcept { return m_ordering->symbolic(); }
    [[nodiscard]] const FsetOrdering& ordering() const noexcept { return *m_ordering; }
    // The time of its ordering, symbolic analysis, conversion and numeric factorisation (AnalysisPartTimings).
    [[nodiscard]] const AnalysisPartTimings& timings() const noexcept { return m_timings; }
    // What it took from the structure cache of its options (nothing without one).
    [[nodiscard]] const FsetReuse& reuse() const noexcept { return m_reuse; }
    // The name of an equation for the messages: the point ID and component.
    [[nodiscard]] base::GridComponent nameOf(std::int32_t equation) const;

private:
    const assembly::DofMap* m_dofMap;
    const linalg::DenseKernels* m_kernels;
    // Built for the solver in its budget, or lent by the structure cache with its bytes reserved there.
    std::shared_ptr<const FsetOrdering> m_ordering;
    // The ordering taken from the structure cache (its slot of a freed factor's panels), or null.
    std::shared_ptr<KeptOrdering> m_kept;
    linalg::SupernodalFactor m_factor;
    AnalysisPartTimings m_timings;
    FsetReuse m_reuse;
};

// The name of a g-set DOF for the messages: its point ID and component (0 for a scalar point).
[[nodiscard]] base::GridComponent equationName(const assembly::DofMap& dofMap, base::DofIndex dof);

} // namespace oos::solutions
