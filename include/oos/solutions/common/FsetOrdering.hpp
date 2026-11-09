#pragma once

#include "oos/assembly/dofs/DofMap.hpp"
#include "oos/assembly/dofs/DofSets.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/CancellationToken.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/ordering/ConstrainedOrdering.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/sparse/DofSelection.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"
#include "oos/solutions/common/AnalysisTimings.hpp"

#include <cstdint>
#include <span>

namespace oos::solutions
{

// The equations of one DOF set of the full model and their symbolic factor (the full path): the nodes with a DOF of the
// set are ordered (ConstrainedOrdering, the node weights their DOF counts in the set, the grid point positions for the
// dissection), analysed (SymbolicFactor), and their DOFs numbered in the factor's order (DofSelection with the
// elimination flags of DofSets). Every matrix assembled on the model's pattern (stiffness, mass, differential
// stiffness) converts to the same SymmetricCsc structure with this selection, so the one symbolic factor serves the
// static factorisation (FsetSolver) and every shift of the eigenvalue solvers (SparsePencil, BucklingPencil).
//
// A set without DOFs (every DOF constrained) has no symbolic factor: equationCount() is 0.
//
// Thread-safety: immutable once built. Ownership: owns the selection and the symbolic factor (budgeted); borrows
// nothing after construction.
class FsetOrdering
{
public:
    // keptSet: the DOFs numbered (the l set of a static solve, the a set of an eigenvalue solve: the f set without the
    // r set of FreeBodySupport, which is the f set without FreeBodySupport). Throws MemoryBudgetError,
    // OperationCancelledError. lastNodes: nodes of the model ordered last, in that order (the FreeBodySupport points of
    // the full path streamed, whose displacements its downward pass needs first); those without a kept DOF are skipped.
    // cuts, componentOfNode: the dissection's cuts of the regions of several components, the component of each node of
    // the model (linalg::DissectionOptions; empty: free cuts).
    FsetOrdering(const linalg::NodeGraph& graph, const assembly::DofMap& dofMap, const assembly::DofSets& sets,
                 assembly::DofSet keptSet, linalg::OrderingMethod ordering, base::MemoryBudget& budget,
                 base::ThreadPool* pool = nullptr, const base::CancellationToken* cancellation = nullptr,
                 std::span<const std::int32_t> lastNodes = {}, const linalg::AmalgamationThresholds& thresholds = {},
                 linalg::ComponentCuts cuts = linalg::ComponentCuts::Free,
                 std::span<const std::int32_t> componentOfNode = {});

    [[nodiscard]] std::int32_t equationCount() const noexcept { return m_selection.keptDofCount(); }
    [[nodiscard]] const linalg::DofSelection& selection() const noexcept { return m_selection; }
    [[nodiscard]] const linalg::SymbolicFactor& symbolic() const noexcept { return m_symbolic; }
    // The time of the ordering and of the symbolic analysis (AnalysisPartTimings::ordering, symbolic).
    [[nodiscard]] const AnalysisPartTimings& timings() const noexcept { return m_timings; }

private:
    linalg::DofSelection m_selection;
    linalg::SymbolicFactor m_symbolic;
    AnalysisPartTimings m_timings;
};

} // namespace oos::solutions
