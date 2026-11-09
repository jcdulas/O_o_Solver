#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/supernodal/PartialFactor.hpp"
#include "oos/linalg/symbolic/SymbolicFactor.hpp"
#include "oos/reduction/plan/PlanGraph.hpp"

#include <cstdint>
#include <span>

namespace oos::reduction
{

// The rate the estimated times are computed with: the lower end of the one-thread rate of the supernodal
// factorisation as measured (7 to 25 GFLOP/s), so that an estimate is an
// upper bound of the time on one thread rather than a promise.
inline constexpr double ESTIMATE_FLOP_RATE = 7.0e9;

// The floating-point operations of the factorisation of one supernode panel of c columns over c + r rows: per column
// with m entries below its diagonal, m(m + 1) for the rank-one update and 2m for W = L·D and the divisions (the
// count of SymbolicFactor::flopCount()).
[[nodiscard]] double panelFlopCount(std::int64_t columns, std::int64_t rowsBelow) noexcept;

// The cost of condensing one unit (the plan reports, per unit, the boundary size before and after absorption, the
// size of K*, the estimated condensation time and memory), from the symbolic analysis of the unit
// with its boundary as trailing set, before any matrix exists:
// - the unit's points with free DOFs are ordered by linalg::ConstrainedOrdering (the rule of the full path: minimum
//   degree, nested dissection above its size thresholds) with the boundary points last, and analysed by
//   SymbolicFactor with the default amalgamation; PartialFactor factorises exactly this structure, so the sizes are
//   those it allocates (no pivot is delayed: the interior is positive definite);
// - interior factor = the panels of the interior supernodes, K* = the trailing block (boundary DOFs squared, stored
//   whole), flops = those of the interior panels (their updates make K*), memory peak = SymbolicFactor's estimate.
// keptFullIsRecommended: K* holds more values than the interior factor ("recommends keeping full a
// unit whose K* would cost more than its interior"): condensing it would make the interface problem larger, not
// smaller.
struct UnitCostEstimate
{
    std::int64_t nodeCount = 0; // its points with free DOFs
    std::int64_t interiorDofCount = 0;
    std::int64_t boundaryDofCount = 0;
    std::int64_t interiorFactorValueCount = 0;
    std::int64_t schurValueCount = 0; // K*: boundaryDofCount²
    double flopCount = 0.0;
    std::uint64_t memoryPeakBytes = 0;
    double estimatedSeconds = 0.0; // on one thread, at ESTIMATE_FLOP_RATE
    bool isKeptFullRecommended = false;
    // The sizes the memory estimates of the condensed path start from (reduction/scheduling/UnitFootprint):
    // the node blocks of its points with free DOFs (the lower triangle with the diagonal: the pattern of its matrix),
    // the entries of the lower triangle of its equations (its SymmetricCsc: Σ_i w_i(w_i + 1)/2 + Σ_(i,j) w_i·w_j over
    // the points and the couplings, w the free DOFs of a point), the rows of its interior panels (their index arrays),
    // its interior supernodes, and the bytes its partial factorisation reserves on one thread
    // (SupernodalFactor::estimateMemoryBytes, an upper bound of PartialFactor's: it counts the trailing supernode as a
    // panel of K*'s size and its front in the working storage, PartialFactor allocates K* once and no such panel).
    std::int64_t blockCount = 0;
    std::int64_t matrixValueCount = 0;
    std::int64_t factorRowCount = 0;
    std::int64_t supernodeCount = 0;
    std::uint64_t factorisationBytes = 0;
    // Its factorisation without its factor on one thread (PartialFactor::estimateWithoutFactor): the
    // condensation that keeps no panel once used, and the recovery by regions of its elimination tree.
    linalg::WithoutFactorEstimate withoutFactor;

    // nodes: the unit's points (sorted); boundary: its boundary points (sorted, a subset of nodes);
    // interfaceRelations: the relations of the connection system (sorted), whose dependent DOFs stay in the unit as
    // boundary DOFs (PlanGraph::unitDofCount). Points with no DOF for the unit are left out. Throws
    // std::invalid_argument for a point the graph does not hold, MemoryBudgetError when the analysis does not fit.
    [[nodiscard]] static UnitCostEstimate estimate(const PlanGraph& graph, std::span<const base::EntityId> nodes,
                                                   std::span<const base::EntityId> boundary,
                                                   std::span<const RelationKey> interfaceRelations,
                                                   base::MemoryBudget& budget, base::ThreadPool* pool = nullptr);

    // The same figures from a symbolic factor whose trailing set is the boundary (for the tests and the interface);
    // blockCount and matrixValueCount, which the symbolic factor does not know, stay 0 (fromGraph gives them).
    [[nodiscard]] static UnitCostEstimate fromSymbolic(const linalg::SymbolicFactor& symbolic);
    // fromSymbolic with the pattern sizes of `graph` (the graph the symbolic factor was analysed on) and its weights.
    [[nodiscard]] static UnitCostEstimate fromGraph(const linalg::SymbolicFactor& symbolic,
                                                    const linalg::NodeGraph& graph,
                                                    std::span<const std::int32_t> weights);
};

} // namespace oos::reduction
