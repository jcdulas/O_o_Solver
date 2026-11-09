#pragma once

#include "oos/base/ids/EntityId.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/linalg/ordering/NodeGraph.hpp"
#include "oos/linalg/ordering/NodePermutation.hpp"
#include "oos/linalg/supernodal/StreamedSolve.hpp"
#include "oos/reduction/plan/PlanGraph.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace oos::reduction
{

// One unit as the interface problem sees it: a condensed unit brings its boundary points (its K* couples all of them:
// a clique), a unit kept full brings all its points and their couplings.
struct InterfacePart
{
    bool isCondensed = true;
    std::span<const base::EntityId> nodes;    // every point of the unit, sorted
    std::span<const base::EntityId> boundary; // its boundary points, sorted
};

// The size of the interface problem, estimated before any condensation: its factor size and memory peak are estimated
// from the symbolic analysis of the quotient graph, before any condensation, and shown in the plan report, so that a
// too large interface factor is seen before the work starts.
//
// The interface problem's points: the boundary points of the condensed units, the points of the units kept full and
// the connection-only points of the connection system, those with free DOFs (every relation eliminated). Its graph:
// the couplings of the model between these points (connections, the elements of the full units, the dependent DOFs
// eliminated at the interface) and the structural pattern of each K* (a clique on the boundary points each connected
// part of the unit's interior touches). Its ordering (orderOnUnits): a
// fill-reducing order of the quotient graph in which the boundary of each condensed unit is one vertex (each K* is a
// clique and counts as one vertex: never a minimum degree on the cliques themselves), expanded unit by unit;
// ConstrainedOrdering chooses the method on the quotient graph by
// its usual rule (minimum degree below its size thresholds, nested dissection above). The symbolic analysis of that
// order gives the factor, the flops and the memory peak.
//
// Thread-safety: a value. Ownership: none.
struct InterfaceEstimate
{
    std::int64_t nodeCount = 0;
    std::int64_t dofCount = 0;
    std::int64_t condensedUnitCount = 0;
    std::int64_t fullUnitCount = 0;
    std::int64_t connectionNodeCount = 0; // the connection-only points with free DOFs
    std::int64_t factorValueCount = 0;    // the panels of the factor
    double flopCount = 0.0;
    std::uint64_t memoryPeakBytes = 0;
    double estimatedSeconds = 0.0; // on one thread, at ESTIMATE_FLOP_RATE
    // The sizes the memory estimates of the condensed path start from (reduction/scheduling): as
    // UnitCostEstimate's, on the interface graph (the K* cliques included) and its factor: the node blocks, the
    // entries of the lower triangle of its equations, the rows of the panels, the supernodes, and the bytes its
    // factorisation reserves on one thread (SupernodalFactor::estimateMemoryBytes).
    std::int64_t blockCount = 0;
    std::int64_t matrixValueCount = 0;
    std::int64_t factorRowCount = 0;
    std::int64_t supernodeCount = 0;
    std::uint64_t factorisationBytes = 0;
    // With `estimatesStreamedSolve`: the solve without its factor (linalg::StreamedSolve on
    // this symbolic factor, one thread: its fixed bytes, a run's values, its accumulator rows, its largest panel, every
    // panel, its flops) and the regions of its backward sweep for the bounds a memory plan takes: the least (its
    // largest panel), the powers of two of bytes between it and its factor, and the factor (every panel kept: none
    // factorised again), in increasing bounds of bytes.
    linalg::StreamedSolveEstimate streamedSolve;
    std::vector<linalg::RegionSweepCost> regionCosts; // bounded by the powers of two of bytes up to the factor (64)

    // The regions of the bound of regionBytes: the tabulated bound at most regionBytes that is the largest (the least
    // region below it; the factor's, nothing factorised again, at or above the factor). Empty without the table.
    [[nodiscard]] linalg::RegionSweepCost regionCostOf(std::uint64_t regionBytes) const noexcept;

    // connectionNodes: the connection-only points of the connection system (sorted); interfaceRelations: its relations
    // (sorted), eliminated at the interface: a boundary DOF they make dependent is replaced by the points it depends
    // on. Throws std::invalid_argument for a point the graph does not hold, MemoryBudgetError when the analysis does
    // not fit.
    // estimatesStreamedSolve: the figures of the solve without the factor too (the condensed plan's interface; not the
    // full path's estimate).
    [[nodiscard]] static InterfaceEstimate estimate(const PlanGraph& graph, std::span<const InterfacePart> parts,
                                                    std::span<const base::EntityId> connectionNodes,
                                                    std::span<const RelationKey> interfaceRelations,
                                                    base::MemoryBudget& budget, base::ThreadPool* pool = nullptr,
                                                    bool estimatesStreamedSolve = false);

    // The order of the interface problem: the quotient graph of the partition unitOfNode (unit u in [0, unitCount);
    // a condensed unit's boundary is one unit, every other point a unit of its own) ordered by ConstrainedOrdering,
    // expanded unit by unit (each unit's points by increasing index). nodeWeights: the DOFs of each node;
    // coordinates: the positions (empty when unknown).
    [[nodiscard]] static linalg::NodePermutation orderOnUnits(
        const linalg::NodeGraph& graph, std::span<const std::int32_t> unitOfNode, std::int32_t unitCount,
        std::span<const std::int32_t> nodeWeights, std::span<const base::Vec3> coordinates, base::MemoryBudget& budget,
        base::ThreadPool* pool = nullptr);
};

} // namespace oos::reduction
