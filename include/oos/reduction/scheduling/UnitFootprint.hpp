#pragma once

#include "oos/reduction/plan/UnitCostEstimate.hpp"

#include <algorithm>
#include <cstdint>

namespace oos::reduction
{

// The sizes a unit's footprint is estimated from, all known before its equations are built: the plan's cost estimate
// (the pattern and the factor of its free DOFs, UnitCostEstimate) and what the analysis knows of the unit (its points,
// its load patterns, the cases).
struct UnitFootprintInputs
{
    const UnitCostEstimate* cost = nullptr; // required
    std::int64_t pointCount = 0;            // its points, the generated fastener points included
    std::int64_t localDofCount = 0;         // the DOFs of those points (its local numbering)
    std::int64_t patternCount = 0;          // its load patterns, the six inertial ones of inertia relief included
    bool hasEnforcedPatterns = false;       // some pattern enforces a displacement (a second matrix of patterns)
    // The absorbed spiders (UnitDescription::spiders): the node blocks their change of basis couples (every leg point
    // with every other and with the reference, Σ (legs + 1)²) and the values of their changes of basis (Σ n·m).
    std::int64_t spiderBlockCount = 0;
    std::int64_t spiderValueCount = 0;
    bool isRecovered = false;
    // Craig-Bampton: the unit keeps its mass on its equations and its reduced boundary mass, and its
    // condensation forms its constraint modes by blocks (CraigBampton::projectOnBoundary). Its modal part (the Lanczos
    // solve of its interior, its modes) is reserved when it is made, not in this estimate.
    bool keepsMass = false;
    std::int64_t caseCount = 0;
    // The columns of the boundary basis its recovery may solve: Q, or the cases when they are fewer (an upper bound:
    // the patterns of every unit and the interface's vectors).
    std::int64_t basisColumnCount = 0;
    // The load basis handed over: the basis vectors one block may read of it at once (the largest
    // constraint set's basis), 0 when no block reads it. Its recovery is then that of a block (isRecovered is implied):
    // recoveryBytes its interior factor, kept once made when every recovery is kept, passBytes what a block's recovery
    // of its interior holds beside it (CondensedBasisColumns: the solutions of the columns on its equations, or its
    // boundary motions H when fewer, its boundary values, its interiors on its local DOFs kept for the block, its
    // inertial loads), regionFixedBytes the streamed forward sweep of those columns when by regions.
    std::int64_t readColumnCount = 0;
};

// What one condensed unit takes from the memory budget (every large allocation is accounted before it is made, from
// the symbolic analysis), estimated before its equations
// exist, as upper bounds of what its objects reserve (each term below names the object it bounds):
// - keptBytes, held from its condensation to the end of the analysis when its equations are kept: its equations
//   (UnitSystem: the local numbering, the roles, the node graph, the SymmetricCsc of its matrixValueCount entries and
//   of its spiders' couplings, the symbolic factor, the rigid-body modes and inertial patterns, the raw boundary
//   blocks), its patterns on the local DOFs, its reduced unit (K* of b² values, f* and the inertial patterns on its b
//   boundary DOFs) and the work of its patterns (k × k);
// - lightBytes, held instead when its equations are released after its condensation: the system
//   without equations made from its boundary data (the numbering, the roles, the relations and spiders of the unit,
//   the singularity rows, the raw boundary blocks, the held DOFs, the boundary labels), with the patterns, the reduced
//   unit and the work as above;
// - buildBytes, the transient of the construction of its equations beyond keptBytes (released before its
//   factorisation): the block pattern, the stiffness and the mass in 6 × 6 blocks, the element groups, the subgraph
//   and the ordering;
// - factorisationBytes, the transient of its condensation on one thread beyond keptBytes: the partial factorisation
//   (UnitCostEstimate::factorisationBytes: the interior panels, the fronts, K*), the loads of its patterns on the
//   equations and their forward sweep, the sweeps of the pattern work;
// - factorBytes, what its interior factor holds once made (the interior panels and their row indices, the per-supernode
//   arrays, the solve plan of its panels, the factorisation's own K*): kept for its recovery or released
//   (FactorRetention);
// - recoveryBytes, what its recovery holds for one constraint set when every unit is recovered at once (UnitRecovery:
//   the basis solutions on its equations and their coefficients, or, solving the cases one by one, the loads of its
//   patterns and its interior factor through the cases), zero when it is not recovered;
// - passBytes, what its pass holds beside its interior factor when the units are recovered in sequence: its recovery
//   data (the basis solutions and coefficients, or the loads of its patterns) and its vectors of
//   one case (the solution, loads and K·u on its equations, its enforced values, displacements, loads and reactions on
//   its local DOFs), zero when it is not recovered;
// - withoutFactorBytes, the transient of its condensation without its factor beyond keptBytes:
//   linalg's streamed factorisation (its fronts, the panels of the units being swept, the skeleton factor, the pivots,
//   the forward sweep's accumulators: UnitCostEstimate::withoutFactor), the loads of its patterns and inertial patterns
//   on its equations, swept in place once, and the pattern work;
// - largestPanelBytes and regionFixedBytes, its recovery without its factor (by regions of its elimination tree): the
//   least region (its largest panel) and what that recovery holds beside the panels a region keeps (the streamed
//   factorisation of its forward sweep, its swept columns): with a region bound its pass holds regionFixedBytes, the
//   bound and its recovery data instead of its factor.
// The estimates are bounds, not measures: the margins of UnitFootprint.cpp cover what the plan's graph does not see
// exactly (the points held in every subcase, which the unit's numbering keeps), measured against the budget's own
// accounting on assemblies of 3 000 plates; whole analyses were run at the estimate.
//
// Thread-safety: a value. Ownership: none.
struct UnitFootprint
{
    std::int64_t unitId = 0;
    double flopCount = 0.0; // its interior factorisation: the cost order of the scheduler
    std::uint64_t keptBytes = 0;
    std::uint64_t lightBytes = 0;
    std::uint64_t buildBytes = 0;
    std::uint64_t factorisationBytes = 0;
    std::uint64_t factorBytes = 0;
    std::uint64_t recoveryBytes = 0;
    std::uint64_t passBytes = 0;
    std::uint64_t withoutFactorBytes = 0;
    std::uint64_t largestPanelBytes = 0;
    std::uint64_t regionFixedBytes = 0;
    bool isRecovered = false;

    // The peak beyond keptBytes while it is condensed: its build, then its factorisation, with its factor or without
    // it (the build's transient is released before the factorisation starts).
    [[nodiscard]] std::uint64_t condensationBytes(bool isWithoutFactor = false) const noexcept
    {
        return std::max(buildBytes, isWithoutFactor ? withoutFactorBytes : factorisationBytes);
    }
    // Whether, with a region bound (0: none), its recovery is made by regions: its factor does not fit the bound.
    [[nodiscard]] bool isRecoveredByRegions(std::uint64_t regionBytes) const noexcept
    {
        return regionBytes > 0 && factorBytes > regionBytes;
    }
    // What it keeps to the end of the analysis, its equations released (true) or kept.
    [[nodiscard]] std::uint64_t keptBytesOf(bool isReleased) const noexcept
    {
        return isReleased ? std::min(lightBytes, keptBytes) : keptBytes;
    }
    // The peak of its pass beyond what it keeps when the units are recovered in sequence (zero when it is not
    // recovered): its equations built again when released (beside their build), its interior factorised again, then
    // its factor, its recovery data and its vectors of one case.
    // With a region bound that its factor exceeds: its recovery by regions instead of its factor.
    [[nodiscard]] std::uint64_t passPeakBytes(bool isReleased, std::uint64_t regionBytes = 0) const noexcept
    {
        if (!isRecovered)
            return 0;
        const std::uint64_t equations = isReleased ? keptBytes - keptBytesOf(true) : 0;
        if (isRecoveredByRegions(regionBytes))
            return equations + std::max(isReleased ? buildBytes : 0,
                                        regionFixedBytes + std::max(regionBytes, largestPanelBytes) + passBytes);
        return equations + std::max({isReleased ? buildBytes : 0, factorisationBytes, factorBytes + passBytes});
    }

    // Throws std::invalid_argument without a cost estimate.
    [[nodiscard]] static UnitFootprint estimate(std::int64_t unitId, const UnitFootprintInputs& inputs);
};

} // namespace oos::reduction
