#pragma once

#include "oos/reduction/plan/InterfaceEstimate.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::reduction
{

// What one condensed unit brings to the interface phase of a constraint set.
struct InterfaceUnitSize
{
    std::int64_t boundaryDofCount = 0;
    std::int64_t patternCount = 0; // its load patterns (without the six inertial ones)
};

// The sizes the interface phase of a constraint set is estimated from, all known before any unit is condensed.
struct InterfaceFootprintInputs
{
    const InterfaceEstimate* estimate = nullptr; // the plan's (its graph with the K* cliques and its factor)
    std::int64_t pointCount = 0;                 // the interface points (every model point not interior to a unit)
    std::int64_t localDofCount = 0;              // their DOFs
    std::int64_t modelNodeCount = 0;             // the points of the model (the maps sized by the DOF map)
    std::int64_t modelDofCount = 0;              // the g set
    std::span<const InterfaceUnitSize> units;
    std::int64_t interfaceVectorCount = 0; // the interface's own load basis
    std::int64_t caseCount = 0;
    bool hasInertiaRelief = false;
    // The bytes a constraint set's relations take (measured on the MPC equations, glue and Gm a set makes,
    // made once before the plan for each distinct selection of the subcases, with the Gm of the interface and of the
    // outputs that are no larger than the set's), whether the model has an m set (the vector of the MPC forces), and
    // whether an output reads the element forces on the g set (the MPC forces, the grid point forces of the units not
    // recovered: their vector).
    std::uint64_t relationBytes = 0;
    bool hasMset = false;
    bool needsElementForces = false;
};

// The interface phase of a constraint set (the interface problem and the boundary basis, the interface factor
// estimated before it is made), in bytes beyond what the condensed units keep, as upper
// bounds of what its objects reserve:
// - modelBytes: what the set's constraints take on the g set (the full path's Gm and DOF sets of the subcases, the
//   relations eliminated at the interface, measured: InterfaceFootprintInputs::relationBytes; the held flags; the MPC
//   force vector with an m set, the element force vector when an output reads it: CondensedStatics::solveGroup); the
//   case vectors are a case's
//   (CondensedMemoryPlan::Inputs::caseBytes);
// - problemBytes: the interface problem (InterfaceProblem) at its peak: its numbering and rigid-body modes, then the
//   largest of its assembly (the stiffness and the singularity stiffness in blocks), its conversion (the blocks and the
//   SymmetricCsc) and its factorisation on one thread (InterfaceEstimate::factorisationBytes, with the margin of the
//   plan's estimate);
// - heldBytes: what the interface problem holds once factorised, through the boundary basis, the recoveries and the
//   cases: its numbering and graph, its SymmetricCsc, its factor (the plan's panels, their rows and solve
//   plan, the per-supernode arrays), within the same margins;
// - basisBytes: the boundary basis (BoundaryBasis: V, P_n, F, R on the min(Q, cases) columns it solves, β), the rows of
//   each unit's boundary on it, the coefficients of every unit and of the interface for the cases, the interface's
//   load vectors, and with inertia relief the support motions and relief loads.
// Q is bounded by the patterns of every unit and the interface's vectors (and the six relief loads).
//
// Without its factor (InterfaceInputs::keepsFactor false): problemWithoutFactorBytes the problem
// made without its factorisation (its assembly, then its conversion), heldWithoutFactorBytes what it holds (heldBytes
// but the factor, its rows and its solve plan), and the solve of the boundary basis's columns streamed
// (solveBytes: linalg::StreamedSolveEstimate on the plan's symbolic factor, within the factor's margin), whose regions
// keep at most a bound of bytes (factorBytes, its panels: every panel kept, nothing factorised again; leastRegionBytes,
// its largest panel: a region holds one); the solve is made before the recoveries, the results of the constraint sets
// before it kept beside it.
//
// Thread-safety: a value. Ownership: none.
struct InterfaceFootprint
{
    std::uint64_t modelBytes = 0;
    std::uint64_t problemBytes = 0;
    std::uint64_t heldBytes = 0;
    std::uint64_t basisBytes = 0;
    std::int64_t basisColumnCount = 0; // the columns the boundary basis may solve: min(Q, cases)
    std::uint64_t problemWithoutFactorBytes = 0;
    std::uint64_t heldWithoutFactorBytes = 0;
    std::uint64_t factorBytes = 0;
    std::uint64_t leastRegionBytes = 0;
    linalg::StreamedSolveEstimate streamedSolve;
    std::vector<linalg::RegionSweepCost> regionCosts; // the plan's (InterfaceEstimate::regionCosts)

    // The peak of the interface phase beside `after`, what the phase adds once the interface problem is factorised
    // (the recoveries, the cases, the results): its relations, then the larger of the problem being made and of the
    // factorised problem with its boundary basis and `after`. Without the factor (isWithoutFactor), the problem made,
    // then the problem held with its boundary basis and the larger of its solve by regions of regionBytes (beside the
    // results, resultBytes) and of `after`.
    [[nodiscard]] std::uint64_t peakBytes(std::uint64_t after, bool isWithoutFactor = false,
                                          std::uint64_t regionBytes = 0, std::uint64_t resultBytes = 0) const noexcept
    {
        if (!isWithoutFactor)
            return modelBytes + std::max(problemBytes, heldBytes + basisBytes + after);
        return modelBytes +
               std::max(problemWithoutFactorBytes,
                        heldWithoutFactorBytes + basisBytes + std::max(solveBytes(regionBytes) + resultBytes, after));
    }
    // The solve of the boundary basis's columns without the factor, by regions of at most regionBytes.
    [[nodiscard]] std::uint64_t solveBytes(std::uint64_t regionBytes) const noexcept;
    // The flops its regions factorise again with this bound (the plan's table; none at the factor or above).
    [[nodiscard]] double recomputedFlops(std::uint64_t regionBytes) const noexcept;

    // Throws std::invalid_argument without the plan's interface estimate.
    [[nodiscard]] static InterfaceFootprint estimate(const InterfaceFootprintInputs& inputs);
};

} // namespace oos::reduction
