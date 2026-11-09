#pragma once

#include <array>
#include <span>

namespace oos::assembly
{

// The ratio given to a component that is not singular.
inline constexpr double NOT_SINGULAR_RATIO = -1.0;

// The singular components of one 3 × 3 diagonal block of a grid point (its translations or its rotations): steps 1 to
// 4 of the grid point singularity processor (GridPointSingularities, which documents the rules), on one block. The
// processor walks every grid point of a DOF map; the condensed path walks its own points with it (each
// unit its interior points, the interface problem its points, on the blocks the full path would see), so that both
// paths constrain the same components.
//
// Thread-safety: a value. Ownership: none.
struct SingularBlock
{
    // Per component of the block: the ratio λ/λmax of the singular direction it is constrained for, or
    // NOT_SINGULAR_RATIO.
    std::array<double, 3> ratios{NOT_SINGULAR_RATIO, NOT_SINGULAR_RATIO, NOT_SINGULAR_RATIO};
    // The singular directions lie along the components chosen (|det V_C| ≥ GridPointSingularities::ALIGNMENT_COSINE).
    bool isAligned = true;

    // block: the 9 values of the block, row-major (its two triangles averaged); isFree: the components in neither the m
    // nor the s set (only those are analysed); pivotZeroThreshold: PARAM EPZERO.
    [[nodiscard]] static SingularBlock analyse(std::span<const double, 9> block, const std::array<bool, 3>& isFree,
                                               double pivotZeroThreshold);
};

} // namespace oos::assembly
