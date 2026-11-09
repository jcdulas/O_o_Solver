#pragma once

#include "oos/design/responses/ResponseValues.hpp"

#include <cstddef>

namespace oos::design
{

// How far an item recomputed for its derivatives may differ from the analysis' value: the same affine function
// of the displacements evaluated in another order of sums, so its rounding, and anything beyond is another quantity (a
// STRAIN request with FIBER, a Bar2 under BeamDistributedLoad read in its station layout) whose derivative must not
// be returned.
//
// Two bounds, either one enough:
//   - relative to the item: VALUE_CHECK_TOLERANCE of the larger of its value and VALUE_CHECK_FLOOR of the response's
//     largest value (an item near zero among large ones);
//   - the rounding of the recovery: ROUNDING_CHECK_TOLERANCE of the magnitude of the products it sums before they
//     cancel. An element load is recovered from K_e·u_e (in the element's frame, its pin flags condensed, its offsets
//     moved), so it rounds at n·ε of the largest Σ_c |K_rc·u_c| of the element (ε = 2.2e-16, n the length of the chain
//     of operations, some tens to a few hundred with the transformations and the sum of the elements of a section),
//     whatever its own derivative: the shear of a pinned strut has no stiffness of its own (its derivative terms are
//     1e-21), yet it is the rounding of the axial products, 1e-10 beside an axial force of 7.6e3 when the strut moves
//     far as a rigid body. 1e-12 (some 4 500 ε) bounds the chains here with a margin. The callers give the products
//     they have: the element's K_e·u_e for the element forces, the largest Σ_d |w_d·u_d| of the rows of a section or
//     cut (its exact rows, the axial row of a strut among them), the item's own terms otherwise. An item below this
//     bound is below its own rounding, and no derivative tells it from another quantity; a real mismatch differs by a
//     fraction of an item above it.
inline constexpr double VALUE_CHECK_TOLERANCE = 1.0e-8;
inline constexpr double VALUE_CHECK_FLOOR = 1.0e-6;
inline constexpr double ROUNDING_CHECK_TOLERANCE = 1.0e-12;

// Thread-safety: stateless.
class ValueCheck
{
public:
    // Whether `recomputed` agrees with item `item` of `values`; `terms`: the magnitude of the products the recovery
    // sums (0 when not known: the bound relative to the item alone).
    [[nodiscard]] static bool agrees(const ResponseValues& values, std::size_t item, double recomputed,
                                     double terms = 0.0) noexcept;
    // OptimisationError naming the response, the item and both values when they do not agree.
    static void require(const ResponseValues& values, std::size_t item, double recomputed, double terms = 0.0);
};

} // namespace oos::design
