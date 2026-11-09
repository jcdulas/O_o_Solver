#pragma once

#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/dense/Inertia.hpp"
#include "oos/linalg/kernels/DenseKernelEntryPoints.hpp"

#include <cstddef>
#include <span>

namespace oos::linalg
{

// The pieces of LDLᵀ arithmetic the baseline code of linalg/dense shares with the kernel families: they use the same
// expressions as kernels/reference and kernels/SimdDenseKernels.ipp (each product rounded; linalg is compiled with
// -ffp-contract=off), so that W = L·D, the 2 × 2 pivots and the inertia are the same bits in every family.

// True when D has a 2 × 2 block on rows p and p+1.
[[nodiscard]] bool startsTwoByTwoBlock(const BlockDiagonalView& d, std::size_t p) noexcept;

// W(row, p) = (L·D)(row, p) for p in [firstColumn, firstColumn + w.size()), L read from the factor `l` (row `row`):
// L(row, p)·d_p, or for a 2 × 2 block on (p, p+1): L(row, p)·d_p + L(row, p+1)·e_p and L(row, p)·e_p +
// L(row, p+1)·d_{p+1} (the formula of ProductTileOperands). d is indexed from firstColumn.
void computeProductRow(ConstDenseMatrixView l, std::size_t row, std::size_t firstColumn, const BlockDiagonalView& d,
                       std::span<double> w) noexcept;

// The coefficients of the inverse of a 2 × 2 pivot [[a, b], [b, c]] (LAPACK dsytf2): for a right-hand side (r0, r1),
// x0 = s·(d11·r0 − r1) and x1 = s·(d22·r1 − r0).
struct TwoByTwoInverse
{
    double d11 = 0.0;
    double d22 = 0.0;
    double s = 0.0;
};
[[nodiscard]] TwoByTwoInverse invertTwoByTwo(double a, double b, double c) noexcept;

// The symmetric interchange of rows and columns s < t of the lower-stored square `a` (the same as kernels/reference).
void swapSymmetric(DenseMatrixView a, std::size_t s, std::size_t t) noexcept;

// The inertia of D: a 1 × 1 block by its sign; a 2 × 2 block by its determinant a·c − b² (negative: one eigenvalue of
// each sign; positive: two of the sign of a; zero: one zero and one of the sign of the trace).
[[nodiscard]] Inertia inertiaOf(const BlockDiagonalView& d) noexcept;

} // namespace oos::linalg
