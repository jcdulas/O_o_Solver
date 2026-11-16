#pragma once

#include "oos/elements/mixedHybrid/core/Matrix.hpp"

namespace oos::elements::mixedHybrid
{

// Symmetric eigenvalue problems and least-norm inverses of the small dense matrices of one element, by Jacobi
// rotations. Every symmetric eigenvalue problem of the mixed hybrid formulation goes through jacobiEigen: the
// reference implementation found general-purpose routines inexact on the matrices of the solid elements
// (reconstruction errors of 2.3e-5 for the singular values of a trace matrix, eigenpairs 4e-2 off on a nearly
// diagonal matrix with a double eigenvalue), while the Jacobi method is accurate to the round-off of the matrix in
// every eigenpair (J. Demmel, K. Veselić, Jacobi's method is more accurate than QR, SIAM Journal on Matrix Analysis
// and Applications 13 (1992) 1204-1245). These matrices have a few hundred rows at most, where its cost does not
// matter.
//
// Thread-safety: pure functions.

// A = V diag(values) Vᵀ: the eigenvalues in the order of the diagonal they converge on (not sorted), the
// eigenvectors as the columns of V (orthonormal).
struct SymmetricEigen
{
    Vector values;
    Matrix vectors;
};

// The eigenvalues and eigenvectors of a symmetric matrix by the cyclic Jacobi method (G. H. Golub, C. F. Van Loan,
// Matrix Computations, 4th ed., 2013, section 8.5): sweeps of plane rotations over the pairs (p, q) taken row by row,
// each rotation chosen to zero the term a_pq,
//   θ = (a_qq − a_pp) / (2 a_pq),  t = sign(θ) / (|θ| + √(θ² + 1)),  c = 1 / √(t² + 1),  s = t c,
// until Σ_{i<j} a_ij² ≤ 1e-32 Σ_i a_ii² (at most 100 sweeps). The whole matrix is read: the caller symmetrises it
// first when it holds round-off asymmetry.
[[nodiscard]] SymmetricEigen jacobiEigen(const Matrix& a);

// The least-norm least-squares inverse A⁺ = (AᵀA)⁺ Aᵀ of a matrix whose columns are well scaled and whose rank
// deficiency is exact (the interior functions of a space, which have no trace): from the eigenpairs (λ_k, v_k) of
// AᵀA by jacobiEigen,
//   (AᵀA)⁺ = Σ_{λ_k > 1e-12 λ_max} v_k v_kᵀ / λ_k,
// the eigenvalues below 1e-12 of the largest taken as zero. x = A⁺ b is the solution of least norm of the least
// squares problem min |A x − b|.
[[nodiscard]] Matrix minimumNormInverse(const Matrix& a);

// A = U diag(values) Vᵀ: for an m × n matrix with m ≥ n, U is m × n with orthonormal columns where the singular value
// is not zero (a zero column otherwise), V is n × n orthogonal; for m < n, the decomposition is that of the transpose
// read backwards (U is m × m, V is n × m). Singular values in the order of the columns they converge on, not sorted.
struct SingularValueDecomposition
{
    Matrix left;
    Vector values;
    Matrix right;
};

// The singular value decomposition by one-sided Jacobi rotations (M. R. Hestenes, Inversion of matrices by
// biorthogonalization and related results, Journal of the Society for Industrial and Applied Mathematics 6 (1958)
// 51-90): the columns of A are made orthogonal by the plane rotations that jacobiEigen would apply to AᵀA, without
// forming AᵀA, so that the small singular values keep their relative accuracy (Demmel and Veselić, 1992). For the
// columns p, q, with α = |a_p|², β = |a_q|², γ = a_p · a_q:
//   ζ = (β − α) / (2γ),  t = sign(ζ) / (|ζ| + √(ζ² + 1)),  c = 1 / √(t² + 1),  s = t c,
// until |γ| ≤ 1e-15 √(αβ) for every pair (at most 100 sweeps). Then σ_j = |a_j| and u_j = a_j / σ_j.
[[nodiscard]] SingularValueDecomposition jacobiSingularValueDecomposition(const Matrix& a);

// The pseudo-inverse A⁺ = V Σ⁺ Uᵀ, the singular values not above `singularValueTolerance` (absolute) taken as zero.
[[nodiscard]] Matrix pseudoInverse(const Matrix& a, double singularValueTolerance);

} // namespace oos::elements::mixedHybrid
