#pragma once

#include "oos/elements/mixedHybrid/core/Matrix.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The factorisations of the small dense matrices of one element (Matrix): Cholesky, Cholesky after a symmetric
// diagonal scaling, LU with partial pivoting. The algorithms are those of G. H. Golub and C. F. Van Loan, Matrix
// Computations, 4th ed., Johns Hopkins University Press, 2013 (sections 3.1 and 3.4 for the triangular systems and
// the LU factorisation, section 4.2 for the Cholesky factorisation), written plainly: these matrices have a few
// hundred rows at most. A factorisation that does not exist (a matrix that is not positive definite, a zero pivot)
// is an empty optional, never an exception: the caller turns it into the refusal of its element.
//
// Symmetric eigenvalue problems and the minimum-norm inverse are in JacobiEigen.hpp.
//
// Thread-safety: the functions are pure; the factorisations are immutable once built.

// ---- Cholesky -------------------------------------------------------------------------------------------------

// The lower factor L of A = L Lᵀ, A symmetric positive definite, of which only the lower triangle (diagonal
// included) is read; the strict upper triangle of L is zero. Empty when a pivot is not positive (A is not positive
// definite to the working precision). Row by row:
//   l_ij = (a_ij − Σ_{k<j} l_ik l_jk) / l_jj  (j < i),   l_ii = (a_ii − Σ_{k<i} l_ik²)^½.
[[nodiscard]] std::optional<Matrix> choleskyLower(const Matrix& a);

// B ← L⁻¹ B in place, L lower triangular (forward substitution, row by row).
void solveLower(const Matrix& lower, Matrix& b);
void solveLower(const Matrix& lower, Vector& b);
// B ← L⁻ᵀ B in place (back substitution with the transpose of L, without forming it).
void solveLowerTransposed(const Matrix& lower, Matrix& b);
void solveLowerTransposed(const Matrix& lower, Vector& b);

// A = L Lᵀ and the solutions that use it.
class Cholesky
{
public:
    // Empty when A is not positive definite (see choleskyLower).
    [[nodiscard]] static std::optional<Cholesky> factor(const Matrix& a);

    [[nodiscard]] const Matrix& lower() const { return m_lower; }
    [[nodiscard]] std::size_t size() const { return m_lower.rows(); }

    // A⁻¹ b = L⁻ᵀ (L⁻¹ b).
    [[nodiscard]] Vector solve(Vector b) const;
    [[nodiscard]] Matrix solve(Matrix b) const;
    // A⁻¹ = XᵀX with X = L⁻¹: symmetric by construction.
    [[nodiscard]] Matrix inverse() const;

private:
    explicit Cholesky(Matrix lower) : m_lower(std::move(lower)) {}
    Matrix m_lower;
};

// The Cholesky factorisation of A after a symmetric diagonal scaling to a unit diagonal:
//   S A S = L Lᵀ,   S = diag(s),   s_i = 1 / √a_ii.
// Where the diagonal of A spans many decades (the compliance of the moments of an elongated shell element, the
// bubbles of a solid), the scaling keeps the digits that an unscaled factorisation or an explicit inverse would lose;
// S A S is the matrix of A in the basis scaled to unit energy, and for it the Cholesky factorisation is accurate to
// within the condition number of the scaled matrix (A. van der Sluis, Condition numbers and equilibration of
// matrices, Numerische Mathematik 14 (1969) 14-23: the diagonal scaling is nearly optimal for a positive definite
// matrix). Then
//   A⁻¹ = S (L Lᵀ)⁻¹ S,   and for a coupling B:  Bᵀ A⁻¹ B = Yᵀ Y with Y = L⁻¹ S B.
class EquilibratedCholesky
{
public:
    // Empty when a diagonal term is not positive or S A S is not positive definite.
    [[nodiscard]] static std::optional<EquilibratedCholesky> factor(const Matrix& a);

    // The scales s_i = 1 / √a_ii.
    [[nodiscard]] const Vector& scales() const { return m_scales; }
    // The lower factor of S A S.
    [[nodiscard]] const Matrix& lower() const { return m_lower; }
    [[nodiscard]] std::size_t size() const { return m_lower.rows(); }

    // Y = L⁻¹ S B: the half of the solution such that Bᵀ A⁻¹ B = Yᵀ Y (a stiffness written as a Gram matrix).
    [[nodiscard]] Matrix halfSolve(Matrix b) const;
    // A⁻¹ B = S L⁻ᵀ L⁻¹ S B.
    [[nodiscard]] Matrix solve(Matrix b) const;
    [[nodiscard]] Vector solve(Vector b) const;
    // A⁻¹ = S (L Lᵀ)⁻¹ S, symmetric by construction.
    [[nodiscard]] Matrix inverse() const;

private:
    EquilibratedCholesky(Vector scales, Matrix lower) : m_scales(std::move(scales)), m_lower(std::move(lower)) {}
    Vector m_scales;
    Matrix m_lower;
};

// ---- LU with partial pivoting ---------------------------------------------------------------------------------

// P A = L U, L unit lower triangular, U upper triangular, P the permutation of the rows that brings the largest
// remaining term of each column to the pivot. A column whose remaining terms are all zero is left as it is and marks
// the matrix singular (an exactly zero pivot): its determinant is then zero and it has no solution nor inverse.
class LuDecomposition
{
public:
    // A must be square.
    explicit LuDecomposition(const Matrix& a);

    [[nodiscard]] bool isSingular() const { return m_singular; }
    [[nodiscard]] std::size_t size() const { return m_factors.rows(); }
    // det A = ±Π u_kk, the sign that of the permutation.
    [[nodiscard]] double determinant() const;
    // A⁻¹ b, empty when A is singular.
    [[nodiscard]] std::optional<Vector> solve(const Vector& b) const;
    [[nodiscard]] std::optional<Matrix> solve(const Matrix& b) const;
    [[nodiscard]] std::optional<Matrix> inverse() const;

private:
    // Solves in place for one right-hand side already permuted.
    void substitute(std::span<double> x) const;

    Matrix m_factors;                 // L below the diagonal (its unit diagonal not stored), U on and above it
    std::vector<std::size_t> m_rowOf; // row k of P A is row m_rowOf[k] of A
    double m_permutationSign = 1.0;
    bool m_singular = false;
};

// A⁻¹ by LU with partial pivoting, empty when A is singular.
[[nodiscard]] std::optional<Matrix> inverse(const Matrix& a);
// det A by LU with partial pivoting.
[[nodiscard]] double determinant(const Matrix& a);

// The inverse of a square matrix after the equilibration of its rows and columns:
//   R = diag(1 / max_j |m_ij|),   C = diag(1 / max_i |(R m)_ij|),   M_s = R m C,   m⁻¹ = C M_s⁻¹ R,
// so that every row and every column of M_s has a largest term of 1 before its LU factorisation. Used where the rows
// of a matrix are functionals of very different scales (the Regge functionals of a shell element). Empty when M_s is
// singular.
[[nodiscard]] std::optional<Matrix> equilibratedInverse(const Matrix& m);

} // namespace oos::elements::mixedHybrid
