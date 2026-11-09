#pragma once

#include "oos/base/memory/BudgetedBuffer.hpp"
#include "oos/base/memory/MemoryBudget.hpp"
#include "oos/linalg/dense/DenseMatrixView.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace oos::reduction
{

// What a DOF of a spider measures: a translation (a length) or a rotation (an angle), which the change of basis scales
// by the spider's characteristic length so that every DOF is a length.
enum class SpiderDofKind : std::uint8_t
{
    Translation,
    Rotation
};

// The weighting matrix of a spider: the reference DOFs as a combination of the leg
// DOFs, u_ref = W·u_legs. For an RigidInterpolation, W is the weighted least-squares rigid fit of the legs (the rows of
// Gm of its REFC DOFs, assembly::RigidElementEquations); for a pierced side of a Fastener or Weld, the
// interpolation of the piercing point from the grid points of the pierced element (elements::interpolatePiercing).
// Row-major, referenceDofCount rows of legDofCount values; the DOFs in the displacement system of their point. Sized by
// one spider, as an element matrix is by one element: std::vector.
struct SpiderWeightMatrix
{
    std::size_t referenceDofCount = 0;
    std::size_t legDofCount = 0;
    std::vector<double> values;
    std::vector<SpiderDofKind> referenceKinds; // referenceDofCount entries
    std::vector<SpiderDofKind> legKinds;       // legDofCount entries
    // The length the rotations are scaled by: the mean distance of the leg points to the reference point (the Lc of
    // the RigidInterpolation), 1 when every leg is on the reference point.
    double characteristicLength = 1.0;

    [[nodiscard]] double at(std::size_t row, std::size_t column) const noexcept
    {
        return values[row * legDofCount + column];
    }
};

// The change of basis that absorbs one spider into its unit (the only way a spider is absorbed):
//     u_legs = W⁺·u_ref + N·q,
// W the weighting matrix (m reference DOFs, n leg DOFs, n ≥ m), W⁺ a right inverse of W (W·W⁺ = I) and N a basis of
// its null space (W·N = 0): T = [W⁺ N] is square (n × n) and invertible, so that Tᵀ·K·T stays symmetric positive
// definite and the unit is condensed by an ordinary partial Cholesky-type factorisation, the q DOFs interior and the
// reference DOFs boundary. Exact: the relation u_ref = W·u_legs holds for every (u_ref, q), and every leg motion is
// one (u_ref, q). The flexibility form K* = (W·K⁻¹·Wᵀ)⁻¹ and Lagrange multipliers are not used: the first needs the
// inverse of a possibly singular K, the second makes the system indefinite.
//
// Construction (Golub & Van Loan, "Matrix Computations", 4th ed., §5.2 Householder QR, §6.1 the null space and the
// minimum-norm solution from a QR factorisation):
// - scaling: S_c and S_r multiply the rotation DOFs of the legs and of the reference by the characteristic length L
//   (1 for translations), W̃ = S_r·W·S_c⁻¹, so that W̃ is dimensionless and its singular values compare;
// - Householder QR of the tall W̃ᵀ = Q·[R; 0] (n × m, m reflectors, R upper triangular m × m), Q = [Q1 Q2] orthogonal:
//   W̃ = Rᵀ·Q1ᵀ, so W̃⁺ = Q1·R⁻ᵀ (the Moore-Penrose pseudo-inverse of W̃) and Ñ = Q2 (orthonormal);
// - in physical units: T = S_c⁻¹·Q·blockdiag(R⁻ᵀ·S_r, I), i.e. W⁺ = S_c⁻¹·W̃⁺·S_r (a weighted pseudo-inverse of W:
//   W·W⁺ = I) and N = S_c⁻¹·Ñ.
// T is never formed: it is applied through the m reflectors, R and the scalings, O(m·n) per vector.
//
// Conditioning (the numerical safeguard, AbsorptionSafeguard): the singular values of W̃ are those of
// R, the square roots of the eigenvalues of Rᵀ·R = W̃·W̃ᵀ (m × m); conditionNumber() = σ_max / σ_min. It is the
// condition number of the scaled change of basis once the reference coordinates are normalised so that σ_max = 1 (a
// scaling of u_ref, which changes no result): the columns of Ñ are orthonormal and orthogonal to those of W̃⁺ (both in
// the orthogonal Q), so the singular values of [W̃⁺ Ñ] are the 1/σ_i of W̃⁺ and the 1 of Ñ. It does not grow with the
// number of legs of a uniform spider: it is 1 for a symmetric one (four legs at the corners of a square around the
// reference) and grows as the legs leave a reference motion badly determined (collinear legs for a rotation they
// should fix). W̃ is rank-deficient when σ_min ≤ RANK_TOLERANCE·σ_max, or when
// n < m: T then does not exist and the spider cannot be absorbed.
//
// The DOF order after the change: the m reference DOFs, then the n − m deformation DOFs q.
//
// Thread-safety: immutable once built; concurrent calls are safe. Ownership: owns the reflectors (budgeted: n·m values,
// the size of W) and R.
class SpiderChangeOfBasis
{
public:
    // Below this ratio σ_min/σ_max of the scaled weighting matrix the spider does not fix every reference DOF. The σ²
    // are eigenvalues of Rᵀ·R, computed to about ε·σ_max² (SmallEigenSolver is backward stable), so ratios σ_min/σ_max
    // are resolved down to about 1e-7; 1e-6 stays above that round-off and far beyond any threshold the safeguard
    // sets (SPIDER_ABSORPTION_MAX_CONDITION, about 50; 3.5 measured on spiders that share no leg).
    static constexpr double RANK_TOLERANCE = 1.0e-6;

    // Throws std::invalid_argument when the sizes of `weights` do not agree (values, kinds) or a value is not finite,
    // MemoryBudgetError when the reflectors do not fit. A rank-deficient W is not an error: isRankDeficient() says so.
    SpiderChangeOfBasis(const SpiderWeightMatrix& weights, base::MemoryBudget& budget);

    [[nodiscard]] std::size_t referenceDofCount() const noexcept { return m_referenceDofCount; }
    [[nodiscard]] std::size_t legDofCount() const noexcept { return m_legDofCount; }
    // n − m, the q DOFs (0 when rank-deficient with n < m).
    [[nodiscard]] std::size_t deformationDofCount() const noexcept
    {
        return m_legDofCount > m_referenceDofCount ? m_legDofCount - m_referenceDofCount : 0;
    }
    [[nodiscard]] bool isRankDeficient() const noexcept { return m_isRankDeficient; }
    // σ_max/σ_min of the scaled W; +infinity when rank-deficient.
    [[nodiscard]] double conditionNumber() const noexcept { return m_conditionNumber; }
    // The singular values of the scaled W, increasing (m of them; empty when n < m).
    [[nodiscard]] std::span<const double> singularValues() const noexcept { return m_singularValues; }

    // rows ← Tᵀ·rows, for a matrix whose n rows are the leg DOFs in the order of W's columns (any number of columns):
    // the rows of a unit's stiffness, mass or loads on the legs. On return row i < m is reference DOF i and row m + k
    // deformation DOF k. Throws std::logic_error when rank-deficient, std::invalid_argument for another row count.
    void transformRows(linalg::DenseMatrixView rows) const;

    // matrix ← Tᵀ·matrix·T, for a symmetric n × n matrix on the leg DOFs (both triangles stored); the result is made
    // exactly symmetric (the mean of the two triangles, in a fixed order). Throws as transformRows, and
    // std::invalid_argument for a matrix that is not n × n.
    void transformSymmetric(linalg::DenseMatrixView matrix) const;

    // u_legs = T·[u_ref; q] = W⁺·u_ref + N·q (the recovery of the legs from the condensed solution). Throws as
    // transformRows, and std::invalid_argument for spans of other sizes.
    void expand(std::span<const double> reference, std::span<const double> deformation, std::span<double> legs) const;

private:
    // Rows [first, n) of every column of `rows` ← H_j·rows, H_j = I − β_j·v_j·v_jᵀ (v_j stored from row j).
    void applyReflector(std::size_t j, linalg::DenseMatrixView rows) const noexcept;
    void requireFullRank(const char* operation) const;

    std::size_t m_referenceDofCount = 0;
    std::size_t m_legDofCount = 0;
    std::vector<double> m_legScales;           // S_c: 1 or L per leg DOF (sized by one spider)
    std::vector<double> m_referenceScales;     // S_r: 1 or L per reference DOF
    base::BudgetedBuffer<double> m_reflectors; // column-major n × m: v_j in rows j..n−1 of column j, v_j[j] = 1
    std::vector<double> m_betas;               // β_j, m of them (0 for a zero column)
    std::vector<double> m_r;                   // R, column-major m × m (upper triangle)
    std::vector<double> m_singularValues;
    double m_conditionNumber = 0.0;
    bool m_isRankDeficient = false;
};

} // namespace oos::reduction
