#pragma once

#include "oos/linalg/dense/DenseMatrixView.hpp"
#include "oos/linalg/dense/Inertia.hpp"

#include <cstddef>
#include <cstdint>

namespace oos::linalg
{

// The eigenproblem a pencil poses, which sets the spectral transformation, the inner product and the meaning of the
// Sturm counts:
//
// - Vibration: K φ = λ M φ, M positive semi-definite. Op = (K − σM)⁻¹·M, self-adjoint in the M-inner product,
//   θ = 1/(λ − σ) (Ericsson & Ruhe 1980); the negative pivots of K − σM count the eigenvalues below σ.
// - Buckling: K φ = λ B φ with K positive definite and B = −K_d indefinite (SOL 105: (K + λ K_d) φ = 0, λ the
//   load factor of the preload). Op = (K − σB)⁻¹·B, self-adjoint in the K-inner product (K·Op = B + σ·B·(K − σB)⁻¹·B
//   is symmetric), with the same θ = 1/(λ − σ). This is Grimes, Lewis & Simon's buckling transformation
//   (K − σB)⁻¹·K = I + σ·Op written without its identity part: the same Krylov spaces and eigenvectors, the infinite
//   roots of the null space of B mapped to θ = 0 (as the massless DOFs of a vibration problem) instead of θ = 1, and
//   σ = 0 allowed. With K positive definite the negative pivots of K − σB count the eigenvalues in (0, σ) for σ > 0
//   and in (σ, 0) for σ < 0 (Sylvester's law of inertia applied to K^(−1/2)·(K − σB)·K^(−1/2) = I − σ·K^(−1/2)·B·
//   K^(−1/2), whose eigenvalues are 1 − σ/λ).
enum class SpectralTransform : unsigned char
{
    Vibration,
    Buckling
};

// What the factorisation of K − σM at one shift gave: its inertia, whose negative count is the Sturm count of σ (the
// number of eigenvalues of K φ = λ M φ below σ, Sylvester's law of inertia applied to K − σM; Grimes, Lewis & Simon,
// SIAM J. Matrix Anal. Appl. 15 (1994) §2).
struct ShiftFactorisation
{
    double shift = 0.0;
    Inertia inertia;

    [[nodiscard]] std::size_t sturmCount() const noexcept { return inertia.negative; }
    // A zero pivot: σ is an eigenvalue (to rounding), K − σM cannot be solved with; the caller moves the shift.
    [[nodiscard]] bool isSingular() const noexcept { return inertia.zero > 0; }
};

// The matrix pencil (K, M) of a symmetric generalised eigenproblem K φ = λ M φ, as the eigenvalue solver sees it: the
// factorisation of K − σM at a shift, the solves with it, the products with M (the right-hand matrix: the mass of a
// vibration problem, B = −K_d of a buckling problem) and with the matrix of the inner product (M, resp. K).
// The caller provides the assembly and the factorisation (linalg stays below assembly): SparsePencil
// implements it on two SymmetricCsc of one pattern; a solution sequence may implement it otherwise (a condensed
// interface problem, a dense test matrix).
//
// A virtual interface at a plug-in point: it is called once per block of vectors, never per term.
//
// Thread-safety: factorise() is called by one thread; solve() and multiplyMass() are const and may use a pool inside,
// but the solver calls them from one thread at a time. Ownership: the implementation owns its factor.
class ShiftedPencil
{
public:
    ShiftedPencil() = default;
    ShiftedPencil(const ShiftedPencil&) = delete;
    ShiftedPencil& operator=(const ShiftedPencil&) = delete;
    ShiftedPencil(ShiftedPencil&&) = delete;
    ShiftedPencil& operator=(ShiftedPencil&&) = delete;
    virtual ~ShiftedPencil() = default;

    [[nodiscard]] virtual std::int32_t dimension() const noexcept = 0;

    // Factorises K − σM with pivoting for an indefinite matrix, replacing the previous factor. A
    // singular factor is returned, not thrown (isSingular()): the solver then moves the shift.
    [[nodiscard]] virtual ShiftFactorisation factorise(double shift) = 0;

    // X ← (K − σM)⁻¹·X with the last factor, X dimension() × r. Precondition: a factor that is not singular.
    virtual void solve(DenseMatrixView block) const = 0;

    // Y ← M·X, X and Y dimension() × r, distinct storage (M the right-hand matrix of the pencil).
    virtual void multiplyMass(ConstDenseMatrixView x, DenseMatrixView y) const = 0;

    // The eigenproblem the pencil poses (Vibration unless overridden).
    [[nodiscard]] virtual SpectralTransform transform() const noexcept { return SpectralTransform::Vibration; }

    // Y ← W·X with W the matrix of the inner product in which Op is self-adjoint: M for a vibration problem (the
    // default), K for a buckling problem. X and Y dimension() × r, distinct storage.
    virtual void multiplyInnerProduct(ConstDenseMatrixView x, DenseMatrixView y) const { multiplyMass(x, y); }

    // min K_ii / M_ii over the equations with M_ii > 0 and K_ii > 0 (+∞ when there is none): the Rayleigh quotient
    // of a unit vector, hence an upper bound of the lowest finite eigenvalue (for a buckling pencil min K_ii / |B_ii|
    // over B_ii ≠ 0, the magnitude of the lowest root on the side of the sign of that B_ii). It gives the solver the
    // scale of the spectrum before the first shift (ShiftStrategy).
    [[nodiscard]] virtual double diagonalRayleighBound() const = 0;
};

} // namespace oos::linalg
