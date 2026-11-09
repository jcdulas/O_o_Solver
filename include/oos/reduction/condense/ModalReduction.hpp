#pragma once

#include "oos/linalg/dense/DenseMatrix.hpp"

#include <cstddef>
#include <cstdint>

namespace oos::reduction
{

// The q set of a Craig-Bampton unit (for SOL 103 and SOL 105; Craig & Bampton, AIAA
// Journal 6 (1968)): with its boundary DOFs a (the rows of K*) and N fixed-interface modes Φ of its interior
// (K_oo·Φ = M_oo·Φ·Λ, the boundary held), the unit's displacements are
//
//     u_o = Ψ·u_a + Φ·q,     Ψ = −K_oo⁻¹·K_oa (the constraint modes; never formed whole: no recovery matrix),
//
// and its reduced matrices on (a, q), with Ψ̃ = [Ψ; I] and Φ̃ = [Φ; 0] on the unit's equations:
//
//     K_CB = [ K*    0   ]      M_CB = [ M_aa   M_aq ]
//            [ 0     K_qq ]            [ M_qa   M_qq ]
//
//     K* = Ψ̃ᵀ·K·Ψ̃ (the static condensation),   K_qq = Φᵀ·K_oo·Φ,
//     K_aq = Ψ̃ᵀ·K·Φ̃ = K_ao·Φ + Ψᵀ·K_oo·Φ = 0   (Ψᵀ = −K_ao·K_oo⁻¹),
//     M_aa = Ψ̃ᵀ·M·Ψ̃,   M_aq = Ψ̃ᵀ·M·Φ̃,   M_qq = Φᵀ·M_oo·Φ.
//
// K_qq and M_qq are Λ and I to the convergence of the modes (the Lanczos tolerance): they are computed, not assumed,
// so that the reduced problem is the exact Rayleigh-Ritz projection on the span of (Ψ̃, Φ̃) and, with every mode of
// the interior, has the eigenvalues of the full problem to round-off. M_aa is the reduced boundary mass
// kept with K* (CachedReduction::boundaryMass); this holds the q part, which depends on the mode request.
//
// Stored in the condensation cache like K* (a reduction entry: the reduced stiffness and mass are stored like K*); the
// modes Φ themselves are recovery data (CachedRecoveryData).
//
// Thread-safety: immutable; concurrent reads are safe. Ownership: owns its matrices (budgeted in the budget given to
// them, which must outlive it).
class ModalReduction
{
public:
    // eigenvalues: N × 1 (λ = ω² of the fixed-interface modes, increasing); modalStiffness and modalMass: N × N
    // symmetric; couplingMass: boundary DOFs × N. Throws std::invalid_argument when the shapes disagree.
    ModalReduction(linalg::DenseMatrix eigenvalues, linalg::DenseMatrix modalStiffness,
                   linalg::DenseMatrix couplingMass, linalg::DenseMatrix modalMass);

    [[nodiscard]] std::size_t modeCount() const noexcept { return m_eigenvalues.rows(); }
    [[nodiscard]] std::size_t boundaryDofCount() const noexcept { return m_couplingMass.rows(); }
    [[nodiscard]] const linalg::DenseMatrix& eigenvalues() const noexcept { return m_eigenvalues; }
    // K_qq (N × N, both triangles).
    [[nodiscard]] const linalg::DenseMatrix& modalStiffness() const noexcept { return m_modalStiffness; }
    // M_aq (boundary DOFs × N).
    [[nodiscard]] const linalg::DenseMatrix& couplingMass() const noexcept { return m_couplingMass; }
    // M_qq (N × N, both triangles).
    [[nodiscard]] const linalg::DenseMatrix& modalMass() const noexcept { return m_modalMass; }
    // The bytes it holds (what its cache entry weighs in memory).
    [[nodiscard]] std::uint64_t byteCount() const noexcept;

private:
    linalg::DenseMatrix m_eigenvalues;
    linalg::DenseMatrix m_modalStiffness;
    linalg::DenseMatrix m_couplingMass;
    linalg::DenseMatrix m_modalMass;
};

} // namespace oos::reduction
