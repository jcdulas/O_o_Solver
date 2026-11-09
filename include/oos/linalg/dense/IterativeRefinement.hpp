#pragma once

#include "oos/linalg/dense/DenseMatrixView.hpp"

#include <cstddef>
#include <vector>

namespace oos::linalg
{

// The iterative refinement of the relieved solution of a free body solved on its conditioned supports, the rule the
// full path and the full path streamed share.
//
// The problem: inertia relief balances the loads by the inertia of a rigid-body acceleration, P' = P − M·Φ·ü with
// ü = (Φᵀ·M·Φ)⁻¹·Φᵀ·P (Φ the six rigid-body motions from the geometry), and the balanced loads deform the body as on
// any statically determinate support; the solvers hold it at six conditioned supports S
// (assembly::chooseConditionedSupports) and solve K_ll·u_l = P'_l. A single solve carries the round-off of the factor
// amplified by the condition number of the body held at S (κ_S, within a few times that of the free body, λ_max/λ_7, on
// the ill-conditioned bench; far above that of a clamped twin: the free body's lowest elastic modes are its softest),
// where the residual r = P'_l − K_ll·u_l carries ε·|K|·|u| when K·u is made from the elements' own displacements (u_m =
// Gm·u_n, then Kgg, then Gmᵀ: the relative motion across a stiff element taken from its points, not from the cancelling
// terms of Knn).
//
// The method: fixed-precision iterative refinement (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed.,
// §12.1) in the gauge M-orthogonal to the rigid motions, u_l ← u_l + K_ll⁻¹·r on the same factor (the full path) or by
// another streamed solve (the full path streamed), with the stopping rule of LAPACK's xPORFS: a step is taken while
// the largest correction of the last one, relative to the largest displacement of its vector, is above TOLERANCE and at
// most DECREASE times the one before (a step that does not halve the correction has reached the noise of the residual),
// and at most MAX_STEPS; a correction larger than the one before is not applied (the noise growing where ε·κ nears
// one). The first correction is an estimate of the error of a single solve (the condensed path's, which is not
// refined).
//
// Measured on a series of ill-conditioned models: the first
// correction 5e-9 on the base model (one step), 1.4e-4 with fasteners 1e8 times stiffer than the skins (refined to
// 5e-10 in three steps), 4e-2 with shells 1e3 times thinner and 8e-2 on the near-mechanism (no convergence below about
// 2e-3: the condition number of the free body there is of the order of 1/ε, ε·κ about 8 and 0.7).
//
// Thread-safety: a value; not shared. Ownership: owns its corrections.
class IterativeRefinement
{
public:
    // A correction at most this share of its vector's largest displacement ends the refinement: the agreement of the
    // paths taken as round-off on large models (1e-9 to 2e-8 on a series of barrel models), the base model's
    // single solve 5e-9 (one step).
    static constexpr double TOLERANCE = 1.0e-8;
    // A step whose correction is above this share of the previous one has reached the noise of the residual (xPORFS).
    static constexpr double DECREASE = 0.5;
    // The most steps (xPORFS's ITMAX is 5; the bench converges in three where it converges).
    static constexpr std::size_t MAX_STEPS = 4;

    // What to do with a step: apply its correction (the first, or one smaller than the one before: a larger one is the
    // residual's noise growing, which the iterate does not take), and whether another step is taken.
    struct Step
    {
        bool isAccepted = false;
        bool continues = false;
    };
    // Records the largest relative correction of a step (largestRelative).
    [[nodiscard]] Step record(double correction);
    [[nodiscard]] const std::vector<double>& corrections() const noexcept { return m_corrections; }

    // The largest |correction| of each column over the largest |displacement| of that column, the largest over the
    // columns (a zero column counts 0). Both views have the same shape (std::invalid_argument otherwise).
    [[nodiscard]] static double largestRelative(ConstDenseMatrixView corrections, ConstDenseMatrixView displacements);

private:
    std::vector<double> m_corrections; // at most MAX_STEPS
};

} // namespace oos::linalg
