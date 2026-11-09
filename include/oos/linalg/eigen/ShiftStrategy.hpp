#pragma once

#include "oos/linalg/eigen/EigenRequest.hpp"
#include "oos/linalg/eigen/SturmCheck.hpp"

#include <cstddef>
#include <span>

namespace oos::linalg
{

// Where a shift is made, and why (the report of the solver lists them).
enum class ShiftPurpose : unsigned char
{
    // Below the whole spectrum (a request from the lowest root): ν = 0 there, the start of the Sturm walk.
    LowestBound,
    // At the lower bound of the range (a Sturm point and the first Lanczos shift).
    LowerBound,
    // At the upper bound of the range (a Sturm point only, unless modes are missing just below it).
    UpperBound,
    // Above the highest eigenvalue found, to extend the verified region (a Sturm point, then a Lanczos run if needed).
    Advance,
    // Inside an interval where the Sturm count says modes are missing.
    FillGap
};

// A shift proposed by the strategy.
struct ShiftChoice
{
    double shift = 0.0;
    ShiftPurpose purpose = ShiftPurpose::Advance;
};

// The shift strategy of the block Lanczos solver ("new shifts when the range requires them"). Pure
// logic: it proposes shifts from the request, the Sturm points and the eigenvalues found, and decides when the request
// is met; BlockLanczos factorises and runs.
//
// The rules (shifts from V1 upward with Sturm checks, a negative shift for the rigid-body modes; Grimes, Lewis & Simon
// 1994 §4):
//
// - Start. A range with a lower bound above 0: a Sturm point at the lower bound, which is also the first Lanczos
//   shift. From the lowest root (lowerBound <= 0): a shift below the spectrum, σ₀ = −RIGID_SHIFT_FRACTION·scale with
//   scale the diagonal Rayleigh bound of the pencil; if ν(σ₀) > 0 (negative roots), σ₀ is lowered by DESCENT_FACTOR
//   until ν(σ₀) = 0. A finite upper bound: a Sturm point there (the count of the range, known before any Lanczos run).
// - Completion. The Sturm walk (SturmCheck::verify) is verified up to the upper bound, or verified over at least
//   modeCount roots above the lower bound.
// - Next shift. In the first interval [a, b) whose count disagrees (b = +∞ when the walk is verified to its highest
//   point and more roots are wanted): above the highest eigenvalue found there, λ_c, half-way to the next Ritz
//   estimate r above it when the last run gives one (the Sturm count there then usually verifies the interval without
//   a run), else λ_c + STEP_FRACTION·(λ_c − a); never at or beyond b (half-way from λ_c to b instead). When no
//   eigenvalue was found in the interval: at the lowest Ritz estimate in it, else its midpoint (a + scale when b is
//   +∞). A proposal that repeats a Lanczos shift already made is replaced by the midpoint of the widest gap between
//   the eigenvalues found in [a, b) and its ends: the roots still missing there lie in a gap or are copies of a
//   repeated root, which a run near them finds since the modes already found are locked out of its basis.
// - Clearance. A shift is kept SHIFT_CLEARANCE (relative) away from every eigenvalue found, so that no eigenvalue sits
//   on a Sturm point (its side would be decided by rounding) and K − σM stays far from singular.
//
// Thread-safety: immutable once built.
class ShiftStrategy
{
public:
    // σ₀ = −RIGID_SHIFT_FRACTION·scale. The rigid-body roots (λ ≈ 0) then have θ = 1/(λ − σ₀) about 10⁶/scale, and the
    // flexible ones θ ≈ 1/λ: the shift lies below the lowest flexible root of any mesh up to about 1 000 elements
    // along an edge (λ₁/scale ~ (π/2N)²), so the low roots converge as with σ = 0, while K − σ₀M stays 10¹⁰ away from
    // the rounding of a singular K (its smallest eigenvalue about 10⁻⁶·‖K‖ against ε·‖K‖).
    static constexpr double RIGID_SHIFT_FRACTION = 1.0e-6;
    // The factor by which σ₀ is lowered while ν(σ₀) > 0.
    static constexpr double DESCENT_FACTOR = 100.0;
    // The descents allowed before the negative roots are declared unbounded (σ₀ down to
    // 10¹⁶·scale·RIGID_SHIFT_FRACTION).
    static constexpr std::size_t MAX_DESCENTS = 8;
    // A shift is kept this far (relative to max(|λ|, |σ|, RIGID_SHIFT_FRACTION·scale)) from the eigenvalues found: 10³
    // times the accuracy of a converged eigenvalue (CONVERGENCE_TOLERANCE squared over the gap, rounding about 10⁻¹³),
    // so that a Sturm point never meets an eigenvalue, and small enough not to push a shift out of a cluster.
    static constexpr double SHIFT_CLEARANCE = 1.0e-8;
    // How far beyond the highest eigenvalue found a shift advances, as a fraction of the extent found from the start
    // of the interval, when no Ritz estimate says where the next root is.
    static constexpr double STEP_FRACTION = 0.5;
    // The relative move of a shift that met an eigenvalue (a singular factor).
    static constexpr double SINGULAR_SHIFT_MOVE = 1.0e-6;

    // scale: the diagonal Rayleigh bound of the pencil (1 when it is not finite and positive). Throws
    // std::invalid_argument when the request asks for every root without an upper bound, or has
    // upperBound <= lowerBound.
    ShiftStrategy(const EigenRequest& request, double scale);

    [[nodiscard]] const EigenRequest& request() const noexcept { return m_request; }
    [[nodiscard]] double scale() const noexcept { return m_scale; }

    // The first shift: below the spectrum (from the lowest root) or at the lower bound.
    [[nodiscard]] ShiftChoice firstShift() const noexcept;
    // A lower shift when ν(shift) > 0 at the start from the lowest root.
    [[nodiscard]] double descend(double shift) const noexcept;
    // A shift next to one that met an eigenvalue, upward or downward (attempt 1, 2, ... moves further each time).
    [[nodiscard]] double moveOffEigenvalue(double shift, bool isUpward, std::size_t attempt) const noexcept;

    // True when the request is met by the eigenvalues verified.
    [[nodiscard]] bool isComplete(const SturmVerification& verification) const noexcept;

    // The eigenvalues found, increasing, that the request asks for once complete: those of [lowerPoint, upperPoint),
    // the Sturm points of the bounds as factorised (−∞ and +∞ when absent), the lowest modeCount of them when it is
    // set; the indices into `sortedEigenvalues`, as [first, first + count).
    struct Selection
    {
        std::size_t first = 0;
        std::size_t count = 0;
    };
    [[nodiscard]] Selection select(std::span<const double> sortedEigenvalues, double lowerPoint,
                                   double upperPoint) const noexcept;

    // The next shift (see the class comment). sortedEigenvalues: every eigenvalue found; ritzEstimates: the eigenvalue
    // estimates of the unconverged Ritz pairs of the last run, increasing; lanczosShifts: the shifts at which a
    // Lanczos run was made. Precondition: !isComplete(verification).
    [[nodiscard]] ShiftChoice nextShift(const SturmVerification& verification,
                                        std::span<const double> sortedEigenvalues,
                                        std::span<const double> ritzEstimates,
                                        std::span<const double> lanczosShifts) const;

    // `shift` moved upward until it clears every eigenvalue found by SHIFT_CLEARANCE.
    [[nodiscard]] double clear(double shift, std::span<const double> sortedEigenvalues) const noexcept;

private:
    [[nodiscard]] double clearanceAt(double value, double shift) const noexcept;
    [[nodiscard]] bool isNearShift(double shift, std::span<const double> lanczosShifts) const noexcept;

    EigenRequest m_request;
    double m_scale = 1.0;
};

} // namespace oos::linalg
