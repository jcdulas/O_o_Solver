#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace oos::design
{

// The first-order approximation of a response in a design cycle (DOPTPRM APRCOD, or a kind the caller asks for).
enum class ApproximationKind : std::uint8_t
{
    Direct,     // APRCOD = 1: linear in the design variables, r ≈ r0 + Σ ∂r/∂x_i·(x_i − x0_i)
    Reciprocal, // linear in the intermediate variables 1/x_i (asked by the caller only: every response)
    Mixed,      // APRCOD = 2 (the default): reciprocal for the responses of the structure, direct for the others
    Convex,     // APRCOD = 3: convex linearisation, direct where ∂f/∂x_i > 0 and reciprocal where ∂f/∂x_i < 0
    Hints       // the intermediate variables of the approximation hints: y_i = x_i^p, p per response and
                // variable (design/aids/ApproximationHints), an option of O_o_Solver beside the APRCOD values
};

[[nodiscard]] std::string_view approximationKindName(ApproximationKind kind) noexcept;
// "direct", "reciprocal", "mixed", "convex"; nothing for another name.
[[nodiscard]] std::optional<ApproximationKind> findApproximationKind(std::string_view name) noexcept;
// The kind of a DesignOptimParams APRCOD (1, 2, 3).
[[nodiscard]] ApproximationKind approximationKindOfCode(std::int64_t approximationCode) noexcept;
// Whether the mixed method approximates a response of this DesignConstrScreen type in the reciprocal variables: the
// grid and element responses (DISP, SPCFORCE, STRESS, STRAIN, FORCE, ESE and the composite ones), whose dependence on
// sizing variables is close to 1/x (exact for a statically determinate structure: a stress is a force over an area);
// WEIGHT, VOLUME and FRMASS are linear in the properties and stay direct, as the equations (EQUA, DesignResp3), whose
// form the approximation cannot know.
[[nodiscard]] bool isReciprocalType(std::string_view responseType) noexcept;

// The first-order expansion of a function f at x0 in the intermediate variables of an ApproximationKind (Vanderplaats,
// Numerical Optimization Techniques for Engineering Design, 4th ed., §7.3; Fleury and Braibant, "Structural
// optimization: a new dual method using mixed variables", Int. J. Numer. Meth. Engng 23, 1986, for the convex
// linearisation):
//
//     f(x) ≈ f0 + Σ_i t_i(x_i),   direct:      t_i = g_i·(x_i − x0_i),               ∂t_i/∂x_i = g_i,
//                                 reciprocal:  t_i = g_i·x0_i·(x_i − x0_i)/x_i,       ∂t_i/∂x_i = g_i·(x0_i/x_i)²,
//
// with g = ∇f(x0); the reciprocal term is the linear expansion in y_i = 1/x_i (∂f/∂y_i = −x0_i²·g_i), so that the
// approximation and its gradient equal f0 and g at x0. A variable takes 1/x only when x0_i > 0 and its lower bound in
// the approximate problem is > 0 (the term would otherwise pass through infinity); else it stays direct.
//
// The hints (ApproximationKind::Hints) generalise the reciprocal term to a power p of the variable, the linear
// expansion in y_i = x_i^p (∂f/∂y_i = g_i·x0_i^(1−p)/p):
//     t_i = g_i·(x_i^p − x0_i^p)/(p·x0_i^(p−1)),   ∂t_i/∂x_i = g_i·(x_i/x0_i)^(p−1),
// p = 1 direct, p = −1 reciprocal; a power other than 1 under the same condition as 1/x.
//
// Thread-safety: immutable. Ownership: owns x0 and the flags.
class ResponseApproximation
{
public:
    ResponseApproximation() = default;
    // `lower`: the lower bounds of the approximate problem (after the move limits).
    ResponseApproximation(std::span<const double> x0, std::span<const double> lower);

    [[nodiscard]] const std::vector<double>& x0() const noexcept { return m_x0; }
    // Whether variable i is taken as 1/x for a function of derivative `gradient` along it. `kind` is Direct, Reciprocal
    // or Convex (Mixed is resolved by the caller per response type).
    [[nodiscard]] bool isReciprocal(ApproximationKind kind, std::size_t i, double gradient) const noexcept;
    // f at x. `powers`: the power of each variable for ApproximationKind::Hints (empty: every one direct).
    [[nodiscard]] double value(ApproximationKind kind, double value0, std::span<const double> gradient0,
                               std::span<const double> x, std::span<const int> powers = {}) const noexcept;
    // ∇f at x, into `gradient`.
    void gradient(ApproximationKind kind, std::span<const double> gradient0, std::span<const double> x,
                  std::span<double> gradient, std::span<const int> powers = {}) const noexcept;

private:
    std::vector<double> m_x0;
    std::vector<std::uint8_t> m_canBeReciprocal;
};

} // namespace oos::design
