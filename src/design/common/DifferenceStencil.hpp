#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

// The difference of a function of the design variables along a direction v, d/dσ f(x0 + σ·v) at σ = 0, for the
// second-order products that difference a first-order quantity (external/VectorJacobianProducts,
// aids/SecondDerivatives). Internal to the design module.
//
// The step. Each moved variable (v_j != 0) has its own scale s_j: |x_j|, or for a variable at zero the width of its
// bounds, or without bounds `stepAtZero / relative` (so that its step is `stepAtZero`). The step along v is
//
//     t = relative · min_j (s_j / |v_j|),
//
// so that no variable moves by more than `relative` of its own scale. One scale for the whole vector (the largest
// |x_j|) moved a thickness of 2 by 7 when a modulus of 70 000 was a variable too: the difference then crossed zero.
//
// The bounds. The states x0 + σ·v stay within [lower, upper] of every moved variable:
//   - central difference at h = min(t, room on the + side, room on the − side) when h >= SMALLEST_STEP_SHARE · t:
//         f' ≈ (f(h) − f(−h)) / (2h)                                   (error O(h²))
//   - else one-sided, toward the side with more room, at h = min(t, room / 2) when h >= SMALLEST_STEP_SHARE · t:
//         f' ≈ ±(−3·f(0) + 4·f(±h) − f(±2h)) / (2h)                    (error O(h²); the three-point forward
//         difference, Abramowitz and Stegun, Handbook of Mathematical Functions, 1964, formula 25.3.4)
//   - else (no usable room on either side: a variable held between bounds closer than the smallest step, or the
//     direction leaves two bounds at once) the central difference at t with the bounds ignored: the bounds are the
//     optimiser's limits, and a model the step makes invalid is refused by its analysis.
// A variable whose bounds are equal is fixed by the optimiser, not limited by the model: it gives no room limit. A
// bound of magnitude NO_BOUND or more (the default of a design variable) is no bound.
namespace oos::design::detail
{

// A design variable bound at or beyond this magnitude constrains nothing (the default bounds of a design variable).
inline constexpr double STENCIL_NO_BOUND = 1.0e20;
// The smallest share of the nominal step a clipped step may keep: below it the rounding of the difference grows by
// more than 1 / SMALLEST_STEP_SHARE and the other form (one-sided, then unclipped) is taken.
inline constexpr double SMALLEST_STEP_SHARE = 0.1;

// One state of a stencil: x0 + offset·v, and the weight of its value in the derivative.
struct StencilPoint
{
    double offset = 0.0;
    double weight = 0.0;
};

// d/dσ f(x0 + σ·v) ≈ Σ_k weight_k · f(x0 + offset_k · v).
struct DifferenceStencil
{
    std::vector<StencilPoint> points; // two (central) or three (one-sided; the first is the state x0, offset 0)
    double step = 0.0;                // h
    bool isOneSided = false;
    bool ignoresBounds = false; // the last form: no usable room, the bounds were ignored
};

[[nodiscard]] inline bool isStencilBound(double bound) noexcept
{
    return std::isfinite(bound) && std::abs(bound) < STENCIL_NO_BOUND;
}

// The nominal step t along v (the comment above). `lower` and `upper` may be empty (no bounds); 0 when v is zero.
[[nodiscard]] inline double directionStep(std::span<const double> x, std::span<const double> v,
                                          std::span<const double> lower, std::span<const double> upper,
                                          double relative, double stepAtZero) noexcept
{
    double smallest = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < x.size(); ++j)
    {
        if (v[j] == 0.0)
            continue;
        double scale = std::abs(x[j]);
        if (!(scale > 0.0))
        {
            const bool hasWidth = j < lower.size() && j < upper.size() && isStencilBound(lower[j]) &&
                                  isStencilBound(upper[j]) && upper[j] > lower[j];
            scale = hasWidth ? upper[j] - lower[j] : stepAtZero / relative;
        }
        smallest = std::min(smallest, scale / std::abs(v[j]));
    }
    return std::isfinite(smallest) ? relative * smallest : 0.0;
}

// The room along +v (sign = +1) or −v (sign = −1) before a moved variable reaches a bound; infinite without bounds.
[[nodiscard]] inline double directionRoom(std::span<const double> x, std::span<const double> v,
                                          std::span<const double> lower, std::span<const double> upper,
                                          double sign) noexcept
{
    double room = std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < x.size() && j < lower.size() && j < upper.size(); ++j)
    {
        const double move = sign * v[j];
        if (move == 0.0 || !(upper[j] > lower[j]))
            continue; // not moved, or fixed by the optimiser (equal bounds): no limit
        const double bound = move > 0.0 ? upper[j] : lower[j];
        if (!isStencilBound(bound))
            continue;
        room = std::min(room, std::max((bound - x[j]) / move, 0.0));
    }
    return room;
}

// The stencil of the nominal step t (directionStep, or a fraction of it for an extrapolation) within the bounds.
// Precondition: t > 0.
[[nodiscard]] inline DifferenceStencil differenceStencil(std::span<const double> x, std::span<const double> v,
                                                         std::span<const double> lower,
                                                         std::span<const double> upper, double t)
{
    const double plusRoom = directionRoom(x, v, lower, upper, 1.0);
    const double minusRoom = directionRoom(x, v, lower, upper, -1.0);
    const auto central = [](double h, bool ignoresBounds)
    {
        return DifferenceStencil{{{h, 0.5 / h}, {-h, -0.5 / h}}, h, false, ignoresBounds};
    };
    const double centralStep = std::min({t, plusRoom, minusRoom});
    if (centralStep >= SMALLEST_STEP_SHARE * t)
        return central(centralStep, false);
    const double sign = plusRoom >= minusRoom ? 1.0 : -1.0;
    const double sidedStep = std::min(t, 0.5 * std::max(plusRoom, minusRoom));
    if (sidedStep >= SMALLEST_STEP_SHARE * t)
    {
        // ±(−3·f(0) + 4·f(±h) − f(±2h)) / (2h).
        const double scale = sign * 0.5 / sidedStep;
        return DifferenceStencil{
            {{0.0, -3.0 * scale}, {sign * sidedStep, 4.0 * scale}, {2.0 * sign * sidedStep, -scale}},
            sidedStep,
            true,
            false};
    }
    return central(t, true);
}

} // namespace oos::design::detail
