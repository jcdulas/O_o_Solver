#pragma once

#include "oos/elements/shells/Laminate.hpp"
#include "oos/elements/shells/ShellMaterial.hpp"

#include <array>

namespace oos::design
{

// The strength ratio of a ply: the factor λ by which its stresses would be multiplied to reach failure, F(λ·σ) = 1
// for the failure criterion F of the laminate's theory (PCOMP FT; PARAM SRCOMPS; the CSTRAT response of DesignResp1).
// With F = a + b where a is the quadratic part and b the linear part of the criterion at σ:
//   - HILL (quadratic only, the allowable chosen by the sign of each stress, which λ > 0 keeps): λ = 1/√F(σ);
//   - HOFF and TSAI (Hoffman, Tsai-Wu: a·λ² + b·λ = 1): λ = (−b + √(b² + 4a)) / (2a), the positive root;
//   - STRN (maximum strain, linear): λ = 1/F(σ).
// The bond strength ratio of the interlaminar shear (SB over the larger transverse shear) is 1/FB. A ply whose
// criterion is 0 (no load) gets UNLOADED_STRENGTH_RATIO.
//
// Thread-safety: stateless.
class StrengthRatio
{
public:
    // The ratio of an unloaded ply: the "no bound" magnitude of the format (DCONSTR's blank bounds are ±1.0E20).
    static constexpr double UNLOADED_STRENGTH_RATIO = 1.0e20;

    // The ratio of the direct stresses of a ply of `material` under the ply stresses `stress` (σ1, σ2, τ12) whose
    // failure index is `failureIndex` (the failure_index column, which gives F for HILL and STRN).
    [[nodiscard]] static double ofDirectStresses(elements::FailureTheory theory,
                                                 const elements::ShellMaterial& material,
                                                 const std::array<double, 3>& stress, double failureIndex) noexcept;
    // The bond strength ratio from the bond failure index FB.
    [[nodiscard]] static double ofBond(double bondFailureIndex) noexcept;
};

} // namespace oos::design
