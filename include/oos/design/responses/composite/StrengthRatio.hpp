#pragma once

#include "oos/elements/shells/Laminate.hpp"
#include "oos/elements/shells/ShellMaterial.hpp"
#include "oos/elements/shells/ShellRecovery.hpp"

#include <array>

namespace oos::design
{

// The strength ratio of a ply, for the CompositeStrengthRatio response of a DesignResp1: the factor λ by which its
// stresses would be multiplied to reach failure, F(λ·σ) = 1 for the failure criterion F of the laminate's theory, and
// the strength ratio of a bond. The same values as the strength ratio table of the laminated shells
// (elements::plyStrengthRatio, elements::bondStrengthRatio, which give the formulas).
//
// Thread-safety: stateless.
class StrengthRatio
{
public:
    // The ratio of an unloaded ply: the "no bound" magnitude of the format (DesignConstr's blank bounds are ±1.0E20).
    static constexpr double UNLOADED_STRENGTH_RATIO = elements::UNLOADED_STRENGTH_RATIO;

    // The ratio of the direct stresses of a ply of `material` under the ply stresses `stress` (σ1, σ2, τ12) whose
    // failure index is `failureIndex` (the failure_index column, which gives F for HILL and STRN).
    [[nodiscard]] static double ofDirectStresses(elements::FailureTheory theory,
                                                 const elements::ShellMaterial& material,
                                                 const std::array<double, 3>& stress, double failureIndex) noexcept;
    // The bond strength ratio from the bond failure index FB.
    [[nodiscard]] static double ofBond(double bondFailureIndex) noexcept;
};

} // namespace oos::design
