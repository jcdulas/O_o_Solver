#pragma once

#include "oos/elements/mixedHybrid/shell/ShellSection.hpp"
#include "oos/elements/shells/ShellSection.hpp"

#include <string>
#include <variant>

namespace oos::elements::mixedHybrid
{

// The section law of the mixed hybrid shell (mixedHybrid::Section, at a reference thickness in its material axes) from
// the section of a shell property as the solver reads it (elements::ShellSection: ShellProp with IsoMat, AnisoMat2D,
// OrthoMat2D; CompProp and CompPropGlobal as their equivalent ShellProp), docs/mixed-hybrid-formulation-spec.md
// section 4.1:
//   A = T G1, D = (12I/T³) T³/12 G2, B = T² couplingMatrix (the sign of the lamination theory, z along the normal:
//   the mixed shell's layer strain is e + z κ), H = κ G T along the material axes (the inverse of the transverse
//   shear compliance; empty for a section without transverse shear flexibility: the Kirchhoff section),
//   I0 = ρ T + NSM, I1 = Σ ρ_k (z_k² − z_{k−1}²) / 2, I2 = Σ ρ_k (z_k³ − z_{k−1}³) / 3 (ρ T³ / 12 for a ShellProp),
//   the thermal resultants per unit ΔT (N_T = thermalA ΔT + thermalB T', M_T = thermalB ΔT + thermalD T').
// The reference thickness is T (the laminate's thickness for a laminate); the element interpolates its corner
// thicknesses.
//
// A membrane-only property (MID2 blank, or a laminate's membrane option) gives SectionParts::MembraneOnly with
// B = D = 0, no transverse shear, the thermal force alone and no rotary inertia (I1 = I2 = 0); a bending-only one (MID1
// blank, or a laminate's bending option) SectionParts::BendingOnly with A = B = 0 and the thermal moments alone, its
// Kirchhoff limit measured on 12 D / t² (spec section 4.1).
//
// Returns the message of a section the formulation does not take instead: no membrane nor bending material, the
// plane-strain option (MID2 = −1), a thickness that is not positive.
[[nodiscard]] std::variant<Section, std::string> mixedSectionOf(const elements::ShellSection& section);

} // namespace oos::elements::mixedHybrid
