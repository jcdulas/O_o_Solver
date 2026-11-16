#pragma once

#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementDefinition.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementLoads.hpp"
#include "oos/elements/mixedHybrid/shell/ShellReferenceElement.hpp"

#include <array>
#include <span>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The stresses of a shell element of the mixed hybrid formulation from its local unknowns x (ShellElement.hpp):
// the moments m = M_σ⁻¹ ((G + C) x − h) of the element's moment space, the membrane strain I x of its membrane space,
// the forces of the partially mixed form N = Ā (e − e0) + B D⁻¹ m, the transverse shear forces Q = H γ; and the
// geometric stiffness of a field of membrane forces.
//
// Reference: tdnns/element.rs (resultants, membrane_forces, geometric_membrane).
//
// Thread-safety: pure functions.

// The generalised forces at the reference point (ξ, η): [N11, N22, N12, M11, M22, M12, Q1, Q2] in the local frame,
// about the surface of the grid points, for the element unknowns x; `momentTerm` the thermal term h of the moments
// (null without temperature) and `temperatures` (T0, T1) there.
// Reference: tdnns/element.rs::resultants
[[nodiscard]] std::array<double, 8> resultants(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                               const ShellElementMatrices& matrices, std::span<const double> x,
                                               const Vector* momentTerm, const std::array<double, 2>& temperatures,
                                               double xi, double eta);

// The membrane forces (local N11, N22, N12, about the surface of the grid points) at the surface points for the
// element unknowns x (`momentTerm`, `temperatures`: the thermal term and the temperatures; null and empty without).
// Reference: tdnns/element.rs::membrane_forces
[[nodiscard]] std::vector<Voigt3> membraneForces(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                                 const ShellElementMatrices& matrices, std::span<const double> x,
                                                 const Vector* momentTerm, const TemperaturesAt& temperatures);

// The geometric stiffness of the membrane forces n (at the surface points): the second variation of the Regge
// interpolant of the Green strain, N : I_R(∂_α u · ∂_β u), written with the membrane functionals. A matrix on the
// element unknowns (displacement block only).
//
// Faithful to the reference, which does not use this function: its weights of the three constant functionals are
// read through covariantToVoigt alone, without the turn T(φ) of the frame of the constant states that
// reggeFunctionals applies, so it agrees with the membrane part of evaluateNonlinear (ShellNonlinear.hpp, the
// geometric stiffness the formulation uses) only where φ = 0 or with `mapReferenceConstants`.
// Reference: tdnns/element.rs::geometric_membrane
[[nodiscard]] Matrix geometricMembrane(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                       const ShellElementMatrices& matrices, std::span<const Voigt3> forces);

} // namespace oos::elements::mixedHybrid
