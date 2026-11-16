#pragma once

#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementDefinition.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"
#include "oos/elements/mixedHybrid/shell/ShellReferenceElement.hpp"

#include <array>
#include <functional>

namespace oos::elements::mixedHybrid
{

// The loads of a shell element of the mixed hybrid formulation on its local unknowns (ShellElement.hpp for their
// order; document 34 of the reference, section 6): a force per unit area (pressure along the normal, traction), the
// thermal load through the free strains, the inertia of a uniform acceleration, and the load stiffness of a pressure
// that follows the surface.
//
// Reference: tdnns/element.rs (area_load, thermal_load, inertia_load, follower_pressure).
//
// Thread-safety: pure functions (the callbacks are called on the calling thread).

// A force per unit area (Cartesian components) at a point of the element.
using AreaForceAt = std::function<SpaceVector(const PointGeometry&)>;
// The temperatures (T0 the rise at the surface, T1 the through-thickness gradient) at a point of the element.
using TemperaturesAt = std::function<std::array<double, 2>(const PointGeometry&)>;
// A pressure at a point of the element.
using PressureAt = std::function<double(const PointGeometry&)>;

// f_3i+c = ∫ N_i f_c dA: the load vector of a force per unit area on the displacement.
// Reference: tdnns/element.rs::area_load
[[nodiscard]] Vector areaLoad(const ShellReferenceElement& re, const ShellElementDefinition& def,
                              const AreaForceAt& force);

// The thermal load: the equivalent forces f = Iᵀ f_R + (G + C)ᵀ M_σ⁻¹ h, the term h = ∫ Sᵀ (κ0 + D⁻¹ Bᵀ e0) dA of the
// moments and the term f_R = ∫ Rᵀ Ā e0 dA of the membrane (on the membrane coefficients), (e0, κ0) the free strains
// (freeStrains). A point where both temperatures are zero adds nothing.
// Reference: the tuple (f, h, fr) of tdnns/element.rs::thermal_load
struct ShellThermalLoad
{
    Vector forces;
    Vector momentTerm;
    Vector membraneTerm;
};

// Reference: tdnns/element.rs::thermal_load
[[nodiscard]] ShellThermalLoad thermalLoad(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                           const ShellElementMatrices& matrices, const TemperaturesAt& temperatures);

// The load of the inertia forces of a uniform acceleration a: ∫ (I0 uᵀ + I1 βᵀ) a dA (the force on the director of
// an offset section included).
// Reference: tdnns/element.rs::inertia_load
[[nodiscard]] Vector inertiaLoad(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                 const SpaceVector& acceleration);

// The load stiffness of a pressure that follows the surface (its symmetric part): the derivative with respect to the
// displacement of ∫ p (φ_,1 × φ_,2) · δφ dξ, with φ = X + u.
// Reference: tdnns/element.rs::follower_pressure
[[nodiscard]] Matrix followerPressure(const ShellReferenceElement& re, const ShellElementDefinition& def,
                                      const PressureAt& pressure);

} // namespace oos::elements::mixedHybrid
