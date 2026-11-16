#pragma once

#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementData.hpp"
#include "oos/elements/mixedHybrid/shell/ShellReferenceElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSection.hpp"
#include "oos/elements/mixedHybrid/shell/ShellTypes.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <span>
#include <utility>

namespace oos::elements::mixedHybrid
{

// The results of one shell element of the mixed hybrid formulation from its local unknowns x (ShellElementData::local
// of the solution): the generalised forces, the state of the section at a point, the stresses of the plies and the
// transverse shear stresses through the thickness.
//
// Conventions: forces [N11, N22, N12, M11, M22, M12, Q1, Q2] and strains [e11, e22, γ12, κ11, κ22, 2 κ12, γ1, γ2] in
// the local frame (e1, e2) of the element (e1 the element x axis projected, tangentFrameXy). The resultants are those
// of the reference surface of the section (about the middle of an offset section), the "grid" resultants those about
// the surface of the grid points. The temperatures of a load case are given by a function of the position (empty: the
// case has no temperature) and its thermal moment term by ShellElementData::thermalMoments (null without).
//
// Reference: tdnns/analysis.rs (the element parts of Analysis::resultants, resultants_grid, state, state_of,
// corner_state, stress, transverse_shear) and kernel.rs::compliance.
//
// Thread-safety: pure functions.

// The temperatures (T0, T1) of a load case at a position of the space.
using TemperatureAtPosition = std::function<std::array<double, 2>(const SpaceVector&)>;

// The state of the shell at one point of an element.
// Reference: analysis.rs::PointState
struct ShellPointState
{
    // The position on the reference surface of the section (the surface of the grid points moved by the offset).
    SpaceVector x{};
    // The local tangent frame (e1 the projection of the element x axis) and the normal.
    SurfaceFrame frame{};
    SpaceVector normal{};
    // [N11, N22, N12, M11, M22, M12, Q1, Q2] in the local frame.
    std::array<double, 8> resultants{};
    // [e11, e22, γ12, κ11, κ22, 2 κ12, γ1, γ2] in the local frame (total strains).
    std::array<double, 8> strains{};
    // The angle from e1 to the material x axis of the section.
    double angle = 0.0;
    double thickness = 0.0;
    // (temperature rise, through-thickness gradient) of the load case at the point.
    std::array<double, 2> thermal{};
};

// The compliance of a section point: [A B; Bᵀ D]⁻¹ on the six membrane and bending components, H⁻¹ on the two shear
// components. Throws InternalError when the section is singular.
// Reference: kernel.rs::compliance
[[nodiscard]] std::array<std::array<double, 8>, 8> sectionCompliance(const SectionPoint& point);

// The generalised forces at (ξ, η) about the surface of the grid points.
// Reference: tdnns/analysis.rs::Analysis::resultants_grid
[[nodiscard]] std::array<double, 8> gridResultants(const ShellReferenceElement& re, const ShellElementData& element,
                                                   double xi, double eta, std::span<const double> x,
                                                   const Vector* momentTerm,
                                                   const TemperatureAtPosition& temperatureAt);

// The generalised forces at (ξ, η) about the reference surface of the section (the moments of an offset section
// about its own middle): N_s = P⁻ᵀ N_g / μ0, M_s = P⁻ᵀ M_g / μ0 − z0 N_s (offsetMap P, offsetArea μ0).
// Reference: tdnns/analysis.rs::Analysis::resultants
[[nodiscard]] std::array<double, 8> sectionResultants(const ShellReferenceElement& re, const ShellElementData& element,
                                                      double xi, double eta, std::span<const double> x,
                                                      const Vector* momentTerm,
                                                      const TemperatureAtPosition& temperatureAt);

// The state at (ξ, η) for given resultants: the strains from the compliance of the section about its own reference
// surface, plus the free strains of the temperatures.
// Reference: tdnns/analysis.rs::Analysis::state_of
[[nodiscard]] ShellPointState pointStateOf(const ShellReferenceElement& re, const ShellElementData& element, double xi,
                                           double eta, const std::array<double, 8>& resultants,
                                           const TemperatureAtPosition& temperatureAt);

// The state at (ξ, η) for the unknowns x.
// Reference: tdnns/analysis.rs::Analysis::state
[[nodiscard]] ShellPointState pointState(const ShellReferenceElement& re, const ShellElementData& element, double xi,
                                         double eta, std::span<const double> x, const Vector* momentTerm,
                                         const TemperatureAtPosition& temperatureAt);

// The state at corner c (the moments are a field of the element: the same as pointState there).
// Reference: tdnns/analysis.rs::Analysis::corner_state
[[nodiscard]] ShellPointState cornerState(const ShellReferenceElement& re, const ShellElementData& element,
                                          std::size_t corner, std::span<const double> x, const Vector* momentTerm,
                                          const TemperatureAtPosition& temperatureAt);

// The stresses at the distance z from the middle of the section: in the material axes of the section and in the axes
// of the ply that contains z (σ = Q_ply (ε + z κ − α (T0 + z T1))).
// Reference: tdnns/analysis.rs::Analysis::stress
[[nodiscard]] std::pair<Voigt3, Voigt3> plyStresses(const ShellElementData& element, const ShellPointState& state,
                                                    double z);

// The transverse shear stresses [τ_xz, τ_yz] at the distance z from the middle of the section, in the material axes of
// the section and in the axes of the ply: the integral through the thickness of the equilibrium of the in-plane
// stresses, from the gradients of N and M (central differences of step 1e-5 in the reference coordinates), the part
// of Q that the gradients do not carry distributed as the equilibrium distribution of the section (PlyShear).
// Reference: tdnns/analysis.rs::Analysis::transverse_shear
[[nodiscard]] std::pair<std::array<double, 2>, std::array<double, 2>> transverseShear(
    const ShellReferenceElement& re, const ShellElementData& element, double xi, double eta, std::span<const double> x,
    const Vector* momentTerm, const TemperatureAtPosition& temperatureAt, double z);

} // namespace oos::elements::mixedHybrid
