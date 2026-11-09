#pragma once

#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/shells/ShellRecovery.hpp"
#include "oos/elements/shells/ShellSection.hpp"
#include "oos/elements/shells/ShellSectionRate.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingBending.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingGeometry.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingWarpOffsets.hpp"
#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingMembrane.hpp"

#include <array>
#include <cstdint>
#include <optional>

namespace oos::elements
{

// One quadrilateral plate formulated in its element frame, the formulation of the ShellQuad4Drilling family
// (ShellQuad4DrillingElement, ShellQuad4DrillingElement), on its 24 local DOFs (u, v, w, θx, θy, θn per corner, node by node;
// translations along x, y, n, rotations about them, on the reference plane moved by ZOFFS): the membrane
// ShellQuad4DrillingMembrane on (u, v, θn) of the corners and the bending ShellQuad4DrillingBending on (w, θx, θy),
//
//   K24 = K_membrane,12 (on u, v, θn) + K_bending,12 (on w, θx, θy).
//
// There is no membrane-bending coupling: the family refuses a section with one (MID4, a laminate with B ≠ 0). The
// family (ShellQuad4DrillingElement) carries K24 to the displacement systems of the grid points with ShellNodeFrames. The
// section is given in the element frame (ShellSection::rotatedToElementFrame). The mass, the loads and the recovery are
// given on the same local DOFs.
//
// Thermal loads: a temperature change uniform over the element (ShellTemperatureChange) gives the resultants N_T, M_T
// of ShellSection::thermalResultants at each Gauss point; N_T loads the membrane as an initial stress, f = Σ_g w det J
// Bᵀ N_T (B the strain matrix of ShellQuad4DrillingMembrane), M_T the bending as an initial curvature κ_T = −D⁻¹ M_T (the
// sign of ShellQuad4DrillingBending's curvatures, ShellQuad4DrillingBending::initialCurvatureLoads); the recovery subtracts them (σ = E
// (ε − α ΔT)).
//
// A warped element is formed flat in its mean plane and joined to its grid points by the rigid offsets of
// ShellQuad4DrillingWarpOffsets: its stiffness, differential stiffness, their rates and its loads are Tᵀ K T and Tᵀ f of the
// flat element's, its recovery reads the displacements of the projections q_P = T q. Its mass is not carried by the
// offsets: it is the mass of the flat element put on the grid points, which lie on the modelled surface (the
// projections are an artefact of the flat formulation), so a lumped mass stays diagonal and its first moments are
// those of the grid points; the difference from Tᵀ M T is of the order of the warp. The links keep their length under
// a temperature change: a free uniform expansion α ΔT X of the grid points moves the projections out of the flat plane
// by α ΔT h_i, which the flat element takes as a twist, so a warped mesh freely heated shows small transverse shear
// forces, of the order of κG t α ΔT h/ℓ (ℓ the element size), that vanish as the mesh is refined.
//
// Thread-safety: immutable once built. A plain value, built per element.
class ShellQuad4DrillingPlate
{
public:
    // `section` in the element frame, `thickness` the thickness at each corner.
    ShellQuad4DrillingPlate(const ShellQuad4DrillingGeometry& geometry, const ShellSection& section,
                   const std::array<double, 4>& thickness);

    [[nodiscard]] const ShellQuad4DrillingGeometry& geometry() const noexcept { return m_geometry; }
    // K24 (24 × 24, local DOFs).
    [[nodiscard]] base::MatN<24> stiffness() const;
    // The differential stiffness on the local DOFs of the membrane resultants (Nx, Ny, Nxy) of the preload, constant
    // over the element (ShellDifferential: the 2 × 2 Gauss rule on the plane coordinates, exact for a parallelogram).
    [[nodiscard]] base::MatN<24> differentialStiffness(const std::array<double, 3>& membraneForce) const;
    // dK24/dT with respect to the ShellProp thickness T when every corner thickness is proportional to it (t_i = r_i·T:
    // a blank corner, or TFLAG = 1), in closed form: the membrane (∫ t Bᵀ G1 B and its rotation penalty ∫ t γ hᵀ h) ∝
    // T, the bending of ShellQuad4DrillingBending::thicknessRate (the curvatures ∝ T³, the saturated edge shear):
    //   dK/dT = K_membrane/T + dK_bending/dT.
    [[nodiscard]] base::MatN<24> thicknessRate(double thickness) const;
    // The rates with respect to T of the resultants of the place states `states` of the local displacements q
    // (placeStates) at fixed q, under the same condition (shellResultantRates, the shear of ShellQuad4DrillingBending).
    [[nodiscard]] std::array<ShellPlaceState, 5> placeStateRates(const std::array<ShellPlaceState, 5>& states,
                                                                 const std::array<double, 24>& q,
                                                                 double thickness) const;
    // dM24/dT under the same condition: the translational mass ∫ ρ t N_i N_j dA ∝ T (the NSM does not follow T), the
    // rotary inertia ∫ ρ t³/12 N_i N_j dA ∝ T³ (mass()).
    [[nodiscard]] base::MatN<24> massThicknessRate(MassFormulation formulation, double thickness) const;
    // d²K24/dT² under the same condition: the membrane has none (degree 1), the bending
    // ShellQuad4DrillingBending::thicknessSecondRate.
    [[nodiscard]] base::MatN<24> thicknessSecondRate(double thickness) const;
    // The second rates of the resultants of the place states at fixed q (shellResultantSecondRates).
    [[nodiscard]] std::array<ShellPlaceState, 5> placeStateSecondRates(const std::array<ShellPlaceState, 5>& states,
                                                                       const std::array<double, 24>& q,
                                                                       double thickness) const;

    // The rates of the plate along a rate of its section at fixed geometry and thickness (the AnisoMat2D terms of
    // ShellSectionRate), every part chained exactly:
    //   dK24 = dK_membrane,12 + dK_bending,12,
    // dK_membrane the membrane's stiffness at the rates (ShellQuad4DrillingMembrane is linear in C and G together),
    // dK_bending the hybrid curvatures' and the saturated edge shear's (ShellQuad4DrillingBending::sectionRate).
    struct SectionRate
    {
        base::MatN<24> stiffness;
        std::optional<ShellQuad4DrillingBending::SectionRate> bending;
    };
    // `withStiffness` false leaves `stiffness` zero and forms only what placeStateSectionRates reads.
    [[nodiscard]] SectionRate sectionRate(const ShellSectionRate& rate, bool withStiffness = true) const;
    // The rates of the place states of the local displacements q (placeStates without a temperature change: `states`)
    // at fixed q along the section rate `rate` (`plateRate` its sectionRate): the membrane strains do not move (the
    // strain matrix of ShellQuad4DrillingMembrane is kinematic), the hybrid moments move with d(H⁻¹ Gm), the curvature κ
    // = D⁻¹ M_h with dM_h and dD, the shear forces with the edge-shear coefficients, N = t G1 ε with dG1, M = −M_h with
    // dM_h; no thickness rate.
    [[nodiscard]] std::array<ShellPlaceState, 5> placeStateSectionRates(const std::array<ShellPlaceState, 5>& states,
                                                                        const std::array<double, 24>& q,
                                                                        const ShellSectionRate& rate,
                                                                        const SectionRate& plateRate) const;

    // The state at the centre and at the four corners (in that order) of the local displacements q: membrane strains
    // and moments evaluated at the centre and at the 2 × 2 Gauss points, the corner values extrapolated bilinearly
    // from the Gauss points (STRESS(CORNER) = BILIN); the transverse shear forces are constant over the element.
    // `gridIds` names the corners in the rows; `temperature` the thermal state of the element (zero: none).
    [[nodiscard]] std::array<ShellPlaceState, 5> placeStates(const std::array<double, 24>& q,
                                                             const std::array<std::int32_t, 4>& gridIds,
                                                             const ShellTemperatureChange& temperature = {}) const;

    // placeStates in parts: what does not depend on the displacements nor on the temperatures (PlaceOperators: the
    // strain matrices of the membrane and the moment interpolation at the centre and the Gauss points, the thickness
    // and flexibility D⁻¹ of each place, the Jacobians of a plate without bending, the weights of the corner
    // extrapolation), what depends on the temperatures only (PlaceThermal), formed once each for many displacements
    // (the unit displacements of the recovery slopes); placeStates(operators, thermal, q, ...) gives the bits of
    // placeStates(q, ...) (the same operations in the same order).
    struct PlaceOperators
    {
        std::array<base::MatN<3, 12>, 5> membrane;            // the centre, then the Gauss points
        std::array<base::MatN<3, 5>, 5> moment;               // the moment interpolation
        std::array<double, 5> thickness{};                    // the centre, then the corners
        std::array<base::Mat3, 5> flexibility;                // D⁻¹ at the place's thickness (with bending)
        std::array<PlaneJacobian, 5> jacobian;                // without bending
        std::array<std::array<double, 4>, 4> extrapolation{}; // corner × Gauss point
    };
    struct PlaceThermal
    {
        std::array<double, 5> initialMoments{};
        std::array<ShellTemperatureChange, 5> temperature; // the element's, ΔT at the place
        std::array<ThermalResultants, 5> resultants;
    };
    [[nodiscard]] PlaceOperators placeOperators() const;
    [[nodiscard]] PlaceThermal placeThermal(const PlaceOperators& operators,
                                            const ShellTemperatureChange& temperature) const;
    [[nodiscard]] std::array<ShellPlaceState, 5> placeStates(const PlaceOperators& operators,
                                                             const PlaceThermal& thermal,
                                                             const std::array<double, 24>& q,
                                                             const std::array<std::int32_t, 4>& gridIds) const;

    // The mass on the local DOFs (see ShellQuad4DrillingElement: lumped row-sum translations, or the consistent translations and
    // the rotary inertia ρ t³/12 on θx, θy), `areaMass` the non-structural mass per unit area added to the section's.
    [[nodiscard]] base::MatN<24> mass(MassFormulation formulation, double areaMass) const;
    // mass() and massThicknessRate() of a plate of this geometry, section and corner thicknesses, without building
    // the plate.
    // PERFORMANCE: the mass reads the geometry, the density and the thicknesses only; the plate's constructor builds
    // the membrane and the hybrid bending, some two thirds of a mass facet's call.
    // Measured: the masses' thickness rates of a gradient ladder model (1/128) took 0.067 s with the plate built.
    [[nodiscard]] static base::MatN<24> massOf(const ShellQuad4DrillingGeometry& geometry, const ShellSection& section,
                                               const std::array<double, 4>& thickness, MassFormulation formulation,
                                               double areaMass);
    [[nodiscard]] static base::MatN<24> massThicknessRateOf(const ShellQuad4DrillingGeometry& geometry,
                                                            const ShellSection& section,
                                                            const std::array<double, 4>& thickness,
                                                            MassFormulation formulation, double nominalThickness);
    // d²M24/dT²: the translational mass and the NSM have none, the rotary inertia ρ t³/12 has ρ t³/(2 T²).
    [[nodiscard]] static base::MatN<24> massThicknessSecondRateOf(const ShellQuad4DrillingGeometry& geometry,
                                                                  const ShellSection& section,
                                                                  const std::array<double, 4>& thickness,
                                                                  MassFormulation formulation, double nominalThickness);
    // The equivalent nodal loads of a pressure p(ξ, η) = Σ N_i p_i: F_i = ∫ N_i p dA on the mean plane, along +n, or
    // along the unit vector `direction` of the element frame when one is given (a FacePressureLoad with CID or N1-N3).
    [[nodiscard]] std::array<double, 24> pressureLoads(const std::array<double, 4>& cornerPressures,
                                                       const std::optional<base::Vec3>& direction = {}) const;
    // The equivalent nodal loads of a temperature change (see the class comment).
    [[nodiscard]] std::array<double, 24> thermalLoads(const ShellTemperatureChange& temperature) const;

private:
    [[nodiscard]] double thicknessAt(double xi, double eta) const;
    // The warp offsets (ShellQuad4DrillingWarpOffsets) of a warped element, the identity otherwise: a matrix and loads of the
    // flat plate carried to the grid points, the displacements of the projections.
    [[nodiscard]] base::MatN<24> onGridPoints(base::MatN<24> k) const;
    [[nodiscard]] std::array<double, 24> loadsOnGridPoints(const std::array<double, 24>& f) const;
    [[nodiscard]] std::array<double, 24> onProjections(const std::array<double, 24>& q) const;
    // The mass of `mass()` with the mass per unit area `areal(t)` and the rotary inertia per unit area `inertia(t)` at
    // a point of thickness t.
    template <typename Areal, typename Inertia>
    [[nodiscard]] static base::MatN<24> massWith(const ShellQuad4DrillingGeometry& geometry,
                                                 const std::array<double, 4>& thickness, MassFormulation formulation,
                                                 const Areal& areal, const Inertia& inertia);
    // The thermal resultants at the 2 × 2 Gauss points: N_T (membrane) and κ_T = −D⁻¹ M_T (bending).
    struct ThermalAtGaussPoints
    {
        std::array<std::array<double, 3>, 4> force{};
        std::array<std::array<double, 3>, 4> curvature{};
    };
    [[nodiscard]] ThermalAtGaussPoints thermalAtGaussPoints(const ShellTemperatureChange& temperature) const;

    ShellQuad4DrillingGeometry m_geometry; // the plate is formed on its plane coordinates, the heights in m_offsets
    ShellSection m_section;
    std::array<double, 4> m_thickness{};
    std::optional<ShellQuad4DrillingWarpOffsets> m_offsets; // a warped element
    std::optional<ShellQuad4DrillingMembrane> m_membrane;
    std::optional<ShellQuad4DrillingBending> m_bending;
};

} // namespace oos::elements
