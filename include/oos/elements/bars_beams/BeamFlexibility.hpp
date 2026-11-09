#pragma once

#include "oos/base/math/MatN.hpp"
#include "oos/elements/bars_beams/BeamLineLoad.hpp"
#include "oos/elements/bars_beams/BeamSection.hpp"
#include "oos/elements/core/IsotropicMaterial.hpp"

#include <array>
#include <span>

namespace oos::elements
{

// The stiffness and the fixed-end loads of a bar or beam, by the flexibility method (Euler-Bernoulli bending with
// transverse shear flexibility, the stiffness of a tapered beam obtained by inverting its integrated flexibility;
// Przemieniecki, Theory of Matrix Structural Analysis, §5.6 and §11):
//
// - The element is a cantilever clamped at end A and loaded at end B by P = [N, Vy, Vz, T, My, Mz] (element system).
//   At a section x the stress resultants are s(x) = S(x)·P: N, Vy, Vz, T unchanged, My(x) = My − (L − x)·Vz,
//   Mz(x) = Mz + (L − x)·Vy (the moments of the forces at B about the section).
// - The complementary energy U* = ½∫ sᵀ·c(x)·s dx with the compliance c: 1/EA, 1/(K1·A·G), 1/(K2·A·G), 1/GJ, and for
//   the moments the inverse of the moment-curvature matrix [My; Mz] = E·[[I2, −I12], [−I12, I1]]·[χy; χz]
//   (χy = θy', χz = θz'; σ = E·(−y·χz + z·χy)). The resultants are those of the element axis (the shear centres of a
//   Beam2); with the neutral axis at (y_n, z_n) from it, the axial and bending terms are those of the neutral axis,
//   c = Eᵀ·c_n·E with My_n = My − z_n·N, Mz_n = Mz + y_n·N.
// - The flexibility F = ∫ Sᵀ·c·S dx gives the end-B deformation relative to the rigid motion of end A, q = F·P with
//   q = B·d: Δu = u_B − u_A, Δv = v_B − v_A − L·θz_A, Δw = w_B − w_A + L·θy_A, Δθ = θ_B − θ_A (d the end DOFs). The
//   natural stiffness is k = F⁻¹ and K = Bᵀ·k·B (12 × 12, end DOFs). For a uniform section this is the familiar
//   matrix with the shear relief Φ = 12·E·I/(K·A·G·L²) per plane (e.g. 12EI/(L³(1 + Φ)) for the transverse terms).
// - A load on the span: q0 = ∫ Sᵀ·c·s₀ dx, s₀(x) the resultants of the loads on [x, L] about x (B free). Clamping B
//   needs P_B = −k·q0, so the equivalent nodal loads (the fixed-end reactions with their sign changed) are
//   f = Bᵀ·k·q0 + [F₀; M₀; 0; 0], (F₀, M₀) the total load and its moment about end A.
//
// The integrals are exact (8-point Gauss-Legendre on polynomials of degree ≤ 4) for a uniform section and polynomial
// loads; a tapered segment is split into BEAM_TAPER_SUBINTERVALS parts, each integrated by the 8-point rule (relative
// error below 1e-12 for a ratio of 8 between the inertias at the two ends of a segment). Between stations the section
// properties vary linearly (a modelling choice of this project for a tapered Beam2).
//
// A component whose stiffness is zero along the whole element (A, J, I1 or I2 zero, or E, G zero) is left out of F,
// and k is zero there: the element carries no load in it. A shear factor ≤ 0, or a zero K·A·G, means no shear
// flexibility (PBAR: 0.0 or blank for K1, K2 is an infinite shear stiffness; extended to A or G zero).
//
// Thread-safety: immutable. Precondition: findSectionProblem(section) found nothing; length > 0.
class BeamFlexibility
{
public:
    BeamFlexibility(const BeamSection& section, const IsotropicMaterial& material, double length);

    // K = Bᵀ·k·B over the end DOFs [u_A θ_A u_B θ_B] in the element system, without pin flags.
    [[nodiscard]] base::MatN<12> stiffness() const;
    // f, the equivalent nodal loads of the loads over the end DOFs, without pin flags.
    [[nodiscard]] std::array<double, 12> equivalentLoads(std::span<const BeamLineLoad> loads) const;

    [[nodiscard]] const base::MatN<6>& flexibility() const noexcept { return m_flexibility; }
    [[nodiscard]] const base::MatN<6>& naturalStiffness() const noexcept { return m_naturalStiffness; }

private:
    // c(x) at a distance x from end A.
    [[nodiscard]] base::MatN<6> complianceAt(double x) const;
    // Calls visit(x, weight) for the Gauss points of [0, L], the segments split at the stations and at `breaks`.
    template <typename Visit>
    void integrate(std::span<const double> breaks, Visit&& visit) const;

    const BeamSection* m_section;
    IsotropicMaterial m_material;
    double m_length;
    std::array<bool, 6> m_isActive{};
    base::MatN<6> m_flexibility;
    base::MatN<6> m_naturalStiffness;
};

// A tapered segment between two stations is integrated on this many equal parts (see BeamFlexibility).
inline constexpr int BEAM_TAPER_SUBINTERVALS = 8;

// q = B·d, the end-B deformation of a cantilever clamped at A (see BeamFlexibility), for an element of length L.
[[nodiscard]] base::MatN<6, 12> cantileverDeformation(double length);

} // namespace oos::elements
