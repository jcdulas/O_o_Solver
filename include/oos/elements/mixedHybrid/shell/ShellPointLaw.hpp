#pragma once

#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSection.hpp"
#include "oos/elements/mixedHybrid/shell/ShellTypes.hpp"

#include <array>
#include <optional>
#include <utility>

namespace oos::elements::mixedHybrid
{

// The section law of a shell element of the mixed hybrid formulation at a point: the section moved to the surface of
// the grid points (offset), the matrices of the partially mixed form, the free thermal strains, and the maps between
// covariant, contravariant and local components (document 34 of the reference, sections 2.3 and 3.2).
//
// Components: covariant strains (11, 22, 12) are ε_αβ in the reference coordinates; local Voigt strains (11, 22,
// 2 × 12) are in the local frame (e1, e2) with engineering shear; local moments (m11, m22, m12) are plain components.
//
// Offset (document 34 of the reference, section 2.3): the unknowns stay on the surface of the grid points and the
// section, at z0 along the normal, acts on the strains e + z0 κ of the offset surface, read with its own basis
// a_α + z0 ∂_α ν, over its area μ0 = j(z0) / j. With P the map from the local strains of the surface of the grid points
// to those of the offset surface:
//   A' = μ0 Pᵀ A P,  B' = μ0 Pᵀ (B + z0 A) P,  D' = μ0 Pᵀ (D + 2 z0 B + z0² A) P,
// the same for the thermal coefficients and μ0 on the inertia. On a flat element P = I and μ0 = 1.
//
// Reference: tdnns/element.rs (section_at, section_geo, offset_area, offset_map, Laws, laws, free_strains,
// cov_to_voigt, voigt_to_cov, con_to_local).
//
// Thread-safety: pure functions.

// The section at a point, moved to the surface of the grid points by the shift of an offset z0:
// B' = B + z0 A, D' = D + 2 z0 B + z0² A, the same for the thermal coefficients and the inertia.
// Reference: tdnns/element.rs::section_at
[[nodiscard]] SectionPoint sectionAt(const ShellSection& section, double offset, double xi, double eta,
                                     const SurfaceFrame& frame);

// The section at a point of an element whose section lies at z0 from the surface of the grid points, moved to that
// surface with the metric of the offset surface (the formulas of the header comment).
// Reference: tdnns/element.rs::section_geo
[[nodiscard]] SectionPoint sectionAtGeometry(const ShellSection& section, double offset, double xi, double eta,
                                             const PointGeometry& g);

// μ0: the area of the offset surface at z0 over that of the surface of the grid points.
// Reference: tdnns/element.rs::offset_area
[[nodiscard]] double offsetArea(const PointGeometry& g, double offset);

// P: the map from the local Voigt strains of the surface of the grid points to those of the surface at z0, for the
// same covariant components. Throws InternalError when the local frame is degenerate.
// Reference: tdnns/element.rs::offset_map
[[nodiscard]] Matrix33 offsetMap(const PointGeometry& g, double offset);

// The matrices of the section used by the mixed form: Ā = A − B D⁻¹ Bᵀ, D⁻¹, D⁻¹ Bᵀ, B and H.
// Reference: tdnns/element.rs::Laws
struct SectionLaws
{
    Matrix33 abar{};
    Matrix33 dinv{};
    Matrix33 dinvBt{};
    Matrix33 b{};
    Matrix22 h{};
};

// Throws InternalError when D is singular (a section without bending stiffness, which the element family refuses
// before building an element).
// Reference: tdnns/element.rs::laws
[[nodiscard]] SectionLaws sectionLaws(const SectionPoint& point);

// The free strains (e0, κ0) of the section for the temperatures (T0, T1), local Voigt with engineering shear:
// (e0, κ0') = [A B; Bᵀ D]⁻¹ (N_T, M_T), and κ0 = κ0' + the part of the free expansion on a curved surface,
// (e0 · K + (e0 · K)ᵀ) in tensor components, K the curvature (a uniform rise of an unrestrained curved shell is a
// homothety without stress; document 34 of the reference, section 2.3). Throws InternalError when the section is
// singular.
// Reference: tdnns/element.rs::free_strains
[[nodiscard]] std::pair<Voigt3, Voigt3> freeStrains(const SectionPoint& point, const Matrix22& curvature,
                                                    const std::array<double, 2>& temperatures);

// T: covariant strain (11, 22, 12) to local Voigt (11, 22, 2 × 12), from c[α][I] = e_I · a^α:
//   ε_IJ = Σ_αβ c[α][I] c[β][J] ε_αβ.
// Reference: tdnns/element.rs::cov_to_voigt
[[nodiscard]] Matrix33 covariantToVoigt(const Matrix22& c);

// Local Voigt strain (11, 22, 2 × 12) to covariant (11, 22, 12), from d[α][I] = e_I · a_α: ε_αβ = d[α][I] d[β][J] ε_IJ.
// Reference: tdnns/element.rs::voigt_to_cov
[[nodiscard]] Voigt3 voigtToCovariant(const Matrix22& d, const Voigt3& v);

// Reference contravariant (S^11, S^22, S^12) to local moments (m11, m22, m12): the double contravariant Piola map
// m^IJ = d[α][I] d[β][J] S^αβ / j², from d[α][I] = e_I · a_α.
// Reference: tdnns/element.rs::con_to_local
[[nodiscard]] Matrix33 contravariantToLocal(const Matrix22& d, double j);

// ---- small fixed-size algebra used by the shell kernel ----------------------------------------------------------

// The inverse of a 3 × 3 matrix by its adjugate; empty when the determinant is zero.
[[nodiscard]] std::optional<Matrix33> inverse33(const Matrix33& m);
// m · v.
[[nodiscard]] Voigt3 times33(const Matrix33& m, const Voigt3& v);
// mᵀ · v.
[[nodiscard]] Voigt3 transposeTimes33(const Matrix33& m, const Voigt3& v);
// a · b.
[[nodiscard]] Matrix33 product33(const Matrix33& a, const Matrix33& b);

} // namespace oos::elements::mixedHybrid
