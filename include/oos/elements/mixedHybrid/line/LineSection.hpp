#pragma once

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"

#include <array>
#include <cstddef>
#include <optional>

namespace oos::elements::mixedHybrid
{

// The data of a line element of the mixed hybrid formulation (a rod, a bar or a beam on an edge): its section, its
// grids, orientation and offsets, and the law of its section at one point. These are the plain inputs of the line
// kernel (LineCore.hpp); the adapters from the solver's cards are written by the element families.
//
// Section axes: x along the line from grid A to grid B, y the first section axis (the part of the orientation vector
// across the line, or of the normal of the shell along the edge), z = x × y. The reference axis of the element is
// the shear centre; the neutral axis may be away from it.
//
// Strains and forces of a section, in that order: (Γ_x, Γ_y, Γ_z, K_x, K_y, K_z), the axial strain, the two shear
// strains, the twist and the two curvatures; (N, V_y, V_z, T, M_y, M_z), conjugate to them, the moments about the
// reference axis. With (y_n, z_n) the neutral axis, the strain energy per length is (document 27 of the reference
// crate, section 3)
//   W = [E A ε_n² + E I1 K_z² + E I2 K_y² − 2 E I12 K_y K_z + G J K_x² + k1 G A Γ_y² + k2 G A Γ_z²] / 2,
//   ε_n = Γ_x + z_n K_y − y_n K_z,
// the elementary theory of the Timoshenko beam (S. P. Timoshenko, J. M. Gere, Mechanics of Materials, 1972; the shear
// factors k1, k2 of the section) with Saint-Venant torsion.
//
// Thread-safety: plain value types; the functions are pure.

// The shear stiffness of a section given without shear flexibility, as a multiple of E A (the convention of the
// shell sections): large enough that the shear strain is negligible (results to 1e-8), finite so that the
// complementary energy of the shear force stays defined.
inline constexpr double RIGID_SHEAR_STIFFNESS_FACTOR = 1.0e4;

// The law of a section at one point of a line.
// Reference: src/line.rs::SectionPoint
struct LineSectionPoint
{
    // Stiffness on (Γ_x, Γ_y, Γ_z, K_x, K_y, K_z) at the reference axis (reference field: d).
    std::array<std::array<double, 6>, 6> stiffness{};
    std::array<double, 3> centroidalBending{}; // Dyy, Dzz, Dyz about the neutral axis.
    // The neutral axis from the reference axis, components (y, z).
    std::array<double, 2> neutralAxis{};
    // Mass per unit length (with the non-structural mass).
    double massPerLength = 0.0;
    double structuralMassPerLength = 0.0;
    double nonStructuralMassPerLength = 0.0;
    std::array<double, 2> massCentre{};
    // Rotary inertias per unit length: about x, about y, about z, and the product of y and z.
    std::array<double, 4> inertia{};
};

// The section of a line element, linear between its ends A and B.
// Reference: src/line.rs::LineSection
struct LineSection
{
    double youngModulus = 0.0;     // e
    double shearModulus = 0.0;     // g
    double density = 0.0;          // rho
    double thermalExpansion = 0.0; // alpha
    // At the two ends A and B, linear between them: the area; I1, the second moment for bending in the plane (x, y);
    // I2, in the plane (x, z); I12, the product of y and z; J, the torsion constant.
    std::array<double, 2> area{};
    std::array<double, 2> i1{};
    std::array<double, 2> i2{};
    std::array<double, 2> i12{};
    std::array<double, 2> torsionConstant{};
    // The shear factors k1, k2 of the planes (x, y) and (x, z): shear stiffnesses k G A. Empty: no shear flexibility
    // (RIGID_SHEAR_STIFFNESS_FACTOR · E A).
    std::optional<std::array<double, 2>> shearFactors;
    // Mass per unit length that is not that of the material.
    double nonStructuralMass = 0.0;
    std::array<double, 2> stationNonStructuralMass{};
    std::array<std::array<double, 2>, 2> massCentre{};
    // The neutral axis from the reference axis, components (y, z), at A and at B.
    std::array<std::array<double, 2>, 2> neutralAxis{};
    // The section bends and shears. False: a rod, axial force and torque only.
    bool bending = true;

    // A rod: axial force and torque only.
    // Reference: src/line.rs::LineSection::rod
    [[nodiscard]] static LineSection rod(double e, double g, double area, double j, double rho);

    // A bar: uniform section, neutral axis on the reference axis, no shear flexibility.
    // Reference: src/line.rs::LineSection::bar
    [[nodiscard]] static LineSection bar(double e, double g, double area, double i1, double i2, double i12, double j,
                                         double rho);

    // A beam: `ends[end] = {area, I1, I2, I12, J}` at A and B, the shear factors, the neutral axis from the shear
    // centre at A and B.
    // Reference: src/line.rs::LineSection::beam
    [[nodiscard]] static LineSection beam(double e, double g, double rho,
                                          const std::array<std::array<double, 5>, 2>& ends,
                                          const std::optional<std::array<double, 2>>& shearFactors,
                                          const std::array<std::array<double, 2>, 2>& neutralAxis);

    // The law of the section at the fraction `fraction` of the chord from A (every property linear between the ends).
    // Reference: src/line.rs::LineSection::at
    [[nodiscard]] LineSectionPoint at(double fraction) const;
};

// A line element between two grids: a rod, a bar or a beam by its section.
// Reference: src/line.rs::LineElement
struct LineElement
{
    // Grids A and B (reference field: nodes), in the caller's numbering.
    std::array<std::size_t, 2> grids{};
    // The index of its section in the caller's list (reference field: prop).
    std::size_t section = 0;
    // The orientation vector: the first section axis y is its part across the line. Empty: the normal of the shell
    // along the edge (the web of a stiffener stands on the shell); required for a bar or a beam on an edge that no
    // shell has (reference field: v).
    std::optional<Vec3Of<double>> orientation;
    // Offsets of the reference axis from the grids A and B, global components, linear between them.
    std::array<Vec3Of<double>, 2> offsets{};
    // Offset of the reference axis along the section axes (y, z), added to `offsets`: without an orientation vector,
    // a distance along the normal of the shell, whatever its curvature (reference field: ecc).
    std::array<double, 2> eccentricity{};

    // The element between grids a and b with section `sectionIndex`, no orientation vector, no offset.
    // Reference: src/line.rs::LineElement::new
    [[nodiscard]] static LineElement between(std::size_t a, std::size_t b, std::size_t sectionIndex);
    // Reference: src/line.rs::LineElement::oriented
    [[nodiscard]] LineElement oriented(const Vec3Of<double>& v) const;
    // Reference: src/line.rs::LineElement::eccentric
    [[nodiscard]] LineElement eccentric(double y, double z) const;
    // Reference: src/line.rs::LineElement::offsets
    [[nodiscard]] LineElement withOffsets(const Vec3Of<double>& atA, const Vec3Of<double>& atB) const;
};

// The forces (N, V_y, V_z, T, M_y, M_z) of the section law `point` for the strains (Γ_x, Γ_y, Γ_z, K_x, K_y, K_z)
// and the free strain e0 of the material along `freeDirection` (the components of the strain Γ of a unit free
// expansion, (1, 0, 0) without an offset):
//   F = D (ε − e0 (f, 0)).
// Reference: src/line.rs::line_forces, its section law part (the strains of the trace of the other shell
// formulation of the reference crate are not ported: the strains come from the caller).
[[nodiscard]] std::array<double, 6> lineForces(const LineSectionPoint& point, const std::array<double, 6>& strains,
                                               double freeStrain, const Vec3Of<double>& freeDirection);

} // namespace oos::elements::mixedHybrid
