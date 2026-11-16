#pragma once

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/line/LineKinematics.hpp"
#include "oos/elements/mixedHybrid/line/LineSection.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace oos::elements::mixedHybrid
{

// The mixed hybrid kernel of a line element (a rod, a bar or a beam; document 34 of the reference crate, section 8.1):
// its stiffness, mass, loads and state from its kinematics at the points of its rule.
//
// The rotation θ of a line is continuous inside an edge only. The line is written in the mixed form of its moments M,
// a vector polynomial of degree k on its edge (components along the section axes at the middle of the line, Legendre
// polynomials P_l(2λ − 1) along it), eliminated on the edge; its rotation Λ at each of its grids is a hybrid unknown
// shared by the lines that meet there:
//   b(M; θ, Λ) = −∫ dM/ds · θ ds + M(B) · Λ_B − M(A) · Λ_A.
// With moments of degree k, dM/ds tests the whole rotation and the line has no mode of zero energy but the rigid
// ones. The shear force of the equilibrium of the line, V = t × dM/ds, carries the shear flexibility of the section
// by its complementary energy ½ ∫ V · C_s V ds: the beam with shear, its rotation across the line eliminated. The
// axial strain t · du/ds stays primal. An offset o of the axis (the end offsets, the eccentricity, the neutral axis)
// couples it with the curvatures through the section law, Γ_axis = Γ_edge + (K × o) · x, treated in the partially
// mixed form of the shells (document 34, section 3.2): with the law on (Γ_x, K) split as [[A, B], [Bᵀ, D]], the
// membrane block is Ā = A − B D⁻¹ Bᵀ and the coupling Mᵀ D⁻¹ Bᵀ Γ_x. An axis away from a curved edge is longer than
// the edge: its strains are those seen from the edge over μ = |t + do/ds| and its length μ ds, so that the law on
// the strains of the edge is TᵀDT / μ, and the mass is μ times that of the edge. Eliminating M,
//   K = K_axial + (G + C)ᵀ M_σ⁻¹ (G + C) = K_axial + Yᵀ Y,   Y = L⁻¹ (G + C),   M_σ = L Lᵀ,
// M_σ the compliance of the moments, G the pairing b, C the coupling. This is the Hellinger-Reissner principle of a
// beam (E. Reissner, On a variational theorem in elasticity, Journal of Mathematics and Physics 29 (1950) 90-95) made
// hybrid at the grids (T. H. H. Pian, Derivation of element stiffness matrices by assumed stress distributions, AIAA
// Journal 2 (1964) 1333-1336).
//
// Local unknowns of a line (n_l = n_xk + 6): the n_xk columns of its kinematics, then the rotations Λ (global) at its
// grid A and at its grid B. For a line alone the n_xk columns are those of its AloneEdge (LineKinematics.hpp gives
// the full ordering); for a line on a shell edge, the unknowns of the shell element.
//
// Moment coefficients: row 3 l + c is the component c (along basis[c]) of degree l (P_l(2λ − 1)), l ≤ degree.
//
// Thread-safety: plain value types; the functions are pure. Ownership: LineData owns its matrices.

// The terms of the axial force at one point of the rule of a line.
// Reference: src/tdnns/line.rs::LineData::axial (one tuple)
struct LineAxialPoint
{
    // The row of the axial strain t · du/ds on the local unknowns (n_l).
    Vector strainRow;
    // Ā = A − B D⁻¹ Bᵀ: the axial stiffness that stays primal.
    double axialStiffness = 0.0;
    // The rows of du/ds across the line, 3 × n_l (the geometric stiffness).
    Matrix transverseRate;
    // ds: the length of the edge this point stands for.
    double weight = 0.0;
    // The moment part of the axial force, (D⁻¹ Bᵀ)ᵀ M_s, on the moment coefficients.
    Vector momentRow;
};

// A line element: its stiffness on the unknowns of its kinematics (with their signs) and the rotations of its two
// grids.
// Reference: src/tdnns/line.rs::LineData
struct LineData
{
    LineSource source;
    // The numbers of its n_l unknowns in the caller's numbering (the kinematics, then the three rotations at A and the
    // three at B), and the sign of each (+1 or −1: the orientation of a shell's edge unknown).
    std::vector<std::size_t> dofs;
    std::vector<double> signs;
    // n_l × n_l, symmetric.
    Matrix stiffness;
    // n_l × n_l, symmetric; when asked.
    std::optional<Matrix> mass;
    // One per point of the rule.
    std::vector<LineAxialPoint> axial;
    // The moment coefficients from the local unknowns, M_σ⁻¹ (G + C) (3 (degree + 1) × n_l; rows of the components
    // without stiffness zero), and M_σ⁻¹ (3 (degree + 1) square, likewise).
    Matrix moments;
    Matrix complianceInverse;
    // The vectors of the moment components (the section axes at the middle of the line) and the degree of the moments.
    SectionAxes basis{};
    std::size_t momentDegree = 0;
    // The shear flexibility is in the moments.
    bool shear = false;
    // ∫ (u + θ × o)ᵀ m μ ds: the integral of the displacement of the axis times the mass per length (n_l × 3); the
    // load of a uniform acceleration a is inertia · a.
    Matrix inertia;
    // The points of the axis at the points of the rule (where the loads are read).
    std::vector<Vec3Of<double>> axisPoints;

    // The local unknowns from the model's: x_i = sign_i u[dof_i].
    // Reference: LineData::local
    [[nodiscard]] Vector local(std::span<const double> modelUnknowns) const;

    // The moment term of a free axial strain e0 given at the points of the rule: g0 = Σ w e0 (D⁻¹ Bᵀ)ᵀ M_s.
    // Reference: LineData::thermal_moments
    [[nodiscard]] Vector thermalMoments(std::span<const double> freeStrains) const;

    // The load of a free axial strain e0 given at the points of the rule (a temperature rise): the primal part
    // Ā e0 on the axial strain, and the moment part.
    // Reference: LineData::thermal_load
    [[nodiscard]] Vector thermalLoad(std::span<const double> freeStrains) const;
};

// Builds a line element from its kinematics at the points of its rule (`points`: kinematics and ds),
// `kinematicsColumns` = n_xk columns, moments of degree `momentDegree`, shear flexibility `shear`; `kinematicsDofs`,
// `kinematicsSigns`: the unknowns of the kinematics; `rotationDofs`: the first of the three rotation unknowns of
// grids A and B (in the reference numbering, 6 a + 3 and 6 b + 3). Returns the message of a problem instead: the
// section axes (sectionAxes), a singular compliance of the moments.
// Reference: src/tdnns/line.rs::line_core
[[nodiscard]] std::variant<LineData, std::string> lineCore(
    const LineSource& source, std::span<const WeightedLineKinematics> points, std::size_t kinematicsColumns,
    std::size_t momentDegree, bool shear, const LineElement& line, const LineSection& section,
    std::span<const std::size_t> kinematicsDofs, std::span<const double> kinematicsSigns,
    const std::array<std::size_t, 2>& rotationDofs, bool withMass);

// The state of a line element at one point.
// Reference: src/analysis.rs::LineState
struct LineState
{
    // The point of the reference axis, and the section axes (x along the line from A to B, y, z).
    Vec3Of<double> point{};
    SectionAxes axes{};
    // (N, V_y, V_z, T, M_y, M_z): axial force, shear forces, torque, bending moments about y and z through the
    // reference axis. Conjugate to the strains.
    std::array<double, 6> forces{};
    // (Γ_x, Γ_y, Γ_z, K_x, K_y, K_z) at the reference axis.
    std::array<double, 6> strains{};
    // The temperature rise of the load case at the point, and the axial strain of its free expansion.
    double temperature = 0.0;
    double freeStrain = 0.0;
};

// The state of a line at the point of kinematics `kinematics`: the forces of the section about its axis and the
// strains at the axis; `x` the local unknowns of the element, `freeStrain` the free axial strain e0 at the point
// (`temperature` the temperature rise), `freeMoments` the moment term of the free strain along the line
// (LineData::thermalMoments; empty without one). Returns the message of a problem of the section axes instead.
// Reference: src/tdnns/line.rs::line_state
[[nodiscard]] std::variant<LineState, std::string> lineState(const LineData& data, const LineKinematics& kinematics,
                                                             const LineElement& line, const LineSection& section,
                                                             std::span<const double> x, double temperature,
                                                             double freeStrain, std::span<const double> freeMoments);

} // namespace oos::elements::mixedHybrid
