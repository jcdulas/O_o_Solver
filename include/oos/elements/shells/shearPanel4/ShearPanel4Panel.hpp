#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"

#include <array>
#include <cstdint>

namespace oos::elements
{

// A shear panel whose height off its mean plane exceeds this fraction of its mean diagonal is warped: its kick forces
// are not formed, and such a panel is refused.
inline constexpr double SHEAR_PANEL_WARP_TOLERANCE = 1.0e-8;
// A corner of the projected panel whose cross product of its two edges is at or below this fraction of the panel area
// is folded or collapsed (a valid ShearPanel4 has all its interior angles below 180°).
inline constexpr double SHEAR_PANEL_MIN_CORNER_RATIO = 1.0e-10;
// ShearPanelProp F1, F2: F at or below this is a fraction of the panel, above it a width in thicknesses.
inline constexpr double SHEAR_PANEL_EFFECTIVENESS_LIMIT = 1.01;

// The section of a shear panel (PSHEAR: T, NSM, F1, F2; the IsoMat of MID).
struct ShearPanelSection
{
    double thickness = 0.0;
    double shearModulus = 0.0;      // G
    double youngModulus = 0.0;      // E, for the extensional rods of F1, F2
    double density = 0.0;           // RHO
    double nonStructuralMass = 0.0; // NSM per unit area
    double f1 = 0.0;                // effectiveness for extension along edges 1-2 and 3-4
    double f2 = 0.0;                // along edges 2-3 and 1-4
};

// Why a shear panel has no formulation.
enum class ShearPanelProblem : std::uint8_t
{
    None,
    Degenerate, // parallel or null diagonals
    Folded,     // a corner of 180° or more, or two corners at one place
    Warped      // the four grid points off one plane
};

// One shear panel (CSHEAR, ShearPanelProp; the panel of Garvey, The quadrilateral shear panel, Aircraft Engineering
// 1951), formulated in its element frame: x along the projection of G1→G2 on the mean plane, n along d13 × d24 (the
// diagonals), y = n × x.
//
// - Shear flows (statically determinate, Garvey): the panel carries tangential shear flows q1 ... q4 on its edges
//   1-2, 2-3, 3-4, 4-1, whose edge forces F_e = s_e q_e L_e u_e (s = −1, +1, −1, +1: a positive shear acts along
//   −u on edges 1-2 and 3-4, along +u on 2-3 and 4-1, u_e the unit edge vector) are in equilibrium in the plane:
//   Σ F_e = 0 and Σ (p_e × F_e)·n = 0. These three equations leave one pattern q_e = c_e q̄, c the null vector of the
//   3 × 4 system normalised to a mean of 1 (c = 1 on a parallelogram, the uniform pure shear).
// - Corner forces: each edge force acts half at each end of its edge (along the edge), the vector b (12) of the
//   corner forces per unit q̄ on (u, v, w) of the four corners, local frame.
// - Stiffness: the generalised displacement Δ = bᵀ u is conjugate to q̄; with the strain energy of the mean flow
//   U = q̄² A / (2 G t) (A the panel area ½ |d13 × d24|), q̄ = Δ / f, f = A / (G t), and K = b bᵀ / f. On a
//   parallelogram the state is the exact uniform shear τ = G γ.
// - Extensional rods (PSHEAR F1, F2): on edges 1-2 and 3-4 of area F1 T A / (L12 + L34) when F1 ≤ 1.01, else
//   0.5 F1 T², and likewise with F2 on edges 2-3 and 1-4; axial stiffness E A_rod / L each (no Poisson coupling).
//
// Design choices of this project: the energy of the mean flow on a panel that is not a parallelogram (Garvey's flows
// vary along the panel) and the even split of the edge forces at the corners.
//
// Thread-safety: immutable once built. A plain value.
class ShearPanel4Panel
{
public:
    // The panel of the corners `positions` (basic system), or the problem that keeps it from being formed.
    struct BuildResult;
    [[nodiscard]] static BuildResult build(const std::array<base::Vec3, 4>& positions,
                                           const ShearPanelSection& section);

    [[nodiscard]] const base::Mat3& axes() const noexcept { return m_axes; } // columns x, y, n in basic
    [[nodiscard]] double area() const noexcept { return m_area; }
    [[nodiscard]] const std::array<double, 4>& edgeLengths() const noexcept { return m_lengths; }
    // The shear flow pattern c (mean 1).
    [[nodiscard]] const std::array<double, 4>& flowPattern() const noexcept { return m_pattern; }

    // K (12 × 12) on (u, v, w) of the four corners in the element frame: the shear panel and its rods.
    [[nodiscard]] base::MatN<12> stiffness() const;
    // The mean shear flow q̄ of the local displacements q (12).
    [[nodiscard]] double meanFlow(const std::array<double, 12>& q) const;
    // The differential stiffness (12 × 12, local translations) of the mean shear flow q̄ of the preload: the membrane
    // initial-stress stiffness of ShellDifferential with Nx = Ny = 0, Nxy = q̄, on the bilinear functions of the
    // projected corners (2 × 2 Gauss). A simplification of this project: the varying flows of a panel that is not a
    // parallelogram and the axial forces of the rods of F1, F2 are left out.
    [[nodiscard]] base::MatN<12> differentialStiffness(double meanFlow) const;
    // The mass (lumped: a quarter of (ρ t + NSM) A on the translations of each corner), `areaMass` added to NSM.
    [[nodiscard]] base::MatN<12> mass(double areaMass) const;

private:
    base::Mat3 m_axes = base::Mat3::identity();
    std::array<double, 4> m_x{};
    std::array<double, 4> m_y{};
    std::array<double, 4> m_lengths{};
    double m_area = 0.0;
    std::array<double, 4> m_pattern{};
    std::array<double, 12> m_cornerForces{}; // b
    double m_flexibility = 0.0;              // f
    ShearPanelSection m_section;
};

struct ShearPanel4Panel::BuildResult
{
    ShearPanel4Panel panel;
    ShearPanelProblem problem = ShearPanelProblem::None;
};

} // namespace oos::elements
