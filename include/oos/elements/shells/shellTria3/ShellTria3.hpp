#pragma once

// The ShellTria3 shell of NASTRAN-95 (the program of NASA, open-sourced by NASA in 2015 under the NASA Open Source
// Agreement): its stiffness, mass, loads, centre recovery and a membrane-stress differential stiffness, implemented
// from the mathematical description of the element (routines tria3d, t3setd, t3bmgd, t3gemd, shgmgd, shcsgd, t3bgbs,
// t3pl4d, tltr3d, stri31/32), not translated from its source. "Spec §n" below numbers the parts of that
// description; each is the work of these NASTRAN-95 routines, the public source to check it against:
//   §1     materials and their rotation to the element frame        premat, shgmgd, shcsgd
//   §3.2   user and element frames (median of the shortest edge)   t3setd, betrns
//   §3.5   shape functions and B rows                              t3bmgd
//   §3.6   constant shear and extra parameter of the edge strains  t3gemd, tria3d
//   §3.7   constitutive matrix and shear compliance Z              tria3d, t3bgbs
//   §3.8   mass and transformation                                 tria3d, tldrs
//   §5.2   FacePressureLoad;  §6.3 thermal loads;  §7.3 recovery              t3pl4d;  tltr3d;  stri31, stri32, shstss,
//   shfors
//
// The element in a few words:
// - linear membrane and bending (constant strains and curvatures on the three nodes), integrated at the three edge
//   midpoints, the thickness interpolated linearly;
// - transverse shear: a constant shear field and one extra parameter fitted to the three tangential edge strains, its
//   stiffness the inverse of a compliance Z adding the transverse shear compliance (MID3) and a "bending relief"
//   compliance built from the bending stiffness and the element lengths along its axes;
// - no drilling stiffness (θz never enters), no rotary inertia.
//
// Frames: the element frame E (x along the median from the vertex opposite the shortest edge, z the normal, origin the
// centroid) holds every matrix; the user frame (x along G1→G2) is the reference of THETA and of the output.
//
// Thread-safety: stateless functions on plain values.

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/shells/PlateConstitutive.hpp"
#include "oos/elements/shells/ShellCentreRecovery.hpp"
#include "oos/elements/shells/ShellScalar.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace oos::elements
{

// The geometry of one ShellTria3 (spec §3.2, §3.5, §3.6), from the positions of its grid points only.
struct ShellTria3Geometry
{
    base::Mat3 axes = base::Mat3::identity(); // columns x, y, z of the element frame, in basic
    base::Vec3 centre;                        // the centroid, in basic
    std::array<double, 3> x{};                // the plane coordinates of the nodes from the centroid
    std::array<double, 3> y{};
    std::array<double, 3> dX{};               // ∂N_a/∂x, ∂N_a/∂y of the linear shapes (constant)
    std::array<double, 3> dY{};
    double determinant = 0.0;                 // 2 × the area
    double lengthX = 0.0;                     // L_x: the length of the median along x
    double lengthY = 0.0;                     // L_y = 2 × area / L_x
    // Edge p runs 1→2, 2→3, 3→1: its direction cosines (c_p, s_p) in the element frame.
    std::array<double, 3> edgeCosine{};
    std::array<double, 3> edgeSine{};
    // The inverse of the edge fit: [γxz⁰, γyz⁰, q] = edgeFit · [γ_t(1), γ_t(2), γ_t(3)] (spec §3.6).
    Matrix3<double> edgeFit;
    double userAngle = 0.0; // the user x axis (G1→G2) from the element x axis, radians
};

struct ShellTria3GeometryResult
{
    ShellTria3Geometry geometry;
    std::string problemField; // the field to name when the geometry is refused; empty when accepted
    std::string problem;
};

// The geometry from the positions of G1 to G3 in the basic system; refused: coincident or collinear grid points.
[[nodiscard]] ShellTria3GeometryResult buildShellTria3Geometry(const std::array<base::Vec3, 3>& positions);

// The ShellTria3 kernel, with the interface of ShellQuad4.
struct ShellTria3
{
    static constexpr std::size_t NODES = 3;
    static constexpr std::size_t DOFS = 18;
    static constexpr std::string_view CARD = "ShellTria3";
    using Geometry = ShellTria3Geometry;

    [[nodiscard]] static ShellTria3GeometryResult buildGeometry(const std::array<base::Vec3, NODES>& positions)
    {
        return buildShellTria3Geometry(positions);
    }

    // The linear triangle has no integration problem once its geometry is accepted.
    [[nodiscard]] static std::optional<std::string> integrationProblem(const Geometry& /*geometry*/,
                                                                       const std::array<double, NODES>& /*thickness*/)
    {
        return std::nullopt;
    }

    // K (18 × 18) in the element frame (spec §3.5 to §3.7).
    template <typename T>
    [[nodiscard]] static ScalarMatrix<T, DOFS> stiffness(const Geometry& geometry,
                                                         const PlateConstitutive<T>& constitutive,
                                                         const std::array<T, NODES>& thickness);

    // The translational mass (spec §3.8): ρ t + NSM per unit area at the edge midpoints, lumped or coupled.
    template <typename T>
    [[nodiscard]] static ScalarMatrix<T, DOFS> mass(const Geometry& geometry, const T& density,
                                                    const T& nonStructuralMass, const std::array<T, NODES>& thickness,
                                                    MassFormulation formulation);

    // The centre recovery operator (spec §7.3; ShellCentreRecovery.hpp).
    template <typename T>
    [[nodiscard]] static CentreRecoveryOperator<T, DOFS> recovery(const Geometry& geometry,
                                                                  const PlateConstitutive<T>& constitutive,
                                                                  const std::array<T, NODES>& thickness,
                                                                  const std::array<T, 2>& fibres,
                                                                  const PlateThermalStrain& thermal);

    // The nodal loads of a pressure (spec §5.2) on the element-frame translations.
    [[nodiscard]] static std::array<double, DOFS> pressureLoads(const Geometry& geometry,
                                                                const std::array<double, 4>& cornerPressures,
                                                                const std::optional<base::Vec3>& direction);

    // The nodal loads of a thermal strain (spec §6.3), in the element frame.
    [[nodiscard]] static std::array<double, DOFS> thermalLoads(const Geometry& geometry,
                                                               const PlateConstitutive<double>& constitutive,
                                                               const std::array<double, NODES>& thickness,
                                                               const PlateThermalStrain& thermal);

    // The differential stiffness of constant membrane forces (element frame; ShellDifferential).
    [[nodiscard]] static base::MatN<DOFS> differentialStiffness(const Geometry& geometry,
                                                                const std::array<double, 3>& membraneForce);
};

} // namespace oos::elements
