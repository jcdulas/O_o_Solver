#pragma once

// The ShellQuad4 shell of NASTRAN-95 (the program of NASA, open-sourced by NASA in 2015 under the NASA Open Source
// Agreement): its stiffness, mass, loads, centre recovery and a membrane-stress differential stiffness, implemented
// from the mathematical description of the element (routines quad4d, q4shps, q4nrms, jacob2, q4bmgd, q4gmgs, trplmd,
// tldrs, plod4d, tlqd4d, squd41/42), not translated from its source. "Spec §n" below numbers the parts of that
// description; each is the work of these NASTRAN-95 routines, the public source to check it against:
//   §1     materials and their rotation to the element frame        premat, q4gmgs, quad4d
//   §2.2   mean plane, user and element frames, nodal normals       quad4d, betrns, q4nrms, jacob2
//   §2.4   material axes, moduli of the shear-correction factors    quad4d
//   §2.5   shape functions, Jacobian, lamina frame, B rows 1-6      q4shps, jacob2, q4bmgd
//   §2.6   assumed transverse shear of the edge strains             q4bmgd, quad4d
//   §2.7   shear-correction factors C_b, C_t and anisotropic ones   quad4d
//   §2.8-9 constitutive matrix, 2 × 2 × 2 integration               quad4d, trplmd
//   §2.10  mass;  §2.11 element to global with the offset            quad4d;  tldrs
//   §5.1   FacePressureLoad;  §6.2 thermal loads;  §7.2 recovery               plod4d;  tlqd4d;  squd41, squd42
//
// The element in a few words:
// - a degenerated (Ahmad) shell on four nodes: the geometry X(ξ, η, ζ) = Σ N_a [X_a + (ζ/2) t_a ν_a] with a normal ν_a
//   at each node, the rotations moving a fibre by (ζ/2) t_a (θ_a × ν_a), integrated 2 × 2 × 2;
// - selective reduced in-plane shear: γxy (membrane and bending) taken at the centre;
// - transverse shear from the four tangential edge strains, interpolated through the corner dual vectors, the parts
//   even and odd across the element weighted by shear-correction factors (C_b = 1.6 for bending, C_t for torsion, and
//   factors for an anisotropic coupling), in series with the shear compliance when MID3 is given;
// - no drilling stiffness: θz has none in a flat element (the solver's automatic constraint of singular DOFs, PARAM
//   AUTOSPC, removes it); no rotary inertia.
//
// Frames: the element frame E (x along the bisector of the diagonals, z the normal of the mean plane, origin the
// centroid) holds every matrix (local DOFs u, v, w, θx, θy, θz of each node, node by node); the user frame (x along
// G1→G2 projected on the mean plane) is the reference of THETA and of the output (userAngle).
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

// The geometry of one ShellQuad4 (spec §2.2, §2.6, §2.7), from the positions of its grid points only.
struct ShellQuad4Geometry
{
    base::Mat3 axes = base::Mat3::identity();  // columns x, y, z of the element frame, in basic
    base::Vec3 centre;                         // the mean of the four grid points, in basic
    std::array<Vector3<double>, 4> position{}; // X_a: the grid points in the element frame, from the centre
    std::array<Vector3<double>, 4> normal{};   // ν_a: the unit nodal normals in the element frame
    // Edge e runs from node e to node e + 1 (cyclic): its unit direction ê_e, length L_e and mean normal μ_e.
    std::array<Vector3<double>, 4> edgeDirection{};
    std::array<Vector3<double>, 4> edgeNormal{};
    std::array<double, 4> edgeLength{};
    std::array<double, 4> cornerCosine{}; // c_a = −ê_a·ê_(a−1), the cosine of the interior angle at node a
    double meanWidthX = 0.0;              // A = ½ |X2 + X3 − X1 − X4|
    double meanWidthY = 0.0;              // B = ½ |Y4 + Y3 − Y1 − Y2|
    double aspect = 0.0;                  // α = min(A, B) / max(A, B)
    double warpFactor = 1.0;              // S_w of the anisotropic shear factor (1 when flat)
    double userAngle = 0.0;               // the user x axis (G1→G2 projected) from the element x axis, radians
};

struct ShellQuad4GeometryResult
{
    ShellQuad4Geometry geometry;
    std::string problemField; // the field to name when the geometry is refused ("node1" ...); empty when accepted
    std::string problem;      // why
};

// The geometry from the positions of G1 to G4 in the basic system; refused: parallel or null diagonals, a degenerate
// nodal normal, two edges along one line (a corner of 0° or 180°).
[[nodiscard]] ShellQuad4GeometryResult buildShellQuad4Geometry(const std::array<base::Vec3, 4>& positions);

// The ShellQuad4 kernel. `thickness` holds the thickness at each node; the constitutive data are in the element frame.
struct ShellQuad4
{
    static constexpr std::size_t NODES = 4;
    static constexpr std::size_t DOFS = 24;
    static constexpr std::string_view CARD = "ShellQuad4";
    using Geometry = ShellQuad4Geometry;

    [[nodiscard]] static ShellQuad4GeometryResult buildGeometry(const std::array<base::Vec3, NODES>& positions)
    {
        return buildShellQuad4Geometry(positions);
    }

    // The problem of an element whose 3-D Jacobian is not positive at an integration point (spec §2.5: "bad or
    // reverse geometry", e.g. a thickness large against a warped or folded element), or nothing.
    [[nodiscard]] static std::optional<std::string> integrationProblem(const Geometry& geometry,
                                                                       const std::array<double, NODES>& thickness);

    // K (24 × 24) in the element frame (spec §2.5 to §2.9).
    template <typename T>
    [[nodiscard]] static ScalarMatrix<T, DOFS> stiffness(const Geometry& geometry,
                                                         const PlateConstitutive<T>& constitutive,
                                                         const std::array<T, NODES>& thickness);

    // The translational mass (spec §2.10): ρ t + NSM per unit area, lumped or coupled, in the element frame.
    template <typename T>
    [[nodiscard]] static ScalarMatrix<T, DOFS> mass(const Geometry& geometry, const T& density,
                                                    const T& nonStructuralMass, const std::array<T, NODES>& thickness,
                                                    MassFormulation formulation);

    // The centre recovery operator (spec §7.2; ShellCentreRecovery.hpp), `fibres` Z1 and Z2, `thermal` the free
    // thermal strain of the element (subtracted from the stresses and forces).
    template <typename T>
    [[nodiscard]] static CentreRecoveryOperator<T, DOFS> recovery(const Geometry& geometry,
                                                                  const PlateConstitutive<T>& constitutive,
                                                                  const std::array<T, NODES>& thickness,
                                                                  const std::array<T, 2>& fibres,
                                                                  const PlateThermalStrain& thermal);

    // The nodal loads of a pressure (spec §5.1), on the element-frame translations: `cornerPressures` P1-P4, along
    // the vector `direction` (element frame, unit) when given, else along the surface normal.
    [[nodiscard]] static std::array<double, DOFS> pressureLoads(const Geometry& geometry,
                                                                const std::array<double, 4>& cornerPressures,
                                                                const std::optional<base::Vec3>& direction);

    // The nodal loads of a thermal strain (spec §6.2), in the element frame.
    [[nodiscard]] static std::array<double, DOFS> thermalLoads(const Geometry& geometry,
                                                               const PlateConstitutive<double>& constitutive,
                                                               const std::array<double, NODES>& thickness,
                                                               const PlateThermalStrain& thermal);

    // The differential stiffness of the membrane forces `membraneForce` (element frame), constant over the element
    // (ShellDifferential), on the mean plane with the 2 × 2 rule.
    [[nodiscard]] static base::MatN<DOFS> differentialStiffness(const Geometry& geometry,
                                                                const std::array<double, 3>& membraneForce);
};

} // namespace oos::elements
