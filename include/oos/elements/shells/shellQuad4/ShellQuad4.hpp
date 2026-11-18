#pragma once

// The ShellQuad4: a linear four-node shell with 24 degrees of freedom, its stiffness, mass, loads, centre recovery and
// membrane-stress differential stiffness. The formulation is stated in docs/shell-quad4-formulation.md ("the
// document": its sections are named in the sources), from the methods of the open literature:
// - shell kinematics with a director at each node, the rotations moving a fibre by (ζ/2) t (θ × n): S. Ahmad,
//   B. M. Irons, O. C. Zienkiewicz, Analysis of thick and thin shell structures by curved finite elements, Int. J.
//   Numer. Methods Eng. 2 (1970);
// - transverse shear strains tied at the mid-sides: E. N. Dvorkin, K.-J. Bathe, A continuum mechanics based four-node
//   shell element for general non-linear analysis, Eng. Comput. 1 (1984);
// - four enhanced parameters on the membrane strains and four on the curvatures, condensed in the element: T. H. H.
//   Pian, K. Sumihara, Rational approach for assumed stress finite elements, Int. J. Numer. Methods Eng. 20 (1984),
//   in the enhanced-strain form of J. C. Simo, M. S. Rifai, A class of mixed assumed strain methods and the method of
//   incompatible modes, Int. J. Numer. Methods Eng. 29 (1990);
// - the residual bending flexibility of the transverse shear: R. H. MacNeal, A simple quadrilateral shell element,
//   Comput. Struct. 8 (1978);
// - a stabilisation of the rotation about the normal: T. J. R. Hughes, F. Brezzi, On drilling degrees of freedom,
//   Comput. Methods Appl. Mech. Eng. 72 (1989).
//
// Frames: the element frame (x along the bisector of the diagonals, z the normal of the plane of the diagonals,
// origin the centroid) holds every matrix (local DOFs u, v, w, θx, θy, θz of each node, node by node); each
// integration point has a local frame of its own on a warped element (ShellQuad4Kernel.hpp); the reference direction
// (G1→G2 projected on the plane normal to z) is that of THETA and of the output (userAngle).
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

// The geometry of one ShellQuad4 (document sections 2 and 3), from the positions of its grid points only.
struct ShellQuad4Geometry
{
    base::Mat3 axes = base::Mat3::identity();  // columns x, y, z of the element frame, in basic
    base::Vec3 centre;                         // the mean of the four grid points, in basic
    std::array<Vector3<double>, 4> position{}; // x_i: the grid points in the element frame, from the centre
    std::array<Vector3<double>, 4> normal{};   // n_i: the unit directors in the element frame
    double meanWidthX = 0.0;                   // a = ½ |x2 + x3 − x1 − x4|
    double meanWidthY = 0.0;                   // b = ½ |y3 + y4 − y1 − y2|
    double userAngle = 0.0; // the reference direction (G1→G2 projected) from the element x axis, radians
};

struct ShellQuad4GeometryResult
{
    ShellQuad4Geometry geometry;
    std::string problemField; // the field to name when the geometry is refused ("node1" ...); empty when accepted
    std::string problem;      // why
};

// The geometry from the positions of G1 to G4 in the basic system; refused: parallel or null diagonals, a degenerate
// director (two edges along one line at a node), two grid points at one position, a folded quadrilateral.
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

    // The problem of an element whose Jacobian is not positive at an integration point, on its top or bottom face (a
    // folded corner, a thickness large against a warped element), or nothing.
    [[nodiscard]] static std::optional<std::string> integrationProblem(const Geometry& geometry,
                                                                       const std::array<double, NODES>& thickness);

    // K (24 × 24) in the element frame (document sections 3 to 9).
    template <typename T>
    [[nodiscard]] static ScalarMatrix<T, DOFS> stiffness(const Geometry& geometry,
                                                         const PlateConstitutive<T>& constitutive,
                                                         const std::array<T, NODES>& thickness);

    // The mass (document section 10) in the element frame: ρ t + NSM per unit area on the translations, lumped or
    // coupled; the coupled form has the rotary inertia ρ t³/12 of the directors.
    template <typename T>
    [[nodiscard]] static ScalarMatrix<T, DOFS> mass(const Geometry& geometry, const T& density,
                                                    const T& nonStructuralMass, const std::array<T, NODES>& thickness,
                                                    MassFormulation formulation);

    // The centre recovery operator (ShellCentreRecovery.hpp; document section 12 at the centre), `fibres` Z1 and Z2,
    // `thermal` the free thermal strain of the element (subtracted from the stresses and forces).
    template <typename T>
    [[nodiscard]] static CentreRecoveryOperator<T, DOFS> recovery(const Geometry& geometry,
                                                                  const PlateConstitutive<T>& constitutive,
                                                                  const std::array<T, NODES>& thickness,
                                                                  const std::array<T, 2>& fibres,
                                                                  const PlateThermalStrain& thermal);

    // The nodal loads of a pressure (document section 12), on the element-frame translations: `cornerPressures`
    // P1-P4, along
    // the vector `direction` (element frame, unit) when given, else along the surface normal.
    [[nodiscard]] static std::array<double, DOFS> pressureLoads(const Geometry& geometry,
                                                                const std::array<double, 4>& cornerPressures,
                                                                const std::optional<base::Vec3>& direction);

    // The nodal loads of a thermal strain (document section 12), in the element frame.
    [[nodiscard]] static std::array<double, DOFS> thermalLoads(const Geometry& geometry,
                                                               const PlateConstitutive<double>& constitutive,
                                                               const std::array<double, NODES>& thickness,
                                                               const PlateThermalStrain& thermal);

    // The differential stiffness of the membrane forces `membraneForce` (element frame), constant over the element
    // (ShellDifferential; document section 12), with the 2 × 2 rule.
    [[nodiscard]] static base::MatN<DOFS> differentialStiffness(const Geometry& geometry,
                                                                const std::array<double, 3>& membraneForce);
};

} // namespace oos::elements
