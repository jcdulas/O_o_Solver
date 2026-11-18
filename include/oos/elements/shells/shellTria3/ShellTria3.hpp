#pragma once

// The ShellTria3: a linear three-node shell with 18 degrees of freedom, its stiffness, mass, loads, centre recovery
// and membrane-stress differential stiffness. It is the ShellQuad4 (docs/shell-quad4-formulation.md) brought to three
// nodes, so that two triangles on the grid points of a quadrilateral behave closely to it; the formulation and the
// measured closeness are in docs/shell-tria3-formulation.md. From the methods of the open literature:
// - constant membrane strains and curvatures on the three nodes (the rotations move a fibre as in the ShellQuad4);
// - transverse shear strains tied at the mid-sides: the tangential shear of each side gives a constant shear and one
//   rotational part (the lowest-order field with continuous tangential components: J.-C. Nédélec, Mixed finite elements
//   in R³, Numer. Math. 35 (1980); as a plate element: P.-S. Lee, K.-J. Bathe, Development of MITC isotropic
//   triangular shell finite elements, Comput. Struct. 82 (2004));
// - the residual bending flexibility of the transverse shear (R. H. MacNeal, A simple quadrilateral shell element,
//   Comput. Struct. 8 (1978)), with the factors of the ShellQuad4 and the widths of the parallelogram the triangle is
//   half of;
// - the stabilisation of the rotation about the normal of the ShellQuad4 (T. J. R. Hughes, F. Brezzi, On drilling
//   degrees of freedom, Comput. Methods Appl. Mech. Eng. 72 (1989)), with a stiffness of the order of the bending one
//   beside its membrane term: the rotation about the normal of a flat element is a bending rotation for the
//   neighbours it meets at an angle.
//
// Frames: the element frame (that of the ShellQuad4 on the parallelogram whose diagonal is the longest side of the
// triangle; z the normal, origin the centroid) holds every matrix (local DOFs u, v, w, θx, θy, θz of each node, node
// by node); the reference direction (G1→G2) is that of THETA and of the output (userAngle).
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

// The geometry of one ShellTria3, from the positions of its grid points only.
struct ShellTria3Geometry
{
    base::Mat3 axes = base::Mat3::identity(); // columns x, y, z of the element frame, in basic
    base::Vec3 centre;                        // the centroid, in basic
    std::array<double, 3> x{};                // the plane coordinates of the nodes from the centroid
    std::array<double, 3> y{};
    std::array<double, 3> dX{}; // ∂N_a/∂x, ∂N_a/∂y of the linear shapes (constant)
    std::array<double, 3> dY{};
    double determinant = 0.0; // 2 × the area: the area of the parallelogram the triangle is half of
    // The widths a and b of that parallelogram along x and y (those of the ShellQuad4 on its four corners).
    double widthX = 0.0;
    double widthY = 0.0;
    // Side p runs 1→2, 2→3, 3→1: its direction cosines (c_p, s_p) in the element frame.
    std::array<double, 3> edgeCosine{};
    std::array<double, 3> edgeSine{};
    // The inverse of the side fit: [γx, γy at the centroid; q] = edgeFit · [γ_t(1), γ_t(2), γ_t(3)], the shear field
    // being γ(x, y) = (γx − q y, γy + q x).
    Matrix3<double> edgeFit;
    double userAngle = 0.0; // the reference direction (G1→G2) from the element x axis, radians
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

    // The flat triangle has no integration problem once its geometry is accepted.
    [[nodiscard]] static std::optional<std::string> integrationProblem(const Geometry& /*geometry*/,
                                                                       const std::array<double, NODES>& /*thickness*/)
    {
        return std::nullopt;
    }

    // K (18 × 18) in the element frame.
    template <typename T>
    [[nodiscard]] static ScalarMatrix<T, DOFS> stiffness(const Geometry& geometry,
                                                         const PlateConstitutive<T>& constitutive,
                                                         const std::array<T, NODES>& thickness);

    // The mass in the element frame: ρ t + NSM per unit area on the translations, lumped or coupled; the coupled
    // form has the rotary inertia ρ t³/12 of the normals.
    template <typename T>
    [[nodiscard]] static ScalarMatrix<T, DOFS> mass(const Geometry& geometry, const T& density,
                                                    const T& nonStructuralMass, const std::array<T, NODES>& thickness,
                                                    MassFormulation formulation);

    // The centre recovery operator (ShellCentreRecovery.hpp), `fibres` Z1 and Z2, `thermal` the free thermal strain
    // of the element (subtracted from the stresses and forces).
    template <typename T>
    [[nodiscard]] static CentreRecoveryOperator<T, DOFS> recovery(const Geometry& geometry,
                                                                  const PlateConstitutive<T>& constitutive,
                                                                  const std::array<T, NODES>& thickness,
                                                                  const std::array<T, 2>& fibres,
                                                                  const PlateThermalStrain& thermal);

    // The nodal loads of a pressure on the element-frame translations: `cornerPressures` P1-P3 (P4 is not used),
    // along the normal or along `direction`.
    [[nodiscard]] static std::array<double, DOFS> pressureLoads(const Geometry& geometry,
                                                                const std::array<double, 4>& cornerPressures,
                                                                const std::optional<base::Vec3>& direction);

    // The nodal loads of a thermal strain, in the element frame.
    [[nodiscard]] static std::array<double, DOFS> thermalLoads(const Geometry& geometry,
                                                               const PlateConstitutive<double>& constitutive,
                                                               const std::array<double, NODES>& thickness,
                                                               const PlateThermalStrain& thermal);

    // The differential stiffness of constant membrane forces (element frame; ShellDifferential).
    [[nodiscard]] static base::MatN<DOFS> differentialStiffness(const Geometry& geometry,
                                                                const std::array<double, 3>& membraneForce);
};

} // namespace oos::elements
