// The checks of the transverse shear moduli a shell section reads from its materials: the modulus of a ply in a
// direction of the laminate (elements::plyShearModulus: G1Z and G2Z turned by the ply angle, a blank one rigid), and
// the modulus a ShellQuad8 or ShellTria6 without shear material takes from its bending material, which is that of
// the material frame whatever the direction of the element's x axis (ShellSection::materialShearTermOf).

#include "CheckRunner.hpp"
#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/MaterialMatrices.hpp"
#include "oos/elements/shells/Laminate.hpp"
#include "oos/elements/shells/MindlinShell.hpp"
#include "oos/elements/shells/ShellSection.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>

using oos::base::Vec3;
using oos::elements::LaminatePly;
using oos::elements::MindlinShell;
using oos::elements::SerendipityQuadShape;
using oos::elements::ShellSection;

namespace
{

constexpr double DEGREE = std::numbers::pi / 180.0;
constexpr double TOLERANCE = 1.0e-12;

// A ply at `degrees` from the material axis with the transverse shear moduli G1Z and G2Z (0: blank).
LaminatePly plyOf(double degrees, double g1z, double g2z)
{
    LaminatePly ply;
    ply.angle = degrees * DEGREE;
    ply.material.transverseShear = {g1z, g2z};
    return ply;
}

// An orthotropic lamina in its material frame, per unit thickness: Q11, Q22, Q12 and Q66 = G12 (the plane-stress
// matrix of a ply whose E1 is fourteen times its E2).
constexpr double Q11 = 140.0;
constexpr double Q22 = 10.0;
constexpr double Q12 = 3.0;
constexpr double G12 = 5.0;
constexpr double THICKNESS = 0.2;
constexpr double SHEAR_RATIO = 5.0 / 6.0;

// The section of that lamina as membrane and bending material, without shear material (MID3 blank), in its material
// frame.
ShellSection orthotropicWithoutShearMaterial()
{
    ShellSection section;
    section.hasMembrane = true;
    section.hasBending = true;
    section.hasThickness = true;
    section.thickness = THICKNESS;
    section.bendingRatio = 1.0;
    section.shearRatio = SHEAR_RATIO;
    for (oos::base::Mat3* modulus : {&section.membraneMatrix, &section.bendingMatrix})
    {
        (*modulus)(0, 0) = Q11;
        (*modulus)(1, 1) = Q22;
        (*modulus)(0, 1) = Q12;
        (*modulus)(1, 0) = Q12;
        (*modulus)(2, 2) = G12;
    }
    return section;
}

// A flat square ShellQuad8 of side 1 on the section `section`, its directors the normal of its plane.
MindlinShell<SerendipityQuadShape> squareOn(const ShellSection& section)
{
    constexpr std::array<std::array<double, 2>, 8> PLANE{
        {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.5, 0.0}, {1.0, 0.5}, {0.5, 1.0}, {0.0, 0.5}}};
    std::array<Vec3, 8> position{};
    std::array<Vec3, 8> director{};
    for (std::size_t a = 0; a < 8; ++a)
    {
        position[a] = Vec3{PLANE[a][0], PLANE[a][1], 0.0};
        director[a] = Vec3{0.0, 0.0, 1.0};
    }
    std::array<double, 4> thickness{};
    thickness.fill(THICKNESS);
    constexpr double NO_DRILLING = 0.0;
    return MindlinShell<SerendipityQuadShape>(position, director, section, thickness, NO_DRILLING);
}

} // namespace

CHECK_CASE("shells: the transverse shear modulus of a ply follows its angle, a blank modulus is rigid")
{
    using oos::elements::plyShearModulus;
    constexpr double G1Z = 4.0;
    constexpr double G2Z = 2.0;
    // G_xz = G1Z cos²θ + G2Z sin²θ, G_yz = G1Z sin²θ + G2Z cos²θ.
    runner.expectNear(plyShearModulus(plyOf(0.0, G1Z, G2Z), 0), G1Z, TOLERANCE, "0 degrees, along x");
    runner.expectNear(plyShearModulus(plyOf(0.0, G1Z, G2Z), 1), G2Z, TOLERANCE, "0 degrees, along y");
    runner.expectNear(plyShearModulus(plyOf(90.0, G1Z, G2Z), 0), G2Z, TOLERANCE, "90 degrees, along x");
    runner.expectNear(plyShearModulus(plyOf(90.0, G1Z, G2Z), 1), G1Z, TOLERANCE, "90 degrees, along y");
    runner.expectNear(plyShearModulus(plyOf(45.0, G1Z, G2Z), 0), 0.5 * (G1Z + G2Z), TOLERANCE, "45 degrees, along x");
    runner.expectNear(plyShearModulus(plyOf(30.0, G1Z, G2Z), 0), 0.75 * G1Z + 0.25 * G2Z, TOLERANCE,
                      "30 degrees, along x");

    // A blank modulus (0) is an infinite one: the ply is rigid (0) in every direction that has a share of that axis,
    // and keeps the other modulus only straight along the other axis.
    constexpr double BLANK = 0.0;
    runner.expect(plyShearModulus(plyOf(0.0, BLANK, G2Z), 0) == 0.0, "G1Z blank at 0 degrees: rigid along x");
    runner.expectNear(plyShearModulus(plyOf(0.0, BLANK, G2Z), 1), G2Z, TOLERANCE,
                      "G1Z blank at 0 degrees: G2Z along y");
    runner.expectNear(plyShearModulus(plyOf(90.0, BLANK, G2Z), 0), G2Z, TOLERANCE,
                      "G1Z blank at 90 degrees: G2Z along x");
    runner.expect(plyShearModulus(plyOf(90.0, BLANK, G2Z), 1) == 0.0, "G1Z blank at 90 degrees: rigid along y");
    runner.expect(plyShearModulus(plyOf(45.0, BLANK, G2Z), 0) == 0.0, "G1Z blank at 45 degrees: rigid along x");
    runner.expect(plyShearModulus(plyOf(45.0, BLANK, G2Z), 1) == 0.0, "G1Z blank at 45 degrees: rigid along y");
    runner.expect(plyShearModulus(plyOf(30.0, BLANK, BLANK), 0) == 0.0, "both blank: rigid along x");
    runner.expect(plyShearModulus(plyOf(30.0, BLANK, BLANK), 1) == 0.0, "both blank: rigid along y");
}

CHECK_CASE("shells: the in-plane shear term of a section is read in its material frame")
{
    const ShellSection material = orthotropicWithoutShearMaterial();
    for (const double degrees : {0.0, 15.0, 30.0, 45.0, -60.0, 90.0})
    {
        const std::string at = " at " + std::to_string(static_cast<int>(degrees)) + " degrees";
        const ShellSection turned = material.rotatedToElementFrame(degrees * DEGREE);
        runner.expectNear(turned.materialShearTermOf(turned.bendingMatrix), G12, 1.0e-10, "G12 of the bending" + at);
        // The rate of the bending matrix along its term (3, 3), turned as the section is: 1 in the material frame;
        // along its term (1, 1): 0.
        oos::base::Mat3 shearTerm;
        shearTerm(2, 2) = 1.0;
        oos::base::Mat3 firstTerm;
        firstTerm(0, 0) = 1.0;
        runner.expectNear(
            turned.materialShearTermOf(oos::elements::rotateToElementFrame(shearTerm, degrees * DEGREE)), 1.0,
            1.0e-12, "the rate along G33" + at);
        runner.expectNear(
            turned.materialShearTermOf(oos::elements::rotateToElementFrame(firstTerm, degrees * DEGREE)), 0.0,
            1.0e-12, "the rate along G11" + at);
    }
    // What the term (3, 3) of the turned matrix is at 45 degrees: (Q11 + Q22 − 2 Q12) / 4, seven times G12 here.
    const ShellSection diagonal = material.rotatedToElementFrame(45.0 * DEGREE);
    runner.expectNear(diagonal.bendingMatrix(2, 2), 0.25 * (Q11 + Q22 - 2.0 * Q12), 1.0e-10,
                      "the term (3, 3) of the bending matrix at 45 degrees");
}

CHECK_CASE("shells: the transverse shear of a ShellQuad8 without shear material does not turn with the element")
{
    // S = TS/T · G12 · t on both axes, whatever the angle of the material axis from the element's x axis.
    const double expected = SHEAR_RATIO * G12 * THICKNESS;
    const ShellSection material = orthotropicWithoutShearMaterial();
    for (const double degrees : {0.0, 30.0, 45.0, 75.0})
    {
        const std::string at = " at " + std::to_string(static_cast<int>(degrees)) + " degrees";
        const auto shell = squareOn(material.rotatedToElementFrame(degrees * DEGREE));
        const oos::base::MatN<2> shear = shell.placeOperators().shearStiffness[0];
        runner.expectNear(shear(0, 0), expected, 1.0e-10, "S11 at the centre" + at);
        runner.expectNear(shear(1, 1), expected, 1.0e-10, "S22 at the centre" + at);
        runner.expectNear(shear(0, 1), 0.0, 1.0e-10, "S12 at the centre" + at);
    }
}
