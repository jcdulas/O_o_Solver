// The checks of the ShellQuad4 kernel (docs/shell-quad4-formulation.md, section 14): its numerical values, its
// zero-energy modes, and the consistency of its loads and of its recovery with its stiffness.

#include "CheckRunner.hpp"
#include "ShellChecks.hpp"
#include "oos/elements/shells/shellQuad4/ShellQuad4.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

using namespace oos::checks::shells;
using oos::elements::PlateConstitutive;
using oos::elements::PlateThermalStrain;
using oos::elements::ShellQuad4;

namespace
{

constexpr std::size_t DOFS = ShellQuad4::DOFS;

// The quadrilateral of the document: flat, or with its third grid point lifted by `warp`.
ShellQuad4::Geometry quadrilateral(double warp)
{
    const std::array<oos::base::Vec3, 4> positions{
        oos::base::Vec3{0.0, 0.0, 0.0}, oos::base::Vec3{1.2, 0.0, 0.0}, oos::base::Vec3{1.0, 0.9, warp},
        oos::base::Vec3{-0.1, 0.8, 0.0}};
    return ShellQuad4::buildGeometry(positions).geometry;
}

constexpr std::array<double, 4> UNIFORM{THICKNESS, THICKNESS, THICKNESS, THICKNESS};

struct DocumentValues
{
    double warp;
    std::array<double, 18> eigenvalues;
    double massTrace;
};

// The eighteen non-zero eigenvalues of the stiffness and the trace of the consistent mass (document section 14).
const std::array<DocumentValues, 2> DOCUMENT{{
    {0.0,
     {2.099463e-4, 6.305329e-4, 6.315727e-4, 2.485962e-3, 4.669159e-3, 5.546885e-3, 6.624600e-3, 7.210094e-3,
      1.706078e-2, 1.015927e-1, 2.908098e-1, 3.286997e-1, 2.864322, 11.95446, 22.46793, 33.57019, 40.52063, 82.02336},
     6.567579e-2},
    {0.1,
     {2.107253e-4, 6.341382e-4, 6.352187e-4, 2.500274e-3, 4.684838e-3, 5.421407e-3, 7.279753e-3, 8.967955e-3,
      1.706005e-2, 1.017682e-1, 2.899826e-1, 3.303452e-1, 2.877774, 12.00676, 22.45304, 33.60344, 40.46907, 81.86705},
     6.592780e-2},
}};

} // namespace

CHECK_CASE("shellQuad4: the eigenvalues of the stiffness and the trace of the mass are those of the formulation")
{
    // The document prints seven digits.
    constexpr double PRINTED = 5.0e-7;
    for (const DocumentValues& expected : DOCUMENT)
    {
        const ShellQuad4::Geometry geometry = quadrilateral(expected.warp);
        const std::vector<double> values =
            sortedEigenvalues<DOFS>(ShellQuad4::stiffness(geometry, isotropicSection(true), UNIFORM));
        const double largest = values.back();
        for (std::size_t k = 0; k < 6; ++k)
            runner.expect(std::fabs(values[k]) < 1.0e-10 * largest, "zero-energy mode " + std::to_string(k + 1) +
                                                                        " has the eigenvalue " +
                                                                        std::to_string(values[k]));
        for (std::size_t k = 0; k < expected.eigenvalues.size(); ++k)
            runner.expect(std::fabs(values[6 + k] / expected.eigenvalues[k] - 1.0) < PRINTED,
                          "eigenvalue " + std::to_string(k + 1) + " of the element warped by " +
                              std::to_string(expected.warp) + " is " + std::to_string(values[6 + k]) + ", expected " +
                              std::to_string(expected.eigenvalues[k]));
        const auto mass = ShellQuad4::mass(geometry, 1.0, 0.0, UNIFORM, oos::elements::MassFormulation::Coupled);
        double trace = 0.0;
        for (std::size_t i = 0; i < DOFS; ++i)
            trace += mass(i, i);
        runner.expect(std::fabs(trace / expected.massTrace - 1.0) < PRINTED,
                      "trace of the mass " + std::to_string(trace) + ", expected " +
                          std::to_string(expected.massTrace));
    }
}

CHECK_CASE("shellQuad4: a rigid motion of a warped element gives no force, with or without shear flexibility")
{
    const ShellQuad4::Geometry geometry = quadrilateral(0.1);
    for (const bool hasShearFlexibility : {true, false})
    {
        const auto k = ShellQuad4::stiffness(geometry, isotropicSection(hasShearFlexibility), UNIFORM);
        double largest = 0.0;
        for (std::size_t i = 0; i < DOFS; ++i)
            largest = std::fmax(largest, k(i, i));
        for (std::size_t mode = 0; mode < 6; ++mode)
        {
            std::array<double, 3> shift{};
            std::array<double, 3> turn{};
            (mode < 3 ? shift : turn)[mode % 3] = 1.0;
            const std::array<double, DOFS> q = rigidMotion<ShellQuad4::NODES>(geometry.position, shift, turn);
            for (std::size_t i = 0; i < DOFS; ++i)
            {
                double force = 0.0;
                for (std::size_t j = 0; j < DOFS; ++j)
                    force += k(i, j) * q[j];
                runner.expect(std::fabs(force) < 1.0e-10 * largest,
                              "rigid motion " + std::to_string(mode + 1) + " gives the force " + std::to_string(force) +
                                  " on DOF " + std::to_string(i));
            }
        }
    }
}

CHECK_CASE("shellQuad4: the loads of a free thermal expansion are balanced by it and leave no force in the element")
{
    // An element, flat or warped, under the uniform free strain ε_x = ε_y = e: the homothety u = e x is its own
    // equilibrium (K q = f) and the recovered forces, moments and stresses vanish. On the warped element the free
    // strain of the dilatation is the strain of the homothety, which has a curvature there (ShellQuad4Kernel.hpp).
    constexpr double EXPANSION = 1.0e-3;
    const PlateConstitutive<double> section = isotropicSection(true);
    PlateThermalStrain thermal;
    thermal.membrane = {EXPANSION, EXPANSION, 0.0};
    thermal.isZero = false;
    for (const double warp : {0.0, 0.1})
    {
        const ShellQuad4::Geometry geometry = quadrilateral(warp);
        std::array<double, DOFS> q{};
        for (std::size_t a = 0; a < 4; ++a)
            for (std::size_t k = 0; k < 3; ++k)
                q[6 * a + k] = EXPANSION * geometry.position[a][k];
        const auto k = ShellQuad4::stiffness(geometry, section, UNIFORM);
        const std::array<double, DOFS> loads = ShellQuad4::thermalLoads(geometry, section, UNIFORM, thermal);
        const double scale = YOUNG * THICKNESS * EXPANSION;
        for (std::size_t i = 0; i < DOFS; ++i)
        {
            double force = 0.0;
            for (std::size_t j = 0; j < DOFS; ++j)
                force += k(i, j) * q[j];
            runner.expect(std::fabs(force - loads[i]) < 1.0e-10 * scale,
                          "warp " + std::to_string(warp) + ", DOF " + std::to_string(i) + ": K q = " +
                              std::to_string(force) + ", load " + std::to_string(loads[i]));
        }
        const std::array<double, 2> fibres{-0.5 * THICKNESS, 0.5 * THICKNESS};
        const auto recovery = ShellQuad4::recovery(geometry, section, UNIFORM, fibres, thermal);
        for (std::size_t row = 0; row < oos::elements::centre_row::MEMBRANE_STRAIN; ++row)
        {
            double value = recovery(row, DOFS);
            for (std::size_t j = 0; j < DOFS; ++j)
                value += recovery(row, j) * q[j];
            // the stresses are per unit thickness against the forces
            const double rowScale = row >= oos::elements::centre_row::STRESS_FIBER1 ? YOUNG * EXPANSION : scale;
            runner.expect(std::fabs(value) < 1.0e-10 * rowScale,
                          "warp " + std::to_string(warp) + ", force or stress row " + std::to_string(row) + " is " +
                              std::to_string(value));
        }
    }
}

CHECK_CASE("shellQuad4: the centre recovery gives the forces and moments of a constant state")
{
    // A rectangle under the constant curvature of u = z θ, w: the rotations θ_y = k x give κ_x (document) = k, the
    // moment of the tables M_x = −D k (M = −∫σ z dz) and the top fibre stress E/(1 − ν²) k t/2.
    const std::array<oos::base::Vec3, 4> positions{oos::base::Vec3{0.0, 0.0, 0.0}, oos::base::Vec3{2.0, 0.0, 0.0},
                                                   oos::base::Vec3{2.0, 1.0, 0.0}, oos::base::Vec3{0.0, 1.0, 0.0}};
    const ShellQuad4::Geometry geometry = ShellQuad4::buildGeometry(positions).geometry;
    const PlateConstitutive<double> section = isotropicSection(true);
    constexpr double CURVATURE = 0.01;
    std::array<double, DOFS> q{};
    for (std::size_t a = 0; a < 4; ++a)
    {
        const double x = geometry.position[a][0];
        q[6 * a + 4] = CURVATURE * x;            // θ_y: β_x = θ_y
        q[6 * a + 2] = -0.5 * CURVATURE * x * x; // w = −k x²/2 at the nodes: no transverse shear at the centre
    }
    const std::array<double, 2> fibres{-0.5 * THICKNESS, 0.5 * THICKNESS};
    const auto recovery = ShellQuad4::recovery(geometry, section, UNIFORM, fibres, PlateThermalStrain{});
    const auto valueOf = [&](std::size_t row)
    {
        double value = recovery(row, DOFS);
        for (std::size_t j = 0; j < DOFS; ++j)
            value += recovery(row, j) * q[j];
        return value;
    };
    const double plane = YOUNG / (1.0 - POISSON * POISSON);
    const double rigidity = plane * THICKNESS * THICKNESS * THICKNESS / 12.0;
    runner.expectNear(valueOf(oos::elements::centre_row::CURVATURE), -CURVATURE, 1.0e-12, "curvature of the tables");
    runner.expectNear(valueOf(oos::elements::centre_row::MOMENT) / (rigidity * CURVATURE), -1.0, 1.0e-10,
                      "M_x / (D k)");
    runner.expectNear(valueOf(oos::elements::centre_row::MOMENT + 1) / (rigidity * CURVATURE), -POISSON, 1.0e-10,
                      "M_y / (D k)");
    runner.expectNear(valueOf(oos::elements::centre_row::STRESS_FIBER2) / (plane * CURVATURE * 0.5 * THICKNESS), 1.0,
                      1.0e-10, "top fibre stress");
    runner.expectNear(valueOf(oos::elements::centre_row::MEMBRANE_FORCE), 0.0, 1.0e-12, "membrane force");
}

namespace
{

using CornerOperators = std::array<oos::elements::CentreRecoveryOperator<double, DOFS>, ShellQuad4::NODES>;

constexpr std::array<double, 2> FIBRES{-0.5 * THICKNESS, 0.5 * THICKNESS};
constexpr std::array<std::array<double, 2>, 4> CORNER_FIBRES{FIBRES, FIBRES, FIBRES, FIBRES};

// The value of one row of a recovery operator at the local displacements q.
template <typename Operator>
double valueOf(const Operator& recovery, std::size_t row, const std::array<double, DOFS>& q)
{
    double value = recovery(row, DOFS);
    for (std::size_t j = 0; j < DOFS; ++j)
        value += recovery(row, j) * q[j];
    return value;
}

} // namespace

CHECK_CASE("shellQuad4: the corner recovery gives the values of the centre in a constant state")
{
    // The flat distorted quadrilateral of the document under a uniform stretch e along x and a constant curvature k
    // (u = e x + z θ_y, θ_y = k x): the strains of the integration points are those of the state and the enhanced
    // strains vanish (Σ j G = 0), so every corner has the forces, moments, stresses and strains of the centre.
    constexpr double STRETCH = 2.0e-3;
    constexpr double CURVATURE = 0.01;
    const ShellQuad4::Geometry geometry = quadrilateral(0.0);
    const PlateConstitutive<double> section = isotropicSection(true);
    std::array<double, DOFS> q{};
    for (std::size_t a = 0; a < 4; ++a)
    {
        const double x = geometry.position[a][0];
        q[6 * a + 0] = STRETCH * x;
        q[6 * a + 4] = CURVATURE * x;
        q[6 * a + 2] = -0.5 * CURVATURE * x * x;
    }
    const auto centre = ShellQuad4::recovery(geometry, section, UNIFORM, FIBRES, PlateThermalStrain{});
    const CornerOperators corners =
        ShellQuad4::cornerRecovery(geometry, section, UNIFORM, CORNER_FIBRES, PlateThermalStrain{});
    const double plane = YOUNG / (1.0 - POISSON * POISSON);
    runner.expectNear(valueOf(centre, oos::elements::centre_row::MEMBRANE_FORCE, q) / (plane * THICKNESS * STRETCH),
                      1.0, 1.0e-10, "N_x of the centre");
    for (std::size_t corner = 0; corner < 4; ++corner)
        for (std::size_t row = 0; row < oos::elements::CENTRE_RECOVERY_ROWS; ++row)
        {
            // the transverse shear of w = −k x²/2 on a distorted element is not that of the state
            if (row == oos::elements::centre_row::SHEAR_FORCE || row == oos::elements::centre_row::SHEAR_FORCE + 1)
                continue;
            const double expected = valueOf(centre, row, q);
            const double scale = row < oos::elements::centre_row::MEMBRANE_STRAIN ? plane * THICKNESS * STRETCH : 1.0;
            runner.expect(std::fabs(valueOf(corners[corner], row, q) - expected) < 1.0e-10 * scale,
                          "corner " + std::to_string(corner + 1) + ", row " + std::to_string(row) + " is " +
                              std::to_string(valueOf(corners[corner], row, q)) + ", the centre has " +
                              std::to_string(expected));
        }
}

CHECK_CASE("shellQuad4: the corner recovery gives the fibre forces of a rectangle bent in its plane")
{
    // A rectangle 2 × 1 in pure in-plane bending, σ_x = E k y: u = k x y, v = −k x²/2 − ν k y²/2. The enhanced
    // membrane strains give ε_y = −ν k y and remove the shear of the bilinear displacements, so the forces of the
    // integration points are N_x = E t k y, and those of the corners E t k (± ½) with no N_y nor N_xy; the centre
    // has none.
    constexpr double CURVATURE = 0.01;
    const std::array<oos::base::Vec3, 4> positions{oos::base::Vec3{0.0, 0.0, 0.0}, oos::base::Vec3{2.0, 0.0, 0.0},
                                                   oos::base::Vec3{2.0, 1.0, 0.0}, oos::base::Vec3{0.0, 1.0, 0.0}};
    const ShellQuad4::Geometry geometry = ShellQuad4::buildGeometry(positions).geometry;
    const PlateConstitutive<double> section = isotropicSection(true);
    std::array<double, DOFS> q{};
    for (std::size_t a = 0; a < 4; ++a)
    {
        const double x = geometry.position[a][0];
        const double y = geometry.position[a][1];
        q[6 * a + 0] = CURVATURE * x * y;
        q[6 * a + 1] = -0.5 * CURVATURE * (x * x + POISSON * y * y);
    }
    const CornerOperators corners =
        ShellQuad4::cornerRecovery(geometry, section, UNIFORM, CORNER_FIBRES, PlateThermalStrain{});
    const double scale = YOUNG * THICKNESS * CURVATURE;
    for (std::size_t corner = 0; corner < 4; ++corner)
    {
        const double y = geometry.position[corner][1];
        const std::string name = "corner " + std::to_string(corner + 1);
        runner.expectNear(valueOf(corners[corner], oos::elements::centre_row::MEMBRANE_FORCE, q) / scale, y, 1.0e-10,
                          name + ": N_x / (E t k)");
        runner.expectNear(valueOf(corners[corner], oos::elements::centre_row::MEMBRANE_FORCE + 1, q) / scale, 0.0,
                          1.0e-10, name + ": N_y / (E t k)");
        runner.expectNear(valueOf(corners[corner], oos::elements::centre_row::MEMBRANE_FORCE + 2, q) / scale, 0.0,
                          1.0e-10, name + ": N_xy / (E t k)");
        // the stresses of both fibres are N_x / t
        runner.expectNear(valueOf(corners[corner], oos::elements::centre_row::STRESS_FIBER1, q) /
                              (YOUNG * CURVATURE),
                          y, 1.0e-10, name + ": stress of the fibre Z1 / (E k)");
    }
}

CHECK_CASE("shellQuad4: the corner recovery gives the transverse shear force of the centre under a uniform shear")
{
    // A rectangle sheared by w = g x without rotation: the tied shears are uniform, their part odd across the element
    // is zero, and the force conjugate to them at each integration point is that of the centre.
    constexpr double SHEAR = 1.0e-3;
    const std::array<oos::base::Vec3, 4> positions{oos::base::Vec3{0.0, 0.0, 0.0}, oos::base::Vec3{2.0, 0.0, 0.0},
                                                   oos::base::Vec3{2.0, 1.0, 0.0}, oos::base::Vec3{0.0, 1.0, 0.0}};
    const ShellQuad4::Geometry geometry = ShellQuad4::buildGeometry(positions).geometry;
    const PlateConstitutive<double> section = isotropicSection(true);
    std::array<double, DOFS> q{};
    for (std::size_t a = 0; a < 4; ++a)
        q[6 * a + 2] = SHEAR * geometry.position[a][0];
    const auto centre = ShellQuad4::recovery(geometry, section, UNIFORM, FIBRES, PlateThermalStrain{});
    const CornerOperators corners =
        ShellQuad4::cornerRecovery(geometry, section, UNIFORM, CORNER_FIBRES, PlateThermalStrain{});
    const double expected = valueOf(centre, oos::elements::centre_row::SHEAR_FORCE, q);
    runner.expect(std::fabs(expected) > 0.0, "the centre has a shear force");
    for (std::size_t corner = 0; corner < 4; ++corner)
    {
        runner.expectNear(valueOf(corners[corner], oos::elements::centre_row::SHEAR_FORCE, q) / expected, 1.0, 1.0e-10,
                          "corner " + std::to_string(corner + 1) + ": Q_x against the centre's");
        runner.expect(std::fabs(valueOf(corners[corner], oos::elements::centre_row::SHEAR_FORCE + 1, q)) <
                          1.0e-10 * std::fabs(expected),
                      "corner " + std::to_string(corner + 1) + " has no Q_y");
    }
}

CHECK_CASE("shellQuad4: the corner recovery gives nothing in a rigid motion nor in a free thermal expansion")
{
    // A warped element in each of its six rigid motions has no strain at its corners, and under the uniform free
    // strain ε_x = ε_y = e with the homothety u = e x no force, moment nor stress (the centre recovery's check).
    constexpr double EXPANSION = 1.0e-3;
    const PlateConstitutive<double> section = isotropicSection(true);
    const ShellQuad4::Geometry geometry = quadrilateral(0.1);
    const CornerOperators atRest =
        ShellQuad4::cornerRecovery(geometry, section, UNIFORM, CORNER_FIBRES, PlateThermalStrain{});
    for (std::size_t mode = 0; mode < 6; ++mode)
    {
        std::array<double, 3> shift{};
        std::array<double, 3> turn{};
        (mode < 3 ? shift : turn)[mode % 3] = 1.0;
        const std::array<double, DOFS> q = rigidMotion<ShellQuad4::NODES>(geometry.position, shift, turn);
        for (std::size_t corner = 0; corner < 4; ++corner)
            for (std::size_t row = 0; row < oos::elements::CENTRE_RECOVERY_ROWS; ++row)
            {
                const double scale = row < oos::elements::centre_row::MEMBRANE_STRAIN ? YOUNG * THICKNESS : 1.0;
                runner.expect(std::fabs(valueOf(atRest[corner], row, q)) < 1.0e-10 * scale,
                              "rigid motion " + std::to_string(mode + 1) + ", corner " + std::to_string(corner + 1) +
                                  ", row " + std::to_string(row) + " is " +
                                  std::to_string(valueOf(atRest[corner], row, q)));
            }
    }
    PlateThermalStrain thermal;
    thermal.membrane = {EXPANSION, EXPANSION, 0.0};
    thermal.isZero = false;
    for (const double warp : {0.0, 0.1})
    {
        const ShellQuad4::Geometry expanded = quadrilateral(warp);
        std::array<double, DOFS> q{};
        for (std::size_t a = 0; a < 4; ++a)
            for (std::size_t k = 0; k < 3; ++k)
                q[6 * a + k] = EXPANSION * expanded.position[a][k];
        const CornerOperators corners =
            ShellQuad4::cornerRecovery(expanded, section, UNIFORM, CORNER_FIBRES, thermal);
        const double scale = YOUNG * THICKNESS * EXPANSION;
        for (std::size_t corner = 0; corner < 4; ++corner)
            for (std::size_t row = 0; row < oos::elements::centre_row::MEMBRANE_STRAIN; ++row)
            {
                const double rowScale = row >= oos::elements::centre_row::STRESS_FIBER1 ? YOUNG * EXPANSION : scale;
                runner.expect(std::fabs(valueOf(corners[corner], row, q)) < 1.0e-10 * rowScale,
                              "warp " + std::to_string(warp) + ", corner " + std::to_string(corner + 1) +
                                  ", force or stress row " + std::to_string(row) + " is " +
                                  std::to_string(valueOf(corners[corner], row, q)));
            }
    }
}
