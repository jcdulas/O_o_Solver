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
    // A flat element under the uniform free strain ε_x = ε_y = e: the displacements u = e x, v = e y are its own
    // equilibrium (K q = f) and the recovered membrane forces vanish.
    constexpr double EXPANSION = 1.0e-3;
    const ShellQuad4::Geometry geometry = quadrilateral(0.0);
    const PlateConstitutive<double> section = isotropicSection(true);
    PlateThermalStrain thermal;
    thermal.membrane = {EXPANSION, EXPANSION, 0.0};
    thermal.isZero = false;
    std::array<double, DOFS> q{};
    for (std::size_t a = 0; a < 4; ++a)
    {
        q[6 * a + 0] = EXPANSION * geometry.position[a][0];
        q[6 * a + 1] = EXPANSION * geometry.position[a][1];
    }
    const auto k = ShellQuad4::stiffness(geometry, section, UNIFORM);
    const std::array<double, DOFS> loads = ShellQuad4::thermalLoads(geometry, section, UNIFORM, thermal);
    const double scale = YOUNG * THICKNESS * EXPANSION;
    for (std::size_t i = 0; i < DOFS; ++i)
    {
        double force = 0.0;
        for (std::size_t j = 0; j < DOFS; ++j)
            force += k(i, j) * q[j];
        runner.expect(std::fabs(force - loads[i]) < 1.0e-10 * scale,
                      "DOF " + std::to_string(i) + ": K q = " + std::to_string(force) + ", load " +
                          std::to_string(loads[i]));
    }
    const std::array<double, 2> fibres{-0.5 * THICKNESS, 0.5 * THICKNESS};
    const auto recovery = ShellQuad4::recovery(geometry, section, UNIFORM, fibres, thermal);
    for (std::size_t row = 0; row < 3; ++row)
    {
        double force = recovery(oos::elements::centre_row::MEMBRANE_FORCE + row, DOFS);
        for (std::size_t j = 0; j < DOFS; ++j)
            force += recovery(oos::elements::centre_row::MEMBRANE_FORCE + row, j) * q[j];
        runner.expect(std::fabs(force) < 1.0e-10 * scale, "membrane force " + std::to_string(row) + " is " +
                                                              std::to_string(force));
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
