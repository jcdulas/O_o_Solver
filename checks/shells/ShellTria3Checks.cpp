// The checks of the ShellTria3 kernel (docs/shell-tria3-formulation.md): its numerical values, its zero-energy modes,
// the consistency of its loads and of its recovery with its stiffness, and two triangles against one ShellQuad4.

#include "CheckRunner.hpp"
#include "ShellChecks.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/shells/shellQuad4/ShellQuad4.hpp"
#include "oos/elements/shells/shellTria3/ShellTria3.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace mh = oos::elements::mixedHybrid;
using namespace oos::checks::shells;
using oos::elements::PlateConstitutive;
using oos::elements::PlateThermalStrain;
using oos::elements::ShellQuad4;
using oos::elements::ShellTria3;

namespace
{

constexpr std::size_t DOFS = ShellTria3::DOFS;
constexpr std::array<double, 3> UNIFORM{THICKNESS, THICKNESS, THICKNESS};

// A triangle without symmetry, out of the planes of the basic axes.
ShellTria3::Geometry scalene()
{
    const std::array<oos::base::Vec3, 3> positions{oos::base::Vec3{0.0, 0.0, 0.0}, oos::base::Vec3{1.2, 0.1, 0.0},
                                                   oos::base::Vec3{0.3, 0.9, 0.2}};
    return ShellTria3::buildGeometry(positions).geometry;
}

std::array<std::array<double, 3>, 3> positionsOf(const ShellTria3::Geometry& geometry)
{
    std::array<std::array<double, 3>, 3> position{};
    for (std::size_t a = 0; a < 3; ++a)
        position[a] = geometry.position[a];
    return position;
}

struct ReferenceValues
{
    bool hasShearFlexibility;
    std::array<double, 12> eigenvalues;
};

// The twelve non-zero eigenvalues of the stiffness of the scalene triangle with the section of the document
// (E = 1000, ν = 0.3, t = 0.05), from an independent implementation of docs/shell-tria3-formulation.md.
const std::array<ReferenceValues, 2> REFERENCE{{
    {true,
     {3.452772614e-04, 3.452772614e-04, 4.584678133e-03, 5.950790620e-03, 6.213710696e-03, 1.436811758e-02,
      2.426647736e-02, 8.562176106e-02, 2.097010749e-01, 2.979552950e+01, 3.463211936e+01, 6.915264118e+01}},
    {false,
     {3.452772614e-04, 3.452772614e-04, 4.584678133e-03, 5.950907217e-03, 6.220952221e-03, 1.436816311e-02,
      2.502438656e-02, 8.613741876e-02, 2.116889453e-01, 2.979552950e+01, 3.463211936e+01, 6.915264118e+01}},
}};
constexpr double REFERENCE_MASS_TRACE = 4.040305034e-02;

// The tip deflection of a cantilever of one cell, L along x and b along y, clamped at x = 0 and loaded by ½ on each
// grid point of its tip, over the deflection of the beam, L³ / (3 E I) + L / (κ G A): the cell is one ShellQuad4
// (`diagonal` 0), or two ShellTria3 on the diagonal 1-3 (1) or 2-4 (2).
double cantileverRatio(double length, double width, int diagonal)
{
    const PlateConstitutive<double> section = isotropicSection(true, 0.0);
    const std::array<oos::base::Vec3, 4> grid{oos::base::Vec3{0.0, 0.0, 0.0}, oos::base::Vec3{length, 0.0, 0.0},
                                              oos::base::Vec3{length, width, 0.0}, oos::base::Vec3{0.0, width, 0.0}};
    mh::Matrix model(24, 24);
    if (diagonal == 0)
    {
        const ShellQuad4::Geometry geometry = ShellQuad4::buildGeometry(grid).geometry;
        const std::array<double, 4> thickness{THICKNESS, THICKNESS, THICKNESS, THICKNESS};
        addToModel<24>(model, ShellQuad4::stiffness(geometry, section, thickness), geometry.axes, {0, 1, 2, 3});
    }
    else
    {
        const std::array<std::array<std::size_t, 3>, 2> triangles =
            diagonal == 1 ? std::array<std::array<std::size_t, 3>, 2>{{{0, 1, 2}, {0, 2, 3}}}
                          : std::array<std::array<std::size_t, 3>, 2>{{{0, 1, 3}, {1, 2, 3}}};
        for (const std::array<std::size_t, 3>& nodes : triangles)
        {
            const ShellTria3::Geometry geometry =
                ShellTria3::buildGeometry({grid[nodes[0]], grid[nodes[1]], grid[nodes[2]]}).geometry;
            addToModel<18>(model, ShellTria3::stiffness(geometry, section, UNIFORM), geometry.axes, nodes);
        }
    }
    // The free DOFs: those of grid points 2 and 3 (the tip).
    mh::Matrix free(12, 12);
    for (std::size_t i = 0; i < 12; ++i)
        for (std::size_t j = 0; j < 12; ++j)
            free(i, j) = model(6 + i, 6 + j);
    mh::Vector load(12, 0.0);
    load[2] = 0.5;
    load[8] = 0.5;
    const std::optional<mh::Cholesky> factor = mh::Cholesky::factor(free);
    if (!factor)
        return 0.0;
    const mh::Vector u = factor->solve(load);
    const double inertia = width * THICKNESS * THICKNESS * THICKNESS / 12.0;
    const double shearArea = SHEAR_FACTOR * (0.5 * YOUNG) * width * THICKNESS;
    const double beam = length * length * length / (3.0 * YOUNG * inertia) + length / shearArea;
    return 0.5 * (u[2] + u[8]) / beam;
}

} // namespace

CHECK_CASE("shellTria3: the eigenvalues of the stiffness and the trace of the mass are those of the formulation")
{
    constexpr double PRINTED = 5.0e-9;
    const ShellTria3::Geometry geometry = scalene();
    for (const ReferenceValues& expected : REFERENCE)
    {
        const PlateConstitutive<double> section = isotropicSection(expected.hasShearFlexibility);
        const std::vector<double> values = sortedEigenvalues<DOFS>(ShellTria3::stiffness(geometry, section, UNIFORM));
        const double largest = values.back();
        for (std::size_t k = 0; k < 6; ++k)
            runner.expect(std::fabs(values[k]) < 1.0e-10 * largest, "zero-energy mode " + std::to_string(k + 1) +
                                                                        " has the eigenvalue " +
                                                                        std::to_string(values[k]));
        for (std::size_t k = 0; k < expected.eigenvalues.size(); ++k)
            runner.expect(std::fabs(values[6 + k] / expected.eigenvalues[k] - 1.0) < PRINTED,
                          "eigenvalue " + std::to_string(k + 1) + " is " + std::to_string(values[6 + k]) +
                              ", expected " + std::to_string(expected.eigenvalues[k]));
    }
    const auto mass = ShellTria3::mass(geometry, 1.0, 0.0, UNIFORM, oos::elements::MassFormulation::Coupled);
    double trace = 0.0;
    for (std::size_t i = 0; i < DOFS; ++i)
        trace += mass(i, i);
    runner.expect(std::fabs(trace / REFERENCE_MASS_TRACE - 1.0) < PRINTED,
                  "trace of the mass " + std::to_string(trace) + ", expected " + std::to_string(REFERENCE_MASS_TRACE));
    // The lumped mass: ρ t area, a third on each translation of each node, nothing on the rotations.
    const auto lumped = ShellTria3::mass(geometry, 1.0, 0.0, UNIFORM, oos::elements::MassFormulation::Lumped);
    const double third = THICKNESS * 0.5 * geometry.determinant / 3.0;
    for (std::size_t i = 0; i < DOFS; ++i)
        runner.expectNear(lumped(i, i), i % 6 < 3 ? third : 0.0, 1.0e-14, "lumped mass of DOF " + std::to_string(i));
}

CHECK_CASE("shellTria3: directors that are not the normal of the triangle give the values of the formulation")
{
    // The scalene triangle with a director at each node 6 to 10 degrees off its normal, as the normals its grid points
    // would share with neighbouring elements: the twelve non-zero eigenvalues of the stiffness, from an independent
    // implementation, six zero-energy modes, and no force under a rigid motion.
    const std::array<oos::base::Vec3, 3> directors{
        oos::base::Vec3{0.11634746723402964, -0.26768324260002146, 0.95645645405285662},
        oos::base::Vec3{-0.06255936396723194, -0.10466846902744856, 0.99253757489148253},
        oos::base::Vec3{0.049310513060364561, -0.13482329824164022, 0.98964193097956865}};
    const std::array<double, 12> expected{3.412781254e-04, 3.450049511e-04, 4.378774519e-03, 5.975466350e-03,
                                          6.544627910e-03, 1.462007979e-02, 2.443904227e-02, 8.569121971e-02,
                                          2.105586843e-01, 2.979652551e+01, 3.463327897e+01, 6.915416522e+01};
    ShellTria3::Geometry geometry = scalene();
    for (std::size_t a = 0; a < 3; ++a)
        for (std::size_t k = 0; k < 3; ++k)
            geometry.normal[a][k] = directors[a][0] * geometry.axes(0, k) + directors[a][1] * geometry.axes(1, k) +
                                    directors[a][2] * geometry.axes(2, k);
    const auto k = ShellTria3::stiffness(geometry, isotropicSection(true), UNIFORM);
    const std::vector<double> values = sortedEigenvalues<DOFS>(k);
    for (std::size_t mode = 0; mode < 6; ++mode)
        runner.expect(std::fabs(values[mode]) < 1.0e-10 * values.back(),
                      "zero-energy mode " + std::to_string(mode + 1) + " has the eigenvalue " +
                          std::to_string(values[mode]));
    for (std::size_t mode = 0; mode < expected.size(); ++mode)
        runner.expect(std::fabs(values[6 + mode] / expected[mode] - 1.0) < 5.0e-9,
                      "eigenvalue " + std::to_string(mode + 1) + " is " + std::to_string(values[6 + mode]) +
                          ", expected " + std::to_string(expected[mode]));
    for (std::size_t mode = 0; mode < 6; ++mode)
    {
        std::array<double, 3> shift{};
        std::array<double, 3> turn{};
        (mode < 3 ? shift : turn)[mode % 3] = 1.0;
        const std::array<double, DOFS> q = rigidMotion<ShellTria3::NODES>(positionsOf(geometry), shift, turn);
        for (std::size_t i = 0; i < DOFS; ++i)
        {
            double force = 0.0;
            for (std::size_t j = 0; j < DOFS; ++j)
                force += k(i, j) * q[j];
            runner.expect(std::fabs(force) < 1.0e-10 * values.back(),
                          "rigid motion " + std::to_string(mode + 1) + " gives the force " + std::to_string(force) +
                              " on DOF " + std::to_string(i));
        }
    }
}

CHECK_CASE("shellTria3: a rigid motion gives no force, with or without shear flexibility")
{
    const ShellTria3::Geometry geometry = scalene();
    for (const bool hasShearFlexibility : {true, false})
    {
        const auto k = ShellTria3::stiffness(geometry, isotropicSection(hasShearFlexibility), UNIFORM);
        double largest = 0.0;
        for (std::size_t i = 0; i < DOFS; ++i)
            largest = std::fmax(largest, k(i, i));
        for (std::size_t mode = 0; mode < 6; ++mode)
        {
            std::array<double, 3> shift{};
            std::array<double, 3> turn{};
            (mode < 3 ? shift : turn)[mode % 3] = 1.0;
            const std::array<double, DOFS> q = rigidMotion<ShellTria3::NODES>(positionsOf(geometry), shift, turn);
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

CHECK_CASE("shellTria3: the loads of a free thermal expansion are balanced by it and leave no force in the element")
{
    // The uniform free strain ε_x = ε_y = e and the free curvature g of ε(z) = e + z g: the displacements u = e x,
    // v = e y, w = −g (x² + y²)/2 with the rotations of its normals, θx = w,y and θy = −w,x, are its own equilibrium
    // (K q = f) and the recovered forces and moments vanish.
    constexpr double EXPANSION = 1.0e-3;
    constexpr double GRADIENT = 2.0e-2;
    const ShellTria3::Geometry geometry = scalene();
    const PlateConstitutive<double> section = isotropicSection(true);
    PlateThermalStrain thermal;
    thermal.membrane = {EXPANSION, EXPANSION, 0.0};
    thermal.gradient = {GRADIENT, GRADIENT, 0.0};
    thermal.isZero = false;
    std::array<double, DOFS> q{};
    for (std::size_t a = 0; a < 3; ++a)
    {
        const double x = geometry.position[a][0];
        const double y = geometry.position[a][1];
        q[6 * a + 0] = EXPANSION * x;
        q[6 * a + 1] = EXPANSION * y;
        q[6 * a + 2] = -0.5 * GRADIENT * (x * x + y * y);
        q[6 * a + 3] = -GRADIENT * y;
        q[6 * a + 4] = GRADIENT * x;
    }
    const auto k = ShellTria3::stiffness(geometry, section, UNIFORM);
    const std::array<double, DOFS> loads = ShellTria3::thermalLoads(geometry, section, UNIFORM, thermal);
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
    const auto recovery = ShellTria3::recovery(geometry, section, UNIFORM, fibres, thermal);
    for (std::size_t row = 0; row < oos::elements::centre_row::MEMBRANE_STRAIN; ++row)
    {
        double value = recovery(row, DOFS);
        for (std::size_t j = 0; j < DOFS; ++j)
            value += recovery(row, j) * q[j];
        runner.expect(std::fabs(value) < 1.0e-10 * scale, "force or stress row " + std::to_string(row) + " is " +
                                                              std::to_string(value));
    }
}

CHECK_CASE("shellTria3: the centre recovery gives the forces and moments of a constant curvature")
{
    // The rotations θy = k x and w = −k x²/2 at the nodes: κ_x = k in ε(z) = e0 + z κ, the moment of the tables
    // M_x = −D k (M = −∫σ z dz), the top fibre stress E/(1 − ν²) k t/2 and no transverse shear force.
    const std::array<oos::base::Vec3, 3> positions{oos::base::Vec3{0.0, 0.0, 0.0}, oos::base::Vec3{2.0, 0.0, 0.0},
                                                   oos::base::Vec3{2.0, 1.0, 0.0}};
    const ShellTria3::Geometry geometry = ShellTria3::buildGeometry(positions).geometry;
    const PlateConstitutive<double> section = isotropicSection(true);
    constexpr double BENDING = 0.01;
    // The basic x axis in the element frame (the element x axis is along a side of the right angle).
    const double cosine = geometry.axes(0, 0);
    const double sine = geometry.axes(0, 1);
    std::array<double, DOFS> q{};
    for (std::size_t a = 0; a < 3; ++a)
    {
        const double x = positions[a][0];
        q[6 * a + 2] = -0.5 * BENDING * x * x;
        // the rotation k x about the basic y axis, in the element frame
        q[6 * a + 3] = -sine * BENDING * x;
        q[6 * a + 4] = cosine * BENDING * x;
    }
    const std::array<double, 2> fibres{-0.5 * THICKNESS, 0.5 * THICKNESS};
    const auto recovery = ShellTria3::recovery(geometry, section, UNIFORM, fibres, PlateThermalStrain{});
    const auto valueOf = [&](std::size_t row)
    {
        double value = recovery(row, DOFS);
        for (std::size_t j = 0; j < DOFS; ++j)
            value += recovery(row, j) * q[j];
        return value;
    };
    // The values along the basic x axis: the invariants of the element-frame rows.
    const double plane = YOUNG / (1.0 - POISSON * POISSON);
    const double rigidity = plane * THICKNESS * THICKNESS * THICKNESS / 12.0;
    using namespace oos::elements::centre_row;
    runner.expectNear(valueOf(CURVATURE) + valueOf(CURVATURE + 1), -BENDING, 1.0e-12, "sum of the curvatures");
    runner.expectNear((valueOf(MOMENT) + valueOf(MOMENT + 1)) / (rigidity * BENDING), -(1.0 + POISSON), 1.0e-10,
                      "(M_x + M_y) / (D k)");
    runner.expectNear((valueOf(STRESS_FIBER2) + valueOf(STRESS_FIBER2 + 1)) / (plane * BENDING * 0.5 * THICKNESS),
                      1.0 + POISSON, 1.0e-10, "sum of the top fibre stresses");
    runner.expectNear(valueOf(MEMBRANE_FORCE), 0.0, 1.0e-12, "membrane force");
    runner.expectNear(valueOf(SHEAR_FORCE) / (rigidity * BENDING), 0.0, 1.0e-9, "shear force Q_x");
    runner.expectNear(valueOf(SHEAR_FORCE + 1) / (rigidity * BENDING), 0.0, 1.0e-9, "shear force Q_y");
}

CHECK_CASE("shellTria3: two triangles bend a cantilever cell as one ShellQuad4")
{
    // The ShellQuad4 gives the deflection of the beam with one element. Two ShellTria3 are within 1 % of it on a cell
    // twice as long as wide, on either diagonal, and tend to it as the cell lengthens.
    runner.expectNear(cantileverRatio(2.0, 1.0, 0), 1.0, 1.0e-9, "one ShellQuad4");
    for (const int diagonal : {1, 2})
    {
        runner.expectNear(cantileverRatio(2.0, 1.0, diagonal), 1.003744278, 1.0e-8,
                          "two ShellTria3, cell 2 x 1, diagonal " + std::to_string(diagonal));
        runner.expectNear(cantileverRatio(10.0, 1.0, diagonal), 1.0, 1.0e-4,
                          "two ShellTria3, cell 10 x 1, diagonal " + std::to_string(diagonal));
    }
}
