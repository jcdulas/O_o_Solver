// The checks of the ShellQuad8 and ShellTria6 kernel (MindlinShell): the numerical values of the tied triangle, the
// zero-energy modes, the consistency of the thermal loads and of the recovery with the stiffness.

#include "CheckRunner.hpp"
#include "ShellChecks.hpp"
#include "oos/elements/shells/MindlinShell.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

using namespace oos::checks::shells;
using oos::base::Vec3;
using oos::elements::MindlinShell;
using oos::elements::QuadraticTriangleShape;
using oos::elements::SerendipityQuadShape;
using oos::elements::ShellPlaceState;
using oos::elements::ShellSection;
using oos::elements::ShellSectionRate;
using oos::elements::ShellTemperatureChange;

namespace
{

constexpr double EXPANSION = 1.2e-5;

// The isotropic section of ShellChecks.hpp as a ShellSection, per unit thickness.
ShellSection isotropic(bool hasShearFlexibility = true)
{
    ShellSection section;
    section.hasMembrane = true;
    section.hasBending = true;
    section.hasShearFlexibility = hasShearFlexibility;
    section.hasThickness = true;
    section.thickness = THICKNESS;
    section.bendingRatio = 1.0;
    section.shearRatio = SHEAR_FACTOR;
    const double plane = YOUNG / (1.0 - POISSON * POISSON);
    for (oos::base::Mat3* modulus : {&section.membraneMatrix, &section.bendingMatrix})
    {
        (*modulus)(0, 0) = plane;
        (*modulus)(1, 1) = plane;
        (*modulus)(0, 1) = POISSON * plane;
        (*modulus)(1, 0) = POISSON * plane;
        (*modulus)(2, 2) = 0.5 * (1.0 - POISSON) * plane;
    }
    if (hasShearFlexibility)
        section.transverseShearStiffness.fill(SHEAR_FACTOR * YOUNG / (2.0 * (1.0 + POISSON)));
    section.membraneExpansion = {EXPANSION, EXPANSION, 0.0};
    section.bendingExpansion = {EXPANSION, EXPANSION, 0.0};
    return section;
}

// The height of the curved surface the nodes of the curved elements lie on, and its unit normal there: the
// directors of these elements, which are not the normals of their own quadratic surface.
double heightAt(double x, double y)
{
    return 0.3 * x * x - 0.2 * x * y + 0.15 * y * y;
}

Vec3 normalAt(double x, double y)
{
    const Vec3 normal{-0.6 * x + 0.2 * y, 0.2 * x - 0.3 * y, 1.0};
    return normal * (1.0 / oos::base::norm(normal));
}

template <std::size_t N>
struct Nodes
{
    std::array<Vec3, N> position{};
    std::array<Vec3, N> director{};
};

// The nodes at the plane positions `plane`, flat with the normal of the plane as director, or on the curved surface.
template <std::size_t N>
Nodes<N> nodesOf(const std::array<std::array<double, 2>, N>& plane, bool isCurved)
{
    Nodes<N> nodes;
    for (std::size_t a = 0; a < N; ++a)
    {
        const double x = plane[a][0];
        const double y = plane[a][1];
        nodes.position[a] = Vec3{x, y, isCurved ? heightAt(x, y) : 0.0};
        nodes.director[a] = isCurved ? normalAt(x, y) : Vec3{0.0, 0.0, 1.0};
    }
    return nodes;
}

// A scalene triangle of straight sides and a quadrilateral of straight sides that is not a parallelogram, their edge
// nodes at the middle of their sides.
constexpr std::array<std::array<double, 2>, 6> TRIANGLE{
    {{0.0, 0.0}, {1.0, 0.1}, {0.2, 0.9}, {0.5, 0.05}, {0.6, 0.5}, {0.1, 0.45}}};
constexpr std::array<std::array<double, 2>, 8> QUADRILATERAL{
    {{0.0, 0.0}, {1.0, 0.1}, {1.2, 0.9}, {-0.1, 0.7}, {0.5, 0.05}, {1.1, 0.5}, {0.55, 0.8}, {-0.05, 0.35}}};

template <typename Shape>
MindlinShell<Shape> shellOf(const Nodes<Shape::NODES>& nodes, const ShellSection& section,
                            double drillingFactor = DRILLING_FACTOR)
{
    std::array<double, Shape::CORNERS> thickness{};
    thickness.fill(THICKNESS);
    return MindlinShell<Shape>(nodes.position, nodes.director, section, thickness, drillingFactor);
}

template <std::size_t DOFS>
std::vector<double> eigenvaluesOf(const oos::base::MatN<DOFS>& k)
{
    oos::elements::ScalarMatrix<double, DOFS> matrix;
    for (std::size_t i = 0; i < DOFS; ++i)
        for (std::size_t j = 0; j < DOFS; ++j)
            matrix(i, j) = k(i, j);
    return sortedEigenvalues<DOFS>(matrix);
}

template <std::size_t N>
std::array<std::array<double, 3>, N> positionsOf(const Nodes<N>& nodes)
{
    std::array<std::array<double, 3>, N> position{};
    for (std::size_t a = 0; a < N; ++a)
        position[a] = {nodes.position[a][0], nodes.position[a][1], nodes.position[a][2]};
    return position;
}

template <std::size_t DOFS>
double largestOf(const std::array<double, DOFS>& values)
{
    double largest = 0.0;
    for (const double value : values)
        largest = std::fmax(largest, std::fabs(value));
    return largest;
}

// The thirty non-zero eigenvalues of the stiffness of the ShellTria6 TRIANGLE with the section of ShellChecks.hpp,
// flat then on the curved surface with its normals as directors: the values of an independent program of the
// formulation (tied transverse shears and membrane strains, seven-point rule, drilling term).
constexpr std::array<double, 30> FLAT_TRIANGLE_EIGENVALUES{
    1.0302017488e-04, 1.0302355163e-04, 1.2677982131e-04, 3.2935952021e-04, 3.2940027670e-04, 3.6463043352e-03,
    3.7386196036e-03, 4.1451505498e-03, 5.1601610265e-03, 6.3515575223e-03, 2.3295301698e-02, 2.7112900382e-02,
    3.9404288983e-02, 3.2238012652e-01, 3.4791540961e-01, 1.1246721615e+00, 6.7170271604e+00, 7.5442365300e+00,
    8.9151509545e+00, 1.2365821688e+01, 2.0628986675e+01, 2.5071946027e+01, 4.0427067946e+01, 4.4496940019e+01,
    4.9248788252e+01, 6.0887553700e+01, 7.8620768740e+01, 1.2420085128e+02, 1.4206316405e+02, 1.9217600848e+02};
constexpr std::array<double, 30> CURVED_TRIANGLE_EIGENVALUES{
    1.0466005218e-04, 1.1073214702e-04, 1.2200014831e-04, 3.3835105705e-04, 3.4996839290e-04, 2.8265818096e-03,
    3.3146152445e-03, 4.5014643338e-03, 5.2545851524e-03, 7.1340433407e-03, 2.2332154047e-02, 2.6686492958e-02,
    3.9923731347e-02, 3.3473684575e-01, 3.5071952303e-01, 1.2696040811e+00, 6.3466570477e+00, 7.2568882960e+00,
    8.7748897042e+00, 1.2488558896e+01, 1.9306719051e+01, 2.6228696892e+01, 3.8923552657e+01, 4.3834977468e+01,
    4.9873838588e+01, 6.3848962229e+01, 7.9281887474e+01, 1.2544212292e+02, 1.4371456798e+02, 1.9738813743e+02};

// A rigid motion of the element gives no force, and the stiffness has `zeroModes` zero eigenvalues.
template <typename Shape>
void expectRigidMotions(oos::checks::CheckRunner& runner, const Nodes<Shape::NODES>& nodes, std::size_t zeroModes,
                        const std::string& what)
{
    const oos::base::MatN<6 * Shape::NODES> k = shellOf<Shape>(nodes, isotropic()).stiffness();
    const std::vector<double> values = eigenvaluesOf(k);
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        const bool isZero = std::fabs(values[i]) < 1.0e-10 * values.back();
        runner.expect(isZero == (i < zeroModes), what + ": eigenvalue " + std::to_string(i + 1) + " is " +
                                                     std::to_string(values[i] / values.back()) + " of the largest");
    }
    const std::array<double, 6 * Shape::NODES> q =
        rigidMotion<Shape::NODES>(positionsOf(nodes), {0.1, -0.2, 0.3}, {0.3, -0.2, 0.5});
    double stiffest = 0.0;
    for (std::size_t i = 0; i < 6 * Shape::NODES; ++i)
        stiffest = std::fmax(stiffest, k(i, i));
    runner.expect(largestOf(k * q) < 1.0e-12 * stiffest, what + ": a rigid motion gives a force");
}

// The loads of a uniform temperature rise are the forces K q of the homothety q = α ΔT x, and the element in that
// homothety has no force, moment nor shear force at any of its places.
template <typename Shape>
void expectFreeExpansion(oos::checks::CheckRunner& runner, const Nodes<Shape::NODES>& nodes, const std::string& what)
{
    constexpr std::size_t DOFS = 6 * Shape::NODES;
    constexpr double RISE = 80.0;
    const MindlinShell<Shape> shell = shellOf<Shape>(nodes, isotropic());
    ShellTemperatureChange temperature;
    temperature.deltaT = RISE;
    std::array<double, DOFS> q{};
    for (std::size_t a = 0; a < Shape::NODES; ++a)
        for (std::size_t k = 0; k < 3; ++k)
            q[6 * a + k] = EXPANSION * RISE * nodes.position[a][k];
    const std::array<double, DOFS> loads = shell.thermalLoads(temperature);
    const std::array<double, DOFS> forces = shell.stiffness() * q;
    double difference = 0.0;
    for (std::size_t i = 0; i < DOFS; ++i)
        difference = std::fmax(difference, std::fabs(loads[i] - forces[i]));
    runner.expect(difference < 1.0e-10 * largestOf(loads),
                  what + ": the thermal loads are off the forces of the homothety by " +
                      std::to_string(difference / largestOf(loads)));
    std::array<std::int32_t, Shape::NODES> ids{};
    const double held = YOUNG * THICKNESS * EXPANSION * RISE; // the force of the expansion held
    for (const ShellPlaceState& state : shell.placeStates(q, ids, temperature))
    {
        runner.expect(largestOf(state.membraneForce) < 1.0e-10 * held, what + ": a membrane force is left");
        runner.expect(largestOf(state.moment) < 1.0e-10 * held * THICKNESS, what + ": a moment is left");
        runner.expect(largestOf(state.shearForce) < 1.0e-10 * held, what + ": a shear force is left");
    }
}

// The local DOFs of the plate field w = (kx x² + ky y² + 2 kxy x y) / 2 with the rotations of its normals
// (θx = ∂w/∂y, θy = −∂w/∂x) and of the plane field of membrane strains (ex, ey, gxy).
template <std::size_t N>
std::array<double, 6 * N> uniformStates(const Nodes<N>& nodes, const std::array<double, 3>& strain,
                                        const std::array<double, 3>& curvature)
{
    std::array<double, 6 * N> q{};
    for (std::size_t a = 0; a < N; ++a)
    {
        const double x = nodes.position[a][0];
        const double y = nodes.position[a][1];
        q[6 * a + 0] = strain[0] * x + 0.5 * strain[2] * y;
        q[6 * a + 1] = 0.5 * strain[2] * x + strain[1] * y;
        q[6 * a + 2] = 0.5 * (curvature[0] * x * x + curvature[1] * y * y + 2.0 * curvature[2] * x * y);
        q[6 * a + 3] = curvature[1] * y + curvature[2] * x;
        q[6 * a + 4] = -(curvature[0] * x + curvature[2] * y);
    }
    return q;
}

// Every place of the element recovers the uniform membrane strains and curvatures, without shear force.
template <typename Shape>
void expectUniformStates(oos::checks::CheckRunner& runner, const Nodes<Shape::NODES>& nodes, bool withCurvature,
                         const std::string& what)
{
    const std::array<double, 3> strain{1.0e-3, -0.4e-3, 0.6e-3};
    const std::array<double, 3> curvature =
        withCurvature ? std::array<double, 3>{1.0e-2, -0.5e-2, 0.7e-2} : std::array<double, 3>{};
    const MindlinShell<Shape> shell = shellOf<Shape>(nodes, isotropic());
    std::array<std::int32_t, Shape::NODES> ids{};
    const double shearScale = YOUNG * THICKNESS * 1.0e-2;
    for (const ShellPlaceState& state : shell.placeStates(uniformStates(nodes, strain, curvature), ids))
    {
        for (std::size_t c = 0; c < 3; ++c)
        {
            runner.expectNear(state.membraneStrain[c] / 1.0e-3, strain[c] / 1.0e-3, 1.0e-10, what + ": strain");
            // The shell curvature (κx, κy, 2κxy) of the field.
            const double expected = (c == 2 ? 2.0 : 1.0) * curvature[c];
            runner.expectNear(state.curvature[c] / 1.0e-2, expected / 1.0e-2, 1.0e-10, what + ": curvature");
        }
        runner.expect(largestOf(state.shearForce) < 1.0e-10 * shearScale, what + ": a shear force is left");
    }
}

// The closed-form rate of the stiffness along a rate of the section is the derivative of the stiffness: every part
// of it is linear in the matrices of the section (the drilling term in the shear modulus of the membrane, the shear
// of a section without shear flexibility in the larger of D11 and D22), so a central difference gives it to round-off.
template <typename Shape>
void expectSectionRate(oos::checks::CheckRunner& runner, const Nodes<Shape::NODES>& nodes, bool hasShearFlexibility,
                       const std::string& what)
{
    constexpr std::size_t DOFS = 6 * Shape::NODES;
    constexpr double STEP = 1.0e-3;
    ShellSection section = isotropic(hasShearFlexibility);
    section.bendingMatrix(0, 0) *= 1.2; // D11 the larger, away from D22
    ShellSectionRate rate;
    rate.movesMembrane = true;
    rate.movesBending = true;
    rate.movesShear = hasShearFlexibility;
    const std::array<double, 6> membrane{310.0, 40.0, -15.0, 220.0, 25.0, 130.0}; // 11, 12, 13, 22, 23, 33
    const std::array<double, 6> bending{-90.0, 30.0, 12.0, 150.0, -20.0, 60.0};
    constexpr std::array<std::array<std::size_t, 2>, 6> TERMS{{{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}}};
    for (std::size_t k = 0; k < 6; ++k)
    {
        const std::size_t i = TERMS[k][0];
        const std::size_t j = TERMS[k][1];
        rate.membraneMatrix(i, j) = rate.membraneMatrix(j, i) = membrane[k];
        rate.bendingMatrix(i, j) = rate.bendingMatrix(j, i) = bending[k];
    }
    if (hasShearFlexibility)
        rate.transverseShearStiffness = {70.0, -40.0};
    const auto stiffnessAt = [&](double step)
    {
        ShellSection moved = section;
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
            {
                moved.membraneMatrix(i, j) += step * rate.membraneMatrix(i, j);
                moved.bendingMatrix(i, j) += step * rate.bendingMatrix(i, j);
            }
        for (std::size_t k = 0; k < 2; ++k)
            moved.transverseShearStiffness[k] += step * rate.transverseShearStiffness[k];
        return shellOf<Shape>(nodes, moved).stiffness();
    };
    const oos::base::MatN<DOFS> above = stiffnessAt(STEP);
    const oos::base::MatN<DOFS> below = stiffnessAt(-STEP);
    const oos::base::MatN<DOFS> closed = shellOf<Shape>(nodes, section).stiffnessSectionRate(rate);
    double largest = 0.0;
    double difference = 0.0;
    for (std::size_t i = 0; i < DOFS; ++i)
        for (std::size_t j = 0; j < DOFS; ++j)
        {
            largest = std::fmax(largest, std::fabs(closed(i, j)));
            difference = std::fmax(difference, std::fabs(closed(i, j) - (above(i, j) - below(i, j)) / (2.0 * STEP)));
        }
    runner.expect(difference < 1.0e-9 * largest, what + ": the section rate is off the derivative by " +
                                                     std::to_string(difference / largest));
}

} // namespace

CHECK_CASE("shellTria6: the eigenvalues of the stiffness are those of the formulation, flat and curved")
{
    constexpr double PRINTED = 5.0e-9;
    constexpr std::size_t DOFS = 36;
    for (const bool isCurved : {false, true})
    {
        const std::array<double, 30>& expected = isCurved ? CURVED_TRIANGLE_EIGENVALUES : FLAT_TRIANGLE_EIGENVALUES;
        const MindlinShell<QuadraticTriangleShape> shell =
            shellOf<QuadraticTriangleShape>(nodesOf<6>(TRIANGLE, isCurved), isotropic());
        const std::vector<double> values = eigenvaluesOf<DOFS>(shell.stiffness());
        for (std::size_t k = 0; k < 6; ++k)
            runner.expect(std::fabs(values[k]) < 1.0e-10 * values.back(),
                          "zero-energy mode " + std::to_string(k + 1) + " has the eigenvalue " +
                              std::to_string(values[k]));
        for (std::size_t k = 0; k < expected.size(); ++k)
            runner.expect(std::fabs(values[6 + k] / expected[k] - 1.0) < PRINTED,
                          "eigenvalue " + std::to_string(k + 1) + " is " + std::to_string(values[6 + k]) +
                              ", expected " + std::to_string(expected[k]));
    }
}

CHECK_CASE("shellTria6 and shellQuad8: the rigid motions are the zero-energy modes")
{
    // The triangle: its six rigid motions. The quadrilateral: these and the two modes its reduced rule leaves an
    // element alone (one of its membrane, one of its bending), which two elements do not share.
    for (const bool isCurved : {false, true})
    {
        const std::string shape = isCurved ? "curved" : "flat";
        expectRigidMotions<QuadraticTriangleShape>(runner, nodesOf<6>(TRIANGLE, isCurved), 6, shape + " ShellTria6");
        expectRigidMotions<SerendipityQuadShape>(runner, nodesOf<8>(QUADRILATERAL, isCurved), 8,
                                                 shape + " ShellQuad8");
    }
}

CHECK_CASE("shellTria6 and shellQuad8: a free thermal expansion is the homothety, without force in the element")
{
    for (const bool isCurved : {false, true})
    {
        const std::string shape = isCurved ? "curved" : "flat";
        expectFreeExpansion<QuadraticTriangleShape>(runner, nodesOf<6>(TRIANGLE, isCurved), shape + " ShellTria6");
        expectFreeExpansion<SerendipityQuadShape>(runner, nodesOf<8>(QUADRILATERAL, isCurved), shape + " ShellQuad8");
    }
}

CHECK_CASE("shellTria6 and shellQuad8: uniform strains and curvatures are recovered at every place")
{
    // The triangle of straight sides: its tied strains are those of the field. The quadrilateral of straight sides:
    // the membrane strains; the curvatures on a parallelogram (the eight nodes of another quadrilateral do not hold
    // the quadratic deflection).
    expectUniformStates<QuadraticTriangleShape>(runner, nodesOf<6>(TRIANGLE, false), true, "ShellTria6");
    expectUniformStates<SerendipityQuadShape>(runner, nodesOf<8>(QUADRILATERAL, false), false, "ShellQuad8");
    constexpr std::array<std::array<double, 2>, 8> PARALLELOGRAM{
        {{0.0, 0.0}, {1.0, 0.1}, {1.3, 0.9}, {0.3, 0.8}, {0.5, 0.05}, {1.15, 0.5}, {0.8, 0.85}, {0.15, 0.4}}};
    expectUniformStates<SerendipityQuadShape>(runner, nodesOf<8>(PARALLELOGRAM, false), true,
                                              "ShellQuad8 parallelogram");
}

CHECK_CASE("shellTria6 and shellQuad8: the section rate of the stiffness is its derivative")
{
    for (const bool hasShearFlexibility : {true, false})
    {
        const std::string shear = hasShearFlexibility ? " with" : " without";
        expectSectionRate<QuadraticTriangleShape>(runner, nodesOf<6>(TRIANGLE, true), hasShearFlexibility,
                                                  "ShellTria6" + shear + " shear flexibility");
        expectSectionRate<SerendipityQuadShape>(runner, nodesOf<8>(QUADRILATERAL, true), hasShearFlexibility,
                                                "ShellQuad8" + shear + " shear flexibility");
    }
}
