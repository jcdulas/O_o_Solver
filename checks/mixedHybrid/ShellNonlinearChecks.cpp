#include "CheckRunner.hpp"
#include "ShellCheckElements.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementStresses.hpp"
#include "oos/elements/mixedHybrid/shell/ShellNonlinear.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <vector>

namespace mh = oos::elements::mixedHybrid;
using namespace oos::checks::shell;

namespace
{

// The elements of the nonlinear checks: a cylindrical quadrilateral and a curved triangle with an offset section
// (the reference's check uses an offset of 0.004 on a section of thickness 0.02, E = 2e5).
mh::ShellSection thinSection()
{
    return mh::ShellSection::uniform(mh::Section::isotropic(2.0e5, 0.3, 0.02, 5.0 / 6.0, 1.0, 0.0), {1.0, 0.0, 0.0});
}

TestElement nonlinearQuadrilateral(std::size_t degree)
{
    return makeElement(cylinderQuadrant(2.0, 1.5), degree, thinSection(), 0.004, {1.0, -1.0, -1.0, 1.0}, false);
}

TestElement nonlinearTriangle(std::size_t degree)
{
    const mh::SpaceVector p0{2.0, 0.0, 0.0};
    const mh::SpaceVector p1{0.0, 0.0, 2.0};
    const mh::SpaceVector p2{2.0, 1.5, 0.0};
    const std::array<mh::SpaceVector, 3> controls{mh::SpaceVector{2.0, 0.0, 2.0}, mh::scl3(mh::add3(p1, p2), 0.5),
                                                  mh::scl3(mh::add3(p2, p0), 0.5)};
    const TestSurface s = rationalTriangle({p0, p1, p2}, controls, {std::numbers::sqrt2 / 2.0, 1.0, 1.0});
    return makeElement(s, degree, thinSection(), 0.004, {1.0, 1.0, -1.0}, false);
}

double maxAbsVector(const mh::Vector& v)
{
    double top = 0.0;
    for (double x : v)
        top = std::fmax(top, std::fabs(x));
    return top;
}

} // namespace

// The linear element is the first variation of the nonlinear one at u = 0: the tangent at the reference is the linear
// stiffness and the internal forces vanish (document 34 of the reference, section 2.5).
// Reference: tests/tdnns.rs nonlinear_element_derivatives (its first part).
CHECK_CASE("shell: nonlinear tangent at the reference is the linear stiffness")
{
    for (std::size_t degree : {3u, 4u})
    {
        for (const TestElement& e : {nonlinearQuadrilateral(degree), nonlinearTriangle(degree)})
        {
            const mh::ShellReferenceElement& re = *e.re;
            const mh::ShellElementDefinition def = e.data.definition();
            const mh::ShellElementMatrices& m = e.data.matrices;
            const mh::NonlinearPreparation preparation = mh::prepareNonlinear(re, def);
            const mh::Vector zero(m.unknownCount, 0.0);
            const mh::NonlinearEvaluation at0 =
                mh::evaluateNonlinear(re, def, m, preparation, zero, nullptr, nullptr, false);
            const double scale = mh::maxAbs(m.stiffness);
            const double tangentError = mh::maxAbs(at0.tangent - m.stiffness) / scale;
            const std::string where =
                (re.cornerCount == 4 ? "quadrilateral" : "triangle") + std::string(" k ") + std::to_string(degree);
            runner.expect(tangentError < 1e-13, where + ": tangent " + std::to_string(tangentError));
            runner.expect(maxAbsVector(at0.forces) < 1e-13 * scale, where + ": internal forces at the reference");
            runner.expect(std::fabs(at0.energy) < 1e-20 * scale, where + ": energy at the reference");
        }
    }
}

// At a state of large displacement the internal forces are the derivative of the energy and the tangent the
// derivative of the internal forces (central differences of step 1e-6).
// Reference: tests/tdnns.rs nonlinear_element_derivatives (its second part).
CHECK_CASE("shell: nonlinear forces and tangent are the derivatives of the energy")
{
    for (const TestElement& e : {nonlinearQuadrilateral(3), nonlinearTriangle(3)})
    {
        const mh::ShellReferenceElement& re = *e.re;
        const mh::ShellElementDefinition def = e.data.definition();
        const mh::ShellElementMatrices& m = e.data.matrices;
        const std::size_t n = m.unknownCount;
        const mh::NonlinearPreparation preparation = mh::prepareNonlinear(re, def);
        Xorshift random;
        mh::Vector x(n);
        for (double& v : x)
            v = 0.2 * random.next();
        mh::Vector d(n);
        for (double& v : d)
            v = random.next();
        const double h = 1e-6;
        mh::Vector xp = x;
        mh::Vector xm = x;
        for (std::size_t i = 0; i < n; ++i)
        {
            xp[i] += h * d[i];
            xm[i] -= h * d[i];
        }
        const auto evaluate = [&](const mh::Vector& state)
        { return mh::evaluateNonlinear(re, def, m, preparation, state, nullptr, nullptr, false); };
        const mh::NonlinearEvaluation at = evaluate(x);
        const mh::NonlinearEvaluation plus = evaluate(xp);
        const mh::NonlinearEvaluation minus = evaluate(xm);
        const double rd = mh::dot(at.forces, d);
        const double de = std::fabs((plus.energy - minus.energy) / (2.0 * h) - rd) / std::fabs(rd);
        const mh::Vector kd = at.tangent * d;
        mh::Vector difference(n);
        for (std::size_t i = 0; i < n; ++i)
            difference[i] = (plus.forces[i] - minus.forces[i]) / (2.0 * h) - kd[i];
        const double dr = maxAbsVector(difference) / maxAbsVector(kd);
        const std::string where = re.cornerCount == 4 ? "quadrilateral" : "triangle";
        runner.expect(de < 1e-6, where + ": energy derivative " + std::to_string(de));
        runner.expect(dr < 1e-6, where + ": tangent " + std::to_string(dr));
        runner.expect(asymmetry(at.tangent) < 1e-14, where + ": tangent symmetric");
    }
}

// Accepting a state moves the start of the kink angles there without changing the energy of that state.
CHECK_CASE("shell: accepting a nonlinear state keeps its energy")
{
    for (const TestElement& e : {nonlinearQuadrilateral(3), nonlinearTriangle(3)})
    {
        const mh::ShellReferenceElement& re = *e.re;
        const mh::ShellElementDefinition def = e.data.definition();
        const mh::ShellElementMatrices& m = e.data.matrices;
        mh::NonlinearPreparation preparation = mh::prepareNonlinear(re, def);
        Xorshift random;
        mh::Vector x(m.unknownCount);
        for (double& v : x)
            v = 0.1 * random.next();
        const double before = mh::evaluateNonlinear(re, def, m, preparation, x, nullptr, nullptr, false).energy;
        mh::acceptNonlinearState(re, def, preparation, x);
        const double after = mh::evaluateNonlinear(re, def, m, preparation, x, nullptr, nullptr, false).energy;
        runner.expectNear(after / before, 1.0, 1e-10, re.cornerCount == 4 ? "quadrilateral" : "triangle");
    }
}

// The geometric stiffness of a uniform membrane force N on a flat affine element: for a transverse displacement w,
// wᵀ K_g w = ∫ N_IJ w_,I w_,J dA (the second variation of the Green strain; the moments of the preload are zero).
CHECK_CASE("shell: geometric stiffness of a uniform membrane force")
{
    const TestSurface s = affineTriangle(
        {mh::SpaceVector{0.0, 0.0, 0.0}, mh::SpaceVector{1.7, 0.3, 0.0}, mh::SpaceVector{0.4, 1.4, 0.0}});
    const TestElement e = makeElement(s, 4, isotropicSection(0.05), 0.0, {1.0, -1.0, 1.0}, false);
    const mh::ShellReferenceElement& re = *e.re;
    const mh::ShellElementDefinition def = e.data.definition();
    const mh::ShellElementMatrices& m = e.data.matrices;
    const double g[2][2] = {{1e-3, 2e-4}, {-1e-4, 5e-4}};
    const mh::Vector preload = unknownsOfField(
        e, [&](const mh::SpaceVector& x)
        { return mh::SpaceVector{g[0][0] * x[0] + g[0][1] * x[1], g[1][0] * x[0] + g[1][1] * x[1], 0.0}; },
        [](const mh::SpaceVector&) { return mh::SpaceVector{}; });
    const double kxx = 2e-3;
    const double kxy = -7e-4;
    const double kyy = 1.2e-3;
    const auto gradient = [&](const mh::SpaceVector& x)
    { return std::array<double, 2>{kxx * x[0] + kxy * x[1], kxy * x[0] + kyy * x[1]}; };
    const mh::Vector w = unknownsOfField(
        e, [&](const mh::SpaceVector& x)
        { return mh::SpaceVector{0.0, 0.0, 0.5 * (kxx * x[0] * x[0] + 2.0 * kxy * x[0] * x[1] + kyy * x[1] * x[1])}; },
        [&](const mh::SpaceVector& x)
        {
            const std::array<double, 2> d = gradient(x);
            return mh::SpaceVector{d[1], -d[0], 0.0};
        });
    const mh::Matrix kg = mh::elementGeometricStiffness(re, def, m, preload, nullptr, nullptr, nullptr);
    const double form = mh::dot(w, kg * w);
    // ∫ N : (∇w ⊗ ∇w) dA, N the (uniform) membrane force of the preload in the local frame
    const std::array<double, 8> r = mh::resultants(re, def, m, preload, nullptr, {0.0, 0.0}, -0.3, -0.4);
    double expected = 0.0;
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const mh::PointGeometry geometry = mh::volumeGeometry(def, re.h1Count, q);
        const std::array<double, 2> d = gradient(geometry.x);
        const double w1 = geometry.e[0][0] * d[0] + geometry.e[0][1] * d[1];
        const double w2 = geometry.e[1][0] * d[0] + geometry.e[1][1] * d[1];
        expected += re.weights[q] * geometry.j * (r[0] * w1 * w1 + r[1] * w2 * w2 + 2.0 * r[2] * w1 * w2);
    }
    runner.expectNear(form / expected, 1.0, 1e-9, "wᵀ Kg w");
    // the geometric stiffness of the membrane forces at the points (the reference's unused geometricMembrane) is the
    // same on this element, whose frame of constant states has φ = 0
    const std::vector<mh::Voigt3> forces = mh::membraneForces(re, def, m, preload, nullptr, {});
    const mh::Matrix km = mh::geometricMembrane(re, def, m, forces);
    runner.expectNear(mh::dot(w, km * w) / expected, 1.0, 1e-9, "wᵀ Kg w from the membrane forces");
}
