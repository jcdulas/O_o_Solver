#include "CheckRunner.hpp"
#include "ShellCheckElements.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementResults.hpp"
#include "oos/elements/mixedHybrid/shell/ShellElementStresses.hpp"
#include "oos/elements/mixedHybrid/shell/ShellNonlinear.hpp"
#include "oos/elements/mixedHybrid/shell/ShellPointLaw.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <vector>

namespace mh = oos::elements::mixedHybrid;
using namespace oos::checks::shell;

namespace
{

constexpr std::size_t DEGREE = 4;

// The test elements: flat and distorted, warped, curved (rational patches), quadrilaterals and triangles.
// (`edgeDegrees`: the degrees of the edges of a transition element; empty: the degree of the element.)
using EdgeDegrees = std::vector<std::size_t>;

TestElement flatQuadrilateral(std::size_t degree = DEGREE, bool withMass = false, const EdgeDegrees& edgeDegrees = {})
{
    const TestSurface s = bilinearQuadrilateral({mh::SpaceVector{0.0, 0.0, 0.0}, mh::SpaceVector{2.2, 0.1, 0.0},
                                                 mh::SpaceVector{1.9, 1.7, 0.0}, mh::SpaceVector{-0.2, 1.2, 0.0}});
    return makeElement(s, degree, isotropicSection(0.05), 0.0, {1.0, -1.0, 1.0, -1.0}, withMass, false, edgeDegrees);
}

TestElement warpedQuadrilateral(std::size_t degree = DEGREE, bool withMass = false,
                                const EdgeDegrees& edgeDegrees = {})
{
    const TestSurface s = bilinearQuadrilateral({mh::SpaceVector{0.0, 0.0, 0.0}, mh::SpaceVector{2.2, 0.1, 0.0},
                                                 mh::SpaceVector{1.9, 1.7, 0.25}, mh::SpaceVector{-0.2, 1.2, 0.0}});
    return makeElement(s, degree, laminateSection({0.6, 0.8, 0.0}), 0.01, {1.0, -1.0, -1.0, 1.0}, withMass, false,
                       edgeDegrees);
}

TestElement cylindricalQuadrilateral(std::size_t degree = DEGREE, bool withMass = false,
                                     const EdgeDegrees& edgeDegrees = {})
{
    return makeElement(cylinderQuadrant(2.0, 1.5), degree, isotropicSection(0.02), 0.0, {1.0, 1.0, -1.0, -1.0},
                       withMass, false, edgeDegrees);
}

TestElement flatTriangle(std::size_t degree = DEGREE, bool withMass = false, const EdgeDegrees& edgeDegrees = {})
{
    const TestSurface s = affineTriangle(
        {mh::SpaceVector{0.0, 0.0, 0.0}, mh::SpaceVector{1.7, 0.3, 0.0}, mh::SpaceVector{0.4, 1.4, 0.0}});
    return makeElement(s, degree, isotropicSection(0.05), 0.0, {1.0, -1.0, 1.0}, withMass, false, edgeDegrees);
}

TestElement curvedTriangle(std::size_t degree = DEGREE, bool withMass = false)
{
    // edge 0 a quarter circle of radius 2 about the y axis, edges 1 and 2 straight
    const mh::SpaceVector p0{2.0, 0.0, 0.0};
    const mh::SpaceVector p1{0.0, 0.0, 2.0};
    const mh::SpaceVector p2{2.0, 1.5, 0.0};
    const std::array<mh::SpaceVector, 3> controls{mh::SpaceVector{2.0, 0.0, 2.0}, mh::scl3(mh::add3(p1, p2), 0.5),
                                                  mh::scl3(mh::add3(p2, p0), 0.5)};
    const TestSurface s = rationalTriangle({p0, p1, p2}, controls, {std::numbers::sqrt2 / 2.0, 1.0, 1.0});
    return makeElement(s, degree, laminateSection({0.0, 1.0, 0.0}), -0.005, {-1.0, 1.0, -1.0}, withMass);
}

struct NamedElement
{
    std::string name;
    TestElement element;
};

std::vector<NamedElement> allElements(std::size_t degree = DEGREE, bool withMass = false)
{
    std::vector<NamedElement> out;
    out.push_back({"flat quadrilateral", flatQuadrilateral(degree, withMass)});
    out.push_back({"warped quadrilateral", warpedQuadrilateral(degree, withMass)});
    out.push_back({"cylindrical quadrilateral", cylindricalQuadrilateral(degree, withMass)});
    out.push_back({"flat triangle", flatTriangle(degree, withMass)});
    out.push_back({"curved triangle", curvedTriangle(degree, withMass)});
    return out;
}

// Transition elements (ShellSpaces.hpp): elements of degree 3 and 4 some of whose edges, or all, have a lower degree.
std::vector<NamedElement> transitionElements()
{
    std::vector<NamedElement> out;
    out.push_back({"flat quadrilateral 3, edges 2 3 3 2", flatQuadrilateral(3, false, {2, 3, 3, 2})});
    out.push_back({"flat quadrilateral 3, edges 2 2 2 2", flatQuadrilateral(3, false, {2, 2, 2, 2})});
    out.push_back({"flat quadrilateral 4, edges 3 2 4 3", flatQuadrilateral(4, false, {3, 2, 4, 3})});
    out.push_back({"warped quadrilateral 3, edges 3 2 3 3", warpedQuadrilateral(3, false, {3, 2, 3, 3})});
    out.push_back({"cylindrical quadrilateral 4, edges 3 4 3 4", cylindricalQuadrilateral(4, false, {3, 4, 3, 4})});
    out.push_back({"flat triangle 3, edges 2 3 2", flatTriangle(3, false, {2, 3, 2})});
    const TestSurface cylinder = cylinderQuadrant(2.0, 1.5);
    out.push_back({"thin curved quadrilateral 4, edges 2 4 2 3",
                   makeElement(cylinder, 4, isotropicSection(0.02), 0.0, {1.0, 1.0, -1.0, -1.0}, false, false,
                               {2, 4, 2, 3})});
    const TestSurface warped =
        bilinearQuadrilateral({mh::SpaceVector{0.0, 0.0, 0.0}, mh::SpaceVector{2.2, 0.1, 0.0},
                               mh::SpaceVector{1.9, 1.7, 0.25}, mh::SpaceVector{-0.2, 1.2, 0.0}});
    out.push_back({"warped quadrilateral 3, edges 2 2 2 2",
                   makeElement(warped, 3, laminateSection({0.6, 0.8, 0.0}), 0.01, {1.0, -1.0, -1.0, 1.0}, false, false,
                               {2, 2, 2, 2})});
    out.push_back({"flat triangle 4, edges 2 2 3",
                   makeElement(affineTriangle({mh::SpaceVector{0.0, 0.0, 0.0}, mh::SpaceVector{1.7, 0.3, 0.0},
                                               mh::SpaceVector{0.4, 1.4, 0.0}}),
                               4, isotropicSection(0.05), 0.0, {1.0, -1.0, 1.0}, false, false, {2, 2, 3})});
    return out;
}

double maxDiagonal(const mh::Matrix& k)
{
    double top = 0.0;
    for (std::size_t i = 0; i < k.rows(); ++i)
        top = std::fmax(top, k(i, i));
    return top;
}

double maxAbsVector(const mh::Vector& v)
{
    double top = 0.0;
    for (double x : v)
        top = std::fmax(top, std::fabs(x));
    return top;
}

} // namespace

// K r = 0 for the six rigid motions of one element (document 34 of the reference, section 4.1: the rational map makes
// ω × X exact; section 4.2: the rotation about a conic edge is exact for k ≥ 3), relative to the largest diagonal
// term: flat, warped, curved by a rational patch, distorted.
CHECK_CASE("shell: rigid motions of one element have no energy")
{
    for (std::size_t degree : {3u, 4u})
    {
        for (const NamedElement& named : allElements(degree))
        {
            const TestElement& e = named.element;
            const mh::Matrix& k = e.data.matrices.stiffness;
            const std::array<mh::Vector, 6> rigid = mh::rigidMotionUnknowns(*e.re, e.data.definition());
            const double kd = maxDiagonal(k);
            for (std::size_t m = 0; m < 6; ++m)
            {
                const mh::Vector kr = k * rigid[m];
                const double err = maxAbsVector(kr) / (kd * maxAbsVector(rigid[m]));
                const double energy = mh::dot(rigid[m], kr) / (kd * mh::dot(rigid[m], rigid[m]));
                runner.expect(err < 1e-10 && std::fabs(energy) < 1e-10, named.name + " k " + std::to_string(degree) +
                                                                            " mode " + std::to_string(m) + ": " +
                                                                            std::to_string(err));
            }
        }
    }
}

// The same for transition elements: the rotation about an edge of a lower degree still holds that of a rigid motion
// (a constant on a straight edge, of degree 2 on a conic one, which an edge of degree 3 holds).
CHECK_CASE("shell: rigid motions of a transition element have no energy")
{
    for (const NamedElement& named : transitionElements())
    {
        const TestElement& e = named.element;
        const mh::Matrix& k = e.data.matrices.stiffness;
        const std::array<mh::Vector, 6> rigid = mh::rigidMotionUnknowns(*e.re, e.data.definition());
        const double kd = maxDiagonal(k);
        for (std::size_t m = 0; m < 6; ++m)
        {
            const mh::Vector kr = k * rigid[m];
            const double err = maxAbsVector(kr) / (kd * maxAbsVector(rigid[m]));
            runner.expect(err < 1e-10, named.name + " mode " + std::to_string(m) + ": " + std::to_string(err));
        }
    }
}

// A transition element has the six rigid modes and no other mode without energy: the moments it keeps hold every
// other motion of its unknowns.
CHECK_CASE("shell: free transition element has exactly the rigid modes")
{
    for (const NamedElement& named : transitionElements())
    {
        const std::vector<double> values = sortedEigenvalues(named.element.data.matrices.stiffness);
        const double top = values.back();
        std::size_t zero = 0;
        for (double v : values)
            if (std::fabs(v) < 1e-11 * top)
                ++zero;
        runner.expect(zero == 6, named.name + ": " + std::to_string(zero) + " zero eigenvalues");
        runner.expect(values.front() > -1e-11 * top, named.name + ": no negative eigenvalue");
    }
}

// The membrane strain of a displacement that a transition element can have is interpolated exactly on an affine
// element: a quadratic displacement in the plane gives the forces N = A ε, ε its exact (linear) strain, at every point,
// whatever the degrees of the edges (whose displacement is of degree 2: the traces of a quadratic are within them).
CHECK_CASE("shell: the linear strain of a quadratic displacement of a transition element is exact")
{
    const auto parallelogram = [](std::size_t degree, const EdgeDegrees& edges)
    {
        return makeElement(bilinearQuadrilateral({mh::SpaceVector{0, 0, 0}, mh::SpaceVector{2, 0.3, 0},
                                                  mh::SpaceVector{2.5, 1.3, 0}, mh::SpaceVector{0.5, 1.0, 0}}),
                           degree, isotropicSection(0.05), 0.0, {1, -1, 1, -1}, false, false, edges);
    };
    const auto triangle = [](std::size_t degree, const EdgeDegrees& edges)
    {
        return makeElement(affineTriangle({mh::SpaceVector{0.0, 0.0, 0.0}, mh::SpaceVector{1.7, 0.3, 0.0},
                                           mh::SpaceVector{0.4, 1.4, 0.0}}),
                           degree, isotropicSection(0.05), 0.0, {1.0, -1.0, 1.0}, false, false, edges);
    };
    std::vector<NamedElement> elements;
    elements.push_back({"parallelogram 3, edges 2 2 2 2", parallelogram(3, {2, 2, 2, 2})});
    elements.push_back({"parallelogram 4, edges 2 3 2 4", parallelogram(4, {2, 3, 2, 4})});
    elements.push_back({"triangle 3, edges 2 2 2", triangle(3, {2, 2, 2})});
    elements.push_back({"triangle 4, edges 2 3 2", triangle(4, {2, 3, 2})});
    for (const NamedElement& named : elements)
    {
        const TestElement& e = named.element;
        const mh::ShellReferenceElement& re = *e.re;
        const mh::ShellElementDefinition def = e.data.definition();
        const mh::Section& section = e.data.section.section();
        // u_x = a x² + b x y + c y², u_y = d x² + f x y + g y²: the strain is linear in the position
        const double a = 1e-3, b = -4e-4, c = 2e-4, d = 3e-4, f = -2e-4, g = 5e-4;
        const mh::Vector x = unknownsOfField(
            e,
            [&](const mh::SpaceVector& p)
            {
                return mh::SpaceVector{a * p[0] * p[0] + b * p[0] * p[1] + c * p[1] * p[1],
                                       d * p[0] * p[0] + f * p[0] * p[1] + g * p[1] * p[1], 0.0};
            },
            [](const mh::SpaceVector&) { return mh::SpaceVector{}; });
        double worst = 0.0;
        double scale = 0.0;
        for (const mh::ReferencePoint& point : {mh::ReferencePoint{-0.5, -0.4}, mh::ReferencePoint{0.2, -0.3},
                                                mh::ReferencePoint{-0.3, 0.3}})
        {
            const mh::PointGeometry geometry = mh::geometryAt(re, def, point[0], point[1]);
            const double px = geometry.x[0];
            const double py = geometry.x[1];
            const double gradient[2][2] = {{2 * a * px + b * py, b * px + 2 * c * py},
                                           {2 * d * px + f * py, f * px + 2 * g * py}};
            const double sym[2][2] = {{gradient[0][0], 0.5 * (gradient[0][1] + gradient[1][0])},
                                      {0.5 * (gradient[0][1] + gradient[1][0]), gradient[1][1]}};
            const auto component = [&](std::size_t i, std::size_t j)
            {
                double sum = 0.0;
                for (std::size_t m = 0; m < 2; ++m)
                    for (std::size_t n = 0; n < 2; ++n)
                        sum += geometry.e[i][m] * sym[m][n] * geometry.e[j][n];
                return sum;
            };
            const mh::Voigt3 strain{component(0, 0), component(1, 1), 2.0 * component(0, 1)};
            const std::array<double, 8> forces =
                mh::resultants(re, def, e.data.matrices, x, nullptr, {0.0, 0.0}, point[0], point[1]);
            for (std::size_t i = 0; i < 3; ++i)
            {
                double n = 0.0;
                for (std::size_t j = 0; j < 3; ++j)
                    n += section.a[i][j] * strain[j];
                worst = std::fmax(worst, std::fabs(forces[i] - n));
                scale = std::fmax(scale, std::fabs(n));
            }
        }
        runner.expect(worst < 1e-9 * scale, named.name + ": " + std::to_string(worst / scale));
    }
}

// The stiffness of a free element has exactly six zero eigenvalues (the rigid motions), counted by the Jacobi method.
CHECK_CASE("shell: free element has exactly the rigid modes")
{
    for (const NamedElement& named : allElements())
    {
        const std::vector<double> values = sortedEigenvalues(named.element.data.matrices.stiffness);
        const double top = values.back();
        std::size_t zero = 0;
        for (double v : values)
            if (std::fabs(v) < 1e-11 * top)
                ++zero;
        runner.expect(zero == 6, named.name + ": " + std::to_string(zero) + " zero eigenvalues");
        runner.expect(values[6] > 1e-9 * top,
                      named.name + ": smallest elastic eigenvalue " + std::to_string(values[6] / top));
        runner.expect(values.front() > -1e-11 * top, named.name + ": no negative eigenvalue");
    }
}

// The stiffness, the mass and the geometric stiffness are symmetric; the mass is positive definite on the
// displacement of a translation and gives the mass of the element.
CHECK_CASE("shell: symmetry of the stiffness, mass and geometric stiffness")
{
    for (const NamedElement& named : allElements(DEGREE, true))
    {
        const TestElement& e = named.element;
        const mh::ShellElementMatrices& m = e.data.matrices;
        runner.expect(asymmetry(m.stiffness) < 1e-14, named.name + ": K symmetric");
        runner.expect(m.mass.has_value() && asymmetry(*m.mass) < 1e-13, named.name + ": M symmetric");
        // a preload: a membrane stretch with some bending (no interior unknowns)
        mh::Vector preload(m.unknownCount, 0.0);
        Xorshift random;
        for (std::size_t i = 0; i < e.re->cornerCount; ++i)
            for (std::size_t c = 0; c < 3; ++c)
                preload[3 * i + c] = 1e-3 * random.next();
        const mh::Matrix kg =
            mh::elementGeometricStiffness(*e.re, e.data.definition(), m, preload, nullptr, nullptr, nullptr);
        runner.expect(asymmetry(kg) < 1e-14, named.name + ": Kg symmetric");
        // the mass of a translation: uᵀ M u = ∫ I0 dA
        const std::array<mh::Vector, 6> rigid = mh::rigidMotionUnknowns(*e.re, e.data.definition());
        const mh::Vector mu = *m.mass * rigid[0];
        double area = 0.0;
        double expected = 0.0;
        const mh::ShellElementDefinition def = e.data.definition();
        for (std::size_t q = 0; q < e.re->points.size(); ++q)
        {
            const mh::PointGeometry g = mh::volumeGeometry(def, e.re->h1Count, q);
            const mh::SectionPoint sp =
                mh::sectionAtGeometry(e.data.section, e.data.offset, e.re->points[q][0], e.re->points[q][1], g);
            area += e.re->weights[q] * g.j;
            expected += e.re->weights[q] * g.j * sp.mass[0];
        }
        runner.expectNear(mh::dot(rigid[0], mu) / expected, 1.0, 1e-12, named.name + ": mass of a translation");
        runner.expect(area > 0.0, named.name + ": area");
    }
}

// The constant states of a flat distorted element (document 34 of the reference, section 4.3): a uniform membrane
// strain gives N = A ε and no moment, a uniform curvature gives M = D κ and no membrane force, at every point.
CHECK_CASE("shell: the mass built alone has the bits of the mass built with the element")
{
    for (const NamedElement& named : allElements(DEGREE, true))
    {
        const TestElement& e = named.element;
        const mh::Matrix& together = *e.data.matrices.mass;
        const mh::Matrix alone = mh::buildShellMass(*e.re, e.data.definition());
        bool isSame = alone.rows() == together.rows() && alone.columns() == together.columns();
        for (std::size_t i = 0; isSame && i < together.rows(); ++i)
            for (std::size_t j = 0; isSame && j < together.columns(); ++j)
                isSame = alone(i, j) == together(i, j);
        runner.expect(isSame, named.name);
    }
}

CHECK_CASE("shell: the bases kept at the centre and the corners are the values of their polynomials")
{
    for (const std::size_t cornerCount : {std::size_t{3}, std::size_t{4}})
    {
        const mh::ShellReferenceElement& re =
            mh::shellReferenceElement(cornerCount, DEGREE, mh::shellPointsPerDirection(DEGREE));
        const std::string shape = cornerCount == 3 ? "triangle" : "quadrilateral";
        runner.expect(re.outputPoints.size() == cornerCount + 1, shape + ": the centre and the corners");
        runner.expect(re.valuesAt(0.123, -0.456) == nullptr, shape + ": no values kept at another point");
        std::vector<mh::ReferencePoint> points{mh::referenceCentre(cornerCount)};
        for (std::size_t c = 0; c < cornerCount; ++c)
            points.push_back(mh::referenceCorner(cornerCount, c));
        for (const mh::ReferencePoint& point : points)
        {
            const mh::ReferencePointValues* kept = re.valuesAt(point[0], point[1]);
            runner.expect(kept != nullptr, shape + ": values kept at an output point");
            if (kept == nullptr)
                continue;
            bool isSame = true;
            for (std::size_t i = 0; i < re.h1Basis.size(); ++i)
                isSame = isSame && kept->h1[i] == re.h1Basis[i][0].jet(point[0], point[1]);
            for (std::size_t i = 0; i < re.hcurlBasis.size(); ++i)
                for (std::size_t c = 0; c < 2; ++c)
                    isSame = isSame && kept->hcurl[i][c] == re.hcurlBasis[i][c].evaluate(point[0], point[1]);
            for (std::size_t j = 0; j < re.momentBasis.size(); ++j)
                for (std::size_t c = 0; c < 3; ++c)
                    isSame = isSame && kept->moments[j][c] == re.momentBasis[j][c].evaluate(point[0], point[1]);
            for (std::size_t m = 0; m < re.reggeBasis.size(); ++m)
                for (std::size_t c = 0; c < 3; ++c)
                    isSame = isSame && kept->regge[m][c] == re.reggeBasis[m][c].evaluate(point[0], point[1]);
            runner.expect(isSame, shape + ": the values of the polynomials, to the bit");
        }
    }
}

CHECK_CASE("shell: the resultants from the coefficients of a solution have the bits of those of the solution")
{
    for (const NamedElement& named : allElements())
    {
        const TestElement& e = named.element;
        const mh::ShellElementMatrices& m = e.data.matrices;
        const mh::ShellElementDefinition def = e.data.definition();
        mh::Vector x(m.unknownCount, 0.0);
        Xorshift random;
        for (double& value : x)
            value = 1e-3 * random.next();
        const mh::ShellSolutionCoefficients coefficients = mh::solutionCoefficients(m, x, nullptr);
        // an output point (the bases kept by the reference element) and another one (their polynomials)
        const mh::ReferencePoint centre = mh::referenceCentre(e.re->cornerCount);
        for (const mh::ReferencePoint& point : {centre, mh::ReferencePoint{-0.31, -0.42}})
        {
            const std::array<double, 8> direct =
                mh::resultants(*e.re, def, m, x, nullptr, {0.0, 0.0}, point[0], point[1]);
            const std::array<double, 8> hoisted =
                mh::resultants(*e.re, def, coefficients, x, {0.0, 0.0}, point[0], point[1]);
            runner.expect(direct == hoisted, named.name);
        }
    }
}

namespace
{

// The constant membrane and bending states of a flat element (document 34 of the reference, section 4.3): a uniform
// membrane strain gives N = A ε and no moment, a uniform curvature M = D κ and no membrane force, at every point.
void checkConstantStates(oos::checks::CheckRunner& runner, const NamedElement& named)
{
    const TestElement& e = named.element;
    const mh::ShellReferenceElement& re = *e.re;
    const mh::ShellElementDefinition def = e.data.definition();
    const mh::Section& section = e.data.section.section();
    // membrane: u = G X with an in-plane gradient
    const double g[2][2] = {{1e-3, 2e-4}, {-1e-4, 5e-4}};
    const mh::Vector xm = unknownsOfField(
        e, [&](const mh::SpaceVector& x)
        { return mh::SpaceVector{g[0][0] * x[0] + g[0][1] * x[1], g[1][0] * x[0] + g[1][1] * x[1], 0.0}; },
        [](const mh::SpaceVector&) { return mh::SpaceVector{}; });
    // bending: w = ½ (kxx x² + 2 kxy x y + kyy y²), θ = ν × β = (w_,y, −w_,x, 0)
    const double kxx = 2e-3;
    const double kxy = -7e-4;
    const double kyy = 1.2e-3;
    const mh::Vector xb = unknownsOfField(
        e,
        [&](const mh::SpaceVector& x)
        {
            const double w = 0.5 * (kxx * x[0] * x[0] + 2.0 * kxy * x[0] * x[1] + kyy * x[1] * x[1]);
            return mh::SpaceVector{0.0, 0.0, w};
        },
        [&](const mh::SpaceVector& x)
        { return mh::SpaceVector{kxy * x[0] + kyy * x[1], -(kxx * x[0] + kxy * x[1]), 0.0}; });
    double worstMembrane = 0.0;
    double worstBending = 0.0;
    const double points[3][2] = {{-0.6, -0.7}, {-0.2, 0.1}, {0.5, -0.9}};
    for (const auto& point : points)
    {
        const double xi = point[0];
        const double eta = point[1];
        const mh::PointGeometry geometry = mh::geometryAt(re, def, xi, eta);
        // the strain and the curvature in the local frame (engineering shear, κ = −∂²w)
        const auto local = [&](const double t[2][2])
        {
            const auto component = [&](std::size_t i, std::size_t j)
            {
                double sum = 0.0;
                for (std::size_t a = 0; a < 2; ++a)
                    for (std::size_t b = 0; b < 2; ++b)
                        sum += geometry.e[i][a] * t[a][b] * geometry.e[j][b];
                return sum;
            };
            return mh::Voigt3{component(0, 0), component(1, 1), 2.0 * component(0, 1)};
        };
        const double sym[2][2] = {{g[0][0], 0.5 * (g[0][1] + g[1][0])}, {0.5 * (g[0][1] + g[1][0]), g[1][1]}};
        const double curvature[2][2] = {{-kxx, -kxy}, {-kxy, -kyy}};
        const mh::Voigt3 strain = local(sym);
        const mh::Voigt3 kappa = local(curvature);
        const std::array<double, 8> rm = mh::resultants(re, def, e.data.matrices, xm, nullptr, {0.0, 0.0}, xi, eta);
        const std::array<double, 8> rb = mh::resultants(re, def, e.data.matrices, xb, nullptr, {0.0, 0.0}, xi, eta);
        const double nScale = section.a[0][0] * 1e-3;
        const double mScale = section.d[0][0] * 2e-3;
        for (std::size_t i = 0; i < 3; ++i)
        {
            double n = 0.0;
            double m = 0.0;
            for (std::size_t j = 0; j < 3; ++j)
            {
                n += section.a[i][j] * strain[j];
                m += section.d[i][j] * kappa[j];
            }
            worstMembrane = std::fmax(worstMembrane, std::fabs(rm[i] - n) / nScale);
            worstMembrane = std::fmax(worstMembrane, std::fabs(rm[3 + i]) / mScale);
            worstBending = std::fmax(worstBending, std::fabs(rb[3 + i] - m) / mScale);
            worstBending = std::fmax(worstBending, std::fabs(rb[i]) / nScale);
        }
        for (std::size_t i = 6; i < 8; ++i)
            worstMembrane = std::fmax(worstMembrane, std::fabs(rm[i]) / nScale);
    }
    runner.expect(worstMembrane < 1e-9, named.name + ": membrane state " + std::to_string(worstMembrane));
    runner.expect(worstBending < 1e-9, named.name + ": bending state " + std::to_string(worstBending));
}

} // namespace

CHECK_CASE("shell: constant membrane and bending states")
{
    // (the transition elements too: a constant state has traces of degree 2 at most, which every edge holds)
    for (const NamedElement& named :
         {NamedElement{"flat quadrilateral", flatQuadrilateral()}, NamedElement{"flat triangle", flatTriangle()},
          NamedElement{"flat quadrilateral 3, edges 2 3 3 2", flatQuadrilateral(3, false, {2, 3, 3, 2})},
          NamedElement{"flat quadrilateral 3, edges 2 2 2 2", flatQuadrilateral(3, false, {2, 2, 2, 2})},
          NamedElement{"flat quadrilateral 4, edges 3 2 4 3", flatQuadrilateral(4, false, {3, 2, 4, 3})},
          NamedElement{"flat triangle 3, edges 2 3 2", flatTriangle(3, false, {2, 3, 2})}})
        checkConstantStates(runner, named);
}

// Every configuration of the degrees of one element (its inside of degree 2, 3 or 4 and each edge from 2 to that
// degree): the rigid motions have no energy and the constant states are exact, on a flat quadrilateral and a flat triangle.
CHECK_CASE("shell: every configuration of degrees holds the rigid motions and the constant states")
{
    std::size_t configurations = 0;
    for (const std::size_t corners : {std::size_t{4}, std::size_t{3}})
        for (const std::size_t degree : {std::size_t{2}, std::size_t{3}, std::size_t{4}})
        {
            std::size_t count = 1;
            for (std::size_t k = 0; k < corners; ++k)
                count *= degree - 1;
            for (std::size_t code = 0; code < count; ++code)
            {
                EdgeDegrees edges;
                std::size_t rest = code;
                std::string name = corners == 4 ? "quadrilateral " : "triangle ";
                name += std::to_string(degree) + ", edges";
                for (std::size_t k = 0; k < corners; ++k)
                {
                    edges.push_back(2 + rest % (degree - 1));
                    rest /= degree - 1;
                    name += " " + std::to_string(edges.back());
                }
                TestElement element = corners == 4
                                          ? makeElement(bilinearQuadrilateral({mh::SpaceVector{0.0, 0.0, 0.0},
                                                                               mh::SpaceVector{2.2, 0.1, 0.0},
                                                                               mh::SpaceVector{1.9, 1.7, 0.0},
                                                                               mh::SpaceVector{-0.2, 1.2, 0.0}}),
                                                        degree, isotropicSection(0.05), 0.0, {1.0, -1.0, 1.0, -1.0},
                                                        false, false, edges)
                                          : makeElement(affineTriangle({mh::SpaceVector{0.0, 0.0, 0.0},
                                                                        mh::SpaceVector{1.7, 0.3, 0.0},
                                                                        mh::SpaceVector{0.4, 1.4, 0.0}}),
                                                        degree, isotropicSection(0.05), 0.0, {1.0, -1.0, 1.0}, false,
                                                        false, edges);
                const mh::Matrix& k = element.data.matrices.stiffness;
                const std::array<mh::Vector, 6> rigid =
                    mh::rigidMotionUnknowns(*element.re, element.data.definition());
                const double kd = maxDiagonal(k);
                double worst = 0.0;
                for (const mh::Vector& motion : rigid)
                    worst = std::fmax(worst, maxAbsVector(k * motion) / (kd * maxAbsVector(motion)));
                runner.expect(worst < 1e-10, name + ": rigid motions " + std::to_string(worst));
                checkConstantStates(runner, NamedElement{name, std::move(element)});
                ++configurations;
            }
        }
    runner.expect(configurations == 1 + 16 + 81 + 1 + 8 + 27,
                  "configurations checked: " + std::to_string(configurations));
}

namespace
{

// The stiffness condensed on the unknowns that are not interior (P = [I; −K_ii⁻¹ K_io]) and P. The condensed
// quantities do not depend on the interior part of the boundary functions of a basis, so that they can be compared
// with the reference implementation, whose triangle basis differs from this one by interior components (its
// general-purpose eigen solver leaves its boundary functions 1e-2 from L2-orthogonal to the interior ones; both are
// dual bases of the same space).
struct Condensation
{
    mh::Matrix stiffness;
    mh::Matrix p;
};

Condensation condensation(const TestElement& e)
{
    const mh::ShellReferenceElement& re = *e.re;
    const mh::Matrix& k = e.data.matrices.stiffness;
    const std::size_t n = k.rows();
    std::vector<bool> isInterior(n, false);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (i < 3 * re.h1Count)
            isInterior[i] = re.h1Kinds[i / 3].isInterior();
        else if (i < 3 * re.h1Count + re.hcurlCount)
            isInterior[i] = re.hcurlKinds[i - 3 * re.h1Count].isInterior();
    }
    const mh::CondensedInterior c = *mh::condenseInterior(k, isInterior);
    Condensation out;
    out.stiffness = c.condensedStiffness(k);
    out.p = mh::Matrix(n, c.outer.size());
    for (std::size_t a = 0; a < c.outer.size(); ++a)
    {
        mh::Vector unit(c.outer.size(), 0.0);
        unit[a] = 1.0;
        const mh::Vector inner = c.recoverInterior(mh::Vector(c.inner.size(), 0.0), unit);
        out.p(c.outer[a], a) = 1.0;
        for (std::size_t b = 0; b < c.inner.size(); ++b)
            out.p(c.inner[b], a) = inner[b];
    }
    return out;
}

void expectRelative(oos::checks::CheckRunner& runner, double value, double expected, const std::string& what)
{
    runner.expect(std::fabs(value - expected) <= 1e-10 * std::fabs(expected),
                  what + ": " + std::to_string(value) + " expected " + std::to_string(expected));
}

} // namespace

// Numbers of the reference implementation (the crate at commit 9f55f0b, through a probe building the same elements
// with its own functions), k = 4, 12 points per direction, relative tolerance 1e-10: the warped quadrilateral with the
// unsymmetric laminate and an offset, the curved triangle (a rational patch) with the laminate and an offset. The
// preload is the field of the pseudo-random boundary unknowns (Xorshift, scaled by 1e-3) with its interior unknowns
// recovered by the condensation.
CHECK_CASE("shell: numbers of the reference implementation")
{
    struct Expected
    {
        double traceKc;
        double topKc;
        double traceMc;
        double traceKgc;
        double gridN22;
        double gridM22;
        double sectionN22;
        double sectionM22;
        double energy;
    };
    const Expected quadExpected{1.29410779805250163e6,  2.63023118509224791e5,   1.39484485003863753e-8,
                                3.34172047910123320e3,  -1.19539713377508896e1,  -2.55888279142862218e-1,
                                -1.19509506353616715e1, -1.36329998926362939e-1, 1.08458432844631830e3};
    const Expected triangleExpected{1.54683205344605818e6,  4.86425495026267250e5,  3.14364262669940638e-8,
                                    -2.54840202744540766e4, -5.44534189023916859e0, 5.54143760329756652e-2,
                                    -5.42975780417045861e0, 2.81271597341201836e-2, 1.69339545994697837e4};
    struct Case
    {
        std::string name;
        TestElement element;
        double xi;
        double eta;
        Expected expected;
    };
    const Case cases[] = {{"warped quadrilateral", warpedQuadrilateral(DEGREE, true), 0.2, -0.3, quadExpected},
                          {"curved triangle", curvedTriangle(DEGREE, true), -0.4, -0.3, triangleExpected}};
    for (const Case& c : cases)
    {
        const TestElement& e = c.element;
        const mh::ShellReferenceElement& re = *e.re;
        const mh::ShellElementDefinition def = e.data.definition();
        const mh::ShellElementMatrices& m = e.data.matrices;
        const Condensation cond = condensation(e);
        const std::size_t no = cond.p.columns();
        expectRelative(runner, mh::trace(cond.stiffness), c.expected.traceKc, c.name + ": trace of K condensed");
        expectRelative(runner, sortedEigenvalues(cond.stiffness).back(), c.expected.topKc,
                       c.name + ": largest eigenvalue of K condensed");
        expectRelative(runner, mh::trace(mh::transposeTimes(cond.p, *m.mass * cond.p)), c.expected.traceMc,
                       c.name + ": trace of Pᵀ M P");
        Xorshift random;
        mh::Vector xo(no);
        for (double& v : xo)
            v = 1e-3 * random.next();
        const mh::Vector x0 = cond.p * xo;
        const mh::Matrix kg = mh::elementGeometricStiffness(re, def, m, x0, nullptr, nullptr, nullptr);
        expectRelative(runner, mh::trace(mh::transposeTimes(cond.p, kg * cond.p)), c.expected.traceKgc,
                       c.name + ": trace of Pᵀ Kg P");
        const std::array<double, 8> grid = mh::gridResultants(re, e.data, c.xi, c.eta, x0, nullptr, {});
        const std::array<double, 8> section = mh::sectionResultants(re, e.data, c.xi, c.eta, x0, nullptr, {});
        expectRelative(runner, grid[1], c.expected.gridN22, c.name + ": N22 about the grid surface");
        expectRelative(runner, grid[4], c.expected.gridM22, c.name + ": M22 about the grid surface");
        expectRelative(runner, section[1], c.expected.sectionN22, c.name + ": N22 of the section");
        expectRelative(runner, section[4], c.expected.sectionM22, c.name + ": M22 of the section");
        mh::Vector large = x0;
        for (double& v : large)
            v *= 100.0;
        const mh::NonlinearPreparation preparation = mh::prepareNonlinear(re, def);
        expectRelative(runner, mh::evaluateNonlinear(re, def, m, preparation, large, nullptr, nullptr, false).energy,
                       c.expected.energy, c.name + ": nonlinear energy at a large displacement");
    }
    // the quadrilateral bases agree with the reference's: the plain stiffness too
    const TestElement quad = warpedQuadrilateral();
    expectRelative(runner, mh::trace(quad.data.matrices.stiffness), 3.90447321216813847e6,
                   "warped quadrilateral: trace K");
    expectRelative(runner, quad.data.matrices.stiffness(0, 0), 2.15641125891758247e3, "warped quadrilateral: K(0, 0)");
}
