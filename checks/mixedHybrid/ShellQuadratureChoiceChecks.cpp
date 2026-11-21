#include "CheckRunner.hpp"
#include "ShellCheckElements.hpp"
#include "oos/elements/mixedHybrid/shell/ShellQuadratureChoice.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <variant>

namespace mh = oos::elements::mixedHybrid;
using namespace oos::checks::shell;

namespace
{

constexpr std::size_t DEGREE = 4;
// The rule of the reference (2k + 4 points per direction), which the choice never exceeds.
constexpr std::size_t REFERENCE_POINTS = 12;

// A parallelogram and a triangle: affine elements.
TestElement parallelogram()
{
    const TestSurface s = bilinearQuadrilateral({mh::SpaceVector{0.0, 0.0, 0.0}, mh::SpaceVector{2.0, 0.2, 0.1},
                                                 mh::SpaceVector{2.5, 1.7, 0.4}, mh::SpaceVector{0.5, 1.5, 0.3}});
    return makeElement(s, DEGREE, isotropicSection(0.05), 0.0, {1.0, -1.0, 1.0, -1.0}, false);
}

TestElement flatTriangle()
{
    const TestSurface s = affineTriangle(
        {mh::SpaceVector{0.0, 0.0, 0.0}, mh::SpaceVector{2.2, 0.1, 0.3}, mh::SpaceVector{0.4, 1.6, -0.2}});
    return makeElement(s, DEGREE, isotropicSection(0.05), 0.0, {1.0, -1.0, 1.0}, false);
}

// A flat quadrilateral that is no parallelogram, and one that is warped too: their maps are bilinear, their integrands
// rational.
TestElement distortedQuadrilateral(double warp)
{
    const TestSurface s = bilinearQuadrilateral({mh::SpaceVector{0.0, 0.0, 0.0}, mh::SpaceVector{2.2, 0.1, 0.0},
                                                 mh::SpaceVector{1.6, 1.9, warp}, mh::SpaceVector{-0.2, 1.1, 0.0}});
    return makeElement(s, DEGREE, isotropicSection(0.05), 0.0, {1.0, -1.0, 1.0, -1.0}, false);
}

// A quarter of a cylinder in one element (a rational patch).
TestElement cylinderQuarter()
{
    return makeElement(cylinderQuadrant(2.0, 1.5), DEGREE, isotropicSection(0.02), 0.0, {1.0, 1.0, -1.0, -1.0},
                       false);
}

std::size_t pointsOf(const TestElement& element, double tolerance)
{
    mh::ShellRuleChoice choice;
    choice.tolerance = tolerance;
    return mh::chooseShellPoints(element.data.cornerCount, DEGREE, element.data.definition(), choice);
}

// The stiffness of the element integrated on the rule of `points` points per direction.
mh::Matrix stiffnessOn(const TestElement& element, std::size_t points)
{
    const mh::ShellReferenceElement& re = mh::shellReferenceElement(element.data.cornerCount, DEGREE, points);
    mh::ShellElementData data = element.data;
    mh::ElementJets jets = mh::elementJets(re, data.weight);
    data.volumeJets = std::move(jets.volume);
    data.edgeJets = std::move(jets.edge);
    data.edgeWeights = std::move(jets.edgeWeight);
    const auto built = mh::buildShellElement(re, data.definition(), false);
    return std::get<mh::ShellElementMatrices>(built).stiffness;
}

// The largest difference of two matrices over the largest term of the second.
double relativeDifference(const mh::Matrix& a, const mh::Matrix& b)
{
    return mh::maxAbs(a - b) / mh::maxAbs(b);
}

} // namespace

CHECK_CASE("shell rule: an affine element takes the smallest rule")
{
    const mh::ShellRuleChoice choice;
    runner.expect(pointsOf(parallelogram(), choice.tolerance) == choice.minimumPoints, "parallelogram");
    runner.expect(pointsOf(flatTriangle(), choice.tolerance) == choice.minimumPoints, "flat triangle");
    // and that rule integrates it to the round-off: the integrands of an affine element are polynomials
    const TestElement element = parallelogram();
    const double difference =
        relativeDifference(stiffnessOn(element, choice.minimumPoints), stiffnessOn(element, REFERENCE_POINTS));
    runner.expect(difference < 1e-11, "stiffness of the parallelogram: " + std::to_string(difference));
}

CHECK_CASE("shell rule: a distorted, warped or curved element takes more points, never more than the reference")
{
    const mh::ShellRuleChoice choice;
    const std::size_t affine = pointsOf(parallelogram(), choice.tolerance);
    for (const TestElement& element : {distortedQuadrilateral(0.0), distortedQuadrilateral(0.3), cylinderQuarter()})
    {
        const std::size_t points = pointsOf(element, choice.tolerance);
        runner.expect(points > affine && points <= REFERENCE_POINTS, "points " + std::to_string(points));
    }
    // a looser tolerance never asks for more points
    for (const TestElement& element : {distortedQuadrilateral(0.0), distortedQuadrilateral(0.3), cylinderQuarter()})
    {
        std::size_t previous = 0;
        for (const double tolerance : {1e-4, 1e-6, 1e-8, 1e-10, 1e-12})
        {
            const std::size_t points = pointsOf(element, tolerance);
            runner.expect(points >= previous, "tolerance " + std::to_string(tolerance));
            previous = points;
        }
    }
}

CHECK_CASE("shell rule: the stiffness on the rule chosen is that of the rule of the reference, to the tolerance")
{
    // The choice measures the convergence of the geometric factors of the element; the stiffness, which multiplies
    // them by its polynomials, follows them within a factor of a hundred.
    constexpr double MARGIN = 100.0;
    for (const double tolerance : {1e-6, 1e-8, 1e-10})
        for (const TestElement& element : {distortedQuadrilateral(0.0), distortedQuadrilateral(0.3)})
        {
            const std::size_t points = pointsOf(element, tolerance);
            if (points == REFERENCE_POINTS)
                continue;
            const double difference =
                relativeDifference(stiffnessOn(element, points), stiffnessOn(element, REFERENCE_POINTS));
            runner.expect(difference < MARGIN * tolerance, "tolerance " + std::to_string(tolerance) + ", " +
                                                               std::to_string(points) + " points: " +
                                                               std::to_string(difference));
        }
}
