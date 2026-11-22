#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/mixedHybrid/model/ShellTraceDefect.hpp"
#include "oos/elements/mixedHybrid/shell/ShellPointLaw.hpp"

#include <algorithm>
#include <cmath>

namespace oos::elements::mixedHybrid
{
namespace
{
struct DirectorMode
{
    std::size_t component;
    Polynomial2 polynomial;
};

// Covariant components of director changes with zero trace on every side.
// A triangle gains two cubic bubbles. On a quad the shear spaces grow from
// Q(2,3) x Q(3,2) to Q(3,4) x Q(4,3). Keep the new interior monomials only.
std::vector<DirectorMode> directorModes(std::size_t corners)
{
    const auto x = Polynomial2::monomial(1, 0);
    const auto y = Polynomial2::monomial(0, 1);
    const auto one = Polynomial2::constant(1);
    if (corners == 3)
    {
        const auto bubble = (x + y) * (one + x) * (one + y);
        return {{0, bubble}, {1, bubble}};
    }
    const auto bubble = (one - x * x) * (one - y * y);
    std::vector<DirectorMode> modes;
    for (const std::size_t c : {0u, 1u})
    {
        const auto& along = c == 0 ? x : y;
        const auto& across = c == 0 ? y : x;
        for (const auto& p : {along, along * across, across * across, along * across * across})
            modes.push_back({c, bubble * p});
    }
    return modes;
}

// The two mass terms that work on a virtual director: I1 a and I2 beta(a).
// ShellPart::affineField carries affine acceleration with zero shear, hence
// beta(a) is the negative tangential projection of grad(a)^T nu.
SpaceVector directorLoad(const BodyAcceleration& field, const PointGeometry& g, const SectionPoint& section)
{
    const auto a = field.at({g.x[0], g.x[1], g.x[2]});
    const auto grad = field.gradient.transposed() * base::Vec3{g.nu[0], g.nu[1], g.nu[2]};
    return sub3(scl3(a.components, section.mass[1]), scl3(grad.components, section.mass[2]));
}
} // namespace

ShellInteriorDefect directorEquilibrium(const ShellPart& part, std::size_t shell, const ShellElementData& element,
                                        std::span<const double> x, const ShellSolutionCoefficients& coefficients,
                                        const BodyAcceleration* acceleration)
{
    static const auto triangle = directorModes(3);
    static const auto quad = directorModes(4);
    const auto& modes = element.cornerCount == 3 ? triangle : quad;
    // Integrate on the same rational map as the pilot. Six Gauss points integrate
    // the local test stiffness exactly on an affine cell with constant section.
    const auto& re = shellReferenceElement(element.cornerCount, element.degree, 6, element.edgeDegrees);
    Matrix stiffness(modes.size(), modes.size());
    Vector residual(modes.size(), 0.0);
    ShellInteriorDefect out{shell};
    const auto& corners = part.input().shells[shell].corners;
    double gradientScale = 0.0;
    for (std::size_t k = 0; k < corners.count; ++k)
    {
        const auto next = (k + 1) % corners.count;
        const auto chord =
            sub3(part.input().grids[corners.at(k)].position, part.input().grids[corners.at(next)].position);
        const SpaceVector du{x[3 * k] - x[3 * next], x[3 * k + 1] - x[3 * next + 1], x[3 * k + 2] - x[3 * next + 2]};
        gradientScale = std::max(gradientScale, norm3(du) / norm3(chord));
    }
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const auto [xi, eta] = re.points[q];
        const auto weight = element.weight.jet(xi, eta);
        const auto g = geometryFrom(
            element.mapCoefficients, [&](std::size_t i) { return rationalJet(re.h1[q * re.h1Count + i], weight); },
            element.axes, true);
        const auto section = sectionAtGeometry(element.section, element.offset, xi, eta, g);
        const auto cov = g.cov();
        const auto toVoigt = covariantToVoigt(cov);
        const auto sv = momentsLocal(
            re, [&](std::size_t j) { return re.moments[q * re.momentCount + j]; }, g, element.mapReferenceConstants);
        const auto localMoment = sv * coefficients.moments;
        const Voigt3 moment{localMoment[0], localMoment[1], localMoment[2]};
        std::array<double, 2> gamma{};
        for (std::size_t i = 0; i < re.hcurlCount; ++i)
        {
            const auto& basis = re.hcurl[q * re.hcurlCount + i];
            for (std::size_t c = 0; c < 2; ++c)
                gamma[c] += x[3 * re.h1Count + i] * (cov[0][c] * basis[0] + cov[1][c] * basis[1]);
        }
        const SpaceVector load = acceleration ? directorLoad(*acceleration, g, section) : SpaceVector{};
        const double w = re.weights[q] * g.j;
        std::array<Voigt3, 8> curvature{}, elasticMoment{};
        std::array<std::array<double, 2>, 8> rotation{}, elasticShear{};
        for (std::size_t i = 0; i < modes.size(); ++i)
        {
            const auto c = modes[i].component;
            const auto p = modes[i].polynomial.jet(xi, eta);
            const auto& connection = g.christoffel[c];
            curvature[i] = times33(toVoigt, {(c == 0 ? p[1] : 0.0) - connection[0][0] * p[0],
                                             (c == 1 ? p[2] : 0.0) - connection[1][1] * p[0],
                                             0.5 * (c == 0 ? p[2] : p[1]) - connection[0][1] * p[0]});
            rotation[i] = {cov[c][0] * p[0], cov[c][1] * p[0]};
            elasticMoment[i] = times33(section.d, curvature[i]);
            for (std::size_t j = 0; j < 2; ++j)
                elasticShear[i][j] = dot(section.h[j], rotation[i]);
            residual[i] += w * (p[0] * dot3(g.ac[c], load) - dot(moment, curvature[i]) - dot(gamma, elasticShear[i]));
            for (std::size_t j = 0; j <= i; ++j)
                stiffness(i, j) += w * (dot(curvature[j], elasticMoment[i]) + dot(rotation[j], elasticShear[i]));
        }
        const double bendingEnergy = dot3(moment, times33(sectionLaws(section).dinv, moment));
        out.referenceEnergy += w * (bendingEnergy + (section.h[0][0] + section.h[1][1]) *
                                                        (gradientScale * gradientScale + dot(gamma, gamma)));
    }
    for (std::size_t i = 0; i < modes.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            stiffness(j, i) = stiffness(i, j);
    if (const auto factor = EquilibratedCholesky::factor(stiffness))
        out.work = dot(residual, factor->solve(residual));
    return out;
}
} // namespace oos::elements::mixedHybrid
