#include "CheckRunner.hpp"
#include "LineCheckSupport.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <variant>

namespace mh = oos::elements::mixedHybrid;
using lineChecks::expectRelative;
using lineChecks::Vec3;
using mh::LineElement;
using mh::LineSection;

namespace
{

// A tapered beam with shear factors and its neutral axis off the reference axis.
LineSection taperedBeam()
{
    LineSection s = LineSection::beam(
        2.0e11, 8.0e10, 7800.0, {{{3.0e-3, 4.0e-6, 9.0e-6, 1.0e-6, 5.0e-6}, {2.0e-3, 2.5e-6, 6.0e-6, 0.5e-6, 3.0e-6}}},
        std::array<double, 2>{0.85, 0.6}, {{{0.01, -0.02}, {0.015, 0.0}}});
    s.nonStructuralMass = 0.5;
    return s;
}

} // namespace

CHECK_CASE("line: section law of a rod, a bar and a tapered beam")
{
    const double e = 2.0e11;
    const double g = 8.0e10;
    // A rod: axial and torsion only.
    const mh::LineSectionPoint rod = LineSection::rod(e, g, 3.0e-3, 5.0e-6, 7800.0).at(0.3);
    expectRelative(runner, rod.stiffness[0][0], e * 3.0e-3, 1e-15, "rod: E A");
    expectRelative(runner, rod.stiffness[3][3], g * 5.0e-6, 1e-15, "rod: G J");
    runner.expect(rod.stiffness[1][1] == 0.0 && rod.stiffness[4][4] == 0.0 && rod.stiffness[5][5] == 0.0,
                  "rod: no shear nor bending stiffness");
    // A bar: shear 1e4 E A without shear factors.
    const mh::LineSectionPoint bar = LineSection::bar(e, g, 3.0e-3, 4.0e-6, 9.0e-6, 1.0e-6, 5.0e-6, 7800.0).at(0.0);
    expectRelative(runner, bar.stiffness[1][1], 1.0e4 * e * 3.0e-3, 1e-15, "bar: rigid shear");
    expectRelative(runner, bar.stiffness[4][4], e * 9.0e-6, 1e-15, "bar: E I2 on K_y");
    expectRelative(runner, bar.stiffness[5][5], e * 4.0e-6, 1e-15, "bar: E I1 on K_z");
    expectRelative(runner, bar.stiffness[4][5], -e * 1.0e-6, 1e-15, "bar: -E I12");
    expectRelative(runner, bar.massPerLength, 7800.0 * 3.0e-3, 1e-15, "bar: mass per length");
    expectRelative(runner, bar.inertia[0], 7800.0 * 13.0e-6, 1e-15, "bar: inertia about x");
    // A tapered beam at λ = 0.25: linear properties, the neutral axis coupling.
    const LineSection beam = taperedBeam();
    const double lam = 0.25;
    const mh::LineSectionPoint p = beam.at(lam);
    const double a = 0.75 * 3.0e-3 + 0.25 * 2.0e-3;
    const double yn = 0.75 * 0.01 + 0.25 * 0.015;
    const double zn = 0.75 * -0.02;
    expectRelative(runner, p.stiffness[0][0], e * a, 1e-14, "beam: E A");
    expectRelative(runner, p.stiffness[0][4], e * a * zn, 1e-14, "beam: E A z_n");
    expectRelative(runner, p.stiffness[0][5], -e * a * yn, 1e-14, "beam: -E A y_n");
    expectRelative(runner, p.stiffness[4][4], e * a * zn * zn + e * (0.75 * 9.0e-6 + 0.25 * 6.0e-6), 1e-14,
                   "beam: E I2 + E A z_n²");
    expectRelative(runner, p.stiffness[1][1], 0.85 * g * a, 1e-14, "beam: k1 G A");
    expectRelative(runner, p.stiffness[2][2], 0.6 * g * a, 1e-14, "beam: k2 G A");
    expectRelative(runner, p.massPerLength, 7800.0 * a + 0.5, 1e-14, "beam: mass with the non-structural mass");
    for (std::size_t r = 0; r < 6; ++r)
        for (std::size_t c = 0; c < 6; ++c)
            runner.expect(p.stiffness[r][c] == p.stiffness[c][r], "beam: symmetric law");
}

CHECK_CASE("line: forces of the section law with a free strain")
{
    const mh::LineSectionPoint p = taperedBeam().at(0.6);
    // A free expansion e0 along x on its own carries no force.
    const double e0 = 3.0e-4;
    const std::array<double, 6> free{e0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const std::array<double, 6> none = mh::lineForces(p, free, e0, {1.0, 0.0, 0.0});
    for (std::size_t r = 0; r < 6; ++r)
        runner.expect(std::fabs(none[r]) <= 1e-12 * p.stiffness[0][0] * e0, "free expansion: no force");
    // Otherwise F = D ε.
    const std::array<double, 6> strains{1e-4, 2e-5, -3e-5, 4e-4, -1e-3, 2e-3};
    const std::array<double, 6> forces = mh::lineForces(p, strains, 0.0, {1.0, 0.0, 0.0});
    for (std::size_t r = 0; r < 6; ++r)
    {
        double expected = 0.0;
        for (std::size_t c = 0; c < 6; ++c)
            expected += p.stiffness[r][c] * strains[c];
        expectRelative(runner, forces[r], expected, 1e-15, "F = D ε, row " + std::to_string(r));
    }
}

CHECK_CASE("line: Legendre polynomials with their derivatives")
{
    for (const double x : {-1.0, -0.37, 0.0, 0.5, 1.0})
    {
        const mh::PolynomialTable table = mh::legendreTable(8, x);
        for (std::size_t n = 0; n <= 8; ++n)
        {
            const auto [v, dv] = mh::legendreWithDerivative(n, x);
            runner.expect(v == mh::legendreValue(n, x), "P_n: the recurrence of the core");
            runner.expectNear(dv, table.derivatives[n], 1e-12, "P'_" + std::to_string(n));
        }
    }
    // P_n(1) = 1, P'_n(1) = n (n + 1) / 2.
    for (std::size_t n = 0; n <= 6; ++n)
    {
        const auto [v, dv] = mh::legendreWithDerivative(n, 1.0);
        runner.expect(v == 1.0, "P_n(1) = 1");
        runner.expect(dv == static_cast<double>(n * (n + 1) / 2), "P'_n(1) = n (n + 1) / 2");
    }
}

CHECK_CASE("line: section axes and their refusals")
{
    const Vec3 t = mh::unit3(Vec3{2.0, 1.0, 0.5});
    const LineSection bar = LineSection::bar(2.0e11, 8.0e10, 1.0e-3, 1.0e-6, 2.0e-6, 0.0, 1.0e-6, 0.0);
    const LineSection rod = LineSection::rod(2.0e11, 8.0e10, 1.0e-3, 1.0e-6, 0.0);
    // The orientation vector: y its part across the line.
    const Vec3 v{0.0, 0.3, 1.0};
    const auto oriented = mh::sectionAxes(LineElement::between(0, 1, 0).oriented(v), bar, t, std::nullopt);
    runner.expect(std::holds_alternative<mh::SectionAxes>(oriented), "oriented bar: axes");
    if (const auto* axes = std::get_if<mh::SectionAxes>(&oriented))
    {
        const Vec3 y = mh::unit3(mh::sub3(v, mh::scl3(t, mh::dot3(v, t))));
        for (std::size_t c = 0; c < 3; ++c)
        {
            runner.expectNear((*axes)[1][c], y[c], 1e-15, "y across the line");
            runner.expectNear((*axes)[2][c], mh::cross3(t, y)[c], 1e-15, "z = x × y");
        }
    }
    // The normal of the shell, when there is no orientation vector.
    const Vec3 n{0.0, 0.0, 1.0};
    const auto onShell = mh::sectionAxes(LineElement::between(0, 1, 0), bar, t, n);
    runner.expect(std::holds_alternative<mh::SectionAxes>(onShell), "bar on a shell edge: axes from the normal");
    // A rod alone takes an axis across the line.
    const auto rodAxes = mh::sectionAxes(LineElement::between(0, 1, 0), rod, t, std::nullopt);
    runner.expect(std::holds_alternative<mh::SectionAxes>(rodAxes), "rod alone: any axis across");
    if (const auto* axes = std::get_if<mh::SectionAxes>(&rodAxes))
        runner.expectNear(mh::dot3((*axes)[1], t), 0.0, 1e-15, "rod: y across the line");
    // Refusals.
    runner.expect(
        std::holds_alternative<std::string>(mh::sectionAxes(LineElement::between(0, 1, 0), bar, t, std::nullopt)),
        "a bar alone without its orientation vector is refused");
    runner.expect(std::holds_alternative<std::string>(
                      mh::sectionAxes(LineElement::between(0, 1, 0).oriented(mh::scl3(t, 3.0)), bar, t, std::nullopt)),
                  "an orientation vector along the line is refused");
    runner.expect(std::holds_alternative<std::string>(mh::sectionAxes(LineElement::between(0, 1, 0), bar, t, t)),
                  "a normal along the line is refused");
    runner.expect(std::holds_alternative<std::string>(mh::aloneEdgeOfLine(
                      LineElement::between(0, 1, 0).oriented(v), bar, {0, 1}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, 4, 12)),
                  "a line of zero length is refused");
}

CHECK_CASE("line: functions and kinematics of an edge alone")
{
    const Vec3 xa{0.3, -0.2, 0.5};
    const Vec3 xb{1.6, 0.4, 1.1};
    const LineSection bar = LineSection::bar(2.0e11, 8.0e10, 1.0e-3, 1.0e-6, 2.0e-6, 0.0, 1.0e-6, 0.0);
    const LineElement line = LineElement::between(0, 1, 0).oriented({0.0, 0.3, 1.0});
    for (const std::size_t k : {2U, 3U, 4U})
    {
        const auto built = mh::aloneEdgeOfLine(line, bar, {0, 1}, xa, xb, k, 12);
        runner.expect(std::holds_alternative<mh::AloneEdge>(built), "edge alone");
        if (!std::holds_alternative<mh::AloneEdge>(built))
            continue;
        const mh::AloneEdge& edge = std::get<mh::AloneEdge>(built);
        runner.expect(edge.unknownCount() == 6 + 3 * (k - 1) + k, "n_x = 6 + 3 (k - 1) + k");
        // The reference numbering: grids, moments, then the twist after the k unknowns of the shear.
        runner.expect(edge.dofs[0] == 0 && edge.dofs[5] == 8 && edge.dofs[6] == 12, "numbering of the grids");
        runner.expect(edge.dofs[6 + 3 * (k - 1)] == 12 + 3 * (k - 1) + k, "numbering of the twist");
        // The functions vanish at the grids but the two linear ones.
        for (const double s : {0.0, 1.0})
        {
            const mh::EdgeFunctions f = edge.functions(s);
            runner.expect(f.values[0] == 1.0 - s && f.values[1] == s, "linear functions at the grids");
            for (std::size_t i = 2; i < f.values.size(); ++i)
                runner.expect(f.values[i] == 0.0, "bubble functions vanish at the grids");
        }
        // Derivatives by central differences.
        const double s = 0.37;
        const double h = 1e-6;
        const mh::EdgeFunctions f = edge.functions(s);
        const mh::EdgeFunctions fp = edge.functions(s + h);
        const mh::EdgeFunctions fm = edge.functions(s - h);
        for (std::size_t i = 0; i < f.values.size(); ++i)
            runner.expectNear(f.derivatives[i], (fp.values[i] - fm.values[i]) / (2.0 * h), 1e-8, "d/ds of a function");
        // θ = t × du/ds and ψ t; the line from the upper grid: t and du/ds change sign, θ does not.
        const mh::LineKinematics forward = edge.kinematics(s, true);
        const mh::LineKinematics backward = edge.kinematics(s, false);
        runner.expect(forward.fraction == s && backward.fraction == 1.0 - s, "fraction from grid A");
        const std::size_t n = edge.unknownCount();
        for (std::size_t col = 0; col < n; ++col)
        {
            const Vec3 du{forward.displacementRate(0, col), forward.displacementRate(1, col),
                          forward.displacementRate(2, col)};
            const Vec3 theta{forward.rotation(0, col), forward.rotation(1, col), forward.rotation(2, col)};
            const Vec3 expected = col < 6 + 3 * (k - 1) ? mh::cross3(forward.tangent, du) : theta;
            for (std::size_t r = 0; r < 3; ++r)
            {
                runner.expectNear(theta[r], expected[r], 1e-14, "θ = t × du/ds");
                runner.expect(backward.rotation(r, col) == forward.rotation(r, col), "θ independent of the direction");
                runner.expect(backward.displacementRate(r, col) == -forward.displacementRate(r, col),
                              "du/ds changes sign with the direction");
            }
        }
        // The twist: θ · t = P_l(2s − 1) / L.
        const double len = edge.length();
        for (std::size_t l = 0; l < k; ++l)
        {
            const std::size_t col = 6 + 3 * (k - 1) + l;
            const Vec3 theta{forward.rotation(0, col), forward.rotation(1, col), forward.rotation(2, col)};
            runner.expectNear(mh::dot3(theta, forward.tangent), mh::legendreValue(l, 2.0 * s - 1.0) / len, 1e-14,
                              "twist function");
        }
    }
}

CHECK_CASE("line: description of the unknowns of a line alone")
{
    const std::size_t k = 4;
    const std::vector<mh::LineUnknown> unknowns = mh::aloneLineUnknowns(k);
    runner.expect(unknowns.size() == 6 + 3 * (k - 1) + k + 6, "n_x + 6 unknowns");
    runner.expect(
        unknowns[4].kind == mh::LineUnknownKind::GridTranslation && unknowns[4].end == 1 && unknowns[4].component == 1,
        "unknown 4: translation y of the upper grid");
    runner.expect(unknowns[6 + 3 * 1 + 2].kind == mh::LineUnknownKind::EdgeFunction && unknowns[11].index == 1 &&
                      unknowns[11].component == 2 && unknowns[11].edgeSlot == 5,
                  "unknown 11: function 1 along the third axis of the frame");
    runner.expect(unknowns[16].kind == mh::LineUnknownKind::EdgeTwist && unknowns[16].index == 1 &&
                      unknowns[16].edgeSlot == 3 * (k - 1) + k + 1,
                  "unknown 16: twist coefficient 1");
    runner.expect(
        unknowns[24].kind == mh::LineUnknownKind::GridRotation && unknowns[24].end == 1 && unknowns[24].component == 2,
        "unknown 24: rotation z at grid B");
}
