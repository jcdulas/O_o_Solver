#include "oos/elements/core/SolidIntegrationRules.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace oos::elements
{

namespace
{

// The classic four-point rule of the tetrahedron (degree 2): L = a at one corner, b at the three others.
constexpr double TETRA_FOUR_POINT_A = 0.58541019662496845446;
constexpr double TETRA_FOUR_POINT_B = 0.13819660112501051518;

[[noreturn]] void refuseRule(const char* rule, std::size_t value, const char* expected)
{
    throw std::invalid_argument(base::concatenate({"solid integration rule: ", rule, " expected ", expected, ", found ",
                                                   std::to_string(value), "; ask for a supported rule"}));
}

// Maps an n-point Gauss-Legendre rule from [−1, 1] to [0, 1] (abscissae (x + 1)/2, weights w/2).
struct UnitRule
{
    std::size_t count = 0;
    std::array<double, 4> abscissae{};
    std::array<double, 4> weights{};
};
[[nodiscard]] UnitRule unitRule(std::size_t points)
{
    const GaussLegendreRule gauss = gaussLegendreRule(points);
    UnitRule rule;
    rule.count = gauss.count;
    for (std::size_t i = 0; i < gauss.count; ++i)
    {
        rule.abscissae[i] = 0.5 * (gauss.abscissae[i] + 1.0);
        rule.weights[i] = 0.5 * gauss.weights[i];
    }
    return rule;
}

// A triangle rule in area coordinates (L2, L3) with weights summing to the area 1/2.
struct TriangleRule
{
    std::size_t count = 0;
    std::array<std::array<double, 3>, 16> points{}; // L2, L3, weight
};

[[nodiscard]] TriangleRule triangleRule(std::size_t degree)
{
    TriangleRule rule;
    if (degree <= 1)
    {
        rule.count = 1;
        rule.points[0] = {1.0 / 3.0, 1.0 / 3.0, 0.5};
        return rule;
    }
    if (degree == 2)
    {
        rule.count = 3;
        rule.points[0] = {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0};
        rule.points[1] = {2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0};
        rule.points[2] = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};
        return rule;
    }
    // Collapsed square (Duffy): L2 = a, L3 = b (1 − a), dA = (1 − a) da db; a polynomial of degree p becomes one of
    // degree p + 1 in a, so n Gauss points per direction are exact for p ≤ 2n − 2.
    const std::size_t n = (degree + 3) / 2;
    const UnitRule u = unitRule(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
        {
            const double a = u.abscissae[i];
            const double b = u.abscissae[j];
            rule.points[rule.count++] = {a, b * (1.0 - a), u.weights[i] * u.weights[j] * (1.0 - a)};
        }
    return rule;
}

[[nodiscard]] SolidRule buildHexahedronRule(std::size_t n)
{
    const GaussLegendreRule g = gaussLegendreRule(n);
    SolidRule rule;
    // ξ varies fastest, then η, then ζ.
    for (std::size_t k = 0; k < n; ++k)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = 0; i < n; ++i)
                rule.points[rule.count++] = {{g.abscissae[i], g.abscissae[j], g.abscissae[k]},
                                             g.weights[i] * g.weights[j] * g.weights[k]};
    return rule;
}

[[nodiscard]] SolidRule buildTetrahedronRule(std::size_t degree)
{
    SolidRule rule;
    if (degree <= 1)
    {
        rule.count = 1;
        rule.points[0] = {{0.25, 0.25, 0.25}, 1.0 / 6.0};
        return rule;
    }
    if (degree == 2)
    {
        const double a = TETRA_FOUR_POINT_A;
        const double b = TETRA_FOUR_POINT_B;
        rule.count = 4;
        rule.points[0] = {{b, b, b}, 1.0 / 24.0};
        rule.points[1] = {{a, b, b}, 1.0 / 24.0};
        rule.points[2] = {{b, a, b}, 1.0 / 24.0};
        rule.points[3] = {{b, b, a}, 1.0 / 24.0};
        return rule;
    }
    if (degree == 3)
    {
        // Five points: the centroid with the weight −4/5 of the volume, and L = ½ at one corner, ⅙ at the others
        // with 9/20 each (Keast, CMAME 55, 1986, rule 2; exact for degree 3, one weight negative).
        rule.count = 5;
        rule.points[0] = {{0.25, 0.25, 0.25}, -4.0 / 5.0 / 6.0};
        rule.points[1] = {{1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0}, 9.0 / 20.0 / 6.0};
        rule.points[2] = {{0.5, 1.0 / 6.0, 1.0 / 6.0}, 9.0 / 20.0 / 6.0};
        rule.points[3] = {{1.0 / 6.0, 0.5, 1.0 / 6.0}, 9.0 / 20.0 / 6.0};
        rule.points[4] = {{1.0 / 6.0, 1.0 / 6.0, 0.5}, 9.0 / 20.0 / 6.0};
        return rule;
    }
    // Collapsed cube: L2 = a, L3 = b (1 − a), L4 = c (1 − a)(1 − b), dV = (1 − a)² (1 − b) da db dc; degree p becomes
    // p + 2 in a, so n points per direction are exact for p ≤ 2n − 3.
    const std::size_t n = (degree + 4) / 2;
    const UnitRule u = unitRule(n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t k = 0; k < n; ++k)
            {
                const double a = u.abscissae[i];
                const double b = u.abscissae[j];
                const double c = u.abscissae[k];
                rule.points[rule.count++] = {
                    {a, b * (1.0 - a), c * (1.0 - a) * (1.0 - b)},
                    u.weights[i] * u.weights[j] * u.weights[k] * (1.0 - a) * (1.0 - a) * (1.0 - b)};
            }
    return rule;
}

[[nodiscard]] SolidRule buildPentahedronRule(std::size_t triangleDegree, std::size_t zetaPoints)
{
    const TriangleRule triangle = triangleRule(triangleDegree);
    const GaussLegendreRule g = gaussLegendreRule(zetaPoints);
    SolidRule rule;
    for (std::size_t k = 0; k < g.count; ++k)
        for (std::size_t t = 0; t < triangle.count; ++t)
            rule.points[rule.count++] = {{triangle.points[t][0], triangle.points[t][1], g.abscissae[k]},
                                         triangle.points[t][2] * g.weights[k]};
    return rule;
}

} // namespace

GaussLegendreRule gaussLegendreRule(std::size_t points)
{
    GaussLegendreRule rule;
    rule.count = points;
    switch (points)
    {
    case 1:
        rule.abscissae = {0.0};
        rule.weights = {2.0};
        break;
    case 2:
        rule.abscissae = {-0.57735026918962576451, 0.57735026918962576451};
        rule.weights = {1.0, 1.0};
        break;
    case 3:
        rule.abscissae = {-0.77459666924148337704, 0.0, 0.77459666924148337704};
        rule.weights = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
        break;
    case 4:
        rule.abscissae = {-0.86113631159405257522, -0.33998104358485626480, 0.33998104358485626480,
                          0.86113631159405257522};
        rule.weights = {0.34785484513745385737, 0.65214515486254614263, 0.65214515486254614263, 0.34785484513745385737};
        break;
    default:
        refuseRule("Gauss-Legendre points", points, "1 to 4");
    }
    return rule;
}

const SolidRule& hexahedronRule(std::size_t pointsPerAxis)
{
    static const std::array<SolidRule, 4> RULES{buildHexahedronRule(1), buildHexahedronRule(2), buildHexahedronRule(3),
                                                buildHexahedronRule(4)};
    if (pointsPerAxis < 1 || pointsPerAxis > 4)
        refuseRule("hexahedron points per axis", pointsPerAxis, "1 to 4");
    return RULES[pointsPerAxis - 1];
}

const SolidRule& tetrahedronRule(std::size_t degree)
{
    static const std::array<SolidRule, 5> RULES{buildTetrahedronRule(1), buildTetrahedronRule(2),
                                                buildTetrahedronRule(3), buildTetrahedronRule(4),
                                                buildTetrahedronRule(5)};
    if (degree < 1 || degree > 5)
        refuseRule("tetrahedron degree", degree, "1 to 5");
    return RULES[degree - 1];
}

const SolidRule& pentahedronRule(std::size_t triangleDegree, std::size_t zetaPoints)
{
    // Six triangle degrees times four ζ rules, built once.
    static const std::array<SolidRule, 24> RULES = []
    {
        std::array<SolidRule, 24> rules{};
        for (std::size_t d = 1; d <= 6; ++d)
            for (std::size_t z = 1; z <= 4; ++z)
                rules[(d - 1) * 4 + (z - 1)] = buildPentahedronRule(d, z);
        return rules;
    }();
    if (triangleDegree < 1 || triangleDegree > 6)
        refuseRule("pentahedron triangle degree", triangleDegree, "1 to 6");
    if (zetaPoints < 1 || zetaPoints > 4)
        refuseRule("pentahedron points along zeta", zetaPoints, "1 to 4");
    return RULES[(triangleDegree - 1) * 4 + (zetaPoints - 1)];
}

} // namespace oos::elements
