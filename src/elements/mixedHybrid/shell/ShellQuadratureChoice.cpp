#include "oos/elements/mixedHybrid/shell/ShellQuadratureChoice.hpp"

#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/shell/ShellGeometry.hpp"
#include "oos/elements/mixedHybrid/shell/ShellPointLaw.hpp"
#include "oos/elements/mixedHybrid/shell/ShellReferenceElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <span>

namespace oos::elements::mixedHybrid
{

namespace
{

// The highest degree, in each reference coordinate, of the polynomial an element multiplies a geometric factor by:
// the product of two functions of degree 4 (the mass, the shear stiffness).
constexpr std::size_t TOP_DEGREE = 8;

// The test polynomials P_a(ξ) P_b(η) (Legendre, on the reference domain, which lies in [−1, 1]²): the constant and
// the highest degree in each coordinate and in both for a quadrilateral, whose functions have degree 4 in each
// coordinate; for a triangle, whose functions have total degree 4, the product of degree 4 in each in place of the
// last.
struct TestDegrees
{
    std::size_t xi;
    std::size_t eta;
};
constexpr std::size_t TEST_COUNT = 4;
constexpr std::array<TestDegrees, TEST_COUNT> QUADRILATERAL_TESTS{
    {{0, 0}, {TOP_DEGREE, 0}, {0, TOP_DEGREE}, {TOP_DEGREE, TOP_DEGREE}}};
constexpr std::array<TestDegrees, TEST_COUNT> TRIANGLE_TESTS{
    {{0, 0}, {TOP_DEGREE, 0}, {0, TOP_DEGREE}, {TOP_DEGREE / 2, TOP_DEGREE / 2}}};
// The test polynomials of an edge, P_a(2τ − 1): the constant, the two highest degrees (one of each parity), and the
// middle one.
constexpr std::array<std::size_t, TEST_COUNT> EDGE_TESTS{0, TOP_DEGREE / 2, TOP_DEGREE - 1, TOP_DEGREE};

// The Legendre polynomials P_0 ... P_8 at x: k P_k = (2k − 1) x P_{k−1} − (k − 1) P_{k−2}.
std::array<double, TOP_DEGREE + 1> legendreAt(double x)
{
    std::array<double, TOP_DEGREE + 1> values{};
    values[0] = 1.0;
    values[1] = x;
    for (std::size_t k = 2; k <= TOP_DEGREE; ++k)
        values[k] = (static_cast<double>(2 * k - 1) * x * values[k - 1] - static_cast<double>(k - 1) * values[k - 2]) /
                    static_cast<double>(k);
    return values;
}

// The geometric factors of an element at one point, in groups of the same dimension (the components of a vector or
// of a tensor). An integral is compared on the largest scale of its group: a component may vanish on an element (a
// component of the normal of a flat plate), and its own scale is then the round-off.
constexpr std::size_t MAX_FACTORS = 64;
struct PointFactors
{
    std::array<double, MAX_FACTORS> values{};
    std::array<std::size_t, MAX_FACTORS> groups{};
    std::size_t count = 0;
    std::size_t group = 0;

    void add(double value)
    {
        values[count] = value;
        groups[count] = group;
        ++count;
    }
    void endGroup() { ++group; }
};

// The integrals of the factors against the test polynomials on one rule, and the integral of the absolute value of
// each factor (its scale).
struct FactorIntegrals
{
    std::array<double, MAX_FACTORS * TEST_COUNT> integrals{}; // [factor · TEST_COUNT + test]
    std::array<double, MAX_FACTORS> scales{};
    std::array<std::size_t, MAX_FACTORS> groups{};
    std::size_t count = 0;

    // The factors of a point of weight `weight`, where the test polynomials are `tests`.
    void add(const PointFactors& point, double weight, const std::array<double, TEST_COUNT>& tests)
    {
        count = point.count;
        for (std::size_t factor = 0; factor < point.count; ++factor)
        {
            const double weighted = weight * point.values[factor];
            groups[factor] = point.groups[factor];
            scales[factor] += std::fabs(weighted);
            for (std::size_t m = 0; m < TEST_COUNT; ++m)
                integrals[factor * TEST_COUNT + m] += weighted * tests[m];
        }
    }

    // The largest change of an integral from `coarser` to this rule, relative to the scale of its group here.
    [[nodiscard]] double changeFrom(const FactorIntegrals& coarser) const
    {
        std::array<double, MAX_FACTORS> groupScales{};
        for (std::size_t factor = 0; factor < count; ++factor)
            groupScales[groups[factor]] = std::max(groupScales[groups[factor]], scales[factor]);
        double largest = 0.0;
        for (std::size_t factor = 0; factor < count; ++factor)
        {
            const double scale = groupScales[groups[factor]];
            if (!(scale > 0.0))
                continue; // a group that vanishes on the element
            for (std::size_t m = 0; m < TEST_COUNT; ++m)
            {
                const std::size_t at = factor * TEST_COUNT + m;
                largest = std::max(largest, std::fabs(integrals[at] - coarser.integrals[at]) / scale);
            }
        }
        return largest;
    }
};

// The integrals of the geometric factors of an element on one rule: those of its surface points and of each edge.
struct RuleIntegrals
{
    FactorIntegrals surface;
    std::array<FactorIntegrals, 4> edges;

    [[nodiscard]] double changeFrom(const RuleIntegrals& coarser, std::size_t edgeCount) const
    {
        double largest = surface.changeFrom(coarser.surface);
        for (std::size_t k = 0; k < edgeCount; ++k)
            largest = std::max(largest, edges[k].changeFrom(coarser.edges[k]));
        return largest;
    }
};

// Whether the section laws vary over the element: a thickness that is not the same at its corners, or an offset
// (the laws of an offset section follow the curvature of the element).
bool hasVaryingSection(const ShellElementDefinition& shape)
{
    const std::array<double, 4>& thickness = shape.section->thickness();
    const bool isUniform =
        thickness[0] == thickness[1] && thickness[1] == thickness[2] && thickness[2] == thickness[3];
    return !isUniform || shape.offset != 0.0;
}

// The geometric factors of the surface integrands of an element at one point of geometry g. They are the products
// the bilinear forms of the element integrate (buildShellElement), without their polynomials, each a group:
//   mass and loads       j;   1 / j;   the normal ν;
//   membrane stiffness   Cᵀ C j,   C the map of the covariant strains to the local frame (covariantToVoigt);
//   moment compliance    Sᵀ S j,   S the Piola map of the moments (contravariantToLocal);
//   pairing              Sᵀ C j and Sᵀ C Γ^λ j, Γ^λ the Christoffel symbols of the curvature of a displacement
//                        (they vanish on an affine element: the group has the scale of Sᵀ C j);
//   shear stiffness      c cᵀ j,   c the covariant components of the local frame (the contravariant metric);
//   constant states      cos² φ, sin² φ, cos φ sin φ of the angle of their frame, times j;
// then the laws of a section that varies over the element, times j: the membrane stiffness, the inverse bending
// stiffness (the coupling of an unsymmetric laminate varies as they do), the shear stiffness, each inertia.
PointFactors surfaceFactors(const PointGeometry& g, const SectionLaws* laws, const SectionPoint* section)
{
    PointFactors factors;
    const double j = g.j;
    factors.add(j);
    factors.endGroup();
    factors.add(1.0 / j);
    factors.endGroup();
    for (std::size_t c = 0; c < 3; ++c)
        factors.add(g.nu[c]);
    factors.endGroup();
    const Matrix22 covariant = g.cov();
    const Matrix33 strainMap = covariantToVoigt(covariant);
    const Matrix33 momentMap = contravariantToLocal(g.con(), j);
    // the symmetric product MᵀM j of a map M
    const auto addGram = [&](const Matrix33& m)
    {
        for (std::size_t a = 0; a < 3; ++a)
            for (std::size_t b = a; b < 3; ++b)
                factors.add((m[0][a] * m[0][b] + m[1][a] * m[1][b] + m[2][a] * m[2][b]) * j);
        factors.endGroup();
    };
    addGram(strainMap);
    addGram(momentMap);
    for (std::size_t a = 0; a < 3; ++a)
        for (std::size_t b = 0; b < 3; ++b)
            factors.add((momentMap[0][a] * strainMap[0][b] + momentMap[1][a] * strainMap[1][b] +
                         momentMap[2][a] * strainMap[2][b]) *
                        j);
    for (std::size_t lambda = 0; lambda < 2; ++lambda)
    {
        const Matrix22& symbols = g.christoffel[lambda];
        const Voigt3 curvature = times33(strainMap, Voigt3{symbols[0][0], symbols[1][1], symbols[0][1]});
        const Voigt3 paired = transposeTimes33(momentMap, curvature);
        for (std::size_t r = 0; r < 3; ++r)
            factors.add(paired[r] * j);
    }
    factors.endGroup();
    for (std::size_t a = 0; a < 2; ++a)
        for (std::size_t b = a; b < 2; ++b)
            factors.add((covariant[a][0] * covariant[b][0] + covariant[a][1] * covariant[b][1]) * j);
    factors.endGroup();
    const double cosine = std::cos(g.phi);
    const double sine = std::sin(g.phi);
    factors.add(cosine * cosine * j);
    factors.add(sine * sine * j);
    factors.add(cosine * sine * j);
    factors.endGroup();
    if (laws == nullptr)
        return factors;
    for (const Matrix33* law : {&laws->abar, &laws->dinv})
    {
        for (std::size_t a = 0; a < 3; ++a)
            for (std::size_t b = a; b < 3; ++b)
                factors.add((*law)[a][b] * j);
        factors.endGroup();
    }
    for (std::size_t a = 0; a < 2; ++a)
        for (std::size_t b = a; b < 2; ++b)
            factors.add(laws->h[a][b] * j);
    factors.endGroup();
    for (std::size_t m = 0; m < 3; ++m)
    {
        factors.add(section->mass[m] * j);
        factors.endGroup();
    }
    return factors;
}

// The geometric factors of the edge integrands at one point of an edge, of geometry g, tangent X_τ of the map along
// the edge and weight W: the length |X_τ| of the edge (the membrane functionals, the loads), and those of the pairing
// of the normal moments with the rotation about the edge: the normal moment S^αβ (μ · a_α)(μ · a_β) / j² of a moment
// function, with the rotation unknown in the measure 1 / (|X_τ| W²), and with the derivative along the conormal
// μ · a^α of a displacement function times the normal.
PointFactors edgeFactors(const PointGeometry& g, const SpaceVector& tangentOfMap, double edgeWeight)
{
    PointFactors factors;
    const double length = norm3(tangentOfMap);
    const SpaceVector tangent = scl3(tangentOfMap, 1.0 / length);
    const SpaceVector conormal = cross3(tangent, g.nu);
    const std::array<double, 2> onCovariant{dot3(conormal, g.a[0]), dot3(conormal, g.a[1])};
    const std::array<double, 2> onContravariant{dot3(conormal, g.ac[0]), dot3(conormal, g.ac[1])};
    const double inverseArea = 1.0 / (g.j * g.j);
    const double inverseWeight = 1.0 / (edgeWeight * edgeWeight);
    factors.add(length);
    factors.endGroup();
    factors.add(1.0 / length);
    factors.endGroup();
    factors.add(inverseWeight);
    factors.endGroup();
    for (std::size_t c = 0; c < 3; ++c)
        factors.add(tangentOfMap[c]);
    factors.endGroup();
    // the normal moments of the three reference components
    const std::array<double, 3> normalMoment{onCovariant[0] * onCovariant[0] * inverseArea,
                                             onCovariant[1] * onCovariant[1] * inverseArea,
                                             onCovariant[0] * onCovariant[1] * inverseArea};
    for (std::size_t s = 0; s < 3; ++s)
        factors.add(normalMoment[s] * length);
    factors.endGroup();
    for (std::size_t s = 0; s < 3; ++s)
        factors.add(normalMoment[s] * inverseWeight);
    factors.endGroup();
    for (std::size_t s = 0; s < 3; ++s)
        for (std::size_t alpha = 0; alpha < 2; ++alpha)
            for (std::size_t c = 0; c < 3; ++c)
                factors.add(normalMoment[s] * onContravariant[alpha] * g.nu[c] * length);
    factors.endGroup();
    return factors;
}

// The geometry of the element at a point from the jets of its displacement functions there (`jetOf(i)`: value, ∂ξ,
// ∂η and, with `second`, the second derivatives) and the jet of its weight W. The map is X = (Σ x_i N_i) / W: the
// three components of the numerator are summed first and divided once, where the tables of an element divide each of
// its functions (elementJets). The geometry is the same to the round-off, for a fifth of the operations: the choice
// of the rule reads several rules for each element.
template <typename JetOf>
PointGeometry geometryOfMap(const ShellElementDefinition& shape, std::size_t functionCount, const JetOf& jetOf,
                            const ReferenceJet& weightJet, bool second)
{
    std::array<ReferenceJet, 3> numerator{};
    const std::size_t terms = second ? 6 : 3;
    for (std::size_t i = 0; i < functionCount; ++i)
    {
        const SpaceVector& x = shape.mapCoefficients[i];
        const auto& jet = jetOf(i);
        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t r = 0; r < terms; ++r)
                numerator[c][r] += x[c] * jet[r];
    }
    const std::array<ReferenceJet, 3> components{rationalJet(numerator[0], weightJet),
                                                 rationalJet(numerator[1], weightJet),
                                                 rationalJet(numerator[2], weightJet)};
    // X = Σ_c e_c X_c: the map of three functions whose coefficients are the axes of the basic system
    static constexpr std::array<SpaceVector, 3> BASIC_AXES{{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
    return geometryFrom(BASIC_AXES, [&](std::size_t c) { return components[c]; }, shape.axes, second);
}

// The integrals of the geometric factors of the element on the rule of `points` points per direction; nothing when
// a point of the element has no area.
std::optional<RuleIntegrals> integralsOn(std::size_t cornerCount, std::size_t degree,
                                         std::span<const std::size_t> edgeDegrees,
                                         const ShellElementDefinition& shape, std::size_t points)
{
    const ShellDisplacementTables& re = shellDisplacementTables(cornerCount, degree, points, edgeDegrees);
    const std::size_t n1 = re.h1Count;
    const bool readsSection = hasVaryingSection(shape);
    const std::array<TestDegrees, TEST_COUNT>& testDegrees = cornerCount == 3 ? TRIANGLE_TESTS : QUADRILATERAL_TESTS;
    RuleIntegrals out;
    std::array<double, TEST_COUNT> tests{};
    for (std::size_t q = 0; q < re.points.size(); ++q)
    {
        const double xi = re.points[q][0];
        const double eta = re.points[q][1];
        const PointGeometry g = geometryOfMap(
            shape, n1, [&](std::size_t i) -> const ReferenceJet& { return re.h1[q * n1 + i]; },
            shape.weight->jet(xi, eta), true);
        if (!(g.j > 0.0) || !std::isfinite(g.j))
            return std::nullopt;
        const std::array<double, TOP_DEGREE + 1> inXi = legendreAt(xi);
        const std::array<double, TOP_DEGREE + 1> inEta = legendreAt(eta);
        for (std::size_t m = 0; m < TEST_COUNT; ++m)
            tests[m] = inXi[testDegrees[m].xi] * inEta[testDegrees[m].eta];
        if (readsSection)
        {
            const SectionPoint section = sectionAtGeometry(*shape.section, shape.offset, xi, eta, g);
            const SectionLaws laws = sectionLaws(section);
            out.surface.add(surfaceFactors(g, &laws, &section), re.weights[q], tests);
        }
        else
        {
            out.surface.add(surfaceFactors(g, nullptr, nullptr), re.weights[q], tests);
        }
    }
    for (std::size_t k = 0; k < re.cornerCount; ++k)
        for (std::size_t gi = 0; gi < re.edgeParameters.size(); ++gi)
        {
            const auto [at, direction] = referenceEdgePointAndDirection(cornerCount, k, re.edgeParameters[gi]);
            const ReferenceJet weightJet = shape.weight->jet(at[0], at[1]);
            const PointGeometry g = geometryOfMap(
                shape, n1, [&](std::size_t i) -> const EdgeJet& { return re.h1Edge[k][gi * n1 + i]; },
                ReferenceJet{weightJet[0], weightJet[1], weightJet[2], 0.0, 0.0, 0.0}, false);
            const SpaceVector tangentOfMap = add3(scl3(g.a[0], direction[0]), scl3(g.a[1], direction[1]));
            if (!(g.j > 0.0) || !(norm3(tangentOfMap) > 0.0))
                return std::nullopt;
            const std::array<double, TOP_DEGREE + 1> along = legendreAt(2.0 * re.edgeParameters[gi] - 1.0);
            for (std::size_t m = 0; m < TEST_COUNT; ++m)
                tests[m] = along[EDGE_TESTS[m]];
            out.edges[k].add(edgeFactors(g, tangentOfMap, weightJet[0]), re.edgeWeights[gi], tests);
        }
    return out;
}

} // namespace

std::size_t chooseShellPoints(std::size_t cornerCount, std::size_t degree, const ShellElementDefinition& shape,
                              const ShellRuleChoice& choice, std::span<const std::size_t> edgeDegrees)
{
    const std::size_t largest = choice.maximumPoints;
    std::size_t points = std::min(choice.minimumPoints, largest);
    if (points == largest)
        return largest;
    std::optional<RuleIntegrals> current = integralsOn(cornerCount, degree, edgeDegrees, shape, points);
    // The change of the integrals from a rule to the next is the error of the first, the error of a Gauss rule on
    // these factors falling by about a constant factor at each point (their singularities are off the element).
    double previousChange = 0.0;
    while (points < largest)
    {
        std::optional<RuleIntegrals> next = integralsOn(cornerCount, degree, edgeDegrees, shape, points + 1);
        if (!current || !next)
            return largest;
        const double change = next->changeFrom(*current, cornerCount);
        if (change <= choice.tolerance)
            return points;
        // PERFORMANCE: two changes give the factor of the fall, and the rule it predicts to reach the tolerance; the
        // rules before it are passed over. The rule kept is always one whose own change was measured below the
        // tolerance: a prediction that falls short only costs the rules after it, read one by one. Measured on 2 304
        // distorted quadrilaterals (WebAssembly, one thread): the choice takes 0.28 ms an element, against 3.2 ms
        // for the matrices of one.
        std::size_t following = points + 1;
        if (previousChange > change && change > 0.0)
        {
            const double fall = change / previousChange;
            const double steps = std::ceil(std::log(choice.tolerance / change) / std::log(fall));
            if (steps >= static_cast<double>(largest - points))
                return largest;
            following = points + static_cast<std::size_t>(std::max(steps, 1.0));
        }
        if (following == points + 1)
        {
            current = std::move(next);
        }
        else
        {
            current = integralsOn(cornerCount, degree, edgeDegrees, shape, following);
        }
        previousChange = following == points + 1 ? change : 0.0;
        points = following;
    }
    return largest;
}

} // namespace oos::elements::mixedHybrid
