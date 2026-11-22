#include "CheckRunner.hpp"
#include "oos/base/threading/ThreadPool.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/shell/ShellReferenceElement.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSpaces.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace mh = oos::elements::mixedHybrid;

namespace
{

std::string text(std::size_t n)
{
    return std::to_string(n);
}

} // namespace

// Reference: tdnns/space.rs tests::dimensions.
CHECK_CASE("shell: dimensions of the spaces")
{
    for (std::size_t p = 0; p < 5; ++p)
    {
        runner.expect(mh::hhjSpace(4, p).size() == 3 * p * p + 10 * p + 5, "quadrilateral moments, p = " + text(p));
        runner.expect(mh::hhjSpace(3, p).size() == 3 * (p + 1) * (p + 2) / 2, "triangle moments, p = " + text(p));
        runner.expect(mh::reggeSpace(4, p).size() == 2 * (p + 1) * (p + 2) + (p + 1) * (p + 1),
                      "quadrilateral membrane, p = " + text(p));
        runner.expect(mh::hcurlSpace(4, p).functions.size() == 2 * (p + 1) * (p + 2),
                      "quadrilateral shear, p = " + text(p));
        if (p > 0)
            runner.expect(mh::hcurlSpace(3, p).functions.size() == (p + 1) * (p + 2), "triangle shear, p = " + text(p));
    }
    for (std::size_t k = 2; k < 6; ++k)
    {
        runner.expect(mh::h1Space(4, k).functions.size() == (k + 1) * (k + 1),
                      "quadrilateral displacement, k = " + text(k));
        runner.expect(mh::h1Space(3, k).functions.size() == (k + 1) * (k + 2) / 2,
                      "triangle displacement, k = " + text(k));
    }
}

// Reference: tdnns/space.rs tests::traces. A function of an edge has a zero trace on the other edges; an interior
// function on every edge.
CHECK_CASE("shell: traces of the edge and interior functions")
{
    const double ts[] = {0.1, 0.37, 0.8};
    for (std::size_t nv : {3u, 4u})
    {
        const mh::SpaceBasis h1 = mh::h1Space(nv, 4);
        for (std::size_t i = 0; i < h1.functions.size(); ++i)
        {
            const mh::FunctionKind& kind = h1.kinds[i];
            for (std::size_t e = 0; e < nv; ++e)
            {
                bool own = false;
                if (kind.place == mh::FunctionKind::Place::Vertex)
                    own = kind.entity == e || (kind.entity + nv - 1) % nv == e;
                else if (kind.place == mh::FunctionKind::Place::Edge)
                    own = kind.entity == e;
                if (own)
                    continue;
                for (double t : ts)
                {
                    const mh::ReferencePoint x = mh::referenceEdgePointAndDirection(nv, e, t).first;
                    runner.expect(std::fabs(h1.functions[i][0].evaluate(x[0], x[1])) < 1e-10,
                                  "displacement nv " + text(nv) + " function " + text(i) + " edge " + text(e));
                }
            }
        }
        const mh::SpaceBasis hc = mh::hcurlSpace(nv, 3);
        for (std::size_t i = 0; i < hc.functions.size(); ++i)
        {
            const mh::FunctionKind& kind = hc.kinds[i];
            for (std::size_t e = 0; e < nv; ++e)
            {
                if (kind.place == mh::FunctionKind::Place::Edge && kind.entity == e)
                    continue;
                for (double t : ts)
                {
                    const auto [x, tv] = mh::referenceEdgePointAndDirection(nv, e, t);
                    const mh::Field& f = hc.functions[i];
                    const double g = f[0].evaluate(x[0], x[1]) * tv[0] + f[1].evaluate(x[0], x[1]) * tv[1];
                    runner.expect(std::fabs(g) < 1e-10,
                                  "shear nv " + text(nv) + " function " + text(i) + " edge " + text(e));
                }
            }
        }
    }
}

// The duality of the bases: the corner values and the edge moments of each displacement function are those of its
// kind (1 for its own functional, 0 for the others).
CHECK_CASE("shell: dual basis of the displacement")
{
    const std::size_t pointsPerDirection = 10;
    for (std::size_t nv : {3u, 4u})
    {
        const mh::SpaceBasis h1 = mh::h1Space(nv, 4);
        double worst = 0.0;
        for (std::size_t i = 0; i < h1.functions.size(); ++i)
        {
            const mh::Polynomial2& f = h1.functions[i][0];
            for (std::size_t v = 0; v < nv; ++v)
            {
                const mh::ReferencePoint x = mh::referenceCorner(nv, v);
                const double expected = (h1.kinds[i] == mh::FunctionKind::vertex(v)) ? 1.0 : 0.0;
                worst = std::fmax(worst, std::fabs(f.evaluate(x[0], x[1]) - expected));
            }
            for (std::size_t e = 0; e < nv; ++e)
            {
                for (std::size_t l = 0; l < 3; ++l)
                {
                    double moment = 0.0;
                    for (const mh::LineRulePoint& point : mh::unitIntervalRule(pointsPerDirection))
                    {
                        const mh::ReferencePoint x = mh::referenceEdgePointAndDirection(nv, e, point.point).first;
                        moment += point.weight * f.evaluate(x[0], x[1]) * mh::legendreValue(l, 2.0 * point.point - 1.0);
                    }
                    const double expected = (h1.kinds[i] == mh::FunctionKind::edge(e, l)) ? 1.0 : 0.0;
                    worst = std::fmax(worst, std::fabs(moment - expected));
                }
            }
        }
        runner.expect(worst < 1e-11, "displacement duality nv " + text(nv) + ": " + std::to_string(worst));
    }
}

// The counts of the reference element at k = 2 … 4: displacement (k + 1)² or (k + 1)(k + 2)/2, shear, moments
// (3p² + 10p + 5 on the quadrilateral) and membrane strains, the membrane functionals matching the membrane space.
CHECK_CASE("shell: reference element tables")
{
    for (std::size_t k = 2; k <= 4; ++k)
    {
        const std::size_t p = k - 1;
        for (std::size_t nv : {3u, 4u})
        {
            const std::size_t n = mh::shellPointsPerDirection(k);
            const mh::ShellReferenceElement& re = mh::shellReferenceElement(nv, k, n);
            const std::string where = "nv " + text(nv) + " k " + text(k);
            const bool quad = nv == 4;
            runner.expect(re.h1Count == (quad ? (k + 1) * (k + 1) : (k + 1) * (k + 2) / 2), "h1 count " + where);
            runner.expect(re.hcurlCount == (quad ? 2 * (p + 1) * (p + 2) : (p + 1) * (p + 2)), "shear count " + where);
            runner.expect(re.momentCount == (quad ? 3 * p * p + 10 * p + 5 : 3 * (p + 1) * (p + 2) / 2),
                          "moment count " + where);
            const std::size_t membrane = quad ? 2 * (p + 1) * (p + 2) + (p + 1) * (p + 1) : 3 * (p + 1) * (p + 2) / 2;
            runner.expect(re.reggeCount == membrane, "membrane count " + where);
            runner.expect(nv * (p + 1) + re.reggeInteriorMoments.size() + 3 == re.reggeCount,
                          "membrane functionals " + where);
            runner.expect(re.points.size() == n * n && re.edgeParameters.size() == n, "rule sizes " + where);
            runner.expect(re.h1.size() == n * n * re.h1Count && re.moments.size() == n * n * re.momentCount,
                          "table sizes " + where);
            runner.expect(re.h1Edge.size() == nv && re.h1Edge[0].size() == n * re.h1Count, "edge table sizes " + where);
            runner.expect(re.reggeTest.size() == n * n * re.reggeInteriorMoments.size(), "test table size " + where);
            // the weights of the rule sum to the area of the reference domain
            double area = 0.0;
            for (double w : re.weights)
                area += w;
            runner.expectNear(area, quad ? 4.0 : 2.0, 1e-13, "area " + where);
        }
    }
    // the cache hands out one object per key, also when asked from several threads
    const mh::ShellReferenceElement* first = &mh::shellReferenceElement(4, 4, 12);
    std::vector<const mh::ShellReferenceElement*> seen(4, nullptr);
    oos::base::ThreadPool pool(4);
    pool.runBatch(seen.size(), [&seen](std::size_t t) { seen[t] = &mh::shellReferenceElement(4, 4, 12); });
    for (const mh::ShellReferenceElement* s : seen)
        runner.expect(s == first, "one shared reference element per key");
}

// The constants of the moment and membrane spaces: the last three functions are the reference constants, the others
// orthonormal and orthogonal to them.
CHECK_CASE("shell: constants of the tensor spaces")
{
    for (std::size_t nv : {3u, 4u})
    {
        const mh::ShellReferenceElement& re = mh::shellReferenceElement(nv, 4, 12);
        for (const std::vector<mh::Field>* basis : {&re.momentBasis, &re.reggeBasis})
        {
            const std::size_t n = basis->size();
            double worstConstant = 0.0;
            double worstOrthogonality = 0.0;
            for (std::size_t c = 0; c < 3; ++c)
            {
                const mh::Field& f = (*basis)[n - 3 + c];
                for (std::size_t comp = 0; comp < 3; ++comp)
                {
                    const double expected = comp == c ? 1.0 : 0.0;
                    worstConstant = std::fmax(worstConstant, std::fabs(f[comp].evaluate(0.3, -0.4) - expected));
                }
            }
            for (std::size_t i = 0; i < n - 3; ++i)
            {
                for (std::size_t j = 0; j < n; ++j)
                {
                    double product = 0.0;
                    for (std::size_t q = 0; q < re.points.size(); ++q)
                    {
                        const mh::ReferencePoint x = re.points[q];
                        for (std::size_t comp = 0; comp < 3; ++comp)
                            product += re.weights[q] * (*basis)[i][comp].evaluate(x[0], x[1]) *
                                       (*basis)[j][comp].evaluate(x[0], x[1]);
                    }
                    const double expected = (i == j) ? 1.0 : 0.0;
                    worstOrthogonality = std::fmax(worstOrthogonality, std::fabs(product - expected));
                }
            }
            runner.expect(worstConstant == 0.0, "constants nv " + text(nv));
            runner.expect(worstOrthogonality < 1e-11,
                          "orthonormal complement nv " + text(nv) + ": " + std::to_string(worstOrthogonality));
        }
    }
}

// The spaces of a transition element (ShellSpaces.hpp; not in the reference): on an edge of degree n the trace of a
// displacement function is a polynomial of degree n at most whose corner values and moments l ≤ n − 2 are those of
// its kind; the tangential trace of a shear function and the normal-normal trace of a moment have the degree n − 1 at
// most. The moments stay orthonormal, their three constants last.
CHECK_CASE("shell: traces of the spaces of a transition element")
{
    struct Pattern
    {
        std::size_t cornerCount;
        std::size_t degree;
        std::vector<std::size_t> edgeDegrees;
    };
    const std::vector<Pattern> patterns{{4, 3, {2, 3, 3, 2}}, {4, 3, {2, 2, 2, 2}}, {4, 4, {3, 2, 4, 3}},
                                        {3, 3, {2, 3, 2}},    {3, 4, {2, 2, 3}},    {3, 4, {4, 3, 4}}};
    const std::vector<mh::LineRulePoint> rule = mh::unitIntervalRule(10);
    // the moment l of a function of the parameter of edge e
    const auto momentOf = [&](std::size_t nv, std::size_t e, std::size_t l, const auto& valueAt)
    {
        double moment = 0.0;
        for (const mh::LineRulePoint& point : rule)
        {
            const mh::ReferencePoint x = mh::referenceEdgePointAndDirection(nv, e, point.point).first;
            moment += point.weight * valueAt(x) * mh::legendreValue(l, 2.0 * point.point - 1.0);
        }
        return moment;
    };
    for (const Pattern& pattern : patterns)
    {
        const std::size_t nv = pattern.cornerCount;
        const std::size_t k = pattern.degree;
        const std::size_t p = k - 1;
        std::string name = (nv == 3 ? "triangle " : "quadrilateral ") + text(k) + ", edges";
        std::vector<std::size_t> traceDegrees; // of the shear and of the moments: one less
        std::size_t lowered = 0;
        for (const std::size_t n : pattern.edgeDegrees)
        {
            name += " " + text(n);
            traceDegrees.push_back(n - 1);
            lowered += k - n;
        }
        const mh::SpaceBasis fullH1 = mh::h1Space(nv, k);
        const mh::SpaceBasis h1 = mh::h1TransitionSpace(fullH1, nv, k, pattern.edgeDegrees);
        runner.expect(h1.functions.size() == fullH1.functions.size() - lowered, name + ": displacement functions");
        double worst = 0.0;
        for (std::size_t i = 0; i < h1.functions.size(); ++i)
        {
            const mh::Polynomial2& f = h1.functions[i][0];
            const auto value = [&](const mh::ReferencePoint& x) { return f.evaluate(x[0], x[1]); };
            for (std::size_t v = 0; v < nv; ++v)
            {
                const double expected = (h1.kinds[i] == mh::FunctionKind::vertex(v)) ? 1.0 : 0.0;
                worst = std::fmax(worst, std::fabs(value(mh::referenceCorner(nv, v)) - expected));
            }
            for (std::size_t e = 0; e < nv; ++e)
            {
                const std::size_t n = pattern.edgeDegrees[e];
                // its own moments, and no Legendre coefficient above the degree of the edge
                for (std::size_t l = 0; l + 1 < n; ++l)
                {
                    const double expected = (h1.kinds[i] == mh::FunctionKind::edge(e, l)) ? 1.0 : 0.0;
                    worst = std::fmax(worst, std::fabs(momentOf(nv, e, l, value) - expected));
                }
                for (std::size_t l = n + 1; l <= k; ++l)
                    worst = std::fmax(worst, std::fabs(momentOf(nv, e, l, value)));
            }
        }
        runner.expect(worst < 1e-10, name + ": displacement traces " + std::to_string(worst));

        const mh::SpaceBasis fullShear = mh::hcurlSpace(nv, p);
        const mh::SpaceBasis shear = mh::hcurlTransitionSpace(fullShear, p, traceDegrees);
        runner.expect(shear.functions.size() == fullShear.functions.size() - lowered, name + ": shear functions");
        worst = 0.0;
        for (const mh::Field& f : shear.functions)
            for (std::size_t e = 0; e < nv; ++e)
            {
                const mh::ReferencePoint d = mh::referenceEdgePointAndDirection(nv, e, 0.0).second;
                const auto tangential = [&](const mh::ReferencePoint& x)
                { return d[0] * f[0].evaluate(x[0], x[1]) + d[1] * f[1].evaluate(x[0], x[1]); };
                for (std::size_t l = traceDegrees[e] + 1; l <= p; ++l)
                    worst = std::fmax(worst, std::fabs(momentOf(nv, e, l, tangential)));
            }
        runner.expect(worst < 1e-10, name + ": shear traces " + std::to_string(worst));

        const std::vector<mh::Field> fullMoments = mh::splitConstants(mh::hhjSpace(nv, p), nv);
        const std::vector<mh::Field> moments = mh::momentTransitionSpace(fullMoments, fullH1, nv, p, traceDegrees);
        // one function less for each rotation and for each displacement function left out
        runner.expect(moments.size() == fullMoments.size() - 2 * lowered, name + ": moment functions");
        worst = 0.0;
        for (const mh::Field& s : moments)
            for (std::size_t e = 0; e < nv; ++e)
            {
                const mh::ReferencePoint d = mh::referenceEdgePointAndDirection(nv, e, 0.0).second;
                const double n1 = d[1];
                const double n2 = -d[0];
                const auto normal = [&](const mh::ReferencePoint& x)
                {
                    return s[0].evaluate(x[0], x[1]) * n1 * n1 + s[1].evaluate(x[0], x[1]) * n2 * n2 +
                           2.0 * s[2].evaluate(x[0], x[1]) * n1 * n2;
                };
                for (std::size_t l = traceDegrees[e] + 1; l <= p; ++l)
                    worst = std::fmax(worst, std::fabs(momentOf(nv, e, l, normal)));
            }
        runner.expect(worst < 1e-10, name + ": moment traces " + std::to_string(worst));
        // Each removed scalar displacement has two in-plane strain directions.
        const std::vector<mh::Field> fullRegge = mh::splitConstants(mh::reggeSpace(nv, p), nv);
        const mh::MembraneTransition membrane = mh::membraneTransition(fullRegge, fullH1, nv, p, traceDegrees);
        runner.expect(membrane.strains.size() == fullRegge.size() - 2 * lowered, name + ": membrane functions");
        runner.expect(membrane.functionals.rows() + 3 == membrane.strains.size(), name + ": membrane functionals");
        // The functions before the three constants are orthonormal, and orthogonal to the constants (the product of
        // the spaces: the sum of the products of the components 11, 22 and 12 over the reference domain).
        worst = 0.0;
        const std::vector<mh::SurfaceRulePoint> surface = mh::referenceSurfaceRule(nv, 10);
        const std::size_t constants = moments.size() - 3;
        for (std::size_t a = 0; a < constants; ++a)
            for (std::size_t b = a; b < moments.size(); ++b)
            {
                double gram = 0.0;
                for (const mh::SurfaceRulePoint& point : surface)
                    for (std::size_t c = 0; c < 3; ++c)
                        gram += point.weight * moments[a][c].evaluate(point.point[0], point.point[1]) *
                                moments[b][c].evaluate(point.point[0], point.point[1]);
                worst = std::fmax(worst, std::fabs(gram - (a == b ? 1.0 : 0.0)));
            }
        runner.expect(worst < 1e-9, name + ": moments orthonormal " + std::to_string(worst));
    }
}

// With every edge at the degree of the element, the transition spaces are the full ones, to the bit.
CHECK_CASE("shell: the transition spaces of an element of one degree are its spaces")
{
    for (std::size_t nv : {3u, 4u})
    {
        const std::vector<std::size_t> edges(nv, 3);
        const std::vector<std::size_t> traces(nv, 2);
        const mh::SpaceBasis h1 = mh::h1Space(nv, 3);
        runner.expect(mh::h1TransitionSpace(h1, nv, 3, edges).functions == h1.functions,
                      "displacement, nv " + text(nv));
        const mh::SpaceBasis shear = mh::hcurlSpace(nv, 2);
        runner.expect(mh::hcurlTransitionSpace(shear, 2, traces).functions == shear.functions, "shear, nv " + text(nv));
        const std::vector<mh::Field> moments = mh::splitConstants(mh::hhjSpace(nv, 2), nv);
        runner.expect(mh::momentTransitionSpace(moments, h1, nv, 2, traces) == moments, "moments, nv " + text(nv));
        const std::vector<mh::Field> regge = mh::splitConstants(mh::reggeSpace(nv, 2), nv);
        const mh::MembraneTransition membrane = mh::membraneTransition(regge, h1, nv, 2, traces);
        runner.expect(membrane.strains == regge && membrane.functionals.empty(),
                      "membrane strains, nv " + text(nv));
    }
}
