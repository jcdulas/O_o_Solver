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
