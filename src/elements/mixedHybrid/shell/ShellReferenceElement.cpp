#include "oos/elements/mixedHybrid/shell/ShellReferenceElement.hpp"

#include "ShellSpacesDetail.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace oos::elements::mixedHybrid
{

namespace
{

// The lowest degree of the displacement: the edge rotation and the shear are of degree k − 1 ≥ 1.
constexpr std::size_t LOWEST_DEGREE = 2;

// x^n by repeated products.
double integerPower(double x, std::size_t n)
{
    double r = 1.0;
    for (std::size_t k = 0; k < n; ++k)
        r *= x;
    return r;
}

// The polynomial and its derivatives ∂ξ, ∂η, ∂ξξ, ∂ξη, ∂ηη.
// Reference: tdnns/space.rs::derivs
std::array<Polynomial2, 6> derivatives(const Polynomial2& p)
{
    const Polynomial2 dx = p.dXi();
    const Polynomial2 dy = p.dEta();
    return {p, dx, dy, dx.dXi(), dx.dEta(), dy.dEta()};
}

std::array<double, 3> evaluate3(const Field& f, const ReferencePoint& x)
{
    return {f[0].evaluate(x[0], x[1]), f[1].evaluate(x[0], x[1]), f[2].evaluate(x[0], x[1])};
}

// The matrix of the membrane functionals on the membrane basis (exact rules): row after row the tangential moments of
// each edge, the interior moments, then the moments of the three constants. Its unisolvence on the reference domain
// is checked here; an element checks its own (with the work of the constant forces in place of the last three).
void checkReggeUnisolvence(const ShellReferenceElement& re)
{
    const std::size_t nv = re.cornerCount;
    const std::size_t nr = re.reggeCount;
    const std::vector<SurfaceRulePoint> surface = referenceSurfaceRule(nv, shellspaces::BUILD_POINTS);
    const std::vector<LineRulePoint> edge = unitIntervalRule(shellspaces::BUILD_POINTS);
    std::vector<ReggeMoment> moments = re.reggeInteriorMoments;
    moments.push_back({0, 0, 0});
    moments.push_back({1, 0, 0});
    moments.push_back({2, 0, 0});
    Matrix functionals(nr, nr);
    for (std::size_t m = 0; m < nr; ++m)
    {
        const Field& f = re.reggeBasis[m];
        std::size_t row = 0;
        for (std::size_t e = 0; e < nv; ++e)
        {
            for (std::size_t l = 0; l <= re.shearDegree; ++l)
            {
                double sum = 0.0;
                for (const LineRulePoint& point : edge)
                {
                    const auto [x, tv] = referenceEdgePointAndDirection(nv, e, point.point);
                    const std::array<double, 3> r = evaluate3(f, x);
                    sum += point.weight * (r[0] * tv[0] * tv[0] + r[1] * tv[1] * tv[1] + 2.0 * r[2] * tv[0] * tv[1]) *
                           legendreValue(l, 2.0 * point.point - 1.0);
                }
                functionals(row++, m) = sum;
            }
        }
        for (const ReggeMoment& moment : moments)
        {
            double sum = 0.0;
            for (const SurfaceRulePoint& point : surface)
                sum += point.weight * f[moment.component].evaluate(point.point[0], point.point[1]) *
                       integerPower(point.point[0], moment.i) * integerPower(point.point[1], moment.j);
            functionals(row++, m) = sum;
        }
    }
    if (LuDecomposition(functionals).isSingular())
        throw base::InternalError("membrane functionals not unisolvent on the reference element");
}

} // namespace

ShellReferenceElement ShellReferenceElement::build(std::size_t cornerCount, std::size_t degree,
                                                   std::size_t pointsPerDirection)
{
    if (degree < LOWEST_DEGREE)
        throw base::InternalError("the shell needs a displacement of degree 2 at least");
    const std::size_t nv = cornerCount;
    const std::size_t p = degree - 1;
    ShellReferenceElement re;
    re.cornerCount = nv;
    re.degree = degree;
    re.shearDegree = p;
    SpaceBasis h1 = h1Space(nv, degree);
    SpaceBasis hc = hcurlSpace(nv, p);
    re.h1Basis = std::move(h1.functions);
    re.h1Kinds = std::move(h1.kinds);
    re.hcurlBasis = std::move(hc.functions);
    re.hcurlKinds = std::move(hc.kinds);
    re.momentBasis = splitConstants(hhjSpace(nv, p), nv);
    re.reggeBasis = splitConstants(reggeSpace(nv, p), nv);
    re.h1Count = re.h1Basis.size();
    re.hcurlCount = re.hcurlBasis.size();
    re.momentCount = re.momentBasis.size();
    re.reggeCount = re.reggeBasis.size();

    for (const SurfaceRulePoint& point : referenceSurfaceRule(nv, pointsPerDirection))
    {
        re.points.push_back(point.point);
        re.weights.push_back(point.weight);
    }
    // the derivatives of every basis polynomial, once
    std::vector<std::array<Polynomial2, 6>> h1Derivatives;
    for (const Field& f : re.h1Basis)
        h1Derivatives.push_back(derivatives(f[0]));
    std::vector<std::array<Polynomial2, 6>> hcurlDerivatives;
    for (const Field& f : re.hcurlBasis)
        hcurlDerivatives.push_back({f[0], f[1], f[0].dXi(), f[0].dEta(), f[1].dXi(), f[1].dEta()});
    for (const ReferencePoint& x : re.points)
    {
        for (const auto& d : h1Derivatives)
        {
            ReferenceJet jet{};
            for (std::size_t r = 0; r < 6; ++r)
                jet[r] = d[r].evaluate(x[0], x[1]);
            re.h1.push_back(jet);
        }
        for (const auto& d : hcurlDerivatives)
        {
            std::array<double, 6> values{};
            for (std::size_t r = 0; r < 6; ++r)
                values[r] = d[r].evaluate(x[0], x[1]);
            re.hcurl.push_back(values);
        }
        for (const Field& f : re.momentBasis)
            re.moments.push_back(evaluate3(f, x));
        for (const Field& f : re.reggeBasis)
            re.regge.push_back(evaluate3(f, x));
    }

    for (const LineRulePoint& point : unitIntervalRule(pointsPerDirection))
    {
        re.edgeParameters.push_back(point.point);
        re.edgeWeights.push_back(point.weight);
    }
    re.h1Edge.resize(nv);
    re.hcurlEdge.resize(nv);
    re.momentsEdge.resize(nv);
    re.reggeEdge.resize(nv);
    for (std::size_t e = 0; e < nv; ++e)
    {
        for (double t : re.edgeParameters)
        {
            const ReferencePoint x = referenceEdgePointAndDirection(nv, e, t).first;
            for (const auto& d : h1Derivatives)
                re.h1Edge[e].push_back(
                    {d[0].evaluate(x[0], x[1]), d[1].evaluate(x[0], x[1]), d[2].evaluate(x[0], x[1])});
            for (const Field& f : re.hcurlBasis)
                re.hcurlEdge[e].push_back({f[0].evaluate(x[0], x[1]), f[1].evaluate(x[0], x[1])});
            for (const Field& f : re.momentBasis)
                re.momentsEdge[e].push_back(evaluate3(f, x));
            for (const Field& f : re.reggeBasis)
                re.reggeEdge[e].push_back(evaluate3(f, x));
        }
    }

    // the membrane functionals: nv (p + 1) tangential edge moments, the interior moments, the three constants
    re.reggeInteriorMoments = reggeInterior(nv, p);
    if (nv * (p + 1) + re.reggeInteriorMoments.size() + 3 != re.reggeCount)
        throw base::InternalError("membrane functionals do not match the membrane space");
    checkReggeUnisolvence(re);

    // the test polynomials of zero mean on the reference domain: the complement of the constants that an affine map
    // of the domain onto itself keeps (a permutation of the corners of the triangle)
    const std::vector<SurfaceRulePoint> exact = referenceSurfaceRule(nv, shellspaces::BUILD_POINTS);
    double area = 0.0;
    for (const SurfaceRulePoint& point : exact)
        area += point.weight;
    std::vector<double> means;
    for (const ReggeMoment& moment : re.reggeInteriorMoments)
    {
        double sum = 0.0;
        for (const SurfaceRulePoint& point : exact)
            sum += point.weight * integerPower(point.point[0], moment.i) * integerPower(point.point[1], moment.j);
        means.push_back(sum / area);
    }
    for (const ReferencePoint& x : re.points)
        for (std::size_t m = 0; m < re.reggeInteriorMoments.size(); ++m)
            re.reggeTest.push_back(integerPower(x[0], re.reggeInteriorMoments[m].i) *
                                       integerPower(x[1], re.reggeInteriorMoments[m].j) -
                                   means[m]);
    return re;
}

const ShellReferenceElement& shellReferenceElement(std::size_t cornerCount, std::size_t degree,
                                                   std::size_t pointsPerDirection)
{
    using Key = std::tuple<std::size_t, std::size_t, std::size_t>;
    static std::mutex mutex;
    static std::map<Key, std::unique_ptr<const ShellReferenceElement>> cache;
    const Key key{cornerCount, degree, pointsPerDirection};
    const std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(key);
    if (found == cache.end())
    {
        auto built = std::make_unique<const ShellReferenceElement>(
            ShellReferenceElement::build(cornerCount, degree, pointsPerDirection));
        found = cache.emplace(key, std::move(built)).first;
    }
    return *found->second;
}

} // namespace oos::elements::mixedHybrid
