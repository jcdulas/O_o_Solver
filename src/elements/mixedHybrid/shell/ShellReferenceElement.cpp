#include "oos/elements/mixedHybrid/shell/ShellReferenceElement.hpp"

#include "ShellSpacesDetail.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"

#include <array>
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
    const std::size_t raw = nv * (re.shearDegree + 1) + re.reggeInteriorMoments.size() + 3;
    Matrix functionals(nr, nr);
    for (std::size_t m = 0; m < nr; ++m)
    {
        const Field& f = re.reggeBasis[m];
        Vector column(raw);
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
                column[row++] = sum;
            }
        }
        for (const ReggeMoment& moment : moments)
        {
            double sum = 0.0;
            for (const SurfaceRulePoint& point : surface)
                sum += point.weight * f[moment.component].evaluate(point.point[0], point.point[1]) *
                       integerPower(point.point[0], moment.i) * integerPower(point.point[1], moment.j);
            column[row++] = sum;
        }
        const Vector kept = re.membraneFunctionals(std::move(column));
        for (std::size_t i = 0; i < nr; ++i)
            functionals(i, m) = kept[i];
    }
    if (LuDecomposition(functionals).isSingular())
        throw base::InternalError("membrane functionals not unisolvent on the reference element");
}

} // namespace

namespace
{

// The bases of `re` at a reference point, from their polynomials.
ReferencePointValues valuesAtPoint(const ShellReferenceElement& re, const ReferencePoint& point)
{
    const double xi = point[0];
    const double eta = point[1];
    ReferencePointValues values;
    values.point = point;
    for (const Field& f : re.h1Basis)
        values.h1.push_back(f[0].jet(xi, eta));
    for (const Field& f : re.hcurlBasis)
        values.hcurl.push_back({f[0].evaluate(xi, eta), f[1].evaluate(xi, eta)});
    for (const Field& f : re.momentBasis)
        values.moments.push_back({f[0].evaluate(xi, eta), f[1].evaluate(xi, eta), f[2].evaluate(xi, eta)});
    for (const Field& f : re.reggeBasis)
        values.regge.push_back({f[0].evaluate(xi, eta), f[1].evaluate(xi, eta), f[2].evaluate(xi, eta)});
    return values;
}

} // namespace

namespace
{

// The spaces of the elements of `cornerCount` corners and degree k with every edge at that degree: the displacement
// and the shear, the moments and the membrane strains with their constants last (splitConstants).
struct FullSpaces
{
    SpaceBasis h1;
    SpaceBasis hcurl;
    std::vector<Field> moments;
    std::vector<Field> regge;
};

// Those of (cornerCount, degree), built on the first call and shared (safe to call from several threads).
// PERFORMANCE: the reference elements of one degree differ by their rule and by the degrees of their edges, and a
// model whose shells take their degree from their shape (ShellDegreeChoice.hpp) builds tens of them; their moments
// and their membrane strains are the same, and take most of the build (their orthonormal bases). Measured on 256
// quadrilaterals with their degrees chosen (WebAssembly, one thread): the reference elements from 379 ms to 126 ms.
const FullSpaces& fullSpacesOf(std::size_t cornerCount, std::size_t degree)
{
    static std::mutex mutex;
    static std::map<std::pair<std::size_t, std::size_t>, std::unique_ptr<const FullSpaces>> cache;
    const std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find({cornerCount, degree});
    if (found == cache.end())
    {
        const std::size_t p = degree - 1;
        auto built = std::make_unique<FullSpaces>();
        built->h1 = h1Space(cornerCount, degree);
        built->hcurl = hcurlSpace(cornerCount, p);
        built->moments = splitConstants(hhjSpace(cornerCount, p), cornerCount);
        built->regge = splitConstants(reggeSpace(cornerCount, p), cornerCount);
        found = cache.emplace(std::pair{cornerCount, degree}, std::move(built)).first;
    }
    return *found->second;
}

} // namespace

namespace
{

// The degree of the trace of each edge of an element, all `degree` where none is given; checks them.
using EdgeDegrees = std::array<std::size_t, 4>;
using ReferenceKey = std::tuple<std::size_t, std::size_t, std::size_t, EdgeDegrees>;

EdgeDegrees edgeDegreesOf(std::size_t cornerCount, std::size_t degree, std::span<const std::size_t> edgeDegrees)
{
    if (cornerCount != 3 && cornerCount != 4)
        throw base::InternalError("the shell needs three or four corners");
    if (degree < LOWEST_DEGREE)
        throw base::InternalError("the shell needs a displacement of degree 2 at least");
    if (!edgeDegrees.empty() && edgeDegrees.size() != cornerCount)
        throw base::InternalError("the shell needs one degree per edge");
    EdgeDegrees degrees{};
    for (std::size_t e = 0; e < cornerCount; ++e)
    {
        degrees[e] = edgeDegrees.empty() ? degree : edgeDegrees[e];
        if (degrees[e] < LOWEST_DEGREE || degrees[e] > degree)
            throw base::InternalError("the degree of an edge of a shell is from 2 to the degree of the shell");
    }
    return degrees;
}

// The functions of the displacement of an element with its tables on a rule.
struct DisplacementPart
{
    SpaceBasis basis;
    ShellDisplacementTables tables;
};

DisplacementPart displacementPart(std::size_t cornerCount, std::size_t degree, std::size_t pointsPerDirection,
                                  std::span<const std::size_t> edgeDegrees)
{
    DisplacementPart part;
    part.basis = h1TransitionSpace(fullSpacesOf(cornerCount, degree).h1, cornerCount, degree, edgeDegrees);
    ShellDisplacementTables& tables = part.tables;
    tables.cornerCount = cornerCount;
    tables.h1Count = part.basis.functions.size();
    for (const SurfaceRulePoint& point : referenceSurfaceRule(cornerCount, pointsPerDirection))
    {
        tables.points.push_back(point.point);
        tables.weights.push_back(point.weight);
    }
    // the derivatives of every basis polynomial, once
    std::vector<std::array<Polynomial2, 6>> h1Derivatives;
    for (const Field& f : part.basis.functions)
        h1Derivatives.push_back(derivatives(f[0]));
    for (const ReferencePoint& x : tables.points)
        for (const auto& d : h1Derivatives)
        {
            ReferenceJet jet{};
            for (std::size_t r = 0; r < 6; ++r)
                jet[r] = d[r].evaluate(x[0], x[1]);
            tables.h1.push_back(jet);
        }
    for (const LineRulePoint& point : unitIntervalRule(pointsPerDirection))
    {
        tables.edgeParameters.push_back(point.point);
        tables.edgeWeights.push_back(point.weight);
    }
    tables.h1Edge.resize(cornerCount);
    for (std::size_t e = 0; e < cornerCount; ++e)
        for (const double tau : tables.edgeParameters)
        {
            const ReferencePoint x = referenceEdgePointAndDirection(cornerCount, e, tau).first;
            for (const auto& d : h1Derivatives)
                tables.h1Edge[e].push_back(
                    {d[0].evaluate(x[0], x[1]), d[1].evaluate(x[0], x[1]), d[2].evaluate(x[0], x[1])});
        }
    return part;
}

} // namespace

std::size_t ShellReferenceElement::firstRotation(std::size_t edge) const
{
    std::size_t first = 0;
    for (std::size_t e = 0; e < edge; ++e)
        first += rotationCount(e);
    return first;
}

ShellReferenceElement ShellReferenceElement::build(std::size_t cornerCount, std::size_t degree,
                                                   std::size_t pointsPerDirection,
                                                   std::span<const std::size_t> edgeDegrees)
{
    const std::size_t nv = cornerCount;
    const std::size_t p = degree - 1;
    ShellReferenceElement re;
    re.cornerCount = nv;
    re.degree = degree;
    re.shearDegree = p;
    const EdgeDegrees degrees = edgeDegreesOf(nv, degree, edgeDegrees);
    re.edgeDegrees.assign(degrees.begin(), degrees.begin() + nv);
    std::vector<std::size_t> shearEdgeDegrees; // of the tangential trace of the shear: one less
    for (const std::size_t edgeDegree : re.edgeDegrees)
        shearEdgeDegrees.push_back(edgeDegree - 1);
    const FullSpaces& full = fullSpacesOf(nv, degree);
    DisplacementPart displacement = displacementPart(nv, degree, pointsPerDirection, re.edgeDegrees);
    SpaceBasis hc = hcurlTransitionSpace(full.hcurl, p, shearEdgeDegrees);
    re.h1Basis = std::move(displacement.basis.functions);
    re.h1Kinds = std::move(displacement.basis.kinds);
    re.hcurlBasis = std::move(hc.functions);
    re.hcurlKinds = std::move(hc.kinds);
    re.momentBasis = momentTransitionSpace(full.moments, full.h1, nv, p, shearEdgeDegrees);
    MembraneTransition membrane = membraneTransition(full.regge, full.h1, nv, p, shearEdgeDegrees);
    re.reggeBasis = std::move(membrane.strains);
    re.membraneCombination = std::move(membrane.functionals);
    re.h1Count = re.h1Basis.size();
    re.hcurlCount = re.hcurlBasis.size();
    re.momentCount = re.momentBasis.size();
    re.reggeCount = re.reggeBasis.size();

    re.points = std::move(displacement.tables.points);
    re.weights = std::move(displacement.tables.weights);
    re.h1 = std::move(displacement.tables.h1);
    re.edgeParameters = std::move(displacement.tables.edgeParameters);
    re.edgeWeights = std::move(displacement.tables.edgeWeights);
    re.h1Edge = std::move(displacement.tables.h1Edge);
    // the derivatives of every basis polynomial of the shear, once
    std::vector<std::array<Polynomial2, 6>> hcurlDerivatives;
    for (const Field& f : re.hcurlBasis)
        hcurlDerivatives.push_back({f[0], f[1], f[0].dXi(), f[0].dEta(), f[1].dXi(), f[1].dEta()});
    for (const ReferencePoint& x : re.points)
    {
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

    re.hcurlEdge.resize(nv);
    re.momentsEdge.resize(nv);
    re.reggeEdge.resize(nv);
    for (std::size_t e = 0; e < nv; ++e)
    {
        for (double t : re.edgeParameters)
        {
            const ReferencePoint x = referenceEdgePointAndDirection(nv, e, t).first;
            for (const Field& f : re.hcurlBasis)
                re.hcurlEdge[e].push_back({f[0].evaluate(x[0], x[1]), f[1].evaluate(x[0], x[1])});
            for (const Field& f : re.momentBasis)
                re.momentsEdge[e].push_back(evaluate3(f, x));
            for (const Field& f : re.reggeBasis)
                re.reggeEdge[e].push_back(evaluate3(f, x));
        }
    }

    // the raw membrane functionals: the tangential moments of each edge (p + 1), the interior moments, the constants
    re.reggeInteriorMoments = reggeInterior(nv, p);
    const std::size_t rawFunctionals = nv * (p + 1) + re.reggeInteriorMoments.size() + 3;
    const bool isFull = re.membraneCombination.empty();
    if ((isFull && rawFunctionals != re.reggeCount) ||
        (!isFull && (re.membraneCombination.columns() + 3 != rawFunctionals ||
                     re.membraneCombination.rows() + 3 != re.reggeCount)))
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
    re.outputPoints.push_back(valuesAtPoint(re, referenceCentre(cornerCount)));
    for (std::size_t c = 0; c < cornerCount; ++c)
        re.outputPoints.push_back(valuesAtPoint(re, referenceCorner(cornerCount, c)));
    return re;
}

Vector ShellReferenceElement::membraneFunctionals(Vector raw) const
{
    if (membraneCombination.empty())
        return raw;
    const std::size_t other = membraneCombination.columns();
    Vector out = membraneCombination * std::span<const double>(raw.data(), other);
    out.insert(out.end(), raw.begin() + static_cast<std::ptrdiff_t>(other), raw.end());
    return out;
}

Vector ShellReferenceElement::rawMembraneWeights(const Vector& weights) const
{
    if (membraneCombination.empty())
        return weights;
    const std::size_t kept = membraneCombination.rows();
    Vector out(membraneCombination.columns(), 0.0);
    for (std::size_t r = 0; r < kept; ++r)
        for (std::size_t i = 0; i < out.size(); ++i)
            out[i] += membraneCombination(r, i) * weights[r];
    out.insert(out.end(), weights.begin() + static_cast<std::ptrdiff_t>(kept), weights.end());
    return out;
}

const ReferencePointValues* ShellReferenceElement::valuesAt(double xi, double eta) const noexcept
{
    for (const ReferencePointValues& values : outputPoints)
        if (values.point[0] == xi && values.point[1] == eta)
            return &values;
    return nullptr;
}

const ShellDisplacementTables& shellDisplacementTables(std::size_t cornerCount, std::size_t degree,
                                                       std::size_t pointsPerDirection,
                                                       std::span<const std::size_t> edgeDegrees)
{
    const EdgeDegrees degrees = edgeDegreesOf(cornerCount, degree, edgeDegrees);
    static std::mutex mutex;
    static std::map<ReferenceKey, std::unique_ptr<const ShellDisplacementTables>> cache;
    const ReferenceKey key{cornerCount, degree, pointsPerDirection, degrees};
    const std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(key);
    if (found == cache.end())
    {
        const auto edges = std::span(degrees).first(cornerCount);
        auto tables = displacementPart(cornerCount, degree, pointsPerDirection, edges).tables;
        found = cache.emplace(key, std::make_unique<const ShellDisplacementTables>(std::move(tables))).first;
    }
    return *found->second;
}

const ShellReferenceElement& shellReferenceElement(std::size_t cornerCount, std::size_t degree,
                                                   std::size_t pointsPerDirection,
                                                   std::span<const std::size_t> edgeDegrees)
{
    const EdgeDegrees degrees = edgeDegreesOf(cornerCount, degree, edgeDegrees);
    static std::mutex mutex;
    static std::map<ReferenceKey, std::unique_ptr<const ShellReferenceElement>> cache;
    const ReferenceKey key{cornerCount, degree, pointsPerDirection, degrees};
    const std::lock_guard<std::mutex> lock(mutex);
    auto found = cache.find(key);
    if (found == cache.end())
    {
        auto built = std::make_unique<const ShellReferenceElement>(
            ShellReferenceElement::build(cornerCount, degree, pointsPerDirection, edgeDegrees));
        found = cache.emplace(key, std::move(built)).first;
    }
    return *found->second;
}

} // namespace oos::elements::mixedHybrid
