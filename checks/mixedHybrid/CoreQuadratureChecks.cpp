#include "CheckRunner.hpp"
#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"
#include "oos/elements/mixedHybrid/core/SolidSpaces.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace mh = oos::elements::mixedHybrid;

namespace
{

// Σ w_g f_i f_j over a rule, for the functions tabulated at its points (values[g][i]).
std::vector<double> gramOf(const std::vector<std::vector<double>>& values, const std::vector<double>& weights)
{
    const std::size_t n = values.empty() ? 0 : values[0].size();
    std::vector<double> gram(n * n, 0.0);
    for (std::size_t g = 0; g < values.size(); ++g)
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                gram[i * n + j] += weights[g] * values[g][i] * values[g][j];
    return gram;
}

// The largest |G_ij| / √(G_ii G_jj) over i ≠ j (zero for an orthogonal family).
double largestCosine(const std::vector<double>& gram, std::size_t n)
{
    double worst = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (i != j)
                worst = std::fmax(worst, std::fabs(gram[i * n + j]) / std::sqrt(gram[i * n + i] * gram[j * n + j]));
    return worst;
}

// Tabulates a solid space at the points of a rule and returns the largest cosine of its Gram matrix.
double solidSpaceCosine(const std::vector<mh::VolumeRulePoint>& rule,
                        const std::function<std::vector<double>(const std::array<double, 3>&)>& space)
{
    std::vector<std::vector<double>> values;
    std::vector<double> weights;
    for (const mh::VolumeRulePoint& p : rule)
    {
        values.push_back(space(p.point));
        weights.push_back(p.weight);
    }
    return largestCosine(gramOf(values, weights), values[0].size());
}

double weightSum(const std::vector<mh::VolumeRulePoint>& rule)
{
    double sum = 0.0;
    for (const mh::VolumeRulePoint& p : rule)
        sum += p.weight;
    return sum;
}

} // namespace

// Port of the reference test gauss_integrates_polynomials.
CHECK_CASE("core: Gauss-Legendre integrates the polynomials of degree 2n - 1")
{
    for (std::size_t n = 1; n < 25; ++n)
    {
        const mh::LineRule rule = mh::gaussLegendre(n);
        for (int degree = 0; degree < static_cast<int>(2 * n); ++degree)
        {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                sum += rule.weights[i] * std::pow(rule.points[i], degree);
            const double exact = degree % 2 == 0 ? 2.0 / (degree + 1.0) : 0.0;
            runner.expect(std::fabs(sum - exact) < 1e-13,
                          "n = " + std::to_string(n) + ", degree " + std::to_string(degree));
        }
    }
}

// Port of the first part of the reference test tetrahedron_basis_is_orthogonal.
CHECK_CASE("core: Gauss-Jacobi integrates against its weight")
{
    const mh::LineRule rule = mh::gaussJacobi(6, 2.0);
    const mh::LineRule fine = mh::gaussLegendre(12);
    for (int degree = 0; degree < 11; ++degree)
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < rule.points.size(); ++i)
            sum += rule.weights[i] * std::pow(rule.points[i], degree);
        // ∫ (1 − x)² x^degree dx by a fine Gauss-Legendre rule.
        double exact = 0.0;
        for (std::size_t i = 0; i < fine.points.size(); ++i)
        {
            const double x = fine.points[i];
            exact += fine.weights[i] * (1.0 - x) * (1.0 - x) * std::pow(x, degree);
        }
        runner.expect(std::fabs(sum - exact) < 1e-13, "Gauss-Jacobi degree " + std::to_string(degree));
    }
    // The nodes increase, the weights are positive and sum to μ0 = 2^(α+1) / (α + 1).
    for (const double alpha : {0.0, 1.0, 2.0})
    {
        const mh::LineRule r = mh::gaussJacobi(7, alpha);
        double total = 0.0;
        for (std::size_t i = 0; i < r.points.size(); ++i)
        {
            total += r.weights[i];
            runner.expect(r.weights[i] > 0.0, "positive weight");
            if (i > 0)
                runner.expect(r.points[i] > r.points[i - 1], "increasing nodes");
        }
        runner.expectNear(total, std::pow(2.0, alpha + 1.0) / (alpha + 1.0), 1e-14, "sum of the weights");
    }
    // α = 0 is the Gauss-Legendre rule.
    const mh::LineRule legendre = mh::gaussLegendre(9);
    const mh::LineRule jacobi = mh::gaussJacobi(9, 0.0);
    for (std::size_t i = 0; i < 9; ++i)
    {
        runner.expectNear(jacobi.points[i], legendre.points[i], 1e-14, "Gauss-Jacobi(0) node");
        runner.expectNear(jacobi.weights[i], legendre.weights[i], 1e-14, "Gauss-Jacobi(0) weight");
    }
}

// Port of the second part of the reference test tetrahedron_basis_is_orthogonal.
CHECK_CASE("core: the tetrahedron basis is orthogonal")
{
    const std::size_t d = 6;
    const std::size_t n = mh::totalDegreeCount3D(d);
    const std::vector<mh::VolumeRulePoint> rule = mh::tetrahedronRule(d + 1);
    runner.expect(std::fabs(weightSum(rule) - 4.0 / 3.0) < 1e-13, "volume of the tetrahedron");
    std::vector<std::vector<double>> values;
    std::vector<double> weights;
    for (const mh::VolumeRulePoint& p : rule)
    {
        values.push_back(mh::tetrahedronBasis(d, p.point[0], p.point[1], p.point[2]));
        weights.push_back(p.weight);
    }
    runner.expect(values[0].size() == n, "size of the basis");
    runner.expect(largestCosine(gramOf(values, weights), n) < 1e-12, "orthogonality");
}

// Port of the reference test triangle_basis_is_orthogonal.
CHECK_CASE("core: the triangle basis is orthogonal and its derivatives are right")
{
    const std::size_t d = 11;
    const std::size_t n = mh::totalDegreeCount2D(d);
    const std::vector<mh::TriangleRulePoint> rule = mh::triangleRule(14);
    double area = 0.0;
    for (const mh::TriangleRulePoint& p : rule)
        area += p.weight;
    runner.expect(std::fabs(area - 2.0) < 1e-14, "area of the triangle");
    std::vector<double> val(n);
    std::vector<double> du(n);
    std::vector<double> dv(n);
    std::vector<std::vector<double>> values;
    std::vector<double> weights;
    for (const mh::TriangleRulePoint& p : rule)
    {
        mh::triangleBasisWithDerivatives(d, p.u, p.v, val, du, dv);
        values.push_back(val);
        weights.push_back(p.weight);
    }
    runner.expect(largestCosine(gramOf(values, weights), n) < 1e-13, "orthogonality");
    // Derivatives against central differences.
    const double u = -0.4;
    const double v = -0.3;
    const double h = 1e-6;
    std::vector<double> a(n);
    std::vector<double> b(n);
    mh::triangleBasisWithDerivatives(d, u, v, val, du, dv);
    mh::triangleBasis(d, u + h, v, a);
    mh::triangleBasis(d, u - h, v, b);
    for (std::size_t i = 0; i < n; ++i)
        runner.expect(std::fabs(0.5 * (a[i] - b[i]) / h - du[i]) < 1e-7 * (1.0 + std::fabs(du[i])),
                      "du " + std::to_string(i));
    mh::triangleBasis(d, u, v + h, a);
    mh::triangleBasis(d, u, v - h, b);
    for (std::size_t i = 0; i < n; ++i)
        runner.expect(std::fabs(0.5 * (a[i] - b[i]) / h - dv[i]) < 1e-7 * (1.0 + std::fabs(dv[i])),
                      "dv " + std::to_string(i));
}

CHECK_CASE("core: the generic triangle space is the triangle basis")
{
    const std::size_t d = 7;
    const std::size_t n = mh::totalDegreeCount2D(d);
    std::vector<double> basis(n);
    mh::triangleBasis(d, 0.15, -0.55, basis);
    const std::vector<double> generic = mh::triangleSpace(d, 0.15, -0.55);
    runner.expect(generic.size() == n, "size of the triangle space");
    for (std::size_t i = 0; i < n; ++i)
        runner.expectNear(generic[i], basis[i], 1e-13, "triangle function " + std::to_string(i));
}

CHECK_CASE("core: the Legendre basis of the square and its derivatives")
{
    const std::size_t d = 5;
    const std::size_t n = mh::totalDegreeCount2D(d);
    std::vector<double> val(n);
    std::vector<double> du(n);
    std::vector<double> dv(n);
    std::vector<double> plain(n);
    const double u = 0.2;
    const double v = -0.7;
    mh::legendreBasisWithDerivatives(d, u, v, val, du, dv);
    mh::legendreBasis(d, u, v, plain);
    std::size_t k = 0;
    for (std::size_t total = 0; total <= d; ++total)
    {
        for (std::size_t i = 0; i <= total; ++i)
        {
            const double expected = mh::legendreValue(i, u) * mh::legendreValue(total - i, v);
            runner.expectNear(val[k], expected, 1e-15, "value");
            runner.expectNear(plain[k], expected, 1e-15, "plain value");
            ++k;
        }
    }
    // P'_n against central differences, and P_3(x) = (5x³ − 3x) / 2.
    const double x = 0.37;
    const double h = 1e-6;
    const mh::PolynomialTable table = mh::legendreTable(8, x);
    for (std::size_t m = 0; m <= 8; ++m)
    {
        const double slope = (mh::legendreValue(m, x + h) - mh::legendreValue(m, x - h)) / (2.0 * h);
        runner.expectNear(table.derivatives[m], slope, 1e-8, "Legendre derivative");
    }
    runner.expectNear(table.values[3], 0.5 * (5.0 * x * x * x - 3.0 * x), 1e-15, "P_3");
    // Jacobi P_n^(α,0) with α = 0 is Legendre; its derivatives against differences.
    const mh::PolynomialTable jacobi = mh::jacobiTable(8, 3.0, x);
    const mh::PolynomialTable jacobiZero = mh::jacobiTable(8, 0.0, x);
    for (std::size_t m = 0; m <= 8; ++m)
    {
        runner.expectNear(jacobiZero.values[m], table.values[m], 1e-14, "Jacobi(0) is Legendre");
        const double up = mh::jacobiTable(8, 3.0, x + h).values[m];
        const double down = mh::jacobiTable(8, 3.0, x - h).values[m];
        runner.expectNear(jacobi.derivatives[m], (up - down) / (2.0 * h), 1e-7, "Jacobi derivative");
    }
}

CHECK_CASE("core: scaled Jacobi polynomials are homogeneous Jacobi polynomials")
{
    const double p = 0.3;
    const double q = 0.6;
    const std::vector<double> scaled = mh::scaledJacobi(6, 3.0, p, q);
    const mh::PolynomialTable plain = mh::jacobiTable(6, 3.0, p / q);
    for (std::size_t k = 0; k <= 6; ++k)
        runner.expectNear(scaled[k], std::pow(q, static_cast<double>(k)) * plain.values[k], 1e-14, "q^k P_k(p / q)");
}

CHECK_CASE("core: the rules of the solids and of the faces")
{
    for (std::size_t n = 1; n <= 6; ++n)
    {
        runner.expectNear(weightSum(mh::hexahedronRule(n)), 8.0, 1e-13, "volume of the hexahedron");
        runner.expectNear(weightSum(mh::pentahedronRule(n)), 4.0, 1e-13, "volume of the pentahedron");
        runner.expectNear(weightSum(mh::pyramidRule(n)), 8.0 / 3.0, 1e-13, "volume of the pyramid");
        runner.expectNear(weightSum(mh::tetrahedronRule(n)), 4.0 / 3.0, 1e-13, "volume of the tetrahedron");
    }
    // ∫ (1 − z)⁴ over the pyramid: q² dz da db with q = (1 − z)/2 gives 4 ∫ (1 − z)⁶ / 4 dz = 2⁷ / 7.
    double moment = 0.0;
    for (const mh::VolumeRulePoint& p : mh::pyramidRule(3))
        moment += p.weight * std::pow(1.0 - p.point[2], 4);
    runner.expectNear(moment, 128.0 / 7.0, 1e-12, "moment of the pyramid");
    // ∫ x² over the pyramid: ∫ (4/3) q⁴ dz with q⁴ = (1 − z)⁴ / 16: (4/3)(1/16)(2⁵/5) = 8/15.
    double xx = 0.0;
    for (const mh::VolumeRulePoint& p : mh::pyramidRule(3))
        xx += p.weight * p.point[0] * p.point[0];
    runner.expectNear(xx, 8.0 / 15.0, 1e-13, "second moment of the pyramid");
    // Faces: the triangle in (L1, L2) has area 1/2, the quadrilateral in (s, t) area 1; ∫ L1 = 1/6, ∫ s t = 1/4.
    double triangleArea = 0.0;
    double firstMoment = 0.0;
    for (const mh::SurfaceRulePoint& p : mh::faceRule(3, 4))
    {
        triangleArea += p.weight;
        firstMoment += p.weight * p.point[0];
    }
    runner.expectNear(triangleArea, 0.5, 1e-14, "area of a triangular face");
    runner.expectNear(firstMoment, 1.0 / 6.0, 1e-14, "moment of a triangular face");
    double quadArea = 0.0;
    double product = 0.0;
    for (const mh::SurfaceRulePoint& p : mh::faceRule(4, 3))
    {
        quadArea += p.weight;
        product += p.weight * p.point[0] * p.point[1];
    }
    runner.expectNear(quadArea, 1.0, 1e-14, "area of a quadrilateral face");
    runner.expectNear(product, 0.25, 1e-14, "moment of a quadrilateral face");
    // The shell domains: square of area 4, triangle of area 2; the edge rule on [0, 1].
    double square = 0.0;
    for (const mh::SurfaceRulePoint& p : mh::referenceSurfaceRule(4, 5))
        square += p.weight;
    double triangle = 0.0;
    for (const mh::SurfaceRulePoint& p : mh::referenceSurfaceRule(3, 5))
        triangle += p.weight;
    double edge = 0.0;
    double edgeMoment = 0.0;
    for (const mh::LineRulePoint& p : mh::unitIntervalRule(4))
    {
        edge += p.weight;
        edgeMoment += p.weight * p.point * p.point * p.point;
    }
    runner.expectNear(square, 4.0, 1e-14, "area of the square");
    runner.expectNear(triangle, 2.0, 1e-14, "area of the triangle");
    runner.expectNear(edge, 1.0, 1e-14, "length of the edge");
    runner.expectNear(edgeMoment, 0.25, 1e-14, "moment on the edge");
}

CHECK_CASE("core: the spaces of the solids are orthogonal and of their dimension")
{
    for (std::size_t n = 0; n <= 4; ++n)
    {
        const auto hexa = [n](const std::array<double, 3>& r) { return mh::hexahedronSpace(n, r); };
        const auto penta = [n](const std::array<double, 3>& r) { return mh::pentahedronSpace(n, r); };
        const auto pyramid = [n](const std::array<double, 3>& r) { return mh::pyramidSpace(n, r); };
        const auto tetra = [n](const std::array<double, 3>& r) { return mh::tetrahedronSpace(n, r); };
        const std::array<double, 3> r{0.1, -0.2, 0.3};
        runner.expect(hexa(r).size() == mh::hexahedronSpaceDimension(n), "dimension of the hexahedron space");
        runner.expect(penta(r).size() == mh::pentahedronSpaceDimension(n), "dimension of the pentahedron space");
        runner.expect(pyramid(r).size() == mh::pyramidSpaceDimension(n), "dimension of the pyramid space");
        runner.expect(tetra(r).size() == mh::tetrahedronSpaceDimension(n), "dimension of the tetrahedron space");
        runner.expect(mh::hexahedronLevels(n).size() == mh::hexahedronSpaceDimension(n), "hexahedron levels");
        runner.expect(mh::pentahedronLevels(n).size() == mh::pentahedronSpaceDimension(n), "pentahedron levels");
        runner.expect(mh::pyramidLevels(n).size() == mh::pyramidSpaceDimension(n), "pyramid levels");
        runner.expect(mh::tetrahedronLevels(n).size() == mh::tetrahedronSpaceDimension(n), "tetrahedron levels");
        if (n == 0)
            continue;
        runner.expect(solidSpaceCosine(mh::hexahedronRule(n + 1), hexa) < 1e-12, "hexahedron space orthogonal");
        runner.expect(solidSpaceCosine(mh::pentahedronRule(n + 1), penta) < 1e-12, "pentahedron space orthogonal");
        runner.expect(solidSpaceCosine(mh::pyramidRule(n + 1), pyramid) < 1e-12, "pyramid space orthogonal");
    }
    // The space of degree m is spanned by the functions of level m at most: their count is the dimension of degree m.
    for (std::size_t m = 0; m <= 4; ++m)
    {
        std::size_t count = 0;
        for (const std::size_t level : mh::pyramidLevels(4))
            count += level <= m ? 1 : 0;
        runner.expect(count == mh::pyramidSpaceDimension(m), "pyramid levels nest the spaces");
    }
    // At the apex the functions with min(i, j) > 0 vanish and the others are finite.
    for (const double value : mh::pyramidSpace(3, std::array<double, 3>{0.0, 0.0, 1.0}))
        runner.expect(std::isfinite(value), "finite at the apex");
}

CHECK_CASE("core: the tetrahedron basis differentiated through Dual")
{
    using D = mh::Dual<double>;
    const double r = 0.1;
    const double s = -0.2;
    const double t = -0.5;
    const double h = 1e-6;
    const std::vector<D> dual = mh::tetrahedronBasis(4, D::variable(r, 0), D::variable(s, 1), D(t));
    const std::vector<double> rPlus = mh::tetrahedronBasis(4, r + h, s, t);
    const std::vector<double> rMinus = mh::tetrahedronBasis(4, r - h, s, t);
    const std::vector<double> sPlus = mh::tetrahedronBasis(4, r, s + h, t);
    const std::vector<double> sMinus = mh::tetrahedronBasis(4, r, s - h, t);
    const std::vector<double> plain = mh::tetrahedronBasis(4, r, s, t);
    for (std::size_t i = 0; i < dual.size(); ++i)
    {
        runner.expectNear(dual[i].value, plain[i], 1e-15, "value");
        runner.expectNear(dual[i].derivatives[0], (rPlus[i] - rMinus[i]) / (2.0 * h), 1e-7, "d/dr");
        runner.expectNear(dual[i].derivatives[1], (sPlus[i] - sMinus[i]) / (2.0 * h), 1e-7, "d/ds");
    }
}
