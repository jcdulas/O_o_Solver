#include "oos/elements/mixedHybrid/model/SolidPartProjections.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"
#include "oos/elements/mixedHybrid/core/Quadrature.hpp"
#include "oos/elements/mixedHybrid/solid/VolumeShape.hpp"

#include <optional>

namespace oos::elements::mixedHybrid
{

namespace
{

// The points of the rule of the projection on an edge: the functions of an edge are of degree p ≤ 4, the curves
// (conics) rational; 12 Gauss points make the Gram matrix exact and the projection of a conic exact to round-off.
constexpr std::size_t EDGE_PROJECTION_POINTS = 12;
// The points per direction of the rule of the projection on a face (faceRule).
constexpr std::size_t FACE_PROJECTION_POINTS = 10;

constexpr std::size_t COMPONENTS = 3;

// G⁻¹ for the Gram matrix of independent functions (an internal error otherwise: the functions of an edge or a face
// are independent by construction).
Matrix gramInverse(const Matrix& gram)
{
    if (gram.rows() == 0)
        return gram;
    std::optional<Matrix> out = inverse(gram);
    if (!out)
        throw base::InternalError("the Gram matrix of the functions of an edge or a face is singular");
    return std::move(*out);
}

} // namespace

std::vector<double> edgeFunctions(std::size_t traceDegree, double s)
{
    const std::size_t count = edgeFunctionCount(traceDegree);
    std::vector<double> out;
    if (count == 0)
        return out;
    // P_m(2s − 1): the homogeneous Jacobi polynomials of weight 0 at q = 1 are the Legendre polynomials
    const std::vector<double> legendre = scaledJacobi(traceDegree - 2, 0.0, 2.0 * s - 1.0, 1.0);
    for (std::size_t m = 0; m < count; ++m)
        out.push_back(s * (1.0 - s) * legendre[m]);
    return out;
}

EdgeProjection EdgeProjection::of(std::size_t traceDegree)
{
    const std::size_t count = edgeFunctionCount(traceDegree);
    const LineRule rule = gaussLegendre(EDGE_PROJECTION_POINTS);
    const std::size_t points = rule.points.size();
    EdgeProjection out;
    std::vector<double> w(points);
    std::vector<std::vector<double>> b(points);
    for (std::size_t q = 0; q < points; ++q)
    {
        // the rule carried from [−1, 1] to [0, 1]
        out.parameters.push_back(0.5 * (rule.points[q] + 1.0));
        w[q] = 0.5 * rule.weights[q];
        b[q] = edgeFunctions(traceDegree, out.parameters[q]);
    }
    Matrix gram(count, count);
    for (std::size_t m = 0; m < count; ++m)
        for (std::size_t n = 0; n < count; ++n)
        {
            double sum = 0.0;
            for (std::size_t q = 0; q < points; ++q)
                sum += w[q] * b[q][m] * b[q][n];
            gram(m, n) = sum;
        }
    const Matrix gi = gramInverse(gram);
    out.weights = Matrix(count, points);
    for (std::size_t m = 0; m < count; ++m)
        for (std::size_t q = 0; q < points; ++q)
        {
            double sum = 0.0;
            for (std::size_t n = 0; n < count; ++n)
                sum += gi(m, n) * w[q] * b[q][n];
            out.weights(m, q) = sum;
        }
    for (std::size_t m = 0; m < count; ++m)
    {
        double lower = 0.0;
        double upper = 0.0;
        for (std::size_t q = 0; q < points; ++q)
        {
            lower += out.weights(m, q) * (1.0 - out.parameters[q]);
            upper += out.weights(m, q) * out.parameters[q];
        }
        out.lowerWeights.push_back(-lower);
        out.upperWeights.push_back(-upper);
    }
    return out;
}

std::vector<SolidPoint> EdgeProjection::project(std::span<const SolidPoint> values, const SolidPoint& lower,
                                                const SolidPoint& upper) const
{
    OOS_ASSERT(values.size() == parameters.size());
    std::vector<SolidPoint> out(weights.rows());
    for (std::size_t m = 0; m < out.size(); ++m)
        for (std::size_t c = 0; c < COMPONENTS; ++c)
        {
            double sum = 0.0;
            for (std::size_t q = 0; q < parameters.size(); ++q)
                sum += weights(m, q) * values[q][c];
            out[m][c] = sum + lowerWeights[m] * lower[c] + upperWeights[m] * upper[c];
        }
    return out;
}

FaceProjection FaceProjection::of(std::span<const base::EntityId> grids, std::size_t traceDegree,
                                  std::size_t faceDegree)
{
    const std::size_t nf = grids.size();
    FaceProjection out;
    out.perEdge = edgeFunctionCount(traceDegree);
    out.interiorCount = faceFunctionCount(faceDegree, nf);
    const std::vector<SurfaceRulePoint> rule = faceRule(nf, FACE_PROJECTION_POINTS);
    for (const SurfaceRulePoint& point : rule)
    {
        out.points.push_back(point.point);
        out.functionValues.push_back(faceFunctions(grids, traceDegree, faceDegree, point.point));
    }
    // the interior functions come after the grid points and the edges
    const std::size_t first = nf + nf * out.perEdge;
    const std::size_t fc = out.interiorCount;
    Matrix gram(fc, fc);
    for (std::size_t m = 0; m < fc; ++m)
        for (std::size_t n = 0; n < fc; ++n)
        {
            double sum = 0.0;
            for (std::size_t q = 0; q < rule.size(); ++q)
                sum += rule[q].weight * out.functionValues[q][first + m] * out.functionValues[q][first + n];
            gram(m, n) = sum;
        }
    const Matrix gi = gramInverse(gram);
    out.weights = Matrix(fc, rule.size());
    for (std::size_t m = 0; m < fc; ++m)
        for (std::size_t q = 0; q < rule.size(); ++q)
        {
            double sum = 0.0;
            for (std::size_t n = 0; n < fc; ++n)
                sum += gi(m, n) * rule[q].weight * out.functionValues[q][first + n];
            out.weights(m, q) = sum;
        }
    return out;
}

std::vector<SolidPoint> FaceProjection::project(std::span<const SolidPoint> values,
                                                std::span<const SolidPoint> gridPositions,
                                                std::span<const std::vector<SolidPoint>> edgeCoefficients) const
{
    const std::size_t nf = gridPositions.size();
    OOS_ASSERT(values.size() == points.size() && edgeCoefficients.size() == nf);
    std::vector<SolidPoint> out(interiorCount, SolidPoint{});
    for (std::size_t q = 0; q < points.size(); ++q)
    {
        const std::vector<double>& f = functionValues[q];
        for (std::size_t c = 0; c < COMPONENTS; ++c)
        {
            // what the surface adds to the grid points and to the edges at this point
            double v = values[q][c];
            for (std::size_t i = 0; i < nf; ++i)
                v -= f[i] * gridPositions[i][c];
            for (std::size_t k = 0; k < nf; ++k)
                for (std::size_t m = 0; m < perEdge; ++m)
                    v -= f[nf + k * perEdge + m] * edgeCoefficients[k][m][c];
            for (std::size_t m = 0; m < interiorCount; ++m)
                out[m][c] += weights(m, q) * v;
        }
    }
    return out;
}

} // namespace oos::elements::mixedHybrid
