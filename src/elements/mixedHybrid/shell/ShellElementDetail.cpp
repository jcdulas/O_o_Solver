#include "ShellElementDetail.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"
#include "oos/elements/mixedHybrid/geometry/ReferenceShapes.hpp"
#include "oos/elements/mixedHybrid/shell/ShellSection.hpp"

namespace oos::elements::mixedHybrid::shelldetail
{

std::vector<std::vector<EdgePointGeometry>> edgeGeometries(const ShellReferenceElement& re,
                                                           const ShellElementDefinition& def)
{
    // the tables of the element are on the rule of the reference element (an element has its own rule)
    OOS_ASSERT(def.edgeJets.size() == re.cornerCount &&
               def.edgeJets.front().size() == re.edgeParameters.size() * re.h1Count);
    std::vector<std::vector<EdgePointGeometry>> out(re.cornerCount);
    for (std::size_t k = 0; k < re.cornerCount; ++k)
    {
        out[k].reserve(re.edgeParameters.size());
        for (std::size_t g = 0; g < re.edgeParameters.size(); ++g)
        {
            EdgePointGeometry point;
            point.tv = referenceEdgePointAndDirection(re.cornerCount, k, re.edgeParameters[g]).second;
            point.g = edgeGeometry(def, re.h1Count, k, g);
            point.xt = add3(scl3(point.g.a[0], point.tv[0]), scl3(point.g.a[1], point.tv[1]));
            out[k].push_back(point);
        }
    }
    return out;
}

std::vector<PointGeometry> volumeGeometries(const ShellReferenceElement& re, const ShellElementDefinition& def)
{
    OOS_ASSERT(def.volumeJets.size() == re.points.size() * re.h1Count);
    std::vector<PointGeometry> out;
    out.reserve(re.points.size());
    for (std::size_t q = 0; q < re.points.size(); ++q)
        out.push_back(volumeGeometry(def, re.h1Count, q));
    return out;
}

std::vector<double> normalMoments(const ShellReferenceElement& re, std::size_t k, std::size_t g,
                                  const PointGeometry& geometry, const SpaceVector& conormal,
                                  bool mapReferenceConstants)
{
    const std::size_t ns = re.momentCount;
    const double ma[2] = {dot3(conormal, geometry.a[0]), dot3(conormal, geometry.a[1])};
    const double me1 = dot3(conormal, geometry.e[0]);
    const double me2 = dot3(conormal, geometry.e[1]);
    const double inverseJ2 = 1.0 / (geometry.j * geometry.j);
    const Matrix33 t = strainTransformation(geometry.phi);
    std::vector<double> out(ns);
    for (std::size_t j = 0; j < ns; ++j)
    {
        if (j < ns - 3 || mapReferenceConstants)
        {
            const std::array<double, 3>& s = re.momentsEdge[k][g * ns + j];
            out[j] = (s[0] * ma[0] * ma[0] + s[1] * ma[1] * ma[1] + 2.0 * s[2] * ma[0] * ma[1]) * inverseJ2;
        }
        else
        {
            const std::array<double, 3>& m = t[j - (ns - 3)];
            out[j] = m[0] * me1 * me1 + m[1] * me2 * me2 + 2.0 * m[2] * me1 * me2;
        }
    }
    return out;
}

void addCongruence(Matrix& target, const Matrix& a, const Matrix33& m, double s)
{
    const std::size_t n = a.columns();
    // ma = m · a
    Matrix ma(3, n);
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t j = 0; j < n; ++j)
            ma(r, j) = m[r][0] * a(0, j) + m[r][1] * a(1, j) + m[r][2] * a(2, j);
    for (std::size_t i = 0; i < n; ++i)
    {
        const double a0 = a(0, i) * s;
        const double a1 = a(1, i) * s;
        const double a2 = a(2, i) * s;
        if (a0 == 0.0 && a1 == 0.0 && a2 == 0.0)
            continue;
        std::span<double> row = target.rowSpan(i);
        for (std::size_t j = 0; j < n; ++j)
            row[j] += a0 * ma(0, j) + a1 * ma(1, j) + a2 * ma(2, j);
    }
}

void addCongruence2(Matrix& target, const Matrix& a, const Matrix22& m, double s)
{
    addCongruence2(target, a, m, s, 0, a.columns());
}

void addCongruence2(Matrix& target, const Matrix& a, const Matrix22& m, double s, std::size_t firstColumn,
                    std::size_t columnCount)
{
    const std::size_t last = firstColumn + columnCount;
    Matrix ma(2, a.columns());
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t j = firstColumn; j < last; ++j)
            ma(r, j) = m[r][0] * a(0, j) + m[r][1] * a(1, j);
    for (std::size_t i = firstColumn; i < last; ++i)
    {
        const double a0 = a(0, i) * s;
        const double a1 = a(1, i) * s;
        if (a0 == 0.0 && a1 == 0.0)
            continue;
        std::span<double> row = target.rowSpan(i);
        for (std::size_t j = firstColumn; j < last; ++j)
            row[j] += a0 * ma(0, j) + a1 * ma(1, j);
    }
}

void addTransposeProduct(Matrix& target, const Matrix& a, const Matrix& b, double s)
{
    addTransposeProduct(target, a, b, s, b.columns());
}

void addTransposeProduct(Matrix& target, const Matrix& a, const Matrix& b, double s, std::size_t columnCount)
{
    for (std::size_t r = 0; r < a.rows(); ++r)
    {
        const std::span<const double> source = b.rowSpan(r);
        for (std::size_t i = 0; i < a.columns(); ++i)
        {
            const double ari = a(r, i) * s;
            if (ari == 0.0)
                continue;
            std::span<double> row = target.rowSpan(i);
            for (std::size_t j = 0; j < columnCount; ++j)
                row[j] += ari * source[j];
        }
    }
}

void addTransposeProduct33(Matrix& target, const Matrix& a, const Matrix33& m, const Matrix& b, double s)
{
    Matrix mb(3, b.columns());
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t j = 0; j < b.columns(); ++j)
            mb(r, j) = m[r][0] * b(0, j) + m[r][1] * b(1, j) + m[r][2] * b(2, j);
    addTransposeProduct(target, a, mb, s);
}

void symmetrise(Matrix& a)
{
    for (std::size_t i = 0; i < a.rows(); ++i)
    {
        for (std::size_t j = 0; j < i; ++j)
        {
            const double mean = 0.5 * (a(i, j) + a(j, i));
            a(i, j) = mean;
            a(j, i) = mean;
        }
    }
}

std::array<double, 3> times3(const Matrix& a, std::span<const double> v)
{
    std::array<double, 3> out{};
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t j = 0; j < a.columns(); ++j)
            out[r] += a(r, j) * v[j];
    return out;
}

double edgeLegendre(std::size_t l, double tau)
{
    return legendreValue(l, 2.0 * tau - 1.0);
}

} // namespace oos::elements::mixedHybrid::shelldetail
