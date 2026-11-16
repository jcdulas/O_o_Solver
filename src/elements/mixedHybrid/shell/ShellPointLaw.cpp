#include "oos/elements/mixedHybrid/shell/ShellPointLaw.hpp"

#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/DenseDecompositions.hpp"
#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/core/Matrix.hpp"

namespace oos::elements::mixedHybrid
{

namespace
{

// μ0 Pᵀ M P.
Matrix33 offsetCongruence(const Matrix33& m, const Matrix33& p, double mu0)
{
    Matrix33 out{};
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            for (std::size_t k = 0; k < 3; ++k)
                for (std::size_t l = 0; l < 3; ++l)
                    out[i][j] += p[k][i] * m[k][l] * p[l][j];
            out[i][j] *= mu0;
        }
    }
    return out;
}

// μ0 Pᵀ v.
Voigt3 offsetResultant(const Voigt3& v, const Matrix33& p, double mu0)
{
    Voigt3 out{};
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t k = 0; k < 3; ++k)
            out[i] += p[k][i] * v[k];
        out[i] *= mu0;
    }
    return out;
}

bool isFlat(const PointGeometry& g)
{
    for (const SpaceVector& d : g.dnu)
        for (double v : d)
            if (v != 0.0)
                return false;
    return true;
}

// The covariant basis of the surface at z0 along the normal: a_α + z0 ∂_α ν.
SurfaceFrame offsetBasis(const PointGeometry& g, double offset)
{
    return {add3(g.a[0], scl3(g.dnu[0], offset)), add3(g.a[1], scl3(g.dnu[1], offset))};
}

} // namespace

std::optional<Matrix33> inverse33(const Matrix33& m)
{
    // cofactors of the first row
    const double c00 = m[1][1] * m[2][2] - m[2][1] * m[1][2];
    const double c01 = m[1][0] * m[2][2] - m[2][0] * m[1][2];
    const double c02 = m[1][0] * m[2][1] - m[2][0] * m[1][1];
    const double det = m[0][0] * c00 - m[0][1] * c01 + m[0][2] * c02;
    if (det == 0.0)
        return std::nullopt;
    Matrix33 r{};
    r[0][0] = c00 / det;
    r[0][1] = -(m[0][1] * m[2][2] - m[2][1] * m[0][2]) / det;
    r[0][2] = (m[0][1] * m[1][2] - m[1][1] * m[0][2]) / det;
    r[1][0] = -c01 / det;
    r[1][1] = (m[0][0] * m[2][2] - m[2][0] * m[0][2]) / det;
    r[1][2] = -(m[0][0] * m[1][2] - m[1][0] * m[0][2]) / det;
    r[2][0] = c02 / det;
    r[2][1] = -(m[0][0] * m[2][1] - m[2][0] * m[0][1]) / det;
    r[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) / det;
    return r;
}

Voigt3 times33(const Matrix33& m, const Voigt3& v)
{
    return {m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2], m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]};
}

Voigt3 transposeTimes33(const Matrix33& m, const Voigt3& v)
{
    return {m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2], m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2]};
}

Matrix33 product33(const Matrix33& a, const Matrix33& b)
{
    Matrix33 out{};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 3; ++k)
                out[i][j] += a[i][k] * b[k][j];
    return out;
}

SectionPoint sectionAt(const ShellSection& section, double offset, double xi, double eta, const SurfaceFrame& frame)
{
    SectionPoint sp = section.at(xi, eta, frame);
    if (offset == 0.0)
        return sp;
    const double z0 = offset;
    const Matrix33 a = sp.a;
    const Matrix33 b = sp.b;
    const Matrix33 d = sp.d;
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t jj = 0; jj < 3; ++jj)
        {
            sp.b[i][jj] = b[i][jj] + z0 * a[i][jj];
            sp.d[i][jj] = d[i][jj] + 2.0 * z0 * b[i][jj] + z0 * z0 * a[i][jj];
        }
        const double ta = sp.thermalA[i];
        const double tb = sp.thermalB[i];
        const double td = sp.thermalD[i];
        sp.thermalB[i] = tb + z0 * ta;
        sp.thermalD[i] = td + 2.0 * z0 * tb + z0 * z0 * ta;
    }
    const std::array<double, 3> m = sp.mass;
    sp.mass = {m[0], m[1] + z0 * m[0], m[2] + 2.0 * z0 * m[1] + z0 * z0 * m[0]};
    return sp;
}

SectionPoint sectionAtGeometry(const ShellSection& section, double offset, double xi, double eta,
                               const PointGeometry& g)
{
    SectionPoint sp = sectionAt(section, offset, xi, eta, g.e);
    if (offset == 0.0 || isFlat(g))
        return sp;
    const double mu0 = offsetArea(g, offset);
    const Matrix33 p = offsetMap(g, offset);
    sp.a = offsetCongruence(sp.a, p, mu0);
    sp.b = offsetCongruence(sp.b, p, mu0);
    sp.d = offsetCongruence(sp.d, p, mu0);
    sp.thermalA = offsetResultant(sp.thermalA, p, mu0);
    sp.thermalB = offsetResultant(sp.thermalB, p, mu0);
    sp.thermalD = offsetResultant(sp.thermalD, p, mu0);
    for (double& m : sp.mass)
        m *= mu0;
    return sp;
}

double offsetArea(const PointGeometry& g, double offset)
{
    const SurfaceFrame a = offsetBasis(g, offset);
    return norm3(cross3(a[0], a[1])) / g.j;
}

Matrix33 offsetMap(const PointGeometry& g, double offset)
{
    const SurfaceFrame a = offsetBasis(g, offset);
    const Matrix22 metric{{{dot3(a[0], a[0]), dot3(a[0], a[1])}, {dot3(a[1], a[0]), dot3(a[1], a[1])}}};
    const double det = metric[0][0] * metric[1][1] - metric[0][1] * metric[1][0];
    const Matrix22 inverse{{{metric[1][1] / det, -metric[0][1] / det}, {-metric[1][0] / det, metric[0][0] / det}}};
    const SurfaceFrame ac{add3(scl3(a[0], inverse[0][0]), scl3(a[1], inverse[0][1])),
                          add3(scl3(a[0], inverse[1][0]), scl3(a[1], inverse[1][1]))};
    const Matrix22 cOffset{{{dot3(g.e[0], ac[0]), dot3(g.e[1], ac[0])}, {dot3(g.e[0], ac[1]), dot3(g.e[1], ac[1])}}};
    const std::optional<Matrix33> gridInverse = inverse33(covariantToVoigt(g.cov()));
    if (!gridInverse)
        throw base::InternalError("degenerate local frame of a shell point");
    return product33(covariantToVoigt(cOffset), *gridInverse);
}

SectionLaws sectionLaws(const SectionPoint& point)
{
    const std::optional<Matrix33> dinv = inverse33(point.d);
    if (!dinv)
        throw base::InternalError("bending stiffness of the section is singular");
    SectionLaws laws;
    laws.dinv = *dinv;
    Matrix33 bt{};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            bt[i][j] = point.b[j][i];
    laws.dinvBt = product33(laws.dinv, bt);
    const Matrix33 bDinvBt = product33(point.b, laws.dinvBt);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            laws.abar[i][j] = point.a[i][j] - bDinvBt[i][j];
    laws.b = point.b;
    laws.h = point.h;
    return laws;
}

std::pair<Voigt3, Voigt3> freeStrains(const SectionPoint& point, const Matrix22& curvature,
                                      const std::array<double, 2>& temperatures)
{
    Matrix m(6, 6);
    Vector r(6);
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            m(i, j) = point.a[i][j];
            m(i, 3 + j) = point.b[i][j];
            m(3 + i, j) = point.b[j][i];
            m(3 + i, 3 + j) = point.d[i][j];
        }
        r[i] = point.thermalA[i] * temperatures[0] + point.thermalB[i] * temperatures[1];
        r[3 + i] = point.thermalB[i] * temperatures[0] + point.thermalD[i] * temperatures[1];
    }
    const std::optional<Matrix> inverseM = inverse(m);
    if (!inverseM)
        throw base::InternalError("singular section");
    const Vector f = *inverseM * r;
    // the free strain as a tensor, times the curvature: the part of the free expansion on a curved surface
    const Matrix22 strain{{{f[0], 0.5 * f[2]}, {0.5 * f[2], f[1]}}};
    Matrix22 ek{};
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            ek[i][j] = strain[i][0] * curvature[0][j] + strain[i][1] * curvature[1][j];
    return {{f[0], f[1], f[2]}, {f[3] + ek[0][0], f[4] + ek[1][1], f[5] + ek[0][1] + ek[1][0]}};
}

Matrix33 covariantToVoigt(const Matrix22& c)
{
    Matrix33 t{};
    const std::size_t pairs[2][2] = {{0, 0}, {1, 1}};
    for (std::size_t row = 0; row < 2; ++row)
    {
        const std::size_t i = pairs[row][0];
        const std::size_t jj = pairs[row][1];
        t[row][0] = c[0][i] * c[0][jj];
        t[row][1] = c[1][i] * c[1][jj];
        t[row][2] = c[0][i] * c[1][jj] + c[1][i] * c[0][jj];
    }
    t[2][0] = 2.0 * c[0][0] * c[0][1];
    t[2][1] = 2.0 * c[1][0] * c[1][1];
    t[2][2] = 2.0 * (c[0][0] * c[1][1] + c[1][0] * c[0][1]);
    return t;
}

Voigt3 voigtToCovariant(const Matrix22& d, const Voigt3& v)
{
    const Matrix22 e{{{v[0], 0.5 * v[2]}, {0.5 * v[2], v[1]}}};
    const auto component = [&](std::size_t a, std::size_t b)
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < 2; ++i)
        {
            double inner = 0.0;
            for (std::size_t j = 0; j < 2; ++j)
                inner += d[a][i] * d[b][j] * e[i][j];
            sum += inner;
        }
        return sum;
    };
    return {component(0, 0), component(1, 1), component(0, 1)};
}

Matrix33 contravariantToLocal(const Matrix22& d, double j)
{
    const double s = 1.0 / (j * j);
    Matrix33 t{};
    const std::size_t pairs[3][2] = {{0, 0}, {1, 1}, {0, 1}};
    for (std::size_t row = 0; row < 3; ++row)
    {
        const std::size_t i = pairs[row][0];
        const std::size_t jj = pairs[row][1];
        t[row][0] = d[0][i] * d[0][jj] * s;
        t[row][1] = d[1][i] * d[1][jj] * s;
        t[row][2] = (d[0][i] * d[1][jj] + d[1][i] * d[0][jj]) * s;
    }
    return t;
}

} // namespace oos::elements::mixedHybrid
