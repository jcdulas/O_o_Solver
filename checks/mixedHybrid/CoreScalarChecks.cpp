#include "CheckRunner.hpp"
#include "oos/base/errors/InternalError.hpp"
#include "oos/elements/mixedHybrid/core/DualScalar.hpp"
#include "oos/elements/mixedHybrid/core/HessianScalar.hpp"
#include "oos/elements/mixedHybrid/core/Polynomial2.hpp"
#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>

namespace mh = oos::elements::mixedHybrid;

namespace
{

using D1 = mh::Dual<double>;
using D2 = mh::Dual<mh::Dual<double>>;

constexpr double STEP = 1e-5;

// A function of two variables that uses every operation of the scalar: f = √(x² + y) sin(x y) / cos(y) − 3 x + x / y.
template <mh::Scalar T>
T testFunction(const T& x, const T& y)
{
    const T root = mh::sqrt(x * x + y);
    return root * mh::sin(x * y) / mh::cos(y) - mh::scale(x, 3.0) + x / y;
}

// A function of three variables for HessianScalar: atan2(x1 x0 + 1.3, √x0 / (x1 − 2)) + x2 (x0 + x2) · |(x0, x1, x2)|.
template <std::size_t N>
mh::HessianScalar<N> hessianFunction(const std::array<mh::HessianScalar<N>, 3>& x)
{
    using H = mh::HessianScalar<N>;
    const H y = x[1] * x[0] + H::constant(1.3);
    const H w = mh::sqrt(x[0]) * mh::reciprocal(x[1] - H::constant(2.0));
    const H length = mh::sqrt(mh::hessianDot(x, x));
    return mh::atan2(y, w) + x[2] * (x[0] + x[2]) * length;
}

double plainHessianFunction(double x0, double x1, double x2)
{
    std::array<mh::HessianScalar<1>, 3> x{mh::HessianScalar<1>::constant(x0), mh::HessianScalar<1>::constant(x1),
                                          mh::HessianScalar<1>::constant(x2)};
    return hessianFunction<1>(x).value;
}

} // namespace

CHECK_CASE("core: Dual derivatives against finite differences")
{
    const double x = 0.7;
    const double y = 0.4;
    const D1 f = testFunction(D1::variable(x, 0), D1::variable(y, 1));
    runner.expectNear(f.value, testFunction(x, y), 1e-15, "value");
    const double dx = (testFunction(x + STEP, y) - testFunction(x - STEP, y)) / (2.0 * STEP);
    const double dy = (testFunction(x, y + STEP) - testFunction(x, y - STEP)) / (2.0 * STEP);
    runner.expectNear(f.derivatives[0], dx, 1e-9, "d/dx");
    runner.expectNear(f.derivatives[1], dy, 1e-9, "d/dy");
    runner.expectNear(mh::valueOf(f), f.value, 0.0, "valueOf");
    const D1 z = mh::zeroValue(f);
    runner.expect(z.value == 0.0 && z.derivatives == f.derivatives, "zeroValue keeps the derivatives");
    const D1 c = mh::constantOf<D1>(2.5);
    runner.expect(c.value == 2.5 && c.derivatives[0] == 0.0 && c.derivatives[1] == 0.0, "constant");
}

CHECK_CASE("core: nested Dual gives the second derivatives")
{
    const double x = 0.7;
    const double y = 0.4;
    // Outer direction 0 is x, outer direction 1 is y; the inner directions are the same variables.
    const D2 xx(D1::variable(x, 0), D1(1.0), D1(0.0));
    const D2 yy(D1::variable(y, 1), D1(0.0), D1(1.0));
    const D2 f = testFunction(xx, yy);
    runner.expectNear(mh::valueOf(f), testFunction(x, y), 1e-15, "value");
    // Second derivatives against differences of the exact first derivatives.
    const auto gradient = [](double a, double b)
    {
        const D1 g = testFunction(D1::variable(a, 0), D1::variable(b, 1));
        return std::array<double, 2>{g.derivatives[0], g.derivatives[1]};
    };
    const std::array<double, 2> gxPlus = gradient(x + STEP, y);
    const std::array<double, 2> gxMinus = gradient(x - STEP, y);
    const std::array<double, 2> gyPlus = gradient(x, y + STEP);
    const std::array<double, 2> gyMinus = gradient(x, y - STEP);
    const double fxx = (gxPlus[0] - gxMinus[0]) / (2.0 * STEP);
    const double fxy = (gyPlus[0] - gyMinus[0]) / (2.0 * STEP);
    const double fyy = (gyPlus[1] - gyMinus[1]) / (2.0 * STEP);
    runner.expectNear(f.derivatives[0].derivatives[0], fxx, 1e-8, "d2/dx2");
    runner.expectNear(f.derivatives[0].derivatives[1], fxy, 1e-8, "d2/dxdy");
    runner.expectNear(f.derivatives[1].derivatives[0], fxy, 1e-8, "d2/dydx");
    runner.expectNear(f.derivatives[1].derivatives[1], fyy, 1e-8, "d2/dy2");
    // The first derivatives sit in the inner parts of the value and in the values of the outer parts.
    runner.expectNear(f.value.derivatives[0], f.derivatives[0].value, 1e-14, "symmetric first derivatives");
}

CHECK_CASE("core: 3-vector helpers on Dual")
{
    using V = mh::Vec3Of<D1>;
    const V a{D1::variable(0.3, 0), D1(-1.2), D1::variable(0.8, 1)};
    const V b = mh::lift3<D1>({0.5, 0.25, -2.0});
    const V u = mh::unit3(a);
    const D1 length = mh::norm3(u);
    runner.expectNear(length.value, 1.0, 1e-15, "unit length");
    runner.expectNear(length.derivatives[0], 0.0, 1e-15, "unit length is constant (0)");
    runner.expectNear(length.derivatives[1], 0.0, 1e-15, "unit length is constant (1)");
    const V n = mh::cross3(a, b);
    runner.expectNear(mh::dot3(n, a).value, 0.0, 1e-15, "cross product orthogonal to a");
    runner.expectNear(mh::dot3(n, b).derivatives[0], 0.0, 1e-15, "cross product orthogonal to b");
    const V sum = mh::sub3(mh::add3(a, b), b);
    for (std::size_t i = 0; i < 3; ++i)
        runner.expectNear(sum[i].value, a[i].value, 1e-15, "add then subtract");
    const std::array<std::array<double, 3>, 3> m{{{1.0, 2.0, 3.0}, {0.0, 1.0, 0.0}, {-1.0, 0.0, 2.0}}};
    const V ma = mh::mat3(m, a);
    runner.expectNear(ma[0].value, 0.3 - 2.4 + 2.4, 1e-15, "matrix times vector");
    runner.expectNear(ma[2].derivatives[1], 2.0, 1e-15, "matrix times vector, derivative");
    const V s = mh::scl3(a, D1(2.0));
    runner.expectNear(s[2].derivatives[1], 2.0, 1e-15, "scaled vector");
}

CHECK_CASE("core: HessianScalar against finite differences")
{
    using H = mh::HessianScalar<3>;
    const std::array<double, 3> x0{0.7, -0.4, 0.25};
    const std::array<H, 3> x{H::variable(x0[0], 0), H::variable(x0[1], 1), H::variable(x0[2], 2)};
    const H f = hessianFunction<3>(x);
    runner.expectNear(f.value, plainHessianFunction(x0[0], x0[1], x0[2]), 1e-15, "value");
    const double h = 1e-4;
    const auto at = [&](std::size_t i, double di, std::size_t j, double dj)
    {
        std::array<double, 3> p = x0;
        p[i] += di;
        p[j] += dj;
        return plainHessianFunction(p[0], p[1], p[2]);
    };
    for (std::size_t i = 0; i < 3; ++i)
    {
        const double slope = (at(i, h, i, 0.0) - at(i, -h, i, 0.0)) / (2.0 * h);
        runner.expectNear(f.gradient[i], slope, 1e-8, "gradient " + std::to_string(i));
        for (std::size_t j = 0; j < 3; ++j)
        {
            const double second =
                (at(i, h, j, h) - at(i, h, j, -h) - at(i, -h, j, h) + at(i, -h, j, -h)) / (4.0 * h * h);
            runner.expectNear(f.hessian[i][j], second, 1e-6, "Hessian " + std::to_string(i) + std::to_string(j));
            runner.expectNear(f.hessian[i][j], f.hessian[j][i], 1e-14, "Hessian symmetric");
        }
    }
    // The unit vector has unit length to second order, and its Hessian is that of a constant.
    const mh::HessianVec3<3> u = mh::hessianUnit(x);
    const H length2 = mh::hessianDot(u, u);
    runner.expectNear(length2.value, 1.0, 1e-15, "unit length");
    for (std::size_t i = 0; i < 3; ++i)
    {
        runner.expectNear(length2.gradient[i], 0.0, 1e-14, "unit length: gradient");
        for (std::size_t j = 0; j < 3; ++j)
            runner.expectNear(length2.hessian[i][j], 0.0, 1e-13, "unit length: Hessian");
    }
    // The cross product of x with itself vanishes with its derivatives.
    const mh::HessianVec3<3> cross = mh::hessianCross(x, x);
    for (std::size_t c = 0; c < 3; ++c)
        runner.expect(cross[c].value == 0.0 && cross[c].hessian[0][1] == 0.0, "x cross x = 0");
    // scale and subtraction.
    const H twice = mh::scale(f, 2.0) - f;
    runner.expectNear(twice.hessian[1][2], f.hessian[1][2], 1e-15, "2 f - f = f");
}

CHECK_CASE("core: Polynomial2 arithmetic, derivatives and jet")
{
    using P = mh::Polynomial2;
    // p = ξ² η (0.5 + 2 ξ) + 0.7 (−1 + 3 η) − 0.25
    const P p = P::monomial(2, 1) * P::linearInXi(0.5, 2.0) + P::linearInEta(-1.0, 3.0).scaled(0.7) - P::constant(0.25);
    const double x = 0.3;
    const double y = -0.7;
    const double value = x * x * y * (0.5 + 2.0 * x) + 0.7 * (-1.0 + 3.0 * y) - 0.25;
    const double px = y * (x + 6.0 * x * x);
    const double py = x * x * (0.5 + 2.0 * x) + 2.1;
    const double pxx = y * (1.0 + 12.0 * x);
    const double pxy = x + 6.0 * x * x;
    const double pyy = 0.0;
    const std::array<double, 6> jet = p.jet(x, y);
    const double expected[6] = {value, px, py, pxx, pxy, pyy};
    for (std::size_t k = 0; k < 6; ++k)
        runner.expectNear(jet[k], expected[k], 1e-15, "jet " + std::to_string(k));
    runner.expectNear(p.evaluate(x, y), value, 1e-15, "evaluate");
    // The jet is formed without the derived polynomials: its values are theirs to the bit, on a polynomial with
    // every coefficient of its degree and on the zero polynomial.
    P full;
    for (std::size_t i = 0; i < 7; ++i)
        for (std::size_t j = 0; j < 6; ++j)
            full.coefficient(i, j) = std::sin(1.0 + 3.0 * static_cast<double>(i) + 0.7 * static_cast<double>(j));
    for (const P& q : {p, full, P::zero(), P::constant(-2.5), P::monomial(0, 5), P::monomial(5, 0)})
    {
        const std::array<double, 6> formed = q.jet(x, y);
        const P dx = q.dXi();
        const P dy = q.dEta();
        const double derived[6] = {q.evaluate(x, y),        dx.evaluate(x, y),        dy.evaluate(x, y),
                                   dx.dXi().evaluate(x, y), dx.dEta().evaluate(x, y), dy.dEta().evaluate(x, y)};
        for (std::size_t k = 0; k < 6; ++k)
            runner.expect(formed[k] == derived[k], "jet " + std::to_string(k) + " is the derived polynomial's value");
    }
    runner.expect(p.dXi().dEta() == p.dEta().dXi(), "mixed derivatives commute");
    runner.expect(p - p == P::zero(), "p - p = 0");
    runner.expect(p.coefficient(3, 1) == 2.0, "coefficient of xi^3 eta");
    // A product beyond the degree bound is a bug of the caller.
    bool thrown = false;
    try
    {
        static_cast<void>(P::monomial(6, 0) * P::monomial(6, 0));
    }
    catch (const oos::base::InternalError&)
    {
        thrown = true;
    }
    runner.expect(thrown, "degree overflow refused");
    // The Legendre polynomial of the spaces.
    runner.expectNear(mh::legendreValue(5, 0.3), mh::legendreTable(5, 0.3).values[5], 1e-16, "P_5");
    runner.expectNear(mh::legendreValue(0, 0.3), 1.0, 0.0, "P_0");
}
