#include "oos/elements/mixedHybrid/core/PolynomialBases.hpp"

namespace oos::elements::mixedHybrid
{

PolynomialTable legendreTable(std::size_t degree, double x)
{
    PolynomialTable table{std::vector<double>(degree + 1, 0.0), std::vector<double>(degree + 1, 0.0)};
    std::vector<double>& p = table.values;
    std::vector<double>& dp = table.derivatives;
    p[0] = 1.0;
    if (degree >= 1)
    {
        p[1] = x;
        dp[1] = 1.0;
    }
    for (std::size_t k = 2; k <= degree; ++k)
    {
        const double kf = static_cast<double>(k);
        p[k] = ((2.0 * kf - 1.0) * x * p[k - 1] - (kf - 1.0) * p[k - 2]) / kf;
        dp[k] = dp[k - 2] + (2.0 * kf - 1.0) * p[k - 1];
    }
    return table;
}

double legendreValue(std::size_t n, double x)
{
    if (n == 0)
        return 1.0;
    double p0 = 1.0;
    double p1 = x;
    for (std::size_t k = 2; k <= n; ++k)
    {
        const double kf = static_cast<double>(k);
        const double p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
        p0 = p1;
        p1 = p2;
    }
    return p1;
}

void legendreBasis(std::size_t degree, double u, double v, std::span<double> out)
{
    const PolynomialTable pu = legendreTable(degree, u);
    const PolynomialTable pv = legendreTable(degree, v);
    std::size_t k = 0;
    for (std::size_t total = 0; total <= degree; ++total)
        for (std::size_t i = 0; i <= total; ++i)
            out[k++] = pu.values[i] * pv.values[total - i];
}

void legendreBasisWithDerivatives(std::size_t degree, double u, double v, std::span<double> values,
                                  std::span<double> du, std::span<double> dv)
{
    const PolynomialTable pu = legendreTable(degree, u);
    const PolynomialTable pv = legendreTable(degree, v);
    std::size_t k = 0;
    for (std::size_t total = 0; total <= degree; ++total)
    {
        for (std::size_t i = 0; i <= total; ++i)
        {
            const std::size_t j = total - i;
            values[k] = pu.values[i] * pv.values[j];
            du[k] = pu.derivatives[i] * pv.values[j];
            dv[k] = pu.values[i] * pv.derivatives[j];
            ++k;
        }
    }
}

PolynomialTable jacobiTable(std::size_t degree, double alpha, double x)
{
    PolynomialTable table{std::vector<double>(degree + 1, 0.0), std::vector<double>(degree + 1, 0.0)};
    std::vector<double>& p = table.values;
    std::vector<double>& dp = table.derivatives;
    p[0] = 1.0;
    if (degree >= 1)
    {
        p[1] = 0.5 * (alpha + (alpha + 2.0) * x);
        dp[1] = 0.5 * (alpha + 2.0);
    }
    for (std::size_t k = 1; k < degree; ++k)
    {
        const double n = static_cast<double>(k);
        const double c = 2.0 * n + alpha;
        const double den = 2.0 * (n + 1.0) * (n + alpha + 1.0) * c;
        const double a1 = (c + 1.0) * (c + 2.0) * c / den;
        const double a0 = (c + 1.0) * alpha * alpha / den;
        const double b = 2.0 * (n + alpha) * n * (c + 2.0) / den;
        p[k + 1] = (a1 * x + a0) * p[k] - b * p[k - 1];
        dp[k + 1] = (a1 * x + a0) * dp[k] + a1 * p[k] - b * dp[k - 1];
    }
    return table;
}

void triangleBasisWithDerivatives(std::size_t degree, double u, double v, std::span<double> values,
                                  std::span<double> du, std::span<double> dv)
{
    const double p = u + 0.5 * (1.0 + v);
    const double q = 0.5 * (1.0 - v);
    // s_i = q^i P_i(p / q) with its derivatives in p and q.
    std::vector<double> s(degree + 1, 0.0);
    std::vector<double> sp(degree + 1, 0.0);
    std::vector<double> sq(degree + 1, 0.0);
    s[0] = 1.0;
    if (degree >= 1)
    {
        s[1] = p;
        sp[1] = 1.0;
    }
    for (std::size_t i = 1; i < degree; ++i)
    {
        const double a = static_cast<double>(2 * i + 1) / static_cast<double>(i + 1);
        const double b = static_cast<double>(i) / static_cast<double>(i + 1);
        s[i + 1] = a * p * s[i] - b * q * q * s[i - 1];
        sp[i + 1] = a * (s[i] + p * sp[i]) - b * q * q * sp[i - 1];
        sq[i + 1] = a * p * sq[i] - b * (2.0 * q * s[i - 1] + q * q * sq[i - 1]);
    }
    // P_j^(2i+1,0)(v) for each i.
    std::vector<PolynomialTable> jacobi;
    jacobi.reserve(degree + 1);
    for (std::size_t i = 0; i <= degree; ++i)
        jacobi.push_back(jacobiTable(degree - i, static_cast<double>(2 * i + 1), v));
    std::size_t k = 0;
    for (std::size_t total = 0; total <= degree; ++total)
    {
        for (std::size_t i = 0; i <= total; ++i)
        {
            const std::size_t j = total - i;
            const double pj = jacobi[i].values[j];
            const double dpj = jacobi[i].derivatives[j];
            values[k] = s[i] * pj;
            du[k] = sp[i] * pj;
            dv[k] = 0.5 * (sp[i] - sq[i]) * pj + s[i] * dpj;
            ++k;
        }
    }
}

void triangleBasis(std::size_t degree, double u, double v, std::span<double> out)
{
    const std::size_t n = totalDegreeCount2D(degree);
    std::vector<double> du(n);
    std::vector<double> dv(n);
    triangleBasisWithDerivatives(degree, u, v, out.first(n), du, dv);
}

} // namespace oos::elements::mixedHybrid
