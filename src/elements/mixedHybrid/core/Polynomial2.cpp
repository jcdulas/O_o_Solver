#include "oos/elements/mixedHybrid/core/Polynomial2.hpp"

#include "oos/base/errors/InternalError.hpp"

namespace oos::elements::mixedHybrid
{

namespace
{

constexpr std::size_t BOUND = Polynomial2::DEGREE_BOUND;

} // namespace

Polynomial2 Polynomial2::monomial(std::size_t i, std::size_t j)
{
    Polynomial2 p;
    p.coefficient(i, j) = 1.0;
    return p;
}

Polynomial2 Polynomial2::constant(double c)
{
    Polynomial2 p;
    p.coefficient(0, 0) = c;
    return p;
}

Polynomial2 Polynomial2::linearInXi(double a, double b)
{
    Polynomial2 p = constant(a);
    p.coefficient(1, 0) = b;
    return p;
}

Polynomial2 Polynomial2::linearInEta(double a, double b)
{
    Polynomial2 p = constant(a);
    p.coefficient(0, 1) = b;
    return p;
}

Polynomial2 Polynomial2::scaled(double s) const
{
    Polynomial2 r = *this;
    for (double& c : r.m_coefficients)
        c *= s;
    return r;
}

// ∂(ξ^i η^j)/∂ξ = i ξ^(i−1) η^j.
Polynomial2 Polynomial2::dXi() const
{
    Polynomial2 r;
    for (std::size_t i = 1; i < BOUND; ++i)
        for (std::size_t j = 0; j < BOUND; ++j)
            r.coefficient(i - 1, j) = static_cast<double>(i) * coefficient(i, j);
    return r;
}

// ∂(ξ^i η^j)/∂η = j ξ^i η^(j−1).
Polynomial2 Polynomial2::dEta() const
{
    Polynomial2 r;
    for (std::size_t i = 0; i < BOUND; ++i)
        for (std::size_t j = 1; j < BOUND; ++j)
            r.coefficient(i, j - 1) = static_cast<double>(j) * coefficient(i, j);
    return r;
}

double Polynomial2::evaluate(double xi, double eta) const
{
    std::array<double, BOUND> powersXi{};
    std::array<double, BOUND> powersEta{};
    powersXi[0] = 1.0;
    powersEta[0] = 1.0;
    for (std::size_t i = 1; i < BOUND; ++i)
    {
        powersXi[i] = powersXi[i - 1] * xi;
        powersEta[i] = powersEta[i - 1] * eta;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < BOUND; ++i)
    {
        double inner = 0.0;
        for (std::size_t j = 0; j < BOUND; ++j)
            inner += coefficient(i, j) * powersEta[j];
        sum += inner * powersXi[i];
    }
    return sum;
}

std::array<double, 6> Polynomial2::jet(double xi, double eta) const
{
    const Polynomial2 dx = dXi();
    const Polynomial2 dy = dEta();
    return {evaluate(xi, eta),          dx.evaluate(xi, eta),        dy.evaluate(xi, eta),
            dx.dXi().evaluate(xi, eta), dx.dEta().evaluate(xi, eta), dy.dEta().evaluate(xi, eta)};
}

Polynomial2 operator+(const Polynomial2& a, const Polynomial2& b)
{
    Polynomial2 r;
    for (std::size_t k = 0; k < r.m_coefficients.size(); ++k)
        r.m_coefficients[k] = a.m_coefficients[k] + b.m_coefficients[k];
    return r;
}

Polynomial2 operator-(const Polynomial2& a, const Polynomial2& b)
{
    Polynomial2 r;
    for (std::size_t k = 0; k < r.m_coefficients.size(); ++k)
        r.m_coefficients[k] = a.m_coefficients[k] - b.m_coefficients[k];
    return r;
}

// (Σ a_ij ξ^i η^j)(Σ b_kl ξ^k η^l) = Σ a_ij b_kl ξ^(i+k) η^(j+l), the zero coefficients skipped.
Polynomial2 operator*(const Polynomial2& a, const Polynomial2& b)
{
    Polynomial2 r;
    for (std::size_t i = 0; i < BOUND; ++i)
    {
        for (std::size_t j = 0; j < BOUND; ++j)
        {
            const double aij = a.coefficient(i, j);
            if (aij == 0.0)
                continue;
            for (std::size_t k = 0; k < BOUND; ++k)
            {
                for (std::size_t l = 0; l < BOUND; ++l)
                {
                    const double bkl = b.coefficient(k, l);
                    if (bkl == 0.0)
                        continue;
                    if (i + k >= BOUND || j + l >= BOUND)
                        throw base::InternalError("polynomial degree beyond the reference tables");
                    r.coefficient(i + k, j + l) += aij * bkl;
                }
            }
        }
    }
    return r;
}

} // namespace oos::elements::mixedHybrid
