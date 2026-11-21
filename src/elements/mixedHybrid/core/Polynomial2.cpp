#include "oos/elements/mixedHybrid/core/Polynomial2.hpp"

#include "oos/base/errors/InternalError.hpp"

namespace oos::elements::mixedHybrid
{

namespace
{

constexpr std::size_t BOUND = Polynomial2::DEGREE_BOUND;

// 1, x, x², ... up to the degree bound.
std::array<double, BOUND> powersOf(double x)
{
    std::array<double, BOUND> powers{};
    powers[0] = 1.0;
    for (std::size_t i = 1; i < BOUND; ++i)
        powers[i] = powers[i - 1] * x;
    return powers;
}

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
    for (std::size_t i = 0; i < m_rows; ++i)
        for (std::size_t j = 0; j < m_columns; ++j)
            r.m_coefficients[i * BOUND + j] *= s;
    return r;
}

// ∂(ξ^i η^j)/∂ξ = i ξ^(i−1) η^j.
Polynomial2 Polynomial2::dXi() const
{
    Polynomial2 r;
    for (std::size_t i = 1; i < m_rows; ++i)
        for (std::size_t j = 0; j < m_columns; ++j)
            r.coefficient(i - 1, j) = static_cast<double>(i) * coefficient(i, j);
    return r;
}

// ∂(ξ^i η^j)/∂η = j ξ^i η^(j−1).
Polynomial2 Polynomial2::dEta() const
{
    Polynomial2 r;
    for (std::size_t i = 0; i < m_rows; ++i)
        for (std::size_t j = 1; j < m_columns; ++j)
            r.coefficient(i, j - 1) = static_cast<double>(j) * coefficient(i, j);
    return r;
}

double Polynomial2::evaluate(double xi, double eta) const
{
    const std::array<double, BOUND> powersXi = powersOf(xi);
    const std::array<double, BOUND> powersEta = powersOf(eta);
    double sum = 0.0;
    for (std::size_t i = 0; i < m_rows; ++i)
    {
        double inner = 0.0;
        for (std::size_t j = 0; j < m_columns; ++j)
            inner += coefficient(i, j) * powersEta[j];
        sum += inner * powersXi[i];
    }
    return sum;
}

// Each derivative is the sum of evaluate() on the coefficients of the derived polynomial, written without forming
// that polynomial: ∂ξ has i c_ij at (i − 1, j), ∂η has j c_ij at (i, j − 1), and the second derivatives derive those
// once more ((i − 1)(i c_ij), j (i c_ij), (j − 1)(j c_ij)). The products and the order of the sums are those of
// dXi(), dEta() and evaluate(), so the values are theirs to the bit.
std::array<double, 6> Polynomial2::jet(double xi, double eta) const
{
    const std::array<double, BOUND> powersXi = powersOf(xi);
    const std::array<double, BOUND> powersEta = powersOf(eta);
    double value = 0.0;
    double dXiValue = 0.0;
    double dEtaValue = 0.0;
    double dXiXi = 0.0;
    double dXiEta = 0.0;
    double dEtaEta = 0.0;
    for (std::size_t i = 0; i < m_rows; ++i)
    {
        const double rowFactor = static_cast<double>(i);
        // the sums over j of row i of the polynomial, of its ∂η and ∂ηη, and of row i − 1 of its ∂ξ and ∂ξη
        double row = 0.0;
        double rowEta = 0.0;
        double rowEtaEta = 0.0;
        double rowXi = 0.0;
        double rowXiEta = 0.0;
        for (std::size_t j = 0; j < m_columns; ++j)
        {
            const double c = coefficient(i, j);
            const double cXi = rowFactor * c;
            row += c * powersEta[j];
            rowXi += cXi * powersEta[j];
            if (j >= 1)
            {
                const double columnFactor = static_cast<double>(j);
                const double cEta = columnFactor * c;
                rowEta += cEta * powersEta[j - 1];
                rowXiEta += (columnFactor * cXi) * powersEta[j - 1];
                if (j >= 2)
                    rowEtaEta += (static_cast<double>(j - 1) * cEta) * powersEta[j - 2];
            }
        }
        value += row * powersXi[i];
        dEtaValue += rowEta * powersXi[i];
        dEtaEta += rowEtaEta * powersXi[i];
        if (i >= 1)
        {
            dXiValue += rowXi * powersXi[i - 1];
            dXiEta += rowXiEta * powersXi[i - 1];
        }
        if (i >= 2)
        {
            // row i − 2 of ∂ξξ: (i − 1)(i c_ij)
            double rowXiXi = 0.0;
            for (std::size_t j = 0; j < m_columns; ++j)
                rowXiXi += (static_cast<double>(i - 1) * (rowFactor * coefficient(i, j))) * powersEta[j];
            dXiXi += rowXiXi * powersXi[i - 2];
        }
    }
    return {value, dXiValue, dEtaValue, dXiXi, dXiEta, dEtaEta};
}

Polynomial2 operator+(const Polynomial2& a, const Polynomial2& b)
{
    Polynomial2 r;
    r.m_rows = std::max(a.m_rows, b.m_rows);
    r.m_columns = std::max(a.m_columns, b.m_columns);
    for (std::size_t i = 0; i < r.m_rows; ++i)
        for (std::size_t j = 0; j < r.m_columns; ++j)
            r.m_coefficients[i * BOUND + j] = a.m_coefficients[i * BOUND + j] + b.m_coefficients[i * BOUND + j];
    return r;
}

Polynomial2 operator-(const Polynomial2& a, const Polynomial2& b)
{
    Polynomial2 r;
    r.m_rows = std::max(a.m_rows, b.m_rows);
    r.m_columns = std::max(a.m_columns, b.m_columns);
    for (std::size_t i = 0; i < r.m_rows; ++i)
        for (std::size_t j = 0; j < r.m_columns; ++j)
            r.m_coefficients[i * BOUND + j] = a.m_coefficients[i * BOUND + j] - b.m_coefficients[i * BOUND + j];
    return r;
}

// (Σ a_ij ξ^i η^j)(Σ b_kl ξ^k η^l) = Σ a_ij b_kl ξ^(i+k) η^(j+l), the zero coefficients skipped.
Polynomial2 operator*(const Polynomial2& a, const Polynomial2& b)
{
    Polynomial2 r;
    for (std::size_t i = 0; i < a.m_rows; ++i)
    {
        for (std::size_t j = 0; j < a.m_columns; ++j)
        {
            const double aij = a.coefficient(i, j);
            if (aij == 0.0)
                continue;
            for (std::size_t k = 0; k < b.m_rows; ++k)
            {
                for (std::size_t l = 0; l < b.m_columns; ++l)
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
