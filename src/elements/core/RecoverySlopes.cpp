#include "oos/elements/core/RecoverySlopes.hpp"

#include <cmath>

namespace oos::elements
{

double planeVonMisesRate(const ColumnForm& form, const std::array<double, 3>& values,
                         const std::array<double, 3>& rates) noexcept
{
    const double a = values[0];
    const double b = values[1];
    const double c = values[2];
    const double k = form.shearFactor;
    const double q = a * a + b * b - a * b + 3.0 * k * k * c * c;
    if (!(q > 0.0))
        return 0.0;
    const double dq = (2.0 * a - b) * rates[0] + (2.0 * b - a) * rates[1] + 6.0 * k * k * c * rates[2];
    return form.scale * dq / (2.0 * std::sqrt(q));
}

double planeVonMisesSecondRate(const ColumnForm& form, const std::array<double, 3>& values,
                               const std::array<double, 3>& ratesA, const std::array<double, 3>& ratesB,
                               const std::array<double, 3>& second) noexcept
{
    const double a = values[0];
    const double b = values[1];
    const double c = values[2];
    const double k2 = form.shearFactor * form.shearFactor;
    const double q = a * a + b * b - a * b + 3.0 * k2 * c * c;
    if (!(q > 0.0))
        return 0.0;
    const std::array<double, 3> gradient{2.0 * a - b, 2.0 * b - a, 6.0 * k2 * c};
    const double dqA = gradient[0] * ratesA[0] + gradient[1] * ratesA[1] + gradient[2] * ratesA[2];
    const double dqB = gradient[0] * ratesB[0] + gradient[1] * ratesB[1] + gradient[2] * ratesB[2];
    const double curvature = 2.0 * ratesA[0] * ratesB[0] - ratesA[0] * ratesB[1] - ratesA[1] * ratesB[0] +
                             2.0 * ratesA[1] * ratesB[1] + 6.0 * k2 * ratesA[2] * ratesB[2];
    const double d2q = curvature + gradient[0] * second[0] + gradient[1] * second[1] + gradient[2] * second[2];
    const double root = std::sqrt(q);
    return form.scale * (d2q / (2.0 * root) - dqA * dqB / (4.0 * q * root));
}

void RecoverySlopes::reset(std::span<const ElementDof> dofs, std::size_t columnCount)
{
    m_dofs.assign(dofs.begin(), dofs.end());
    m_columnCount = columnCount;
    m_places.clear();
    m_values.clear();
}

std::size_t RecoverySlopes::appendRow(ResultPlace place)
{
    m_places.push_back(place);
    m_values.resize(m_values.size() + m_columnCount * m_dofs.size(), 0.0);
    return m_places.size() - 1;
}

std::span<double> RecoverySlopes::slopes(std::size_t row, std::size_t column)
{
    return std::span<double>(m_values).subspan((row * m_columnCount + column) * m_dofs.size(), m_dofs.size());
}

std::span<const double> RecoverySlopes::slopes(std::size_t row, std::size_t column) const
{
    return std::span<const double>(m_values).subspan((row * m_columnCount + column) * m_dofs.size(), m_dofs.size());
}

} // namespace oos::elements
