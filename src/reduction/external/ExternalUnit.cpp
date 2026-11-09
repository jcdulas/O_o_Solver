#include "oos/reduction/external/ExternalUnit.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/reduction/cache/CacheHasher.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace oos::reduction
{

namespace
{

// The design stamp of a unit no variable drives (design::EMPTY_DESIGN_STAMP, which this layer cannot include): an
// external unit's stamp is never it.
constexpr std::uint64_t EMPTY_STAMP = 0;

bool isBefore(const UnitDofLabel& a, const UnitDofLabel& b) noexcept
{
    return a.id != b.id ? a.id < b.id : a.component < b.component;
}

std::string labelText(const UnitDofLabel& label)
{
    return base::concatenate(
        {"(grid ", std::to_string(label.id), ", component ", std::to_string(label.component), ")"});
}

// Throws when `matrix` is not n × n or holds a value that is not finite.
void requireSquare(const linalg::DenseMatrix& matrix, std::size_t n, const std::string& zone, const char* what)
{
    if (matrix.rows() != n || matrix.columns() != n)
        throw std::invalid_argument(
            base::concatenate({"external unit of ", zone, ": expected ", what, " of ", std::to_string(n), " x ",
                               std::to_string(n), " (one row and one column per label), found ",
                               std::to_string(matrix.rows()), " x ", std::to_string(matrix.columns())}));
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            if (!std::isfinite(matrix(i, j)))
                throw std::invalid_argument(base::concatenate(
                    {"external unit of ", zone, ": ", what, " holds ", std::to_string(matrix(i, j)), " at row ",
                     std::to_string(i), ", column ", std::to_string(j), "; expected finite values"}));
}

void mixMatrix(CacheHasher& hasher, const linalg::DenseMatrix& matrix) noexcept
{
    hasher.mix(static_cast<std::uint64_t>(matrix.rows()));
    hasher.mix(static_cast<std::uint64_t>(matrix.columns()));
    for (std::size_t j = 0; j < matrix.columns(); ++j)
        for (std::size_t i = 0; i < matrix.rows(); ++i)
            hasher.mix(matrix(i, j));
}

} // namespace

ExternalUnit::ExternalUnit(ExternalUnitParts parts) : m_parts(std::move(parts))
{
    const std::string zone = displayName();
    const std::size_t n = m_parts.labels.size();
    if (n == 0)
        throw std::invalid_argument(
            base::concatenate({"external unit of ", zone, ": expected the labels of its contour DOFs, found none"}));
    for (const UnitDofLabel& label : m_parts.labels.span())
        if (label.kind != UnitPointKind::Grid || label.component < 1 || label.component > 6)
            throw std::invalid_argument(base::concatenate(
                {"external unit of ", zone, ": expected labels of grid points with a component from 1 to 6, found ",
                 labelText(label), "; an external unit acts on the contour grid points of its zone"}));
    m_rowsByLabel.resize(n);
    std::iota(m_rowsByLabel.begin(), m_rowsByLabel.end(), std::size_t{0});
    std::stable_sort(m_rowsByLabel.begin(), m_rowsByLabel.end(),
                     [&](std::size_t a, std::size_t b) { return isBefore(m_parts.labels[a], m_parts.labels[b]); });
    for (std::size_t k = 1; k < n; ++k)
        if (m_parts.labels[m_rowsByLabel[k - 1]] == m_parts.labels[m_rowsByLabel[k]])
            throw std::invalid_argument(base::concatenate(
                {"external unit of ", zone, ": the label ", labelText(m_parts.labels[m_rowsByLabel[k]]), " names rows ",
                 std::to_string(m_rowsByLabel[k - 1]), " and ", std::to_string(m_rowsByLabel[k]),
                 "; give each contour DOF once"}));
    requireSquare(m_parts.stiffness, n, zone, "the stiffness K*");
    if (m_parts.mass)
        requireSquare(*m_parts.mass, n, zone, "the mass M*");
    std::vector<std::size_t> variables; // bounded by the derivatives
    for (const ExternalUnitDerivative& derivative : m_parts.derivatives)
    {
        const std::string what =
            base::concatenate({"the derivative of K* with respect to variable ", std::to_string(derivative.variable)});
        requireSquare(derivative.stiffness, n, zone, what.c_str());
        if (derivative.mass)
        {
            const std::string massWhat = base::concatenate(
                {"the derivative of M* with respect to variable ", std::to_string(derivative.variable)});
            requireSquare(*derivative.mass, n, zone, massWhat.c_str());
            if (!m_parts.mass)
                throw std::invalid_argument(base::concatenate({"external unit of ", zone,
                                                               ": a derivative of M* is given without M*; give the "
                                                               "mass, or no derivative of it"}));
        }
        variables.push_back(derivative.variable);
    }
    std::sort(variables.begin(), variables.end());
    if (std::adjacent_find(variables.begin(), variables.end()) != variables.end())
        throw std::invalid_argument(
            base::concatenate({"external unit of ", zone,
                               ": two derivatives are given with respect to the same variable; give one "
                               "per variable"}));
    m_designStamp = computeDesignStamp();
}

std::string ExternalUnit::displayName() const
{
    return std::string("zone ").append(m_parts.zoneName.empty() ? std::to_string(m_parts.zoneId) : m_parts.zoneName);
}

std::optional<std::size_t> ExternalUnit::rowOf(const UnitDofLabel& label) const noexcept
{
    const auto found = std::lower_bound(m_rowsByLabel.begin(), m_rowsByLabel.end(), label,
                                        [&](std::size_t row, const UnitDofLabel& wanted)
                                        { return isBefore(m_parts.labels[row], wanted); });
    if (found == m_rowsByLabel.end() || !(m_parts.labels[*found] == label))
        return std::nullopt;
    return *found;
}

std::uint64_t ExternalUnit::computeDesignStamp() const noexcept
{
    CacheHasher hasher;
    hasher.mix(m_parts.zoneId);
    hasher.mix(std::string_view(m_parts.identity));
    hasher.mix(static_cast<std::uint64_t>(m_parts.labels.size()));
    for (const UnitDofLabel& label : m_parts.labels.span())
    {
        hasher.mix(label.id);
        hasher.mix(static_cast<std::int32_t>(label.component));
    }
    mixMatrix(hasher, m_parts.stiffness);
    hasher.mix(static_cast<std::uint64_t>(m_parts.mass ? 1 : 0));
    if (m_parts.mass)
        mixMatrix(hasher, *m_parts.mass);
    hasher.mix(static_cast<std::uint64_t>(m_parts.derivatives.size()));
    for (const ExternalUnitDerivative& derivative : m_parts.derivatives)
    {
        hasher.mix(static_cast<std::uint64_t>(derivative.variable));
        mixMatrix(hasher, derivative.stiffness);
        hasher.mix(static_cast<std::uint64_t>(derivative.mass ? 1 : 0));
        if (derivative.mass)
            mixMatrix(hasher, *derivative.mass);
    }
    hasher.mix(static_cast<std::uint64_t>(m_parts.loads ? 1 : 0));
    return hasher.value() == EMPTY_STAMP ? 1 : hasher.value();
}

} // namespace oos::reduction
