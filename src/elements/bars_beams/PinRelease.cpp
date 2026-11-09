#include "oos/elements/bars_beams/PinRelease.hpp"

#include <cmath>
#include <cstddef>

namespace oos::elements
{

std::optional<PinRelease> PinRelease::make(const base::MatN<12>& stiffness, const std::array<bool, 12>& isReleased)
{
    PinRelease release;
    release.m_isReleased = isReleased;
    release.m_stiffness = stiffness;
    std::array<std::size_t, 12> released{};
    std::size_t count = 0;
    for (std::size_t i = 0; i < 12; ++i)
    {
        if (isReleased[i])
            released[count++] = i;
    }

    // Cholesky K_rr = L·Lᵀ (at most 12 × 12, in place in `l`), then K_rr⁻¹ = L⁻ᵀ·L⁻¹ column by column.
    base::MatN<12> l;
    for (std::size_t i = 0; i < count; ++i)
        for (std::size_t j = 0; j < count; ++j)
            l(i, j) = stiffness(released[i], released[j]);
    for (std::size_t j = 0; j < count; ++j)
    {
        const double original = l(j, j);
        double pivot = original;
        for (std::size_t k = 0; k < j; ++k)
            pivot -= l(j, k) * l(j, k);
        if (!(original > 0.0) || pivot <= PIN_RELEASE_PIVOT_RATIO * original)
            return std::nullopt;
        l(j, j) = std::sqrt(pivot);
        for (std::size_t i = j + 1; i < count; ++i)
        {
            double sum = l(i, j);
            for (std::size_t k = 0; k < j; ++k)
                sum -= l(i, k) * l(j, k);
            l(i, j) = sum / l(j, j);
        }
    }
    for (std::size_t column = 0; column < count; ++column)
    {
        std::array<double, 12> y{};
        for (std::size_t i = 0; i < count; ++i) // L·y = e_column
        {
            double sum = i == column ? 1.0 : 0.0;
            for (std::size_t k = 0; k < i; ++k)
                sum -= l(i, k) * y[k];
            y[i] = sum / l(i, i);
        }
        for (std::size_t i = count; i-- > 0;) // Lᵀ·x = y
        {
            double sum = y[i];
            for (std::size_t k = i + 1; k < count; ++k)
                sum -= l(k, i) * y[k];
            y[i] = sum / l(i, i);
        }
        for (std::size_t i = 0; i < count; ++i)
            release.m_inverse(released[i], released[column]) = y[i];
    }

    // K_c = K_aa − K_ar·K_rr⁻¹·K_ra on the kept rows and columns.
    for (std::size_t i = 0; i < 12; ++i)
        for (std::size_t j = 0; j < 12; ++j)
        {
            if (isReleased[i] || isReleased[j])
                continue;
            double correction = 0.0;
            for (std::size_t p = 0; p < count; ++p)
                for (std::size_t q = 0; q < count; ++q)
                    correction += stiffness(i, released[p]) * release.m_inverse(released[p], released[q]) *
                                  stiffness(released[q], j);
            release.m_condensed(i, j) = stiffness(i, j) - correction;
        }
    // Exactly symmetric, as K is.
    for (std::size_t i = 0; i < 12; ++i)
        for (std::size_t j = i + 1; j < 12; ++j)
            release.m_condensed(i, j) = release.m_condensed(j, i) =
                0.5 * (release.m_condensed(i, j) + release.m_condensed(j, i));
    return release;
}

std::array<double, 12> PinRelease::condenseLoads(const std::array<double, 12>& loads) const
{
    // y = K_rr⁻¹·f_r, then f_c = f_a − K_ar·y.
    const std::array<double, 12> y = m_inverse * loads;
    std::array<double, 12> condensed{};
    for (std::size_t i = 0; i < 12; ++i)
    {
        if (m_isReleased[i])
            continue;
        double sum = loads[i];
        for (std::size_t j = 0; j < 12; ++j)
        {
            if (m_isReleased[j])
                sum -= m_stiffness(i, j) * y[j];
        }
        condensed[i] = sum;
    }
    return condensed;
}

std::array<double, 12> PinRelease::completeDisplacements(const std::array<double, 12>& kept,
                                                         const std::array<double, 12>& loads) const
{
    // d_r = K_rr⁻¹·(f_r − K_ra·d_a).
    std::array<double, 12> rightHandSide{};
    for (std::size_t i = 0; i < 12; ++i)
    {
        if (!m_isReleased[i])
            continue;
        double sum = loads[i];
        for (std::size_t j = 0; j < 12; ++j)
        {
            if (!m_isReleased[j])
                sum -= m_stiffness(i, j) * kept[j];
        }
        rightHandSide[i] = sum;
    }
    const std::array<double, 12> released = m_inverse * rightHandSide;
    std::array<double, 12> complete = kept;
    for (std::size_t i = 0; i < 12; ++i)
    {
        if (m_isReleased[i])
            complete[i] = released[i];
    }
    return complete;
}

} // namespace oos::elements
