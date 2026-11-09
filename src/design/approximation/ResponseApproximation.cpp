#include "oos/design/approximation/ResponseApproximation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>

namespace oos::design
{

namespace
{

// The DesignConstrScreen types the mixed method takes in reciprocal variables (isReciprocalType); the response kinds of
// the optimisation aids are displacements of the structure, as DISP.
constexpr std::array<std::string_view, 14> RECIPROCAL_TYPES{"Displacement",
                                                            "ConstraintForce",
                                                            "Stress",
                                                            "Strain",
                                                            "Force",
                                                            "StrainEnergy",
                                                            "CompositeStress",
                                                            "CompositeStrain",
                                                            "CompositeFailure",
                                                            "CompositeStrengthRatio",
                                                            "Compliance",
                                                            "MeanDisplacement",
                                                            "DeflectionToThickness",
                                                            "Rotation"};

} // namespace

std::string_view approximationKindName(ApproximationKind kind) noexcept
{
    switch (kind)
    {
    case ApproximationKind::Direct:
        return "direct";
    case ApproximationKind::Reciprocal:
        return "reciprocal";
    case ApproximationKind::Mixed:
        return "mixed";
    case ApproximationKind::Convex:
        return "convex";
    case ApproximationKind::Hints:
        return "hints";
    }
    return "";
}

std::optional<ApproximationKind> findApproximationKind(std::string_view name) noexcept
{
    for (const ApproximationKind kind : {ApproximationKind::Direct, ApproximationKind::Reciprocal,
                                         ApproximationKind::Mixed, ApproximationKind::Convex, ApproximationKind::Hints})
        if (approximationKindName(kind) == name)
            return kind;
    return std::nullopt;
}

ApproximationKind approximationKindOfCode(std::int64_t approximationCode) noexcept
{
    switch (approximationCode)
    {
    case 1:
        return ApproximationKind::Direct;
    case 3:
        return ApproximationKind::Convex;
    default:
        return ApproximationKind::Mixed;
    }
}

bool isReciprocalType(std::string_view responseType) noexcept
{
    return std::find(RECIPROCAL_TYPES.begin(), RECIPROCAL_TYPES.end(), responseType) != RECIPROCAL_TYPES.end();
}

ResponseApproximation::ResponseApproximation(std::span<const double> x0, std::span<const double> lower)
    : m_x0(x0.begin(), x0.end()), m_canBeReciprocal(x0.size(), 0)
{
    for (std::size_t i = 0; i < x0.size(); ++i)
        m_canBeReciprocal[i] = x0[i] > 0.0 && i < lower.size() && lower[i] > 0.0 ? 1 : 0;
}

bool ResponseApproximation::isReciprocal(ApproximationKind kind, std::size_t i, double gradient) const noexcept
{
    if (m_canBeReciprocal[i] == 0)
        return false;
    switch (kind)
    {
    case ApproximationKind::Reciprocal:
        return true;
    case ApproximationKind::Convex:
        return gradient < 0.0;
    default:
        return false;
    }
}

double ResponseApproximation::value(ApproximationKind kind, double value0, std::span<const double> gradient0,
                                    std::span<const double> x, std::span<const int> powers) const noexcept
{
    double value = value0;
    for (std::size_t i = 0; i < m_x0.size(); ++i)
    {
        const double step = x[i] - m_x0[i];
        if (kind == ApproximationKind::Hints)
        {
            const int power = i < powers.size() ? powers[i] : 1;
            if (power == 1 || m_canBeReciprocal[i] == 0)
                value += gradient0[i] * step;
            else
            {
                // t = g·(x^p − x0^p)/(p·x0^(p−1)) (class comment).
                const double p = static_cast<double>(power);
                value += gradient0[i] * (std::pow(x[i], p) - std::pow(m_x0[i], p)) / (p * std::pow(m_x0[i], p - 1.0));
            }
            continue;
        }
        value += isReciprocal(kind, i, gradient0[i]) ? gradient0[i] * m_x0[i] * step / x[i] : gradient0[i] * step;
    }
    return value;
}

void ResponseApproximation::gradient(ApproximationKind kind, std::span<const double> gradient0,
                                     std::span<const double> x, std::span<double> gradient,
                                     std::span<const int> powers) const noexcept
{
    for (std::size_t i = 0; i < m_x0.size(); ++i)
    {
        if (kind == ApproximationKind::Hints)
        {
            const int power = i < powers.size() ? powers[i] : 1;
            gradient[i] = power == 1 || m_canBeReciprocal[i] == 0
                              ? gradient0[i]
                              : gradient0[i] * std::pow(x[i] / m_x0[i], static_cast<double>(power) - 1.0);
            continue;
        }
        const double ratio = m_x0[i] / x[i];
        gradient[i] = isReciprocal(kind, i, gradient0[i]) ? gradient0[i] * ratio * ratio : gradient0[i];
    }
}

} // namespace oos::design
