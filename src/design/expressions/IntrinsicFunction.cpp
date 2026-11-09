#include "oos/design/expressions/IntrinsicFunction.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>
#include <string_view>

namespace oos::design
{

namespace
{

constexpr double PI_VALUE = 3.14159265358979323846;
constexpr double LN10 = 2.30258509299404568402;
constexpr double QUIET_NAN = std::numeric_limits<double>::quiet_NaN();

// The constants of the A-weighting terms Ta1 and Ta2 of DBA and INVDBA (the A-weighting curve of
// IEC 61672: P1 to P4 are its pole frequencies in hertz; K1 and K3 make the gain 0 dB at 1 kHz).
constexpr double A_WEIGHTING_K1 = 2.242882e+16;
constexpr double A_WEIGHTING_K3 = 1.562339;
constexpr double A_WEIGHTING_P1 = 20.598997;
constexpr double A_WEIGHTING_P2 = 107.65265;
constexpr double A_WEIGHTING_P3 = 737.86223;
constexpr double A_WEIGHTING_P4 = 12194.22;

constexpr int ANY = VARIADIC_MAX_ARGUMENTS;

// In the order of the enum.
constexpr std::array<IntrinsicSignature, INTRINSIC_FUNCTION_COUNT> SIGNATURES{{
    {"ABS", 1, 1, "all reals"},
    {"ACOS", 1, 1, "-1 <= x <= 1"},
    {"ACOSH", 1, 1, "x >= 1"},
    {"ASIN", 1, 1, "-1 <= x <= 1"},
    {"ASINH", 1, 1, "all reals"},
    {"ATAN", 1, 1, "all reals"},
    {"ATAN2", 2, 2, "(x, y) != (0, 0)"},
    {"ATANH", 1, 1, "-1 < x < 1"},
    {"ATANH2", 2, 2, "y != 0 and -1 < x/y < 1"},
    {"AVG", 1, ANY, "all reals"},
    {"COS", 1, 1, "all reals"},
    {"COSH", 1, 1, "all reals"},
    {"DB", 2, 2, "P/PREF > 0"},
    {"DBA", 3, 3, "P/PREF > 0 and F != 0"},
    {"DIM", 2, 2, "all reals"},
    {"EXP", 1, 1, "all reals"},
    {"INVDB", 2, 2, "PREF > 0"},
    {"INVDBA", 3, 3, "PREF > 0 and F != 0"},
    {"LOG", 1, 1, "x > 0"},
    {"LOG10", 1, 1, "x > 0"},
    {"LOGX", 2, 2, "x > 0, x != 1 and y > 0"},
    {"MAX", 1, ANY, "all reals"},
    {"MIN", 1, ANY, "all reals"},
    {"MOD", 2, 2, "y != 0"},
    {"PI", 1, 1, "all reals"},
    {"RSS", 1, ANY, "all reals"},
    {"SIN", 1, 1, "all reals"},
    {"SINH", 1, 1, "all reals"},
    {"SQRT", 1, 1, "x >= 0"},
    {"SSQ", 1, ANY, "all reals"},
    {"SUM", 1, ANY, "all reals"},
    {"TAN", 1, 1, "all reals"},
    {"TANH", 1, 1, "all reals"},
}};

IntrinsicValue faulty(DomainFault fault) noexcept
{
    return {QUIET_NAN, fault};
}

// The A-weighting of DBA and INVDBA, 10·log10(Ta1) + 10·log10(Ta2) in decibels, and its derivative with respect to F:
//   Ta1 = K3·F⁴ / ((F² + P2²)·(F² + P3²)),   Ta2 = K1·F⁴ / ((F² + P1²)²·(F² + P4²)²)
//   d/dF = 10/ln10 · [ (4/F - 2F/(F²+P2²) - 2F/(F²+P3²)) + (4/F - 4F/(F²+P1²) - 4F/(F²+P4²)) ]
// Precondition: F != 0.
struct Weighting
{
    double decibels = 0.0;
    double derivative = 0.0;
};

Weighting aWeighting(double frequency) noexcept
{
    const double f2 = frequency * frequency;
    const double d1 = f2 + A_WEIGHTING_P1 * A_WEIGHTING_P1;
    const double d2 = f2 + A_WEIGHTING_P2 * A_WEIGHTING_P2;
    const double d3 = f2 + A_WEIGHTING_P3 * A_WEIGHTING_P3;
    const double d4 = f2 + A_WEIGHTING_P4 * A_WEIGHTING_P4;
    const double ta1 = A_WEIGHTING_K3 * f2 * f2 / (d2 * d3);
    const double ta2 = A_WEIGHTING_K1 * f2 * f2 / (d1 * d1 * d4 * d4);
    const double slope1 = 4.0 / frequency - 2.0 * frequency / d2 - 2.0 * frequency / d3;
    const double slope2 = 4.0 / frequency - 4.0 * frequency / d1 - 4.0 * frequency / d4;
    return {10.0 * std::log10(ta1) + 10.0 * std::log10(ta2), 10.0 / LN10 * (slope1 + slope2)};
}

// Writes the partials when they are asked for (an empty span means values only).
void setPartials(std::span<double> partials, std::initializer_list<double> values) noexcept
{
    if (partials.empty())
        return;
    std::size_t index = 0;
    for (const double value : values)
        partials[index++] = value;
}

// One-argument functions. The derivative of each is the textbook one; at the edge of a domain where the value exists
// but the slope is infinite (SQRT at 0, ACOS and ASIN at ±1, ACOSH at 1) the partial is left infinite.
IntrinsicValue unary(IntrinsicFunction function, double x, std::span<double> partials) noexcept
{
    switch (function)
    {
    case IntrinsicFunction::Abs:
        // Not differentiable at 0: the right derivative (+1) is taken there, so that the gradient is defined
        // everywhere and does not depend on the sign of a zero.
        setPartials(partials, {x >= 0.0 ? 1.0 : -1.0});
        return {std::fabs(x)};
    case IntrinsicFunction::Acos:
        if (!(std::fabs(x) <= 1.0))
            return faulty(DomainFault::OutsideDomain);
        setPartials(partials, {-1.0 / std::sqrt((1.0 - x) * (1.0 + x))});
        return {std::acos(x)};
    case IntrinsicFunction::Acosh:
        if (!(x >= 1.0))
            return faulty(DomainFault::OutsideDomain);
        setPartials(partials, {1.0 / std::sqrt((x - 1.0) * (x + 1.0))});
        return {std::acosh(x)};
    case IntrinsicFunction::Asin:
        if (!(std::fabs(x) <= 1.0))
            return faulty(DomainFault::OutsideDomain);
        setPartials(partials, {1.0 / std::sqrt((1.0 - x) * (1.0 + x))});
        return {std::asin(x)};
    case IntrinsicFunction::Asinh:
        setPartials(partials, {1.0 / std::sqrt(x * x + 1.0)});
        return {std::asinh(x)};
    case IntrinsicFunction::Atan:
        setPartials(partials, {1.0 / (1.0 + x * x)});
        return {std::atan(x)};
    case IntrinsicFunction::Atanh:
        if (!(std::fabs(x) < 1.0))
            return faulty(DomainFault::OutsideDomain);
        setPartials(partials, {1.0 / ((1.0 - x) * (1.0 + x))});
        return {std::atanh(x)};
    case IntrinsicFunction::Cos:
        setPartials(partials, {-std::sin(x)});
        return {std::cos(x)};
    case IntrinsicFunction::Cosh:
        setPartials(partials, {std::sinh(x)});
        return {std::cosh(x)};
    case IntrinsicFunction::Exp:
    {
        const double value = std::exp(x);
        setPartials(partials, {value});
        return {value};
    }
    case IntrinsicFunction::Log:
        if (!(x > 0.0))
            return faulty(DomainFault::OutsideDomain);
        setPartials(partials, {1.0 / x});
        return {std::log(x)};
    case IntrinsicFunction::Log10:
        if (!(x > 0.0))
            return faulty(DomainFault::OutsideDomain);
        setPartials(partials, {1.0 / (x * LN10)});
        return {std::log10(x)};
    case IntrinsicFunction::Pi:
        // Multiples of pi: PI(x) = π·x.
        setPartials(partials, {PI_VALUE});
        return {PI_VALUE * x};
    case IntrinsicFunction::Sin:
        setPartials(partials, {std::cos(x)});
        return {std::sin(x)};
    case IntrinsicFunction::Sinh:
        setPartials(partials, {std::cosh(x)});
        return {std::sinh(x)};
    case IntrinsicFunction::Sqrt:
    {
        if (!(x >= 0.0))
            return faulty(DomainFault::OutsideDomain);
        const double value = std::sqrt(x);
        setPartials(partials, {0.5 / value});
        return {value};
    }
    case IntrinsicFunction::Tan:
    {
        const double value = std::tan(x);
        setPartials(partials, {1.0 + value * value});
        return {value};
    }
    case IntrinsicFunction::Tanh:
    {
        const double value = std::tanh(x);
        setPartials(partials, {1.0 - value * value});
        return {value};
    }
    default:
        return faulty(DomainFault::OutsideDomain);
    }
}

IntrinsicValue binary(IntrinsicFunction function, double x, double y, std::span<double> partials) noexcept
{
    switch (function)
    {
    case IntrinsicFunction::Atan2:
    {
        // ATAN2(x, y) = tan⁻¹(x/y) with the quadrant of (y, x), as Fortran's ATAN2(Y, X) with the numerator first.
        if (x == 0.0 && y == 0.0)
            return faulty(DomainFault::OutsideDomain);
        const double radius2 = x * x + y * y;
        setPartials(partials, {y / radius2, -x / radius2});
        return {std::atan2(x, y)};
    }
    case IntrinsicFunction::Atanh2:
    {
        // ATANH2(x, y) = tanh⁻¹(x/y). With u = x/y: d/dx = 1/(y(1-u²)), d/dy = -u/(y(1-u²)).
        if (y == 0.0)
            return faulty(DomainFault::DivisionByZero);
        const double u = x / y;
        if (!(std::fabs(u) < 1.0))
            return faulty(DomainFault::OutsideDomain);
        const double slope = 1.0 / ((1.0 - u) * (1.0 + u));
        setPartials(partials, {slope / y, -slope * u / y});
        return {std::atanh(u)};
    }
    case IntrinsicFunction::Db:
    {
        // DB(P, PREF) = 20·log10(P/PREF).
        if (y == 0.0)
            return faulty(DomainFault::DivisionByZero);
        const double ratio = x / y;
        if (!(ratio > 0.0))
            return faulty(DomainFault::OutsideDomain);
        setPartials(partials, {20.0 / (LN10 * x), -20.0 / (LN10 * y)});
        return {20.0 * std::log10(ratio)};
    }
    case IntrinsicFunction::Dim:
        // DIM(x, y) = x - MIN(x, y): x - y when x > y, else 0. At x = y the value is 0 and so are the partials
        // (the branch x <= y is taken).
        if (x > y)
        {
            setPartials(partials, {1.0, -1.0});
            return {x - y};
        }
        setPartials(partials, {0.0, 0.0});
        return {0.0};
    case IntrinsicFunction::Invdb:
    {
        // INVDB(DB, PREF) = 10^(DB/20 + log10(PREF)).
        if (!(y > 0.0))
            return faulty(DomainFault::OutsideDomain);
        const double value = std::pow(10.0, x / 20.0 + std::log10(y));
        setPartials(partials, {value * LN10 / 20.0, value / y});
        return {value};
    }
    case IntrinsicFunction::Logx:
    {
        // LOGX(x, y) = log_x(y) = ln(y)/ln(x).
        if (!(x > 0.0) || !(y > 0.0))
            return faulty(DomainFault::OutsideDomain);
        const double lnX = std::log(x);
        if (lnX == 0.0)
            return faulty(DomainFault::DivisionByZero);
        const double value = std::log(y) / lnX;
        setPartials(partials, {-value / (x * lnX), 1.0 / (y * lnX)});
        return {value};
    }
    case IntrinsicFunction::Mod:
    {
        // MOD(x, y) = x - y·INT(x/y), INT truncating toward zero (Fortran), computed exactly by fmod. The
        // partials hold between the jumps: d/dx = 1, d/dy = -INT(x/y).
        if (y == 0.0)
            return faulty(DomainFault::DivisionByZero);
        const double value = std::fmod(x, y);
        setPartials(partials, {1.0, -std::round((x - value) / y)});
        return {value};
    }
    default:
        return faulty(DomainFault::OutsideDomain);
    }
}

IntrinsicValue ternary(IntrinsicFunction function, std::span<const double> arguments,
                       std::span<double> partials) noexcept
{
    const double level = arguments[0];
    const double reference = arguments[1];
    const double frequency = arguments[2];
    if (function == IntrinsicFunction::Dba)
    {
        // DBA(P, PREF, F) = 20·log10(P/PREF) + 10·log10(Ta1) + 10·log10(Ta2).
        if (reference == 0.0)
            return faulty(DomainFault::DivisionByZero);
        if (!(level / reference > 0.0) || frequency == 0.0)
            return faulty(DomainFault::OutsideDomain);
        const Weighting weighting = aWeighting(frequency);
        setPartials(partials, {20.0 / (LN10 * level), -20.0 / (LN10 * reference), weighting.derivative});
        return {20.0 * std::log10(level / reference) + weighting.decibels};
    }
    // INVDBA(DBA, PREF, F) = 10^((DBA - 10·log10(Ta1) - 10·log10(Ta2))/20 + log10(PREF)), the inverse of
    // DBA in P (the "+ log10(PREF)" term is read from INVDB, of which INVDBA is the A-weighted form).
    if (!(reference > 0.0) || frequency == 0.0)
        return faulty(DomainFault::OutsideDomain);
    const Weighting weighting = aWeighting(frequency);
    const double value = std::pow(10.0, (level - weighting.decibels) / 20.0 + std::log10(reference));
    const double slope = value * LN10 / 20.0;
    setPartials(partials, {slope, value / reference, -slope * weighting.derivative});
    return {value};
}

IntrinsicValue variadic(IntrinsicFunction function, std::span<const double> arguments,
                        std::span<double> partials) noexcept
{
    const std::size_t count = arguments.size();
    switch (function)
    {
    case IntrinsicFunction::Max:
    case IntrinsicFunction::Min:
    {
        // Not differentiable at a tie: the first argument (in the written order) that holds the extreme takes the
        // whole derivative, the same rule whatever the point, so that the gradient is deterministic.
        const bool isMax = function == IntrinsicFunction::Max;
        std::size_t chosen = 0;
        for (std::size_t index = 1; index < count; ++index)
        {
            if (isMax ? arguments[index] > arguments[chosen] : arguments[index] < arguments[chosen])
                chosen = index;
        }
        if (!partials.empty())
        {
            for (std::size_t index = 0; index < count; ++index)
                partials[index] = index == chosen ? 1.0 : 0.0;
        }
        return {arguments[chosen]};
    }
    case IntrinsicFunction::Avg:
    case IntrinsicFunction::Sum:
    {
        // Summed in the written order (a fixed order: the same bits whatever the batch).
        double sum = 0.0;
        for (const double value : arguments)
            sum += value;
        const bool isAverage = function == IntrinsicFunction::Avg;
        const double scale = isAverage ? 1.0 / static_cast<double>(count) : 1.0;
        if (!partials.empty())
        {
            for (std::size_t index = 0; index < count; ++index)
                partials[index] = scale;
        }
        return {isAverage ? sum / static_cast<double>(count) : sum};
    }
    case IntrinsicFunction::Ssq:
    case IntrinsicFunction::Rss:
    {
        // SSQ = Σ xᵢ², RSS = sqrt(Σ xᵢ²); ∂RSS/∂xᵢ = xᵢ/RSS, which does not exist at 0 (0/0 gives NaN).
        double sum = 0.0;
        for (const double value : arguments)
            sum += value * value;
        const bool isRoot = function == IntrinsicFunction::Rss;
        const double value = isRoot ? std::sqrt(sum) : sum;
        if (!partials.empty())
        {
            for (std::size_t index = 0; index < count; ++index)
                partials[index] = isRoot ? arguments[index] / value : 2.0 * arguments[index];
        }
        return {value};
    }
    default:
        return faulty(DomainFault::OutsideDomain);
    }
}

bool isVariadic(IntrinsicFunction function) noexcept
{
    return signatureOf(function).maxArguments == ANY;
}

} // namespace

const IntrinsicSignature& signatureOf(IntrinsicFunction function) noexcept
{
    return SIGNATURES[static_cast<std::size_t>(function)];
}

std::optional<IntrinsicFunction> findIntrinsicFunction(std::string_view name) noexcept
{
    for (std::size_t index = 0; index < SIGNATURES.size(); ++index)
    {
        if (SIGNATURES[index].name == name)
            return static_cast<IntrinsicFunction>(index);
    }
    return std::nullopt;
}

IntrinsicValue evaluateIntrinsic(IntrinsicFunction function, std::span<const double> arguments,
                                 std::span<double> partials) noexcept
{
    if (isVariadic(function))
        return variadic(function, arguments, partials);
    switch (arguments.size())
    {
    case 1:
        return unary(function, arguments[0], partials);
    case 2:
        return binary(function, arguments[0], arguments[1], partials);
    default:
        return ternary(function, arguments, partials);
    }
}

} // namespace oos::design
