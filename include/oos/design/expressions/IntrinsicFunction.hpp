#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace oos::design
{

// The intrinsic functions of a DesignEquation equation: exactly the set of functions of the DesignEquation entry of the
// input format. A function outside that set (SIGN, DBLE, INT, ...) is not part of the language and is refused by the
// program reader, so that a deck never relies on a function the format does not define.
enum class IntrinsicFunction : std::uint8_t
{
    Abs,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atan2,
    Atanh,
    Atanh2,
    Avg,
    Cos,
    Cosh,
    Db,
    Dba,
    Dim,
    Exp,
    Invdb,
    Invdba,
    Log,
    Log10,
    Logx,
    Max,
    Min,
    Mod,
    Pi,
    Rss,
    Sin,
    Sinh,
    Sqrt,
    Ssq,
    Sum,
    Tan,
    Tanh
};

// The number of intrinsic functions (the enum's values are 0 .. INTRINSIC_FUNCTION_COUNT - 1).
inline constexpr std::size_t INTRINSIC_FUNCTION_COUNT = 33;

// DesignEquation: AVG, MAX, MIN, RSS, SSQ and SUM are limited to fewer than 97 arguments.
inline constexpr int VARIADIC_MAX_ARGUMENTS = 96;

// What the format defines of a function: its name, how many arguments it takes, and where it is defined (for
// messages).
struct IntrinsicSignature
{
    std::string_view name; // as a deck writes it, upper case
    int minArguments = 1;
    int maxArguments = 1;
    std::string_view domain; // the domain of definition, in words, "x >= 0"; "all reals" when there is no limit
};

[[nodiscard]] const IntrinsicSignature& signatureOf(IntrinsicFunction function) noexcept;

// The function of a name (upper case, as the contract writes it); nothing for a name outside the set.
[[nodiscard]] std::optional<IntrinsicFunction> findIntrinsicFunction(std::string_view name) noexcept;

// Why a function or an operator could not give a value at a point.
enum class DomainFault : std::uint8_t
{
    None,
    OutsideDomain,       // an argument outside the domain of definition (SQRT of a negative, ACOS of 1.5, ...)
    DivisionByZero,      // a quotient by zero (the operator /, MOD with y = 0, 0 ** -1, ...)
    NotFinite,           // the value overflows or is not a number although the arguments are in the domain
    DerivativeNotFinite, // the value exists but a partial derivative is infinite (SQRT at 0, ACOS at 1, ...)
    IntegerOverflow      // integer arithmetic (between integer constants) beyond the exactly representable integers
};

// The value of a function at one point and its fault (None when the value is valid).
struct IntrinsicValue
{
    double value = 0.0;
    DomainFault fault = DomainFault::None;
};

// Evaluates `function` at `arguments` (their count within the signature's bounds: the program reader checks it). When
// `partials` is not empty (same size as `arguments`), it receives the partial derivative of the function with
// respect to each argument; a partial that does not exist (infinite at the edge of the domain) is left infinite or NaN
// and the caller decides whether it matters (it does only for an argument that depends on a free variable).
//
// The definitions, and the derivative conventions at the points where a function is not differentiable (ABS at 0,
// ties of MAX and MIN, the jumps of MOD), are stated in the source next to each function.
[[nodiscard]] IntrinsicValue evaluateIntrinsic(IntrinsicFunction function, std::span<const double> arguments,
                                               std::span<double> partials) noexcept;

} // namespace oos::design
