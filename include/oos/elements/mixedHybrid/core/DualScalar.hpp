#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>

namespace oos::elements::mixedHybrid
{

// Scalars for forward-mode differentiation. Every geometric quantity of an element of the mixed hybrid formulation
// (tangents, normal, traces, interior extension) is written once, as a template on its scalar type, and evaluated with
// `double` for its value or with `Dual<T>` for its first derivatives along two directions. Nesting `Dual` gives the
// higher derivatives: `Dual<Dual<double>>` carries the second derivatives. The derivatives are exact (the chain rule
// on each operation), with no step size.
//
// A scalar type T provides +, −, ·, / and unary −, and the free functions below: constantOf<T>(x), valueOf(a),
// sqrt(a), sin(a), cos(a), zeroValue(a) and scale(a, s). They are overloaded for double here, and found for Dual by
// argument-dependent lookup.
//
// Thread-safety: plain value types and pure functions.

// ---- double -----------------------------------------------------------------------------------------------------

// The constant x in the scalar type T (its derivative parts zero).
template <class T>
[[nodiscard]] constexpr T constantOf(double x)
{
    return T(x);
}

// The value of a scalar with all its derivative parts dropped.
[[nodiscard]] constexpr double valueOf(double x)
{
    return x;
}

[[nodiscard]] inline double sqrt(double x)
{
    return std::sqrt(x);
}
[[nodiscard]] inline double sin(double x)
{
    return std::sin(x);
}
[[nodiscard]] inline double cos(double x)
{
    return std::cos(x);
}

// The same derivative parts with the value set to zero: used where a value is zero by construction and only
// round-off would remain. A double has no derivative part.
[[nodiscard]] constexpr double zeroValue(double)
{
    return 0.0;
}

// a·s, s a plain number.
[[nodiscard]] constexpr double scale(double a, double s)
{
    return a * s;
}

// ---- Dual -------------------------------------------------------------------------------------------------------

// A value and its derivatives along two directions (d/dξ and d/dη, or any two directions). T is double or another
// Dual. The rules are those of the derivative of a sum, a product and a quotient, applied to each direction.
template <class T>
struct Dual
{
    T value{};
    std::array<T, 2> derivatives{};

    constexpr Dual() = default;
    // The constant c: both derivatives zero.
    constexpr explicit Dual(double c) : value(constantOf<T>(c)) {}
    constexpr Dual(T valueIn, T derivative0, T derivative1) : value(valueIn), derivatives{derivative0, derivative1} {}

    // The independent variable number `axis` (0 or 1) at the value `valueIn`: its derivative along that axis is 1.
    [[nodiscard]] static constexpr Dual variable(T valueIn, std::size_t axis)
    {
        Dual x(valueIn, constantOf<T>(0.0), constantOf<T>(0.0));
        x.derivatives[axis] = constantOf<T>(1.0);
        return x;
    }
};

template <class T>
[[nodiscard]] constexpr Dual<T> operator+(const Dual<T>& a, const Dual<T>& b)
{
    return {a.value + b.value, a.derivatives[0] + b.derivatives[0], a.derivatives[1] + b.derivatives[1]};
}

template <class T>
[[nodiscard]] constexpr Dual<T> operator-(const Dual<T>& a, const Dual<T>& b)
{
    return {a.value - b.value, a.derivatives[0] - b.derivatives[0], a.derivatives[1] - b.derivatives[1]};
}

// (a b)' = a' b + a b'.
template <class T>
[[nodiscard]] constexpr Dual<T> operator*(const Dual<T>& a, const Dual<T>& b)
{
    return {a.value * b.value, a.derivatives[0] * b.value + a.value * b.derivatives[0],
            a.derivatives[1] * b.value + a.value * b.derivatives[1]};
}

// (a / b)' = (a' − q b') / b with q = a / b.
template <class T>
[[nodiscard]] constexpr Dual<T> operator/(const Dual<T>& a, const Dual<T>& b)
{
    const T inverse = constantOf<T>(1.0) / b.value;
    const T quotient = a.value * inverse;
    return {quotient, (a.derivatives[0] - quotient * b.derivatives[0]) * inverse,
            (a.derivatives[1] - quotient * b.derivatives[1]) * inverse};
}

template <class T>
[[nodiscard]] constexpr Dual<T> operator-(const Dual<T>& a)
{
    return {-a.value, -a.derivatives[0], -a.derivatives[1]};
}

template <class T>
[[nodiscard]] constexpr double valueOf(const Dual<T>& a)
{
    return valueOf(a.value);
}

// (√a)' = a' / (2 √a).
template <class T>
[[nodiscard]] Dual<T> sqrt(const Dual<T>& a)
{
    const T root = sqrt(a.value);
    const T half = constantOf<T>(0.5) / root;
    return {root, a.derivatives[0] * half, a.derivatives[1] * half};
}

// (sin a)' = a' cos a.
template <class T>
[[nodiscard]] Dual<T> sin(const Dual<T>& a)
{
    const T s = sin(a.value);
    const T c = cos(a.value);
    return {s, a.derivatives[0] * c, a.derivatives[1] * c};
}

// (cos a)' = −a' sin a.
template <class T>
[[nodiscard]] Dual<T> cos(const Dual<T>& a)
{
    const T s = sin(a.value);
    const T c = cos(a.value);
    return {c, -(a.derivatives[0] * s), -(a.derivatives[1] * s)};
}

template <class T>
[[nodiscard]] constexpr Dual<T> zeroValue(const Dual<T>& a)
{
    return {zeroValue(a.value), a.derivatives[0], a.derivatives[1]};
}

// a·s for a plain number s: every part scaled.
template <class T>
[[nodiscard]] constexpr Dual<T> scale(const Dual<T>& a, double s)
{
    return {scale(a.value, s), scale(a.derivatives[0], s), scale(a.derivatives[1], s)};
}

// ---- the concept ------------------------------------------------------------------------------------------------

// What the generic code of the formulation asks of its scalar type: double and every nesting of Dual.
template <class T>
concept Scalar = std::copyable<T> && requires(T a, T b, double x) {
    { a + b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
    { -a } -> std::convertible_to<T>;
    { constantOf<T>(x) } -> std::same_as<T>;
    { valueOf(a) } -> std::same_as<double>;
    { sqrt(a) } -> std::convertible_to<T>;
    { sin(a) } -> std::convertible_to<T>;
    { cos(a) } -> std::convertible_to<T>;
    { zeroValue(a) } -> std::convertible_to<T>;
    { scale(a, x) } -> std::convertible_to<T>;
};

// ---- 3-vectors --------------------------------------------------------------------------------------------------

template <class T>
using Vec3Of = std::array<T, 3>;

template <Scalar T>
[[nodiscard]] Vec3Of<T> add3(const Vec3Of<T>& a, const Vec3Of<T>& b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

template <Scalar T>
[[nodiscard]] Vec3Of<T> sub3(const Vec3Of<T>& a, const Vec3Of<T>& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

// a·s, s a scalar of the same type.
template <Scalar T>
[[nodiscard]] Vec3Of<T> scl3(const Vec3Of<T>& a, const T& s)
{
    return {a[0] * s, a[1] * s, a[2] * s};
}

template <Scalar T>
[[nodiscard]] T dot3(const Vec3Of<T>& a, const Vec3Of<T>& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <Scalar T>
[[nodiscard]] Vec3Of<T> cross3(const Vec3Of<T>& a, const Vec3Of<T>& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

template <Scalar T>
[[nodiscard]] T norm3(const Vec3Of<T>& a)
{
    return sqrt(dot3(a, a));
}

// a / |a|; the caller makes sure a is not zero.
template <Scalar T>
[[nodiscard]] Vec3Of<T> unit3(const Vec3Of<T>& a)
{
    const T inverseNorm = constantOf<T>(1.0) / norm3(a);
    return scl3(a, inverseNorm);
}

// A vector of plain numbers as constants of the scalar type T.
template <Scalar T>
[[nodiscard]] Vec3Of<T> lift3(const std::array<double, 3>& a)
{
    return {constantOf<T>(a[0]), constantOf<T>(a[1]), constantOf<T>(a[2])};
}

// m·a, m a 3 × 3 matrix of plain numbers given by rows.
template <Scalar T>
[[nodiscard]] Vec3Of<T> mat3(const std::array<std::array<double, 3>, 3>& m, const Vec3Of<T>& a)
{
    const auto row = [&](std::size_t i) { return scale(a[0], m[i][0]) + scale(a[1], m[i][1]) + scale(a[2], m[i][2]); };
    return {row(0), row(1), row(2)};
}

} // namespace oos::elements::mixedHybrid
