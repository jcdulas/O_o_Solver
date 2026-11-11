#pragma once

#include <cmath>

namespace oos::elements
{

// A scalar and its first two derivatives along one input direction. The ordinary shell kernel evaluates all
// three together using the product, quotient and square-root chain rules; no perturbation or step size is used.
// `second` is the second derivative itself, rather than the coefficient of the quadratic Taylor term.
struct SecondOrderScalar
{
    double value = 0.0;
    double first = 0.0;
    double second = 0.0;

    constexpr SecondOrderScalar() = default;
    constexpr SecondOrderScalar(double valueIn, double firstIn = 0.0, double secondIn = 0.0)
        : value(valueIn), first(firstIn), second(secondIn)
    {
    }
    [[nodiscard]] constexpr double real() const noexcept { return value; }
    friend bool operator==(const SecondOrderScalar&, const SecondOrderScalar&) = default;

    SecondOrderScalar& operator+=(const SecondOrderScalar& other)
    {
        value += other.value;
        first += other.first;
        second += other.second;
        return *this;
    }
    SecondOrderScalar& operator-=(const SecondOrderScalar& other)
    {
        value -= other.value;
        first -= other.first;
        second -= other.second;
        return *this;
    }
    SecondOrderScalar& operator*=(const SecondOrderScalar& other)
    {
        const SecondOrderScalar result{value * other.value, first * other.value + value * other.first,
            second * other.value + 2.0 * first * other.first + value * other.second};
        *this = result;
        return *this;
    }
    SecondOrderScalar& operator/=(const SecondOrderScalar& other)
    {
        const double quotient = value / other.value;
        const double slope = (first - quotient * other.first) / other.value;
        const double curvature = (second - quotient * other.second - 2.0 * slope * other.first) / other.value;
        *this = {quotient, slope, curvature};
        return *this;
    }
};

[[nodiscard]] inline SecondOrderScalar operator+(SecondOrderScalar a, const SecondOrderScalar& b) { return a += b; }
[[nodiscard]] inline SecondOrderScalar operator-(SecondOrderScalar a, const SecondOrderScalar& b) { return a -= b; }
[[nodiscard]] inline SecondOrderScalar operator*(SecondOrderScalar a, const SecondOrderScalar& b) { return a *= b; }
[[nodiscard]] inline SecondOrderScalar operator/(SecondOrderScalar a, const SecondOrderScalar& b) { return a /= b; }
[[nodiscard]] inline SecondOrderScalar operator-(const SecondOrderScalar& a)
{
    return {-a.value, -a.first, -a.second};
}

// Found through argument-dependent lookup by ShellScalar::squareRoot.
[[nodiscard]] inline SecondOrderScalar sqrt(const SecondOrderScalar& a)
{
    const double root = std::sqrt(a.value);
    if (root == 0.0 && a.first == 0.0 && a.second == 0.0)
        return {};
    const double slope = a.first / (2.0 * root);
    return {root, slope, (a.second - 2.0 * slope * slope) / (2.0 * root)};
}

} // namespace oos::elements
