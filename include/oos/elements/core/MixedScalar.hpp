#pragma once

namespace oos::elements
{

// A value with two first partials and their mixed partial. Ordinary arithmetic propagates the exact product and
// quotient rules. Seeding both first partials with the same direction gives its second derivative, without a step.
struct MixedScalar
{
    double value = 0.0;
    double first = 0.0;
    double other = 0.0;
    double mixed = 0.0;

    constexpr MixedScalar() = default;
    constexpr MixedScalar(double valueIn, double firstIn = 0.0, double otherIn = 0.0, double mixedIn = 0.0)
        : value(valueIn), first(firstIn), other(otherIn), mixed(mixedIn)
    {
    }
    MixedScalar& operator+=(const MixedScalar& b)
    {
        value += b.value;
        first += b.first;
        other += b.other;
        mixed += b.mixed;
        return *this;
    }
    MixedScalar& operator-=(const MixedScalar& b)
    {
        value -= b.value;
        first -= b.first;
        other -= b.other;
        mixed -= b.mixed;
        return *this;
    }
    MixedScalar& operator*=(const MixedScalar& b)
    {
        const MixedScalar result{value * b.value, first * b.value + value * b.first,
            other * b.value + value * b.other,
            mixed * b.value + first * b.other + other * b.first + value * b.mixed};
        *this = result;
        return *this;
    }
    MixedScalar& operator/=(const MixedScalar& b)
    {
        const double quotient = value / b.value;
        const double p = (first - quotient * b.first) / b.value;
        const double q = (other - quotient * b.other) / b.value;
        const double pq = (mixed - p * b.other - q * b.first - quotient * b.mixed) / b.value;
        *this = {quotient, p, q, pq};
        return *this;
    }
};

[[nodiscard]] inline MixedScalar operator+(MixedScalar a, const MixedScalar& b) { return a += b; }
[[nodiscard]] inline MixedScalar operator-(MixedScalar a, const MixedScalar& b) { return a -= b; }
[[nodiscard]] inline MixedScalar operator*(MixedScalar a, const MixedScalar& b) { return a *= b; }
[[nodiscard]] inline MixedScalar operator/(MixedScalar a, const MixedScalar& b) { return a /= b; }
[[nodiscard]] inline MixedScalar operator-(const MixedScalar& a)
{
    return {-a.value, -a.first, -a.other, -a.mixed};
}

} // namespace oos::elements
