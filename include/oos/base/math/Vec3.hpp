#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace oos::base
{

// A 3-vector of doubles: positions, directions, forces and moments in a frame the variable name states. A plain value
// type; every operation has a fixed order of evaluation.
struct Vec3
{
    std::array<double, 3> components{0.0, 0.0, 0.0};

    constexpr Vec3() = default;
    constexpr Vec3(double x, double y, double z) : components{x, y, z} {}

    [[nodiscard]] constexpr double operator[](std::size_t i) const { return components[i]; }
    [[nodiscard]] constexpr double& operator[](std::size_t i) { return components[i]; }
    [[nodiscard]] constexpr double x() const { return components[0]; }
    [[nodiscard]] constexpr double y() const { return components[1]; }
    [[nodiscard]] constexpr double z() const { return components[2]; }

    friend constexpr Vec3 operator+(Vec3 a, Vec3 b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
    friend constexpr Vec3 operator-(Vec3 a, Vec3 b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
    friend constexpr Vec3 operator-(Vec3 a) { return {-a[0], -a[1], -a[2]}; }
    friend constexpr Vec3 operator*(double s, Vec3 a) { return {s * a[0], s * a[1], s * a[2]}; }
    friend constexpr Vec3 operator*(Vec3 a, double s) { return s * a; }
    friend constexpr Vec3 operator/(Vec3 a, double s) { return {a[0] / s, a[1] / s, a[2] / s}; }
    friend constexpr bool operator==(const Vec3&, const Vec3&) = default;
};

[[nodiscard]] constexpr double dot(Vec3 a, Vec3 b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

[[nodiscard]] constexpr Vec3 cross(Vec3 a, Vec3 b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

[[nodiscard]] inline double norm(Vec3 a)
{
    return std::sqrt(dot(a, a));
}

// The unit vector along a; the caller makes sure a is not zero (an element with coincident nodes is refused earlier).
[[nodiscard]] inline Vec3 normalized(Vec3 a)
{
    return a / norm(a);
}

} // namespace oos::base
