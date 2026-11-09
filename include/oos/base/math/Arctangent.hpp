#pragma once

#include <cmath>
#include <cstddef>

namespace oos::base
{

// The arctangent of y/x in [−π, π], the quadrant from the signs of y and x (std::atan2's conventions: atan2(±0, x) is
// ±0 for x > 0 or x = +0, ±π for x < 0 or x = −0; atan2(y, ±0) is ±π/2 for y ≠ 0), by the arithmetic of fdlibm's
// atan2 and atan (Sun Microsystems, 1993, e_atan2.c and s_atan.c: the argument reduced to |x| < 7/16 about
// atan(1/2), atan(1), atan(3/2) or by 1/x, then an odd polynomial of degree 23 whose error is below one unit in the
// last place, the reduction's constants as a high and a low part), for finite arguments.
//
// It replaces the library's atan2 where a value is computed for many rows (the angle of the plane principal values):
// about 12 ns against 22 for MinGW-w64's UCRT, inlined, and the same bits on
// every target (the library's atan2 differs between the C runtimes); it differs from the UCRT's by at most 2 units in
// the last place (on 17 % of 10⁷ random arguments of magnitudes 1e−6 to 1e6).
[[nodiscard]] inline double arctangent2(double y, double x) noexcept
{
    // atan(1/2), atan(1), atan(3/2), atan(∞), each as a high part and a low part (fdlibm s_atan.c, atanhi, atanlo).
    constexpr double ARCTANGENT_HIGH[4] = {4.63647609000806093515e-01, 7.85398163397448278999e-01,
                                           9.82793723247329054082e-01, 1.57079632679489655800e+00};
    constexpr double ARCTANGENT_LOW[4] = {2.26987774529616870924e-17, 3.06161699786838301793e-17,
                                          1.39033110312309984516e-17, 6.12323399573676603587e-17};
    // The coefficients of the odd polynomial of atan on |x| < 7/16 (fdlibm s_atan.c, aT).
    constexpr double COEFFICIENTS[11] = {
        3.33333333333329318027e-01,  -1.99999999998764832476e-01, 1.42857142725034663711e-01,
        -1.11111104054623557880e-01, 9.09088713343650656196e-02,  -7.69187620504482999495e-02,
        6.66107313738753120669e-02,  -5.83357013379057348645e-02, 4.97687799461593236017e-02,
        -3.65315727442169155270e-02, 1.62858201153657823623e-02};
    // π and π/2 rounded, and the low part of π (fdlibm e_atan2.c, pi, pi_o_2, pi_lo).
    constexpr double PI = 3.14159265358979311600e+00;
    constexpr double HALF_PI = 1.57079632679489655800e+00;
    constexpr double PI_LOW = 1.2246467991473531772e-16;
    if (y == 0.0)
        return std::signbit(x) ? std::copysign(PI, y) : y;
    if (x == 0.0)
        return std::copysign(HALF_PI, y);
    // atan(|y/x|): an |y/x| beyond the double range gives ∞, whose arctangent is π/2 by the last reduction (1/∞ = 0).
    double t = std::abs(y / x);
    int reduction = -1;
    if (t >= 0.4375)
    {
        if (t < 0.6875)
        {
            reduction = 0;
            t = (2.0 * t - 1.0) / (2.0 + t);
        }
        else if (t < 1.1875)
        {
            reduction = 1;
            t = (t - 1.0) / (t + 1.0);
        }
        else if (t < 2.4375)
        {
            reduction = 2;
            t = (t - 1.5) / (1.0 + 1.5 * t);
        }
        else
        {
            reduction = 3;
            t = -1.0 / t;
        }
    }
    const double z = t * t;
    const double w = z * z;
    // The odd and even terms of the polynomial in w apart (fdlibm's evaluation order).
    const double odd =
        z * (COEFFICIENTS[0] +
             w * (COEFFICIENTS[2] +
                  w * (COEFFICIENTS[4] + w * (COEFFICIENTS[6] + w * (COEFFICIENTS[8] + w * COEFFICIENTS[10])))));
    const double even =
        w *
        (COEFFICIENTS[1] + w * (COEFFICIENTS[3] + w * (COEFFICIENTS[5] + w * (COEFFICIENTS[7] + w * COEFFICIENTS[9]))));
    const double angle = reduction < 0
                             ? t - t * (odd + even)
                             : ARCTANGENT_HIGH[reduction] - ((t * (odd + even) - ARCTANGENT_LOW[reduction]) - t);
    return x > 0.0 ? std::copysign(angle, y) : std::copysign(PI - (angle - PI_LOW), y);
}

} // namespace oos::base
