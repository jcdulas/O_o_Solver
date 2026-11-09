#include "LdltArithmetic.hpp"

#include <utility>

namespace oos::linalg
{
namespace
{

void countSign(double value, Inertia& inertia) noexcept
{
    if (value > 0.0)
        ++inertia.positive;
    else if (value < 0.0)
        ++inertia.negative;
    else
        ++inertia.zero;
}

} // namespace

bool startsTwoByTwoBlock(const BlockDiagonalView& d, std::size_t p) noexcept
{
    return d.subdiagonal != nullptr && p + 1 < d.size && d.subdiagonal[p] != 0.0;
}

void computeProductRow(ConstDenseMatrixView l, std::size_t row, std::size_t firstColumn, const BlockDiagonalView& d,
                       std::span<double> w) noexcept
{
    const auto lAt = [&](std::size_t p) { return l(row, firstColumn + p); };
    const auto dAt = [&](std::size_t p) { return d.diagonal[p * d.diagonalStride]; };
    std::size_t p = 0;
    while (p < w.size())
    {
        if (startsTwoByTwoBlock(d, p) && p + 1 < w.size())
        {
            const double e = d.subdiagonal[p];
            w[p] = lAt(p) * dAt(p) + lAt(p + 1) * e;
            w[p + 1] = lAt(p) * e + lAt(p + 1) * dAt(p + 1);
            p += 2;
        }
        else
        {
            w[p] = lAt(p) * dAt(p);
            p += 1;
        }
    }
}

TwoByTwoInverse invertTwoByTwo(double a, double b, double c) noexcept
{
    const double d11 = c / b;
    const double d22 = a / b;
    const double t = 1.0 / (d11 * d22 - 1.0);
    return {d11, d22, t / b};
}

void swapSymmetric(DenseMatrixView a, std::size_t s, std::size_t t) noexcept
{
    const std::size_t n = a.rows;
    for (std::size_t j = 0; j < s; ++j)
        std::swap(a(s, j), a(t, j));
    std::swap(a(s, s), a(t, t));
    for (std::size_t i = s + 1; i < t; ++i)
        std::swap(a(i, s), a(t, i));
    for (std::size_t i = t + 1; i < n; ++i)
        std::swap(a(i, s), a(i, t));
}

Inertia inertiaOf(const BlockDiagonalView& d) noexcept
{
    Inertia inertia;
    std::size_t p = 0;
    while (p < d.size)
    {
        const double a = d.diagonal[p * d.diagonalStride];
        if (startsTwoByTwoBlock(d, p))
        {
            const double b = d.subdiagonal[p];
            const double c = d.diagonal[(p + 1) * d.diagonalStride];
            const double determinant = a * c - b * b;
            if (determinant < 0.0)
            {
                ++inertia.positive;
                ++inertia.negative;
            }
            else if (determinant > 0.0)
            {
                countSign(a, inertia);
                countSign(a, inertia);
            }
            else
            {
                ++inertia.zero;
                countSign(a + c, inertia);
            }
            p += 2;
        }
        else
        {
            countSign(a, inertia);
            p += 1;
        }
    }
    return inertia;
}

} // namespace oos::linalg
