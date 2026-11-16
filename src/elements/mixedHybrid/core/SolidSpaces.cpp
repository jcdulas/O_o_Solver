#include "oos/elements/mixedHybrid/core/SolidSpaces.hpp"

namespace oos::elements::mixedHybrid
{

std::size_t tetrahedronSpaceDimension(std::size_t n)
{
    return totalDegreeCount3D(n);
}

// Σ_{m=0}^{n} (2m + 1)(n − m + 1): the pairs (i, j) with max(i, j) = m, times the degrees k ≤ n − m.
std::size_t pyramidSpaceDimension(std::size_t n)
{
    std::size_t count = 0;
    for (std::size_t m = 0; m <= n; ++m)
        count += (2 * m + 1) * (n - m + 1);
    return count;
}

std::size_t pentahedronSpaceDimension(std::size_t n)
{
    return (n + 1) * (n + 1) * (n + 2) / 2;
}

std::size_t hexahedronSpaceDimension(std::size_t n)
{
    return (n + 1) * (n + 1) * (n + 1);
}

std::vector<std::size_t> tetrahedronLevels(std::size_t n)
{
    std::vector<std::size_t> out;
    out.reserve(tetrahedronSpaceDimension(n));
    for (std::size_t total = 0; total <= n; ++total)
        for (std::size_t i = 0; i <= total; ++i)
            for (std::size_t j = 0; j <= total - i; ++j)
                out.push_back(total);
    return out;
}

std::vector<std::size_t> pyramidLevels(std::size_t n)
{
    std::vector<std::size_t> out;
    out.reserve(pyramidSpaceDimension(n));
    for (std::size_t m = 0; m <= n; ++m)
        for (std::size_t i = 0; i <= m; ++i)
            for (std::size_t j = 0; j <= m; ++j)
            {
                if (std::max(i, j) != m)
                    continue;
                for (std::size_t k = 0; k <= n - m; ++k)
                    out.push_back(m + k);
            }
    return out;
}

std::vector<std::size_t> pentahedronLevels(std::size_t n)
{
    std::vector<std::size_t> out;
    out.reserve(pentahedronSpaceDimension(n));
    for (std::size_t total = 0; total <= n; ++total)
        for (std::size_t i = 0; i <= total; ++i)
            for (std::size_t k = 0; k <= n; ++k)
                out.push_back(std::max(total, k));
    return out;
}

std::vector<std::size_t> hexahedronLevels(std::size_t n)
{
    std::vector<std::size_t> out;
    out.reserve(hexahedronSpaceDimension(n));
    for (std::size_t i = 0; i <= n; ++i)
        for (std::size_t j = 0; j <= n; ++j)
            for (std::size_t k = 0; k <= n; ++k)
                out.push_back(std::max(std::max(i, j), k));
    return out;
}

} // namespace oos::elements::mixedHybrid
