#include "oos/elements/shells/shellQuad4Drilling/ShellQuad4DrillingWarpOffsets.hpp"

#include <array>

namespace oos::elements
{

namespace
{

// The local DOFs of corner j, node by node.
constexpr std::size_t U = 0;
constexpr std::size_t V = 1;
constexpr std::size_t THETA_X = 3;
constexpr std::size_t THETA_Y = 4;

[[nodiscard]] constexpr std::size_t local(std::size_t node, std::size_t dof)
{
    return 6 * node + dof;
}

} // namespace

std::array<double, 24> ShellQuad4DrillingWarpOffsets::projected(const std::array<double, 24>& q) const
{
    // u_P = u − h θy, v_P = v + h θx; w and the rotations pass the link unchanged.
    std::array<double, 24> onProjections = q;
    for (std::size_t i = 0; i < 4; ++i)
    {
        onProjections[local(i, U)] -= height[i] * q[local(i, THETA_Y)];
        onProjections[local(i, V)] += height[i] * q[local(i, THETA_X)];
    }
    return onProjections;
}

std::array<double, 24> ShellQuad4DrillingWarpOffsets::loadsOnGridPoints(const std::array<double, 24>& f) const
{
    // Tᵀ f: the in-plane forces carried by a link of length h make the moments h × force at the grid point,
    // M_x = h F_y, M_y = −h F_x.
    std::array<double, 24> onGridPoints = f;
    for (std::size_t i = 0; i < 4; ++i)
    {
        onGridPoints[local(i, THETA_X)] += height[i] * f[local(i, V)];
        onGridPoints[local(i, THETA_Y)] -= height[i] * f[local(i, U)];
    }
    return onGridPoints;
}

base::MatN<24> ShellQuad4DrillingWarpOffsets::matrixOnGridPoints(const base::MatN<24>& k) const
{
    // Tᵀ K T as the column operations of K T, then the same operations on the rows: T is the identity but for the two
    // terms of each corner, T[u_i, θy_i] = −h_i and T[v_i, θx_i] = h_i.
    base::MatN<24> onGridPoints = k;
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t r = 0; r < 24; ++r)
        {
            onGridPoints(r, local(i, THETA_Y)) -= height[i] * onGridPoints(r, local(i, U));
            onGridPoints(r, local(i, THETA_X)) += height[i] * onGridPoints(r, local(i, V));
        }
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t c = 0; c < 24; ++c)
        {
            onGridPoints(local(i, THETA_Y), c) -= height[i] * onGridPoints(local(i, U), c);
            onGridPoints(local(i, THETA_X), c) += height[i] * onGridPoints(local(i, V), c);
        }
    return onGridPoints;
}

} // namespace oos::elements
