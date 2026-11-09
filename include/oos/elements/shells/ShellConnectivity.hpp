#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"
#include "oos/input/tables/ConnectivityView.hpp"

#include <array>
#include <cstdint>

namespace oos::elements
{

// The grid point columns of a shell batch, read once per facet call: nodesRows (O_o_Reader: the rows of G1, G2 ... in
// the node table, the row null when a grid point is unknown) and nodes (their IDs, for the messages and the outputs).
struct ShellConnectivity
{
    input::ConnectivityView rows;
    input::ConnectivityView ids;
    base::ValidityView valid;

    explicit ShellConnectivity(const ElementBatchView& view);
};

// The grid points of a shell of N corners: their node table rows and IDs, their positions in basic and the rotations
// of their displacement systems.
template <std::size_t N>
struct ShellCorners
{
    std::array<std::int32_t, N> rows{};
    std::array<std::int32_t, N> ids{};
    std::array<base::Vec3, N> positions{};
    std::array<base::Mat3, N> cdFrames{};
};

// The corners of the shell of batch row `row`; an unknown grid point (field G1, G2 ...) or a grid point named twice is
// a problem.
template <std::size_t N>
[[nodiscard]] Resolution<ShellCorners<N>> resolveShellCorners(const ElementBatchView& view,
                                                              const ShellConnectivity& connectivity, std::int64_t row);

} // namespace oos::elements
