#pragma once

#include "oos/input/tables/PointIndex.hpp"

#include <cstdint>

namespace oos::elements
{

// One degree of freedom of an element: a component of a grid point in that grid point's displacement system CD
// (component 1, 2, 3 the translations, 4, 5, 6 the rotations, as the input format numbers them), or the single DOF of
// a scalar point (component 0). Points are named by their row in the snapshot (NodeTable, or its scalar point table),
// so that the assembler maps a DOF to its equation without looking an ID up.
struct ElementDof
{
    input::PointKind kind = input::PointKind::Grid;
    std::uint8_t component = 1;
    std::int32_t row = 0;

    // The DOF of a component (1 to 6) of the grid point of a row.
    [[nodiscard]] static constexpr ElementDof grid(std::int32_t row, int component)
    {
        return {input::PointKind::Grid, static_cast<std::uint8_t>(component), row};
    }
    // The DOF of the scalar point of a row.
    [[nodiscard]] static constexpr ElementDof scalar(std::int32_t row) { return {input::PointKind::Scalar, 0, row}; }

    friend constexpr bool operator==(const ElementDof&, const ElementDof&) = default;
};

} // namespace oos::elements
