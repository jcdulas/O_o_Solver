#pragma once

#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementDof.hpp"
#include "oos/elements/core/ElementFamily.hpp"
#include "oos/elements/core/ElementProblem.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace oos::elements
{

// How a Fastener or Weld reaches the model: its two piercing points A and B (GA and GB, given on the card or
// found by projection on the shells), each moving with the reference point of an RigidInterpolation interpolation
// element whose independent points are the grid points of the shell element it pierces, weighted by the shape functions
// of that element at the point (PiercingInterpolation), so that the motions of A and B in the basic system are
//     w = (u_A, θ_A, u_B, θ_B) = M·u,   u the DOFs of the grid points of both pierced elements (six each, in their
//     displacement systems, each grid point once; the interpolation reads their translations only).
// A side connected directly (CWELD ALIGN, the GS of a Weld ELEMID without SHIDB) is that grid point itself.
//
// The auxiliary points GA and GB of a fastener are eliminated at element level by default (PWELD MSET = OFF, the
// default: the constraint equations are condensed into the element, whose matrix couples the grid points of the
// pierced elements directly): the element matrix is K = Mᵀ·k·M, k the 12 × 12 stiffness of the fastener over w. This is
// the linear elimination of the m set done inside the element (exact, a change of basis: never a flexibility form or a
// Lagrange multiplier). When GA or GB is a grid point of the model, assembly::FastenerConstraints also makes its six
// DOFs dependent on the grid points of the pierced element by the same rows (so that it is recovered and anything else
// attached to it acts on the pierced element).
//
// The sizes are those of one element (at most 8 grid points of the pierced element per side): std::vector is the
// right container.
struct FastenerConnection
{
    base::Vec3 pointA;
    base::Vec3 pointB;
    std::vector<std::int32_t> gridRows; // the grid points of the elements both sides pierce, each once
    std::vector<double> map;            // M: 12 rows of 6·gridRows.size(), row-major

    [[nodiscard]] std::size_t dofCount() const noexcept { return 6 * gridRows.size(); }
    // The DOFs of u: six per grid point of a pierced element, in the order of gridRows.
    [[nodiscard]] std::vector<ElementDof> dofs() const;
    // K = Mᵀ·k·M over dofs(), row-major and exactly symmetric.
    [[nodiscard]] std::vector<double> condense(const base::MatN<12>& k) const;
    // w = M·u from the displacements of the model.
    [[nodiscard]] std::array<double, 12> pointMotions(const NodalDisplacements& displacements) const;
};

// The connection of the fastener of batch row `row` from the fastener piercing table of the snapshot (the piercing
// points of O_o_Reader's fastenerPiercings and the shape functions of the elements they pierce). A side without a row
// in the table is the grid point `directGrids[side]` of the card itself (CWELD ALIGN: GA and GB connect directly, and
// no table is read; the GS end of a point-to-patch Weld), or a problem when that is 0.
//
// Problems (field PID for a missing table; for a side the reader did not place, the field and the reason it gives:
// its tolerance and the nearest element; IDA or IDB otherwise): no fastenerPiercings table, no piercing point for a
// side and no direct grid point, a grid point of the pierced element the model does not define, shape functions that
// are not an interpolation (interpolatePiercing), coincident piercing points.
[[nodiscard]] Resolution<FastenerConnection> resolveFastenerConnection(const ElementBatchView& view, std::int64_t row,
                                                                       std::array<std::int32_t, 2> directGrids);

// The piercing points A and B closer than this (model length units) give no fastener axis.
inline constexpr double FASTENER_COINCIDENT_DISTANCE = 1.0e-8;

// The axes of a fastener whose x-axis runs from A to B, when the property gives no system (PFAST MCID = −1, Weld
// MCID = −1): y is the basic axis least aligned with x, made perpendicular to it, z = x × y (the first of the basic
// axes X, Y, Z wins a tie): the input format places the element y-axis perpendicular to the element axis, towards the
// closest basic axis. The columns of the result are x, y, z in the basic system.
[[nodiscard]] base::Mat3 fastenerAxes(base::Vec3 pointA, base::Vec3 pointB);

} // namespace oos::elements
