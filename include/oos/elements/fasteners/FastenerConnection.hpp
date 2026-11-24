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
// found by projection on the shells), each moving with the normal fibre of the shell of its side
// (PiercingInterpolation): the translation and the rotation of the shell read at the four corners of the square of
// the cross-section of the fastener, each on its shell element (a Fastener, a Weld of the patch forms), or at the
// point itself by the shape functions of the shell element it pierces (the other forms of a Weld, a fastener
// without corners); the rotation about the normal of the sheet from the translations around the point. The motions
// of A and B in the basic system are then
//     w = (u_A, θ_A, u_B, θ_B) = M·u,   u the DOFs of the grid points both sides are tied to (six each, in their
//     displacement systems, each grid point once: their translations and their rotations).
// A side connected directly (Weld ALIGN, the GS of a Weld ELEMID without SHIDB) is that grid point itself. A side
// that pierces a shell of a formulation with unknowns besides its grid points reads the displacement and the rotation
// of the shells over the same square, over their own DOFs, instead (resolveSideMotion,
// FormulationModel::shellPointMotion).
//
// The auxiliary points GA and GB of a fastener are eliminated at element level by default (WeldProp MSET = OFF, the
// default: the constraint equations are condensed into the element, whose matrix couples the grid points of the
// pierced elements directly): the element matrix is K = Mᵀ·k·M, k the 12 × 12 stiffness of the fastener over w. This is
// the linear elimination of the m set done inside the element (exact, a change of basis: never a flexibility form or a
// Lagrange multiplier). When GA or GB is a grid point of the model, assembly::FastenerConstraints also makes its six
// DOFs dependent on the grid points of the pierced element by the same rows (so that it is recovered and anything else
// attached to it acts on the pierced element).
//
// The sizes are those of a few elements (the grid points of the elements under the four corners of a side, or of
// the patch of its point): std::vector is the right container.
struct FastenerConnection
{
    base::Vec3 pointA;
    base::Vec3 pointB;
    // The DOFs both sides read, each once: six per grid point a side is tied to, or the DOFs of the shells of a
    // formulation with entities under the footprint of a side.
    std::vector<ElementDof> dofList;
    std::vector<double> map; // M: 12 rows of dofList.size(), row-major

    [[nodiscard]] std::size_t dofCount() const noexcept { return dofList.size(); }
    // The DOFs of u, in the order of dofList.
    [[nodiscard]] std::vector<ElementDof> dofs() const { return dofList; }
    // K = Mᵀ·k·M over dofs(), row-major and exactly symmetric.
    [[nodiscard]] std::vector<double> condense(const base::MatN<12>& k) const;
    // w = M·u from the displacements of the model.
    [[nodiscard]] std::array<double, 12> pointMotions(const NodalDisplacements& displacements) const;
};

// The connection of the fastener of batch row `row` from the fastener piercing table of the snapshot (the piercing
// points of O_o_Model's fastenerPiercings and the shape functions of the elements they pierce). A side without a row
// in the table is the grid point `directGrids[side]` of the card itself (Weld ALIGN: GA and GB connect directly, and
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

// The axes of a fastener whose x-axis runs from A to B, when the property gives no system (FastenerProp MCID = −1, Weld
// MCID = −1): y is the basic axis least aligned with x, made perpendicular to it, z = x × y (the first of the basic
// axes X, Y, Z wins a tie): the input format places the element y-axis perpendicular to the element axis, towards the
// closest basic axis. The columns of the result are x, y, z in the basic system. Two components of x that differ by
// less than FASTENER_AXIS_TIE are a tie: a fastener along a basic axis has the two others at round-off of zero, and
// the y-axis must not depend on which of the two round-offs is the smaller.
inline constexpr double FASTENER_AXIS_TIE = 1.0e-6;
[[nodiscard]] base::Mat3 fastenerAxes(base::Vec3 pointA, base::Vec3 pointB);

} // namespace oos::elements
