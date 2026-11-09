#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/input/tables/FastenerPiercingTable.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace oos::elements
{

// One grid point of the shell element a fastener side pierces (the pierced element): its position and displacement
// system in the basic system, and the shape function of the element at the piercing point.
struct PiercedGrid
{
    base::Vec3 position;
    base::Mat3 frame; // the displacement system of the grid point, columns in the basic system
    double shapeValue = 0.0;
};

// The map from the DOFs of the grid points of the element one side pierces (six per grid point, in its displacement
// system, grid after grid) to the motion of the piercing point P in the basic system, (u_P, θ_P) = M·u_grids: 6 rows
// of 6·gridCount, row-major.
struct PiercingMap
{
    std::size_t gridCount = 0;
    std::vector<double> values;

    [[nodiscard]] double at(std::size_t row, std::size_t column) const { return values[row * 6 * gridCount + column]; }
};

// The shape functions of the pierced element at the piercing point sum to 1; a larger departure means the table does
// not hold an interpolation.
inline constexpr double PIERCING_WEIGHT_SUM_TOLERANCE = 1.0e-8;

// The smallest weight of a grid point of the pierced element, as a fraction of its largest shape function. A piercing
// point on an edge or at a grid point of the element gives zero shape functions to the other grid points, and the
// rotation about that edge (or every rotation) would be determined by nothing; with this floor those grid points
// determine it, and the result is within about this fraction of the limit of the interpolation as the point
// approaches the edge. Negative shape functions (the corners of a quadratic element) are raised to it too, since a
// least-squares weight must not be negative. A choice of this implementation.
inline constexpr double PIERCING_WEIGHT_FLOOR = 1.0e-3;

// The smallest pivot of the scaled normal matrix of the interpolation, relative to its largest diagonal term: below
// it the grid points of the pierced element are collinear or coincide, and leave a rotation of the piercing point
// undetermined.
inline constexpr double PIERCING_PIVOT_TOLERANCE = 1.0e-10;

// The interpolation of the piercing point P of one fastener side from the grid points of the shell element it
// pierces: P moves with the reference point of an RigidInterpolation interpolation element (an element of NASTRAN-95,
// NASA's open-source release, among its rigid elements in mis/criggp.f; implemented for the RigidInterpolation card by
// assembly/constraints/RigidElementEquations, which the elements layer cannot call) whose independent points are the
// grid points of that element with their translations (components 123), weighted by the shape functions N_i of the
// element at P. The motion of P is the weighted least-squares rigid motion of the grid translations:
//
//     (u_P, θ_P) = argmin Σ w_i·|R_i·u_i − (u_P + θ_P × r_i)|²,   r_i = x_i − x_P,   w_i = max(N_i, f·max_j N_j),
//
// R_i the rotation from the displacement system of grid i to the basic system, f = PIERCING_WEIGHT_FLOOR. With the rows
// B_i = [I, −[r_i]×] the normal equations give (u_P, θ_P) = A⁻¹·Σ w_i·B_iᵀ·R_i·u_i, A = Σ w_i·B_iᵀ·B_i, solved with
// the rotations scaled by Lc, the mean distance of the grid points to P (so that A is of order 1 and the result does
// not depend on the length unit, as the RigidInterpolation of RigidElementEquations). For a linear element whose shape
// functions are all above the floor, Σ N_i·r_i = 0 (the isoparametric map gives x_P = Σ N_i·x_i), so u_P = Σ
// N_i·R_i·u_i: the shape-function interpolation of the translations, and θ_P the weighted mean rotation of the pierced
// element. The shell rotations take no part (columns zero). The map is exact for every rigid motion of the pierced
// element.
//
// Returns the message of a problem instead: no grid point, shape functions that do not sum to 1, grid points all on
// P, or collinear grid points (a rotation undetermined).
[[nodiscard]] std::variant<PiercingMap, std::string> interpolatePiercing(base::Vec3 piercingPoint,
                                                                         std::span<const PiercedGrid> grids);

// The map of a side connected directly to a grid point of the model (CWELD ALIGN): (u_P, θ_P) = (R·u, R·θ).
[[nodiscard]] PiercingMap directGridMap(const base::Mat3& gridFrame);

// A grid point of the model as the interpolation of a side reads it: its row (in the node table and the DOF map),
// its position and its displacement system in the basic system.
struct SideGrid
{
    std::int32_t row = 0;
    base::Vec3 position;
    base::Mat3 frame;
};

// The lookup of a grid point by ID (nothing when the model does not define it). Each caller gives its own: the element
// families read the snapshot, the assembly and the reduction the DOF map; both give the same rows and geometry.
using SideGridLookup = std::function<std::optional<SideGrid>(std::int32_t gridId)>;

// One side of a fastener resolved from the fastener piercing table: its piercing point, the rows of the grid points
// of the element it pierces (in the order of the table) and the map of interpolatePiercing over them.
struct SideInterpolation
{
    base::Vec3 point;
    std::vector<std::int32_t> gridRows;
    PiercingMap map;
};

// The interpolation of the side on rows `rows` of the table (the one row of its piercing point). The element
// families, assembly::FastenerConstraints and reduction::FastenerPiercing all call this, so that they tie a side to the
// element it pierces by the same map. Returns the message of a problem instead: no row, a grid point the lookup does
// not find, the problems of interpolatePiercing.
[[nodiscard]] std::variant<SideInterpolation, std::string>
interpolateSide(const input::FastenerPiercingTable& table, input::PiercingRows rows, const SideGridLookup& findGrid);

} // namespace oos::elements
