#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementDof.hpp"
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

class FormulationModel;

// One grid point of the shell element a fastener side pierces (the pierced element): its position and displacement
// system in the basic system, and the shape function of the element at the piercing point.
struct PiercedGrid
{
    base::Vec3 position;
    base::Mat3 frame; // the displacement system of the grid point, columns in the basic system
    double shapeValue = 0.0;
};

// The map from the DOFs of the grid points one side is tied to (six per grid point, in its displacement system, grid
// after grid: their translations and their rotations) to the motion of the piercing point P in the basic system,
// (u_P, θ_P) = M·u_grids: 6 rows of 6·gridCount, row-major.
struct PiercingMap
{
    std::size_t gridCount = 0;
    std::vector<double> values;

    [[nodiscard]] double at(std::size_t row, std::size_t column) const { return values[row * 6 * gridCount + column]; }
};

// The shape functions of the pierced element at the piercing point sum to 1, and so do the weights of the auxiliary
// points of a footprint and the shape functions of each of them; a larger departure means the table does not hold an
// interpolation (interpolatePiercing and interpolateFootprint then return a problem).
inline constexpr double PIERCING_WEIGHT_SUM_TOLERANCE = 1.0e-8;

// The smallest weight of a grid point in the rigid fit that gives the rotation of the piercing point about the normal
// of the sheet, as a fraction of the largest shape function. A piercing point on an edge or at a grid point of the
// element gives zero shape functions to the other grid points, and that rotation would be determined by nothing;
// with this floor those grid points determine it, and the result is within about this fraction of the limit of the
// fit as the point approaches the edge. Negative shape functions (the corners of a quadratic element) are raised
// to it too, since a least-squares weight must not be negative. A choice of this implementation.
inline constexpr double PIERCING_WEIGHT_FLOOR = 1.0e-3;

// The smallest pivot of the scaled normal matrix of the interpolation, relative to its largest diagonal term: below
// it the grid points of the pierced element are collinear or coincide, and leave a rotation of the piercing point
// undetermined.
inline constexpr double PIERCING_PIVOT_TOLERANCE = 1.0e-10;

// The motion of the piercing point P of a fastener side, in the ClassicFormulation: P moves with the normal fibre
// of the shell under it, as the kinematics of a shell with transverse shear give it,
//
//     u_P = ū + θ_P × (P − X̄),      θ_P = (I − n·nᵀ)·β + n·nᵀ·θ_fit,
//
// with, at the point X̄ of the sheet under P (n its normal there):
//   - ū the translation of the sheet, by the shape functions of the element over the translations of its grid
//     points;
//   - β the rotation of the shell, by the same shape functions over the rotations of its grid points: the rotation
//     of its normal fibre, which a fastener through the thickness follows. Only its part about the axes of the
//     plane of the sheet is read: the rotation of a grid point about the normal is not a motion of the shell (its
//     stiffness is the drilling stabilisation);
//   - θ_fit the rotation of the least-squares rigid motion of the translations of the grid points around P, read
//     about the normal only: the rotation of the sheet in its plane, ½(∂v/∂x − ∂u/∂y).
// The map is exact for every rigid motion of the shells.
//
// The rotation about the axes of the plane is not taken from the translations (the slope of the sheet between its
// grid points), as an interpolation element over translations does: an element with transverse shear bends by the
// rotations of its grid points, and the slope of its deflection over the width of one element is not the rotation
// of the sheet at a grid point of it. On the flange of a stiffener one element wide, bolted to a skin along one of
// its edges and lifted by the other, the slope across the flange tied the lifted edge to the skin through the arm
// and the rotational springs of each fastener: the edge lifted by 0.62 of what it does with the same bolts written
// as springs between the grid points (0.21 with bolts a hundred times stiffer); with the rotations of the grid
// points it lifts by 0.97 and 0.93 of it (wasm/test/validation/fasteners.test.mjs, section 13).
//
// interpolatePiercing is the side tied to the grid points of the shell element it pierces (a Weld of a point form, a
// fastener without corners): `grids` are those of the patch of the point (input::FastenerPiercingTable::patchGrids:
// the pierced element first, then, for a point on a grid point or on a side of it, the other grid points of the shell
// elements that hold it and continue the sheet, with a shape function of zero). X̄ = Σ N_i·x_i, ū and β are read
// with the shape functions N_i of the pierced element at P; θ_fit is that of
//
//     argmin Σ w_i·|R_i·u_i − (u + θ × r_i)|²,   r_i = x_i − x_P,   w_i = max(N_i, f·max_j N_j),
//
// R_i the rotation from the displacement system of grid i to the basic system, f = PIERCING_WEIGHT_FLOOR (the
// least-squares fit of a RigidInterpolation interpolation element over translations: an element of NASTRAN-95, NASA's
// open-source release, among its rigid elements in mis/criggp.f; solved with the rotations scaled by Lc, the mean
// distance of the grid points to P, so that the result does not depend on the length unit). The grid points of the
// patch beyond the pierced element take part in this fit only: on a grid point of the mesh the rotation of the
// sheet in its plane is then that of the elements all around it, not of one of them.
//
// Returns the message of a problem instead: no grid point, shape functions that do not sum to 1, grid points all on
// P, or collinear grid points (a rotation undetermined).
[[nodiscard]] std::variant<PiercingMap, std::string> interpolatePiercing(base::Vec3 piercingPoint,
                                                                         std::span<const PiercedGrid> grids);

// A grid point of the model as the interpolation of a side reads it: its row (in the node table and the DOF map),
// its position and its displacement system in the basic system.
struct SideGrid
{
    std::int32_t row = 0;
    base::Vec3 position;
    base::Mat3 frame;
};

// One auxiliary point of a fastener side: a corner of the square that stands for its cross-section on the shells
// (input/tables/PiercingFootprint.hpp), on one shell element: its weight, the grid points of that element (their
// indices among the grid points given with the auxiliary points) and the shape functions of the element at the
// point.
struct AuxiliaryPoint
{
    double weight = 0.0;
    std::vector<std::size_t> gridIndices;
    std::vector<double> shapeValues;
};

// The motion of the piercing point P of a fastener side tied over its cross-section (the formulas of
// interpolatePiercing): the sheet is read at the auxiliary points, the corners of the square of the fastener, each
// on its shell element,
//
//     u_k = Σ_j N_kj·R_j·u_j,   β_k = Σ_j N_kj·R_j·θ_j   at X_k = Σ_j N_kj·x_j,
//
// and ū = Σ w_k·u_k at X̄ = Σ w_k·X_k, β = Σ w_k·β_k, θ_fit the rotation of the least-squares rigid motion of the
// u_k about P, n the normal of the square. For the four corners of a square with equal weights, ū is the mean of
// their translations and n·θ_fit = ½(∂v/∂x − ∂u/∂y) of the bilinear surface through them, in axes x, y of the
// square. The size of the tie is then that of the fastener, not that of the element, and the joint stops softening
// once the elements are as small as the fastener. The map is over the six DOFs of each grid point of `grids`.
//
// The tie holds the mean translation of the corners, not each of them: the sheet strains freely under the fastener,
// and under the load of the fastener that mean is less than the motion at the piercing point (in a strip of
// stiffness EA loaded along its length, by about F·d/(2·EA), d the half side of the square). The fastener is so
// held a little more stiffly than at a point, which shows only beside a fastener about as stiff as the sheet over
// the length d: in a row of such fasteners the end ones carry more, and the inner ones may carry a small load of
// the other sign (wasm/test/validation/fasteners.test.mjs, section 4).
//
// Returns the message of a problem instead, as interpolatePiercing.
[[nodiscard]] std::variant<PiercingMap, std::string> interpolateFootprint(base::Vec3 piercingPoint,
                                                                          std::span<const SideGrid> grids,
                                                                          std::span<const AuxiliaryPoint> points);

// The map of a side connected directly to a grid point of the model (Weld ALIGN): (u_P, θ_P) = (R·u, R·θ).
[[nodiscard]] PiercingMap directGridMap(const base::Mat3& gridFrame);

// The lookup of a grid point by ID (nothing when the model does not define it). Each caller gives its own: the element
// families read the snapshot, the assembly and the reduction the DOF map; both give the same rows and geometry.
using SideGridLookup = std::function<std::optional<SideGrid>(std::int32_t gridId)>;

// One side of a fastener resolved from the fastener piercing table: its piercing point, the rows of the grid points
// that move it and the map over them: the grid points under the corners of its footprint and the map of
// interpolateFootprint (input::FastenerPiercingTable::cornerFootprint), or without corners the grid points of the
// patch of the point and the map of interpolatePiercing.
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
[[nodiscard]] std::variant<SideInterpolation, std::string> interpolateSide(const input::FastenerPiercingTable& table,
                                                                           input::PiercingRows rows,
                                                                           const SideGridLookup& findGrid);

// The motion of the piercing point P of one side as rows over DOFs of the model, (u_P, θ_P) = map·u(dofs) in the basic
// system: over the six DOFs of each grid point of interpolateSide, or, when the pierced shell is an element of a
// formulation with unknowns besides its grid points, over the DOFs of the shells under the footprint of the
// fastener: the mean of the displacement and of the rotation of their surface (FormulationModel::shellPointMotion,
// docs/mixed-hybrid-formulation-spec.md section 5.10), their grid translations and the unknowns of their edges and
// interiors.
struct SideMotion
{
    base::Vec3 point;
    std::vector<ElementDof> dofs;
    std::vector<double> map; // 6 rows of dofs.size(), row-major
};

// The motion of a side over the six DOFs of each grid point of SideInterpolation::gridRows.
[[nodiscard]] SideMotion sideMotionOf(const SideInterpolation& side);

// The motion of the side on rows `rows` of the table: through `formulation` when it has the pierced shell (null: no
// such formulation), else through the grid points of the pierced element (interpolateSide). The element families,
// assembly::FastenerConstraints and the glue of the fastener points call this, so that they tie a side alike. Returns
// the message of a problem instead, as interpolateSide.
[[nodiscard]] std::variant<SideMotion, std::string> resolveSideMotion(const input::FastenerPiercingTable& table,
                                                                      input::PiercingRows rows,
                                                                      const SideGridLookup& findGrid,
                                                                      const FormulationModel* formulation);

} // namespace oos::elements
