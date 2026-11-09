#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/MatN.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementDof.hpp"
#include "oos/elements/core/ElementProblem.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace oos::elements
{

// The geometry of a Bar2 or Beam2 (the orientation vector, OFFT, the offsets and the pin flags of the Bar2 and
// Beam2 cards), built in this order:
//
// - End points: end A = x_GA + w_A, end B = x_GB + w_B (basic system), w the offset vectors WA, WB.
// - Element system: x from end A to end B; the orientation vector v either from GA to G0 (v = x_G0 − x_GA, then moved
//   to end A) or X1-X3; z = x × v / |x × v|, y = z × x.
// - OFFT: three letters, the first for the orientation vector X1-X3 (G: the displacement system of
//   GA, B: the basic system), the second and third for the offsets WA and WB (G: the displacement system of GA, resp.
//   GB; O: the offset system). Design choice: the offset system has its x-axis from GA to GB (not from
//   the offset ends, which it serves to define) and its y and z axes from v as the element system does.
// - Pin flags PA, PB: digits 1-6 of the element-system DOFs released at end A, B (the bar does not connect them to
//   the grid point); the stiffness is condensed on them (PinRelease).
//
// The element's DOFs are the six of each end point in the element system, d = [u_A θ_A u_B θ_B] (the "end DOFs"); the
// grid point DOFs g = [u_GA θ_GA u_GB θ_GB] are in the displacement systems CD. d = T·g with, per end,
// T_end = [[Eᵀ·R, −Eᵀ·[w]×·R], [0, Eᵀ·R]] (R the CD-to-basic rotation of the grid point, E the element axes as columns
// in basic, w the offset in basic): the rigid offset u_end = u + θ × w, then the element axes.
//
// Not supported (refused by resolveBeamGeometry): OFFT with the letter E or an old numeric code; CD rotations are
// those O_o_Reader evaluates at the grid point (curvilinear CD allowed).
struct BeamGeometry
{
    std::int32_t rowA = 0;
    std::int32_t rowB = 0;
    base::Mat3 frameA = base::Mat3::identity(); // CD-to-basic rotation of GA
    base::Mat3 frameB = base::Mat3::identity(); // of GB
    base::Vec3 offsetA;                         // w_A in basic
    base::Vec3 offsetB;                         // w_B in basic
    base::Mat3 axes = base::Mat3::identity();   // columns: the element x, y, z axes in basic
    double length = 0.0;                        // between the end points
    std::array<bool, 12> isReleased{};          // the pin flags, on the end DOFs [A 1-6, B 1-6]

    // T, the map from the grid point DOFs to the end DOFs (d = T·g).
    [[nodiscard]] base::MatN<12> endTransform() const;
    // T without the offsets, per end [[Eᵀ·R, 0], [0, Eᵀ·R]]: what carries the mass of a bar or beam to its grid points
    // under ModelParams OFFDEF ELMOFF, the default (input::OffsetDefinition; only LROFF carries the masses through the
    // offsets).
    [[nodiscard]] base::MatN<12> rotationTransform() const;
    // The grid point DOFs [GA 1-6, GB 1-6], in the order of T's columns.
    [[nodiscard]] std::array<ElementDof, 12> dofs() const;
    [[nodiscard]] bool hasPinFlags() const noexcept;

private:
    // T with the offsets `offsets` (in basic) at ends A and B.
    [[nodiscard]] base::MatN<12> transformThrough(const std::array<base::Vec3, 2>& offsets) const;
};

// The grid point and orientation columns that Bar2 and Beam2 share, read once per facet call.
struct BeamConnectionColumns
{
    std::span<const std::int32_t> ga;
    std::span<const std::int32_t> gb;
    base::FixedListView<double> x;
    std::span<const std::int32_t> g0;
    base::Utf8View offsetKind;
    std::span<const std::int32_t> pa;
    std::span<const std::int32_t> pb;
    base::FixedListView<double> offsetA; // the end offsets as given on the card (OFFT says in which system)
    base::FixedListView<double> offsetB;

    explicit BeamConnectionColumns(const ElementBatchView& view);
};

// End points closer than this factor times max(1, |x_GA|, |x_GB|) give no element axis: refused as degenerate, as
// for the rods (a zero-length bar has no element system).
inline constexpr double BEAM_ZERO_LENGTH_FACTOR = 1.0e-12;

// The orientation vector is refused as parallel to the axis when |x × v| ≤ this factor times |v| (the element y and z
// axes are then undefined).
inline constexpr double BEAM_PARALLEL_ORIENTATION_FACTOR = 1.0e-10;

// The geometry of the element of batch row `row`; an unknown grid point, an orientation vector that is zero or
// parallel to the axis, a zero length, an OFFT or pin flag not supported, are problems.
[[nodiscard]] Resolution<BeamGeometry> resolveBeamGeometry(const ElementBatchView& view,
                                                           const BeamConnectionColumns& columns, std::int64_t row);

} // namespace oos::elements
