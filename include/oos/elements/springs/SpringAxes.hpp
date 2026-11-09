#pragma once

#include "oos/base/math/Mat3.hpp"
#include "oos/base/math/Vec3.hpp"
#include "oos/elements/core/ElementBatchView.hpp"
#include "oos/elements/core/ElementProblem.hpp"

#include <cstdint>
#include <optional>

namespace oos::elements
{

// GA and GB closer than this (model length units) give no element x-axis: CID must then be given (CBUSH1D and
// Gap2, as Bushing: BUSH_COINCIDENT_DISTANCE).
inline constexpr double SPRING_COINCIDENT_DISTANCE = 1.0e-4;

// The orientation vector is parallel to the axis when |x × v| ≤ this factor times |v|.
inline constexpr double SPRING_PARALLEL_ORIENTATION_FACTOR = 1.0e-12;

// O_o_Reader gives a blank CID of Bushing1D and Gap2 as −1 (0 is the basic system).
inline constexpr std::int32_t BLANK_SPRING_SYSTEM = -1;

// Where the element axes of a two-point spring come from (CBUSH1D, Gap2): the system CID when it is given, else x
// from GA to GB and, when the element needs y and z, the orientation vector v from the grid point GO (v = x_GO − x_GA)
// or from X1-X3 (in the displacement system of GA), z = x × v / |x × v|, y = z × x (the Bar2 convention).
struct SpringAxesInput
{
    base::Vec3 positionA;
    base::Vec3 positionB;
    base::Mat3 frameA;       // the CD-to-basic rotation of GA (X1-X3 are in that system)
    bool isGrounded = false; // GB blank
    std::int32_t frameId = BLANK_SPRING_SYSTEM;
    bool needsOrientation = false; // Gap2: y and z are needed; Bushing1D: x only
    std::int32_t go = 0;
    std::optional<base::Vec3> x; // X1-X3 when given
};

// The element axes as the columns of a rotation (element to basic), or the problem: CID unknown or curvilinear
// (field CID), CID missing for a grounded or zero-length element (field CID), GO unknown (field GO), no orientation
// or one parallel to the axis (field X1 or GO).
[[nodiscard]] Resolution<base::Mat3> resolveSpringAxes(const ElementBatchView& view, std::int64_t row,
                                                       const SpringAxesInput& input);

} // namespace oos::elements
