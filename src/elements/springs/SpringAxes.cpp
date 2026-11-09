#include "oos/elements/springs/SpringAxes.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"

#include <cmath>
#include <string>

namespace oos::elements
{

// The messages name the fields by those of Gap2 (Gap2.frameId); Bushing1D, the other card resolved here,
// has the same keywords for them.
Resolution<base::Mat3> resolveSpringAxes(const ElementBatchView& view, std::int64_t row, const SpringAxesInput& input)
{
    using Result = Resolution<base::Mat3>;
    if (input.frameId != BLANK_SPRING_SYSTEM)
    {
        Resolution<input::CoordinateFrame> frame = resolveRectangularFrame(view, row, input.frameId, "frameId");
        if (!frame)
            return Result::failure(std::move(frame.problem));
        return Result::success(frame.value->axes);
    }
    const base::Vec3 span = input.positionB - input.positionA;
    const double length = base::norm(span);
    if (input.isGrounded || length < SPRING_COINCIDENT_DISTANCE)
        return Result::failure(elementProblem(
            view, row, problem_code::INVALID_FIELD, "frameId",
            base::concatenate(
                {"expected a Gap2.frameId, found it blank with ",
                 input.isGrounded
                     ? std::string("Gap2.nodeB blank")
                     : std::string("Gap2.nodeA and Gap2.nodeB ").append(std::to_string(length)).append(" apart"),
                 " (", view.card(),
                 ": the element axis needs Gap2.frameId when Gap2.nodeB is blank or "
                 "Gap2.nodeA and Gap2.nodeB coincide)"})));
    const base::Vec3 xAxis = span / length;
    if (!input.needsOrientation)
    {
        // Only the axis matters (CBUSH1D): any y, z completing it.
        const base::Vec3 helper = std::abs(xAxis[0]) < 0.9 ? base::Vec3{1.0, 0.0, 0.0} : base::Vec3{0.0, 1.0, 0.0};
        const base::Vec3 zAxis = base::normalized(base::cross(xAxis, helper));
        return Result::success(base::matrixFromRows(xAxis, base::cross(zAxis, xAxis), zAxis).transposed());
    }
    base::Vec3 orientation;
    const char* orientationField = "orientation1";
    if (input.go > 0)
    {
        orientationField = "orientationNode";
        Resolution<std::int32_t> rowO = resolveGridRow(view, row, input.go, "orientationNode");
        if (!rowO)
            return Result::failure(std::move(rowO.problem));
        orientation = view.positionInBasic(*rowO.value) - input.positionA;
    }
    else if (input.x)
        orientation = input.frameA * *input.x;
    const double orientationLength = base::norm(orientation);
    if (orientationLength == 0.0)
        return Result::failure(elementProblem(view, row, problem_code::INVALID_FIELD, "orientation1",
                                              "expected an orientation (Gap2.orientationNode, or "
                                              "Gap2.orientation1-Gap2.orientation3 not all zero) or a "
                                              "Gap2.frameId, found none; give Gap2.orientationNode, "
                                              "Gap2.orientation1-Gap2.orientation3 or Gap2.frameId"));
    const base::Vec3 zDirection = base::cross(xAxis, orientation);
    if (base::norm(zDirection) <= SPRING_PARALLEL_ORIENTATION_FACTOR * orientationLength)
        return Result::failure(
            elementProblem(view, row, problem_code::DEGENERATE_GEOMETRY, orientationField,
                           "expected an orientation vector not parallel to "
                           "Gap2.nodeA-Gap2.nodeB, found a parallel one; the "
                           "element y and z axes are undefined: choose another Gap2.orientationNode or "
                           "Gap2.orientation1-Gap2.orientation3"));
    const base::Vec3 zAxis = base::normalized(zDirection);
    return Result::success(base::matrixFromRows(xAxis, base::cross(zAxis, xAxis), zAxis).transposed());
}

} // namespace oos::elements
