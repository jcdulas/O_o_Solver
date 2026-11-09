#include "oos/elements/shells/ShellPressure.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <cmath>
#include <string>
#include <utility>

namespace oos::elements
{

namespace
{

// A direction vector shorter than this (relative to its largest component scale of 1) has no direction.
constexpr double MIN_DIRECTION_LENGTH = 1.0e-12;

} // namespace

PressureDirection pressureDirectionOf(const ElementBatchView& view, const ElementPressure& pressure,
                                      const base::Mat3& elementAxes)
{
    PressureDirection result;
    if (!pressure.hasDirection)
        return result;
    const std::optional<input::CoordinateFrame> frame = view.index().findCoordinateFrame(pressure.directionFrameId);
    if (!frame)
    {
        result.problem =
            base::concatenate({"expected the coordinate system ", std::to_string(pressure.directionFrameId),
                               " of the pressure direction, found none in the model; define it or "
                               "correct FacePressureLoad.frameId"});
        return result;
    }
    if (frame->kind != input::CoordinateFrame::Kind::Rectangular)
    {
        result.problem = base::concatenate(
            {"expected a rectangular system for the pressure direction, found the "
             "curvilinear system ",
             std::to_string(pressure.directionFrameId),
             "; give FacePressureLoad.directionX-FacePressureLoad.directionZ in a rectangular system"});
        return result;
    }
    const base::Vec3 local{pressure.direction[0], pressure.direction[1], pressure.direction[2]};
    const double length = std::sqrt(local[0] * local[0] + local[1] * local[1] + local[2] * local[2]);
    if (!(length > MIN_DIRECTION_LENGTH))
    {
        result.problem =
            "expected a direction FacePressureLoad.directionX-FacePressureLoad.directionZ of non-zero "
            "length, found a null vector; give "
            "the direction or leave FacePressureLoad.frameId and "
            "FacePressureLoad.directionX-FacePressureLoad.directionZ blank for a normal pressure";
        return result;
    }
    // N in basic, then in the element frame: v_e = Eᵀ (axes · N / |N|).
    base::Vec3 basic{};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            basic[i] += frame->axes(i, k) * local[k] / length;
    base::Vec3 element{};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t k = 0; k < 3; ++k)
            element[i] += elementAxes(k, i) * basic[k];
    result.inElementFrame = element;
    return result;
}

void checkShellPressures(const ElementBatchView& view, std::span<const ElementPressure> pressures,
                         base::DiagnosticSink& sink)
{
    for (const ElementPressure& pressure : pressures)
    {
        if (!pressure.hasDirection)
            continue;
        // The element axes do not change whether a direction can be used: the identity serves the check.
        const PressureDirection direction = pressureDirectionOf(view, pressure, base::Mat3::identity());
        if (direction.problem.empty())
            continue;
        base::Diagnostic diagnostic;
        diagnostic.code = "unsupported-load";
        diagnostic.card = std::string(pressure.card);
        diagnostic.id = pressure.setId;
        diagnostic.field = "frameId";
        diagnostic.message = base::concatenate(
            {direction.problem, " (on ", view.card(), " ", std::to_string(view.elementId(pressure.elementRow)), ")"});
        sink.report(std::move(diagnostic));
    }
}

} // namespace oos::elements
