#include "oos/elements/core/ElementReferences.hpp"

#include "oos/base/errors/ContractError.hpp"
#include "oos/base/errors/ErrorMessage.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>

namespace oos::elements
{

Resolution<std::int32_t> resolveGridRow(const ElementBatchView& view, std::int64_t row, std::int32_t gridId,
                                        const char* field)
{
    const std::optional<std::int32_t> gridRow = view.index().points().findGrid(gridId);
    if (!gridRow)
        return Resolution<std::int32_t>::failure(elementProblem(
            view, row, problem_code::MISSING_POINT, field,
            base::concatenate({"expected the ID of a grid point, found ", std::to_string(gridId),
                               ", which the model does not define as a grid point; define it or correct the card"})));
    return Resolution<std::int32_t>::success(*gridRow);
}

std::int32_t checkedGridRow(const ElementBatchView& view, std::int32_t givenRow, std::int32_t gridId)
{
    const std::span<const std::int32_t> nodeIds = view.snapshot().nodes().ids();
    if (givenRow >= 0 && static_cast<std::size_t>(givenRow) < nodeIds.size() &&
        nodeIds[static_cast<std::size_t>(givenRow)] == gridId)
        return givenRow;
    const std::optional<std::int32_t> gridRow = view.index().points().findGrid(gridId);
    if (!gridRow)
        throw base::ContractError(base::concatenate(
            {view.card(), ": expected grid point ", std::to_string(gridId),
             " in the node table, found none; the element table and the node table come from different snapshots: "
             "take every table of the snapshot from one model"}));
    return *gridRow;
}

Resolution<input::CoordinateFrame> resolveRectangularFrame(const ElementBatchView& view, std::int64_t row,
                                                           std::int32_t frameId, const char* field)
{
    using Result = Resolution<input::CoordinateFrame>;
    const std::optional<input::CoordinateFrame> frame = view.index().findCoordinateFrame(frameId);
    if (!frame)
        return Result::failure(elementProblem(
            view, row, problem_code::MISSING_COORDINATE_SYSTEM, field,
            base::concatenate({"expected the ID of a coordinate system, found ", std::to_string(frameId),
                               ", which the model does not define (or the snapshot was taken without coordsTable); "
                               "define the coordinate system or correct the field"})));
    if (frame->kind != input::CoordinateFrame::Kind::Rectangular)
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, field,
            base::concatenate({"expected a rectangular coordinate system, found the ",
                               frame->kind == input::CoordinateFrame::Kind::Cylindrical ? "cylindrical" : "spherical",
                               " system ", std::to_string(frameId), "; ", view.card(),
                               " in a curvilinear system is not supported yet: use a rectangular system"})));
    return Result::success(*frame);
}

} // namespace oos::elements
