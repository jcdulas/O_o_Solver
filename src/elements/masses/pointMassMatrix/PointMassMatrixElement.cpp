#include "oos/elements/masses/pointMassMatrix/PointMassMatrixElement.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/core/MatrixKernels.hpp"

#include <string>

namespace oos::elements
{

namespace
{

constexpr std::int64_t LOWER_TRIANGLE_TERMS = 21; // M11, M21, M22, ..., M66

} // namespace

PointMassMatrixElement::Columns::Columns(const base::ContractReader& reader)
    : g(reader.values<std::int32_t>("node")),
      frameId(reader.values<std::int32_t>("frameId")),
      m(reader.fixedLists<double>("massMatrix"))
{
}

Resolution<PointMassMatrixElement::Mass> PointMassMatrixElement::resolve(const ElementBatchView& view, const Columns& columns,
                                                               std::int64_t row)
{
    using Result = Resolution<Mass>;
    const auto r = static_cast<std::size_t>(row);
    if (columns.m.width() != LOWER_TRIANGLE_TERMS)
        return Result::failure(elementProblem(
            view, row, problem_code::INVALID_FIELD, "massTerm11",
            base::concatenate({"expected the 21 terms of the lower triangle in the column m, found ",
                               std::to_string(columns.m.width()), "; the table does not follow the data contract"})));
    Resolution<std::int32_t> gridRow = resolveGridRow(view, row, columns.g[r], "node");
    if (!gridRow)
        return Result::failure(std::move(gridRow.problem));
    Resolution<input::CoordinateFrame> frame = resolveRectangularFrame(view, row, columns.frameId[r], "frameId");
    if (!frame)
        return Result::failure(std::move(frame.problem));
    // The lower triangle row by row: term (i, j), j ≤ i, at i·(i + 1)/2 + j.
    const std::span<const double> terms = columns.m[row];
    base::MatN<6> massInFrame;
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j <= i; ++j)
        {
            massInFrame(i, j) = terms[i * (i + 1) / 2 + j];
            massInFrame(j, i) = massInFrame(i, j);
        }
    const base::Mat3 frameToDisplacement = frame.value->axes.transposed() * view.displacementFrame(*gridRow.value);
    return Result::success(
        {*gridRow.value, toDisplacementFrames<6>(massInFrame, {frameToDisplacement, frameToDisplacement})});
}

void PointMassMatrixElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view.columns());
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
