#include "oos/elements/masses/pointMassInertia/PointMassInertiaElement.hpp"

#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/core/MatrixKernels.hpp"

namespace oos::elements
{

namespace
{

// PointMassInertia: CID −1 gives the centre of gravity and the inertia in the basic system.
constexpr std::int32_t BASIC_FRAME_ID = -1;

} // namespace

base::MatN<6> rigidBodyMass(double mass, base::Vec3 offsetInBasic, const base::Mat3& inertiaAtCentre)
{
    const base::Mat3 cross = crossProductMatrix(offsetInBasic);
    const base::Mat3 coupling = mass * cross;
    const base::Mat3 rotational = inertiaAtCentre - mass * (cross * cross);
    base::MatN<6> result;
    for (std::size_t r = 0; r < 3; ++r)
    {
        result(r, r) = mass;
        for (std::size_t c = 0; c < 3; ++c)
        {
            result(r, 3 + c) = -coupling(r, c);
            result(3 + r, c) = coupling(r, c);
            result(3 + r, 3 + c) = rotational(r, c);
        }
    }
    return result;
}

PointMassInertiaElement::Columns::Columns(const base::ContractReader& reader)
    : g(reader.values<std::int32_t>("node")),
      frameId(reader.values<std::int32_t>("frameId")),
      m(reader.values<double>("mass")),
      x(reader.fixedLists<double>("offset")),
      i(reader.fixedLists<double>("inertia"))
{
}

Resolution<PointMassInertiaElement::Mass> PointMassInertiaElement::resolve(const ElementBatchView& view, const Columns& columns,
                                                               std::int64_t row)
{
    using Result = Resolution<Mass>;
    const auto r = static_cast<std::size_t>(row);
    Resolution<std::int32_t> gridRow = resolveGridRow(view, row, columns.g[r], "node");
    if (!gridRow)
        return Result::failure(std::move(gridRow.problem));
    const std::span<const double> x = columns.x[row];
    const std::span<const double> terms = columns.i[row]; // I11, I21, I22, I31, I32, I33
    const base::Mat3 inertia = base::matrixFromRows({terms[0], -terms[1], -terms[3]}, {-terms[1], terms[2], -terms[4]},
                                                    {-terms[3], -terms[4], terms[5]});
    const base::Vec3 gridPosition = view.positionInBasic(*gridRow.value);
    base::Vec3 offset;
    base::Mat3 inertiaInBasic = inertia;
    if (columns.frameId[r] == BASIC_FRAME_ID)
        offset = base::Vec3{x[0], x[1], x[2]} - gridPosition;
    else
    {
        Resolution<input::CoordinateFrame> frame = resolveRectangularFrame(view, row, columns.frameId[r], "frameId");
        if (!frame)
            return Result::failure(std::move(frame.problem));
        const base::Mat3& axes = frame.value->axes;
        offset = axes * base::Vec3{x[0], x[1], x[2]};
        inertiaInBasic = axes * inertia * axes.transposed();
    }
    const base::Mat3 frameOfGrid = view.displacementFrame(*gridRow.value);
    const base::MatN<6> massInBasic = rigidBodyMass(columns.m[r], offset, inertiaInBasic);
    return Result::success({*gridRow.value, toDisplacementFrames<6>(massInBasic, {frameOfGrid, frameOfGrid})});
}

void PointMassInertiaElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view.columns());
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
