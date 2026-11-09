#include "oos/elements/fasteners/fastener/FastenerElement.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/core/MatrixKernels.hpp"

#include <string>

namespace oos::elements
{

namespace
{

// FastenerProp: MCID = −1 (default) gives the fastener system from the geometry; MFLAG = 1 makes MCID absolute.
constexpr std::int32_t GEOMETRIC_SYSTEM = -1;
constexpr std::uint8_t ABSOLUTE_SYSTEM = 1;

void setBlock(base::MatN<6, 12>& h, std::size_t blockRow, std::size_t blockColumn, const base::Mat3& block)
{
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            h(3 * blockRow + i, 3 * blockColumn + j) = block(i, j);
}

} // namespace

std::vector<input::FieldRule> FastenerElement::fieldRules() const
{
    return {{"FastenerProp", "alphaCoefficient", 0.0, "the thermal expansion of a fastener is not supported"},
            {"FastenerProp", "coincidentLength", 0.0, "the coincidence length of a fastener is not supported"}};
}

FastenerElement::Columns::Columns(const ElementBatchView& view) : properties(view, "FastenerProp")
{
    if (properties.table == nullptr)
        return;
    const base::ContractReader& fastenerProp = properties.table->reader();
    matFrameId = fastenerProp.values<std::int32_t>("matFrameId");
    mflag = fastenerProp.values<std::uint8_t>("absoluteFrame");
    kt = fastenerProp.fixedLists<double>("translationalStiffness");
    kr = fastenerProp.fixedLists<double>("rotationalStiffness");
    mass = fastenerProp.values<double>("mass");
}

Resolution<FastenerElement::Fastener> FastenerElement::resolve(const ElementBatchView& view,
                                                                       const Columns& columns, std::int64_t row)
{
    using Result = Resolution<Fastener>;
    Fastener fastener;
    Resolution<PropertyRef> property = resolveProperty(view, columns.properties, row);
    if (!property)
        return Result::failure(std::move(property.problem));
    const std::int32_t p = property.value->row;
    const auto pRow = static_cast<std::size_t>(p);
    const std::int32_t propId = columns.properties.propIds[static_cast<std::size_t>(row)];
    // The piercing points come from the table; a Fastener always has one on each side (no direct connection).
    Resolution<FastenerConnection> connection = resolveFastenerConnection(view, row, {0, 0});
    if (!connection)
        return Result::failure(std::move(connection.problem));
    fastener.connection = std::move(*connection.value);

    base::Mat3 axes;
    if (columns.matFrameId[pRow] == GEOMETRIC_SYSTEM)
        axes = fastenerAxes(fastener.connection.pointA, fastener.connection.pointB);
    else if (columns.mflag[pRow] == ABSOLUTE_SYSTEM)
    {
        Resolution<input::CoordinateFrame> frame =
            resolveRectangularFrame(view, row, columns.matFrameId[pRow], "FastenerProp.matFrameId");
        if (!frame)
            return Result::failure(std::move(frame.problem));
        axes = frame.value->axes;
    }
    else
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, "propId",
            base::concatenate(
                {"FastenerProp ", std::to_string(propId),
                 ": expected FastenerProp.matFrameId −1, or FastenerProp.matFrameId with "
                 "FastenerProp.absoluteFrame 1, found FastenerProp.matFrameId ",
                 std::to_string(columns.matFrameId[pRow]),
                 " with FastenerProp.absoluteFrame 0; the relative fastener system is not supported: give "
                 "FastenerProp.absoluteFrame 1 (FastenerProp.matFrameId absolute) or leave "
                 "FastenerProp.matFrameId blank"})));
    const std::span<const double> kt = columns.kt[p];
    const std::span<const double> kr = columns.kr[p];
    fastener.stiffness = {kt[0], kt[1], kt[2], kr[0], kr[1], kr[2]};
    fastener.mass = columns.mass[pRow];

    // H over w = (u_A, θ_A, u_B, θ_B) in basic, the spring at the middle p of A-B (FastenerElement.hpp).
    const base::Vec3 middle = 0.5 * (fastener.connection.pointA + fastener.connection.pointB);
    const base::Mat3 toFastener = axes.transposed();
    setBlock(fastener.relativeMotion, 0, 0, -1.0 * toFastener); // −u_A
    setBlock(fastener.relativeMotion, 0, 1,
             toFastener * crossProductMatrix(middle - fastener.connection.pointA)); // −θ_A × (p − x_A)
    setBlock(fastener.relativeMotion, 0, 2, toFastener);                            // u_B
    setBlock(fastener.relativeMotion, 0, 3,
             -1.0 * (toFastener * crossProductMatrix(middle - fastener.connection.pointB))); // θ_B × (p − x_B)
    setBlock(fastener.relativeMotion, 1, 1, -1.0 * toFastener);
    setBlock(fastener.relativeMotion, 1, 3, toFastener);
    return Result::success(std::move(fastener));
}

void FastenerElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
