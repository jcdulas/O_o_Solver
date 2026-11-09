#include "oos/elements/bars_beams/BeamGeometry.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/core/MatrixKernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace oos::elements
{

namespace
{

constexpr std::size_t PIN_DIGITS = 6;

// The released end DOFs of a pin flag (digits 1-6, each once, in any order); false when the flag has another digit or
// a repeated one.
bool readPinFlag(std::int32_t flag, std::span<bool> released)
{
    if (flag < 0)
        return false;
    for (std::int32_t rest = flag; rest > 0; rest /= 10)
    {
        const std::int32_t digit = rest % 10;
        if (digit < 1 || digit > static_cast<std::int32_t>(PIN_DIGITS) || released[static_cast<std::size_t>(digit - 1)])
            return false;
        released[static_cast<std::size_t>(digit - 1)] = true;
    }
    return true;
}

// The element axes (columns in basic) from the axis direction and the orientation vector: z = x × v / |x × v|,
// y = z × x (the element coordinate system of a Bar2 or Beam2). Nothing when v is parallel to x.
std::optional<base::Mat3> axesFrom(base::Vec3 xAxis, base::Vec3 orientation)
{
    const base::Vec3 zDirection = base::cross(xAxis, orientation);
    if (base::norm(zDirection) <= BEAM_PARALLEL_ORIENTATION_FACTOR * base::norm(orientation))
        return std::nullopt;
    const base::Vec3 zAxis = base::normalized(zDirection);
    const base::Vec3 yAxis = base::cross(zAxis, xAxis);
    return base::matrixFromRows(xAxis, yAxis, zAxis).transposed();
}

// The reader's offsetKind column names the three frames: the orientation vector, then the offsets at ends A and B.
std::optional<std::array<bool, 3>> globalOffsetFrames(std::string_view kind)
{
    constexpr std::array<std::pair<std::string_view, std::array<bool, 3>>, 8> KINDS{{
        {"OffsetsGlobalGlobalGlobal", {true, true, true}},
        {"OffsetsGlobalGlobalOrient", {true, true, false}},
        {"OffsetsGlobalOrientGlobal", {true, false, true}},
        {"OffsetsGlobalOrientOrient", {true, false, false}},
        {"OffsetsBasicGlobalGlobal", {false, true, true}},
        {"OffsetsBasicGlobalOrient", {false, true, false}},
        {"OffsetsBasicOrientGlobal", {false, false, true}},
        {"OffsetsBasicOrientOrient", {false, false, false}},
    }};
    for (const auto& [name, frames] : KINDS)
        if (kind == name)
            return frames;
    return std::nullopt;
}

} // namespace

BeamConnectionColumns::BeamConnectionColumns(const ElementBatchView& view)
    : ga(view.columns().values<std::int32_t>("nodeA")),
      gb(view.columns().values<std::int32_t>("nodeB")),
      x(view.columns().fixedLists<double>("orientation", base::NullPolicy::Allow)),
      g0(view.columns().values<std::int32_t>("orientationNode", base::NullPolicy::Allow)),
      offsetKind(view.columns().texts("offsetKind")),
      pa(view.columns().values<std::int32_t>("pinFlagsA")),
      pb(view.columns().values<std::int32_t>("pinFlagsB")),
      offsetA(view.columns().fixedLists<double>("offsetA")),
      offsetB(view.columns().fixedLists<double>("offsetB"))
{
}

base::MatN<12> BeamGeometry::endTransform() const
{
    return transformThrough({offsetA, offsetB});
}

base::MatN<12> BeamGeometry::rotationTransform() const
{
    return transformThrough({base::Vec3{}, base::Vec3{}});
}

base::MatN<12> BeamGeometry::transformThrough(const std::array<base::Vec3, 2>& offsets) const
{
    base::MatN<12> t;
    const base::Mat3 toElement = axes.transposed();
    const std::array<base::Mat3, 2> frames{frameA, frameB};
    for (std::size_t end = 0; end < 2; ++end)
    {
        const base::Mat3 rotation = toElement * frames[end];
        // u_end = u + θ × w = u − [w]×·θ.
        const base::Mat3 arm = -1.0 * (toElement * crossProductMatrix(offsets[end]) * frames[end]);
        const std::size_t first = 6 * end;
        for (std::size_t i = 0; i < 3; ++i)
            for (std::size_t j = 0; j < 3; ++j)
            {
                t(first + i, first + j) = rotation(i, j);
                t(first + i, first + 3 + j) = arm(i, j);
                t(first + 3 + i, first + 3 + j) = rotation(i, j);
            }
    }
    return t;
}

std::array<ElementDof, 12> BeamGeometry::dofs() const
{
    std::array<ElementDof, 12> result;
    for (int c = 0; c < 6; ++c)
    {
        result[static_cast<std::size_t>(c)] = ElementDof::grid(rowA, c + 1);
        result[static_cast<std::size_t>(6 + c)] = ElementDof::grid(rowB, c + 1);
    }
    return result;
}

bool BeamGeometry::hasPinFlags() const noexcept
{
    return std::any_of(isReleased.begin(), isReleased.end(), [](bool flag) { return flag; });
}

Resolution<BeamGeometry> resolveBeamGeometry(const ElementBatchView& view, const BeamConnectionColumns& columns,
                                             std::int64_t row)
{
    using Result = Resolution<BeamGeometry>;
    const auto r = static_cast<std::size_t>(row);
    BeamGeometry geometry;
    Resolution<std::int32_t> rowA = resolveGridRow(view, row, columns.ga[r], "nodeA");
    if (!rowA)
        return Result::failure(std::move(rowA.problem));
    Resolution<std::int32_t> rowB = resolveGridRow(view, row, columns.gb[r], "nodeB");
    if (!rowB)
        return Result::failure(std::move(rowB.problem));
    geometry.rowA = *rowA.value;
    geometry.rowB = *rowB.value;

    const auto globalFrames = globalOffsetFrames(columns.offsetKind[row]);
    if (!globalFrames)
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, "offsetKind",
            base::concatenate({"expected Bar2.offsetKind to name the orientation frame (Global or Basic) "
                               "and both offset frames (Global or Orient), such as OffsetsGlobalGlobalGlobal, found \"",
                               columns.offsetKind[row], "\"; choose a supported offset frame combination"})));
    if (!readPinFlag(columns.pa[r], std::span<bool>(geometry.isReleased).subspan(0, 6)))
        return Result::failure(
            elementProblem(view, row, problem_code::INVALID_FIELD, "pinFlagsA",
                           base::concatenate({"expected the digits 1 to 6 without repeat, found ",
                                              std::to_string(columns.pa[r]), "; correct the pin flag"})));
    if (!readPinFlag(columns.pb[r], std::span<bool>(geometry.isReleased).subspan(6, 6)))
        return Result::failure(
            elementProblem(view, row, problem_code::INVALID_FIELD, "pinFlagsB",
                           base::concatenate({"expected the digits 1 to 6 without repeat, found ",
                                              std::to_string(columns.pb[r]), "; correct the pin flag"})));

    const base::Vec3 positionA = view.positionInBasic(geometry.rowA);
    const base::Vec3 positionB = view.positionInBasic(geometry.rowB);
    geometry.frameA = view.displacementFrame(geometry.rowA);
    geometry.frameB = view.displacementFrame(geometry.rowB);

    // The orientation vector v in basic.
    base::Vec3 orientation;
    const char* orientationField = "orientation1";
    if (columns.g0[r] > 0)
    {
        orientationField = "orientationNode";
        Resolution<std::int32_t> rowO = resolveGridRow(view, row, columns.g0[r], "orientationNode");
        if (!rowO)
            return Result::failure(std::move(rowO.problem));
        orientation = view.positionInBasic(*rowO.value) - positionA;
    }
    else
    {
        const std::span<const double> components = columns.x[row];
        const base::Vec3 given{components[0], components[1], components[2]};
        orientation = (*globalFrames)[0] ? geometry.frameA * given : given;
    }
    if (base::norm(orientation) == 0.0)
        return Result::failure(
            elementProblem(view, row, problem_code::INVALID_FIELD, orientationField,
                           "expected an orientation vector (Bar2.orientation1-Bar2.orientation3 not all zero, "
                           "or a "
                           "Bar2.orientationNode away from Bar2.nodeA), found a zero vector; give the "
                           "orientation of the element y axis"));

    const double scale = std::max({1.0, base::norm(positionA), base::norm(positionB)});
    const auto parallelProblem = [&]
    {
        return Result::failure(elementProblem(
            view, row, problem_code::DEGENERATE_GEOMETRY, orientationField,
            "expected an orientation vector not parallel to the element axis, found a parallel one; the element y "
            "and z axes are undefined: choose another orientation"));
    };

    // The offsets in basic: G in the displacement system of the grid point, O in the offset system.
    const std::span<const double> wa = columns.offsetA[row];
    const std::span<const double> wb = columns.offsetB[row];
    const base::Vec3 givenA{wa[0], wa[1], wa[2]};
    const base::Vec3 givenB{wb[0], wb[1], wb[2]};
    base::Mat3 offsetAxes = base::Mat3::identity();
    if (!(*globalFrames)[1] || !(*globalFrames)[2])
    {
        const base::Vec3 gridSpan = positionB - positionA;
        if (base::norm(gridSpan) <= BEAM_ZERO_LENGTH_FACTOR * scale)
            return Result::failure(
                elementProblem(view, row, problem_code::DEGENERATE_GEOMETRY, "offsetKind",
                               "expected Bar2.nodeA and Bar2.nodeB apart for offsets in the offset system (its "
                               "x-axis is Bar2.nodeA-Bar2.nodeB), found them coincident; give the offsets in "
                               "the displacement systems (Bar2.offsetKind OffsetsGlobalGlobalGlobal)"));
        const std::optional<base::Mat3> axes = axesFrom(base::normalized(gridSpan), orientation);
        if (!axes)
            return parallelProblem();
        offsetAxes = *axes;
    }
    geometry.offsetA = (*globalFrames)[1] ? geometry.frameA * givenA : offsetAxes * givenA;
    geometry.offsetB = (*globalFrames)[2] ? geometry.frameB * givenB : offsetAxes * givenB;

    const base::Vec3 span = (positionB + geometry.offsetB) - (positionA + geometry.offsetA);
    geometry.length = base::norm(span);
    if (geometry.length <= BEAM_ZERO_LENGTH_FACTOR * scale)
        return Result::failure(elementProblem(
            view, row, problem_code::DEGENERATE_GEOMETRY, "nodeB",
            base::concatenate({"expected end points apart, found a length of ", std::to_string(geometry.length),
                               " between end A and end B (offsets included); correct the grid points or offsets"})));
    const std::optional<base::Mat3> axes = axesFrom(span / geometry.length, orientation);
    if (!axes)
        return parallelProblem();
    geometry.axes = *axes;
    return Result::success(geometry);
}

} // namespace oos::elements
