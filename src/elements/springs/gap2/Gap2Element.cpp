#include "oos/elements/springs/gap2/Gap2Element.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/elements/core/ElementReferences.hpp"
#include "oos/elements/springs/SpringAxes.hpp"

#include <cmath>
#include <string>

namespace oos::elements
{

std::vector<input::FieldRule> Gap2Element::fieldRules() const
{
    return {{"GapProp", "preload", 0.0, "a gap preload GapProp.preload is not supported in a linear analysis"}};
}

Gap2Element::Columns::Columns(const ElementBatchView& view)
    : ga(view.columns().values<std::int32_t>("nodeA")),
      gb(view.columns().values<std::int32_t>("nodeB")),
      go(view.columns().values<std::int32_t>("orientationNode", base::NullPolicy::Allow)),
      x(view.columns().fixedLists<double>("orientation", base::NullPolicy::Allow)),
      isXGiven(view.columns().validity("orientation")),
      frameId(view.columns().values<std::int32_t>("frameId")),
      properties(view, "GapProp")
{
    if (properties.table == nullptr)
        return;
    const base::ContractReader& gapProp = properties.table->reader();
    u0 = gapProp.values<double>("initialOpening");
    ka = gapProp.values<double>("closedStiffness");
    kb = gapProp.values<double>("openStiffness");
    kt = gapProp.values<double>("transverseStiffness");
}

Resolution<Gap2Element::Gap> Gap2Element::resolve(const ElementBatchView& view, const Columns& columns,
                                                        std::int64_t row)
{
    using Result = Resolution<Gap>;
    const auto r = static_cast<std::size_t>(row);
    Gap gap;
    Resolution<std::int32_t> rowA = resolveGridRow(view, row, columns.ga[r], "nodeA");
    if (!rowA)
        return Result::failure(std::move(rowA.problem));
    gap.rowA = *rowA.value;
    Resolution<std::int32_t> rowB = resolveGridRow(view, row, columns.gb[r], "nodeB");
    if (!rowB)
        return Result::failure(std::move(rowB.problem));
    gap.rowB = *rowB.value;
    Resolution<PropertyRef> property = resolveProperty(view, columns.properties, row);
    if (!property)
        return Result::failure(std::move(property.problem));
    const auto p = static_cast<std::size_t>(property.value->row);
    // GapProp: closed when U0 ≤ 0 (KA axially, KT transversely), open when U0 > 0 (KB axially, nothing across).
    gap.isClosed = columns.u0[p] <= 0.0;
    gap.stiffness = gap.isClosed ? std::array<double, 3>{columns.ka[p], columns.kt[p], columns.kt[p]}
                                 : std::array<double, 3>{columns.kb[p], 0.0, 0.0};

    SpringAxesInput axesInput;
    axesInput.positionA = view.positionInBasic(gap.rowA);
    axesInput.positionB = view.positionInBasic(gap.rowB);
    axesInput.frameA = view.displacementFrame(gap.rowA);
    axesInput.frameId = columns.frameId[r];
    axesInput.needsOrientation = true;
    axesInput.go = columns.go.empty() ? 0 : columns.go[r];
    if (columns.isXGiven.isValid(row))
    {
        const std::span<const double> components = columns.x[row];
        if (!std::isnan(components[0]))
            axesInput.x = base::Vec3{components[0], components[1], components[2]};
    }
    Resolution<base::Mat3> axes = resolveSpringAxes(view, row, axesInput);
    if (!axes)
        return Result::failure(std::move(axes.problem));

    // H = Eᵀ·[−R_A, R_B]: the relative displacement in the element system.
    const base::Mat3 toElement = axes.value->transposed();
    const base::Mat3 blockA = toElement * axesInput.frameA;
    const base::Mat3 blockB = toElement * view.displacementFrame(gap.rowB);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
        {
            gap.relativeMotion(i, j) = -blockA(i, j);
            gap.relativeMotion(i, 3 + j) = blockB(i, j);
        }
    return Result::success(gap);
}

void Gap2Element::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
