#include "oos/elements/rods/rod2InlineSection/Rod2InlineSectionElement.hpp"

namespace oos::elements
{

Rod2InlineSectionElement::Columns::Columns(const ElementBatchView& view)
    : connectivity(view),
      materials(view),
      matId(view.columns().values<std::int32_t>("matId")),
      a(view.columns().values<double>("area")),
      j(view.columns().values<double>("torsionConstant")),
      c(view.columns().values<double>("torsionStressCoefficient")),
      nonStructMass(view.columns().values<double>("nonStructMass"))
{
}

Resolution<RodData> Rod2InlineSectionElement::resolve(const ElementBatchView& view, const Columns& columns,
                                                    std::int64_t row)
{
    using Result = Resolution<RodData>;
    Resolution<RodGeometry> geometry = resolveRodGeometry(view, columns.connectivity, row);
    if (!geometry)
        return Result::failure(std::move(geometry.problem));
    const auto i = static_cast<std::size_t>(row);
    Resolution<IsotropicMaterial> material =
        resolveIsotropicMaterial(view, columns.materials, row, columns.matId[i], "matId");
    if (!material)
        return Result::failure(std::move(material.problem));
    const RodSection section{columns.a[i], columns.j[i], columns.c[i], columns.nonStructMass[i]};
    return Result::success({*geometry.value, section, *material.value});
}

void Rod2InlineSectionElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
