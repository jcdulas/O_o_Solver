#include "oos/elements/rods/rod2/Rod2Element.hpp"

namespace oos::elements
{

Rod2Element::Columns::Columns(const ElementBatchView& view)
    : connectivity(view), properties(view, "RodProp"), materials(view)
{
    if (properties.table == nullptr)
        return;
    const base::ContractReader& rodProp = properties.table->reader();
    matId = rodProp.values<std::int32_t>("matId");
    a = rodProp.values<double>("area");
    j = rodProp.values<double>("torsionConstant");
    c = rodProp.values<double>("torsionStressCoefficient");
    nonStructMass = rodProp.values<double>("nonStructMass");
}

Resolution<RodData> Rod2Element::resolve(const ElementBatchView& view, const Columns& columns, std::int64_t row)
{
    using Result = Resolution<RodData>;
    Resolution<RodGeometry> geometry = resolveRodGeometry(view, columns.connectivity, row);
    if (!geometry)
        return Result::failure(std::move(geometry.problem));
    Resolution<PropertyRef> property = resolveProperty(view, columns.properties, row);
    if (!property)
        return Result::failure(std::move(property.problem));
    const auto p = static_cast<std::size_t>(property.value->row);
    Resolution<IsotropicMaterial> material =
        resolveIsotropicMaterial(view, columns.materials, row, columns.matId[p], "RodProp.matId");
    if (!material)
        return Result::failure(std::move(material.problem));
    const RodSection section{columns.a[p], columns.j[p], columns.c[p], columns.nonStructMass[p]};
    return Result::success({*geometry.value, section, *material.value});
}

void Rod2Element::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
