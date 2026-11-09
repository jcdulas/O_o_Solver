#include "oos/elements/springs/springScalarProperty/SpringScalarPropertyElement.hpp"

namespace oos::elements
{

SpringScalarPropertyElement::Columns::Columns(const ElementBatchView& view)
    : s1(view.columns().values<std::int32_t>("scalarPoint1")),
      s2(view.columns().values<std::int32_t>("scalarPoint2")),
      properties(view, "SpringProp")
{
}

Resolution<SpringScalarPropertyElement::Spring> SpringScalarPropertyElement::resolve(const ElementBatchView& view, const Columns& columns,
                                                                   std::int64_t row)
{
    const auto i = static_cast<std::size_t>(row);
    // Scalar points carry the component 0 (CELAS3: S1, S2 are scalar point IDs).
    Resolution<ScalarConnection> connection =
        resolveScalarConnection(view, row, columns.s1[i], 0, columns.s2[i], 0, true);
    if (!connection)
        return Resolution<Spring>::failure(std::move(connection.problem));
    Resolution<PropertyRef> property = resolveProperty(view, columns.properties, row);
    if (!property)
        return Resolution<Spring>::failure(std::move(property.problem));
    const base::ContractReader& springProp = property.value->table->reader();
    const auto p = static_cast<std::size_t>(property.value->row);
    return Resolution<Spring>::success({*connection.value, springProp.values<double>("stiffness")[p],
                                        springProp.values<double>("stressCoefficient")[p]});
}

void SpringScalarPropertyElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
