#include "oos/elements/springs/springDofProperty/SpringDofPropertyElement.hpp"

namespace oos::elements
{

SpringDofPropertyElement::Columns::Columns(const ElementBatchView& view)
    : g1(view.columns().values<std::int32_t>("node1")),
      c1(view.columns().values<std::int32_t>("components1")),
      g2(view.columns().values<std::int32_t>("node2")),
      c2(view.columns().values<std::int32_t>("components2")),
      properties(view, "SpringProp")
{
}

Resolution<SpringDofPropertyElement::Spring> SpringDofPropertyElement::resolve(const ElementBatchView& view, const Columns& columns,
                                                                   std::int64_t row)
{
    const auto i = static_cast<std::size_t>(row);
    Resolution<ScalarConnection> connection =
        resolveScalarConnection(view, row, columns.g1[i], columns.c1[i], columns.g2[i], columns.c2[i], false);
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

void SpringDofPropertyElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
