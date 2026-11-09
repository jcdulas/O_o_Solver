#include "oos/elements/masses/massScalarProperty/MassScalarPropertyElement.hpp"

namespace oos::elements
{

MassScalarPropertyElement::Columns::Columns(const ElementBatchView& view)
    : s1(view.columns().values<std::int32_t>("scalarPoint1")),
      s2(view.columns().values<std::int32_t>("scalarPoint2")),
      properties(view, "MassProp")
{
}

Resolution<MassScalarPropertyElement::Mass> MassScalarPropertyElement::resolve(const ElementBatchView& view, const Columns& columns,
                                                             std::int64_t row)
{
    const auto i = static_cast<std::size_t>(row);
    Resolution<ScalarConnection> connection =
        resolveScalarConnection(view, row, columns.s1[i], 0, columns.s2[i], 0, true);
    if (!connection)
        return Resolution<Mass>::failure(std::move(connection.problem));
    Resolution<PropertyRef> property = resolveProperty(view, columns.properties, row);
    if (!property)
        return Resolution<Mass>::failure(std::move(property.problem));
    const double mass =
        property.value->table->reader().values<double>("mass")[static_cast<std::size_t>(property.value->row)];
    return Resolution<Mass>::success({*connection.value, mass});
}

void MassScalarPropertyElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
