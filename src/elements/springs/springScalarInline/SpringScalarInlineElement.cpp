#include "oos/elements/springs/springScalarInline/SpringScalarInlineElement.hpp"

namespace oos::elements
{

SpringScalarInlineElement::Columns::Columns(const base::ContractReader& reader)
    : k(reader.values<double>("stiffness")),
      s1(reader.values<std::int32_t>("scalarPoint1")),
      s2(reader.values<std::int32_t>("scalarPoint2"))
{
}

Resolution<SpringScalarInlineElement::Spring> SpringScalarInlineElement::resolve(const ElementBatchView& view, const Columns& columns,
                                                                   std::int64_t row)
{
    const auto i = static_cast<std::size_t>(row);
    Resolution<ScalarConnection> connection =
        resolveScalarConnection(view, row, columns.s1[i], 0, columns.s2[i], 0, true);
    if (!connection)
        return Resolution<Spring>::failure(std::move(connection.problem));
    return Resolution<Spring>::success({*connection.value, columns.k[i]});
}

void SpringScalarInlineElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view.columns());
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
