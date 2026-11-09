#include "oos/elements/springs/springDofInline/SpringDofInlineElement.hpp"

namespace oos::elements
{

SpringDofInlineElement::Columns::Columns(const base::ContractReader& reader)
    : k(reader.values<double>("stiffness")),
      g1(reader.values<std::int32_t>("node1")),
      c1(reader.values<std::int32_t>("components1")),
      g2(reader.values<std::int32_t>("node2")),
      c2(reader.values<std::int32_t>("components2")),
      s(reader.values<double>("stressCoefficient"))
{
}

Resolution<SpringDofInlineElement::Spring> SpringDofInlineElement::resolve(const ElementBatchView& view, const Columns& columns,
                                                                   std::int64_t row)
{
    const auto i = static_cast<std::size_t>(row);
    Resolution<ScalarConnection> connection =
        resolveScalarConnection(view, row, columns.g1[i], columns.c1[i], columns.g2[i], columns.c2[i], false);
    if (!connection)
        return Resolution<Spring>::failure(std::move(connection.problem));
    return Resolution<Spring>::success({*connection.value, columns.k[i], columns.s[i]});
}

void SpringDofInlineElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view.columns());
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
