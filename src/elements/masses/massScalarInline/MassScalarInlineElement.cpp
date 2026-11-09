#include "oos/elements/masses/massScalarInline/MassScalarInlineElement.hpp"

namespace oos::elements
{

MassScalarInlineElement::Columns::Columns(const ElementBatchView& view)
    : m(view.columns().values<double>("mass")),
      s1(view.columns().values<std::int32_t>("scalarPoint1")),
      s2(view.columns().values<std::int32_t>("scalarPoint2"))
{
}

Resolution<MassScalarInlineElement::Mass> MassScalarInlineElement::resolve(const ElementBatchView& view, const Columns& columns,
                                                             std::int64_t row)
{
    const auto i = static_cast<std::size_t>(row);
    Resolution<ScalarConnection> connection =
        resolveScalarConnection(view, row, columns.s1[i], 0, columns.s2[i], 0, true);
    if (!connection)
        return Resolution<Mass>::failure(std::move(connection.problem));
    const double mass = columns.m[i];
    return Resolution<Mass>::success({*connection.value, mass});
}

void MassScalarInlineElement::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
