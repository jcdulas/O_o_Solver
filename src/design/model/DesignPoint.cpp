#include "oos/design/model/DesignPoint.hpp"

#include "oos/design/model/DesignStamps.hpp"

namespace oos::design
{

namespace
{

[[nodiscard]] std::vector<double> valuesOf(const std::vector<FieldValue>& fields)
{
    std::vector<double> values;
    values.reserve(fields.size());
    for (const FieldValue& field : fields)
        values.push_back(field.value);
    return values;
}

} // namespace

DesignPoint::DesignPoint(const DesignModel& model, const FieldTargets& targets, std::span<const double> x,
                         std::span<const SuppliedField> supplied, std::span<const double> reference)
    : m_x(x.begin(), x.end()), m_values(model.evaluate(x, supplied))
{
    const std::vector<double> shapes = valuesOf(m_values.shapes);
    m_overlay = targets.overlayOf(model.shapes(), valuesOf(m_values.fields), shapes, reference);
    m_designStamps = computeDesignStamps(model, targets, m_values.fields, shapes);
}

} // namespace oos::design
