#include "oos/design/sensitivity/FieldPerturbation.hpp"

#include "oos/base/errors/ErrorMessage.hpp"
#include "oos/input/snapshot/ColumnOverlay.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace oos::design
{

namespace
{

[[nodiscard]] std::vector<double> valuesOf(const std::vector<FieldValue>& quantities)
{
    std::vector<double> values;
    values.reserve(quantities.size());
    for (const FieldValue& quantity : quantities)
        values.push_back(quantity.value);
    return values;
}

} // namespace

double fieldStep(double value) noexcept
{
    return value != 0.0 ? FIELD_STEP_RELATIVE * std::abs(value) : FIELD_STEP_AT_ZERO;
}

double fieldStep(double value, double scale) noexcept
{
    const double floor = FIELD_STEP_SCALE_FRACTION * scale;
    if (!(floor > 0.0) || std::abs(value) >= floor)
        return fieldStep(value);
    return FIELD_STEP_RELATIVE * floor;
}

double fieldStepScale(const input::ModelSnapshot& point, const FieldTarget& target)
{
    // The diagonal of a AnisoMat2D's [[G11, G12, G13], [G12, G22, G23], [G13, G23, G33]]: its items 0, 3 and 5.
    constexpr std::array<std::size_t, 3> DIAGONAL{0, 3, 5};
    double scale = 0.0;
    for (const TargetCell& cell : target.cells)
    {
        if (cell.column != "stiffness")
            continue;
        const base::ContractReader* reader = point.findTable(cell.table);
        if (reader == nullptr || !reader->hasColumn("stiffness") ||
            !reader->columnType("stiffness").starts_with("fixed_list<float64>") ||
            !reader->validity("stiffness").isValid(cell.row))
            continue;
        const std::span<const double> terms =
            reader->fixedLists<double>("stiffness", base::NullPolicy::Allow)[cell.row];
        for (const std::size_t k : DIAGONAL)
            if (k < terms.size() && std::isfinite(terms[k]))
                scale = std::max(scale, std::abs(terms[k]));
    }
    return scale;
}

FieldPerturbation::Side FieldPerturbation::makeSide(const std::shared_ptr<const input::ModelSnapshot>& base,
                                                    const input::ColumnOverlay& overlay,
                                                    const elements::ElementRegistry& registry,
                                                    base::MemoryBudget& budget)
{
    Side side;
    side.snapshot = std::make_shared<const input::ModelSnapshot>(overlay.applyTo(*base, base));
    side.index = std::make_shared<const elements::SnapshotIndex>(*side.snapshot, budget);
    side.set = std::make_shared<const assembly::ElementSet>(*side.index, registry);
    return side;
}

FieldPerturbation::FieldPerturbation(const std::shared_ptr<const input::ModelSnapshot>& base, const DesignModel& model,
                                     const FieldTargets& targets, const DesignValues& values, std::size_t quantity,
                                     double step, const elements::ElementRegistry& registry, base::MemoryBudget& budget)
    : m_quantity(quantity)
{
    const std::size_t fieldCount = values.fields.size();
    if (quantity >= fieldCount + values.shapes.size() || !(step > 0.0))
        throw std::invalid_argument(base::concatenate(
            {"FieldPerturbation: expected a quantity below ", std::to_string(fieldCount + values.shapes.size()),
             " and a positive step, found quantity ", std::to_string(quantity), " and step ", std::to_string(step)}));
    std::vector<double> fields = valuesOf(values.fields);
    std::vector<double> shapes = valuesOf(values.shapes);
    const std::vector<double> pointFields = fields; // the reference of the section meshes
    double& moved = quantity < fieldCount ? fields[quantity] : shapes[quantity - fieldCount];
    const double value = moved;
    const std::array<double, FIELD_SIDE_COUNT> sides{value + step, value - step, value + step / 2.0,
                                                     value - step / 2.0};
    m_width = sides[0] - sides[1];
    m_halfWidth = sides[2] - sides[3];
    for (std::size_t k = 0; k < FIELD_SIDE_COUNT; ++k)
    {
        moved = sides[k];
        m_sides[k] = makeSide(base, targets.overlayOf(model.shapes(), fields, shapes, pointFields), registry, budget);
    }
}

FieldPerturbation::FieldPerturbation(const std::shared_ptr<const input::ModelSnapshot>& base,
                                     const FieldTargets& targets, const std::vector<FieldValue>& fields,
                                     std::size_t relation, const elements::ElementRegistry& registry,
                                     base::MemoryBudget& budget)
    : m_quantity(relation)
{
    if (!targets.movedRows().empty())
        throw std::invalid_argument(
            "FieldPerturbation: the design model moves grid points (DesignNodeMove), so a perturbation must keep them "
            "at the "
            "design point; build it from the design model and the design values");
    if (relation >= fields.size())
        throw std::invalid_argument(
            base::concatenate({"FieldPerturbation: expected a relation below ", std::to_string(fields.size()),
                               ", found ", std::to_string(relation)}));
    std::vector<double> values = valuesOf(fields);
    const std::vector<double> pointFields = values; // the reference of the section meshes
    const double value = values[relation];
    const double step = fieldStep(value, fieldStepScale(*base, targets[relation]));
    const std::array<double, FIELD_SIDE_COUNT> sides{value + step, value - step, value + step / 2.0,
                                                     value - step / 2.0};
    m_width = sides[0] - sides[1];
    m_halfWidth = sides[2] - sides[3];
    const ShapeRelations noShapes;
    for (std::size_t k = 0; k < FIELD_SIDE_COUNT; ++k)
    {
        values[relation] = sides[k];
        m_sides[k] = makeSide(base, targets.overlayOf(noShapes, values, {}, pointFields), registry, budget);
    }
}

FieldPerturbation::FieldPerturbation(const std::shared_ptr<const input::ModelSnapshot>& point,
                                     const elements::SnapshotIndex& pointIndex, const FieldTarget& target, double value,
                                     std::size_t relation, double step, const elements::ElementRegistry& registry,
                                     const FieldTargets* targets, std::span<const double> pointFields,
                                     const SectionLinearisation* sections)
    : m_quantity(relation)
{
    if (!(step > 0.0))
        throw std::invalid_argument(
            base::concatenate({"FieldPerturbation: expected a positive step, found ", std::to_string(step),
                               " for relation ", std::to_string(relation)}));
    if (&pointIndex.snapshot() != point.get())
        throw std::invalid_argument("FieldPerturbation: expected the index of the design point's snapshot");
    const std::array<double, FIELD_SIDE_COUNT> sides{value + step, value - step, value + step / 2.0,
                                                     value - step / 2.0};
    m_width = sides[0] - sides[1];
    m_halfWidth = sides[2] - sides[3];
    const bool isDerived = targets != nullptr && targets->derivedProperties().groupOf(relation).has_value();
    if (isDerived && pointFields.size() <= relation)
        throw std::invalid_argument(
            "FieldPerturbation: a field of a designed laminate or section needs the field "
            "values of the design point");
    std::vector<double> fields(pointFields.begin(), pointFields.end());
    for (std::size_t k = 0; k < FIELD_SIDE_COUNT; ++k)
    {
        input::ColumnOverlay overlay;
        for (const TargetCell& cell : target.cells)
            overlay.setValue(cell.table, cell.column, {cell.row, cell.item, cell.scale * sides[k]});
        if (isDerived)
        {
            fields[relation] = sides[k];
            if (sections != nullptr)
                targets->addDerivedCells(overlay, fields, relation, sections);
            else
            {
                // The library at the side's dimensions on the meshes of the point's.
                if (const std::optional<std::size_t> group = targets->derivedProperties().groupOf(relation))
                    targets->derivedProperties().addCells(overlay, fields, group, targets->sectionLibrary(), nullptr,
                                                          pointFields);
            }
        }
        Side& side = m_sides[k];
        side.snapshot = std::make_shared<const input::ModelSnapshot>(overlay.applyTo(*point, point));
        side.index = std::make_shared<const elements::SnapshotIndex>(*side.snapshot, pointIndex);
        side.set = std::make_shared<const assembly::ElementSet>(*side.index, registry);
    }
}

FieldPerturbation::FieldPerturbation(const std::shared_ptr<const input::ModelSnapshot>& point,
                                     const elements::SnapshotIndex& pointIndex, std::span<const BatchedField> fields,
                                     const elements::ElementRegistry& registry)
    : m_quantity(fields.empty() ? 0 : fields.front().relation)
{
    if (&pointIndex.snapshot() != point.get())
        throw std::invalid_argument("FieldPerturbation: expected the index of the design point's snapshot");
    std::array<input::ColumnOverlay, FIELD_SIDE_COUNT> overlays;
    m_members.reserve(fields.size());
    for (const BatchedField& field : fields)
    {
        if (!(field.step > 0.0) || field.target == nullptr)
            throw std::invalid_argument(
                base::concatenate({"FieldPerturbation: expected a target and a positive step, found step ",
                                   std::to_string(field.step), " for relation ", std::to_string(field.relation)}));
        // The sides of the third constructor, cell by cell.
        const std::array<double, FIELD_SIDE_COUNT> sides{field.value + field.step, field.value - field.step,
                                                         field.value + field.step / 2.0,
                                                         field.value - field.step / 2.0};
        m_members.push_back({field.relation, sides[0] - sides[1], sides[2] - sides[3]});
        for (std::size_t k = 0; k < FIELD_SIDE_COUNT; ++k)
            for (const TargetCell& cell : field.target->cells)
                overlays[k].setValue(cell.table, cell.column, {cell.row, cell.item, cell.scale * sides[k]});
    }
    if (!m_members.empty())
    {
        m_width = m_members.front().width;
        m_halfWidth = m_members.front().halfWidth;
    }
    for (std::size_t k = 0; k < FIELD_SIDE_COUNT; ++k)
    {
        Side& side = m_sides[k];
        side.snapshot = std::make_shared<const input::ModelSnapshot>(overlays[k].applyTo(*point, point));
        side.index = std::make_shared<const elements::SnapshotIndex>(*side.snapshot, pointIndex);
        side.set = std::make_shared<const assembly::ElementSet>(*side.index, registry);
    }
}

FieldPerturbation::FieldPerturbation(std::size_t quantity, double width, double halfWidth,
                                     const std::array<Side, FIELD_SIDE_COUNT>& sides)
    : m_quantity(quantity), m_width(width), m_halfWidth(halfWidth), m_sides(sides)
{
}

FieldPerturbation FieldPerturbation::member(std::size_t k) const
{
    if (k >= m_members.size())
        throw std::invalid_argument(
            base::concatenate({"FieldPerturbation::member: expected a member below ", std::to_string(m_members.size()),
                               ", found ", std::to_string(k)}));
    const Member& field = m_members[k];
    return {field.relation, field.width, field.halfWidth, m_sides};
}

double FieldPerturbation::derivative(double plus, double minus, double halfPlus, double halfMinus) const noexcept
{
    // Richardson: D(h) = (f₊ − f₋)/(2h), D(h/2) likewise; f' = (4·D(h/2) − D(h))/3.
    const double coarse = (plus - minus) / m_width;
    const double fine = (halfPlus - halfMinus) / m_halfWidth;
    return (4.0 * fine - coarse) / 3.0;
}

void FieldPerturbation::derivatives(std::span<const double> plus, std::span<const double> minus,
                                    std::span<const double> halfPlus, std::span<const double> halfMinus,
                                    std::span<double> derivative) const noexcept
{
    for (std::size_t k = 0; k < derivative.size(); ++k)
        derivative[k] = this->derivative(plus[k], minus[k], halfPlus[k], halfMinus[k]);
}

} // namespace oos::design
