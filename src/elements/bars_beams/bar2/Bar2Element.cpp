#include "oos/elements/bars_beams/bar2/Bar2Element.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <string>

namespace oos::elements
{

Bar2Element::Columns::Columns(const ElementBatchView& view)
    : connection(view),
      properties(view, "BarProp"),
      materials(view),
      barlProperties(view, "BarShapeProp"),
      sections(view)
{
    if (barlProperties.table != nullptr)
        barlMatId = barlProperties.table->reader().values<std::int32_t>("matId");
    const input::CardTable* table = properties.table;
    if (table == nullptr)
        return;
    const base::ContractReader& reader = table->reader();
    matId = reader.values<std::int32_t>("matId");
    a = reader.values<double>("area");
    i1 = reader.values<double>("bendingInertia1");
    i2 = reader.values<double>("bendingInertia2");
    i12 = reader.values<double>("productInertia12");
    j = reader.values<double>("torsionConstant");
    nonStructMass = reader.values<double>("nonStructMass");
    const char* names[4] = {"stressPointC", "stressPointD", "stressPointE", "stressPointF"};
    for (std::size_t p = 0; p < 4; ++p)
        points[p] = reader.fixedLists<double>(names[p]);
    k = reader.fixedLists<double>("shearFactor");
}

Resolution<BeamData> Bar2Element::resolve(const ElementBatchView& view, const Columns& columns, std::int64_t row)
{
    using Result = Resolution<BeamData>;
    BeamData beam;
    Resolution<BeamGeometry> geometry = resolveBeamGeometry(view, columns.connection, row);
    if (!geometry)
        return Result::failure(std::move(geometry.problem));
    beam.geometry = *geometry.value;

    const std::int32_t propId = columns.properties.propIds[static_cast<std::size_t>(row)];
    const bool isBarl = columns.properties.valid.isValid(row) && columns.properties.types[row] == "BarShapeProp";
    Resolution<PropertyRef> property = resolveProperty(view, isBarl ? columns.barlProperties : columns.properties, row);
    if (!property)
        return Result::failure(std::move(property.problem));
    const auto p = static_cast<std::size_t>(property.value->row);
    Resolution<IsotropicMaterial> material = resolveIsotropicMaterial(
        view, columns.materials, row, isBarl ? columns.barlMatId[p] : columns.matId[p], "BarProp.matId");
    if (!material)
        return Result::failure(std::move(material.problem));
    beam.material = *material.value;

    if (isBarl)
    {
        Resolution<BeamSection> section =
            resolveTabulatedSection(view, row, columns.sections, "BarShapeProp", propId, {}, {});
        if (!section)
            return Result::failure(std::move(section.problem));
        beam.section = *section.value;
    }
    else
    {
        BeamStation station;
        station.area = columns.a[p];
        station.inertia1 = columns.i1[p];
        station.inertia2 = columns.i2[p];
        station.productOfInertia = columns.i12[p];
        station.torsionalConstant = columns.j[p];
        station.nonStructuralMass = columns.nonStructMass[p];
        for (std::size_t point = 0; point < 4; ++point)
        {
            const std::span<const double> yz = columns.points[point][property.value->row];
            station.recoveryPoints[point] = {yz[0], yz[1]};
        }
        beam.section.count = 2;
        beam.section.stations[0] = station;
        station.position = 1.0;
        beam.section.stations[1] = station;
        const std::span<const double> k = columns.k[property.value->row];
        beam.section.shearFactor1 = k[0];
        beam.section.shearFactor2 = k[1];
    }
    if (const std::optional<SectionProblem> problem = findSectionProblem(beam.section))
        return Result::failure(elementProblem(view, row, problem_code::INVALID_FIELD, problem->field,
                                              base::concatenate({isBarl ? "BarShapeProp " : "BarProp ",
                                                                 std::to_string(propId), ": ", problem->message})));
    if (!hasStablePinFlags(beam))
        return Result::failure(unstablePinFlags(view, row, beam.geometry));
    return Result::success(beam);
}

void Bar2Element::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
