#include "oos/elements/rods/tube2/Tube2Element.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <numbers>
#include <string>

namespace oos::elements
{

RodSection tubeSection(double outerDiameter, double thickness)
{
    // TubeProp: D_i = OD − 2T, A = π(OD² − D_i²)/4, J = π(OD⁴ − D_i⁴)/32; T = 0 is a solid rod (D_i = 0).
    const double innerDiameter = thickness > 0.0 ? outerDiameter - 2.0 * thickness : 0.0;
    const double outerSquared = outerDiameter * outerDiameter;
    const double innerSquared = innerDiameter * innerDiameter;
    RodSection section;
    section.area = std::numbers::pi * (outerSquared - innerSquared) / 4.0;
    section.torsionalConstant = std::numbers::pi * (outerSquared * outerSquared - innerSquared * innerSquared) / 32.0;
    section.stressCoefficient = outerDiameter / 2.0; // the torsional stress at the outer fibre
    return section;
}

Tube2Element::Columns::Columns(const ElementBatchView& view)
    : connectivity(view), properties(view, "TubeProp"), materials(view)
{
    if (properties.table == nullptr)
        return;
    const base::ContractReader& tubeProp = properties.table->reader();
    matId = tubeProp.values<std::int32_t>("matId");
    od = tubeProp.values<double>("outerDiameter");
    t = tubeProp.values<double>("thickness", base::NullPolicy::Allow);
    isThicknessGiven = tubeProp.validity("thickness");
    nonStructMass = tubeProp.values<double>("nonStructMass");
    od2 = tubeProp.values<double>("outerDiameter2", base::NullPolicy::Allow);
    isOd2Given = tubeProp.validity("outerDiameter2");
}

Resolution<RodData> Tube2Element::resolve(const ElementBatchView& view, const Columns& columns, std::int64_t row)
{
    using Result = Resolution<RodData>;
    Resolution<RodGeometry> geometry = resolveRodGeometry(view, columns.connectivity, row);
    if (!geometry)
        return Result::failure(std::move(geometry.problem));
    Resolution<PropertyRef> property = resolveProperty(view, columns.properties, row);
    if (!property)
        return Result::failure(std::move(property.problem));
    const std::int32_t p = property.value->row;
    const auto i = static_cast<std::size_t>(p);
    const double outerDiameter = columns.od[i];
    const double thickness = columns.isThicknessGiven.isValid(p) ? columns.t[i] : 0.0;
    if (columns.isOd2Given.isValid(p) && columns.od2[i] != outerDiameter)
        return Result::failure(
            elementProblem(view, row, problem_code::UNSUPPORTED_FIELD, "TubeProp.outerDiameter2",
                           base::concatenate({"expected TubeProp.outerDiameter2 blank or equal to "
                                              "TubeProp.outerDiameter = ",
                                              std::to_string(outerDiameter), " on its TubeProp, found ",
                                              std::to_string(columns.od2[i]),
                                              "; tapered tubes are not supported yet: use a constant diameter"})));
    if (thickness > outerDiameter / 2.0)
        return Result::failure(elementProblem(
            view, row, problem_code::INVALID_FIELD, "TubeProp.thickness",
            base::concatenate({"expected a wall thickness TubeProp.thickness at most TubeProp.outerDiameter/2 = ",
                               std::to_string(outerDiameter / 2.0), " on its TubeProp, found ",
                               std::to_string(thickness)})));
    Resolution<IsotropicMaterial> material =
        resolveIsotropicMaterial(view, columns.materials, row, columns.matId[i], "TubeProp.matId");
    if (!material)
        return Result::failure(std::move(material.problem));
    RodSection section = tubeSection(outerDiameter, thickness);
    section.nonStructuralMass = columns.nonStructMass[i];
    return Result::success({*geometry.value, section, *material.value});
}

void Tube2Element::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
}

} // namespace oos::elements
