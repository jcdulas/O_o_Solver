#include "oos/elements/bars_beams/beam2/Beam2Element.hpp"

#include "oos/base/errors/ErrorMessage.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <sstream>
#include <string>

namespace oos::elements
{

namespace
{

// O_o_Reader's codes of the BeamProp SO field (BeamStressOutput).
constexpr std::uint8_t STRESS_OUTPUT_NO = 0;
constexpr std::uint8_t STRESS_OUTPUT_YES = 1;

// The BeamProp field of the warping coefficient, given at end A and at end B.
constexpr const char* WARPING_COEFFICIENT_FIELD = "warpingCoefficient";
constexpr std::array<const char*, 2> BEAM_END_NAMES{"A", "B"};

// The first non-zero value of a fixed list of BeamProp, or nothing: the fields refused (or ignored) when given.
std::optional<std::size_t> firstNonZero(std::span<const double> values)
{
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (values[i] != 0.0)
            return i;
    }
    return std::nullopt;
}

// A field value as a message prints it: six significant digits ("1e-12", not std::to_string's "0.000000").
std::string formatValue(double value)
{
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

// The information that the warping coefficient of BeamProp `propId` is not used: `value`, its first non-zero CW
// (`field` 0 for CW(A), 1 for CW(B)). The Beam2 fields SA, SB are the scalar points of the warping DOFs; without
// them the torsional stiffness of warping is left out, and CW enters the torsion equation only through the warping term
// E·CW·d⁴θ/dx⁴ (Vlasov's thin-walled beam theory). check() refuses SA and SB, so no Beam2 has the warping DOFs, and
// CW enters no matrix: the torsion is Saint-Venant's, G·J.
base::Diagnostic ignoredWarping(std::int32_t propId, std::size_t field, double value)
{
    base::Diagnostic diagnostic;
    diagnostic.severity = base::Severity::Information;
    diagnostic.code = problem_code::IGNORED_FIELD;
    diagnostic.card = "BeamProp";
    diagnostic.id = propId;
    diagnostic.field = WARPING_COEFFICIENT_FIELD;
    diagnostic.message = base::concatenate(
        {"BeamProp ", std::to_string(propId), ", field ", WARPING_COEFFICIENT_FIELD, " at end ", BEAM_END_NAMES[field],
         " = ", formatValue(value),
         " is not used: the warping coefficient acts only through the warping DOFs of a Beam2 (its scalar points "
         "of warping), and a Beam2 without them has none (without them the torsional stiffness due to warping is "
         "not considered), so its torsion is G·J alone; nothing to do (remove the warping coefficients to drop this "
         "note)"});
    return diagnostic;
}

} // namespace

Beam2Element::Columns::Columns(const ElementBatchView& view)
    : connection(view),
      sa(view.columns().values<std::int32_t>("warpingPointA")),
      sb(view.columns().values<std::int32_t>("warpingPointB")),
      properties(view, "BeamProp"),
      materials(view),
      beamlProperties(view, "BeamShapeProp"),
      sections(view)
{
    if (beamlProperties.table != nullptr)
    {
        const base::ContractReader& beaml = beamlProperties.table->reader();
        beamlMatId = beaml.values<std::int32_t>("matId");
        beamlPositions = beaml.lists<double>("stationRatio");
        beamlOutputs = beaml.lists<std::uint8_t>("stationOutput");
    }
    const input::CardTable* table = properties.table;
    if (table == nullptr)
        return;
    const base::ContractReader& reader = table->reader();
    matId = reader.values<std::int32_t>("matId");
    stationOutput = reader.lists<std::uint8_t>("stationOutput");
    stationPosition = reader.lists<double>("stationRatio");
    stationSection = input::NestedListView(reader, "stationSection");
    stationStress = input::NestedListView(reader, "stationStress");
    k = reader.fixedLists<double>("shearFactor");
    s = reader.fixedLists<double>("shearRelief");
    nsi = reader.fixedLists<double>("nsmInertia");
    cw = reader.fixedLists<double>("warpingCoefficient");
    m = reader.fixedLists<double>("nsmCenter");
    n = reader.fixedLists<double>("neutralAxis");
}

Resolution<BeamData> Beam2Element::resolve(const ElementBatchView& view, const Columns& columns, std::int64_t row)
{
    using Result = Resolution<BeamData>;
    const auto r = static_cast<std::size_t>(row);
    BeamData beam;
    Resolution<BeamGeometry> geometry = resolveBeamGeometry(view, columns.connection, row);
    if (!geometry)
        return Result::failure(std::move(geometry.problem));
    beam.geometry = *geometry.value;
    // Warping needs SA and SB > 0, its DOFs the points SA, SB; without them the warping stiffness is left out, which
    // is what this formulation computes.
    if (columns.sa[r] != 0 || columns.sb[r] != 0)
    {
        const bool isSa = columns.sa[r] != 0;
        return Result::failure(elementProblem(
            view, row, problem_code::UNSUPPORTED_FIELD, isSa ? "warpingPointA" : "warpingPointB",
            base::concatenate({"expected Beam2.warpingPointA and Beam2.warpingPointB blank, found the warping "
                               "point ",
                               isSa ? "Beam2.warpingPointA " : "Beam2.warpingPointB ",
                               std::to_string(isSa ? columns.sa[r] : columns.sb[r]),
                               "; the warping DOFs of Beam2 are not supported: remove Beam2.warpingPointA and "
                               "Beam2.warpingPointB (the beam then warps freely and the BeamProp's "
                               "BeamProp.warpingCoefficient is not used)"})));
    }

    const std::int32_t propId = columns.properties.propIds[r];
    if (columns.properties.valid.isValid(row))
    {
        const std::string_view type = columns.properties.types[row];
        if (type == "BeamShapeProp")
            return resolveBeaml(view, columns, row, std::move(beam));
        if (type == "BeamLumpedAreaProp")
            return Result::failure(elementProblem(
                view, row, problem_code::UNSUPPORTED_FIELD, "propId",
                base::concatenate({"expected a BeamProp, found the BeamLumpedAreaProp ", std::to_string(propId),
                                   "; BeamLumpedAreaProp is not supported: give the section as a BeamProp"})));
    }
    Resolution<PropertyRef> property = resolveProperty(view, columns.properties, row);
    if (!property)
        return Result::failure(std::move(property.problem));
    const std::int32_t p = property.value->row;

    struct Refused
    {
        const base::FixedListView<double>* values;
        std::array<const char*, 4> fields;
        const char* what;
    };
    // CW(A), CW(B) are not refused: without SA and SB (refused above) they are not used (ignoredWarping, reported by
    // check() once per BeamProp).
    const std::array<Refused, 2> refusedFields{{
        {&columns.s,
         {"BeamProp.shearRelief1", "BeamProp.shearRelief2", "", ""},
         "the shear relief coefficients of taper are not supported"},
        {&columns.nsi,
         {"BeamProp.nsmInertia(A)", "BeamProp.nsmInertia(B)", "", ""},
         "the mass moment of inertia of the BeamProp.nonStructMass is not supported"},
    }};
    for (const Refused& refused : refusedFields)
    {
        const std::span<const double> values = (*refused.values)[p];
        if (const std::optional<std::size_t> index = firstNonZero(values))
            return Result::failure(elementProblem(
                view, row, problem_code::UNSUPPORTED_FIELD, refused.fields[*index],
                base::concatenate({"BeamProp ", std::to_string(propId), ": expected ", refused.fields[*index],
                                   " blank or 0.0, found ", formatValue(values[*index]), "; ", refused.what,
                                   ": remove the field"})));
    }

    Resolution<IsotropicMaterial> material = resolveIsotropicMaterial(
        view, columns.materials, row, columns.matId[static_cast<std::size_t>(p)], "BeamProp.matId");
    if (!material)
        return Result::failure(std::move(material.problem));
    beam.material = *material.value;

    // The stations in increasing X/XB (the card may give them in any order); a BeamProp of end A alone is uniform.
    const std::span<const double> positions = columns.stationPosition[p];
    const std::span<const std::uint8_t> outputs = columns.stationOutput[p];
    std::array<std::size_t, BEAM_MAX_STATIONS> order{};
    const std::size_t count = std::min(positions.size(), BEAM_MAX_STATIONS);
    for (std::size_t i = 0; i < count; ++i)
        order[i] = i;
    std::stable_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(count),
                     [&](std::size_t a, std::size_t b) { return positions[a] < positions[b]; });
    BeamSection& section = beam.section;
    // N1, N2 (the neutral axis) and M1, M2 (the NSM's centre of gravity) from the shear centre at ends A and B (end B
    // defaults to end A: O_o_Reader gives them so), linear in X/XB between them (an assumption, as the other
    // properties of an intermediate station).
    const std::span<const double> n = columns.n[p];
    const std::span<const double> m = columns.m[p];
    // (1 − ξ)·A + ξ·B: the end values exactly at the ends.
    const auto between = [](std::span<const double> ends, std::size_t component, double position)
    { return (1.0 - position) * ends[component] + position * ends[2 + component]; };
    // The recovery points of the card are from the shear centre (PBEAM), those of
    // end A in that frame for a YESA station; a BeamStation's are from its neutral axis.
    std::array<std::array<double, 2>, 4> endAPoints{};
    for (std::size_t i = 0; i < count; ++i)
    {
        const std::size_t source = order[i];
        const std::span<const double> values = columns.stationSection.item(p, static_cast<std::int64_t>(source));
        const std::span<const double> points = columns.stationStress.item(p, static_cast<std::int64_t>(source));
        BeamStation station;
        station.position = positions[source];
        station.area = values[0];
        station.inertia1 = values[1];
        station.inertia2 = values[2];
        station.productOfInertia = values[3];
        station.torsionalConstant = values[4];
        station.nonStructuralMass = values[5];
        const std::uint8_t so = outputs[source];
        std::array<std::array<double, 2>, 4> fromShearCentre = endAPoints; // YESA: the points of end A
        if (so == STRESS_OUTPUT_YES || (i == 0 && so == STRESS_OUTPUT_NO))
            for (std::size_t point = 0; point < 4; ++point)
                fromShearCentre[point] = {points[2 * point], points[2 * point + 1]};
        if (i == 0)
            endAPoints = fromShearCentre;
        for (std::size_t k = 0; k < 2; ++k)
        {
            station.neutralAxis[k] = between(n, k, station.position);
            station.massCentre[k] = between(m, k, station.position);
        }
        for (std::size_t point = 0; point < 4; ++point)
            for (std::size_t k = 0; k < 2; ++k)
                station.recoveryPoints[point][k] = fromShearCentre[point][k] - station.neutralAxis[k];
        station.hasOutput = so != STRESS_OUTPUT_NO;
        section.stations[section.count++] = station;
    }
    if (section.count == 1)
    {
        // A BeamProp of end A alone: end B has its section, its own N and M (a taper of the offsets alone) and the
        // recovery points of end A from the shear centre.
        BeamStation endB = section.stations[0];
        endB.position = 1.0;
        for (std::size_t k = 0; k < 2; ++k)
        {
            endB.neutralAxis[k] = between(n, k, 1.0);
            endB.massCentre[k] = between(m, k, 1.0);
        }
        for (std::size_t point = 0; point < 4; ++point)
            for (std::size_t k = 0; k < 2; ++k)
                endB.recoveryPoints[point][k] = endAPoints[point][k] - endB.neutralAxis[k];
        section.stations[section.count++] = endB;
    }
    // The ends always have their row (an end without stress points gives the axial stress at its four points).
    section.stations[0].hasOutput = true;
    section.stations[section.count - 1].hasOutput = true;
    if (section.stations[0].position != 0.0 || section.stations[section.count - 1].position != 1.0)
        return Result::failure(elementProblem(
            view, row, problem_code::INVALID_FIELD, "BeamProp.stationRatio",
            base::concatenate({"BeamProp ", std::to_string(propId),
                               ": expected stations from BeamProp.stationRatio 0.0 (end A) to 1.0 (end B), found ",
                               std::to_string(section.stations[0].position), " to ",
                               std::to_string(section.stations[section.count - 1].position),
                               "; correct the stations"})));
    const std::span<const double> k = columns.k[p];
    section.shearFactor1 = k[0];
    section.shearFactor2 = k[1];
    if (const std::optional<SectionProblem> problem = findSectionProblem(section))
        return Result::failure(
            elementProblem(view, row, problem_code::INVALID_FIELD, problem->field,
                           base::concatenate({"BeamProp ", std::to_string(propId), ": ", problem->message})));
    if (!hasStablePinFlags(beam))
        return Result::failure(unstablePinFlags(view, row, beam.geometry));
    return Result::success(beam);
}

Resolution<BeamData> Beam2Element::resolveBeaml(const ElementBatchView& view, const Columns& columns,
                                                   std::int64_t row, BeamData beam)
{
    using Result = Resolution<BeamData>;
    const std::int32_t propId = columns.properties.propIds[static_cast<std::size_t>(row)];
    Resolution<PropertyRef> property = resolveProperty(view, columns.beamlProperties, row);
    if (!property)
        return Result::failure(std::move(property.problem));
    const std::int32_t p = property.value->row;
    Resolution<IsotropicMaterial> material = resolveIsotropicMaterial(
        view, columns.materials, row, columns.beamlMatId[static_cast<std::size_t>(p)], "BeamShapeProp.matId");
    if (!material)
        return Result::failure(std::move(material.problem));
    beam.material = *material.value;
    Resolution<BeamSection> section = resolveTabulatedSection(view, row, columns.sections, "BeamShapeProp", propId,
                                                              columns.beamlPositions[p], columns.beamlOutputs[p]);
    if (!section)
        return Result::failure(std::move(section.problem));
    beam.section = *section.value;
    if (const std::optional<SectionProblem> problem = findSectionProblem(beam.section))
        return Result::failure(
            elementProblem(view, row, problem_code::INVALID_FIELD, problem->field,
                           base::concatenate({"BeamShapeProp ", std::to_string(propId), ": ", problem->message})));
    if (!hasStablePinFlags(beam))
        return Result::failure(unstablePinFlags(view, row, beam.geometry));
    return Result::success(beam);
}

void Beam2Element::check(const ElementBatchView& view, base::DiagnosticSink& sink) const
{
    const Columns columns(view);
    checkEveryRow(view, sink, [&](std::int64_t row) { return resolve(view, columns, row); });
    // One information per BeamProp whose CW is given (ignoredWarping), in the order of the BeamProp table: whatever
    // Beam2 uses it, CW enters no computation, and a note per element would repeat it for every beam of a section.
    if (columns.properties.table == nullptr)
        return;
    for (std::size_t p = 0; p < columns.properties.tableIds.size(); ++p)
    {
        const std::span<const double> cw = columns.cw[static_cast<std::int64_t>(p)];
        if (const std::optional<std::size_t> index = firstNonZero(cw))
            sink.report(ignoredWarping(columns.properties.tableIds[p], *index, cw[*index]));
    }
}

} // namespace oos::elements
